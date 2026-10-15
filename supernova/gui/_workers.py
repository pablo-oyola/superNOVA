"""SLEPc worker processes that solve frequency windows side by side.

MUMPS stops scaling past four to six OpenMP threads on the machines this runs
on, so one process with many threads leaves cores idle. The windows of a scan
are independent: given the same assembled pencil, each is one
shift-and-invert solve. So the pool runs several worker processes with a few
threads each, sends every worker the pencil once, and hands out windows as
workers come free. Each worker keeps its eigensolver and retargets it, as
:meth:`~supernova.solve.stability.StabilitySolver.solve_pep_windows` does in
one process.

Workers return :class:`~supernova.solve.stability.WindowSolution` -- plain
arrays -- and the caller packs them, because packing needs the solver, which
only the caller has.

Workers are separate processes, not threads, for two reasons: slepc4py holds
the GIL for the whole of a solve, so threads would take turns; and a stop can
then terminate a worker in the middle of a factorization, which no thread can
be made to do. The thread count is fixed when a worker starts, because the
OpenMP runtime reads it once; changing it replaces the workers.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import io
import itertools
import multiprocessing
from multiprocessing.connection import wait
import os
import sys
import time
import traceback

# set before a worker starts, so its OpenMP runtime and BLAS read them
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

_pencil_keys = itertools.count(1)


@contextlib.contextmanager
def without_main_module():
    """Start spawned processes without re-running the parent's main script.

    A spawned child imports the parent's ``__main__`` before its target runs.
    The processes here never need it, and a script that opens the window
    without an ``if __name__ == "__main__"`` guard would open a second one in
    the child, which multiprocessing then aborts. Hiding the main module's
    name and path for the moment of ``start()`` -- when the child's launch
    data are taken -- leaves the child nothing to import.
    """
    main = sys.modules.get("__main__")
    saved = {name: getattr(main, name) for name in ("__spec__", "__file__")
             if main is not None and hasattr(main, name)}
    try:
        if "__spec__" in saved:
            main.__spec__ = None
        if "__file__" in saved:
            del main.__file__
        yield
    finally:
        for name, value in saved.items():
            setattr(main, name, value)


def new_pencil_key() -> int:
    """A key that no earlier pencil sent to the pool has used."""
    return next(_pencil_keys)


def _worker_main(conn) -> None:
    """A worker's loop: hold one pencil, solve the windows sent for it.

    Messages in: ``("pencil", key, pencil)``, ``("window", key, index,
    target_s, region)`` and ``("stop",)``. Out, one per window: ``("window",
    index, solution or None, error or None, log, seconds)``.
    """
    from supernova.eigen.dispatch import build_eigensolver
    from supernova.solve.stability import WindowSolution

    key, pencil, eigensolver = None, None, None

    def discard():
        nonlocal eigensolver
        if eigensolver is not None:
            with contextlib.suppress(Exception):
                eigensolver.destroy()
        eigensolver = None

    try:
        while True:
            try:
                message = conn.recv()
            except (EOFError, OSError):
                return
            if message[0] == "stop":
                return
            if message[0] == "pencil":
                discard()
                _, key, pencil = message
                continue

            _, window_key, index, target_s, region = message
            started = time.perf_counter()
            log = io.StringIO()
            try:
                if window_key != key:
                    raise RuntimeError("This worker holds a different pencil.")
                with contextlib.redirect_stdout(log):
                    if eigensolver is None:
                        eigensolver = build_eigensolver(
                            gamma_s=pencil["gamma_s"], T0=pencil["T0"],
                            T1=pencil["T1"], Tminus1=pencil["Tminus1"],
                            target_s=target_s, region=region, **pencil["settings"])
                    else:
                        eigensolver.retarget(target_s, region)
                    result = eigensolver.solve(allow_empty=True)
                solution = WindowSolution.from_solver(eigensolver, result)
            except Exception as error:
                discard()               # a failed solve may leave it unusable
                conn.send(("window", index, None, f"{type(error).__name__}: {error}",
                           log.getvalue() + traceback.format_exc(),
                           time.perf_counter() - started))
            else:
                conn.send(("window", index, solution, None, log.getvalue(),
                           time.perf_counter() - started))
    finally:
        discard()


@dataclass
class _Worker:
    process: multiprocessing.process.BaseProcess
    conn: object
    pencil: int | None = None        # the key of the pencil it holds
    window: int | None = None        # the window it is solving


class WindowPool:
    """A fixed set of worker processes that solve windows of one pencil."""

    def __init__(self, workers: int = 1, threads: int = 1):
        self.workers = 0
        self.threads = 0
        self._pool: list[_Worker] = []
        self.configure(workers, threads)

    def configure(self, workers: int, threads: int) -> None:
        """Set the shape; a change replaces the workers at the next solve."""
        workers, threads = max(1, int(workers)), max(1, int(threads))
        if (workers, threads) != (self.workers, self.threads):
            self.close()
            self.workers, self.threads = workers, threads

    # ------------------------------------------------------------ processes

    def _start_worker(self) -> _Worker:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        saved = {name: os.environ.get(name) for name in THREAD_VARIABLES}
        try:
            for name in THREAD_VARIABLES:
                os.environ[name] = str(self.threads)
            process = context.Process(target=_worker_main, args=(child,),
                                      name="supernova-window-worker", daemon=True)
            with without_main_module():
                process.start()
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        child.close()
        return _Worker(process, parent)

    def _ensure(self) -> None:
        self._pool = [worker for worker in self._pool if worker.process.is_alive()]
        while len(self._pool) < self.workers:
            self._pool.append(self._start_worker())

    @staticmethod
    def _kill(worker: _Worker) -> None:
        worker.process.terminate()
        worker.process.join(2.0)
        if worker.process.is_alive():
            worker.process.kill()
            worker.process.join(1.0)
        with contextlib.suppress(Exception):
            worker.conn.close()

    def close(self) -> None:
        """Stop every worker; the next solve starts fresh ones."""
        for worker in self._pool:
            with contextlib.suppress(Exception):
                worker.conn.send(("stop",))
        for worker in self._pool:
            worker.process.join(2.0)
            if worker.process.is_alive():
                self._kill(worker)
        self._pool = []

    # -------------------------------------------------------------- solving

    def solve(self, key: int, pencil: dict, windows, on_window, cancel=None,
              poll: float = 0.2) -> list[int]:
        """Solve ``windows`` -- ``(index, target_s, region)`` -- on ``pencil``.

        ``on_window(index, solution, error, log, seconds)`` is called in this
        process as each window finishes, in completion order; ``solution`` is
        None when the window failed, ``error`` says why. A worker that dies
        costs only its window: it is replaced and the rest carry on.

        ``cancel`` is a ``threading.Event``. When it is set the busy workers are
        terminated at once, mid-solve, and replaced at the next solve.

        Returns the indices of the windows never solved.
        """
        self._ensure()
        pending = list(windows)
        for worker in self._pool:
            worker.window = None

        def assign(worker: _Worker) -> None:
            if worker.pencil != key:
                worker.conn.send(("pencil", key, pencil))
                worker.pencil = key
            index, target_s, region = pending.pop(0)
            worker.conn.send(("window", key, index, target_s, region))
            worker.window = index

        unsolved = []
        try:
            for worker in self._pool:
                if pending:
                    assign(worker)
            while any(worker.window is not None for worker in self._pool):
                if cancel is not None and cancel.is_set():
                    for worker in self._pool:
                        if worker.window is not None:
                            unsolved.append(worker.window)
                            self._kill(worker)
                    self._pool = [w for w in self._pool if w.process.is_alive()]
                    break
                busy = {worker.conn: worker for worker in self._pool
                        if worker.window is not None}
                for conn in wait(list(busy), timeout=poll):
                    worker = busy[conn]
                    index = worker.window
                    try:
                        _, index, solution, error, log, seconds = conn.recv()
                    except (EOFError, OSError):
                        worker.process.join(1.0)
                        on_window(index, None, "the worker process died (exit code "
                                  f"{worker.process.exitcode})", "", 0.0)
                        self._kill(worker)
                        self._pool.remove(worker)
                        if pending:
                            replacement = self._start_worker()
                            self._pool.append(replacement)
                            assign(replacement)
                        continue
                    worker.window = None
                    on_window(index, solution, error, log, seconds)
                    if pending and not (cancel is not None and cancel.is_set()):
                        assign(worker)
        except BaseException:
            # A worker still busy would answer the next solve with this one's
            # window; it cannot be trusted, so it goes.
            for worker in self._pool:
                if worker.window is not None:
                    self._kill(worker)
            self._pool = [w for w in self._pool if w.process.is_alive()]
            raise
        return unsolved + [index for index, _, _ in pending]
