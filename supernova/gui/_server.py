"""The compute process: everything slow happens here, never in the window.

A thread cannot keep the window alive through a solve. slepc4py, pyCOCOS and
the numba kernels hold the GIL for the whole of each call, so a worker thread
starves the GUI thread for as long as a factorization takes, and the desktop
then declares the window hung. So the window starts this process once and
talks to it over a pipe: it sends requests, and gets back progress, log text,
and results made of plain arrays.

The process owns the loaded equilibria -- the window holds only their
snapshots, which name them by ``handle`` -- the continuum operators, and a
:class:`~supernova.gui._workers.WindowPool` that solves frequency windows in
further processes. It never runs an eigensolver itself.

Requests are ``(kind, job, *args)`` and run one at a time, in order:

``("load_equilibrium", job, state)``
    an :class:`~supernova.gui._equilibrium.EquilibriumState` as a dict;
    answers with the view, ``equilibrium`` stripped and ``handle`` set
``("preview", job, handle, state)``
    a :class:`~supernova.gui._solver.SolverState` as a dict; answers with a
    continuum preview
``("solve", job, handle, state, title)``
    answers with a :class:`~supernova.gui._solver.SolverResult`, stripped
    likewise
``("refine", job, handle, state, request, title)``
    re-solves the modes in ``request`` on graded meshes; answers with a
    :class:`~supernova.gui._solver.RefineResult`

plus ``("cancel", job)``, acted on at once, and ``("shutdown",)``. Replies are
``("progress", job, text)``, ``("log", job, text)``, ``("done", job, result)``
and ``("failed", job, message, traceback)``.
"""

from __future__ import annotations

from collections import OrderedDict
import dataclasses
import queue
import threading
import traceback

# equilibria held for handles; the window may keep solving on an older one
KEEP_EQUILIBRIA = 3


class _Sender:
    """``conn.send`` from two threads, one message at a time."""

    def __init__(self, conn):
        self._conn = conn
        self._lock = threading.Lock()

    def send(self, message) -> None:
        with self._lock:
            self._conn.send(message)


class _Log:
    """A file-like object whose writes become ``("log", job, text)`` replies."""

    def __init__(self, sender: _Sender, job: int):
        self._sender, self._job = sender, job

    def write(self, text: str) -> int:
        if text:
            self._sender.send(("log", self._job, text))
        return len(text)

    def flush(self) -> None:
        pass


class _Server:
    """Runs the requests; holds what they leave behind."""

    def __init__(self, sender: _Sender):
        from supernova.gui._equilibrium import EquilibriumModel
        from supernova.gui._solver import SolverModel
        from supernova.gui._workers import WindowPool

        self.sender = sender
        self.equilibrium_model = EquilibriumModel()
        self.solver_model = SolverModel()
        self.pool = WindowPool()
        self.equilibria: OrderedDict[int, object] = OrderedDict()
        self._handles: dict[tuple, int] = {}
        self._next_handle = 1

    def run(self, kind: str, job: int, args: tuple, cancel: threading.Event):
        def progress(text: str) -> None:
            self.sender.send(("progress", job, text))

        if kind == "load_equilibrium":
            from supernova.gui._equilibrium import EquilibriumState

            state = EquilibriumState.from_dict(args[0])
            view = self.equilibrium_model.load(state, progress=progress)
            handle = self._remember(state.equilibrium_key(), view.equilibrium)
            return dataclasses.replace(view, equilibrium=None, handle=handle)

        if kind == "preview":
            from supernova.gui._solver import SolverState

            handle, state = args
            return self.solver_model.preview(
                self._equilibrium(handle), SolverState.from_dict(state),
                progress=progress)

        if kind == "solve":
            from supernova.gui._solver import SolverState

            handle, state, title = args
            state = SolverState.from_dict(state)
            self.pool.configure(state.workers, state.threads_per_worker)
            result = self.solver_model.solve(
                self._equilibrium(handle), state, title=title, progress=progress,
                log=_Log(self.sender, job), cancel=cancel, pool=self.pool)
            return dataclasses.replace(result, equilibrium=None, handle=handle)

        if kind == "refine":
            from supernova.gui._solver import SolverState

            handle, state, request, title = args
            state = SolverState.from_dict(state)
            self.pool.configure(state.workers, state.threads_per_worker)
            result = self.solver_model.refine(
                self._equilibrium(handle), state, request, title=title,
                progress=progress, cancel=cancel, pool=self.pool)
            return dataclasses.replace(result, handle=handle)

        raise ValueError(f"Unknown request {kind!r}.")

    def _remember(self, key: tuple, equilibrium) -> int:
        """The handle for this equilibrium, reusing one for an unchanged load."""
        handle = self._handles.get(key)
        if handle is None or self.equilibria.get(handle) is not equilibrium:
            handle = self._next_handle
            self._next_handle += 1
            self._handles[key] = handle
        self.equilibria[handle] = equilibrium
        self.equilibria.move_to_end(handle)
        while len(self.equilibria) > KEEP_EQUILIBRIA:
            dropped, _ = self.equilibria.popitem(last=False)
            self._handles = {k: h for k, h in self._handles.items() if h != dropped}
        return handle

    def _equilibrium(self, handle):
        if handle not in self.equilibria:
            raise ValueError(
                "The compute process no longer holds that equilibrium (it keeps "
                f"the last {KEEP_EQUILIBRIA}, and loses them all if it restarts). "
                "Load it again in the Equilibrium tab.")
        self.equilibria.move_to_end(handle)
        return self.equilibria[handle]

    def close(self) -> None:
        self.pool.close()


def serve(conn) -> None:
    """The compute process's main function."""
    sender = _Sender(conn)
    requests: queue.Queue = queue.Queue()
    cancels: dict[int, threading.Event] = {}

    def listen() -> None:
        # A thread of its own, so a cancel arrives while a request runs.
        while True:
            try:
                message = conn.recv()
            except (EOFError, OSError):         # the window has gone
                requests.put(None)
                return
            if message[0] == "shutdown":
                requests.put(None)
                return
            if message[0] == "cancel":
                event = cancels.get(message[1])
                if event is not None:
                    event.set()
                continue
            cancels[message[1]] = threading.Event()
            requests.put(message)

    threading.Thread(target=listen, name="supernova-compute-listener",
                     daemon=True).start()
    server = _Server(sender)
    try:
        while True:
            message = requests.get()
            if message is None:
                return
            kind, job, *args = message
            cancel = cancels.get(job) or threading.Event()
            try:
                if cancel.is_set():
                    raise RuntimeError("Stopped before it started.")
                result = server.run(kind, job, tuple(args), cancel)
            except Exception as error:
                sender.send(("failed", job, f"{type(error).__name__}: {error}",
                             traceback.format_exc()))
            else:
                sender.send(("done", job, result))
            finally:
                cancels.pop(job, None)
    finally:
        server.close()
