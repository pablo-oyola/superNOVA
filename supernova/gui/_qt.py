"""Pieces the equilibrium and solver tabs share: the compute client and widgets.

Everything slow runs in a separate process (:mod:`supernova.gui._server`), not
a thread. slepc4py, pyCOCOS and the numba kernels hold the GIL for the whole
of each call, so a worker thread would freeze the window for as long as a
factorization takes -- long enough for the desktop to call it hung. The
window only ever waits on a pipe.

The client starts that process when it is first needed, sends it one request
per job, and turns its replies into Qt signals: a reader thread receives them
and emits them on each job, and since the jobs live on the GUI thread the
signals arrive there, queued. Requests run one at a time, in the order sent,
which also keeps an equilibrium from being loaded and solved on at once.
"""

from __future__ import annotations

import atexit
import contextlib
from dataclasses import MISSING, fields
import itertools
import multiprocessing
import threading

import matplotlib
matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets

MUTED = "#4B5A72"
WARNING = "#9C5D14"
ERROR = "#A33636"


# ----------------------------------------------------------------------------
# The compute process
# ----------------------------------------------------------------------------

class Job(QtCore.QObject):
    """One request to the compute process.

    Lives on the GUI thread. Its signals are emitted by the client's reader
    thread and so arrive here queued, in the order the process sent them.
    """

    progressed = QtCore.Signal(str)
    logged = QtCore.Signal(str)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str, str)

    def __init__(self, ident: int, kind: str):
        super().__init__()
        self.id = ident
        self.kind = kind


class _Link:
    """The process and its pipe, as plain Python.

    Kept apart from the QObject so the exit hook can stop the process even
    after Qt has deleted the window that owned it.
    """

    def __init__(self):
        self.process = None
        self.conn = None
        self._lock = threading.Lock()

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def start(self, reader) -> None:
        from supernova.gui._server import serve
        from supernova.gui._workers import without_main_module

        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=serve, args=(child,),
                                  name="supernova-compute")
        with without_main_module():
            process.start()
        _hook_exit()
        child.close()
        self.process, self.conn = process, parent
        threading.Thread(target=reader, args=(parent, process),
                         name="supernova-compute-reader", daemon=True).start()

    def send(self, message) -> None:
        with self._lock:
            self.conn.send(message)

    def stop(self, wait: float = 5.0) -> None:
        process, conn = self.process, self.conn
        self.process = self.conn = None
        if process is None:
            return
        with contextlib.suppress(Exception):
            with self._lock:
                conn.send(("shutdown",))
        process.join(wait)
        if process.is_alive():
            process.terminate()
            process.join(2.0)
        if process.is_alive():
            process.kill()
            process.join(1.0)
        with contextlib.suppress(Exception):
            conn.close()


_LINKS: list[_Link] = []
_hooked = False


def _stop_links() -> None:
    for link in _LINKS:
        link.stop(wait=2.0)


def _hook_exit() -> None:
    """Make the compute processes exit before multiprocessing waits for them.

    multiprocessing registers an exit hook that joins every live child when it
    starts its first process. Exit hooks run last-registered first, so this
    one, registered just after that start, runs before it -- and tells the
    children to exit, rather than leaving that hook to wait on a process that
    is waiting for requests. Without it, a script that ends with the window
    still open never exits.
    """
    global _hooked
    if not _hooked:
        atexit.register(_stop_links)
        _hooked = True


class ComputeClient(QtCore.QObject):
    """Sends jobs to the compute process and reports what comes back."""

    busyChanged = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._link = _Link()
        _LINKS.append(self._link)
        self._jobs: dict[int, Job] = {}     # sent, not yet answered
        self._live: set[Job] = set()        # kept alive until the answer is handled
        self._ids = itertools.count(1)
        self._closing = False

    def start(self) -> None:
        """Start the process now rather than at the first job; it takes seconds."""
        if not self._link.alive:
            self._closing = False
            self._link.start(self._read)

    def submit(self, kind: str, *args, done=None, failed=None, progressed=None,
               logged=None) -> Job:
        job = Job(next(self._ids), kind)
        for signal, slot in ((job.done, done), (job.failed, failed),
                             (job.progressed, progressed), (job.logged, logged)):
            if slot is not None:
                signal.connect(slot)
        # bound methods of an object on the GUI thread: run there, queued,
        # after the caller's own slots
        job.done.connect(self._finished)
        job.failed.connect(self._finished)
        self._jobs[job.id] = job
        self._live.add(job)
        if len(self._jobs) == 1:
            self.busyChanged.emit(True)
        try:
            self.start()
            self._link.send((kind, job.id, *args))
        except Exception as error:
            message = f"Could not reach the compute process: {error}"
            self._jobs.pop(job.id, None)
            QtCore.QTimer.singleShot(0, lambda: job.failed.emit(message, ""))
        return job

    def cancel(self, job: Job | None) -> None:
        """Ask the process to stop ``job``; it still reports, done or failed."""
        if job is not None and job.id in self._jobs and self._link.alive:
            with contextlib.suppress(Exception):
                self._link.send(("cancel", job.id))

    @property
    def busy(self) -> bool:
        return bool(self._jobs)

    def _finished(self, *_) -> None:
        # The reader thread dropped the job from the pending ones when its
        # answer came; only now, with the answer handled, may it be deleted --
        # a queued signal from a deleted object is silently lost.
        self._live.discard(self.sender())
        if not self._jobs:
            self.busyChanged.emit(False)

    def _read(self, conn, process) -> None:
        """The reader thread: one loop per process, until its pipe closes."""
        while True:
            try:
                message = conn.recv()
            except (EOFError, OSError):
                break
            job = self._jobs.get(message[1])
            if job is None:
                continue
            kind = message[0]
            if kind == "progress":
                job.progressed.emit(message[2])
            elif kind == "log":
                job.logged.emit(message[2])
            elif kind == "done":
                self._jobs.pop(job.id, None)
                job.done.emit(message[2])
            elif kind == "failed":
                self._jobs.pop(job.id, None)
                job.failed.emit(message[2], message[3])
        if self._closing:
            return
        # It went without being asked to: nothing pending will ever answer.
        process.join(1.0)
        reason = (f"The compute process stopped (exit code {process.exitcode}); "
                  "it restarts with the next request, without the equilibria it held.")
        for ident in list(self._jobs):
            job = self._jobs.pop(ident, None)
            if job is not None:
                job.failed.emit(reason, "")

    def shutdown(self) -> None:
        """Stop the process; whatever is running is abandoned."""
        self._closing = True
        self._jobs.clear()
        self._live.clear()
        self._link.stop()


# ----------------------------------------------------------------------------
# Widgets
# ----------------------------------------------------------------------------

def figure_page(window, figsize=(6.4, 4.8)):
    """A figure with its navigation toolbar, ready to drop into a layout."""
    figure = Figure(figsize=figsize, layout="constrained")
    canvas = FigureCanvasQTAgg(figure)
    page = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(NavigationToolbar2QT(canvas, window))
    layout.addWidget(canvas, 1)
    return page, figure, canvas


def toolbar_is_busy(canvas) -> bool:
    """True while the navigation toolbar's pan or zoom is active."""
    return canvas.toolbar is not None and bool(canvas.toolbar.mode)


def double_spin(low, high, value, decimals, step=None, tooltip=None):
    spin = QtWidgets.QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setDecimals(decimals)
    spin.setValue(value)
    if step is not None:
        spin.setSingleStep(step)
    if tooltip:
        spin.setToolTip(tooltip)
    return spin


def int_spin(low, high, value, step=1, tooltip=None):
    spin = QtWidgets.QSpinBox()
    spin.setRange(low, high)
    spin.setValue(value)
    spin.setSingleStep(step)
    if tooltip:
        spin.setToolTip(tooltip)
    return spin


class SciEdit(QtWidgets.QLineEdit):
    """A float field that reads and writes scientific notation.

    QDoubleSpinBox rounds to a fixed number of decimals, which turns a
    tolerance of 1e-10 into 0.
    """

    valueChanged = QtCore.Signal(float)

    def __init__(self, value: float = 0.0, *, minimum=None, tooltip=None, parent=None):
        super().__init__(parent)
        validator = QtGui.QDoubleValidator(self)
        validator.setNotation(QtGui.QDoubleValidator.ScientificNotation)
        if minimum is not None:
            validator.setBottom(minimum)
        self.setValidator(validator)
        self._value = float(value)
        self.setText(f"{self._value:g}")
        if tooltip:
            self.setToolTip(tooltip)
        self.editingFinished.connect(self._commit)

    def value(self) -> float:
        return self._value

    def setValue(self, value: float) -> None:
        self._value = float(value)
        self.setText(f"{self._value:g}")

    def _commit(self) -> None:
        try:
            value = float(self.text())
        except ValueError:
            self.setText(f"{self._value:g}")
            return
        if value != self._value:
            self._value = value
            self.valueChanged.emit(value)
        self.setText(f"{self._value:g}")


class Collapsible(QtWidgets.QWidget):
    """A titled section whose body folds away; for the long tail of knobs."""

    def __init__(self, title: str, body: QtWidgets.QWidget, *, expanded=False,
                 parent=None):
        super().__init__(parent)
        self.button = QtWidgets.QToolButton()
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setChecked(expanded)
        self.button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.button.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.body = body
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.button)
        layout.addWidget(body)
        self.button.toggled.connect(self._toggle)
        self._toggle(expanded)

    def _toggle(self, expanded: bool) -> None:
        self.button.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self.body.setVisible(expanded)


def _humanize(name: str) -> str:
    return name.replace("_", " ")


class DataclassForm(QtWidgets.QWidget):
    """An editor for an options dataclass, built from its fields.

    The value is the set of overrides -- fields that differ from the
    dataclass default -- so a session records only what the user changed,
    and a field added to the class later shows up here on its own.
    """

    changed = QtCore.Signal()

    def __init__(self, cls, *, skip=(), parent=None):
        super().__init__(parent)
        self.cls = cls
        self._defaults = {}
        self._editors = {}
        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(4, 2, 4, 2)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        for item in fields(cls):
            if item.name in skip or item.default is MISSING:
                continue
            default = item.default
            self._defaults[item.name] = default
            editor = self._editor_for(default)
            self._editors[item.name] = editor
            label = QtWidgets.QLabel(_humanize(item.name))
            label.setToolTip(item.name)
            form.addRow(label, editor)
        reset = QtWidgets.QPushButton("Reset to defaults")
        reset.clicked.connect(self.reset)
        form.addRow(reset)

    def _editor_for(self, default):
        if isinstance(default, bool):
            editor = QtWidgets.QCheckBox()
            editor.setChecked(default)
            editor.toggled.connect(self.changed)
        elif isinstance(default, int):
            editor = int_spin(-10**9, 10**9, default)
            editor.valueChanged.connect(self.changed)
        elif isinstance(default, float):
            editor = SciEdit(default)
            editor.valueChanged.connect(self.changed)
        else:
            editor = QtWidgets.QLineEdit(self._format(default))
            editor.editingFinished.connect(self.changed)
        return editor

    @staticmethod
    def _format(value) -> str:
        if isinstance(value, tuple):
            return ", ".join(f"{item:g}" if isinstance(item, float) else str(item)
                             for item in value)
        return str(value)

    def _read(self, name):
        editor, default = self._editors[name], self._defaults[name]
        if isinstance(editor, QtWidgets.QCheckBox):
            return editor.isChecked()
        if isinstance(editor, (QtWidgets.QSpinBox, SciEdit)):
            return editor.value()
        text = editor.text().strip()
        if isinstance(default, tuple):
            kind = type(default[0]) if default else float
            return tuple(kind(part) for part in text.split(",") if part.strip())
        return text

    def _write(self, name, value) -> None:
        editor = self._editors[name]
        if isinstance(editor, QtWidgets.QCheckBox):
            editor.setChecked(bool(value))
        elif isinstance(editor, (QtWidgets.QSpinBox, SciEdit)):
            editor.setValue(value)
        else:
            editor.setText(self._format(tuple(value) if isinstance(value, list) else value))

    def overrides(self) -> dict:
        """Fields that differ from the default; raises ValueError on bad text."""
        out = {}
        for name, default in self._defaults.items():
            value = self._read(name)
            if value != default:
                out[name] = list(value) if isinstance(value, tuple) else value
        return out

    def set_overrides(self, overrides: dict) -> None:
        blocked = self.blockSignals(True)
        for name, default in self._defaults.items():
            self._write(name, overrides.get(name, default))
        self.blockSignals(blocked)

    def reset(self) -> None:
        self.set_overrides({})
        self.changed.emit()


def set_rows(table: QtWidgets.QTableWidget, headers, rows, colours=None) -> None:
    """Fill a read-only table. ``colours[i]`` tints row i's last column."""
    table.setSortingEnabled(False)
    table.clear()
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setRowCount(len(rows))
    for row, values in enumerate(rows):
        for column, value in enumerate(values):
            item = QtWidgets.QTableWidgetItem()
            if isinstance(value, (int, np.integer)):
                item.setData(QtCore.Qt.DisplayRole, int(value))
            elif isinstance(value, (float, np.floating)):
                item.setData(QtCore.Qt.DisplayRole, float(value))
            else:
                item.setText(str(value))
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            table.setItem(row, column, item)
        if colours and colours[row]:
            colour = QtGui.QColor(colours[row])
            colour.setAlpha(60)
            table.item(row, len(headers) - 1).setBackground(colour)
    table.resizeColumnsToContents()
