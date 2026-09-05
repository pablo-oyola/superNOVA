"""A small window for computing and looking at the continuum.

The continuum solves in 50-200 ms, so this is a live tool: change a control and
the plot follows. Only the equilibrium is slow -- seconds for a real eqdsk --
so it loads on a worker thread with the reason showing in the status bar, and
is reused for every subsequent edit.

The window owns no physics. It edits a :class:`~supernova.gui._state.
ContinuumState`, hands it to :class:`~supernova.gui._model.ContinuumModel`, and
draws what comes back; the state round-trips through JSON so a session can be
reproduced without the GUI.
"""

from __future__ import annotations

from pathlib import Path
import sys
import traceback

import matplotlib
matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets

from supernova.gui._model import (
    FAMILY_COLOURS,
    FAMILY_NAMES,
    ContinuumModel,
    ContinuumResult,
)
from supernova.gui._state import (
    BRANCH_LABELS,
    BRANCHES,
    COORDINATES,
    ContinuumState,
)

SPECIES = ("H", "D", "T", "He3", "He4")
DEBOUNCE_MS = 250
CURSOR_GRAB_FRACTION = 0.02      # of the y-axis span, for grabbing the line


class _SolveWorker(QtCore.QThread):
    """Runs one solve off the GUI thread."""

    progressed = QtCore.Signal(str)
    solved = QtCore.Signal(object)
    failed = QtCore.Signal(str, str)

    def __init__(self, model: ContinuumModel, state: ContinuumState, parent=None):
        super().__init__(parent)
        self._model = model
        self._state = state

    def run(self) -> None:
        try:
            result = self._model.solve(
                self._state, progress=self.progressed.emit
            )
        except Exception as error:                    # shown, never swallowed
            self.failed.emit(f"{type(error).__name__}: {error}",
                             traceback.format_exc())
        else:
            self.solved.emit(result)


class ContinuumWindow(QtWidgets.QMainWindow):
    """Controls on the left, the continuum on the right."""

    def __init__(self, state: ContinuumState | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("superNOVA - Continuum")
        self.resize(1180, 760)

        self.state = state or ContinuumState()
        self.model = ContinuumModel()
        self.result: ContinuumResult | None = None
        self._worker: _SolveWorker | None = None
        self._restack = False          # a change arrived while solving
        self._explicit = False         # this solve came from the toolbar
        self._cursor_line = None
        self._cut_artists = []
        self._loading_ui = False       # suppress signals while writing widgets

        self._build_ui()
        self._write_state_to_widgets()
        self._schedule_solve()

    # ---------------------------------------------------------------- layout

    def _build_ui(self) -> None:
        self.figure = Figure(figsize=(6.4, 4.8), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        self._draw_placeholder()

        plot_side = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_side)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        plot_layout.addWidget(self.canvas, 1)

        controls = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(controls)
        form.setContentsMargins(10, 10, 10, 10)
        form.setSpacing(10)
        form.addWidget(self._equilibrium_box())
        form.addWidget(self._density_box())
        form.addWidget(self._continuum_box())
        form.addWidget(self._view_box())
        form.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(controls)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(330)
        scroll.setMaximumWidth(430)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(plot_side)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self._build_toolbar()
        self.status = self.statusBar()
        self.status.showMessage("Choose an equilibrium and a density profile.")

        self._dragging = False
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self._start_solve)

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("Actions")
        bar.setMovable(False)
        self._action(bar, "Solve now", self._solve_now, "Ctrl+Return")
        bar.addSeparator()
        self.export_action = self._action(
            bar, "Export data...", self._export_dataset)
        self.figure_action = self._action(bar, "Save figure", self._save_figure)
        self.export_action.setEnabled(False)
        self.figure_action.setEnabled(False)
        bar.addSeparator()
        self._action(bar, "Save session", self._save_session, "Ctrl+S")
        self._action(bar, "Load session", self._load_session, "Ctrl+O")

    def _action(self, bar, text, slot, shortcut=None):
        action = bar.addAction(text)
        action.triggered.connect(slot)
        if shortcut:
            action.setShortcut(shortcut)
        return action

    # ------------------------------------------------------------ the groups

    def _equilibrium_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Equilibrium")
        grid = QtWidgets.QGridLayout(box)

        self.eqdsk_edit = QtWidgets.QLineEdit()
        self.eqdsk_edit.setPlaceholderText("path to an .eqdsk / .geqdsk")
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._browse_eqdsk)
        grid.addWidget(self.eqdsk_edit, 0, 0, 1, 2)
        grid.addWidget(browse, 0, 2)

        self.cocos_spin = QtWidgets.QSpinBox()
        self.cocos_spin.setRange(1, 18)
        self.cocos_spin.setToolTip(
            "COCOS of the file as written. A refined equilibrium records its own "
            "in the sidecar; guessing here is the classic way to get a plausible "
            "but wrong q."
        )
        grid.addWidget(QtWidgets.QLabel("COCOS"), 1, 0)
        grid.addWidget(self.cocos_spin, 1, 1)

        self.boozer_min = self._double(0.0, 0.999, 0.005, 3)
        self.boozer_max = self._double(0.001, 1.0, 0.995, 3)
        grid.addWidget(QtWidgets.QLabel("Boozer domain"), 2, 0)
        grid.addWidget(self.boozer_min, 2, 1)
        grid.addWidget(self.boozer_max, 2, 2)

        for widget in (self.eqdsk_edit, self.cocos_spin,
                       self.boozer_min, self.boozer_max):
            self._connect(widget)
        return box

    def _density_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Density profile")
        grid = QtWidgets.QGridLayout(box)

        self.density_edit = QtWidgets.QLineEdit()
        self.density_edit.setPlaceholderText("two-column profile file")
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._browse_density)
        grid.addWidget(self.density_edit, 0, 0, 1, 2)
        grid.addWidget(browse, 0, 2)

        self.density_x = QtWidgets.QComboBox()
        self.density_x.addItems(COORDINATES)
        self.density_x.setToolTip("What the file's first column is.")
        grid.addWidget(QtWidgets.QLabel("column is"), 1, 0)
        grid.addWidget(self.density_x, 1, 1, 1, 2)

        self.density_interp = QtWidgets.QComboBox()
        self.density_interp.addItems(COORDINATES)
        self.density_interp.setToolTip(
            "Which variable the interpolation runs in. Not the same choice as "
            "the column, and not equivalent: the existing case scripts read a "
            "psi_n column and interpolate in rho_pol."
        )
        grid.addWidget(QtWidgets.QLabel("interpolate in"), 2, 0)
        grid.addWidget(self.density_interp, 2, 1, 1, 2)

        self.density_units = QtWidgets.QLineEdit()
        grid.addWidget(QtWidgets.QLabel("units"), 3, 0)
        grid.addWidget(self.density_units, 3, 1, 1, 2)

        for widget in (self.density_edit, self.density_x,
                       self.density_interp, self.density_units):
            self._connect(widget)
        return box

    def _continuum_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Continuum")
        grid = QtWidgets.QGridLayout(box)

        self.ntor_spin = QtWidgets.QSpinBox()
        self.ntor_spin.setRange(-64, 64)
        grid.addWidget(QtWidgets.QLabel("n"), 0, 0)
        grid.addWidget(self.ntor_spin, 0, 1)

        self.gamma_combo = QtWidgets.QComboBox()
        self.gamma_combo.addItems(["0", "5/3"])
        self.gamma_combo.setToolTip(
            "Exactly 0 removes the sound coupling; it is a different problem, "
            "not a limit."
        )
        grid.addWidget(QtWidgets.QLabel("gamma_s"), 0, 2)
        grid.addWidget(self.gamma_combo, 0, 3)

        self.species_combo = QtWidgets.QComboBox()
        self.species_combo.addItems(SPECIES)
        grid.addWidget(QtWidgets.QLabel("species"), 1, 0)
        grid.addWidget(self.species_combo, 1, 1)

        self.which_combo = QtWidgets.QComboBox()
        for key in BRANCHES:
            self.which_combo.addItem(BRANCH_LABELS[key], key)
        grid.addWidget(QtWidgets.QLabel("branches"), 1, 2)
        grid.addWidget(self.which_combo, 1, 3)

        self.classify_check = QtWidgets.QCheckBox(
            "label families (Alfvenic / sound / mixed)")
        self.classify_check.setToolTip(
            "Classifies every root from its eigenvector. Needs the full "
            "spectrum and costs about a second rather than 0.1 s, so it is "
            "off by default."
        )
        grid.addWidget(self.classify_check, 3, 0, 1, 4)

        self.mmax_spin = QtWidgets.QSpinBox()
        self.mmax_spin.setRange(1, 512)
        self.mmax_spin.setToolTip("Poloidal harmonics retained. Cost is flat in mmax.")
        grid.addWidget(QtWidgets.QLabel("mmax"), 2, 0)
        grid.addWidget(self.mmax_spin, 2, 1)

        self.npsi_spin = QtWidgets.QSpinBox()
        self.npsi_spin.setRange(8, 4096)
        self.npsi_spin.setSingleStep(32)
        self.npsi_spin.setToolTip("Radial points. This is what the solve time scales with.")
        grid.addWidget(QtWidgets.QLabel("radial points"), 2, 2)
        grid.addWidget(self.npsi_spin, 2, 3)

        for widget in (self.ntor_spin, self.gamma_combo, self.species_combo,
                       self.which_combo, self.mmax_spin, self.npsi_spin,
                       self.classify_check):
            self._connect(widget)
        self.which_combo.currentIndexChanged.connect(self._sync_classify_enabled)
        return box

    def _sync_classify_enabled(self, *_) -> None:
        """Family labels only mean something for the full spectrum."""
        full = self.which_combo.currentData() == "full"
        self.classify_check.setEnabled(full)
        if not full and self.classify_check.isChecked():
            self.classify_check.setChecked(False)

    def _view_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("View")
        grid = QtWidgets.QGridLayout(box)

        self.fmax_spin = self._double(0.0, 1.0e6, 0.0, 1)
        self.fmax_spin.setToolTip("Upper frequency limit. 0 autoscales to the 99th percentile.")
        grid.addWidget(QtWidgets.QLabel("f max [kHz]"), 0, 0)
        grid.addWidget(self.fmax_spin, 0, 1, 1, 2)

        self.band_low = self._double(0.0, 1.0e6, 0.0, 1)
        self.band_high = self._double(0.0, 1.0e6, 0.0, 1)
        grid.addWidget(QtWidgets.QLabel("shade band"), 1, 0)
        grid.addWidget(self.band_low, 1, 1)
        grid.addWidget(self.band_high, 1, 2)

        self.cursor_spin = self._double(0.0, 1.0e6, 0.0, 2)
        self.cursor_spin.setToolTip(
            "A mode at this frequency. Drag the dashed line on the plot, or "
            "type here. 0 hides it."
        )
        grid.addWidget(QtWidgets.QLabel("mode f [kHz]"), 2, 0)
        grid.addWidget(self.cursor_spin, 2, 1, 1, 2)

        self.cuts_label = QtWidgets.QLabel("Drag the dashed line to place a mode.")
        self.cuts_label.setWordWrap(True)
        self.cuts_label.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse)
        self.cuts_label.setStyleSheet("color: #4B5A72;")
        grid.addWidget(self.cuts_label, 3, 0, 1, 3)

        for widget in (self.fmax_spin, self.band_low, self.band_high):
            self._connect(widget)
        # The cursor never re-solves; it only redraws what is already computed.
        self.cursor_spin.valueChanged.connect(self._on_cursor_changed)
        return box

    def _double(self, low, high, value, decimals):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin

    def _connect(self, widget) -> None:
        """Route each widget's 'the user changed this' signal to one slot."""
        if isinstance(widget, QtWidgets.QLineEdit):
            widget.editingFinished.connect(self._on_edit)
        elif isinstance(widget, QtWidgets.QComboBox):
            widget.currentIndexChanged.connect(self._on_edit)
        elif isinstance(widget, QtWidgets.QAbstractButton):
            widget.toggled.connect(self._on_edit)
        else:
            widget.valueChanged.connect(self._on_edit)

    # ------------------------------------------------------------- state I/O

    def _write_state_to_widgets(self) -> None:
        self._loading_ui = True
        s = self.state
        self.eqdsk_edit.setText(s.eqdsk)
        self.cocos_spin.setValue(s.cocos_in)
        self.boozer_min.setValue(s.boozer_min)
        self.boozer_max.setValue(s.boozer_max)
        self.density_edit.setText(s.density_file)
        self.density_units.setText(s.density_units)
        self.density_x.setCurrentText(s.density_x)
        self.density_interp.setCurrentText(s.density_interpolate_in)
        self.ntor_spin.setValue(s.ntor)
        self.gamma_combo.setCurrentText("0" if s.gamma_s == 0.0 else "5/3")
        self.species_combo.setCurrentText(s.species)
        index = self.which_combo.findData(s.which)
        self.which_combo.setCurrentIndex(max(index, 0))
        self.classify_check.setChecked(s.classify)
        self.mmax_spin.setValue(s.mmax)
        self.npsi_spin.setValue(s.npsi)
        self.fmax_spin.setValue(s.fmax_khz)
        self.band_low.setValue(s.band_low_khz)
        self.band_high.setValue(s.band_high_khz)
        self.cursor_spin.setValue(s.cursor_khz)
        self._sync_classify_enabled()
        self._loading_ui = False

    def _read_widgets_into_state(self) -> None:
        s = self.state
        s.eqdsk = self.eqdsk_edit.text().strip()
        s.cocos_in = self.cocos_spin.value()
        s.boozer_min = self.boozer_min.value()
        s.boozer_max = self.boozer_max.value()
        s.density_file = self.density_edit.text().strip()
        s.density_units = self.density_units.text().strip() or "m^-3"
        s.density_x = self.density_x.currentText()
        s.density_interpolate_in = self.density_interp.currentText()
        s.ntor = self.ntor_spin.value()
        s.gamma_s = 0.0 if self.gamma_combo.currentText() == "0" else 5.0 / 3.0
        s.species = self.species_combo.currentText()
        s.which = self.which_combo.currentData()
        s.classify = self.classify_check.isChecked()
        s.mmax = self.mmax_spin.value()
        s.npsi = self.npsi_spin.value()
        s.fmax_khz = self.fmax_spin.value()
        s.band_low_khz = self.band_low.value()
        s.band_high_khz = self.band_high.value()
        s.cursor_khz = self.cursor_spin.value()

    # -------------------------------------------------------------- solving

    def _on_edit(self, *_) -> None:
        if self._loading_ui:
            return
        self._read_widgets_into_state()
        self._schedule_solve()

    def _schedule_solve(self) -> None:
        problems = self.state.validate()
        if problems:
            self.status.showMessage(problems[0])
            return
        self._debounce.start()

    def _solve_now(self) -> None:
        """Toolbar action: solve, and say loudly if it fails."""
        self._explicit = True
        self._start_solve()

    def _start_solve(self) -> None:
        self._read_widgets_into_state()
        problems = self.state.validate()
        if problems:
            self.status.showMessage(problems[0])
            return
        if self._worker is not None and self._worker.isRunning():
            self._restack = True          # coalesce; re-run when this finishes
            return
        self.status.showMessage("Solving...")
        self._worker = _SolveWorker(self.model, ContinuumState(**vars(self.state)))
        self._worker.progressed.connect(self.status.showMessage)
        self._worker.solved.connect(self._on_solved)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.start()

    def _on_worker_done(self) -> None:
        if self._restack:
            self._restack = False
            self._start_solve()

    def _on_solved(self, result: ContinuumResult) -> None:
        self._explicit = False
        self.result = result
        self.export_action.setEnabled(True)
        self.figure_action.setEnabled(True)
        self._draw(result)
        note = []
        if result.reloaded_equilibrium:
            note.append("equilibrium reloaded")
        elif result.rebuilt_continuum:
            note.append("continuum rebuilt")
        if result.classify_note:
            note.append("families unavailable")
            self.cuts_label.setText(
                f"<span style='color:#9C5D14'>{result.classify_note}</span>")
        self.status.showMessage(
            f"{result.n_branches} branches in {result.solve_seconds * 1e3:.0f} ms"
            f"  |  n = {result.ntor}, gamma_s = {result.gamma_s:g}"
            f"  |  q {result.q_min:.2f} - {result.q_max:.2f}"
            + ("  |  " + ", ".join(note) if note else "")
        )

    def _on_failed(self, message: str, detail: str) -> None:
        """Always report in the status bar; interrupt only when asked to solve.

        Edits are debounced and re-solve on their own, so a half-typed path
        would otherwise raise a modal dialog on every keystroke.
        """
        self.status.showMessage(message)
        self.cuts_label.setText(
            f"<span style='color:#A33636'>{message}</span>")
        if not self._explicit:
            return
        self._explicit = False
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("The continuum could not be solved")
        box.setText(message)
        box.setDetailedText(detail)
        box.exec()

    # -------------------------------------------------------------- drawing

    def _draw_placeholder(self) -> None:
        self.axes.clear()
        self._forget_cursor_artists()
        self.axes.text(0.5, 0.5, "Choose an equilibrium and a density profile",
                       ha="center", va="center", transform=self.axes.transAxes,
                       color="0.45")
        self.axes.set_xlabel(r"$\psi_N$")
        self.axes.set_ylabel("f [kHz]")
        self.axes.set_xlim(0.0, 1.0)
        self.axes.grid(alpha=0.25)

    def _draw(self, result: ContinuumResult) -> None:
        self.axes.clear()
        self._forget_cursor_artists()
        khz = result.frequency_khz

        if result.family_code is None:
            self.axes.plot(result.psi_n, khz, lw=0.9, alpha=0.85, color="#2B5EA6")
        else:
            # One scatter per family so the legend names them. Marker colour is
            # the only encoding here; the branches themselves are not tracked
            # across radius, so joining them with lines would imply a
            # continuity the classifier does not claim.
            psi = np.repeat(result.psi_n[:, None], khz.shape[1], axis=1)
            for code in np.unique(result.family_code):
                mask = (result.family_code == code) & np.isfinite(khz)
                if not mask.any():
                    continue
                self.axes.plot(
                    psi[mask], khz[mask], linestyle="none", marker=".",
                    markersize=2.6, alpha=0.8,
                    color=FAMILY_COLOURS.get(int(code), "#78879D"),
                    label=FAMILY_NAMES.get(int(code), str(code)),
                )

        if self.state.band_high_khz > self.state.band_low_khz:
            self.axes.axhspan(self.state.band_low_khz, self.state.band_high_khz,
                              color="#0E7466", alpha=0.10, zorder=0,
                              label="band")

        finite = khz[np.isfinite(khz)]
        if self.state.fmax_khz > 0.0:
            top = self.state.fmax_khz
        elif finite.size:
            # The full spectrum reaches hundreds of kHz; autoscaling to the
            # maximum squashes everything anyone wants to look at.
            top = float(np.percentile(finite, 99.0)) * 1.05
        else:
            top = 1.0
        self.axes.set_ylim(0.0, max(top, 1.0e-6))
        self.axes.set_xlim(0.0, 1.0)
        self.axes.set_xlabel(r"$\psi_N$")
        self.axes.set_ylabel("f [kHz]")
        self.axes.set_title(
            f"{result.which.replace('_', ' ')} continuum, "
            f"n = {result.ntor}, $\\gamma_s$ = {result.gamma_s:g}"
        )
        self.axes.grid(alpha=0.25)
        self._draw_cursor(result)
        if self.axes.get_legend_handles_labels()[0]:
            self.axes.legend(loc="upper left", frameon=False, fontsize=8,
                             markerscale=4)
        self.canvas.draw_idle()

    def _forget_cursor_artists(self) -> None:
        """Drop the cursor artists, whether or not the axes still holds them.

        ``Axes.clear()`` detaches every artist without telling anyone who kept
        a reference, and matplotlib refuses to remove an artist twice. So a
        redraw that begins with clear() would otherwise leave this list full of
        artists that raise NotImplementedError on removal -- which is what
        happened when the branch selector changed while a mode line was up.
        """
        for artist in self._cut_artists:
            try:
                artist.remove()
            except (NotImplementedError, ValueError):
                pass                      # already detached by Axes.clear()
        self._cut_artists = []
        self._cursor_line = None

    def _draw_cursor(self, result: ContinuumResult) -> None:
        """The mode line and the places it meets the continuum."""
        self._forget_cursor_artists()

        frequency = self.state.cursor_khz
        if frequency <= 0.0:
            self.cuts_label.setText("Drag the dashed line to place a mode.")
            return

        self._cursor_line = self.axes.axhline(
            frequency, color="#A33636", lw=1.3, ls="--", zorder=5)
        self._cut_artists.append(self._cursor_line)

        cuts = result.cuts(frequency)
        if cuts.size:
            marks = self.axes.plot(
                cuts, np.full(cuts.size, frequency), linestyle="none",
                marker="o", markersize=6, markerfacecolor="none",
                markeredgecolor="#A33636", markeredgewidth=1.4, zorder=6)
            self._cut_artists.extend(marks)
            listed = ", ".join(f"{value:.3f}" for value in cuts[:12])
            more = "" if cuts.size <= 12 else f" (+{cuts.size - 12} more)"
            where = "radius" if cuts.size == 1 else "radii"
            self.cuts_label.setText(
                f"<b>{frequency:.2f} kHz cuts the continuum at "
                f"{cuts.size} {where}:</b><br>psi_N = {listed}{more}"
            )
        else:
            self.cuts_label.setText(
                f"<b>{frequency:.2f} kHz is gap-resonance free</b><br>"
                "no continuum crossing at this frequency."
            )

    # ------------------------------------------------------ cursor dragging

    def _on_press(self, event) -> None:
        """Grab the mode line, or place it if it is not shown yet."""
        if event.inaxes is not self.axes or event.ydata is None:
            return
        if self.canvas.toolbar is not None and self.canvas.toolbar.mode:
            return                      # pan/zoom is active; leave it alone
        low, high = self.axes.get_ylim()
        tolerance = CURSOR_GRAB_FRACTION * (high - low)
        near = (self.state.cursor_khz > 0.0
                and abs(event.ydata - self.state.cursor_khz) <= tolerance)
        if near or self.state.cursor_khz <= 0.0:
            self._dragging = True
            self._set_cursor(event.ydata)

    def _on_motion(self, event) -> None:
        if not self._dragging or event.inaxes is not self.axes:
            return
        if event.ydata is not None:
            self._set_cursor(event.ydata)

    def _on_release(self, _event) -> None:
        self._dragging = False

    def _set_cursor(self, frequency_khz: float) -> None:
        """Move the mode line. Never re-solves -- the arrays already exist."""
        frequency_khz = float(max(0.0, frequency_khz))
        self.state.cursor_khz = frequency_khz
        self._loading_ui = True
        self.cursor_spin.setValue(frequency_khz)
        self._loading_ui = False
        if self.result is not None:
            self._draw_cursor(self.result)
            self.canvas.draw_idle()

    def _on_cursor_changed(self, value: float) -> None:
        if self._loading_ui:
            return
        self._set_cursor(value)

    # -------------------------------------------------------------- actions

    def _browse_eqdsk(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open equilibrium", self.eqdsk_edit.text() or str(Path.cwd()),
            "Equilibria (*.eqdsk *.geqdsk *.gfile);;All files (*)")
        if path:
            self.eqdsk_edit.setText(path)
            self._on_edit()

    def _browse_density(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open density profile",
            self.density_edit.text() or str(Path.cwd()),
            "Profiles (*.dat *.txt *.csv);;All files (*)")
        if path:
            self.density_edit.setText(path)
            self._on_edit()

    def _export_dataset(self) -> None:
        if self.result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export continuum", "continuum.h5",
            "HDF5 (*.h5 *.hdf5);;NetCDF (*.nc);;All files (*)")
        if not path:
            return
        try:
            engine = self.model.save(self.result, path)
        except Exception as error:
            self.status.showMessage(f"Could not write {path}: {error}")
            return
        self.status.showMessage(f"Wrote {path} via {engine}")

    def _save_figure(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save figure", "continuum.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self.figure.savefig(path, dpi=180, bbox_inches="tight")
            self.status.showMessage(f"Wrote {path}")

    def _save_session(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save session", "continuum_session.json", "JSON (*.json)")
        if path:
            self._read_widgets_into_state()
            self.state.to_json(path)
            self.status.showMessage(f"Wrote {path}")

    def _load_session(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load session", str(Path.cwd()), "JSON (*.json)")
        if not path:
            return
        try:
            self.state = ContinuumState.from_json(path)
        except (ValueError, OSError) as error:
            self.status.showMessage(f"Could not load session: {error}")
            return
        self._write_state_to_widgets()
        self._schedule_solve()


def main(argv=None) -> int:
    """Open the window. Returns the Qt exit code."""
    argv = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv[:1])
    state = None
    if len(argv) > 1 and argv[1].endswith(".json"):
        state = ContinuumState.from_json(argv[1])
    window = ContinuumWindow(state)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
