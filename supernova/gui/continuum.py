"""The continuum tab: look at the continuum of the loaded equilibrium.

The continuum solves in 50-200 ms, so this is a live tool: change a control and
the plot follows. It is solved on the equilibrium the Equilibrium tab loaded --
the same object, held by the compute process -- so there is nothing here to
load, and no second copy that could drift from it.

The tab owns no physics. It edits a :class:`~supernova.gui._state.
ContinuumState` and sends it to the compute process, where a
:class:`~supernova.gui._model.ContinuumModel` solves it on that equilibrium;
it draws what comes back. Its state goes into the application's session file
(:mod:`supernova.gui._session`), so a session can be reproduced without the GUI.
"""

from __future__ import annotations

import sys

import matplotlib
matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets

from supernova.gui._qt import (
    ComputeClient,
    HomeToolbar,
    ViewMemory,
    ask_session_to_open,
    ask_session_to_save,
)
from supernova.gui._model import (
    FAMILY_COLOURS,
    FAMILY_NAMES,
    ContinuumModel,
    ContinuumResult,
)
from supernova.gui._state import (
    BRANCH_LABELS,
    BRANCHES,
    ContinuumState,
)
from supernova.gui._session import read_session, write_session

SPECIES = ("H", "D", "T", "He3", "He4")
DEBOUNCE_MS = 250
CURSOR_GRAB_FRACTION = 0.02      # of the y-axis span, for grabbing the line


class ContinuumWindow(QtWidgets.QMainWindow):
    """Controls on the left, the continuum on the right."""

    def __init__(self, state: ContinuumState | None = None,
                 compute: ComputeClient | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("superNOVA - Continuum")
        self.resize(1180, 760)

        self.state = state or ContinuumState()
        self.model = ContinuumModel()  # only for writing results out
        self.view = None               # EquilibriumView from the Equilibrium tab
        self.result: ContinuumResult | None = None
        self.session_owner = None      # in the app: the window whose session the buttons save
        self._owns_compute = compute is None
        self.compute = compute or ComputeClient(self)
        self._solving = False
        self._job = None
        self._job_handle = None        # the equilibrium the running solve is on
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
        self.canvas.view_memory = ViewMemory()
        self.axes = self.figure.add_subplot(111)
        self._draw_placeholder()

        plot_side = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_side)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(HomeToolbar(self.canvas, self, self._redraw))
        plot_layout.addWidget(self.canvas, 1)

        controls = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(controls)
        form.setContentsMargins(10, 10, 10, 10)
        form.setSpacing(10)
        form.addWidget(self._equilibrium_box())
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
        self.status.showMessage("Load an equilibrium in the Equilibrium tab.")

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
        layout = QtWidgets.QVBoxLayout(box)
        self.equilibrium_label = QtWidgets.QLabel()
        self.equilibrium_label.setWordWrap(True)
        layout.addWidget(self.equilibrium_label)
        self._refresh_equilibrium_label()
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
        self.species_combo.setEnabled(False)
        self.species_combo.setToolTip(
            "Set in the Equilibrium tab, next to the density: every tab uses the "
            "same ions.")
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
        self.mmax_spin.setRange(0, 512)
        self.mmax_spin.setSpecialValueText("auto")
        self.mmax_spin.setToolTip(
            "Poloidal harmonics retained. Cost is flat in mmax. auto (0) is "
            "ceil(n q) at rho_pol = 0.95, which follows n.")
        grid.addWidget(QtWidgets.QLabel("mmax"), 2, 0)
        grid.addWidget(self.mmax_spin, 2, 1)

        self.npsi_spin = QtWidgets.QSpinBox()
        self.npsi_spin.setRange(8, 4096)
        self.npsi_spin.setSingleStep(32)
        self.npsi_spin.setToolTip("Radial points. This is what the solve time scales with.")
        grid.addWidget(QtWidgets.QLabel("radial points"), 2, 2)
        grid.addWidget(self.npsi_spin, 2, 3)

        for widget in (self.ntor_spin, self.gamma_combo,
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

        # f max sets the axis explicitly, so it replaces a zoom (connected first,
        # so the redraw it triggers already uses it)
        self.fmax_spin.valueChanged.connect(self.canvas.view_memory.forget)
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
        s.ntor = self.ntor_spin.value()
        s.gamma_s = 0.0 if self.gamma_combo.currentText() == "0" else 5.0 / 3.0
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
        problems = self._problems()
        if problems:
            self.status.showMessage(problems[0])
            return
        self._debounce.start()

    def _problems(self) -> list[str]:
        problems = self.state.validate()
        if self.view is None:
            problems.insert(0, "Load an equilibrium in the Equilibrium tab.")
        elif self.view.density_is_placeholder:
            problems.insert(0, "The equilibrium has no density profile, so the kHz "
                               "scale would be wrong. Add one in the Equilibrium tab.")
        return problems

    def _solve_now(self) -> None:
        """Toolbar action: solve, and say loudly if it fails."""
        self._explicit = True
        self._start_solve()

    def _start_solve(self) -> None:
        self._read_widgets_into_state()
        problems = self._problems()
        if problems:
            self.status.showMessage(problems[0])
            return
        if self._solving:
            self._restack = True          # coalesce; re-run when this finishes
            return
        self._solving = True
        self._job_handle = self.view.handle
        self.status.showMessage("Solving...")
        self._job = self.compute.submit(
            "continuum", self.view.handle, self.state.to_dict(),
            done=self._on_done, failed=self._on_job_failed,
            progressed=self.status.showMessage)

    def _finish_job(self) -> bool:
        """End the running job; True if its answer is still wanted."""
        self._solving = False
        current = self.view is not None and self._job_handle == self.view.handle
        if self._restack or not current:
            self._restack = False
            self._start_solve()
        return current

    def _on_done(self, result: ContinuumResult) -> None:
        if self._finish_job():
            self._on_solved(result)

    def _on_job_failed(self, message: str, detail: str) -> None:
        if self._finish_job():
            self._on_failed(message, detail)

    # ---------------------------------------------------------- equilibrium

    def set_equilibrium(self, view) -> None:
        """Take the equilibrium the Equilibrium tab passes on, and solve on it.

        Its ions come with it: the species is the Equilibrium tab's.
        """
        self.view = view
        if self.state.species != view.species:
            self.state.species = view.species
            self._write_state_to_widgets()
        self._refresh_equilibrium_label()
        self._schedule_solve()

    def _refresh_equilibrium_label(self) -> None:
        view = self.view
        if view is None:
            self.equilibrium_label.setText(
                "<span style='color:#B45309'>Nothing loaded. Load an equilibrium "
                "in the Equilibrium tab; this tab solves on that one.</span>")
            return
        s = view.summary
        text = (f"<b>{_escape(view.title)}</b><br>q {s.get('q_min', float('nan')):.3f}"
                f" - {s.get('q_edge', float('nan')):.3f}, B<sub>axis</sub> = "
                f"{s['B_axis_T']:.3f} T, {view.psi_n.size} x {view.theta.size} Boozer grid")
        if view.density_is_placeholder:
            text += ("<br><span style='color:#A33636'>No density profile: the kHz scale "
                     "would be wrong. Give the Equilibrium tab a density file.</span>")
        elif "n_axis_m3" in s:
            text += (f", n<sub>axis</sub> = {s['n_axis_m3']:.3g} m<sup>-3</sup> of "
                     f"{_escape(view.species)}")
        self.equilibrium_label.setText(text)

    def _on_solved(self, result: ContinuumResult) -> None:
        self._explicit = False
        self.result = result
        self.export_action.setEnabled(True)
        self.figure_action.setEnabled(True)
        if result.reloaded_equilibrium:
            # a new equilibrium: the zoom chosen on the old one no longer applies
            self.canvas.view_memory.forget()
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
        self.axes.text(0.5, 0.5, "Load an equilibrium in the Equilibrium tab",
                       ha="center", va="center", transform=self.axes.transAxes,
                       color="0.45")
        self.axes.set_xlabel(r"$\psi_N$")
        self.axes.set_ylabel("f [kHz]")
        self.axes.set_xlim(0.0, 1.0)
        self.axes.grid(alpha=0.25)

    def _draw(self, result: ContinuumResult) -> None:
        memory = self.canvas.view_memory
        with memory.drawing():
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
            memory.keep(self.axes, "continuum")
            self.canvas.draw_idle()

    def _redraw(self) -> None:
        """Draw what is computed again, with the default limits (the toolbar's Home)."""
        if self.result is not None:
            self._draw(self.result)
        else:
            self._draw_placeholder()
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

    def session_results(self) -> dict:
        """The continuum on show, filed under its equilibrium's key.

        It goes into a session for the record; opening the session solves it
        again, which takes milliseconds once the equilibrium is loaded.
        """
        if self.result is None or self.view is None:
            return {}
        return {"result": self.result, "equilibrium_key": self.view.key}

    def _save_session(self) -> None:
        if self.session_owner is not None:      # in the app: the whole session
            self.session_owner.save_session()
            return
        path = ask_session_to_save(self, "Save session", "continuum_session.h5")
        if not path:
            return
        self._read_widgets_into_state()
        try:
            write_session(path, continuum=self.state,
                          results={"continuum": self.session_results()})
        except (ValueError, OSError, TypeError) as error:
            self.status.showMessage(f"Could not save the session: {error}")
            return
        self.status.showMessage(f"Wrote {path}")

    def _load_session(self) -> None:
        if self.session_owner is not None:
            self.session_owner.open_session()
            return
        path = ask_session_to_open(self, "Load session")
        if not path:
            return
        try:
            loaded = read_session(path)
        except (ValueError, OSError, TypeError) as error:
            self.status.showMessage(f"Could not load session: {error}")
            return
        if "continuum" not in loaded:
            self.status.showMessage(f"{path} holds no continuum settings.")
            return
        self.state = loaded["continuum"]
        if self.view is not None:
            self.state.species = self.view.species    # the Equilibrium tab's
        self._write_state_to_widgets()
        self._schedule_solve()
        if "continuum_equilibrium" in loaded:
            self.status.showMessage(
                "Loaded the continuum settings. The session's equilibrium file and "
                "density belong in the Equilibrium tab now: open it there.")

    def shutdown(self) -> None:
        if self._solving:
            self.compute.cancel(self._job)
        if self._owns_compute:
            self.compute.shutdown()

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)


def _escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def main(argv=None) -> int:
    """Open superNOVA on this tab: it needs the Equilibrium tab's equilibrium."""
    from supernova.gui.app import main as app_main

    return app_main(argv, start_tab="Continuum")


if __name__ == "__main__":
    raise SystemExit(main())
