"""The solver tab: the simple pass, every knob it has, and what it did.

It solves on the equilibrium the equilibrium tab passes on: one radial grid,
the band cut into windows, one shift-and-invert solve per window, and a mode
filter on every root found -- Gerrit Kramer's MODELOCATE, or the global
complex-path test with MODELOCATE's spike, core and edge tests kept. Before a
solve it shows the continuum for the chosen n, how the band will be cut, and
the gaps in it: hover a gap to see how wide its modes should be and what
element size resolves them, which the resolution guide also lists against the
current grid. After one it shows the modes -- gap modes against the ones the
filter rejects, and why -- and what each window did: how many roots converged
against how many were asked for, and why SLEPc stopped. That is where a search
goes wrong: a window that converges nev roots may be hiding more, and one that
hits maxit found nothing. Selected modes can then be refined: re-solved on
meshes graded towards them until their frequency converges.

The window owns no physics. It edits a :class:`~supernova.gui._solver.
SolverState` and sends it to the compute process (:mod:`supernova.gui.
_server`), which assembles the pencil once and spreads the windows over a pool
of worker processes; it draws the :class:`~supernova.gui._solver.
SolverResult` that comes back. Nothing slow runs in this process, so the
window stays responsive through a solve of any length.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from PySide6 import QtCore, QtGui, QtWidgets

from supernova.app.schema import FIELD_TITLES
from supernova.gui._qt import (
    ERROR,
    MUTED,
    WARNING,
    Collapsible,
    ComputeClient,
    DataclassForm,
    SciEdit,
    double_spin,
    figure_page,
    int_spin,
    set_rows,
    toolbar_is_busy,
)
from supernova.gui._equilibrium import same_equilibrium
from supernova.gui._solver import (
    FACTORIZATIONS,
    FIELDS,
    FILTER_TITLES,
    LOCATE_ORDER,
    LOCATE_STYLE,
    MODE_FILTERS,
    PARTS,
    PEP_TYPES,
    SOUND_MODELS,
    SOUND_TITLES,
    SPECIES,
    ST_TYPES,
    STATUS_COLOURS,
    STATUS_HINTS,
    WHICH,
    ContinuumPreview,
    RefineResult,
    SolverModel,
    SolverResult,
    SolverState,
    locate_family,
)
from supernova.solve.modelocate import ModeLocateThresholds

PREVIEW_DEBOUNCE_MS = 400
PICK_PIXELS = 12
CONTINUUM_COLOUR = "#708090"
SELECTED_COLOUR = "#A33636"
MODE_HEADERS = ("mode", "f [kHz]", "Im/Re omega", "filter", "peak rho_pol",
                "centre psi_N", "width", "dominant m", "backward error", "window")
REFINE_HEADERS = ("mode", "mesh", "elements", "h at mode [mm]", "f [kHz]", "change [kHz]",
                  "Im/Re omega", "correlation", "FWHM [mm]", "backward error")
GAP_COLOUR = "#7A4FB5"
RESOLUTION_COLOURS = {"resolved": "#159A8A", "marginal": "#E39A31", "coarse": "#D64B5F"}
WINDOW_HEADERS = ("window", "range [kHz]", "converged", "kept", "nev",
                  "iterations", "SLEPc stopped", "seconds", "status")


class SolverWindow(QtWidgets.QMainWindow):
    """Knobs on the left; spectrum, mode structure, windows and log on the right."""

    def __init__(self, state: SolverState | None = None,
                 compute: ComputeClient | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("superNOVA - Solver")
        self.resize(1320, 860)

        self.state = state or SolverState()
        self.view = None                     # EquilibriumView from the other tab
        self.preview: ContinuumPreview | None = None
        self.result: SolverResult | None = None
        self.selected: int | None = None
        self.refinement: RefineResult | None = None
        self._refining = False
        self._refine_job = None
        self._gap_points = []                # (pixel-space picking) gap estimates drawn
        self._owns_compute = compute is None
        self.compute = compute or ComputeClient(self)
        self._solving = False
        self._solve_job = None
        self._previewing = False
        self._repreview = False
        self._loading_ui = False
        self.spectrum_axes = None

        self._build_ui()
        self._write_state_to_widgets()
        self._update_derived()
        self._refresh_equilibrium_label()
        self._draw_spectrum()
        self._draw_mode()

    # ---------------------------------------------------------------- layout

    def _build_ui(self) -> None:
        controls = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(controls)
        form.setContentsMargins(10, 10, 10, 10)
        form.setSpacing(8)
        form.addWidget(self._equilibrium_box())
        form.addWidget(self._problem_box())
        form.addWidget(self._search_box())
        form.addWidget(self._grid_box())
        form.addWidget(Collapsible("Eigensolver and linear solves", self._eigensolver_box()))
        form.addWidget(self._filter_box())
        form.addWidget(self._refine_box())
        form.addWidget(self._parallel_box())
        form.addWidget(self._output_box())
        buttons = QtWidgets.QHBoxLayout()
        self.solve_button = QtWidgets.QPushButton("Solve")
        self.solve_button.setToolTip("Search every window, then run MODELOCATE.")
        self.solve_button.clicked.connect(self.solve)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setToolTip(
            "Stop now: busy workers are terminated mid-solve, the remaining "
            "windows are skipped, and what was found so far is kept.")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        buttons.addWidget(self.solve_button, 3)
        buttons.addWidget(self.stop_button, 1)
        form.addLayout(buttons)
        form.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(controls)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(380)
        scroll.setMaximumWidth(500)

        self.pages = QtWidgets.QTabWidget()
        self.pages.addTab(self._spectrum_page(), "Spectrum")
        self.pages.addTab(self._mode_page(), "Mode structure")
        self.pages.addTab(self._windows_page(), "Windows")
        self.pages.addTab(self._refine_page(), "Refinement")
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.log.setPlaceholderText(
            "A summary of each solve; tick 'verbose solver log' for the "
            "assembly and SLEPc detail.")
        self.pages.addTab(self.log, "Log")

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(self.pages)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self._build_toolbar()
        self.status = self.statusBar()

        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(PREVIEW_DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self._start_preview)

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("Actions")
        bar.setMovable(False)
        self.solve_action = self._action(bar, "Solve", self.solve, "Ctrl+Return")
        bar.addSeparator()
        self.export_action = self._action(bar, "Export modes...", self._export)
        self.export_action.setEnabled(False)
        self._action(bar, "Save figure", self._save_figure)
        bar.addSeparator()
        self._action(bar, "Save session", self._save_session)
        self._action(bar, "Load session", self._load_session)

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
        return box

    def _problem_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Problem")
        grid = QtWidgets.QGridLayout(box)
        self.ntor_spin = int_spin(-64, 64, 4, tooltip="Toroidal mode number.")
        self.mmax_spin = int_spin(1, 1024, 40, tooltip=(
            "Highest poloidal harmonic. The band itself is chosen from q: "
            "about n q_min - 3 to n q_max + 3, capped here."))
        grid.addWidget(QtWidgets.QLabel("n / mmax"), 0, 0)
        grid.addWidget(self.ntor_spin, 0, 1)
        grid.addWidget(self.mmax_spin, 0, 2)
        self.max_dm_spin = int_spin(1, 256, 5, tooltip=(
            "Largest harmonic difference |m - m'| coupled by the operators."))
        self.regularizer_edit = SciEdit(0.0, minimum=0.0, tooltip=(
            "Added to the inertia to keep the pencil regular near the axis; "
            "1e-5 in the ITPA benchmark."))
        grid.addWidget(QtWidgets.QLabel("max dm / regularizer"), 1, 0)
        grid.addWidget(self.max_dm_spin, 1, 1)
        grid.addWidget(self.regularizer_edit, 1, 2)
        self.gamma_combo = QtWidgets.QComboBox()
        self.gamma_combo.setEditable(True)
        self.gamma_combo.addItems(["5/3", "0", "1e-3"])
        self.gamma_combo.setToolTip(
            "Adiabatic index. Exactly 0 removes the sound coupling and takes "
            "the generalized linear backend; anything else is the quadratic "
            "PEP. Accepts fractions like 5/3.")
        self.species_combo = QtWidgets.QComboBox()
        self.species_combo.addItems(SPECIES)
        grid.addWidget(QtWidgets.QLabel("gamma_s / species"), 2, 0)
        grid.addWidget(self.gamma_combo, 2, 1)
        grid.addWidget(self.species_combo, 2, 2)
        self.sound_combo = QtWidgets.QComboBox()
        for name in SOUND_MODELS:
            self.sound_combo.addItem(SOUND_TITLES[name], name)
        self.sound_combo.setToolTip(
            "Ideal MHD keeps the parallel sound: the slow-sound continuum and "
            "everything that resonates with it.\n"
            "NOVA ksound=1 drops only the parallel-acoustic term (the 1/s "
            "tensor), as legacy NOVA does by default, standing in for ion Landau "
            "damping: the slow-sound continuum disappears, while gamma_s p still "
            "enters the fast-wave compressibility and the geodesic coupling.\n"
            "gamma_s = 0 is different: it removes the plasma compressibility "
            "altogether, geodesic coupling included.")
        grid.addWidget(QtWidgets.QLabel("sound"), 3, 0)
        grid.addWidget(self.sound_combo, 3, 1, 1, 2)
        for widget in (self.ntor_spin, self.mmax_spin, self.max_dm_spin,
                       self.regularizer_edit, self.species_combo, self.sound_combo):
            self._connect(widget)
        self.gamma_combo.lineEdit().editingFinished.connect(self._on_edit)
        self.gamma_combo.activated.connect(self._on_edit)
        return box

    def _search_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Frequency search")
        grid = QtWidgets.QGridLayout(box)
        self.f_low = double_spin(0.001, 1.0e6, 50.0, 3, 1.0)
        self.f_high = double_spin(0.001, 1.0e6, 150.0, 3, 1.0)
        grid.addWidget(QtWidgets.QLabel("band [kHz]"), 0, 0)
        grid.addWidget(self.f_low, 0, 1)
        grid.addWidget(self.f_high, 0, 2)
        self.window_width = double_spin(0.001, 1.0e6, 2.5, 3, 0.5, tooltip=(
            "The size of each subrange. Each is a separate shift-and-invert "
            "solve targeted at its centre; the last one may be narrower."))
        self.overlap = double_spin(0.0, 0.95, 0.0, 3, 0.05, tooltip=(
            "Fraction of a window shared with the next. A root found by two "
            "windows is kept once."))
        grid.addWidget(QtWidgets.QLabel("window [kHz] / overlap"), 1, 0)
        grid.addWidget(self.window_width, 1, 1)
        grid.addWidget(self.overlap, 1, 2)
        self.nev_spin = int_spin(1, 4096, 8, tooltip=(
            "Roots requested per window. A window that converges this many "
            "is 'saturated' -- it may hold more."))
        self.vscale = double_spin(0.001, 100.0, 0.25, 3, 0.05, tooltip=(
            "Imaginary semi-axis of each elliptical region, relative to its "
            "real semi-axis, in s = omega**2."))
        grid.addWidget(QtWidgets.QLabel("nev / region vscale"), 2, 0)
        grid.addWidget(self.nev_spin, 2, 1)
        grid.addWidget(self.vscale, 2, 2)
        self.tol_edit = SciEdit(1.0e-8, minimum=0.0, tooltip="SLEPc convergence tolerance.")
        self.maxit_spin = int_spin(1, 100000, 200, tooltip="SLEPc iteration limit per window.")
        grid.addWidget(QtWidgets.QLabel("tol / maxit"), 3, 0)
        grid.addWidget(self.tol_edit, 3, 1)
        grid.addWidget(self.maxit_spin, 3, 2)
        self.windows_label = QtWidgets.QLabel()
        self.windows_label.setWordWrap(True)
        self.windows_label.setStyleSheet(f"color: {MUTED};")
        grid.addWidget(self.windows_label, 4, 0, 1, 3)
        for widget in (self.f_low, self.f_high, self.window_width, self.overlap,
                       self.nev_spin, self.vscale, self.tol_edit, self.maxit_spin):
            self._connect(widget)
        return box

    def _grid_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Radial grid")
        grid = QtWidgets.QGridLayout(box)
        self.spans_spin = int_spin(2, 100000, 150, 10, tooltip="Equal B-spline spans in x.")
        self.order_spin = int_spin(1, 9, 3, tooltip="B-spline order of the high space.")
        self.quad_spin = int_spin(1, 32, 6, tooltip="Gauss points per span.")
        grid.addWidget(QtWidgets.QLabel("spans"), 0, 0)
        grid.addWidget(self.spans_spin, 0, 1, 1, 2)
        grid.addWidget(QtWidgets.QLabel("order / quadrature"), 1, 0)
        grid.addWidget(self.order_spin, 1, 1)
        grid.addWidget(self.quad_spin, 1, 2)
        self.rho_min = double_spin(0.0, 0.999, 0.0, 3, 0.01, tooltip=(
            "Inner wall, in rho_pol: xi^psi = 0 there. 0 is the inner edge of "
            "the Boozer domain."))
        self.rho_max = double_spin(0.001, 1.0, 1.0, 3, 0.01, tooltip=(
            "Outer wall, in rho_pol. 1 is the outer edge of the Boozer domain. "
            "The ITPA benchmark puts the walls at 0.05 and 0.95."))
        grid.addWidget(QtWidgets.QLabel("walls rho_pol"), 2, 0)
        grid.addWidget(self.rho_min, 2, 1)
        grid.addWidget(self.rho_max, 2, 2)
        note = QtWidgets.QLabel(
            "The spans are spread between the walls. MODELOCATE's spike and "
            "continuum tests compare neighbouring breakpoints of this grid, so "
            "its verdicts depend on the spans.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {MUTED};")
        grid.addWidget(note, 3, 0, 1, 3)
        self.target_combo = QtWidgets.QComboBox()
        self.target_combo.addItem("about 1e-5 in frequency", "1e-5")
        self.target_combo.addItem("about 1e-4 in frequency", "1e-4")
        self.target_combo.setToolTip(
            "What the resolution guide asks of the elements at each gap: "
            "h <= min(w/4, d/2) for about 1e-5, h <= min(w, d) for about 1e-4, "
            "with w the mode width and d the distance of the gap edge's "
            "singular points (calibrated on ITPA n=6 and DIII-D n=3).")
        grid.addWidget(QtWidgets.QLabel("resolution guide"), 4, 0)
        grid.addWidget(self.target_combo, 4, 1, 1, 2)
        self.resolution_label = QtWidgets.QLabel()
        self.resolution_label.setWordWrap(True)
        self.resolution_label.setTextFormat(QtCore.Qt.RichText)
        grid.addWidget(self.resolution_label, 5, 0, 1, 3)
        for widget in (self.spans_spin, self.order_spin, self.quad_spin,
                       self.rho_min, self.rho_max):
            self._connect(widget)
        self.target_combo.currentIndexChanged.connect(self._on_view_edit)
        return box

    def _eigensolver_box(self) -> QtWidgets.QWidget:
        body = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(body)
        form.setContentsMargins(4, 2, 4, 2)
        self.pep_combo = QtWidgets.QComboBox()
        self.pep_combo.addItems(PEP_TYPES)
        self.pep_combo.setToolTip("SLEPc PEP type. Ignored at gamma_s = 0, which is an EPS.")
        self.st_combo = QtWidgets.QComboBox()
        self.st_combo.addItems(ST_TYPES)
        self.which_combo = QtWidgets.QComboBox()
        self.which_combo.addItems(WHICH)
        self.factorization_combo = QtWidgets.QComboBox()
        self.factorization_combo.addItems(FACTORIZATIONS)
        self.factorization_combo.setToolTip(
            "The shift-and-invert factorization. MUMPS LU is the production "
            "choice; the QR ones are for when it runs out of memory or pivots.")
        self.icntl_spin = int_spin(0, 100000, 200, 20, tooltip=(
            "MUMPS ICNTL(14): workspace headroom in percent. The default 20 "
            "fails with INFOG(1) = -9 under SLEPc 3.25. 0 leaves it unset."))
        self.extra_options = QtWidgets.QPlainTextEdit()
        self.extra_options.setPlaceholderText("key value  (one PETSc option per line)")
        self.extra_options.setFixedHeight(60)
        form.addRow("PEP type", self.pep_combo)
        form.addRow("spectral transform", self.st_combo)
        form.addRow("which", self.which_combo)
        form.addRow("factorization", self.factorization_combo)
        form.addRow("MUMPS ICNTL(14)", self.icntl_spin)
        form.addRow("extra PETSc options", self.extra_options)
        for widget in (self.pep_combo, self.st_combo, self.which_combo,
                       self.factorization_combo, self.icntl_spin):
            self._connect(widget)
        self.extra_options.textChanged.connect(self._on_edit)
        return body

    def _filter_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Mode filter")
        layout = QtWidgets.QVBoxLayout(box)
        self.filter_combo = QtWidgets.QComboBox()
        for name in MODE_FILTERS:
            self.filter_combo.addItem(FILTER_TITLES[name], name)
        self.filter_combo.setToolTip(
            "MODELOCATE judges each envelope on the grid: a gap mode unless it "
            "is a one-point spike, in the core, at the edge, or above the "
            "continuum level next to a continuum crossing.\n"
            "The global path keeps the spike, core and edge tests and replaces "
            "the continuum one: every remaining mode is re-solved on the complex "
            "radial paths +alpha and -alpha. A discrete mode stays put; a "
            "continuum approximant moves. Costs two extra solves per mode.")
        layout.addWidget(self.filter_combo)
        self.filter_intro = QtWidgets.QLabel()
        self.filter_intro.setWordWrap(True)
        self.filter_intro.setStyleSheet(f"color: {MUTED};")
        layout.addWidget(self.filter_intro)
        self.locate_form = DataclassForm(ModeLocateThresholds)
        self.locate_form.changed.connect(self._on_edit)
        layout.addWidget(self.locate_form)

        self.path_box = QtWidgets.QWidget()
        path = QtWidgets.QGridLayout(self.path_box)
        path.setContentsMargins(0, 0, 0, 0)
        self.alpha_max_spin = double_spin(0.001, 0.5, 0.05, 4, 0.01, tooltip=(
            "The largest path amplitude; the path is z = x + i alpha (x - x0)(x1 - x)."))
        self.kappa_spin = double_spin(0.05, 0.95, 0.4, 2, 0.05, tooltip=(
            "Path height at a continuum extremum, as a fraction of the distance "
            "d of its singular points. Invariance breaks at 1; below 0.5 is safe "
            "on a resolved mesh."))
        path.addWidget(QtWidgets.QLabel("alpha max / kappa"), 0, 0)
        path.addWidget(self.alpha_max_spin, 0, 1)
        path.addWidget(self.kappa_spin, 0, 2)
        self.path_tolerance_edit = SciEdit(3.0e-4, minimum=0.0, tooltip=(
            "Largest relative change of omega on a path for a mode to count as "
            "invariant. It must exceed the path's own discretization error, "
            "which is larger on a coarse mesh (a resolved gap mode moves "
            "1e-5 - 1e-4; a continuum approximant 1e-3 and more)."))
        self.path_correlation_spin = double_spin(0.1, 1.0, 0.9, 3, 0.01, tooltip=(
            "Smallest harmonic correlation with the real-axis mode for a path "
            "root to be the same mode."))
        path.addWidget(QtWidgets.QLabel("tolerance / correlation"), 1, 0)
        path.addWidget(self.path_tolerance_edit, 1, 1)
        path.addWidget(self.path_correlation_spin, 1, 2)
        self.path_nev_spin = int_spin(1, 64, 4, tooltip=(
            "Roots per path solve, nearest the mode. Keep it small: the region "
            "is wide."))
        path.addWidget(QtWidgets.QLabel("roots per path solve"), 2, 0)
        path.addWidget(self.path_nev_spin, 2, 1)
        layout.addWidget(self.path_box)
        for widget in (self.filter_combo, self.alpha_max_spin, self.kappa_spin,
                       self.path_tolerance_edit, self.path_correlation_spin,
                       self.path_nev_spin):
            self._connect(widget)

        self.gap_only_check = QtWidgets.QCheckBox("show gap modes only")
        self.gap_only_check.toggled.connect(self._on_view_edit)
        layout.addWidget(self.gap_only_check)
        return box

    def _refine_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Refine selected modes")
        grid = QtWidgets.QGridLayout(box)
        intro = QtWidgets.QLabel(
            "Re-solve the modes selected in the table (Ctrl/Shift for several) "
            "on meshes graded towards them, each halving the element, and watch "
            "the frequency converge. A gap mode converges; a continuum "
            "approximant keeps moving or dissolves.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {MUTED};")
        grid.addWidget(intro, 0, 0, 1, 3)
        self.refine_h_spin = double_spin(0.0, 1.0e4, 0.0, 3, 0.1, tooltip=(
            "Element size at each mode on the finest mesh, in mm of rho_tor * a. "
            "0 picks it from the resolution guide: the guide's element for the "
            "mode's width and gap-edge distance, at most a quarter of the "
            "current element."))
        self.refine_levels_spin = int_spin(1, 8, 3, tooltip="Meshes, coarse to fine.")
        grid.addWidget(QtWidgets.QLabel("finest [mm] / meshes"), 1, 0)
        grid.addWidget(self.refine_h_spin, 1, 1)
        grid.addWidget(self.refine_levels_spin, 1, 2)
        self.refine_growth_spin = double_spin(0.01, 0.9, 0.15, 2, 0.05, tooltip=(
            "How fast the elements grow back to the base size away from a mode: "
            "neighbours differ by at most a factor 1 / (1 - growth)."))
        self.refine_path_check = QtWidgets.QCheckBox("path-test on the finest mesh")
        self.refine_path_check.setToolTip(
            "Also put the refined modes through the global-path test, where it "
            "means most: on a mesh that resolves them.")
        grid.addWidget(QtWidgets.QLabel("growth"), 2, 0)
        grid.addWidget(self.refine_growth_spin, 2, 1)
        grid.addWidget(self.refine_path_check, 3, 0, 1, 3)
        self.refine_button = QtWidgets.QPushButton("Refine selected modes")
        self.refine_button.clicked.connect(self.refine)
        self.refine_button.setEnabled(False)
        grid.addWidget(self.refine_button, 4, 0, 1, 3)
        for widget in (self.refine_h_spin, self.refine_levels_spin,
                       self.refine_growth_spin, self.refine_path_check):
            self._connect(widget)
        return box

    def _parallel_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Parallel windows")
        grid = QtWidgets.QGridLayout(box)
        cores = os.cpu_count() or 1
        self.workers_spin = int_spin(1, 256, 2, tooltip=(
            "Worker processes solving windows side by side. Each holds its own "
            "copy of the pencil and its own factorization, so memory grows "
            "with the count."))
        self.threads_spin = int_spin(1, 256, 4, tooltip=(
            "OpenMP threads for each worker's MUMPS factorization and BLAS. "
            "MUMPS stops speeding up past about 4-6 threads, so beyond that "
            "more workers beat more threads. Changing either restarts the "
            "workers, which takes a few seconds."))
        grid.addWidget(QtWidgets.QLabel("workers / threads each"), 0, 0)
        grid.addWidget(self.workers_spin, 0, 1)
        grid.addWidget(self.threads_spin, 0, 2)
        self.parallel_label = QtWidgets.QLabel()
        self.parallel_label.setWordWrap(True)
        grid.addWidget(self.parallel_label, 1, 0, 1, 3)
        self._cores = cores
        for widget in (self.workers_spin, self.threads_spin):
            self._connect(widget)
        return box

    def _output_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Output")
        grid = QtWidgets.QGridLayout(box)
        self.points_spin = int_spin(16, 100000, 512, 64, tooltip=(
            "Radial points the eigenfunctions are drawn on. MODELOCATE does "
            "not use them; it judges on the grid's breakpoints."))
        grid.addWidget(QtWidgets.QLabel("reconstruction points"), 0, 0)
        grid.addWidget(self.points_spin, 0, 1)
        self.verbose_check = QtWidgets.QCheckBox("verbose solver log")
        self.verbose_check.setToolTip(
            "Send the assembly and eigensolver detail to the Log page.")
        grid.addWidget(self.verbose_check, 1, 0, 1, 2)
        self.preview_check = QtWidgets.QCheckBox("preview the continuum before solving")
        grid.addWidget(self.preview_check, 2, 0, 1, 2)
        for widget in (self.points_spin, self.verbose_check, self.preview_check):
            self._connect(widget)
        return box

    def _spectrum_page(self) -> QtWidgets.QWidget:
        page, self.spectrum_figure, self.spectrum_canvas = figure_page(self, (8.0, 5.5))
        self.spectrum_canvas.mpl_connect("button_press_event", self._on_spectrum_click)
        self.spectrum_canvas.mpl_connect("motion_notify_event", self._on_spectrum_hover)
        self.mode_table = QtWidgets.QTableWidget()
        self.mode_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.mode_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.mode_table.itemSelectionChanged.connect(self._on_table_selection)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(page)
        splitter.addWidget(self.mode_table)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        return splitter

    def _mode_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        self.field_combo = QtWidgets.QComboBox()
        for name in FIELDS:
            self.field_combo.addItem(FIELD_TITLES.get(name, name), name)
        self.part_combo = QtWidgets.QComboBox()
        self.part_combo.addItems(PARTS)
        row.addWidget(QtWidgets.QLabel("field"))
        row.addWidget(self.field_combo)
        row.addWidget(QtWidgets.QLabel("part"))
        row.addWidget(self.part_combo)
        row.addStretch(1)
        layout.addLayout(row)
        figure, self.mode_figure, self.mode_canvas = figure_page(self, (8.0, 5.5))
        layout.addWidget(figure, 1)
        self.reason_label = QtWidgets.QLabel()
        self.reason_label.setWordWrap(True)
        self.reason_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.reason_label)
        self.field_combo.currentIndexChanged.connect(self._on_view_edit)
        self.part_combo.currentIndexChanged.connect(self._on_view_edit)
        return page

    def _windows_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.window_summary = QtWidgets.QLabel("Solve to see what each window did.")
        self.window_summary.setWordWrap(True)
        layout.addWidget(self.window_summary)
        self.window_table = QtWidgets.QTableWidget()
        self.window_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.window_table.itemSelectionChanged.connect(self._on_window_selection)
        layout.addWidget(self.window_table, 1)
        self.window_message = QtWidgets.QLabel()
        self.window_message.setWordWrap(True)
        self.window_message.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.window_message)
        legend = QtWidgets.QLabel("<br>".join(
            f"<span style='color:{STATUS_COLOURS[key]}'>&#9632;</span> "
            f"<b>{key}</b>: {hint}" for key, hint in STATUS_HINTS.items()))
        legend.setWordWrap(True)
        layout.addWidget(legend)
        return page

    def _refine_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.refine_summary = QtWidgets.QLabel(
            "Select modes in the Spectrum table, then press 'Refine selected modes'.")
        self.refine_summary.setWordWrap(True)
        self.refine_summary.setTextFormat(QtCore.Qt.RichText)
        layout.addWidget(self.refine_summary)
        figure, self.refine_figure, self.refine_canvas = figure_page(self, (8.0, 4.0))
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(figure)
        self.refine_table = QtWidgets.QTableWidget()
        self.refine_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        splitter.addWidget(self.refine_table)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)
        return page

    def _connect(self, widget) -> None:
        if isinstance(widget, QtWidgets.QComboBox):
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
        self.mmax_spin.setValue(s.mmax)
        self.max_dm_spin.setValue(s.max_dm)
        self.regularizer_edit.setValue(s.regularizer)
        self.gamma_combo.setEditText(_format_gamma(s.gamma_s))
        self.species_combo.setCurrentText(s.species)
        self.sound_combo.setCurrentIndex(max(self.sound_combo.findData(s.sound_model), 0))
        self.f_low.setValue(s.f_low_khz)
        self.f_high.setValue(s.f_high_khz)
        self.window_width.setValue(s.window_width_khz)
        self.overlap.setValue(s.window_overlap)
        self.nev_spin.setValue(s.nev)
        self.vscale.setValue(s.region_vscale)
        self.tol_edit.setValue(s.tol)
        self.maxit_spin.setValue(s.maxit)
        self.spans_spin.setValue(s.spans)
        self.order_spin.setValue(s.order)
        self.quad_spin.setValue(s.quadrature_order)
        self.rho_min.setValue(s.rho_min)
        self.rho_max.setValue(s.rho_max)
        self.pep_combo.setCurrentText(s.pep_type)
        self.st_combo.setCurrentText(s.st_type)
        self.which_combo.setCurrentText(s.which)
        self.factorization_combo.setCurrentText(s.factorization)
        self.icntl_spin.setValue(s.mumps_icntl_14)
        self.extra_options.setPlainText(s.extra_options)
        self.locate_form.set_overrides(s.modelocate)
        self.filter_combo.setCurrentIndex(max(self.filter_combo.findData(s.mode_filter), 0))
        self.alpha_max_spin.setValue(s.path_alpha_max)
        self.kappa_spin.setValue(s.path_kappa)
        self.path_tolerance_edit.setValue(s.path_tolerance)
        self.path_correlation_spin.setValue(s.path_correlation)
        self.path_nev_spin.setValue(s.path_nev)
        self.target_combo.setCurrentIndex(max(self.target_combo.findData(s.resolution_target), 0))
        self.refine_h_spin.setValue(s.refine_h_mm)
        self.refine_levels_spin.setValue(s.refine_levels)
        self.refine_growth_spin.setValue(s.refine_growth)
        self.refine_path_check.setChecked(s.refine_path_check)
        self.gap_only_check.setChecked(s.gap_only)
        self.points_spin.setValue(s.reconstruction_points)
        self.workers_spin.setValue(s.workers)
        self.threads_spin.setValue(s.threads_per_worker)
        self.verbose_check.setChecked(s.verbose)
        self.preview_check.setChecked(s.continuum_preview)
        self.field_combo.setCurrentIndex(max(self.field_combo.findData(s.field), 0))
        self.part_combo.setCurrentText(s.part)
        self._loading_ui = False

    def _read_widgets_into_state(self) -> None:
        """Copy the widgets into the state; raises ValueError on unreadable text."""
        s = self.state
        s.ntor = self.ntor_spin.value()
        s.mmax = self.mmax_spin.value()
        s.max_dm = self.max_dm_spin.value()
        s.regularizer = self.regularizer_edit.value()
        s.gamma_s = _parse_gamma(self.gamma_combo.currentText())
        s.species = self.species_combo.currentText()
        s.sound_model = self.sound_combo.currentData()
        s.f_low_khz = self.f_low.value()
        s.f_high_khz = self.f_high.value()
        s.window_width_khz = self.window_width.value()
        s.window_overlap = self.overlap.value()
        s.nev = self.nev_spin.value()
        s.region_vscale = self.vscale.value()
        s.tol = self.tol_edit.value()
        s.maxit = self.maxit_spin.value()
        s.spans = self.spans_spin.value()
        s.order = self.order_spin.value()
        s.quadrature_order = self.quad_spin.value()
        s.rho_min = self.rho_min.value()
        s.rho_max = self.rho_max.value()
        s.pep_type = self.pep_combo.currentText()
        s.st_type = self.st_combo.currentText()
        s.which = self.which_combo.currentText()
        s.factorization = self.factorization_combo.currentText()
        s.mumps_icntl_14 = self.icntl_spin.value()
        s.extra_options = self.extra_options.toPlainText()
        s.modelocate = self.locate_form.overrides()
        s.mode_filter = self.filter_combo.currentData()
        s.path_alpha_max = self.alpha_max_spin.value()
        s.path_kappa = self.kappa_spin.value()
        s.path_tolerance = self.path_tolerance_edit.value()
        s.path_correlation = self.path_correlation_spin.value()
        s.path_nev = self.path_nev_spin.value()
        s.resolution_target = self.target_combo.currentData()
        s.refine_h_mm = self.refine_h_spin.value()
        s.refine_levels = self.refine_levels_spin.value()
        s.refine_growth = self.refine_growth_spin.value()
        s.refine_path_check = self.refine_path_check.isChecked()
        s.gap_only = self.gap_only_check.isChecked()
        s.reconstruction_points = self.points_spin.value()
        s.workers = self.workers_spin.value()
        s.threads_per_worker = self.threads_spin.value()
        s.verbose = self.verbose_check.isChecked()
        s.continuum_preview = self.preview_check.isChecked()
        s.field = self.field_combo.currentData()
        s.part = self.part_combo.currentText()

    def set_state(self, state: SolverState) -> None:
        self.state = state
        self._write_state_to_widgets()
        self._on_edit()

    # ---------------------------------------------------------------- edits

    def _on_edit(self, *_) -> None:
        if self._loading_ui:
            return
        try:
            self._read_widgets_into_state()
        except ValueError as error:
            self.status.showMessage(f"Unreadable value: {error}")
            return
        self._update_derived()
        problems = self.state.validate()
        if problems:
            self.status.showMessage(problems[0])
        elif self._result_is_stale():
            self.status.showMessage("Settings changed since the last solve.")
        # Compare with what is drawn, not with the previous state: set_state
        # swaps the state in before this runs.
        if self.preview is None or self.preview.key != self.state.preview_key():
            self._schedule_preview()
        self._draw_spectrum()

    def _on_view_edit(self, *_) -> None:
        if self._loading_ui:
            return
        self.state.field = self.field_combo.currentData()
        self.state.part = self.part_combo.currentText()
        self.state.gap_only = self.gap_only_check.isChecked()
        self.state.resolution_target = self.target_combo.currentData()
        self._update_resolution_guide()
        if self.state.gap_only and self.result is not None and self.selected is not None:
            if self.result.labels[self.selected] != "gap":
                gaps = np.flatnonzero(self.result.labels == "gap")
                self.selected = int(gaps[0]) if gaps.size else None
        self._fill_mode_table()
        self._draw_spectrum()
        self._draw_mode()

    def _update_derived(self) -> None:
        s = self.state
        path = s.mode_filter == "path"
        self.path_box.setVisible(path)
        self.filter_intro.setText(
            "MODELOCATE's spike, core and edge tests run first; every mode they "
            "pass is then re-solved on the paths +alpha and -alpha, alpha kept "
            "below kappa x the singular-point distance of the continuum extrema "
            "next to it. The continuum level is not used." if path else
            "Each mode's envelope sum_m |xi_m| is judged on the grid's "
            "breakpoints: a gap mode unless it peaks on one point, inside the "
            "core edge, outside the boundary edge, or above the continuum level "
            "next to a continuum crossing. Edges are in rho_pol.")
        self._update_resolution_guide()
        self._update_refine_button()
        used = s.workers * s.threads_per_worker
        colour = WARNING if used > self._cores else MUTED
        self.parallel_label.setText(
            f"<span style='color:{colour}'>{s.workers} x {s.threads_per_worker} = "
            f"{used} of this machine's {self._cores} hardware threads"
            + ("; oversubscribed, so they will slow each other down" if used > self._cores
               else "") + "</span>")
        try:
            windows = s.windows_khz() if not s.validate() else []
        except ValueError:
            windows = []
        if not windows:
            self.windows_label.setText("")
            return
        widths = [high - low for low, high in windows]
        text = (f"{len(windows)} window{'s' if len(windows) != 1 else ''} of "
                f"{max(widths):.3g} kHz")
        if min(widths) < max(widths) - 1e-9:
            text += f" (the last {min(widths):.3g} kHz)"
        text += (f", each asking for {s.nev} roots -- {len(windows) * s.nev} "
                 "at most over the band.")
        self.windows_label.setText(text)

    def _gaps_in_band(self) -> list:
        """The preview's gaps that overlap the band, while the preview is current."""
        preview = self.preview
        if preview is None or preview.key != self.state.preview_key():
            return []
        low, high = self.state.band
        return [gap for gap in preview.gaps if gap.f_high > low and gap.f_low < high]

    def _resolution(self, gap) -> tuple[str, float, float]:
        """``(verdict, current element, guide element)`` at ``gap``, in mm."""
        current = self.preview.element_mm(gap, self.state)
        f = min(max(gap.f_center, self.state.f_low_khz), self.state.f_high_khz)
        guide = gap.element_mm(self.state.resolution_target, f)
        if not np.isfinite(guide) or current <= guide:
            return "resolved", current, guide
        return ("marginal" if current <= 2.0 * guide else "coarse"), current, guide

    def _update_resolution_guide(self) -> None:
        gaps = self._gaps_in_band()
        if self.preview is None or self.view is None:
            self.resolution_label.setText(
                f"<span style='color:{MUTED}'>The guide needs the continuum preview.</span>")
            return
        if not gaps:
            self.resolution_label.setText(
                f"<span style='color:{MUTED}'>No Alfven gap in the band.</span>")
            return
        lines = []
        for gap in gaps:
            verdict, current, guide = self._resolution(gap)
            colour = RESOLUTION_COLOURS[verdict]
            lines.append(
                f"<span style='color:{colour}'>&#9679;</span> <b>{gap.kind}</b> "
                f"m={'/'.join(map(str, gap.m))} at rho_pol {np.sqrt(gap.psi_n):.3f}: "
                f"w ~ {gap.width_mm:.3g} mm; elements {current:.3g} mm, want &le; "
                f"{guide:.3g} mm")
        self.resolution_label.setText(
            "<br>".join(lines) + f"<br><span style='color:{MUTED}'>Hover a gap on the "
            "spectrum for the details.</span>")

    def _result_is_stale(self) -> bool:
        return self.result is not None and (
            self.result.state.solve_key() != self.state.solve_key()
            or not same_equilibrium(self.result, self.view))

    # ---------------------------------------------------------- equilibrium

    def set_equilibrium(self, view) -> None:
        """Take the equilibrium the equilibrium tab passes on."""
        self.view = view
        self.preview = None
        self._update_resolution_guide()
        self._refresh_equilibrium_label()
        self._schedule_preview()
        self._draw_spectrum()

    def _refresh_equilibrium_label(self) -> None:
        view = self.view
        if view is None:
            self.equilibrium_label.setText(
                f"<span style='color:{WARNING}'>Nothing loaded. Load an "
                "equilibrium in the Equilibrium tab.</span>")
            return
        s = view.summary
        text = (f"<b>{_escape(view.title)}</b><br>q {s.get('q_min', float('nan')):.3f}"
                f" - {s.get('q_edge', float('nan')):.3f}, B<sub>axis</sub> = "
                f"{s['B_axis_T']:.3f} T, {view.psi_n.size} x {view.theta.size} Boozer grid")
        if view.density_is_placeholder:
            text += (f"<br><span style='color:{ERROR}'>No density profile: the kHz "
                     "scale would be wrong. Give the Equilibrium tab a density "
                     "file before solving.</span>")
        elif "n_axis_m3" in s:
            text += f", n<sub>axis</sub> = {s['n_axis_m3']:.3g} m<sup>-3</sup>"
        self.equilibrium_label.setText(text)

    # -------------------------------------------------------------- preview

    def _schedule_preview(self) -> None:
        if self.view is not None and self.state.continuum_preview:
            self._preview_timer.start()

    def _start_preview(self) -> None:
        if self.view is None or not self.state.continuum_preview:
            return
        if self.state.validate():
            return
        if self._previewing:
            self._repreview = True
            return
        self._previewing = True
        self.compute.submit(
            "preview", self.view.handle, self.state.to_dict(), done=self._on_preview,
            failed=self._on_preview_failed, progressed=self._progress)

    def _on_preview(self, preview: ContinuumPreview) -> None:
        self._previewing = False
        self.preview = preview
        self._update_resolution_guide()
        # a preview queued behind a solve must not hide the solve's summary
        if not self._solving and (self.result is None or self._result_is_stale()):
            gaps = len(self._gaps_in_band())
            self.status.showMessage(
                f"{preview.which} continuum for n = {preview.key[0]} in "
                f"{preview.seconds:.2f} s; {gaps} gap{'s' if gaps != 1 else ''} in the band")
        self._draw_spectrum()
        if self._repreview:
            self._repreview = False
            self._start_preview()

    def _on_preview_failed(self, message: str, _detail: str) -> None:
        self._previewing = False
        self.status.showMessage(f"Continuum preview failed: {message}")
        if self._repreview:
            self._repreview = False
            self._start_preview()

    def _progress(self, message: str) -> None:
        # a preview's progress must not replace a fresh solve's summary
        quiet = self.result is not None and not self._result_is_stale()
        if self._solving or (self._previewing and not quiet):
            self.status.showMessage(message)

    # -------------------------------------------------------------- solving

    def solve(self) -> None:
        try:
            self._read_widgets_into_state()
        except ValueError as error:
            self.status.showMessage(f"Unreadable value: {error}")
            return
        problems = self.state.validate()
        if self.view is None:
            problems.insert(0, "Load an equilibrium in the Equilibrium tab first.")
        elif self.view.density_is_placeholder:
            problems.insert(0, "The equilibrium has no density profile, so the kHz "
                               "scale would be wrong. Add one in the Equilibrium tab.")
        if self._solving:
            problems.insert(0, "A solve is already running.")
        if problems:
            self.status.showMessage(problems[0])
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Information)
            box.setWindowTitle("Cannot solve yet")
            box.setText(problems[0])
            box.exec()
            return

        self._solving = True
        self.solve_button.setEnabled(False)
        self.solve_action.setEnabled(False)
        self.stop_button.setEnabled(True)
        state, title = self.state.copy(), self.view.title
        self._append_log(
            f"\n=== solve on {title}: n = {state.ntor}, {state.f_low_khz:g}-"
            f"{state.f_high_khz:g} kHz in {state.window_count()} windows of "
            f"{state.window_width_khz:g} kHz, nev = {state.nev}, {state.spans} spans, "
            f"gamma_s = {state.gamma_s:g} ({SOUND_TITLES[state.sound_model]}), "
            f"regularizer = {state.regularizer:g}, filter {FILTER_TITLES[state.mode_filter]}, "
            f"{state.workers} worker(s) x {state.threads_per_worker} thread(s)\n")
        self.status.showMessage("Solving...")
        self._solve_job = self.compute.submit(
            "solve", self.view.handle, state.to_dict(), title, done=self._on_solved,
            failed=self._on_failed, progressed=self._progress, logged=self._append_log)

    def stop(self) -> None:
        """Stop now: busy workers are terminated; what was found is kept."""
        if self._solving:
            self.compute.cancel(self._solve_job)
            self.stop_button.setEnabled(False)
            self.status.showMessage("Stopping...")
        elif self._refining:
            self.compute.cancel(self._refine_job)
            self.stop_button.setEnabled(False)
            self.status.showMessage("Stopping the refinement...")

    def _finish_solving(self) -> None:
        self._solving = False
        self.solve_button.setEnabled(True)
        self.solve_action.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _on_solved(self, result: SolverResult) -> None:
        self._finish_solving()
        self.result = result
        self.export_action.setEnabled(result.mode_count > 0)
        labels = result.labels
        gaps = np.flatnonzero(labels == "gap")
        self.selected = (int(gaps[0]) if gaps.size
                         else (0 if result.mode_count else None))
        self._fill_mode_table()
        self._fill_window_table()
        self._draw_spectrum()
        self._draw_mode()
        statuses = {}
        for window in result.windows:
            statuses[window.status] = statuses.get(window.status, 0) + 1
        counts = ", ".join(f"{count} {label}" for label, count
                           in sorted(result.label_counts().items()))
        message = (f"{result.mode_count} modes ({gaps.size} gap) in "
                   f"{result.seconds:.1f} s  |  windows: "
                   + ", ".join(f"{n} {key}" for key, n in statuses.items()))
        self.status.showMessage(message)
        self._update_refine_button()
        self._append_log(message + (f"\n{FILTER_TITLES[result.filter_method]}: " + counts
                                    if counts else "") + "\n")

    def _on_failed(self, message: str, detail: str) -> None:
        self._finish_solving()
        self.status.showMessage(message)
        self._append_log(f"FAILED: {message}\n{detail}\n")
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("The solve failed")
        box.setText(message)
        box.setDetailedText(detail)
        box.exec()

    def _append_log(self, text: str) -> None:
        cursor = self.log.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    # -------------------------------------------------------------- tables

    def _visible_rows(self) -> list[dict]:
        if self.result is None or self._result_is_stale():
            return []
        rows = self.result.mode_rows()
        if self.state.gap_only:
            rows = [row for row in rows if row["label"] == "gap"]
        return rows

    def _fill_mode_table(self) -> None:
        rows = self._visible_rows()
        set_rows(self.mode_table, MODE_HEADERS, [
            (row["mode"], round(row["frequency_khz"], 4),
             float(f"{row['growth_ratio']:.3e}"), row["label"],
             round(row["peak_rho"], 4), round(row["center"], 4),
             round(row["width"], 4), row["dominant_m"],
             float(f"{row['backward_error']:.3e}"), row["window"])
            for row in rows])
        for index, row in enumerate(rows):
            style = LOCATE_STYLE.get(locate_family(row["label"]))
            if style:
                tint = QtGui.QColor(style["color"])
                tint.setAlpha(60)
                self.mode_table.item(index, 3).setBackground(tint)
        self.mode_table.setSortingEnabled(True)
        self._select_table_row(self.selected)

    def _fill_window_table(self) -> None:
        result = self.result
        windows = result.windows if result else ()
        set_rows(self.window_table, WINDOW_HEADERS, [
            (w.window, f"{w.range_khz[0]:.3f} - {w.range_khz[1]:.3f}", w.nconv,
             w.kept, w.nev, w.iterations, w.reason_text, round(w.seconds, 2), w.status)
            for w in windows], colours=[STATUS_COLOURS.get(w.status) for w in windows])
        self.window_message.setText("")
        if result is None:
            return
        statuses = {}
        for window in windows:
            statuses[window.status] = statuses.get(window.status, 0) + 1
        self.window_summary.setText(
            f"<b>{len(windows)} windows</b>: "
            + ", ".join(f"{n} {key}" for key, n in statuses.items())
            + ". Select a row to see its SLEPc message, if it raised one.")

    def _on_window_selection(self) -> None:
        if self.result is None:
            return
        rows = self.window_table.selectionModel().selectedRows()
        if not rows:
            return
        index = int(self.window_table.item(rows[0].row(), 0).data(QtCore.Qt.DisplayRole))
        window = self.result.windows[index]
        self.window_message.setText(
            _escape(window.message) if window.message
            else f"window {index}: {STATUS_HINTS.get(window.status, '')}")

    def _select_table_row(self, mode) -> None:
        table = self.mode_table
        blocked = table.blockSignals(True)
        table.clearSelection()
        if mode is not None:
            for row in range(table.rowCount()):
                item = table.item(row, 0)
                if item is not None and int(item.data(QtCore.Qt.DisplayRole)) == mode:
                    table.selectRow(row)
                    table.scrollToItem(item)
                    break
        table.blockSignals(blocked)

    def _on_table_selection(self) -> None:
        self._update_refine_button()
        rows = self.mode_table.selectionModel().selectedRows()
        if not rows:
            return
        current = self.mode_table.currentRow()
        row = current if any(index.row() == current for index in rows) else rows[0].row()
        item = self.mode_table.item(row, 0)
        self.select_mode(int(item.data(QtCore.Qt.DisplayRole)), from_table=True)

    def selected_modes(self) -> list[int]:
        """Every mode selected in the table, in table order."""
        modes = []
        for index in self.mode_table.selectionModel().selectedRows():
            item = self.mode_table.item(index.row(), 0)
            if item is not None:
                modes.append(int(item.data(QtCore.Qt.DisplayRole)))
        return modes

    def select_mode(self, mode: int, from_table: bool = False) -> None:
        self.selected = int(mode)
        if not from_table:
            self._select_table_row(self.selected)
        self._draw_spectrum()
        self._draw_mode()

    # -------------------------------------------------------------- drawing

    def _continuum_to_draw(self):
        """The solve's own continuum while it is current, else the preview."""
        if self.result is not None and not self._result_is_stale():
            return self.result.continuum_x, self.result.continuum_khz
        if (self.preview is not None and self.state.continuum_preview
                and self.preview.key == self.state.preview_key()):
            return self.preview.x, self.preview.frequency_khz
        return None

    def _draw_spectrum(self) -> None:
        figure = self.spectrum_figure
        figure.clear()
        ax = figure.add_subplot(111)
        self.spectrum_axes = ax
        s = self.state
        stale = self._result_is_stale()
        low, high = s.f_low_khz, s.f_high_khz
        margin = 0.06 * max(high - low, 1.0e-9)

        continuum = self._continuum_to_draw()
        if continuum is not None:
            ax.plot(continuum[0], continuum[1], color=CONTINUUM_COLOUR, lw=0.6, alpha=0.6)

        # windows: the solve's own while current, otherwise the planned ones
        current = self.result is not None and not stale
        if current:
            ranges = [w.range_khz for w in self.result.windows]
            statuses = [w.status for w in self.result.windows]
        else:
            try:
                ranges = s.windows_khz() if not s.validate() else []
            except ValueError:
                ranges = []
            statuses = []
        for index, (bottom, top) in enumerate(ranges):
            ax.axhspan(bottom, top, color="#2B5EA6" if index % 2 else "#0E7466",
                       alpha=0.09, lw=0)
            ax.axhline(bottom, color="0.6", lw=0.3)
        strip = ax.get_yaxis_transform()
        for (bottom, top), status in zip(ranges, statuses):
            ax.add_patch(Rectangle((1.0, bottom), 0.018, top - bottom, transform=strip,
                                   color=STATUS_COLOURS.get(status, "#78879D"),
                                   clip_on=False, lw=0))
        for edge in (low, high):
            ax.axhline(edge, color="0.3", lw=0.8, ls="--")
        if current:
            for wall in self.result.walls:
                if 0.0 < wall < 1.0:
                    ax.axvline(wall, color="0.3", lw=0.8, ls=":")

        self._gap_points = []
        for gap in self._gaps_in_band():
            verdict, _, _ = self._resolution(gap)
            ax.add_patch(Rectangle((gap.psi_n - 0.006, gap.f_low), 0.012, gap.f_high - gap.f_low,
                                   color=GAP_COLOUR, alpha=0.12, lw=0, zorder=2))
            ax.plot(gap.psi_n, gap.f_center, marker="h", markersize=11, linestyle="none",
                    markerfacecolor="white", markeredgecolor=RESOLUTION_COLOURS[verdict],
                    markeredgewidth=1.8, zorder=4)
            ax.annotate(gap.kind, (gap.psi_n, gap.f_center), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=7,
                        color=GAP_COLOUR, zorder=4)
            self._gap_points.append(gap)

        rows = self._visible_rows()
        if rows:
            families = [locate_family(row["label"]) for row in rows]
            for family in LOCATE_ORDER:
                chosen = [row for row, value in zip(rows, families) if value == family]
                if not chosen:
                    continue
                style = LOCATE_STYLE[family]
                gap = family == "gap"
                ax.errorbar([row["center"] for row in chosen],
                            [row["frequency_khz"] for row in chosen],
                            xerr=[row["width"] for row in chosen],
                            fmt=style["marker"], color=style["color"],
                            markersize=7 if gap else 5, elinewidth=0.8, capsize=0,
                            alpha=1.0 if gap else 0.6, zorder=6 if gap else 5,
                            label=f"{style['title']} ({len(chosen)})")
            selected = [row for row in rows if row["mode"] == self.selected]
            if selected:
                row = selected[0]
                ax.plot(row["center"], row["frequency_khz"], marker="o", markersize=15,
                        markerfacecolor="none", markeredgecolor=SELECTED_COLOUR,
                        markeredgewidth=1.6, zorder=7)
            ax.legend(loc="upper left", fontsize=7, frameon=False)
        elif continuum is not None:
            handles = [Line2D([], [], color=CONTINUUM_COLOUR, lw=0.8, label="continuum")]
            if self._gap_points:
                handles.append(Line2D([], [], marker="h", linestyle="none", markersize=9,
                                      markerfacecolor="white", markeredgecolor=GAP_COLOUR,
                                      label="gap centre: hover for the mode size"))
            ax.legend(handles=handles, loc="upper left", fontsize=7, frameon=False)
        if self.result is not None and stale:
            ax.text(0.01, 0.99, "settings changed since the last solve; Solve to update",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8,
                    color=WARNING)
        if self.view is None:
            ax.text(0.5, 0.5, "Load an equilibrium in the Equilibrium tab",
                    transform=ax.transAxes, ha="center", va="center", color="0.45")

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(max(0.0, low - margin), high + margin)
        ax.set_xlabel(r"$\psi_N$ (mode centre, bar = radial width; dotted: walls)")
        ax.set_ylabel("f [kHz]")
        ax.set_title(f"n = {s.ntor}: {len(ranges)} windows of {s.window_width_khz:g} kHz, "
                     f"nev = {s.nev}, {s.spans} spans", fontsize=10)
        ax.grid(alpha=0.2)
        self.spectrum_canvas.draw_idle()

    def _draw_mode(self) -> None:
        figure = self.mode_figure
        figure.clear()
        ax = figure.add_subplot(111)
        result = self.result
        if (result is None or not result.mode_count or self.selected is None
                or self._result_is_stale()):
            ax.text(0.5, 0.5, "Solve, then pick a mode on the spectrum or in the table",
                    ha="center", va="center", transform=ax.transAxes, color="0.45")
            ax.set_axis_off()
            self.reason_label.setText("")
            self.mode_canvas.draw_idle()
            return
        mode = self.selected
        row = result.mode_rows()[mode]
        values = result.field(mode, self.state.field, self.state.part)
        colours = _harmonic_colours(result.m_pol.size)
        for index, m in enumerate(result.m_pol):
            ax.plot(result.x, values[:, index], color=colours[index], lw=1.1,
                    label=f"m = {m}")
        # what MODELOCATE judged: the excluded core and edge, the crossings,
        # and the envelope on the breakpoints
        ax.axvspan(0.0, result.psi_core, color="0.85", alpha=0.6, lw=0)
        ax.axvspan(result.psi_edge, 1.0, color="0.85", alpha=0.6, lw=0)
        for wall in result.walls:
            if 0.0 < wall < 1.0:
                ax.axvline(wall, color="0.3", lw=0.8, ls=":")
        crossings = result.crossings[mode]
        if crossings.size:
            ax.plot(crossings, np.zeros(crossings.size), transform=ax.get_xaxis_transform(),
                    linestyle="none", marker="|", markersize=14, color=CONTINUUM_COLOUR)
        ax.axhline(0.0, color="0.3", lw=0.5)
        envelope_axes = ax.twinx()
        points, envelope = result.envelope(mode)
        envelope_axes.plot(points, envelope, color="k", lw=0.9, ls="--",
                           marker=".", markersize=3, label="MODELOCATE envelope")
        thresholds = result.state.locate_thresholds()
        envelope_axes.axhline(thresholds.continuum_level, color=CONTINUUM_COLOUR,
                              lw=0.6, ls=":")
        envelope_axes.set_ylim(0.0, 1.05)
        envelope_axes.set_ylabel("envelope / peak", fontsize=8)
        envelope_axes.tick_params(labelsize=8)

        family = locate_family(row["label"])
        style = LOCATE_STYLE.get(family, {"title": row["label"]})
        ax.set_title(f"mode {mode}: {row['frequency_khz']:.4f} kHz, "
                     f"Im/Re omega = {row['growth_ratio']:.2e} -- {row['label']}",
                     fontsize=10)
        ax.set_xlabel(r"$\psi_N$ (shaded: MODELOCATE core and edge; ticks: continuum "
                      "crossings; dotted: walls)")
        ax.set_ylabel(f"{self.state.part}  "
                      f"{FIELD_TITLES.get(self.state.field, self.state.field)}")
        ax.set_xlim(0.0, 1.0)
        ax.grid(alpha=0.2)
        if result.m_pol.size <= 30:
            handles, labels = ax.get_legend_handles_labels()
            extra, extra_labels = envelope_axes.get_legend_handles_labels()
            ax.legend(handles + extra, labels + extra_labels, fontsize=7,
                      ncol=2 if result.m_pol.size > 10 else 1, frameon=False,
                      loc="upper right")
        why = result.why(row["label"])
        if np.isfinite(row["path_alpha"]):
            why += (f" (alpha = {row['path_alpha']:.3g}; worst change "
                    f"{row['path_drift']:.2g}, worst correlation {row['path_correlation']:.3f})")
        self.reason_label.setText(
            f"<b>{_escape(style['title'])}</b>: {_escape(why)}.<br>"
            f"<span style='color:{MUTED}'>envelope peak at rho_pol = "
            f"{row['peak_rho']:.4f}; dominant m = {row['dominant_m']}; width "
            f"{row['width']:.4f}; backward error {row['backward_error']:.2e}; "
            f"{crossings.size} continuum crossing(s); found in window "
            f"{row['window']}.</span>")
        self.mode_canvas.draw_idle()

    def _on_spectrum_click(self, event) -> None:
        rows = self._visible_rows()
        if (not rows or event.inaxes is not self.spectrum_axes
                or toolbar_is_busy(self.spectrum_canvas)):
            return
        points = np.array([[row["center"], row["frequency_khz"]] for row in rows])
        pixels = self.spectrum_axes.transData.transform(points)
        distance = np.hypot(pixels[:, 0] - event.x, pixels[:, 1] - event.y)
        nearest = int(np.argmin(distance))
        if distance[nearest] <= PICK_PIXELS:
            self.select_mode(rows[nearest]["mode"])

    def _on_spectrum_hover(self, event) -> None:
        if not self._gap_points or event.inaxes is not self.spectrum_axes:
            QtWidgets.QToolTip.hideText()
            return
        points = np.array([[gap.psi_n, gap.f_center] for gap in self._gap_points])
        pixels = self.spectrum_axes.transData.transform(points)
        distance = np.hypot(pixels[:, 0] - event.x, pixels[:, 1] - event.y)
        nearest = int(np.argmin(distance))
        if distance[nearest] > PICK_PIXELS:
            QtWidgets.QToolTip.hideText()
            return
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(),
                                    self._gap_tooltip(self._gap_points[nearest]),
                                    self.spectrum_canvas)

    def _gap_tooltip(self, gap) -> str:
        verdict, current, guide = self._resolution(gap)
        rough = gap.element_mm("1e-4")
        fine = gap.element_mm("1e-5")
        margin = 0.01 * (gap.f_high - gap.f_low)
        near_edge = gap.edge_distance_mm(gap.f_high - margin)
        colour = RESOLUTION_COLOURS[verdict]
        return (
            f"<b>{gap.kind} gap</b>, m = {'/'.join(map(str, gap.m))}, at rho_pol "
            f"{np.sqrt(gap.psi_n):.3f} (r = rho_tor a = {gap.r_mm:.0f} mm)<br>"
            f"{gap.f_low:.3f} - {gap.f_high:.3f} kHz<br>"
            f"q = {gap.q:.3f}, s = {gap.shear:.3g}, eps = {gap.epsilon:.3g}, "
            f"m = n q = {gap.m_eff:.1f}<br><br>"
            f"<b>mode width</b> w ~ {gap.width_mm:.3g} mm ({gap.regime})<br>"
            f"&nbsp;&nbsp;(r0/m) eps/|s| = {gap.width_shear_mm:.3g} mm, "
            f"(r0/m) sqrt(eps/|s|) = {gap.width_low_shear_mm:.3g} mm<br>"
            f"<b>gap-edge distance</b> d = {gap.edge_distance_mm(gap.f_center):.3g} mm "
            f"at the centre, {near_edge:.3g} mm 1% of the gap from an edge, 0 at it<br><br>"
            f"elements: &le; {rough:.3g} mm for ~1e-4, &le; {fine:.3g} mm for ~1e-5<br>"
            f"<span style='color:{colour}'><b>this grid: {current:.3g} mm -- "
            f"{verdict}</b></span> against {guide:.3g} mm<br>"
            f"<span style='color:{MUTED}'>A mode nearer an edge has finer structure: "
            "refine it, or add spans.</span>")

    # ------------------------------------------------------------- refining

    def _update_refine_button(self) -> None:
        ready = (self.result is not None and not self._result_is_stale()
                 and not self._refining and not self._solving
                 and bool(self.selected_modes()))
        self.refine_button.setEnabled(ready)

    def refine(self) -> None:
        """Re-solve the selected modes on meshes graded towards them."""
        try:
            self._read_widgets_into_state()
        except ValueError as error:
            self.status.showMessage(f"Unreadable value: {error}")
            return
        modes = self.selected_modes()
        result = self.result
        problems = self.state.validate()
        if result is None or self._result_is_stale():
            problems.insert(0, "Solve first: refinement starts from the current solve.")
        elif not modes:
            problems.insert(0, "Select one or more modes in the table first.")
        if self._refining or self._solving:
            problems.insert(0, "A solve or a refinement is already running.")
        if problems:
            self.status.showMessage(problems[0])
            return
        peak = np.asarray(result.located["peak_rho"].values, dtype=float) ** 2
        request = {
            "modes": modes,
            "omega": [complex(result.omega[mode]) for mode in modes],
            "frequency_khz": [float(result.frequency_khz[mode]) for mode in modes],
            "peak_psi": [float(peak[mode]) for mode in modes],
            "fields": np.asarray(result.fields["xi_psi"])[modes],
        }
        # what the result was solved with, plus this tab's refinement settings
        state = result.state.copy()
        for name in ("refine_h_mm", "refine_levels", "refine_growth", "refine_path_check",
                     "resolution_target", "path_alpha_max", "path_kappa",
                     "path_tolerance", "path_correlation", "path_nev", "workers",
                     "threads_per_worker"):
            setattr(state, name, getattr(self.state, name))
        self._refining = True
        self._update_refine_button()
        self.solve_button.setEnabled(False)
        self.solve_action.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._append_log(f"\n=== refine modes {', '.join(map(str, modes))}: "
                         f"{state.refine_levels} meshes, finest "
                         + (f"{state.refine_h_mm:g} mm" if state.refine_h_mm > 0
                            else "from the resolution guide") + "\n")
        self.status.showMessage("Refining...")
        self._refine_job = self.compute.submit(
            "refine", self.view.handle, state.to_dict(), request, self.view.title,
            done=self._on_refined, failed=self._on_refine_failed,
            progressed=self._refine_progress)

    def _refine_progress(self, message: str) -> None:
        if self._refining:
            self.status.showMessage(message)

    def _finish_refining(self) -> None:
        self._refining = False
        self.solve_button.setEnabled(True)
        self.solve_action.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._update_refine_button()

    def _on_refined(self, refinement: RefineResult) -> None:
        self._finish_refining()
        self.refinement = refinement
        self._fill_refine_table()
        self._draw_refinement()
        self.pages.setCurrentIndex(self.pages.count() - 2)     # the Refinement page
        message = f"Refined {len(refinement.modes)} mode(s) in {refinement.seconds:.1f} s"
        self.status.showMessage(message)
        lines = [message]
        for mode in refinement.modes:
            rows = [row for row in refinement.rows_for(mode)
                    if np.isfinite(row["frequency_khz"])]
            if rows:
                lines.append(f"  mode {mode}: " + " -> ".join(
                    f"{row['frequency_khz']:.4f}" for row in rows) + " kHz"
                    + (f"; path test: {refinement.path[mode]['label']}"
                       if mode in refinement.path else ""))
        self._append_log("\n".join(lines) + "\n")

    def _on_refine_failed(self, message: str, detail: str) -> None:
        self._finish_refining()
        self.status.showMessage(f"Refinement failed: {message}")
        self._append_log(f"REFINE FAILED: {message}\n{detail}\n")

    def _fill_refine_table(self) -> None:
        refinement = self.refinement
        rows = []
        for mode in (refinement.modes if refinement else ()):
            previous = np.nan
            for row in refinement.rows_for(mode):
                change = row["frequency_khz"] - previous
                rows.append((mode, row["level"] + 1, row["elements"], round(row["h_mm"], 3),
                             round(row["frequency_khz"], 5),
                             float(f"{change:.2e}") if np.isfinite(change) else "",
                             float(f"{row['growth_ratio']:.2e}"),
                             round(row["correlation"], 4), round(row["fwhm_mm"], 1),
                             float(f"{row['backward_error']:.2e}")))
                previous = row["frequency_khz"]
        set_rows(self.refine_table, REFINE_HEADERS, rows)
        if refinement is None:
            return
        verdicts = []
        for mode in refinement.modes:
            verdict = refinement.path.get(mode)
            if verdict:
                colour = LOCATE_STYLE.get(verdict["label"], {}).get("color", MUTED)
                verdicts.append(
                    f"mode {mode}: <span style='color:{colour}'><b>{verdict['label']}</b>"
                    f"</span> on the finest mesh (alpha = {verdict['alpha']:.3g}, change "
                    f"{verdict['drift']:.2g}, correlation {verdict['correlation']:.3f})")
        self.refine_summary.setText(
            f"<b>{len(refinement.modes)} mode(s)</b>, {refinement.state.refine_levels} "
            "meshes graded towards them, each halving the element at the modes. A gap "
            "mode's frequency settles; a continuum approximant keeps moving."
            + ("<br>" + "<br>".join(verdicts) if verdicts else ""))

    def _draw_refinement(self) -> None:
        figure = self.refine_figure
        figure.clear()
        refinement = self.refinement
        if refinement is None or not refinement.rows:
            self.refine_canvas.draw_idle()
            return
        converge, shapes = figure.subplots(1, 2)
        colours = _harmonic_colours(max(len(refinement.modes), 2))
        for colour, mode in zip(colours, refinement.modes):
            rows = [row for row in refinement.rows_for(mode) if np.isfinite(row["frequency_khz"])]
            if len(rows) < 2:
                continue
            h = np.array([row["h_mm"] for row in rows])
            f = np.array([row["frequency_khz"] for row in rows])
            change = np.maximum(np.abs(f[:-1] - f[-1]), 1.0e-7)
            converge.loglog(h[:-1], change, marker="o", color=colour,
                            label=f"mode {mode}: {f[-1]:.4f} kHz")
        converge.invert_xaxis()
        plain = FuncFormatter(lambda value, _: f"{value:g}")
        converge.xaxis.set_major_formatter(plain)
        converge.xaxis.set_minor_formatter(plain)
        converge.set_xlabel("element at the mode [mm]")
        converge.set_ylabel("|f - f(finest)| [kHz]")
        converge.set_title("frequency against mesh", fontsize=9)
        converge.grid(alpha=0.2, which="both")
        if converge.lines:
            converge.legend(fontsize=7, frameon=False)
        mode = (self.selected if self.selected in refinement.envelopes
                else refinement.modes[0])
        envelopes = refinement.envelopes.get(mode, np.empty((0, 0)))
        for level, envelope in enumerate(envelopes):
            shapes.plot(refinement.x, envelope, lw=1.0 + 0.4 * level,
                        label=f"mesh {level + 1}")
        shapes.set_xlabel(r"$\psi_N$")
        shapes.set_ylabel(r"$|\xi^\psi|$ envelope")
        shapes.set_title(f"mode {mode} on each mesh", fontsize=9)
        shapes.grid(alpha=0.2)
        if envelopes.size:
            peak = refinement.x[np.argmax(envelopes[-1])]
            shapes.set_xlim(max(0.0, peak - 0.15), min(1.0, peak + 0.15))
            shapes.legend(fontsize=7, frameon=False)
        figure.tight_layout()
        self.refine_canvas.draw_idle()

    # -------------------------------------------------------------- actions

    def _export(self) -> None:
        if self.result is None or not self.result.mode_count:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export modes", "modes.nc", "NetCDF (*.nc);;All files (*)")
        if not path:
            return
        try:
            SolverModel.export(self.result, path)
        except Exception as error:
            self.status.showMessage(f"Could not export: {error}")
            return
        self.status.showMessage(f"Wrote {path}")

    def _save_figure(self) -> None:
        figure, name = ((self.mode_figure, "mode") if self.pages.currentIndex() == 1
                        else (self.spectrum_figure, "spectrum"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save figure", f"solver_{name}.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            figure.savefig(path, dpi=180, bbox_inches="tight")
            self.status.showMessage(f"Wrote {path}")

    def _save_session(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save solver session", "solver_session.json", "JSON (*.json)")
        if not path:
            return
        try:
            self._read_widgets_into_state()
        except ValueError as error:
            self.status.showMessage(f"Unreadable value: {error}")
            return
        self.state.to_json(path)
        self.status.showMessage(f"Wrote {path}")

    def _load_session(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load solver session", str(Path.cwd()), "JSON (*.json)")
        if not path:
            return
        try:
            state = SolverState.from_json(path)
        except (ValueError, OSError, TypeError) as error:
            self.status.showMessage(f"Could not load session: {error}")
            return
        self.set_state(state)

    def shutdown(self) -> None:
        if self._solving:
            self.compute.cancel(self._solve_job)
        if self._refining:
            self.compute.cancel(self._refine_job)
        if self._owns_compute:
            self.compute.shutdown()

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)


def _parse_gamma(text: str) -> float:
    text = text.strip()
    if "/" in text:
        numerator, _, denominator = text.partition("/")
        return float(numerator) / float(denominator)
    return float(text)


def _format_gamma(value: float) -> str:
    return "5/3" if np.isclose(value, 5.0 / 3.0, rtol=0.0, atol=1e-12) else f"{value:g}"


def _harmonic_colours(count: int):
    from matplotlib import colormaps

    cmap = colormaps["viridis"]
    return [cmap(value) for value in np.linspace(0.0, 0.9, max(count, 1))]


def _escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def main(argv=None) -> int:
    """Open the solver tab on its own; it has nothing to solve until given a view."""
    argv = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv[:1])
    state = SolverState.from_json(argv[1]) if len(argv) > 1 else None
    window = SolverWindow(state)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
