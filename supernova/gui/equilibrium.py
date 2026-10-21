"""The equilibrium tab: load an equilibrium, look inside it, re-solve it.

Loading computes the magnetic coordinates with the code itself -- pyCOCOS's
Boozer transform for an eqdsk, the analytic transform for ITPA -- and the tab
draws what the solver will see: the Boozer grid over the flux surfaces, the
1-D profiles, and any metric element along one flux surface in the
solver-facing angle. Loading is slow (tens of seconds at production
resolution), so it runs in the compute process and only when asked.

TokaMaker re-solves an eqdsk as a fixed-boundary equilibrium whose q matches
the geometry-derived Boozer q. It runs as a separate process, streams its
output into the log page, and the refined file is loaded when it finishes,
with the previous profiles kept on the plots for comparison.

Every load is passed on (``equilibriumChanged``) for the other tabs to take
over. A load that follows a re-solve is passed on only if "pass it on" was
ticked when the re-solve finished; otherwise the other tabs keep what they
had, and the refined equilibrium stays here to be looked at.

The window owns no physics. It edits an :class:`~supernova.gui._equilibrium.
EquilibriumState` and draws the :class:`~supernova.gui._equilibrium.
EquilibriumView` that comes back.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from matplotlib.lines import Line2D
from PySide6 import QtCore, QtGui, QtWidgets

from supernova.gui._equilibrium import (
    PROFILE_AXES,
    SOURCE_LABELS,
    SOURCES,
    SPECIES,
    SURFACE_GROUPS,
    SURFACE_LABELS,
    SURFACE_NAMES,
    EquilibriumState,
    same_equilibrium,
    EquilibriumView,
)
from supernova.gui._qt import (
    ERROR,
    MUTED,
    WARNING,
    ComputeClient,
    SciEdit,
    ask_session_to_open,
    ask_session_to_save,
    double_spin,
    figure_page,
    int_spin,
    toolbar_is_busy,
)
from supernova.gui._refine_job import RESULT_MARKER
from supernova.gui._session import read_session, write_session
from supernova.gui._state import COORDINATES

SURFACE_COLOUR = "#2B5EA6"
SPOKE_COLOUR = "#9C5D14"
SELECTED_COLOUR = "#A33636"
BEFORE_COLOUR = "#78879D"
REDRAW_MS = 60
PAGES = ("Shape", "Profiles", "Surface", "TokaMaker log")


def tokamaker_available() -> bool:
    return importlib.util.find_spec("OpenFUSIONToolkit") is not None


class EquilibriumWindow(QtWidgets.QMainWindow):
    """Controls on the left; shape, profiles, one surface and a log on the right."""

    # (view, share): share is False for a re-solve the user kept to this tab
    equilibriumChanged = QtCore.Signal(object, bool)

    def __init__(self, state: EquilibriumState | None = None,
                 compute: ComputeClient | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("superNOVA - Equilibrium")
        self.resize(1280, 820)

        self.state = state or EquilibriumState()
        self._requested_key = None           # the inputs of the load in flight
        self._loaded_key = None              # ...and of the one on show
        self.view: EquilibriumView | None = None
        self.before: EquilibriumView | None = None   # what a refinement replaced
        self._owns_compute = compute is None
        self.compute = compute or ComputeClient(self)
        self._loading = False
        self._reload = False                 # inputs changed during a load
        self._comparison_pending = False     # the next load is a refinement's
        self._share_refined = True           # ...and whether to pass it on
        self.shared = True                   # whether the view on show was passed on
        self.session_owner = None            # in the app: the window whose session the buttons save
        self._loading_ui = False
        self._process: QtCore.QProcess | None = None
        self._process_output = ""
        self._spec_path: str | None = None
        self._profile_axes = []
        self.shape_axes = None

        self._build_ui()
        self._write_state_to_widgets()
        self._sync_source_page()
        self._draw_all()
        if not self.state.validate():
            self.load()

    # ---------------------------------------------------------------- layout

    def _build_ui(self) -> None:
        controls = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(controls)
        form.setContentsMargins(10, 10, 10, 10)
        form.setSpacing(10)
        form.addWidget(self._source_box())
        form.addWidget(self._surface_box())
        form.addWidget(self._display_box())
        form.addWidget(self._refine_box())
        form.addWidget(self._summary_box())
        form.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(controls)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(470)

        self.pages = QtWidgets.QTabWidget()
        page, self.shape_figure, self.shape_canvas = figure_page(self, (6.0, 7.0), self._draw_shape)
        self.pages.addTab(page, "Shape")
        page, self.profile_figure, self.profile_canvas = figure_page(self, (11.0, 6.0), self._draw_profiles)
        self.pages.addTab(page, "Profiles")
        page, self.surface_figure, self.surface_canvas = figure_page(self, (8.0, 7.0), self._draw_surface)
        self.pages.addTab(page, "Surface")
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.log.setPlaceholderText("TokaMaker output appears here while it runs.")
        self.pages.addTab(self.log, "TokaMaker log")

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(self.pages)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self._build_toolbar()
        self.status = self.statusBar()
        self.status.showMessage("Choose an eqdsk or the ITPA case, then Load.")

        self.shape_canvas.mpl_connect("button_press_event", self._on_shape_click)
        self.profile_canvas.mpl_connect("button_press_event", self._on_profile_click)

        self._redraw_timer = QtCore.QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(REDRAW_MS)
        self._redraw_timer.timeout.connect(self._draw_all)

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("Actions")
        bar.setMovable(False)
        self.load_action = self._action(bar, "Load equilibrium", self.load, "Ctrl+L")
        bar.addSeparator()
        self.export_action = self._action(bar, "Export data...", self._export_dataset)
        self.figure_action = self._action(bar, "Save figure", self._save_figure)
        self.export_action.setEnabled(False)
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

    def _source_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Equilibrium")
        layout = QtWidgets.QVBoxLayout(box)

        self.source_combo = QtWidgets.QComboBox()
        for key in SOURCES:
            self.source_combo.addItem(SOURCE_LABELS[key], key)
        layout.addWidget(self.source_combo)

        self.source_stack = QtWidgets.QStackedWidget()
        self.source_stack.addWidget(self._eqdsk_page())
        self.source_stack.addWidget(self._itpa_page())
        layout.addWidget(self.source_stack)

        species_row = QtWidgets.QHBoxLayout()
        species_row.addWidget(QtWidgets.QLabel("ion species"))
        self.species_combo = QtWidgets.QComboBox()
        self.species_combo.addItems(SPECIES)
        self.species_combo.setToolTip(
            "The ions. With the density they set the mass density, and so the "
            "Alfven speed, so the continuum and solver tabs use this one -- they "
            "cannot differ. Changing it does not reload: the geometry does not "
            "depend on it.")
        species_row.addWidget(self.species_combo, 1)
        layout.addLayout(species_row)
        self.species_combo.currentIndexChanged.connect(self._on_species_edit)

        self.load_button = QtWidgets.QPushButton("Load equilibrium")
        self.load_button.setToolTip(
            "Read the input and compute the magnetic coordinates. Tens of "
            "seconds for an eqdsk at production resolution.")
        self.load_button.clicked.connect(self.load)
        layout.addWidget(self.load_button)

        self.source_combo.currentIndexChanged.connect(self._sync_source_page)
        self._connect(self.source_combo, self._on_input_edit)
        return box

    def _eqdsk_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)

        self.eqdsk_edit = QtWidgets.QLineEdit()
        self.eqdsk_edit.setPlaceholderText("path to an .eqdsk / .geqdsk")
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._browse_eqdsk)
        grid.addWidget(self.eqdsk_edit, 0, 0, 1, 3)
        grid.addWidget(browse, 0, 3)

        self.cocos_spin = int_spin(1, 18, 5, tooltip=(
            "COCOS of the file as written. A refined equilibrium records its "
            "own in the sidecar, and a TokaMaker re-solve here sets it for you."))
        grid.addWidget(QtWidgets.QLabel("COCOS"), 1, 0)
        grid.addWidget(self.cocos_spin, 1, 1)

        self.density_edit = QtWidgets.QLineEdit()
        self.density_edit.setPlaceholderText("density profile (optional)")
        self.density_edit.setToolTip(
            "Two-column density profile. Geometry does not need it; the solver "
            "tab does, because the density sets the Alfven frequency. Without "
            f"one a uniform placeholder is used and flagged.")
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._browse_density)
        grid.addWidget(self.density_edit, 2, 0, 1, 3)
        grid.addWidget(browse, 2, 3)

        self.density_x = QtWidgets.QComboBox()
        self.density_x.addItems(COORDINATES)
        self.density_x.setToolTip("What the file's first column is.")
        self.density_interp = QtWidgets.QComboBox()
        self.density_interp.addItems(COORDINATES)
        self.density_interp.setToolTip("Which variable the interpolation runs in.")
        grid.addWidget(QtWidgets.QLabel("column / interp."), 3, 0)
        grid.addWidget(self.density_x, 3, 1)
        grid.addWidget(self.density_interp, 3, 2, 1, 2)
        self.density_units = QtWidgets.QLineEdit()
        grid.addWidget(QtWidgets.QLabel("units"), 4, 0)
        grid.addWidget(self.density_units, 4, 1, 1, 3)

        self.boozer_min = double_spin(0.0, 0.999, 0.005, 3, 0.001, tooltip=(
            "Inner edge of the Boozer domain, in rho_pol."))
        self.boozer_max = double_spin(0.001, 1.0, 0.995, 3, 0.001, tooltip=(
            "Outer edge of the Boozer domain, in rho_pol. The separatrix itself "
            "has no straight-field-line angle."))
        grid.addWidget(QtWidgets.QLabel("Boozer rho_pol"), 5, 0)
        grid.addWidget(self.boozer_min, 5, 1)
        grid.addWidget(self.boozer_max, 5, 2, 1, 2)

        self.boozer_npsi = int_spin(16, 4096, 511, 32, tooltip=(
            "Surfaces in the Boozer grid. 511 x 512 is the production default; "
            "128 x 128 loads several times faster for a first look."))
        self.boozer_ntheta = int_spin(16, 4096, 512, 32, tooltip="Poloidal points.")
        grid.addWidget(QtWidgets.QLabel("surfaces x angles"), 6, 0)
        grid.addWidget(self.boozer_npsi, 6, 1)
        grid.addWidget(self.boozer_ntheta, 6, 2, 1, 2)

        self.density_uniform = SciEdit(0.0, minimum=0.0, tooltip=(
            "A constant density in m^-3, used when no file is given -- the "
            "ITPA benchmark's 2e19, say. 0 leaves the flagged placeholder."))
        grid.addWidget(QtWidgets.QLabel("or uniform [m^-3]"), 8, 0)
        grid.addWidget(self.density_uniform, 8, 1, 1, 3)

        self.m_max_spin = int_spin(1, 1024, 128, tooltip=(
            "Poloidal harmonics kept when equilibrium quantities are Fourier "
            "decomposed for the operators. Must cover the span of the solver's "
            "harmonic band."))
        grid.addWidget(QtWidgets.QLabel("equilibrium m_max"), 7, 0)
        grid.addWidget(self.m_max_spin, 7, 1)

        for widget in (self.eqdsk_edit, self.cocos_spin, self.density_edit,
                       self.density_uniform,
                       self.density_x, self.density_interp, self.density_units,
                       self.boozer_min, self.boozer_max, self.boozer_npsi,
                       self.boozer_ntheta, self.m_max_spin):
            self._connect(widget, self._on_input_edit)
        return page

    def _itpa_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)
        self.itpa_aminor = double_spin(0.01, 100.0, 1.0, 3, 0.1)
        self.itpa_rmajor = double_spin(0.02, 1000.0, 10.0, 3, 0.1)
        self.itpa_b0 = double_spin(0.01, 100.0, 3.0, 3, 0.1)
        self.itpa_q0 = double_spin(0.01, 100.0, 1.71, 3, 0.01)
        self.itpa_qedge = double_spin(0.01, 100.0, 1.87, 3, 0.01)
        self.itpa_bt_sign = QtWidgets.QComboBox()
        self.itpa_bt_sign.addItem("+1", 1)
        self.itpa_bt_sign.addItem("-1", -1)
        self.itpa_ns = int_spin(4, 4096, 256, 32)
        self.itpa_ntheta = int_spin(8, 4096, 256, 32)
        self.itpa_m_max = int_spin(1, 1024, 20)
        form.addRow("a [m]", self.itpa_aminor)
        form.addRow("R0 [m]", self.itpa_rmajor)
        form.addRow("B0 [T]", self.itpa_b0)
        form.addRow("q0", self.itpa_q0)
        form.addRow("q edge", self.itpa_qedge)
        form.addRow("sign of B_phi", self.itpa_bt_sign)
        form.addRow("surfaces", self.itpa_ns)
        form.addRow("angles", self.itpa_ntheta)
        form.addRow("equilibrium m_max", self.itpa_m_max)
        note = QtWidgets.QLabel(
            "Circular, large aspect ratio, analytic coordinates and its own "
            "density -- the published TAE benchmark.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {MUTED};")
        form.addRow(note)
        for widget in (self.itpa_aminor, self.itpa_rmajor, self.itpa_b0,
                       self.itpa_q0, self.itpa_qedge, self.itpa_bt_sign,
                       self.itpa_ns, self.itpa_ntheta, self.itpa_m_max):
            self._connect(widget, self._on_input_edit)
        return page

    def _surface_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Flux surface")
        layout = QtWidgets.QVBoxLayout(box)

        row = QtWidgets.QHBoxLayout()
        self.surface_spin = double_spin(0.0, 1.0, 0.5, 4, 0.01, tooltip=(
            "psi_N of the surface to inspect. Click the shape or a profile to "
            "pick one."))
        self.surface_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.surface_slider.setRange(0, 1000)
        row.addWidget(QtWidgets.QLabel("psi_N"))
        row.addWidget(self.surface_spin)
        row.addWidget(self.surface_slider, 1)
        layout.addLayout(row)
        self.surface_label = QtWidgets.QLabel("")
        self.surface_label.setStyleSheet(f"color: {MUTED};")
        layout.addWidget(self.surface_label)

        self.quantity_list = QtWidgets.QListWidget()
        self.quantity_list.setToolTip(
            "Quantities to plot against the solver-facing Boozer angle. Mixed "
            "components with one Theta index carry the sign of the q > 0 "
            "orientation.")
        bold = QtGui.QFont()
        bold.setBold(True)
        for title, items in SURFACE_GROUPS:
            header = QtWidgets.QListWidgetItem(title)
            header.setFlags(QtCore.Qt.NoItemFlags)
            header.setFont(bold)
            self.quantity_list.addItem(header)
            for key, _, name in items:
                item = QtWidgets.QListWidgetItem("    " + name)
                item.setData(QtCore.Qt.UserRole, key)
                item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Unchecked)
                self.quantity_list.addItem(item)
        self.quantity_list.setMinimumHeight(220)
        layout.addWidget(self.quantity_list)

        self.normalize_check = QtWidgets.QCheckBox("divide each by its mean over Theta")
        layout.addWidget(self.normalize_check)

        self.surface_spin.valueChanged.connect(self._on_surface_spin)
        self.surface_slider.valueChanged.connect(self._on_surface_slider)
        self.quantity_list.itemChanged.connect(self._on_view_edit)
        self._connect(self.normalize_check, self._on_view_edit)
        return box

    def _display_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Display")
        grid = QtWidgets.QGridLayout(box)
        self.grid_surfaces = int_spin(1, 200, 12, tooltip="Boozer surfaces drawn on the shape.")
        self.grid_spokes = int_spin(1, 256, 16, tooltip="Lines of constant Theta drawn.")
        grid.addWidget(QtWidgets.QLabel("surfaces / spokes"), 0, 0)
        grid.addWidget(self.grid_surfaces, 0, 1)
        grid.addWidget(self.grid_spokes, 0, 2)
        self.profile_x = QtWidgets.QComboBox()
        for key in PROFILE_AXES:
            self.profile_x.addItem({"psi_n": "psi_N", "rho_pol": "rho_pol"}[key], key)
        grid.addWidget(QtWidgets.QLabel("profile axis"), 1, 0)
        grid.addWidget(self.profile_x, 1, 1, 1, 2)
        self.contours_check = QtWidgets.QCheckBox(
            "flux contours from the file (should sit on the Boozer surfaces)")
        grid.addWidget(self.contours_check, 2, 0, 1, 3)
        for widget in (self.grid_surfaces, self.grid_spokes, self.profile_x,
                       self.contours_check):
            self._connect(widget, self._on_view_edit)
        return box

    def _refine_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Re-solve with TokaMaker (fixed boundary)")
        self.refine_box = box
        grid = QtWidgets.QGridLayout(box)
        intro = QtWidgets.QLabel(
            "Holds the pressure and the boundary, and fits the shape of FF' so "
            "the equilibrium's q matches the geometry-derived Boozer q. The "
            "refined file is loaded when it finishes.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {MUTED};")
        grid.addWidget(intro, 0, 0, 1, 4)

        self.refine_dir = QtWidgets.QLineEdit()
        self.refine_dir.setPlaceholderText("<eqdsk dir>/tokamaker_refined")
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._browse_refine_dir)
        grid.addWidget(QtWidgets.QLabel("output dir"), 1, 0)
        grid.addWidget(self.refine_dir, 1, 1, 1, 2)
        grid.addWidget(browse, 1, 3)
        self.refine_stem = QtWidgets.QLineEdit()
        self.refine_stem.setPlaceholderText("<eqdsk stem>_tokamaker")
        grid.addWidget(QtWidgets.QLabel("stem / COCOS"), 2, 0)
        grid.addWidget(self.refine_stem, 2, 1, 1, 2)
        self.refine_cocos = int_spin(1, 18, 7, tooltip="COCOS the refined file is written in.")
        grid.addWidget(self.refine_cocos, 2, 3)

        self.refine_coarse = double_spin(0.001, 1.0, 0.030, 4, 0.005, tooltip=(
            "Coarse mesh size [m]; the q fit runs on this mesh."))
        self.refine_fine = double_spin(0.001, 1.0, 0.015, 4, 0.005, tooltip=(
            "Final mesh size [m]."))
        grid.addWidget(QtWidgets.QLabel("mesh dx coarse / fine"), 3, 0)
        grid.addWidget(self.refine_coarse, 3, 1)
        grid.addWidget(self.refine_fine, 3, 2)
        self.refine_sequence = QtWidgets.QLineEdit()
        self.refine_sequence.setToolTip(
            "Mesh sizes the solution is carried through, strictly decreasing "
            "from the coarse size and ending at the fine one.")
        grid.addWidget(QtWidgets.QLabel("refine through"), 4, 0)
        grid.addWidget(self.refine_sequence, 4, 1, 1, 3)

        self.refine_order = int_spin(1, 5, 3, tooltip="Finite-element order.")
        self.refine_knots = int_spin(3, 64, 13, tooltip="Knots of the fitted FF' shape.")
        grid.addWidget(QtWidgets.QLabel("FE order / FF' knots"), 5, 0)
        grid.addWidget(self.refine_order, 5, 1)
        grid.addWidget(self.refine_knots, 5, 2)
        self.refine_iterations = int_spin(1, 500, 20, tooltip="Outer Gauss-Newton iterations.")
        self.refine_rms = SciEdit(2.0e-2, minimum=0.0, tooltip=(
            "Target absolute q RMS. The 13-knot problem floors near 1.7e-2 on "
            "the default coarse mesh."))
        grid.addWidget(QtWidgets.QLabel("iterations / target RMS"), 6, 0)
        grid.addWidget(self.refine_iterations, 6, 1)
        grid.addWidget(self.refine_rms, 6, 2)
        self.refine_threads = int_spin(1, 256, 2, tooltip="TokaMaker threads.")
        grid.addWidget(QtWidgets.QLabel("threads"), 7, 0)
        grid.addWidget(self.refine_threads, 7, 1)
        self.refine_current = QtWidgets.QCheckBox("also fit the total current")
        self.refine_current.setToolTip(
            "Off keeps the recorded plasma current and fits only the FF' shape.")
        grid.addWidget(self.refine_current, 7, 2, 1, 2)
        self.refine_nr = int_spin(33, 4097, 257, 32)
        self.refine_nz = int_spin(33, 4097, 257, 32)
        grid.addWidget(QtWidgets.QLabel("export grid"), 8, 0)
        grid.addWidget(self.refine_nr, 8, 1)
        grid.addWidget(self.refine_nz, 8, 2)

        self.refine_share = QtWidgets.QCheckBox(
            "pass the re-solved equilibrium on to the continuum and solver tabs")
        self.refine_share.setToolTip(
            "Read when the re-solve finishes. Ticked, those tabs switch to the "
            "refined file; unticked, they keep the equilibrium they have and "
            "the refined one is only shown here. Ticking it afterwards does not "
            "pass on an earlier re-solve -- re-solve again.")
        grid.addWidget(self.refine_share, 9, 0, 1, 4)
        self.refine_button = QtWidgets.QPushButton("Re-solve with TokaMaker")
        self.refine_button.clicked.connect(self._start_refinement)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel_refinement)
        self.cancel_button.setEnabled(False)
        grid.addWidget(self.refine_button, 10, 0, 1, 3)
        grid.addWidget(self.cancel_button, 10, 3)
        self.refine_note = QtWidgets.QLabel("")
        self.refine_note.setWordWrap(True)
        grid.addWidget(self.refine_note, 11, 0, 1, 4)
        if not tokamaker_available():
            self.refine_note.setText(
                f"<span style='color:{WARNING}'>OpenFUSIONToolkit is not "
                "importable from this Python, so TokaMaker cannot run.</span>")

        for widget in (self.refine_dir, self.refine_stem, self.refine_cocos,
                       self.refine_coarse, self.refine_fine, self.refine_sequence,
                       self.refine_order, self.refine_knots, self.refine_iterations,
                       self.refine_rms, self.refine_threads, self.refine_current,
                       self.refine_nr, self.refine_nz, self.refine_share):
            self._connect(widget, self._on_refine_edit)
        return box

    def _summary_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Summary")
        layout = QtWidgets.QVBoxLayout(box)
        self.summary_label = QtWidgets.QLabel("Nothing loaded yet.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.summary_label)
        return box

    def _connect(self, widget, slot) -> None:
        if isinstance(widget, QtWidgets.QLineEdit):
            widget.editingFinished.connect(slot)
        elif isinstance(widget, QtWidgets.QComboBox):
            widget.currentIndexChanged.connect(slot)
        elif isinstance(widget, QtWidgets.QAbstractButton):
            widget.toggled.connect(slot)
        else:
            widget.valueChanged.connect(slot)

    # ------------------------------------------------------------- state I/O

    def _write_state_to_widgets(self) -> None:
        self._loading_ui = True
        s = self.state
        self.source_combo.setCurrentIndex(max(self.source_combo.findData(s.source), 0))
        self.species_combo.setCurrentText(s.species)
        self.eqdsk_edit.setText(s.eqdsk)
        self.cocos_spin.setValue(s.cocos_in)
        self.density_edit.setText(s.density_file)
        self.density_uniform.setValue(s.density_uniform_m3)
        self.density_units.setText(s.density_units)
        self.density_x.setCurrentText(s.density_x)
        self.density_interp.setCurrentText(s.density_interpolate_in)
        self.boozer_min.setValue(s.boozer_min)
        self.boozer_max.setValue(s.boozer_max)
        self.boozer_npsi.setValue(s.boozer_npsi)
        self.boozer_ntheta.setValue(s.boozer_ntheta)
        self.m_max_spin.setValue(s.m_max)
        self.itpa_aminor.setValue(s.itpa_aminor)
        self.itpa_rmajor.setValue(s.itpa_rmajor)
        self.itpa_b0.setValue(s.itpa_b0)
        self.itpa_q0.setValue(s.itpa_q0)
        self.itpa_qedge.setValue(s.itpa_qedge)
        self.itpa_bt_sign.setCurrentIndex(max(self.itpa_bt_sign.findData(s.itpa_bt_sign), 0))
        self.itpa_ns.setValue(s.itpa_ns)
        self.itpa_ntheta.setValue(s.itpa_ntheta)
        self.itpa_m_max.setValue(s.itpa_m_max)
        self.surface_spin.setValue(s.surface_psi_n)
        self.surface_slider.setValue(int(round(s.surface_psi_n * 1000)))
        chosen = set(s.surface_quantities)
        for row in range(self.quantity_list.count()):
            item = self.quantity_list.item(row)
            key = item.data(QtCore.Qt.UserRole)
            if key:
                item.setCheckState(QtCore.Qt.Checked if key in chosen
                                   else QtCore.Qt.Unchecked)
        self.normalize_check.setChecked(s.normalize_surface)
        self.grid_surfaces.setValue(s.grid_surfaces)
        self.grid_spokes.setValue(s.grid_spokes)
        self.profile_x.setCurrentIndex(max(self.profile_x.findData(s.profile_x), 0))
        self.contours_check.setChecked(s.show_flux_contours)
        self.refine_dir.setText(s.refine_output_dir)
        self.refine_stem.setText(s.refine_stem)
        self.refine_cocos.setValue(s.refine_output_cocos)
        self.refine_coarse.setValue(s.refine_coarse_dx)
        self.refine_fine.setValue(s.refine_fine_dx)
        self.refine_sequence.setText(s.refine_dx_sequence)
        self.refine_order.setValue(s.refine_fe_order)
        self.refine_knots.setValue(s.refine_ffp_knots)
        self.refine_iterations.setValue(s.refine_max_iterations)
        self.refine_rms.setValue(s.refine_target_rms)
        self.refine_threads.setValue(s.refine_nthreads)
        self.refine_current.setChecked(s.refine_fit_current)
        self.refine_nr.setValue(s.refine_export_nr)
        self.refine_nz.setValue(s.refine_export_nz)
        self.refine_share.setChecked(s.refine_share)
        self._loading_ui = False

    def _read_widgets_into_state(self) -> None:
        s = self.state
        s.source = self.source_combo.currentData()
        s.species = self.species_combo.currentText()
        s.eqdsk = self.eqdsk_edit.text().strip()
        s.cocos_in = self.cocos_spin.value()
        s.density_file = self.density_edit.text().strip()
        s.density_uniform_m3 = self.density_uniform.value()
        s.density_units = self.density_units.text().strip() or "m^-3"
        s.density_x = self.density_x.currentText()
        s.density_interpolate_in = self.density_interp.currentText()
        s.boozer_min = self.boozer_min.value()
        s.boozer_max = self.boozer_max.value()
        s.boozer_npsi = self.boozer_npsi.value()
        s.boozer_ntheta = self.boozer_ntheta.value()
        s.m_max = self.m_max_spin.value()
        s.itpa_aminor = self.itpa_aminor.value()
        s.itpa_rmajor = self.itpa_rmajor.value()
        s.itpa_b0 = self.itpa_b0.value()
        s.itpa_q0 = self.itpa_q0.value()
        s.itpa_qedge = self.itpa_qedge.value()
        s.itpa_bt_sign = self.itpa_bt_sign.currentData()
        s.itpa_ns = self.itpa_ns.value()
        s.itpa_ntheta = self.itpa_ntheta.value()
        s.itpa_m_max = self.itpa_m_max.value()
        s.surface_psi_n = self.surface_spin.value()
        s.surface_quantities = [
            self.quantity_list.item(row).data(QtCore.Qt.UserRole)
            for row in range(self.quantity_list.count())
            if self.quantity_list.item(row).data(QtCore.Qt.UserRole)
            and self.quantity_list.item(row).checkState() == QtCore.Qt.Checked
        ]
        s.normalize_surface = self.normalize_check.isChecked()
        s.grid_surfaces = self.grid_surfaces.value()
        s.grid_spokes = self.grid_spokes.value()
        s.profile_x = self.profile_x.currentData()
        s.show_flux_contours = self.contours_check.isChecked()
        s.refine_output_dir = self.refine_dir.text().strip()
        s.refine_stem = self.refine_stem.text().strip()
        s.refine_output_cocos = self.refine_cocos.value()
        s.refine_coarse_dx = self.refine_coarse.value()
        s.refine_fine_dx = self.refine_fine.value()
        s.refine_dx_sequence = self.refine_sequence.text().strip()
        s.refine_fe_order = self.refine_order.value()
        s.refine_ffp_knots = self.refine_knots.value()
        s.refine_max_iterations = self.refine_iterations.value()
        s.refine_target_rms = self.refine_rms.value()
        s.refine_nthreads = self.refine_threads.value()
        s.refine_fit_current = self.refine_current.isChecked()
        s.refine_export_nr = self.refine_nr.value()
        s.refine_export_nz = self.refine_nz.value()
        s.refine_share = self.refine_share.isChecked()

    def set_state(self, state: EquilibriumState) -> None:
        """Replace every control's value, and load if the state is complete."""
        self.state = state
        self._write_state_to_widgets()
        self._sync_source_page()
        self._draw_all()
        if not self.state.validate():
            self.load()

    # ---------------------------------------------------------------- edits

    def _sync_source_page(self, *_) -> None:
        source = self.source_combo.currentData()
        self.source_stack.setCurrentIndex(SOURCES.index(source))
        self.refine_box.setEnabled(source == "eqdsk")

    def _on_species_edit(self, *_) -> None:
        """Pass the new ions on with the loaded equilibrium; nothing reloads."""
        if self._loading_ui:
            return
        self.state.species = self.species_combo.currentText()
        if self.view is None or self.view.species == self.state.species:
            return
        self.view = dataclasses.replace(self.view, species=self.state.species)
        self._write_summary()
        if self.shared:
            self.equilibriumChanged.emit(self.view, True)
        self.status.showMessage(f"Ion species {self.state.species}: passed on to the "
                                "continuum and solver tabs" if self.shared else
                                f"Ion species {self.state.species}")

    def _on_input_edit(self, *_) -> None:
        """An input changed: the loaded equilibrium is now stale, not reloaded."""
        if self._loading_ui:
            return
        self._read_widgets_into_state()
        stale = (self.view is not None
                 and self.state.equilibrium_key() != self._loaded_key)
        self.load_button.setText("Load equilibrium (inputs changed)" if stale
                                 else "Load equilibrium")
        problems = self.state.validate()
        if problems:
            self.status.showMessage(problems[0])
        elif stale:
            self.status.showMessage("Inputs changed; press Load to recompute.")

    def _on_view_edit(self, *_) -> None:
        if self._loading_ui:
            return
        self._read_widgets_into_state()
        problems = self.state.view_problems()
        if problems:
            self.status.showMessage(problems[0])
            return
        self._redraw_timer.start()

    def _on_refine_edit(self, *_) -> None:
        if self._loading_ui:
            return
        self._read_widgets_into_state()

    def _on_surface_spin(self, value: float) -> None:
        if self._loading_ui:
            return
        self.set_surface(value)

    def _on_surface_slider(self, value: int) -> None:
        if self._loading_ui:
            return
        self.set_surface(value / 1000.0)

    def set_surface(self, psi_n: float) -> None:
        """Pick the flux surface; redraws, never reloads."""
        psi_n = float(np.clip(psi_n, 0.0, 1.0))
        self.state.surface_psi_n = psi_n
        self._loading_ui = True
        self.surface_spin.setValue(psi_n)
        self.surface_slider.setValue(int(round(psi_n * 1000)))
        self._loading_ui = False
        self._redraw_timer.start()

    # -------------------------------------------------------------- loading

    def load(self) -> None:
        self._read_widgets_into_state()
        problems = self.state.validate()
        if problems:
            self.status.showMessage(problems[0])
            return
        if self._loading:
            self._reload = True
            return
        self._loading = True
        self.load_button.setEnabled(False)
        self.load_action.setEnabled(False)
        self.status.showMessage("Loading...")
        self._requested_key = self.state.equilibrium_key()
        self.compute.submit(
            "load_equilibrium", self.state.to_dict(), done=self._on_loaded,
            failed=self._on_load_failed, progressed=self.status.showMessage)

    def _finish_loading(self) -> bool:
        self._loading = False
        self.load_button.setEnabled(True)
        self.load_action.setEnabled(True)
        if self._reload:
            self._reload = False
            self.load()
            return True
        return False

    def _on_loaded(self, view: EquilibriumView) -> None:
        refined = self._comparison_pending
        if not refined:
            self.before = None
        self._comparison_pending = False
        changed = not same_equilibrium(view, self.view)
        self.view = view
        self._loaded_key = self._requested_key
        if changed:
            self.shared = self._share_refined if refined else True
        self.export_action.setEnabled(True)
        self.load_button.setText("Load equilibrium")
        self._write_summary()
        if changed:
            # a new equilibrium: the zooms chosen on the old one no longer apply
            for canvas in (self.shape_canvas, self.profile_canvas, self.surface_canvas):
                canvas.view_memory.forget()
        self._draw_all()
        message = (f"{view.title}: {view.psi_n.size} x {view.theta.size} Boozer grid "
                   f"in {view.load_seconds:.1f} s")
        if view.density_is_placeholder:
            message += "  |  placeholder density: geometry only"
        self.status.showMessage(message)
        if changed:
            if not self.shared:
                message += "  |  kept to this tab"
                self.status.showMessage(message)
            self.equilibriumChanged.emit(view, self.shared)
        self._finish_loading()

    def _on_load_failed(self, message: str, detail: str) -> None:
        self._comparison_pending = False
        self.status.showMessage(message)
        self.summary_label.setText(
            f"<span style='color:{ERROR}'>Could not load: {_escape(message)}</span>")
        if self._finish_loading():
            return
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("The equilibrium could not be loaded")
        box.setText(message)
        box.setDetailedText(detail)
        box.exec()

    def _write_summary(self) -> None:
        view = self.view
        if view is None:
            self.summary_label.setText("Nothing loaded yet.")
            return
        s = view.summary
        lines = [f"<b>{_escape(view.title)}</b> ({view.source}, loaded in "
                 f"{view.load_seconds:.1f} s)"]
        geometry = (f"R<sub>axis</sub> = {s['R_axis_m']:.3f} m, a = {s['a_minor_m']:.3f} m, "
                    f"B<sub>axis</sub> = {s['B_axis_T']:.3f} T")
        if "Ip_MA" in s:
            geometry += f", I<sub>p</sub> = {s['Ip_MA']:.3f} MA"
        lines.append(geometry)
        if "q_min" in s:
            q = (f"q: {s['q_inner']:.3f} innermost, {s['q_min']:.3f} minimum, "
                 f"{s['q_edge']:.3f} outermost")
            if "q95" in s:
                q += f", q<sub>95</sub> = {s['q95']:.3f}"
            lines.append(q)
        low, high = s["psi_n_range"]
        lines.append(f"Boozer grid: {s['surfaces']} surfaces x {s['poloidal_points']} "
                     f"angles over psi_N {low:.3g} - {high:.3g}")
        lines.append("Theta orientation: as computed" if s["theta_orientation"] == 1
                     else "Theta orientation: reflected so that q &gt; 0")
        if "q_boozer_vs_file_rms" in s:
            rms = s["q_boozer_vs_file_rms"]
            colour = WARNING if rms > 0.01 else MUTED
            lines.append(
                f"<span style='color:{colour}'>Boozer q vs the file's q: "
                f"{100 * rms:.2f} % RMS"
                + (" -- an inconsistent equilibrium; TokaMaker can fix it"
                   if rms > 0.01 else "") + "</span>")
        if view.density_is_placeholder:
            lines.append(f"<span style='color:{WARNING}'>No density profile: a "
                         f"uniform placeholder is used. Geometry only -- the solver "
                         f"tab needs a real density.</span>")
        elif "n_axis_m3" in s:
            lines.append(f"n<sub>axis</sub> = {s['n_axis_m3']:.3g} m<sup>-3</sup> "
                         f"of {_escape(view.species)}")
        if not self.shared:
            lines.append(f"<span style='color:{WARNING}'>Kept to this tab: the "
                         "continuum and solver tabs still use the equilibrium they "
                         "had. Re-solve with 'pass it on' ticked to hand this one "
                         "over.</span>")
        if self.before is not None:
            lines.append(f"<span style='color:{BEFORE_COLOUR}'>Dotted grey on the "
                         f"profiles: {_escape(self.before.title)} before refinement."
                         f"</span>")
        self.summary_label.setText("<br>".join(lines))

    # -------------------------------------------------------------- drawing

    def _draw_all(self) -> None:
        self._draw_shape()
        self._draw_profiles()
        self._draw_surface()

    @staticmethod
    def _placeholder(figure, text):
        figure.clear()
        axes = figure.add_subplot(111)
        axes.text(0.5, 0.5, text, ha="center", va="center",
                  transform=axes.transAxes, color="0.45")
        axes.set_axis_off()
        return axes

    def _closed(self, values):
        return values if self.view.theta_closes else np.append(values, values[:1])

    def _draw_shape(self) -> None:
        memory = self.shape_canvas.view_memory
        with memory.drawing():
            figure = self.shape_figure
            view = self.view
            if view is None:
                self.shape_axes = self._placeholder(figure, "Load an equilibrium")
                self.shape_canvas.draw_idle()
                return
            figure.clear()
            ax = figure.add_subplot(111)
            self.shape_axes = ax
            s = self.state
            npsi, ntheta = view.R.shape
            surfaces = np.unique(np.linspace(0, npsi - 1, s.grid_surfaces).round().astype(int))
            periodic = ntheta - 1 if view.theta_closes else ntheta
            spokes = np.unique((np.arange(s.grid_spokes) * periodic / s.grid_spokes).astype(int))
            selected = view.surface_index(s.surface_psi_n)
            handles = []

            if view.limiter is not None:
                ax.plot(view.limiter[:, 0], view.limiter[:, 1], color="0.2", lw=1.0)
            if s.show_flux_contours and view.flux_map is not None:
                R, Z, psi_n, inside = view.flux_map
                levels = np.unique(view.psi_n[surfaces])
                levels = levels[(levels > 1.0e-3) & (levels < 1.0)]
                # Inside the plasma only: outside it, a re-solved file's flux is
                # whatever the export padded it with, and the private-flux region
                # repeats psi_N < 1 below the X-point.
                core = np.where(inside, psi_n, np.nan)
                if levels.size:
                    ax.contour(R, Z, core.T, levels=levels, colors="0.55",
                               linewidths=1.6, alpha=0.35)
                    handles.append(Line2D([], [], color="0.55", lw=1.6, alpha=0.35,
                                          label=r"$\psi_N$ contours of the file"))

            ax.plot(view.boundary[:, 0], view.boundary[:, 1], color="k", lw=1.5)
            for index in surfaces:
                ax.plot(self._closed(view.R[index]), self._closed(view.Z[index]),
                        color=SURFACE_COLOUR, lw=0.8)
            for index in spokes:
                ax.plot(view.R[:, index], view.Z[:, index], color=SPOKE_COLOUR,
                        lw=1.6 if index == 0 else 0.6)
            handles.append(Line2D([], [], color=SURFACE_COLOUR, lw=0.8, label="Boozer surfaces"))
            handles.append(Line2D([], [], color=SPOKE_COLOUR, lw=0.6,
                                  label=r"constant $\Theta$ (bold: $\Theta=0$)"))

            # which way Theta increases, on the outermost surface
            start, end = periodic // 16, periodic // 8
            ax.annotate("", xy=(view.R[-1, end], view.Z[-1, end]),
                        xytext=(view.R[-1, start], view.Z[-1, start]),
                        arrowprops=dict(arrowstyle="->", color=SPOKE_COLOUR, lw=1.6))
            ax.text(view.R[-1, end], view.Z[-1, end], r"  $\Theta$", color=SPOKE_COLOUR)

            ax.plot(self._closed(view.R[selected]), self._closed(view.Z[selected]),
                    color=SELECTED_COLOUR, lw=2.0)
            handles.append(Line2D([], [], color=SELECTED_COLOUR, lw=2.0,
                                  label=rf"$\psi_N$ = {view.psi_n[selected]:.4f}"))
            ax.plot(*view.axis, marker="+", color="k", markersize=10)

            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel("R [m]")
            ax.set_ylabel("Z [m]")
            ax.set_title(f"{view.title}: magnetic coordinates, {npsi} x {ntheta}",
                         fontsize=10)
            ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=False)
            ax.grid(alpha=0.2)
            memory.keep(ax, "shape")
            self.shape_canvas.draw_idle()

    def _draw_profiles(self) -> None:
        memory = self.profile_canvas.view_memory
        with memory.drawing():
            figure = self.profile_figure
            view = self.view
            if view is None:
                self._profile_axes = []
                self._placeholder(figure, "Load an equilibrium")
                self.profile_canvas.draw_idle()
                return
            figure.clear()
            axes = figure.subplots(2, 4, sharex=True)
            self._profile_axes = list(axes.ravel())
            kind = self.state.profile_x
            x = view.profile_x(kind)
            p = view.profiles
            before = self.before
            xb = before.profile_x(kind) if before is not None else None
            selected = view.profile_x(kind)[view.surface_index(self.state.surface_psi_n)]

            def overlay(ax, name, scale=1.0):
                if before is not None and name in before.profiles:
                    ax.plot(xb, before.profiles[name] * scale, color=BEFORE_COLOUR,
                            ls=":", lw=1.4, label="before")

            ax = axes[0, 0]
            ax.plot(x, p["q"], color=SURFACE_COLOUR, label="Boozer")
            if view.file_q is not None:
                fx = np.sqrt(view.file_q[0]) if kind == "rho_pol" else view.file_q[0]
                ax.plot(fx, view.file_q[1], color=SPOKE_COLOUR, ls="--", lw=1.0, label="file")
            overlay(ax, "q")
            ax.set_title("safety factor q", fontsize=9)
            ax.legend(fontsize=7, frameon=False)

            ax = axes[0, 1]
            ax.plot(x, p["shear"], color=SURFACE_COLOUR)
            overlay(ax, "shear")
            ax.set_title(r"shear $\rho_{pol}\,q'/q$", fontsize=9)

            ax = axes[0, 2]
            ax.plot(x, p["pressure"] / 1.0e3, color=SURFACE_COLOUR)
            overlay(ax, "pressure", 1.0e-3)
            ax.set_title("pressure [kPa]", fontsize=9)

            ax = axes[0, 3]
            ax.plot(x, p["density"] / 1.0e19,
                    color=WARNING if view.density_is_placeholder else SURFACE_COLOUR)
            overlay(ax, "density", 1.0e-19)
            ax.set_title("density [1e19 m^-3]", fontsize=9)
            if view.density_is_placeholder:
                ax.text(0.5, 0.5, "uniform placeholder:\nno density file given",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=8, color=WARNING)

            ax = axes[1, 0]
            for name, colour in (("F", SURFACE_COLOUR), ("I", SPOKE_COLOUR), ("h", "#0E7466")):
                ax.plot(x, p[name], color=colour, label=name)
            ax.set_title("F = g, I, h = qF + I  [T m]", fontsize=9)
            ax.legend(fontsize=7, frameon=False)

            ax = axes[1, 1]
            if "B_avg" in p:
                ax.fill_between(x, p["B_min"], p["B_max"], color=SURFACE_COLOUR, alpha=0.18,
                                label="min - max")
                ax.plot(x, p["B_avg"], color=SURFACE_COLOUR, label="surface average")
                ax.legend(fontsize=7, frameon=False)
            ax.set_title("|B| on each surface [T]", fontsize=9)

            ax = axes[1, 2]
            ax.plot(x, p["dp_dpsi"] / 1.0e3, color=SURFACE_COLOUR)
            overlay(ax, "dp_dpsi", 1.0e-3)
            ax.set_title(r"$dp/d\psi$ [kPa rad/Wb]", fontsize=9)

            ax = axes[1, 3]
            ax.plot(x, p["ffprime"], color=SURFACE_COLOUR)
            overlay(ax, "ffprime")
            ax.set_title(r"$F\,dF/d\psi$ [T rad]", fontsize=9)

            label = r"$\rho_{pol}$" if kind == "rho_pol" else r"$\psi_N$"
            for ax in self._profile_axes:
                ax.axvline(selected, color=SELECTED_COLOUR, lw=0.9, ls="--", alpha=0.8)
                ax.grid(alpha=0.25)
                ax.tick_params(labelsize=8)
            for ax in axes[1]:
                ax.set_xlabel(label)
            for (row, column), axis in np.ndenumerate(axes):
                memory.keep(axis, f"profile {row} {column}")
            self.profile_canvas.draw_idle()

    def _draw_surface(self) -> None:
        memory = self.surface_canvas.view_memory
        with memory.drawing():
            figure = self.surface_figure
            view = self.view
            if view is None:
                self._placeholder(figure, "Load an equilibrium")
                self.surface_canvas.draw_idle()
                self.surface_label.setText("")
                return
            names = [name for name in self.state.surface_quantities if view.available(name)]
            data = view.surface(self.state.surface_psi_n, names)
            self.surface_label.setText(
                f"nearest Boozer surface: psi_N = {data['psi_n']:.5f} "
                f"(surface {data['index'] + 1} of {view.psi_n.size})")
            if not names:
                self._placeholder(figure, "Tick some quantities in the list")
                self.surface_canvas.draw_idle()
                return
            figure.clear()
            axes = figure.subplots(len(names), 1, sharex=True, squeeze=False)[:, 0]
            theta = data["theta"] / np.pi
            for ax, name in zip(axes, names):
                values = data["values"][name]
                units = view.units_of(name)
                label = SURFACE_LABELS[name]
                ylabel = f"{label}\n[{_units(units)}]" if units else label
                if self.state.normalize_surface:
                    mean = np.nanmean(view.periodic(values))
                    if np.isfinite(mean) and mean != 0.0:
                        values = values / mean
                        ylabel = f"{label}\n/ {mean:.4g} {_units(units)}".rstrip()
                ax.plot(theta, values, color=SURFACE_COLOUR, lw=1.1)
                ax.set_ylabel(ylabel, fontsize=8)
                ax.grid(alpha=0.25)
                ax.tick_params(labelsize=8)
            axes[-1].set_xlabel(r"$\Theta/\pi$ (solver-facing Boozer angle)")
            axes[-1].set_xlim(0.0, 2.0)
            axes[-1].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
            figure.suptitle(rf"$\psi_N$ = {data['psi_n']:.5f}  ({view.title})", fontsize=10)
            for axis, name in zip(axes, names):
                memory.keep(axis, f"surface {name}")
            self.surface_canvas.draw_idle()

        # --------------------------------------------------------- picking

    def _on_shape_click(self, event) -> None:
        if (self.view is None or event.inaxes is not self.shape_axes
                or event.xdata is None or toolbar_is_busy(self.shape_canvas)):
            return
        distance = (self.view.R - event.xdata) ** 2 + (self.view.Z - event.ydata) ** 2
        index = np.unravel_index(np.nanargmin(distance), distance.shape)[0]
        self.set_surface(float(self.view.psi_n[index]))

    def _on_profile_click(self, event) -> None:
        if (self.view is None or event.inaxes not in self._profile_axes
                or event.xdata is None or toolbar_is_busy(self.profile_canvas)):
            return
        x = float(event.xdata)
        self.set_surface(x * x if self.state.profile_x == "rho_pol" else x)

    # ---------------------------------------------------------- refinement

    def _start_refinement(self) -> None:
        self._read_widgets_into_state()
        problems = self.state.refinement_problems()
        if not tokamaker_available():
            problems.insert(0, "OpenFUSIONToolkit is not importable from this Python.")
        if self._process is not None:
            problems.insert(0, "A TokaMaker run is already in progress.")
        if problems:
            self.refine_note.setText(f"<span style='color:{ERROR}'>"
                                     f"{_escape(problems[0])}</span>")
            self.status.showMessage(problems[0])
            return

        spec = self.state.refinement_spec()
        handle, self._spec_path = tempfile.mkstemp(prefix="supernova_refine_",
                                                   suffix=".json")
        with os.fdopen(handle, "w") as stream:
            json.dump(spec, stream, indent=2)

        process = QtCore.QProcess(self)
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert("MPLBACKEND", "Agg")
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._on_refine_output)
        process.finished.connect(self._on_refine_finished)
        process.errorOccurred.connect(self._on_refine_error)
        self._process = process
        self._process_output = ""

        self.log.clear()
        self.log.appendPlainText(
            f"$ {sys.executable} -m supernova.gui._refine_job {self._spec_path}\n"
            f"source  {spec['source_eqdsk']} (COCOS {spec['source_cocos']})\n"
            f"output  {spec['output_dir']}/{spec['stem']}.eqdsk "
            f"(COCOS {spec['output_cocos']})\n")
        self.pages.setCurrentIndex(PAGES.index("TokaMaker log"))
        self.refine_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.refine_note.setText(f"<span style='color:{MUTED}'>TokaMaker is running; "
                                 "minutes, not seconds.</span>")
        self.status.showMessage("TokaMaker is re-solving the equilibrium...")
        process.start(sys.executable, ["-u", "-m", "supernova.gui._refine_job",
                                       self._spec_path])

    def _on_refine_output(self) -> None:
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardOutput()).decode(errors="replace")
        self._process_output += text
        cursor = self.log.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def _on_refine_error(self, error) -> None:
        if self.sender() is not self._process:
            return                      # cancelled; already reported
        if error == QtCore.QProcess.FailedToStart:
            self._refinement_done(None, "TokaMaker could not be started.")

    def _on_refine_finished(self, exit_code: int, _status) -> None:
        if self.sender() is not self._process:
            return                      # cancelled; already reported
        self._on_refine_output()
        result = None
        for line in reversed(self._process_output.splitlines()):
            if line.startswith(RESULT_MARKER):
                try:
                    result = json.loads(line[len(RESULT_MARKER):])
                except ValueError:
                    result = None
                break
        if result is None:
            self._refinement_done(None, f"TokaMaker exited with code {exit_code} "
                                        "and reported no result.")
        elif "error" in result:
            self._refinement_done(None, result["error"])
        else:
            self._refinement_done(result, "")

    def _cancel_refinement(self) -> None:
        if self._process is None:
            return
        process = self._process
        self._process = None
        process.kill()
        process.waitForFinished(2000)
        self.log.appendPlainText("\nCancelled.")
        self._refinement_done(None, "TokaMaker was cancelled.", quiet=True)

    def _refinement_done(self, diagnostics, message: str, quiet: bool = False) -> None:
        self._process = None
        if self._spec_path:
            Path(self._spec_path).unlink(missing_ok=True)
            self._spec_path = None
        self.refine_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

        if diagnostics is None:
            self.refine_note.setText(f"<span style='color:{ERROR}'>{_escape(message)}"
                                     "</span>")
            self.status.showMessage(message)
            if not quiet:
                box = QtWidgets.QMessageBox(self)
                box.setIcon(QtWidgets.QMessageBox.Warning)
                box.setWindowTitle("TokaMaker did not produce an equilibrium")
                box.setText(message)
                box.setDetailedText("\n".join(self._process_output.splitlines()[-60:]))
                box.exec()
            return

        note = (f"Refined: q fit RMS {diagnostics['fit_rms_error']:.3g}, fine-mesh "
                f"q {100 * diagnostics['fine_q_relative_rms']:.2f} % RMS, I_p error "
                f"{100 * diagnostics['plasma_current_relative_error']:.3f} %. "
                f"Wrote {diagnostics['output_eqdsk']}")
        # the choice is read now, when the re-solve finishes
        self._share_refined = bool(self.state.refine_share)
        note += (" Passing it on to the continuum and solver tabs."
                 if self._share_refined else " Kept to this tab, as asked.")
        self.refine_note.setText(f"<span style='color:{MUTED}'>{_escape(note)}</span>")
        self.log.appendPlainText("\n" + note)
        self.before = self.view
        self._comparison_pending = True
        self.state.eqdsk = str(diagnostics["output_eqdsk"])
        self.state.cocos_in = int(diagnostics["output_cocos"])
        self._write_state_to_widgets()
        self.pages.setCurrentIndex(PAGES.index("Profiles"))
        self.load()

    # -------------------------------------------------------------- actions

    def _browse_eqdsk(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open equilibrium", self.eqdsk_edit.text() or str(Path.cwd()),
            "Equilibria (*.eqdsk *.geqdsk *.gfile g*);;All files (*)")
        if path:
            self.eqdsk_edit.setText(path)
            self._on_input_edit()

    def _browse_density(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open density profile", self.density_edit.text() or str(Path.cwd()),
            "Profiles (*.dat *.txt *.csv);;All files (*)")
        if path:
            self.density_edit.setText(path)
            self._on_input_edit()

    def _browse_refine_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Refinement output directory",
            self.refine_dir.text() or str(Path(self.state.eqdsk or Path.cwd()).parent))
        if path:
            self.refine_dir.setText(path)
            self._on_refine_edit()

    def _export_dataset(self) -> None:
        if self.view is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export equilibrium data", "equilibrium.h5",
            "HDF5 (*.h5 *.hdf5);;NetCDF (*.nc);;All files (*)")
        if not path:
            return
        engine = "h5netcdf" if Path(path).suffix.lower() in (".h5", ".hdf5") else "netcdf4"
        try:
            self.view.to_dataset().to_netcdf(path, engine=engine)
        except Exception as error:
            self.status.showMessage(f"Could not write {path}: {error}")
            return
        self.status.showMessage(f"Wrote {path} via {engine}")

    def _save_figure(self) -> None:
        figure = (self.shape_figure, self.profile_figure,
                  self.surface_figure)[min(self.pages.currentIndex(), 2)]
        name = PAGES[min(self.pages.currentIndex(), 2)].lower()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save figure", f"equilibrium_{name}.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            figure.savefig(path, dpi=180, bbox_inches="tight")
            self.status.showMessage(f"Wrote {path}")

    def _save_session(self) -> None:
        if self.session_owner is not None:      # in the app: the whole session
            self.session_owner.save_session()
            return
        path = ask_session_to_save(self, "Save equilibrium session", "equilibrium_session.h5")
        if not path:
            return
        self._read_widgets_into_state()
        try:
            write_session(path, equilibrium=self.state)
        except (ValueError, OSError, TypeError) as error:
            self.status.showMessage(f"Could not save the session: {error}")
            return
        self.status.showMessage(f"Wrote {path}")

    def _load_session(self) -> None:
        if self.session_owner is not None:
            self.session_owner.open_session()
            return
        path = ask_session_to_open(self, "Load equilibrium session")
        if not path:
            return
        try:
            loaded = read_session(path)
        except (ValueError, OSError, TypeError) as error:
            self.status.showMessage(f"Could not load session: {error}")
            return
        state = loaded.get("equilibrium") or loaded.get("continuum_equilibrium")
        if state is None:
            self.status.showMessage(f"{path} holds no equilibrium settings.")
            return
        self.set_state(state)

    def shutdown(self) -> None:
        """Stop what this tab started: the TokaMaker process and, if owned, the thread."""
        if self._process is not None:
            self._cancel_refinement()
        if self._owns_compute:
            self.compute.shutdown()

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)


def _units(units: str) -> str:
    return units.replace("**", "^")


def _escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def main(argv=None) -> int:
    """Open the equilibrium tab on its own."""
    argv = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv[:1])
    state = EquilibriumState.from_json(argv[1]) if len(argv) > 1 else None
    window = EquilibriumWindow(state)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
