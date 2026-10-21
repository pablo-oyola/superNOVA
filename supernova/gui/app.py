"""The superNOVA window: equilibrium, continuum and solver, one tab each.

Each tab has its own toolbar and status bar. This one arranges them and
passes the equilibrium on: every equilibrium the Equilibrium tab
loads goes to the Continuum and Solver tabs -- the same object, which the
compute process holds -- unless it is a TokaMaker re-solve the user kept to the
Equilibrium tab (its "pass it on" box). The Continuum and Solver tabs have no
equilibrium inputs of their own.

All three tabs share one compute process (see :mod:`supernova.gui._qt` for why
it is a process), which holds the loaded equilibria, solves the continuum and
runs the solver's worker pool; it starts with the window.

A session (:mod:`supernova.gui._session`) holds all three tabs' settings and
the solver's pass and refinement; the tabs' own Save and Load session buttons
save and open that same whole session. Opening one loads its equilibrium again
and puts the pass back once that load is done. JSON sessions from before still
open, settings only; one from the old single window sends its continuum
settings to the Continuum tab and its equilibrium file and density to the
Equilibrium tab's form, to be loaded there.
"""

from __future__ import annotations

import sys
import traceback

from PySide6 import QtCore, QtGui, QtWidgets

from supernova.gui._qt import ComputeClient, ask_session_to_open, ask_session_to_save
from supernova.gui._session import read_session, write_session
from supernova.gui.continuum import ContinuumWindow
from supernova.gui.equilibrium import EquilibriumWindow
from supernova.gui.solver import SolverWindow

TAB_NAMES = ("Equilibrium", "Continuum", "Solver")


class SupernovaWindow(QtWidgets.QMainWindow):
    """Three tabs and a File menu."""

    def __init__(self, sessions: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("superNOVA")
        self.resize(1400, 900)
        sessions = sessions or {}

        self.compute = ComputeClient(self)
        # It takes a few seconds to import the stack; start it while the
        # window is being built rather than at the first Load.
        self.compute.start()
        self.equilibrium = EquilibriumWindow(sessions.get("equilibrium"), self.compute)
        self.continuum = ContinuumWindow(sessions.get("continuum"), self.compute)
        self.solver = SolverWindow(sessions.get("solver"), self.compute)
        self.equilibrium.equilibriumChanged.connect(self._on_equilibrium)
        for tab in (self.equilibrium, self.continuum, self.solver):
            tab.session_owner = self

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        for name, page in zip(TAB_NAMES, (self.equilibrium, self.continuum, self.solver)):
            # a QMainWindow can be a page: it keeps its toolbar and status bar
            page.setWindowFlags(QtCore.Qt.Widget)
            self.tabs.addTab(page, name)
        self.setCentralWidget(self.tabs)

        self.busy_label = QtWidgets.QLabel()
        self.tabs.setCornerWidget(self.busy_label, QtCore.Qt.TopRightCorner)
        self.compute.busyChanged.connect(self._on_busy)
        self._build_menu()

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("&File")
        open_action = menu.addAction("Open session...")
        # Ctrl+O and Ctrl+S belong to the continuum tab's own session buttons,
        # which do the same
        open_action.setShortcut("Ctrl+Shift+O")
        open_action.triggered.connect(self.open_session)
        save_action = menu.addAction("Save session...")
        save_action.setShortcut("Ctrl+Shift+S")
        save_action.triggered.connect(self.save_session)
        menu.addSeparator()
        quit_action = menu.addAction("Quit")
        quit_action.setShortcut(QtGui.QKeySequence.Quit)
        quit_action.triggered.connect(self.close)

    # ------------------------------------------------- sharing between tabs

    def _on_equilibrium(self, view, share: bool) -> None:
        if not share:
            return
        self.continuum.set_equilibrium(view)
        self.solver.set_equilibrium(view)

    def _on_busy(self, busy: bool) -> None:
        self.busy_label.setText("  working...  " if busy else "")

    def save_session(self) -> None:
        path = ask_session_to_save(self, "Save session", "supernova_session.h5")
        if path:
            self.write_session(path)

    def write_session(self, path) -> None:
        """Every tab's settings, the continuum, and the solver's pass and refinement."""
        try:
            self.equilibrium._read_widgets_into_state()
            self.continuum._read_widgets_into_state()
            self.solver._read_widgets_into_state()
            write_session(path, equilibrium=self.equilibrium.state,
                          continuum=self.continuum.state, solver=self.solver.state,
                          results={"continuum": self.continuum.session_results(),
                                   "solver": self.solver.session_results()})
        except (ValueError, OSError, TypeError) as error:
            self._say(f"Could not save the session: {error}")
            return
        self._say(f"Wrote {path}")

    def open_session(self) -> None:
        path = ask_session_to_open(self, "Open session")
        if path:
            self.load_session(path)

    def _say(self, message: str, timeout: int = 5000) -> None:
        """On the window's status bar, and on the tab on show, whose bar is nearer."""
        self.statusBar().showMessage(message, timeout)
        self.tabs.currentWidget().statusBar().showMessage(message, timeout)

    def load_session(self, path) -> None:
        try:
            sessions = read_session(path)
        except (ValueError, OSError, TypeError) as error:
            self._say(f"Could not load the session: {error}")
            return
        legacy = sessions.pop("continuum_equilibrium", None)
        # the continuum is solved again, in milliseconds, once the equilibrium is in
        results = sessions.pop("results", {})
        if "equilibrium" in sessions:
            self.equilibrium.set_state(sessions["equilibrium"])
        elif legacy is not None:
            # the old single window's file and density: the Equilibrium tab's now
            self.equilibrium.set_state(legacy)
        if "continuum" in sessions:
            self.continuum.state = sessions["continuum"]
            if self.continuum.view is not None:
                self.continuum.state.species = self.continuum.view.species
            self.continuum._write_state_to_widgets()
            self.continuum._schedule_solve()
        if "solver" in sessions:
            self.solver.set_state(sessions["solver"])
        if "solver" in results:
            self.solver.restore(results["solver"])
        if legacy is not None and "equilibrium" not in sessions:
            self.tabs.setCurrentIndex(TAB_NAMES.index("Equilibrium"))
            self.statusBar().showMessage(
                f"Loaded {path}: its equilibrium is in the Equilibrium tab's form -- "
                "press Load there.", 10000)
            return
        self._say(f"Loaded {path}")

    def closeEvent(self, event) -> None:
        self.equilibrium.shutdown()
        self.continuum.shutdown()
        self.compute.shutdown()
        super().closeEvent(event)


def main(argv=None, start_tab: str | None = None) -> int:
    """Open the window, with an optional session file. Returns the Qt exit code."""
    argv = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv[:1])
    window = SupernovaWindow()
    _report_errors(window)
    if len(argv) > 1:
        window.load_session(argv[1])
    elif start_tab is not None:
        window.tabs.setCurrentIndex(TAB_NAMES.index(start_tab))
    window.show()
    return app.exec()


def _report_errors(window: SupernovaWindow) -> None:
    """Show in the window what a slot raises; Qt only prints it to the terminal.

    The traceback still goes to the terminal, and also to the Solver tab's Log,
    with one line on the status bar of the tab on show.
    """
    default = sys.excepthook

    def hook(kind, error, trace):
        default(kind, error, trace)
        try:
            text = "".join(traceback.format_exception(kind, error, trace))
            window.solver._append_log(f"\n=== error in the window ===\n{text}")
            window._say(f"Error: {kind.__name__}: {error} -- the traceback is in the "
                        "Solver tab's Log", 20000)
        except Exception:                 # the window may be half gone
            pass

    sys.excepthook = hook


if __name__ == "__main__":
    raise SystemExit(main())
