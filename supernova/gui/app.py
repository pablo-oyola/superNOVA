"""The superNOVA window: equilibrium, continuum and solver, one tab each.

Each tab is a window in its own right -- they open standalone too -- with its
own toolbar, status bar and session file. This one arranges them and carries
the equilibrium choices between them:

* every equilibrium the equilibrium tab loads goes to the solver tab, and an
  eqdsk's file, COCOS, Boozer domain and density file are written into the
  continuum tab -- unless it is a TokaMaker re-solve the user kept to the
  equilibrium tab (its "pass it on" box);
* an empty equilibrium tab takes the continuum tab's file and density when it
  is opened, without loading them.

The continuum tab is the existing continuum window, unchanged; it loads its
own copy of the equilibrium from the choices it is given.

The equilibrium and solver tabs share one compute process (see
:mod:`supernova.gui._qt` for why it is a process), which holds the loaded
equilibria and runs the solver's worker pool; it starts with the window. The
continuum tab keeps its own short-lived worker threads, as before.

A session written from the File menu holds all three tabs' states; a
continuum-only session from the old single window still opens.
"""

from __future__ import annotations

from pathlib import Path
import sys

from PySide6 import QtCore, QtGui, QtWidgets

from supernova.gui._qt import ComputeClient
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
        self.continuum = ContinuumWindow(sessions.get("continuum"))
        self.solver = SolverWindow(sessions.get("solver"), self.compute)
        self.equilibrium.equilibriumChanged.connect(self._on_equilibrium)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        for name, page in zip(TAB_NAMES, (self.equilibrium, self.continuum, self.solver)):
            # a QMainWindow can be a page: it keeps its toolbar and status bar
            page.setWindowFlags(QtCore.Qt.Widget)
            self.tabs.addTab(page, name)
        self.setCentralWidget(self.tabs)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.busy_label = QtWidgets.QLabel()
        self.tabs.setCornerWidget(self.busy_label, QtCore.Qt.TopRightCorner)
        self.compute.busyChanged.connect(self._on_busy)
        self._build_menu()

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("&File")
        open_action = menu.addAction("Open session...")
        # Ctrl+O and Ctrl+S belong to the continuum tab's own session buttons
        open_action.setShortcut("Ctrl+Shift+O")
        open_action.triggered.connect(self._open_session)
        save_action = menu.addAction("Save session...")
        save_action.setShortcut("Ctrl+Shift+S")
        save_action.triggered.connect(self._save_session)
        menu.addSeparator()
        quit_action = menu.addAction("Quit")
        quit_action.setShortcut(QtGui.QKeySequence.Quit)
        quit_action.triggered.connect(self.close)

    # ------------------------------------------------- sharing between tabs

    def _on_equilibrium(self, view, share: bool) -> None:
        if not share:
            return
        self.solver.set_equilibrium(view)
        self._share_with_continuum(view.inputs)

    def _share_with_continuum(self, inputs: dict) -> None:
        """Write the equilibrium tab's choices into the continuum tab.

        Only what differs is written, so a load that changes nothing for the
        continuum does not make it reload its own copy.
        """
        if not inputs:
            return                      # ITPA: the continuum tab cannot take it
        continuum = self.continuum
        continuum._read_widgets_into_state()
        state = continuum.state
        changed = {name: value for name, value in inputs.items()
                   if getattr(state, name) != value}
        if not changed:
            return
        for name, value in changed.items():
            setattr(state, name, value)
        continuum._write_state_to_widgets()
        continuum._schedule_solve()
        continuum.status.showMessage(
            "Took the equilibrium tab's " + ", ".join(sorted(changed))
            + "; reloading.")

    def _on_tab_changed(self, index: int) -> None:
        """Opening an empty equilibrium tab offers it the continuum tab's choices."""
        if TAB_NAMES[index] != "Equilibrium":
            return
        equilibrium = self.equilibrium
        equilibrium._read_widgets_into_state()
        state = equilibrium.state
        if state.source != "eqdsk" or state.eqdsk or equilibrium.view is not None:
            return
        self.continuum._read_widgets_into_state()
        theirs = self.continuum.state
        if not theirs.eqdsk:
            return
        for name in ("eqdsk", "cocos_in", "boozer_min", "boozer_max"):
            setattr(state, name, getattr(theirs, name))
        if theirs.density_file and not state.density_file:
            for name in ("density_file", "density_units", "density_x",
                         "density_interpolate_in"):
                setattr(state, name, getattr(theirs, name))
        equilibrium._write_state_to_widgets()
        equilibrium.status.showMessage(
            "Filled in from the continuum tab's equilibrium; press Load to compute it.")

    def _on_busy(self, busy: bool) -> None:
        self.busy_label.setText("  working...  " if busy else "")

    def _save_session(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save session", "supernova_session.json", "JSON (*.json)")
        if not path:
            return
        try:
            self.equilibrium._read_widgets_into_state()
            self.continuum._read_widgets_into_state()
            self.solver._read_widgets_into_state()
            write_session(path, equilibrium=self.equilibrium.state,
                          continuum=self.continuum.state, solver=self.solver.state)
        except (ValueError, OSError) as error:
            self.statusBar().showMessage(f"Could not save the session: {error}")
            return
        self.statusBar().showMessage(f"Wrote {path}", 5000)

    def _open_session(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open session", str(Path.cwd()), "JSON (*.json)")
        if path:
            self.load_session(path)

    def load_session(self, path) -> None:
        try:
            sessions = read_session(path)
        except (ValueError, OSError, TypeError) as error:
            self.statusBar().showMessage(f"Could not load the session: {error}")
            return
        if "equilibrium" in sessions:
            self.equilibrium.set_state(sessions["equilibrium"])
        if "continuum" in sessions:
            self.continuum.state = sessions["continuum"]
            self.continuum._write_state_to_widgets()
            self.continuum._schedule_solve()
        if "solver" in sessions:
            self.solver.set_state(sessions["solver"])
        if set(sessions) == {"continuum"}:
            self.tabs.setCurrentIndex(TAB_NAMES.index("Continuum"))
        self.statusBar().showMessage(f"Loaded {path}", 5000)

    def closeEvent(self, event) -> None:
        self.equilibrium.shutdown()
        self.compute.shutdown()
        super().closeEvent(event)


def main(argv=None) -> int:
    """Open the window, with an optional session file. Returns the Qt exit code."""
    argv = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv[:1])
    sessions = read_session(argv[1]) if len(argv) > 1 else None
    window = SupernovaWindow(sessions)
    if sessions and set(sessions) == {"continuum"}:
        window.tabs.setCurrentIndex(TAB_NAMES.index("Continuum"))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
