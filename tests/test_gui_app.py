"""The tabbed window, driven offscreen.

Like test_gui_window, each test runs in a subprocess with
QT_QPA_PLATFORM=offscreen, so the variable never leaks into the rest of the
session. The equilibrium used is the analytic ITPA case on a small grid, so
nothing here waits on a Boozer transform; the one solve is the ITPA band.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("PySide6")


def run_in_offscreen_qt(body: str) -> str:
    """Execute ``body`` with an open SupernovaWindow called ``window``."""
    script = textwrap.dedent('''
        import time
        from PySide6 import QtCore, QtWidgets
        from supernova.gui.app import SupernovaWindow
        from supernova.gui._equilibrium import EquilibriumState
        from supernova.gui._solver import SolverState

        app = QtWidgets.QApplication([])
        QtWidgets.QMessageBox.exec = lambda self: print("DIALOG", self.text())
        window = SupernovaWindow()

        def wait_until(condition, timeout=300):
            started = time.time()
            while not condition():
                app.processEvents(QtCore.QEventLoop.AllEvents, 50)
                if time.time() - started > timeout:
                    raise TimeoutError
                time.sleep(0.01)
            app.processEvents()

        def load_itpa():
            window.equilibrium.set_state(
                EquilibriumState(source="itpa", itpa_ns=48, itpa_ntheta=48))
            wait_until(lambda: window.equilibrium.view is not None
                       and not window.equilibrium._loading)
    ''') + textwrap.dedent(body)

    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        env=env, timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def test_the_window_has_three_tabs_and_the_continuum_one_is_the_old_window():
    out = run_in_offscreen_qt('''
        print("TABS", [window.tabs.tabText(i) for i in range(window.tabs.count())])
        print("CONTINUUM", type(window.continuum).__name__)
        print("SOLVER_SAYS", window.solver.equilibrium_label.text())
    ''')
    assert "TABS ['Equilibrium', 'Continuum', 'Solver']" in out
    assert "CONTINUUM ContinuumWindow" in out
    assert "Load an equilibrium" in out


def test_a_loaded_equilibrium_reaches_the_solver_tab():
    out = run_in_offscreen_qt('''
        load_itpa()
        print("SAME", window.solver.view is window.equilibrium.view)
        print("LABEL", window.solver.equilibrium_label.text())
        print("EXPORT", window.equilibrium.export_action.isEnabled())
    ''')
    assert "SAME True" in out
    assert "ITPA" in out
    assert "EXPORT True" in out


def test_picking_a_surface_moves_the_spin_the_slider_and_the_plots():
    out = run_in_offscreen_qt('''
        load_itpa()
        eq = window.equilibrium
        eq.set_surface(0.73)
        wait_until(lambda: not eq._redraw_timer.isActive())
        print("SPIN", eq.surface_spin.value(), "SLIDER", eq.surface_slider.value())
        print("LABEL", eq.surface_label.text())
        print("AXES", len(eq.surface_figure.axes))
    ''')
    assert "SPIN 0.73 SLIDER 730" in out
    assert "nearest Boozer surface" in out
    assert "AXES 4" in out          # the four quantities ticked by default


def test_a_placeholder_density_blocks_the_solve_with_a_reason():
    out = run_in_offscreen_qt('''
        from supernova.gui._equilibrium import EquilibriumModel
        view = EquilibriumModel().load(
            EquilibriumState(source="itpa", itpa_ns=16, itpa_ntheta=16))
        import dataclasses
        window.solver.set_equilibrium(dataclasses.replace(view, density_is_placeholder=True))
        window.solver.solve()
        print("SOLVING", window.solver._solving)
        print("STATUS", window.solver.status.currentMessage())
    ''')
    assert "SOLVING False" in out
    assert "density" in out


def test_the_planned_windows_are_drawn_before_any_solve():
    out = run_in_offscreen_qt('''
        solver = window.solver
        solver.set_state(SolverState(f_low_khz=60.0, f_high_khz=76.0,
                                     window_width_khz=4.0))
        spans = [p for p in solver.spectrum_axes.patches]
        print("BANDS", len(spans))
        print("TEXT", solver.windows_label.text())
    ''')
    assert "BANDS 4" in out
    assert "4 windows of 4 kHz" in out


def test_the_modelocate_form_round_trips_its_overrides():
    out = run_in_offscreen_qt('''
        solver = window.solver
        solver.set_state(SolverState(modelocate={"core_edge": 0.1,
                                                 "spike_level": 0.5}))
        solver._read_widgets_into_state()
        print("LOCATE", sorted(solver.state.modelocate.items()))
        solver.locate_form.reset()
        print("RESET", solver.state.modelocate)
    ''')
    assert "LOCATE [('core_edge', 0.1), ('spike_level', 0.5)]" in out
    assert "RESET {}" in out


def test_solving_itpa_fills_the_tables_and_a_settings_change_marks_it_stale():
    pytest.importorskip("slepc4py")
    out = run_in_offscreen_qt('''
        load_itpa()
        solver = window.solver
        solver.set_state(SolverState(ntor=6, mmax=20, max_dm=5, regularizer=1e-5,
                                     gamma_s=0.0, species="H", f_low_khz=68.0,
                                     f_high_khz=72.0, window_width_khz=4.0, nev=4,
                                     tol=1e-7, maxit=80, spans=100))
        solver.solve()
        wait_until(lambda: solver.result is not None and not solver._solving)
        everything = solver.mode_table.rowCount()
        print("MODES", everything > 0, "WINDOWS", solver.window_table.rowCount())
        print("SELECTED_IS_GAP", solver.result.labels[solver.selected])
        solver.gap_only_check.setChecked(True)
        print("FILTERED", 0 < solver.mode_table.rowCount() < everything)
        print("STALE_BEFORE", solver._result_is_stale())
        solver.nev_spin.setValue(5)
        print("STALE_AFTER", solver._result_is_stale())
    ''')
    assert "MODES True WINDOWS 1" in out
    assert "SELECTED_IS_GAP gap" in out           # a gap mode is picked first
    assert "FILTERED True" in out
    assert "STALE_BEFORE False" in out
    assert "STALE_AFTER True" in out


def test_closing_the_window_with_an_idle_worker_exits_cleanly():
    """Regression: an idle QThread worker aborted the process on exit."""
    out = run_in_offscreen_qt('''
        load_itpa()
        print("DONE")
    ''')
    assert "DONE" in out


def test_a_new_state_gets_its_own_continuum_preview_after_one_has_arrived():
    """Regression: set_state compared the new preview key with itself.

    Once the first preview had landed, loading a session with another n left
    the old continuum on screen and never computed the new one.
    """
    out = run_in_offscreen_qt('''
        load_itpa()
        solver = window.solver
        wait_until(lambda: solver.preview is not None)
        print("FIRST", solver.preview.key[0])
        solver.set_state(SolverState(ntor=6, mmax=20, gamma_s=0.0, species="H"))
        wait_until(lambda: solver.preview.key == solver.state.preview_key(), timeout=120)
        print("SECOND", solver.preview.key[0], solver.preview.which)
    ''')
    assert "FIRST 4" in out
    assert "SECOND 6 alfven" in out


# ------------------------------------------------------- sharing between tabs

def test_a_shared_load_reaches_the_solver_and_writes_the_continuum_choices():
    out = run_in_offscreen_qt('''
        import dataclasses
        from supernova.gui._equilibrium import EquilibriumModel
        view = EquilibriumModel().load(
            EquilibriumState(source="itpa", itpa_ns=16, itpa_ntheta=16))
        inputs = {"eqdsk": "/nowhere/shot.geqdsk", "cocos_in": 7,
                  "boozer_min": 0.01, "boozer_max": 0.98,
                  "density_file": "/nowhere/ne.dat", "density_units": "m^-3",
                  "density_x": "psi_n", "density_interpolate_in": "rho_pol"}
        window._on_equilibrium(dataclasses.replace(view, inputs=inputs), True)
        c = window.continuum
        print("SOLVER", window.solver.view is not None)
        print("CONTINUUM", c.eqdsk_edit.text(), c.cocos_spin.value(),
              c.density_edit.text(), c.boozer_max.value())
    ''')
    assert "SOLVER True" in out
    assert "CONTINUUM /nowhere/shot.geqdsk 7 /nowhere/ne.dat 0.98" in out


def test_an_itpa_load_leaves_the_continuum_tab_alone():
    out = run_in_offscreen_qt('''
        load_itpa()
        print("SOLVER", window.solver.view is window.equilibrium.view)
        print("CONTINUUM_EQDSK", repr(window.continuum.state.eqdsk))
    ''')
    assert "SOLVER True" in out
    assert "CONTINUUM_EQDSK ''" in out


def test_a_re_solve_kept_to_its_tab_is_not_passed_on():
    """The 'pass it on' box, as read when the re-solve finished."""
    out = run_in_offscreen_qt('''
        from supernova.gui._equilibrium import EquilibriumModel
        load_itpa()
        eq, solver = window.equilibrium, window.solver
        first = solver.view
        refined = EquilibriumModel().load(
            EquilibriumState(source="itpa", itpa_ns=24, itpa_ntheta=24))
        eq._comparison_pending, eq._share_refined = True, False
        eq._on_loaded(refined)
        print("KEPT", solver.view is first, eq.view is refined)
        print("SUMMARY_SAYS", "Kept to this tab" in eq.summary_label.text())
        again = EquilibriumModel().load(
            EquilibriumState(source="itpa", itpa_ns=20, itpa_ntheta=20))
        eq._comparison_pending, eq._share_refined = True, True
        eq._on_loaded(again)
        print("PASSED", solver.view is again)
    ''')
    assert "KEPT True True" in out
    assert "SUMMARY_SAYS True" in out
    assert "PASSED True" in out


def test_an_empty_equilibrium_tab_takes_the_continuum_tabs_choices():
    out = run_in_offscreen_qt('''
        c = window.continuum
        c.eqdsk_edit.setText("/nowhere/shot.geqdsk")
        c.density_edit.setText("/nowhere/ne.dat")
        c.cocos_spin.setValue(7)
        window.tabs.setCurrentIndex(1)
        window.tabs.setCurrentIndex(0)
        eq = window.equilibrium
        print("FILLED", eq.eqdsk_edit.text(), eq.density_edit.text(),
              eq.cocos_spin.value(), eq._loading)
    ''')
    assert "FILLED /nowhere/shot.geqdsk /nowhere/ne.dat 7 False" in out


# ------------------------------------------------------ the compute process

def test_the_window_stays_responsive_through_a_solve():
    """Regression: a solve on a thread starved the GUI until the desktop
    declared it hung. Now it runs in another process; the GUI thread must
    keep servicing a 50 ms timer throughout."""
    pytest.importorskip("slepc4py")
    out = run_in_offscreen_qt('''
        ticks = []
        timer = QtCore.QTimer()
        timer.setInterval(50)
        timer.timeout.connect(lambda: ticks.append(time.perf_counter()))
        timer.start()
        load_itpa()
        solver = window.solver
        solver.set_state(SolverState(ntor=6, mmax=20, max_dm=5, regularizer=1e-5,
                                     gamma_s=5 / 3, species="H", f_low_khz=60.0,
                                     f_high_khz=76.0, window_width_khz=4.0, nev=8,
                                     tol=1e-8, maxit=200, spans=200, workers=2,
                                     threads_per_worker=1))
        started = time.perf_counter()
        solver.solve()
        wait_until(lambda: solver.result is not None and not solver._solving)
        during = [t for t in ticks if t >= started]
        print("SOLVED", solver.result.mode_count > 0)
        print("WORST_STALL", max(b - a for a, b in zip(during, during[1:])))
    ''')
    assert "SOLVED True" in out
    stall = float(out.split("WORST_STALL")[1].split()[0])
    assert stall < 1.0, stall


def test_stop_ends_a_solve_in_the_middle():
    pytest.importorskip("slepc4py")
    out = run_in_offscreen_qt('''
        load_itpa()
        solver = window.solver
        solver.set_state(SolverState(ntor=6, mmax=20, max_dm=5, regularizer=1e-5,
                                     gamma_s=5 / 3, species="H", f_low_khz=50.0,
                                     f_high_khz=90.0, window_width_khz=1.0, nev=8,
                                     tol=1e-8, maxit=200, spans=200, workers=1,
                                     threads_per_worker=1))
        solver.solve()
        QtCore.QTimer.singleShot(3000, solver.stop)
        wait_until(lambda: not solver._solving)
        statuses = [w.status for w in solver.result.windows]
        print("SKIPPED", statuses.count("skipped") > 0, "TOTAL", len(statuses))
    ''')
    assert "SKIPPED True TOTAL 40" in out
