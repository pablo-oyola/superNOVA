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
                EquilibriumState(source="itpa", itpa_ns=48, itpa_ntheta=48, species="H"))
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


def test_the_window_has_three_tabs():
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


def test_legend_picks_hide_lines_and_the_continuum_view_puts_zero_on_the_mode():
    pytest.importorskip("slepc4py")
    out = run_in_offscreen_qt('''
        from matplotlib.backend_bases import MouseEvent
        load_itpa()
        solver = window.solver
        solver.set_state(SolverState(ntor=6, mmax=20, max_dm=5, regularizer=1e-5,
                                     gamma_s=0.0, species="H", f_low_khz=68.0,
                                     f_high_khz=72.0, window_width_khz=4.0, nev=4,
                                     tol=1e-7, maxit=80, spans=100))
        solver.solve()
        wait_until(lambda: solver.result is not None and not solver._solving)
        print("DEFAULT_VIEW", solver.state.mode_view)
        solver.mode_view_combo.setCurrentIndex(solver.mode_view_combo.findData("harmonics"))
        canvas = solver.mode_canvas

        def click(label, dblclick=False):
            canvas.draw()
            legend = canvas.figure.axes[-1].get_legend()
            text = next(t for t in legend.get_texts() if t.get_text() == label)
            box = text.get_window_extent()
            x, y = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
            MouseEvent("button_press_event", canvas, x, y, button=1,
                       dblclick=dblclick)._process()
            app.processEvents()

        def shown():
            return sorted(line.get_label() for ax in canvas.figure.axes
                          for line in ax.get_lines()
                          if line.get_visible() and not line.get_label().startswith("_"))

        everything = shown()
        click("m = 10")
        print("HIDDEN", "m = 10" not in shown(), len(shown()) == len(everything) - 1)
        click("m = 10")
        print("BACK", shown() == everything)
        click("m = 11"); click("m = 11", dblclick=True)
        print("ALONE", shown())
        click("m = 11"); click("m = 11", dblclick=True)
        print("ALL_AGAIN", shown() == everything)

        click("m = 10")
        solver.mode_view_combo.setCurrentIndex(solver.mode_view_combo.findData("continuum"))
        print("VIEW", solver.state.mode_view, solver._result_is_stale())
        ax, twin = canvas.figure.axes[0], canvas.figure.axes[-1]
        canvas.draw()
        f = solver.result.mode_rows()[solver.selected]["frequency_khz"]
        (low, high), (bottom, top) = ax.get_ylim(), twin.get_ylim()
        print("ZERO_ON_F", abs((f - low) / (high - low) + bottom / (top - bottom)) < 1e-9)
        print("STILL_HIDDEN", "m = 10" not in shown(), "continuum" in shown())

        # many harmonics: the strongest are named, the rest share one entry
        import supernova.gui.solver as module
        module.LEGEND_HARMONICS = 4
        solver.mode_toggles.hidden.clear()
        solver._draw_mode()
        legend = canvas.figure.axes[-1].get_legend()
        names = [text.get_text() for text in legend.get_texts()]
        print("NAMED", sum(name.startswith("m = ") for name in names),
              module.WEAKER_HARMONICS in names)
        click(module.WEAKER_HARMONICS)
        twin = canvas.figure.axes[-1]
        print("WEAKER_HIDDEN", sum(line.get_visible() for line in twin.get_lines()
                                   if line.get_label() == "_nolegend_"),
              len(shown()))
    ''')
    assert "HIDDEN True True" in out
    assert "BACK True" in out
    assert "ALONE ['m = 11']" in out
    assert "ALL_AGAIN True" in out
    assert "VIEW continuum False" in out           # a view change is not a new solve
    assert "ZERO_ON_F True" in out
    assert "STILL_HIDDEN True True" in out
    assert "DEFAULT_VIEW continuum" in out
    assert "NAMED 4 True" in out
    assert "WEAKER_HIDDEN 0 6" in out              # 4 named + continuum + mode frequency


def test_a_saved_session_brings_the_solve_back_without_solving_again(tmp_path):
    """Save after a solve; a fresh window reloads the equilibrium and takes the
    pass back onto it, with the same mode picked -- and solves nothing."""
    pytest.importorskip("slepc4py")
    path = tmp_path / "session.h5"
    out = run_in_offscreen_qt(f'''
        load_itpa()
        solver = window.solver
        solver.set_state(SolverState(ntor=6, mmax=20, max_dm=5, regularizer=1e-5,
                                     gamma_s=0.0, species="H", f_low_khz=68.0,
                                     f_high_khz=72.0, window_width_khz=2.0, nev=4,
                                     tol=1e-7, maxit=80, spans=100))
        solver.solve()
        wait_until(lambda: solver.result is not None and not solver._solving)
        solver.selected = solver.result.mode_count - 1
        window.write_session(r"{path}")
        before = solver.result

        fresh = SupernovaWindow()
        fresh.compute.submit = lambda kind, *a, _submit=fresh.compute.submit, **k: (
            print("SUBMITTED", kind) or _submit(kind, *a, **k))
        fresh.load_session(r"{path}")
        wait_until(lambda: fresh.solver.result is not None)
        after = fresh.solver.result
        print("SAME_MODES", list(after.frequency_khz) == list(before.frequency_khz))
        print("SAME_WINDOWS", after.windows == before.windows)
        print("ON_THE_NEW_LOAD", after.handle == fresh.solver.view.handle,
              fresh.solver._result_is_stale())
        print("PICKED", fresh.solver.selected == before.mode_count - 1)
        print("TABLE", fresh.solver.mode_table.rowCount() == before.mode_count)
        print("LOG", "session restored" in fresh.solver.log.toPlainText())
        fresh.close()
    ''')
    assert "SAME_MODES True" in out
    assert "SAME_WINDOWS True" in out
    assert "ON_THE_NEW_LOAD True False" in out
    assert "PICKED True" in out
    assert "TABLE True" in out
    assert "LOG True" in out
    assert "SUBMITTED solve" not in out
    assert "SUBMITTED load_equilibrium" in out


def test_the_path_tolerance_relabels_a_path_pass_without_solving_again():
    """The global path runs on its own; its thresholds are applied after the solve."""
    pytest.importorskip("slepc4py")
    out = run_in_offscreen_qt('''
        load_itpa()
        solver = window.solver
        solver.set_state(SolverState(ntor=6, mmax=20, max_dm=5, regularizer=1e-5,
                                     gamma_s=0.0, species="H", f_low_khz=68.0,
                                     f_high_khz=72.0, window_width_khz=4.0, nev=6,
                                     tol=1e-7, maxit=80, spans=100, mode_filter="path"))
        print("MODELOCATE_FORM_HIDDEN", solver.locate_form.isHidden())
        solver.solve()
        wait_until(lambda: solver.result is not None and not solver._solving)
        result = solver.result
        print("NO_MODELOCATE_LABELS", not set(result.labels) & {"spike", "core", "edge"})
        headers = [solver.mode_table.horizontalHeaderItem(i).text()
                   for i in range(solver.mode_table.columnCount())]
        print("PATH_COLUMNS", "path drift" in headers and "path correlation" in headers)
        windows = [solver.window_table.horizontalHeaderItem(i).text()
                   for i in range(solver.window_table.columnCount())]
        print("DROPPED_COLUMN", "dropped" in windows)
        before = list(result.labels)
        solver.compute.submit = lambda kind, *a, **k: print("SUBMITTED", kind)
        solver.path_tolerance_edit.setText("0.01")          # typed, as a user would
        solver.path_tolerance_edit.editingFinished.emit()
        after = list(solver.result.labels)
        print("RELABELLED", after != before and after.count("gap") > before.count("gap"))
        print("STALE", solver._result_is_stale())
        import numpy as np
        print("SAME_DRIFT", np.array_equal(solver.result.located["path_drift"].values,
                                           result.located["path_drift"].values,
                                           equal_nan=True))
    ''')
    assert "MODELOCATE_FORM_HIDDEN True" in out
    assert "NO_MODELOCATE_LABELS True" in out
    assert "PATH_COLUMNS True" in out
    assert "DROPPED_COLUMN True" in out
    assert "RELABELLED True" in out
    assert "STALE False" in out
    assert "SAME_DRIFT True" in out
    assert "SUBMITTED solve" not in out


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

def test_a_shared_load_reaches_the_continuum_and_solver_tabs():
    out = run_in_offscreen_qt('''
        from supernova.gui._equilibrium import EquilibriumModel
        view = EquilibriumModel().load(
            EquilibriumState(source="itpa", itpa_ns=16, itpa_ntheta=16))
        window._on_equilibrium(view, True)
        print("SAME", window.continuum.view is view, window.solver.view is view)
        print("LABEL", window.continuum.equilibrium_label.text())
    ''')
    assert "SAME True True" in out
    assert "ITPA" in out


def test_the_continuum_tab_solves_on_the_loaded_equilibrium():
    """Even the analytic ITPA case, which the tab could not load on its own."""
    out = run_in_offscreen_qt('''
        load_itpa()
        c = window.continuum
        c.ntor_spin.setValue(6)
        wait_until(lambda: c.result is not None and c.result.ntor == 6 and not c._solving)
        print("SAME", c.view is window.equilibrium.view)
        print("BRANCHES", c.result.n_branches > 0)
        print("RADII", round(float(c.result.psi_n[0]), 4), round(float(c.result.psi_n[-1]), 4))
    ''')
    assert "SAME True" in out
    assert "BRANCHES True" in out


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
        print("KEPT", solver.view is first, eq.view is refined,
              window.continuum.view is first)
        print("SUMMARY_SAYS", "Kept to this tab" in eq.summary_label.text())
        again = EquilibriumModel().load(
            EquilibriumState(source="itpa", itpa_ns=20, itpa_ntheta=20))
        eq._comparison_pending, eq._share_refined = True, True
        eq._on_loaded(again)
        print("PASSED", solver.view is again)
    ''')
    assert "KEPT True True True" in out
    assert "SUMMARY_SAYS True" in out
    assert "PASSED True" in out


def test_an_old_continuum_session_fills_the_equilibrium_form(tmp_path):
    """Its continuum settings go to the Continuum tab, its file and density to
    the Equilibrium tab's form, unloaded."""
    import json

    path = tmp_path / "old.json"
    path.write_text(json.dumps({"eqdsk": "/nowhere/shot.geqdsk", "cocos_in": 7,
                                "density_file": "/nowhere/ne.dat", "ntor": 4}))
    out = run_in_offscreen_qt(f'''
        window.load_session(r"{path}")
        eq = window.equilibrium
        print("FILLED", eq.eqdsk_edit.text(), eq.density_edit.text(),
              eq.cocos_spin.value(), eq._loading)
        print("NTOR", window.continuum.ntor_spin.value())
        print("TAB", window.tabs.tabText(window.tabs.currentIndex()))
    ''')
    assert "FILLED /nowhere/shot.geqdsk /nowhere/ne.dat 7 False" in out
    assert "NTOR 4" in out
    assert "TAB Equilibrium" in out


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
