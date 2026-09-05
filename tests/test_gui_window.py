"""The Qt window itself, driven offscreen.

Run in a subprocess with QT_QPA_PLATFORM=offscreen. That is deliberate: setting
that variable at module scope would apply to the whole pytest session, which is
the mistake that had NUMBA_DISABLE_JIT silently disabling the compiled kernels
for every test that ran after it.

These exercise the parts that have logic -- the frequency cursor, the family
markers, session round-trips -- against an injected result, so no equilibrium
is loaded and nothing here takes longer than a second.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("PySide6")


def run_in_offscreen_qt(body: str) -> str:
    """Execute ``body`` in a subprocess with an offscreen Qt platform."""
    script = textwrap.dedent('''
        import numpy as np
        from PySide6 import QtWidgets
        from supernova.gui.continuum import ContinuumWindow
        from supernova.gui._model import ContinuumResult
        from supernova.gui._state import ContinuumState

        def make_result(**overrides):
            psi = np.linspace(0.0, 1.0, 101)
            khz = np.column_stack([50.0 + 100.0 * psi, 200.0 - 100.0 * psi])
            fields = dict(psi_n=psi, frequency_khz=khz, which="full", ntor=2,
                          gamma_s=5 / 3, q_min=1.0, q_max=4.0,
                          solve_seconds=0.1, reloaded_equilibrium=False,
                          rebuilt_continuum=False)
            fields.update(overrides)
            return ContinuumResult(**fields)

        app = QtWidgets.QApplication([])
        window = ContinuumWindow()
    ''') + textwrap.dedent(body)

    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        env=env, timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def test_the_window_opens_without_an_equilibrium_and_says_what_it_needs():
    out = run_in_offscreen_qt('''
        print("TITLE", window.windowTitle())
        print("STATUS", window.status.currentMessage())
        print("EXPORT_ENABLED", window.export_action.isEnabled())
    ''')
    assert "Continuum" in out
    assert "equilibrium" in out.lower()
    assert "EXPORT_ENABLED False" in out


def test_moving_the_cursor_redraws_the_cuts_without_re_solving():
    out = run_in_offscreen_qt('''
        window.result = make_result()
        window._draw(window.result)
        window._set_cursor(100.0)
        print("CUTS_TEXT", window.cuts_label.text().replace("\\n", " "))
        print("STATE_CURSOR", window.state.cursor_khz)
        print("SPIN", window.cursor_spin.value())
    ''')
    assert "STATE_CURSOR 100.0" in out
    assert "SPIN 100.0" in out
    assert "0.500" in out and "1.000" in out      # both branches cross


def test_a_gap_frequency_reports_no_crossing():
    out = run_in_offscreen_qt('''
        window.result = make_result()
        window._draw(window.result)
        window._set_cursor(10.0)
        print("TEXT", window.cuts_label.text().replace("\\n", " "))
    ''')
    assert "no continuum crossing" in out


def test_the_cursor_cannot_be_dragged_negative():
    out = run_in_offscreen_qt('''
        window.result = make_result()
        window._draw(window.result)
        window._set_cursor(-42.0)
        print("CURSOR", window.state.cursor_khz)
    ''')
    assert "CURSOR 0.0" in out


def test_family_codes_draw_one_labelled_series_per_family():
    out = run_in_offscreen_qt('''
        codes = np.zeros((101, 2), dtype=np.int8)
        codes[:, 1] = 1                      # second branch is sound
        codes[:20, 0] = 2                    # part of the first is mixed
        window.result = make_result(family_code=codes)
        window._draw(window.result)
        labels = sorted(window.axes.get_legend_handles_labels()[1])
        print("LABELS", ",".join(labels))
    ''')
    assert "Alfvenic" in out and "sound" in out and "mixed" in out


def test_selecting_a_pure_branch_disables_and_clears_classification():
    out = run_in_offscreen_qt('''
        # Start on the full spectrum, where classification is meaningful...
        window.which_combo.setCurrentIndex(window.which_combo.findData("full"))
        window.classify_check.setChecked(True)
        print("ON_FULL_ENABLED", window.classify_check.isEnabled())
        print("ON_FULL_CHECKED", window.classify_check.isChecked())
        # ...then move to a pure branch, which has one family by construction.
        window.which_combo.setCurrentIndex(
            window.which_combo.findData("slow_sound"))
        print("ENABLED", window.classify_check.isEnabled())
        print("CHECKED", window.classify_check.isChecked())
    ''')
    assert "ON_FULL_ENABLED True" in out
    assert "ON_FULL_CHECKED True" in out
    assert "ENABLED False" in out
    assert "CHECKED False" in out


def test_the_branch_combo_offers_the_three_readable_names():
    out = run_in_offscreen_qt('''
        combo = window.which_combo
        pairs = [(combo.itemData(i), combo.itemText(i))
                 for i in range(combo.count())]
        print("PAIRS", pairs)
    ''')
    assert "slow sound only" in out
    assert "no sound (Alfven only)" in out
    assert "full sound-Alfven" in out


def test_a_session_survives_a_round_trip_through_the_window(tmp_path):
    session = tmp_path / "session.json"
    out = run_in_offscreen_qt(f'''
        window.ntor_spin.setValue(7)
        window.npsi_spin.setValue(512)
        window.cursor_spin.setValue(88.5)
        window._read_widgets_into_state()
        window.state.to_json(r"{session}")

        fresh = ContinuumWindow(ContinuumState.from_json(r"{session}"))
        print("NTOR", fresh.ntor_spin.value())
        print("NPSI", fresh.npsi_spin.value())
        print("CURSOR", fresh.cursor_spin.value())
    ''')
    assert "NTOR 7" in out
    assert "NPSI 512" in out
    assert "CURSOR 88.5" in out


def test_redrawing_with_a_mode_line_up_does_not_crash():
    """Regression: change the branch while the frequency cursor is shown.

    _draw begins with Axes.clear(), which detaches every artist. The cursor
    artists were still referenced afterwards, and matplotlib refuses to remove
    an artist twice, so the redraw raised NotImplementedError inside the solve
    callback and left the plot broken.
    """
    out = run_in_offscreen_qt('''
        window.result = make_result()
        window._draw(window.result)
        window._set_cursor(118.0)
        print("CUTS_BEFORE", len(window._cut_artists))

        # what the user did: switch to another branch set, which re-solves
        # and redraws from scratch while the mode line is up
        window.which_combo.setCurrentIndex(
            window.which_combo.findData("alfven"))
        window.result = make_result(which="alfven")
        window._draw(window.result)
        print("SURVIVED_REDRAW", window.state.cursor_khz)

        # and again, twice more, to be sure nothing accumulates
        for _ in range(2):
            window._draw(window.result)
        print("CUTS_AFTER", len(window._cut_artists))
        print("OK")
    ''')
    assert "CUTS_BEFORE 2" in out          # the line plus its crossing markers
    assert "SURVIVED_REDRAW 118.0" in out
    assert "CUTS_AFTER 2" in out           # redrawn, not accumulated
    assert "OK" in out


def test_the_placeholder_can_follow_a_drawn_cursor():
    """Clearing back to the empty state must not trip over stale artists."""
    out = run_in_offscreen_qt('''
        window.result = make_result()
        window._draw(window.result)
        window._set_cursor(118.0)
        window._draw_placeholder()
        print("ARTISTS", len(window._cut_artists))
        print("OK")
    ''')
    assert "ARTISTS 0" in out
    assert "OK" in out
