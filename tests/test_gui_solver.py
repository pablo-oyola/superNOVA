"""The solver tab's state and solve, without Qt.

Most of this needs no equilibrium: the state, its validation, the window
layout and the PETSc options are plain data. Then the tab's simple pass, on
two equilibria:

* the analytic ITPA case, for the mechanics: it must find the roots
  test_itpa_acceptance pins -- a regression pin, not the benchmark TAE -- and
  survive a window in which SLEPc converges nothing;
* the corrected ITPA equilibrium, for the physics: between walls at rho_pol
  0.05 and 0.95 it must find the benchmark's global m = 10/11 TAE at
  65.766 kHz (the solver5 reference run). That file lives outside the
  repository, so the test skips without it.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading

import numpy as np
import pytest

from supernova.gui._session import read_session, write_session
from supernova.gui._solver import (
    SolverState,
    locate_family,
    options_from_overrides,
    parse_petsc_options,
)

# test_itpa_acceptance's regression pin for the analytic case
ACCEPTANCE_PIN_KHZ = (69.95411566599043, 70.14986464493438,
                      70.33584493960305, 70.65894649445084)
# the corrected-equilibrium ITPA TAE, from the solver5 reference run
ITPA_TAE_KHZ = 65.76578119821716
CORRECTED_ITPA = (Path(__file__).resolve().parents[1] / "no_sync_ref" / "ITPA"
                  / "Nikolai" / "itpa_tokamaker_nikolai_corrected.geqdsk")


# --------------------------------------------------------------- validation

def test_the_defaults_are_complete():
    assert SolverState().validate() == []


@pytest.mark.parametrize("overrides, expected", [
    ({"ntor": 0}, "n = 0"),
    ({"f_low_khz": 150.0, "f_high_khz": 50.0}, "frequency band"),
    ({"window_width_khz": 0.0}, "window width"),
    ({"window_width_khz": 1.0e-4}, "too many"),
    ({"window_overlap": 1.0}, "overlap"),
    ({"nev": 0}, "at least one root"),
    ({"spans": 1}, "spans"),
    ({"rho_min": 0.9, "rho_max": 0.1}, "walls"),
    ({"factorization": "cholesky"}, "factorization"),
    ({"regularizer": -1.0}, "regularizer"),
    ({"modelocate": {"lowedge": 0.1}}, "lowedge"),
    ({"workers": 0}, "worker processes"),
    ({"threads_per_worker": 0}, "threads per worker"),
])
def test_validation_catches_each_impossible_setting(overrides, expected):
    problems = SolverState(**overrides).validate()
    assert any(expected in p for p in problems), problems


# ------------------------------------------------------------------ windows

def test_the_windows_are_the_ones_band_cuts():
    """Width is the size of each subrange; the last one takes what is left."""
    state = SolverState(f_low_khz=50.0, f_high_khz=150.0, window_width_khz=3.0)
    windows = state.windows_khz()
    assert len(windows) == state.window_count() == 34
    assert windows[0] == (50.0, 53.0)
    assert windows[-1] == pytest.approx((149.0, 150.0))


def test_overlapping_windows_step_by_the_unshared_part():
    state = SolverState(f_low_khz=60.0, f_high_khz=76.0, window_width_khz=4.0,
                        window_overlap=0.5)
    windows = state.windows_khz()
    assert windows[:2] == [(60.0, 64.0), (62.0, 66.0)]
    assert len(windows) == state.window_count() == 8


# ------------------------------------------------------------------ options

def test_mumps_gets_its_workspace_headroom_and_extra_options_win():
    state = SolverState(mumps_icntl_14=300,
                        extra_options="st_ksp_type gmres\n-st_pc_type=ilu  # why not")
    options = state.petsc_options().as_dict()
    assert options["st_mat_mumps_icntl_14"] == "300"
    assert options["st_ksp_type"] == "gmres"
    assert options["st_pc_type"] == "ilu"


def test_the_qr_factorizations_carry_no_mumps_option():
    options = SolverState(factorization="spqr").petsc_options().as_dict()
    assert "st_mat_mumps_icntl_14" not in options


def test_petsc_option_lines_skip_comments_and_blanks():
    assert parse_petsc_options("\n# a comment\n  -a 1\nb=2\n") == {"a": "1", "b": "2"}


def test_modelocate_overrides_reach_the_thresholds():
    thresholds = SolverState(modelocate={"core_edge": 0.1}).locate_thresholds()
    assert thresholds.core_edge == 0.1
    assert thresholds.boundary_edge == 0.96          # untouched default


def test_overrides_turn_json_lists_back_into_tuples():
    import supernova as sn

    options = options_from_overrides(sn.AdaptiveDiscoveryOptions, {"nev_passes": [2, 4]})
    assert options.nev_passes == (2, 4)


def test_a_combined_modelocate_label_is_drawn_as_its_first_flag():
    assert locate_family("spike|core") == "spike"
    assert locate_family("gap") == "gap"


# ------------------------------------------------------------------ keys

def test_the_solve_key_ignores_what_only_changes_the_drawing():
    base = SolverState()
    for field, value in (("field", "p1"), ("part", "abs"),
                         ("continuum_preview", False), ("gap_only", True),
                         ("workers", 7), ("threads_per_worker", 1)):
        changed = SolverState.from_dict({**base.to_dict(), field: value})
        assert changed.solve_key() == base.solve_key(), field


def test_the_solve_key_sees_the_filter_thresholds():
    assert (SolverState(modelocate={"spike_level": 0.5}).solve_key()
            != SolverState().solve_key())


# ------------------------------------------------------------------ sessions

def test_a_three_tab_session_round_trips(tmp_path):
    from supernova.gui._equilibrium import EquilibriumState
    from supernova.gui._state import ContinuumState

    path = tmp_path / "session.json"
    states = dict(equilibrium=EquilibriumState(source="itpa", refine_share=False),
                  continuum=ContinuumState(ntor=5),
                  solver=SolverState(nev=3, modelocate={"core_edge": 0.1}))
    write_session(path, **states)
    assert read_session(path) == states


def test_a_continuum_only_session_from_the_old_window_still_loads(tmp_path):
    from supernova.gui._state import ContinuumState

    path = tmp_path / "old.json"
    ContinuumState(ntor=7, npsi=300).to_json(path)
    loaded = read_session(path)
    assert set(loaded) == {"continuum"}
    assert loaded["continuum"].ntor == 7


def test_an_unknown_section_is_refused_by_name(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"solver": {}, "plasma": {}}))
    with pytest.raises(ValueError, match="plasma"):
        read_session(path)


# ------------------------------------------------------------------ the solve

@pytest.fixture(scope="module")
def itpa_pass():
    """The ITPA band in four windows; 64-68 kHz converges nothing at maxit 80."""
    pytest.importorskip("slepc4py")
    from supernova.gui._equilibrium import EquilibriumModel, EquilibriumState
    from supernova.gui._solver import SolverModel

    view = EquilibriumModel().load(EquilibriumState(source="itpa"))
    state = SolverState(ntor=6, mmax=20, max_dm=5, regularizer=1.0e-5,
                        gamma_s=0.0, species="H", f_low_khz=60.0, f_high_khz=76.0,
                        window_width_khz=4.0, nev=4, tol=1.0e-7, maxit=80, spans=100)
    model = SolverModel()
    return model, view, state, model.solve(view.equilibrium, state, title=view.title)


def test_the_simple_pass_finds_the_acceptance_pin(itpa_pass):
    *_, result = itpa_pass
    for frequency in ACCEPTANCE_PIN_KHZ:
        error = np.min(np.abs(result.frequency_khz - frequency)) / frequency
        assert error < 1.0e-8, (frequency, result.frequency_khz)


def test_a_window_that_converges_nothing_is_a_row_not_the_end(itpa_pass):
    *_, result = itpa_pass
    statuses = [window.status for window in result.windows]
    assert len(statuses) == 4
    assert statuses[1] == "unconverged"
    assert (result.windows[1].nconv, result.windows[1].reason) == (0, "-1")
    assert result.windows[2].kept >= 4                # the sweep went on


def test_every_mode_gets_a_modelocate_verdict(itpa_pass):
    *_, result = itpa_pass
    assert result.labels.size == result.mode_count
    flags = {flag for label in result.labels for flag in label.split("|")}
    assert flags <= {"gap", "continuum", "spike", "core", "edge", "ideal"}


def test_modes_come_sorted_by_frequency_without_duplicates(itpa_pass):
    *_, result = itpa_pass
    assert np.all(np.diff(result.frequency_khz) > 0.0)


def test_stopping_skips_the_remaining_windows(itpa_pass):
    model, view, state, _ = itpa_pass
    cancel = threading.Event()
    cancel.set()
    result = model.solve(view.equilibrium, state, cancel=cancel)
    assert {window.status for window in result.windows} == {"skipped"}
    assert result.mode_count == 0


def test_radii_are_physical_psi_n(itpa_pass):
    """The analytic case's Boozer domain is (0, 1), so x and psi_N agree."""
    *_, result = itpa_pass
    assert result.walls == pytest.approx((0.0, 1.0))
    assert np.all((result.center > 0.0) & (result.center < 1.0))
    assert result.psi_core == pytest.approx(0.05 ** 2)


def test_walls_map_to_solver_x_and_clip_to_the_boozer_domain(itpa_pass):
    from supernova.gui._solver import radial_domain

    _, view, *_ = itpa_pass
    low, high = radial_domain(view.equilibrium, 0.05, 0.95)
    assert (low, high) == pytest.approx((0.05 ** 2, 0.95 ** 2), rel=1.0e-3)
    assert radial_domain(view.equilibrium, 0.0, 1.0) == (0.0, 1.0)


def test_the_export_carries_the_verdicts_and_the_windows(itpa_pass, tmp_path):
    import xarray as xr

    from supernova.gui._solver import SolverModel

    *_, result = itpa_pass
    path = SolverModel.export(result, tmp_path / "modes.nc")
    with xr.open_dataset(path) as dataset:
        assert list(dataset["mode_label"].values) == list(result.labels)
        assert list(dataset["window_status"].values) == [
            window.status for window in result.windows]
        assert json.loads(dataset.attrs["gui_settings"])["spans"] == 100


@pytest.mark.skipif(not CORRECTED_ITPA.is_file(),
                    reason="the corrected ITPA equilibrium is not in this checkout")
def test_the_corrected_itpa_equilibrium_gives_the_benchmark_tae():
    """The ITPA main result, through the tab's own load and solve."""
    pytest.importorskip("slepc4py")
    from supernova.gui._equilibrium import EquilibriumModel, EquilibriumState
    from supernova.gui._solver import SolverModel

    view = EquilibriumModel().load(EquilibriumState(
        eqdsk=str(CORRECTED_ITPA), cocos_in=7, density_uniform_m3=2.0e19,
        boozer_min=0.025, boozer_max=0.96, boozer_npsi=256, boozer_ntheta=256,
        m_max=20))
    assert not view.density_is_placeholder
    state = SolverState(ntor=6, mmax=20, max_dm=3, regularizer=1.0e-5,
                        gamma_s=5.0 / 3.0, species="H", f_low_khz=60.0,
                        f_high_khz=76.0, window_width_khz=16.0, nev=20, tol=1.0e-7,
                        maxit=80, spans=100, rho_min=0.05, rho_max=0.95)
    result = SolverModel().solve(view.equilibrium, state, title=view.title)

    tae = int(np.argmin(np.abs(result.frequency_khz - ITPA_TAE_KHZ)))
    assert abs(result.frequency_khz[tae] - ITPA_TAE_KHZ) / ITPA_TAE_KHZ < 1.0e-3
    assert result.dominant_m[tae] in (10, 11)
    assert result.walls == pytest.approx((0.05 ** 2, 0.95 ** 2), rel=1.0e-3)
    assert result.labels[tae] == "gap"


# ------------------------------------------------------------- worker pool

def test_a_pool_of_workers_finds_what_one_process_finds(itpa_pass):
    """Same pencil, windows spread over two processes: same roots."""
    from supernova.gui._workers import WindowPool

    model, view, state, serial = itpa_pass
    pool = WindowPool(workers=2, threads=1)
    try:
        pooled = model.solve(view.equilibrium, state, pool=pool)
    finally:
        pool.close()
    np.testing.assert_allclose(pooled.frequency_khz, serial.frequency_khz, rtol=1.0e-8)
    assert [w.status for w in pooled.windows] == [w.status for w in serial.windows]
    assert list(pooled.labels) == list(serial.labels)


def test_stopping_a_pool_terminates_its_busy_workers(itpa_pass):
    from supernova.gui._workers import WindowPool

    model, view, state, _ = itpa_pass
    cancel = threading.Event()
    cancel.set()
    pool = WindowPool(workers=2, threads=1)
    try:
        result = model.solve(view.equilibrium, state, pool=pool, cancel=cancel)
        assert {window.status for window in result.windows} == {"skipped"}
        assert not any(worker.process.is_alive() for worker in pool._pool)
    finally:
        pool.close()
