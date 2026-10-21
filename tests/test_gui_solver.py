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
    for field, value in (("field", "p1"), ("part", "abs"), ("mode_view", "harmonics"),
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

    path = tmp_path / "session.h5"
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


def _small_result():
    """A pass in miniature: three modes from two of three windows."""
    import xarray as xr

    from supernova.gui._solver import SolverResult, WindowRow

    rng = np.random.default_rng(0)
    found_by = np.array([0, 1, 1])
    omega = rng.normal(size=3) + 1j * rng.normal(size=3)
    dataset = xr.Dataset(
        {"omega": ("eigenpair", omega),
         "eigenvector": (("eigenpair", "dof"),
                         rng.normal(size=(3, 6)) + 1j * rng.normal(size=(3, 6)))},
        coords={"eigenpair": np.arange(3), "window": ("eigenpair", found_by),
                "dof": np.arange(6)},
        attrs={"ntor": 6, "apply_bcs": True, "sound_model": "ideal"})
    located = xr.Dataset(
        {"mode_label": ("eigenpair", np.array(["gap", "continuum", "spike|core"])),
         "envelope": (("eigenpair", "x"), rng.random((3, 5)))},
        coords={"eigenpair": np.arange(3), "x": np.linspace(0.0, 1.0, 5)},
        attrs={"method": "modelocate", "core_edge": 0.05})
    windows = (WindowRow(0, (60.0, 68.0), 4, nconv=1, kept=1),
               WindowRow(1, (68.0, 76.0), 4, nconv=2, kept=2, reason="-1",
                         status="unconverged"),
               WindowRow(2, (76.0, 84.0), 4, status="failed", message="no luck"))
    return SolverResult(
        state=SolverState(nev=4), title="test", windows=windows, dataset=dataset,
        frequency_khz=np.array([65.0, 69.0, 70.0]), omega=omega,
        center=rng.random(3), width=rng.random(3), dominant_m=np.array([10, 11, 10]),
        backward_error=np.array([1e-12, 1e-11, np.nan]), window_of=found_by,
        located=located, crossings=(np.empty(0), np.array([0.2]), np.array([0.3, 0.5])),
        x=np.linspace(0.0, 1.0, 5), m_pol=np.array([10, 11]),
        fields={"xi_psi": rng.normal(size=(3, 5, 2)) + 1j * rng.normal(size=(3, 5, 2))},
        envelope_x=np.linspace(0.0, 1.0, 5), continuum_x=np.linspace(0.0, 1.0, 4),
        continuum_khz=np.array([[60.0, np.nan]] * 4), psi_core=0.0025, psi_edge=0.92,
        walls=(0.0025, 0.9025), seconds=1.5, equilibrium=None, handle=7)


def test_a_session_keeps_the_pass_window_by_window(tmp_path):
    """Everything a pass holds comes back as it was, the eigenpairs filed by window."""
    import dataclasses

    import h5py
    import xarray as xr

    from supernova.gui._model import ContinuumResult
    from supernova.gui._solver import RefineResult

    result = _small_result()
    refinement = RefineResult(
        state=SolverState(refine_levels=2), title="test", modes=(0,),
        rows=({"mode": 0, "level": 0, "h_mm": 4.0, "frequency_khz": float("nan")},
              {"mode": 0, "level": 1, "h_mm": 2.0, "frequency_khz": 65.1}),
        x=np.linspace(0.0, 1.0, 5), envelopes={0: np.ones((1, 5))},
        path={0: {"label": "gap", "alpha": 0.01, "drift": 1e-8, "correlation": 0.999}},
        seconds=3.0, handle=7)
    continuum = ContinuumResult(
        psi_n=np.linspace(0.0, 1.0, 4), frequency_khz=np.ones((4, 2)), which="alfven",
        ntor=6, gamma_s=0.0, q_min=1.0, q_max=2.0, solve_seconds=0.01,
        reloaded_equilibrium=True, rebuilt_continuum=False)
    key = ("itpa", 1.0, 10.0, 3.0, 1.71, 1.87, 1, 48, 48, 20)
    path = tmp_path / "session.h5"
    write_session(path, solver=SolverState(nev=4), results={
        "continuum": {"result": continuum, "equilibrium_key": key},
        "solver": {"result": result, "refinement": refinement, "selected": 2,
                   "log": "solved\n" * 20000, "equilibrium_key": key}})

    with h5py.File(path, "r") as h5:
        windows = h5["solver/result/windows"]
        assert list(windows) == ["w000", "w001", "w002"]
        assert windows["w001/eigenpairs/omega"].shape == (2,)
        assert "eigenpairs" not in windows["w002"]       # it found nothing

    loaded = read_session(path)
    assert loaded["solver"] == SolverState(nev=4)
    solver = loaded["results"]["solver"]
    assert solver["equilibrium_key"] == key and solver["selected"] == 2
    assert solver["log"] == "solved\n" * 20000            # longer than an attribute holds
    back = solver["result"]
    xr.testing.assert_identical(back.dataset, result.dataset)
    xr.testing.assert_identical(back.located, result.located)
    assert back.windows == result.windows and back.state == result.state
    assert back.handle is None                    # the compute process names it anew
    for item in dataclasses.fields(result):
        mine, theirs = getattr(result, item.name), getattr(back, item.name)
        if isinstance(mine, np.ndarray):
            np.testing.assert_array_equal(theirs, mine)
            assert theirs.dtype == mine.dtype, item.name
    for mine, theirs in zip(result.crossings, back.crossings, strict=True):
        np.testing.assert_array_equal(theirs, mine)
    np.testing.assert_array_equal(back.fields["xi_psi"], result.fields["xi_psi"])
    assert back.labels[2] == "spike|core"

    refined = solver["refinement"]
    assert refined.path == refinement.path and refined.modes == (0,)
    assert np.isnan(refined.rows[0]["frequency_khz"]) and refined.rows[1] == refinement.rows[1]
    np.testing.assert_array_equal(refined.envelopes[0], refinement.envelopes[0])
    continuum_back = loaded["results"]["continuum"]["result"]
    assert continuum_back.which == "alfven" and continuum_back.reloaded_equilibrium is True
    np.testing.assert_array_equal(continuum_back.frequency_khz, continuum.frequency_khz)


def test_path_verdicts_follow_the_thresholds_without_a_solve():
    """Every mode keeps its drift and correlation; moving a threshold relabels."""
    import dataclasses

    import xarray as xr

    from supernova.gui._solver import SolverResult

    small = _small_result()
    located = small.located.copy()
    located["path_drift"] = ("eigenpair", np.array([2.0e-5, 6.0e-4, np.nan]))
    located["path_correlation"] = ("eigenpair", np.array([0.99, 0.95, np.nan]))
    located["path_alpha"] = ("eigenpair", np.array([0.05, 0.05, 0.05]))
    located["path_ideal"] = ("eigenpair", np.array([False, False, False]))
    located.attrs["method"] = "global_path"
    result = dataclasses.replace(small, located=located,
                                 state=SolverState(mode_filter="path"))
    assert isinstance(result, SolverResult)

    strict = result.with_path_thresholds(3.0e-4, 0.9)
    assert list(strict.labels) == ["gap", "continuum", "inconclusive"]
    loose = strict.with_path_thresholds(1.0e-3, 0.9)
    assert list(loose.labels) == ["gap", "gap", "inconclusive"]
    assert list(loose.with_path_thresholds(1.0e-3, 0.97).labels) == [
        "gap", "continuum", "inconclusive"]
    # the measurements and the solve itself are untouched
    xr.testing.assert_identical(loose.located["path_drift"], located["path_drift"])
    assert loose.state.solve_key() == result.state.solve_key()
    assert loose.state.path_tolerance == 1.0e-3
    # a MODELOCATE pass has nothing to re-judge
    assert small.with_path_thresholds(1.0, 0.0) is small


def test_one_tabs_old_json_file_opens_in_that_tab(tmp_path):
    """The tabs' own buttons used to save their bare state; those open again."""
    from supernova.gui._equilibrium import EquilibriumState

    for name, state in (("equilibrium", EquilibriumState(source="itpa", itpa_ns=40)),
                        ("solver", SolverState(nev=7))):
        path = tmp_path / f"{name}.json"
        state.to_json(path)
        assert read_session(path) == {name: state}


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
