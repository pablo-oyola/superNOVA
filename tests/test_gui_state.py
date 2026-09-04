"""The continuum window's state and caching, without Qt or physics.

The window is deliberately thin: it edits a ContinuumState and hands it to a
ContinuumModel. Both are plain objects, so what is worth testing -- validation,
session round-trips, and which expensive step gets skipped on which edit -- is
testable without a display and without loading an equilibrium.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from supernova.gui._state import BRANCHES, COORDINATES, ContinuumState


# --------------------------------------------------------------- validation

def test_an_empty_state_asks_for_the_two_files_it_needs():
    problems = ContinuumState().validate()
    assert any("equilibrium" in p.lower() for p in problems)
    assert any("density" in p.lower() for p in problems)


def test_validation_reports_missing_files_by_path(tmp_path):
    state = ContinuumState(eqdsk=str(tmp_path / "absent.eqdsk"),
                           density_file=str(tmp_path / "absent.dat"))
    problems = state.validate()
    assert any("absent.eqdsk" in p for p in problems)
    assert any("absent.dat" in p for p in problems)


def _valid(tmp_path, **overrides) -> ContinuumState:
    eqdsk = tmp_path / "eq.eqdsk"; eqdsk.write_text("x")
    density = tmp_path / "ne.dat"; density.write_text("x")
    return ContinuumState(eqdsk=str(eqdsk), density_file=str(density), **overrides)


def test_a_complete_state_has_nothing_to_report(tmp_path):
    assert _valid(tmp_path).validate() == []


@pytest.mark.parametrize("overrides, expected", [
    ({"boozer_min": 0.9, "boozer_max": 0.1}, "Boozer domain"),
    ({"mmax": 0}, "mmax"),
    ({"npsi": 4}, "Radial points"),
    ({"which": "sound"}, "Branch"),
    ({"band_low_khz": 150.0, "band_high_khz": 50.0}, "Band"),
])
def test_validation_catches_each_impossible_setting(tmp_path, overrides, expected):
    problems = _valid(tmp_path, **overrides).validate()
    assert any(expected in p for p in problems), problems


def test_a_zero_band_is_not_an_error(tmp_path):
    """Both zero means 'draw no band', which is the default."""
    assert _valid(tmp_path, band_low_khz=0.0, band_high_khz=0.0).validate() == []


# ------------------------------------------------------------------ session

def test_a_session_round_trips_through_json(tmp_path):
    state = _valid(tmp_path, ntor=7, gamma_s=0.0, species="T", mmax=64,
                   npsi=512, which="full", band_low_khz=50.0, band_high_khz=150.0)
    path = tmp_path / "session.json"
    state.to_json(path)
    assert ContinuumState.from_json(path) == state


def test_loading_a_session_with_an_unknown_key_says_which(tmp_path):
    path = tmp_path / "session.json"
    path.write_text(json.dumps({"ntor": 3, "flux_surface_count": 17}))
    with pytest.raises(ValueError, match="flux_surface_count"):
        ContinuumState.from_json(path)


# -------------------------------------------------------- what invalidates what

def test_the_equilibrium_key_ignores_continuum_settings():
    """Changing n or mmax must not trigger a Boozer transform."""
    base = ContinuumState()
    for field, value in (("ntor", 9), ("mmax", 99), ("npsi", 1024),
                         ("which", "full"), ("gamma_s", 0.0), ("species", "T"),
                         ("fmax_khz", 200.0), ("band_low_khz", 10.0)):
        changed = ContinuumState(**{**vars(base), field: value})
        assert changed.equilibrium_key() == base.equilibrium_key(), field


@pytest.mark.parametrize("field, value", [
    ("eqdsk", "/other.eqdsk"),
    ("cocos_in", 7),
    ("boozer_min", 0.01),
    ("boozer_max", 0.99),
    ("density_file", "/other.dat"),
    ("density_x", "rho_pol"),
    ("density_interpolate_in", "psi_n"),
    ("density_units", "cm^-3"),
])
def test_the_equilibrium_key_changes_when_its_inputs_do(field, value):
    base = ContinuumState()
    changed = ContinuumState(**{**vars(base), field: value})
    assert changed.equilibrium_key() != base.equilibrium_key()


def test_only_gamma_and_species_rebuild_the_continuum():
    base = ContinuumState()
    assert ContinuumState(**{**vars(base), "gamma_s": 0.0}).continuum_key() \
        != base.continuum_key()
    assert ContinuumState(**{**vars(base), "species": "H"}).continuum_key() \
        != base.continuum_key()
    assert ContinuumState(**{**vars(base), "ntor": 8}).continuum_key() \
        == base.continuum_key()


# ----------------------------------------------------------- model caching

def _alfven_registry():
    """A registry with omegaA, which normally comes from the equilibrium."""
    import unyt
    registry = unyt.UnitRegistry()
    registry.add("omegaA", 5.0e5, unyt.dimensions.rate)
    return registry


class _FakeContinuum:
    def __init__(self):
        self.solves = 0

    def solve(self, **kwargs):
        import numpy as np
        import unyt
        self.solves += 1
        npsi = np.asarray(kwargs["psi_vals"]).size
        values = np.linspace(0.1, 4.0, npsi)[:, None] ** 2
        return unyt.unyt_array(values, "omegaA**2", registry=_alfven_registry())


class _FakeCase:
    def __init__(self, counters):
        self.counters = counters
        self.continuum = _FakeContinuum()
        self.equilibrium = SimpleNamespace(
            qprof=SimpleNamespace(data=__import__("numpy").linspace(1.0, 3.0, 8)))

    def with_continuum(self, *, gamma_s=None, species=None, dense=True):
        self.counters["continuum"] += 1
        fresh = _FakeCase(self.counters)
        return fresh


@pytest.fixture
def counting_model(monkeypatch, tmp_path):
    """A model whose expensive steps only count themselves."""
    from supernova.gui import _model

    counters = {"equilibrium": 0, "continuum": 0}

    def fake_from_eqdsk(*args, **kwargs):
        counters["equilibrium"] += 1
        return _FakeCase(counters)

    monkeypatch.setattr(_model.Case, "from_eqdsk", staticmethod(fake_from_eqdsk))
    monkeypatch.setattr(_model, "profile_from_columns",
                        lambda *a, **k: object())
    return _model.ContinuumModel(), counters, _valid(tmp_path)


def test_changing_n_reuses_the_equilibrium_and_the_continuum(counting_model):
    model, counters, state = counting_model
    model.solve(state)
    assert counters == {"equilibrium": 1, "continuum": 0}

    state.ntor = 4
    result = model.solve(state)
    assert counters == {"equilibrium": 1, "continuum": 0}
    assert not result.reloaded_equilibrium and not result.rebuilt_continuum


def test_changing_gamma_rebuilds_only_the_continuum(counting_model):
    model, counters, state = counting_model
    model.solve(state)
    state.gamma_s = 0.0
    result = model.solve(state)
    assert counters == {"equilibrium": 1, "continuum": 1}
    assert result.rebuilt_continuum and not result.reloaded_equilibrium


def test_changing_the_eqdsk_reloads_everything(counting_model, tmp_path):
    model, counters, state = counting_model
    model.solve(state)
    other = tmp_path / "other.eqdsk"; other.write_text("x")
    state.eqdsk = str(other)
    result = model.solve(state)
    assert counters["equilibrium"] == 2
    assert result.reloaded_equilibrium


def test_the_model_refuses_an_invalid_state_before_touching_anything(counting_model):
    model, counters, state = counting_model
    state.mmax = 0
    with pytest.raises(ValueError, match="mmax"):
        model.solve(state)
    assert counters == {"equilibrium": 0, "continuum": 0}


def test_branch_and_coordinate_vocabularies_are_what_the_solver_accepts():
    assert BRANCHES == ("slow_sound", "alfven", "full")
    assert COORDINATES == ("psi_n", "rho_pol")


# ------------------------------------------------------------- the cursor

def _result(psi=None, khz=None, **overrides):
    """A ContinuumResult built from given curves, for the geometry tests."""
    import numpy as np
    from supernova.gui._model import ContinuumResult

    psi = np.linspace(0.0, 1.0, 101) if psi is None else psi
    if khz is None:
        khz = np.column_stack([50.0 + 100.0 * psi, 200.0 - 100.0 * psi])
    fields = dict(psi_n=psi, frequency_khz=khz, which="full", ntor=2,
                  gamma_s=5 / 3, q_min=1.0, q_max=4.0, solve_seconds=0.1,
                  reloaded_equilibrium=False, rebuilt_continuum=False)
    fields.update(overrides)
    return ContinuumResult(**fields)


def test_cuts_finds_a_crossing_on_a_rising_branch():
    """f = 100 kHz meets 50 + 100 psi at psi = 0.5."""
    import numpy as np
    result = _result(khz=np.column_stack(
        [50.0 + 100.0 * np.linspace(0.0, 1.0, 101)]))
    np.testing.assert_allclose(result.cuts(100.0), [0.5], atol=1e-12)


def test_cuts_finds_one_crossing_per_branch():
    import numpy as np
    cuts = _result().cuts(125.0)
    # rising branch reaches 125 at 0.75, falling branch at 0.75 as well
    np.testing.assert_allclose(cuts, [0.75], atol=1e-12)
    np.testing.assert_allclose(_result().cuts(100.0), [0.5, 1.0], atol=1e-12)


def test_a_frequency_below_every_branch_has_no_cuts():
    assert _result().cuts(10.0).size == 0


def test_a_frequency_above_every_branch_has_no_cuts():
    assert _result().cuts(1.0e6).size == 0


def test_cuts_ignores_a_nonpositive_or_nonfinite_frequency():
    import numpy as np
    assert _result().cuts(0.0).size == 0
    assert _result().cuts(-5.0).size == 0
    assert _result().cuts(np.nan).size == 0


def test_cuts_tolerates_branches_that_are_partly_nan():
    """Branches drop out where the local problem has no real root."""
    import numpy as np
    psi = np.linspace(0.0, 1.0, 101)
    khz = np.column_stack([50.0 + 100.0 * psi, np.full(psi.size, np.nan)])
    np.testing.assert_allclose(_result(psi, khz).cuts(100.0), [0.5], atol=1e-12)


def test_cuts_are_sorted_and_deduplicated():
    import numpy as np
    psi = np.linspace(0.0, 1.0, 101)
    curve = 50.0 + 100.0 * psi
    result = _result(psi, np.column_stack([curve, curve]))   # two identical branches
    np.testing.assert_allclose(result.cuts(100.0), [0.5], atol=1e-12)


# ------------------------------------------------------------------ export

def test_export_picks_hdf5_for_h5_and_netcdf4_for_nc(tmp_path):
    import numpy as np
    import xarray as xr
    from supernova.gui._model import ContinuumModel

    model = ContinuumModel()
    result = _result(family_code=np.zeros((101, 2), dtype=np.int8))

    h5 = tmp_path / "continuum.h5"
    assert model.save(result, h5) == "h5netcdf"
    nc = tmp_path / "continuum.nc"
    assert model.save(result, nc) == "netcdf4"

    for path in (h5, nc):
        with xr.open_dataset(path) as loaded:
            np.testing.assert_allclose(
                loaded["frequency_khz"].values, result.frequency_khz)
            assert loaded.attrs["which"] == "full"
            assert loaded.attrs["ntor"] == 2
            assert "family_code" in loaded


def test_an_hdf5_export_is_readable_by_h5py(tmp_path):
    """'via the xarray interface' still has to produce real HDF5."""
    h5py = pytest.importorskip("h5py")
    from supernova.gui._model import ContinuumModel

    path = tmp_path / "continuum.h5"
    ContinuumModel().save(_result(), path)
    with h5py.File(path, "r") as handle:
        assert "frequency_khz" in handle


def test_family_codes_carry_their_meaning_into_the_file(tmp_path):
    import numpy as np
    import xarray as xr
    from supernova.gui._model import ContinuumModel

    codes = np.tile(np.array([0, 1], dtype=np.int8), (101, 1))
    path = tmp_path / "c.h5"
    ContinuumModel().save(_result(family_code=codes), path)
    with xr.open_dataset(path) as loaded:
        assert "Alfvenic" in loaded["family_code"].attrs["labels"]
        assert "mixed" in loaded["family_code"].attrs["labels"]


def test_a_result_without_classification_writes_no_family_variable(tmp_path):
    import xarray as xr
    from supernova.gui._model import ContinuumModel

    path = tmp_path / "c.nc"
    ContinuumModel().save(_result(), path)
    with xr.open_dataset(path) as loaded:
        assert "family_code" not in loaded


def test_classification_is_rejected_for_the_pure_branches(tmp_path):
    state = _valid(tmp_path, which="slow_sound", classify=True)
    assert any("full sound-Alfven" in p for p in state.validate())
    assert _valid(tmp_path, which="full", classify=True).validate() == []
