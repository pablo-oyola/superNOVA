"""The continuum tab's state and caching, without Qt or physics.

The tab is deliberately thin: it edits a ContinuumState, and the compute
process hands it to a ContinuumModel together with the equilibrium the
Equilibrium tab loaded. Both are plain objects, so what is worth testing --
validation, session round-trips, and which expensive step gets skipped on which
edit -- is testable without a display and without loading an equilibrium.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from supernova.gui._state import BRANCHES, COORDINATES, ContinuumState, split_legacy


# --------------------------------------------------------------- validation

def test_the_default_state_needs_nothing_more():
    """The equilibrium comes from the Equilibrium tab, so there are no files here."""
    assert ContinuumState().validate() == []


def _valid(tmp_path, **overrides) -> ContinuumState:
    return ContinuumState(**overrides)


@pytest.mark.parametrize("overrides, expected", [
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


def test_an_old_session_keeps_its_equilibrium_inputs_aside(tmp_path):
    """Sessions from when this tab loaded its own equilibrium still open."""
    from supernova.gui._session import read_session

    path = tmp_path / "old.json"
    path.write_text(json.dumps({"eqdsk": "/shot.geqdsk", "cocos_in": 7,
                                "density_file": "/ne.dat", "boozer_max": 0.98,
                                "ntor": 4, "mmax": 30, "species": "T"}))
    state = ContinuumState.from_json(path)
    assert (state.ntor, state.mmax) == (4, 30)
    settings, legacy = split_legacy(json.loads(path.read_text()))
    assert "eqdsk" not in settings and legacy["cocos_in"] == 7
    sessions = read_session(path)
    equilibrium = sessions["continuum_equilibrium"]
    assert (equilibrium.eqdsk, equilibrium.cocos_in, equilibrium.boozer_max,
            equilibrium.species) == ("/shot.geqdsk", 7, 0.98, "T")


def test_loading_a_session_with_an_unknown_key_says_which(tmp_path):
    path = tmp_path / "session.json"
    path.write_text(json.dumps({"ntor": 3, "flux_surface_count": 17}))
    with pytest.raises(ValueError, match="flux_surface_count"):
        ContinuumState.from_json(path)


# -------------------------------------------------------- what invalidates what

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


def _fake_equilibrium():
    import numpy as np

    return SimpleNamespace(
        qprof=SimpleNamespace(data=np.linspace(1.0, 3.0, 8)),
        magspace=SimpleNamespace(coords={"psi": SimpleNamespace(values=np.linspace(0, 1, 8))}),
        registry=_alfven_registry())


@pytest.fixture
def counting_model(monkeypatch):
    """A model whose continuum construction only counts itself."""
    from supernova import continuum as continuum_module
    from supernova.gui import _model

    counters = {"continuum": 0}

    def fake_continuum(equilibrium, **kwargs):
        counters["continuum"] += 1
        return _FakeContinuum()

    monkeypatch.setattr(continuum_module, "superContinuum", fake_continuum)
    return _model.ContinuumModel(), counters, ContinuumState(), _fake_equilibrium()


def test_changing_n_reuses_the_continuum(counting_model):
    model, counters, state, equilibrium = counting_model
    first = model.solve(state, equilibrium, key=1)
    assert counters == {"continuum": 1} and first.reloaded_equilibrium

    state.ntor = 4
    result = model.solve(state, equilibrium, key=1)
    assert counters == {"continuum": 1}
    assert not result.reloaded_equilibrium and not result.rebuilt_continuum


def test_changing_gamma_rebuilds_only_the_continuum(counting_model):
    model, counters, state, equilibrium = counting_model
    model.solve(state, equilibrium, key=1)
    state.gamma_s = 0.0
    result = model.solve(state, equilibrium, key=1)
    assert counters == {"continuum": 2}
    assert result.rebuilt_continuum and not result.reloaded_equilibrium


def test_another_equilibrium_rebuilds_and_says_so(counting_model):
    model, counters, state, equilibrium = counting_model
    model.solve(state, equilibrium, key=1)
    result = model.solve(state, _fake_equilibrium(), key=2)
    assert counters == {"continuum": 2}
    assert result.reloaded_equilibrium


def test_the_model_refuses_an_invalid_state_before_touching_anything(counting_model):
    model, counters, state, equilibrium = counting_model
    state.mmax = 0
    with pytest.raises(ValueError, match="mmax"):
        model.solve(state, equilibrium, key=1)
    assert counters == {"continuum": 0}


def test_radii_are_the_physical_psi_n(counting_model):
    import numpy as np

    model, _, state, equilibrium = counting_model
    x = equilibrium.magspace.coords["psi"].values
    equilibrium.magspace.coords["psi_physical_norm"] = SimpleNamespace(values=0.01 + 0.9 * x)
    result = model.solve(state, equilibrium, key=1)
    assert np.isclose(result.psi_n[0], 0.01) and np.isclose(result.psi_n[-1], 0.91)


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


# --------------------------------------------- classification feasibility

def _feasibility_case(q_min=1.71, q_max=1.87, available=48):
    """A Case-shaped stub carrying only what the feasibility check reads."""
    import numpy as np
    return SimpleNamespace(
        equilibrium=SimpleNamespace(
            qprof=SimpleNamespace(data=np.linspace(q_min, q_max, 32))),
        continuum=SimpleNamespace(H=SimpleNamespace(m_pol=np.arange(available + 1))),
    )


@pytest.mark.parametrize("ntor", [1, 2, 3, 4, 6, 10])
def test_a_band_away_from_m_zero_is_classifiable(ntor):
    """The operators convolve, so only the band's span matters, not |m|.

    This is the case the classifier used to reject: at ITPA's q the band sits
    at m = 7..15 for n = 6, far from zero but only eight wide.
    """
    from supernova.gui._model import classification_feasibility

    assert classification_feasibility(_feasibility_case(), ntor, 20) == (True, "")


def test_an_equilibrium_with_too_few_harmonics_cannot_classify():
    from supernova.gui._model import classification_feasibility

    ok, why = classification_feasibility(_feasibility_case(available=4), 6, 20)
    assert ok is False
    assert "mode difference of 8" in why
    assert "carries only 4" in why
    assert "m_max at least 8" in why
    assert "branches are still correct" in why


def test_shear_not_n_is_what_outruns_a_finite_equilibrium():
    """The span is n*(q_max - q_min) + 6, so wide shear is the real cost.

    ITPA's q barely moves (1.71 to 1.87), so even n = 60 spans only 17
    harmonics. A MAST-like profile reaching q = 15 spans more than 48 by n = 4.
    """
    from supernova.gui._model import classification_feasibility

    narrow = _feasibility_case(1.71, 1.87, available=48)
    assert classification_feasibility(narrow, 60, 200)[0] is True

    wide = _feasibility_case(1.0, 15.0, available=48)
    assert classification_feasibility(wide, 2, 200)[0] is True
    assert classification_feasibility(wide, 4, 200)[0] is False


def test_an_empty_band_is_reported_as_such():
    from supernova.gui._model import classification_feasibility

    ok, why = classification_feasibility(_feasibility_case(), 6, mmax=-20)
    assert ok is False
    assert "raise mmax" in why
