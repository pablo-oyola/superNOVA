"""The equilibrium tab's state and snapshot, without Qt.

The state tests need no equilibrium at all. The snapshot tests use the
analytic ITPA case, which builds in well under a second, and check what the
tab draws against identities that must hold whatever the drawing code does:
the covariant metric is the inverse of the contravariant one, so
sqrt(g_zeta_zeta) is R; and the Boozer Jacobian is h / B**2.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from supernova.gui._equilibrium import (
    SURFACE_QUANTITIES,
    EquilibriumModel,
    EquilibriumState,
)


# --------------------------------------------------------------- validation

def test_an_empty_eqdsk_state_asks_for_the_file_but_not_for_a_density():
    problems = EquilibriumState().validate()
    assert any("equilibrium file" in p for p in problems)
    assert not any("density" in p.lower() for p in problems)


def test_the_itpa_defaults_are_complete():
    assert EquilibriumState(source="itpa").validate() == []


@pytest.mark.parametrize("overrides, expected", [
    ({"boozer_min": 0.9, "boozer_max": 0.1}, "Boozer domain"),
    ({"density_uniform_m3": -1.0}, "uniform density"),
    ({"boozer_npsi": 8}, "resolution"),
    ({"surface_psi_n": 1.5}, "flux surface"),
    ({"surface_quantities": ["not_a_metric"]}, "not_a_metric"),
    ({"profile_x": "rho_tor"}, "Profile axis"),
])
def test_validation_catches_each_impossible_eqdsk_setting(tmp_path, overrides, expected):
    eqdsk = tmp_path / "eq.eqdsk"; eqdsk.write_text("x")
    problems = EquilibriumState(eqdsk=str(eqdsk), **overrides).validate()
    assert any(expected in p for p in problems), problems


@pytest.mark.parametrize("overrides, expected", [
    ({"itpa_rmajor": 0.5}, "major radius"),
    ({"itpa_bt_sign": 0}, "+1 or -1"),
    ({"itpa_ns": 2}, "surfaces"),
])
def test_validation_catches_each_impossible_itpa_setting(overrides, expected):
    problems = EquilibriumState(source="itpa", **overrides).validate()
    assert any(expected in p for p in problems), problems


def test_the_equilibrium_key_ignores_the_view():
    """Moving the surface or ticking a quantity must never reload."""
    base = EquilibriumState(source="itpa")
    for field, value in (("surface_psi_n", 0.2), ("grid_surfaces", 3),
                         ("surface_quantities", ["B"]), ("profile_x", "rho_pol"),
                         ("refine_fine_dx", 0.01)):
        changed = EquilibriumState.from_dict({**base.to_dict(), field: value})
        assert changed.equilibrium_key() == base.equilibrium_key(), field


def test_a_uniform_density_changes_what_is_loaded():
    base = EquilibriumState(eqdsk="shot.geqdsk")
    assert (EquilibriumState(eqdsk="shot.geqdsk", density_uniform_m3=2e19)
            .equilibrium_key() != base.equilibrium_key())


def test_the_key_of_one_source_ignores_the_other_sources_settings():
    base = EquilibriumState(source="itpa")
    assert (EquilibriumState(source="itpa", eqdsk="other").equilibrium_key()
            == base.equilibrium_key())


def test_a_session_round_trips_through_json(tmp_path):
    state = EquilibriumState(source="itpa", itpa_q0=1.5, surface_psi_n=0.25,
                             surface_quantities=["B", "cov_theta_theta"])
    path = tmp_path / "eq.json"
    state.to_json(path)
    assert EquilibriumState.from_json(path) == state


def test_loading_a_session_with_an_unknown_key_says_which(tmp_path):
    path = tmp_path / "eq.json"
    path.write_text(json.dumps({"source": "itpa", "boozer_lambda": 3}))
    with pytest.raises(ValueError, match="boozer_lambda"):
        EquilibriumState.from_json(path)


# ------------------------------------------------------------------ sharing

def test_itpa_offers_the_continuum_tab_nothing():
    assert EquilibriumState(source="itpa").shared_inputs() == {}


def test_an_eqdsk_without_density_does_not_offer_one():
    """A placeholder must never overwrite the continuum tab's real density."""
    inputs = EquilibriumState(eqdsk="shot.geqdsk", cocos_in=7).shared_inputs()
    assert inputs["eqdsk"] == "shot.geqdsk" and inputs["cocos_in"] == 7
    assert "density_file" not in inputs


def test_shared_choices_use_the_continuum_tabs_names():
    from dataclasses import fields

    from supernova.gui._state import ContinuumState

    inputs = EquilibriumState(eqdsk="shot.geqdsk", density_file="ne.dat").shared_inputs()
    assert set(inputs) <= {f.name for f in fields(ContinuumState)}
    assert inputs["density_file"] == "ne.dat"


# --------------------------------------------------------------- refinement

def _refinable(tmp_path, **overrides) -> EquilibriumState:
    eqdsk = tmp_path / "shot.eqdsk"; eqdsk.write_text("x")
    return EquilibriumState(eqdsk=str(eqdsk), **overrides)


def test_the_refinement_defaults_are_accepted_and_land_beside_the_source(tmp_path):
    state = _refinable(tmp_path)
    assert state.refinement_problems() == []
    spec = state.refinement_spec()
    assert spec["output_dir"] == str((tmp_path / "tokamaker_refined").resolve())
    assert spec["stem"] == "shot_tokamaker"
    assert spec["refinement_dx_sequence"] == [0.025, 0.020, 0.015]


def test_the_run_info_fits_the_forty_characters_tokamaker_allows(tmp_path):
    """TokaMaker refuses a longer g-file header after the whole fit has run."""
    eqdsk = tmp_path / ("a_very_long_shot_name_" * 4 + ".eqdsk")
    eqdsk.write_text("x")
    assert len(EquilibriumState(eqdsk=str(eqdsk)).refinement_spec()["run_info"]) <= 40


@pytest.mark.parametrize("overrides, expected", [
    ({"refine_fine_dx": 0.05}, "finer"),
    ({"refine_dx_sequence": "0.02, 0.025, 0.015"}, "decrease"),
    ({"refine_dx_sequence": "0.025, 0.020"}, "end at the fine"),
    ({"refine_dx_sequence": "0.025, oops"}, "numbers"),
])
def test_refinement_problems_are_caught_before_tokamaker_starts(tmp_path, overrides, expected):
    problems = _refinable(tmp_path, **overrides).refinement_problems()
    assert any(expected in p for p in problems), problems


def test_the_itpa_case_cannot_be_refined():
    assert EquilibriumState(source="itpa").refinement_problems()


def test_the_refinement_spec_builds_the_config_the_package_validates(tmp_path):
    pytest.importorskip("OpenFUSIONToolkit")
    from supernova.gui._refine_job import config_from_spec

    config = config_from_spec(_refinable(tmp_path, refine_max_iterations=7,
                                         refine_fit_current=True).refinement_spec())
    assert config.fit.max_iterations == 7
    assert config.fit.fit_plasma_current is True
    assert config.refinement_dx_sequence[-1] == config.fine_dx


# ----------------------------------------------------------------- snapshot

@pytest.fixture(scope="module")
def itpa_view():
    return EquilibriumModel().load(EquilibriumState(source="itpa", itpa_ns=64,
                                                    itpa_ntheta=64))


def test_the_itpa_view_offers_every_surface_quantity(itpa_view):
    assert [name for name in SURFACE_QUANTITIES if not itpa_view.available(name)] == []


def test_the_covariant_metric_inverts_the_contravariant_one(itpa_view):
    """g_zeta_zeta = R**2 exactly, because e_zeta = R e_phi in Boozer angles."""
    for psi_n in (0.2, 0.5, 0.9):
        surface = itpa_view.surface(psi_n, ["cov_zeta_zeta", "R"])
        values = surface["values"]
        np.testing.assert_allclose(np.sqrt(values["cov_zeta_zeta"]), values["R"],
                                   rtol=1.0e-4)


def test_the_drawn_grid_has_the_orientation_of_the_metric(itpa_view):
    """g_psi_Theta from finite differences of R and Z has the metric's sign.

    The positive-q reflection flips exactly the one-Theta-index components;
    a grid drawn in the unreflected angle would correlate at -1.
    """
    index = itpa_view.surface_index(0.5)
    span = (itpa_view.equilibrium.boozer.coords.psi0.attrs["psi_boundary"]
            - itpa_view.equilibrium.boozer.coords.psi0.attrs["psi_axis"])
    dpsi = (itpa_view.psi_n[index + 1] - itpa_view.psi_n[index - 1]) * span
    R_psi = (itpa_view.R[index + 1] - itpa_view.R[index - 1]) / dpsi
    Z_psi = (itpa_view.Z[index + 1] - itpa_view.Z[index - 1]) / dpsi
    R_theta = np.gradient(itpa_view.R[index], itpa_view.theta)
    Z_theta = np.gradient(itpa_view.Z[index], itpa_view.theta)
    metric = itpa_view.surface(itpa_view.psi_n[index], ["cov_psi_theta"])
    inner = slice(2, -2)
    correlation = np.corrcoef(metric["values"]["cov_psi_theta"][inner],
                              (R_psi * R_theta + Z_psi * Z_theta)[inner])[0, 1]
    assert correlation > 0.99


def test_the_coordinate_jacobian_matches_h_over_b_squared(itpa_view):
    surface = itpa_view.surface(0.5, ["jac", "jac_boozer"])
    np.testing.assert_allclose(surface["values"]["jac"],
                               surface["values"]["jac_boozer"], rtol=1.0e-4)


def test_the_surface_nearest_the_request_is_the_one_returned(itpa_view):
    surface = itpa_view.surface(0.31, ["B"])
    assert abs(surface["psi_n"] - 0.31) <= np.max(np.diff(itpa_view.psi_n))
    assert surface["values"]["B"].shape == itpa_view.theta.shape


def test_the_derivative_of_a_constant_f_is_exactly_zero(itpa_view):
    """ITPA's F is constant; round-off must not plot as structure."""
    assert np.all(itpa_view.profiles["ffprime"] == 0.0)


def test_the_view_summarises_q_and_flags_nothing_for_itpa(itpa_view):
    summary = itpa_view.summary
    assert summary["q_inner"] == pytest.approx(1.71, rel=1e-3)
    assert summary["q_edge"] == pytest.approx(1.87, rel=1e-3)
    assert not itpa_view.density_is_placeholder


def test_the_view_exports_to_a_dataset(itpa_view):
    dataset = itpa_view.to_dataset()
    assert {"q", "gpsi_psi", "R", "Z"} <= set(dataset.data_vars)
    assert dataset["R"].attrs["units"] == "m"


def test_an_unchanged_state_reuses_the_loaded_view():
    model = EquilibriumModel()
    state = EquilibriumState(source="itpa", itpa_ns=16, itpa_ntheta=16)
    first = model.load(state)
    state.surface_psi_n = 0.9
    assert model.load(state) is first
