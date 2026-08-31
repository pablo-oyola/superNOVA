"""The user-facing API: unit conversion, windowing, and result reconstruction.

These test the parts of :mod:`supernova.api` that carry real arithmetic, and in
particular that :meth:`supernova.api.Band.region` reproduces the hand-rolled
``freq_to_elipse`` every case script used to carry.  Building a real
equilibrium is far too slow for a unit test, so the unit registry is stubbed
where one is needed -- what matters here is the algebra, not the Boozer
transform.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import unyt

import supernova as sn
from supernova.api import Band, PetscOptions, RadialGrid, profile_from_columns


class _StubCase:
    """A Case-shaped object whose registry maps kHz to omegaA linearly."""

    def __init__(self, khz_per_omega_a: float = 500.0):
        registry = unyt.UnitRegistry()
        registry.add("omegaA", khz_per_omega_a * 1000.0, unyt.dimensions.rate)
        self.registry = registry


@pytest.fixture
def stub_case():
    return _StubCase()


# --------------------------------------------------------------------------
# Band
# --------------------------------------------------------------------------

def test_band_rejects_a_reversed_or_degenerate_range():
    with pytest.raises(ValueError, match="must increase"):
        Band(150.0, 50.0)
    with pytest.raises(ValueError, match="must increase"):
        Band(50.0, 50.0)


def test_band_refuses_to_convert_before_it_knows_the_equilibrium():
    """omegaA is a property of the equilibrium, so conversion needs a Case."""
    with pytest.raises(ValueError, match="not attached to a Case"):
        Band(50.0, 150.0).omega_range


def test_band_region_matches_the_hand_rolled_conversion(stub_case):
    """Reproduce freq_to_elipse from no_sync/MAST/window_search.py exactly."""
    band = Band(50.0, 51.0, units="kHz", vscale=0.25).for_case(stub_case)
    target_omega, region = band.region()

    frequency = unyt.unyt_array([50.0, 51.0], "kHz", registry=stub_case.registry)
    omega_min, omega_max = 2.0 * np.pi * frequency.to("omegaA").v
    s_min, s_max = omega_min**2, omega_max**2
    expected_center = 0.5 * (s_min + s_max)
    expected_radius = 0.5 * (s_max - s_min)

    assert region["type"] == "ellipse"
    assert region["center"] == pytest.approx(expected_center, rel=1e-14)
    assert region["radius"] == pytest.approx(expected_radius, rel=1e-14)
    assert region["vscale"] == 0.25
    assert complex(target_omega) == pytest.approx(np.sqrt(expected_center), rel=1e-14)


def test_band_round_trips_frequency_through_normalized_omega(stub_case):
    band = Band(50.0, 150.0).for_case(stub_case)
    frequencies = np.array([50.0, 83.0, 150.0])
    np.testing.assert_allclose(
        band.to_frequency(band.to_omega(frequencies)), frequencies, rtol=1e-13
    )


def test_band_windows_tile_the_range_and_stop_at_the_top():
    windows = Band(50.0, 150.0).windows(1.0)
    assert len(windows) == 100
    assert windows[0] == (50.0, 51.0)
    assert windows[-1] == (149.0, 150.0)
    assert all(high <= 150.0 for _, high in windows)


def test_band_windows_overlap_by_the_requested_fraction():
    windows = Band(0.0, 10.0).windows(2.0, overlap=0.5)
    starts = [low for low, _ in windows]
    assert starts[1] - starts[0] == pytest.approx(1.0)
    assert windows[0] == (0.0, 2.0)


def test_band_windows_reject_impossible_shapes():
    with pytest.raises(ValueError, match="width must be positive"):
        Band(50.0, 150.0).windows(0.0)
    with pytest.raises(ValueError, match="0 <= overlap < 1"):
        Band(50.0, 150.0).windows(1.0, overlap=1.0)


def test_a_narrow_band_still_yields_one_window():
    assert Band(50.0, 50.5).windows(1.0) == [(50.0, 50.5)]


# --------------------------------------------------------------------------
# RadialGrid and options
# --------------------------------------------------------------------------

def test_uniform_grid_builds_the_requested_bspline_space():
    grid = RadialGrid.uniform(spans=12, order=3, quad_order=6)
    assert grid.spans == 12
    space = grid.build()
    assert space.order == 3
    np.testing.assert_allclose(grid.knots[[0, -1]], [0.0, 1.0])


def test_uniform_grid_rejects_an_empty_mesh():
    with pytest.raises(ValueError, match="at least 1"):
        RadialGrid.uniform(spans=0)


def test_petsc_options_name_the_factorization_instead_of_spelling_it():
    assert PetscOptions("mumps").as_dict() == {
        "st_ksp_type": "preonly",
        "st_pc_type": "lu",
        "st_pc_factor_mat_solver_type": "mumps",
    }
    assert PetscOptions("spqr").as_dict()["st_pc_type"] == "qr"


def test_petsc_options_extra_entries_win():
    options = PetscOptions("mumps", extra={"st_ksp_type": "gmres"})
    assert options.as_dict()["st_ksp_type"] == "gmres"


def test_petsc_options_reject_an_unknown_factorization():
    with pytest.raises(ValueError, match="Unknown factorization"):
        PetscOptions("lapack")


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------

def test_profile_from_columns_reproduces_the_case_script_interpolation(tmp_path):
    """The scripts interpolate in rho_pol even though the column is psi_N."""
    path = tmp_path / "ne.dat"
    psi = np.linspace(0.0, 1.0, 21)
    density = 1.0 - psi**2
    path.write_text("psi ne\n" + "\n".join(
        f"{a:.6f} {b:.6f}" for a, b in zip(psi, density)))

    profile = profile_from_columns(path, units="m^-3", points=33)
    values = np.asarray(profile.data.values).squeeze()

    target = np.linspace(0.0, 1.0, 33)
    expected = np.interp(np.sqrt(target), np.sqrt(psi), density)
    np.testing.assert_allclose(values, expected, rtol=1e-12)


def test_interpolating_in_psi_gives_a_different_profile(tmp_path):
    """The two choices are not equivalent, which is why both are exposed."""
    path = tmp_path / "ne.dat"
    psi = np.linspace(0.0, 1.0, 9)
    path.write_text("psi ne\n" + "\n".join(
        f"{a:.6f} {1.0 - a**2:.6f}" for a in psi))

    in_rho = np.asarray(profile_from_columns(path, units="m^-3").data.values)
    in_psi = np.asarray(
        profile_from_columns(path, units="m^-3", interpolate_in="psi_n").data.values)
    assert not np.allclose(in_rho, in_psi)


def test_profile_from_columns_accepts_a_rho_pol_column(tmp_path):
    path = tmp_path / "ne.dat"
    rho = np.linspace(0.0, 1.0, 21)
    path.write_text("rho ne\n" + "\n".join(
        f"{a:.9f} {1.0 - a:.9f}" for a in rho))

    profile = profile_from_columns(path, units="m^-3", x="rho_pol", points=33)
    values = np.asarray(profile.data.values).squeeze()
    # Linear in rho_pol, and the interpolation runs in rho_pol, so it is exact.
    np.testing.assert_allclose(
        values, 1.0 - np.sqrt(np.linspace(0.0, 1.0, 33)), atol=1e-9)


def test_profile_from_columns_rejects_unknown_coordinates(tmp_path):
    path = tmp_path / "ne.dat"
    path.write_text("a b\n0.0 1.0\n1.0 0.0")
    with pytest.raises(ValueError, match="x must be"):
        profile_from_columns(path, units="m^-3", x="toroidal")
    with pytest.raises(ValueError, match="interpolate_in must be"):
        profile_from_columns(path, units="m^-3", interpolate_in="rho_tor")


# --------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------

def test_the_public_names_are_reachable_from_the_package_root():
    for name in ("Case", "Band", "RadialGrid", "Scan", "ModeSet",
                 "PetscOptions", "profile_from_columns", "profile_from_eqdsk"):
        assert hasattr(sn, name), name
