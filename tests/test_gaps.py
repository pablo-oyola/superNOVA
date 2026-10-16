"""Continuum gaps, mode-size estimates, path amplitudes and graded meshes."""

import numpy as np

from supernova.api import RadialGrid
from supernova.radial.grading import graded_breakpoints
from supernova.solve.gaps import (
    ContinuumTip,
    Gap,
    RadialGeometry,
    continuum_tips,
    estimate_gap,
    find_gaps,
    harmonic_correlation,
    mode_width_m,
    path_alpha,
)


def avoided_crossing(x, centre=0.4, f0=70.0, slope=60.0, half_gap=4.0):
    """Two branches f0 +- slope (x - centre) split by a toroidal gap."""
    split = np.sqrt((slope * (x - centre)) ** 2 + half_gap**2)
    return np.stack([f0 - split, f0 + split, 3.0 * f0 + 0.0 * x], axis=1)


def test_an_avoided_crossing_is_one_gap_with_its_edges_and_centre():
    x = np.linspace(0.0, 1.0, 401)
    gaps = find_gaps(continuum_tips(x, avoided_crossing(x)))
    assert len(gaps) == 1
    gap = gaps[0]
    assert np.isclose(gap.x, 0.4, atol=2e-3)
    assert np.isclose(gap.f_low, 66.0, atol=1e-2) and np.isclose(gap.f_high, 74.0, atol=1e-2)
    assert gap.lower.is_maximum and not gap.upper.is_maximum


def test_a_crossing_of_uncoupled_branches_is_not_a_gap():
    x = np.linspace(0.0, 1.0, 401)
    f = np.stack([70.0 - 60.0 * (x - 0.4), 70.0 + 60.0 * (x - 0.4)], axis=1)
    assert find_gaps(continuum_tips(x, f)) == []


def test_the_singular_pair_distance_follows_the_parabola_and_vanishes_at_the_edge():
    # s(x) = s_tip - c (x - x_tip)^2 below a maximum: s = s_tip + c d^2 off axis
    tip = ContinuumTip(x=0.3, f_khz=60.0, s2=-2.0 * 5.0e4, level=0)
    f = 61.0
    assert np.isclose(tip.distance(f), np.sqrt((f**2 - 60.0**2) / 5.0e4))
    assert tip.distance(59.0) == 0.0                 # real crossings below the maximum
    gap = Gap(tip, ContinuumTip(x=0.3, f_khz=70.0, s2=2.0 * 5.0e4, level=1))
    assert gap.distance(60.0 + 1e-9) < 1e-5 and gap.distance(80.0) == 0.0


def test_path_alpha_keeps_the_height_below_kappa_d_at_the_facing_tip():
    tip = ContinuumTip(x=0.25, f_khz=72.0, s2=4.0e5, level=1)        # a minimum
    alpha, limiting = path_alpha(71.5, [tip], (0.0, 1.0), kappa=0.5, alpha_max=1.0)
    assert limiting is tip
    assert np.isclose(alpha * 0.25 * 0.75, 0.5 * tip.distance(71.5))
    assert path_alpha(72.5, [tip], (0.0, 1.0), alpha_max=0.05)[0] == 0.05   # not faced


class _Equilibrium:
    """q = 1.7 + 0.2 psi_N on psi_N in (0, 1); a = 1 m, R0 = 10 m."""

    def __init__(self):
        import xarray as xr

        x = np.linspace(0.0, 1.0, 201)
        self.qprof = type("P", (), {"data": xr.DataArray(1.7 + 0.2 * x, coords={"x": x},
                                                          dims=["x"])})()
        self.magspace = xr.Dataset(coords={"psi": x})
        self.aminor, self.Rmajor = 1.0, 10.0


def test_geometry_and_the_width_regimes():
    geometry = RadialGeometry.from_equilibrium(_Equilibrium())
    assert np.isclose(geometry.r[-1], 1.0) and np.all(np.diff(geometry.r) > 0.0)
    x = 0.5
    r0, q0 = geometry.at("r", x), geometry.at("q", x)
    s, eps = geometry.shear(x), r0 / 10.0
    w, w_shear, w_low = mode_width_m(geometry, 6, x)
    assert np.isclose(w_shear, r0 / (6 * q0) * eps / abs(s))
    assert np.isclose(w_low, r0 / (6 * q0) * np.sqrt(eps / abs(s)))
    assert np.isclose(w, min(w_shear, w_low))
    gap = Gap(ContinuumTip(x, 60.0, -1.0e5, 0), ContinuumTip(x, 70.0, 1.0e5, 1))
    estimate = estimate_gap(gap, geometry, 6, kind="TAE", m=(10, 11))
    assert np.isclose(estimate.width_mm, w * 1e3)
    assert estimate.element_mm("1e-5") <= estimate.element_mm("1e-4")
    assert estimate.edge_distance_mm(60.0 + 1e-9) < 1e-2


def test_graded_breakpoints_reach_the_local_size_and_relax_smoothly():
    points = graded_breakpoints((0.0, 1.0), 0.02, [0.3], [0.0005], growth=0.15)
    sizes = np.diff(points)
    assert points[0] == 0.0 and points[-1] == 1.0
    assert sizes.min() < 0.0006 and sizes.max() <= 0.02 + 1e-12
    ratio = np.maximum(sizes[1:] / sizes[:-1], sizes[:-1] / sizes[1:])
    assert ratio.max() <= 1.0 / (1.0 - 0.15) + 1e-9
    near = points[np.argmin(np.abs(points - 0.3))]
    assert abs(near - 0.3) < 0.001
    grid = RadialGrid.graded(50, [0.3], [0.0005], domain=(0.0, 1.0))
    assert np.array_equal(grid.knots, points)


def test_harmonic_correlation_ignores_phase_and_scale():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(30, 4)) + 1j * rng.normal(size=(30, 4))
    assert np.isclose(harmonic_correlation(a, 3.0 * np.exp(0.7j) * a), 1.0)
    assert harmonic_correlation(a, rng.normal(size=(30, 4))) < 0.5


def test_path_amplitudes_share_levels():
    from supernova.gui._solver import _alpha_level

    assert _alpha_level(0.2, 0.05) == 0.05
    assert np.isclose(_alpha_level(0.0486, 0.05), 0.05 / np.sqrt(2.0))
    level = _alpha_level(0.0104, 0.05)
    assert level <= 0.0104 and level > 0.0104 / np.sqrt(2.0)
