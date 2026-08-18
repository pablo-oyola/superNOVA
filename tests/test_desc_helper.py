import os
from types import SimpleNamespace

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-supernova")

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("desc")
from desc.grid import LinearGrid

from supernova.core.continuum import superContinuum
from supernova.core.desc_helper import (
    DESCSolver,
    _dtheta,
    _local_magnetic_shear_desc,
    _profile_values,
    _reflect_boozer_surface_arrays,
    _rho_tor_from_q,
)
from supernova.core.solver import superSolver
from supernova.radial.bspline import BSplineFE
from supernova.radial.profile import Profile


def _profile(x, values, units, name):
    return Profile(
        x=x,
        F_nm=np.asarray(values, dtype=np.float64),
        attrs={"units": units, "name": name},
    )


def test_rho_tor_mapping_accepts_same_sign_nonmonotonic_q():
    x = np.linspace(0.0, 1.0, 9)
    constant = _rho_tor_from_q(x, np.full_like(x, 2.0))
    np.testing.assert_allclose(constant, np.sqrt(x))

    positive = _rho_tor_from_q(x, 2.0 + 0.25 * np.sin(2.0 * np.pi * x))
    negative = _rho_tor_from_q(x, -(2.0 + 0.25 * np.sin(2.0 * np.pi * x)))
    assert np.all(np.diff(positive) > 0.0)
    np.testing.assert_allclose(negative, positive)


@pytest.mark.parametrize(
    "q",
    [
        np.array([1.0, 1.0, 0.0, 1.0]),
        np.array([1.0, 1.0, -1.0, -1.0]),
    ],
)
def test_rho_tor_mapping_rejects_zero_or_sign_changing_q(q):
    with pytest.raises(ValueError, match="non-zero|one sign"):
        _rho_tor_from_q(np.linspace(0.0, 1.0, q.size), q)


def test_from_eqdsk_extracts_profiles_on_density_grid():
    rhop = np.linspace(0.0, 1.0, 6)
    profiles = xr.Dataset(
        {
            "q": xr.DataArray(1.5 + rhop**2, dims=("rhop",), coords={"rhop": rhop}),
            "pres": xr.DataArray(2.0e3 * (1.0 - rhop**2), dims=("rhop",), coords={"rhop": rhop}),
            "rho": xr.DataArray(rhop, dims=("rhop",), coords={"rhop": rhop}),
        }
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    boundary = xr.Dataset(
        {
            "R": xr.DataArray(3.0 + np.cos(theta), dims=("point",)),
            "z": xr.DataArray(np.sin(theta), dims=("point",)),
        }
    )
    fluxdata = xr.Dataset()
    fluxdata.attrs.update({"Raxis": 3.0, "zaxis": 0.0})

    def flux_surface(rhop_value):
        return xr.Dataset(
            {
                "R": xr.DataArray(3.0 + rhop_value * np.cos(theta), dims=("point",)),
                "z": xr.DataArray(rhop_value * np.sin(theta), dims=("point",)),
            }
        )

    fake_eqdsk = SimpleNamespace(
        cocos_internal=1,
        phiclockwise_internal=False,
        flux_normalization_internal="Wb/rad",
        gs_profs=profiles,
        boundary=boundary,
        fluxdata=fluxdata,
        R_axis=3.0,
        z_axis=0.0,
        Bdata=SimpleNamespace(Baxis=xr.DataArray(1.0)),
        flux_surface=flux_surface,
    )
    x = np.linspace(0.0, 1.0, 9)
    density = _profile(x, 1.0e19 * (1.0 + x), "m**-3", "density")

    solver = DESCSolver.from_eqdsk(
        fake_eqdsk,
        density,
        use_surface_guess=False,
        boundary_m_max=1,
    )

    np.testing.assert_allclose(solver.qin.x, x)
    pressure_values = solver.pressure.data.sel(mpol=0, ntor=0).values.real
    np.testing.assert_allclose(pressure_values, 2.0e3 * (1.0 - x))
    expected_rho_tor = _rho_tor_from_q(
        x,
        solver.qin.data.sel(mpol=0, ntor=0).values.real,
    )
    np.testing.assert_allclose(solver.rho_tor_input, expected_rho_tor)
    np.testing.assert_allclose(solver.desc_pressure.knots, expected_rho_tor)
    np.testing.assert_allclose(solver.desc_density.knots, expected_rho_tor)
    np.testing.assert_allclose(solver.desc_iota.knots, expected_rho_tor)
    assert solver.cocos_internal == 1
    assert solver.phiclockwise_internal is False
    assert solver.flux_normalization_internal == "Wb/rad"

    truncated = DESCSolver.from_eqdsk(
        fake_eqdsk,
        density,
        use_surface_guess=False,
        rhop_boundary=0.8,
        boundary_m_max=1,
        boundary_k_max=0,
    )
    x_eqdsk = 0.8**2 * x
    np.testing.assert_allclose(
        _profile_values(truncated.qin),
        np.interp(x_eqdsk, rhop**2, profiles.q.values),
    )
    np.testing.assert_allclose(
        _profile_values(truncated.pressure),
        np.interp(x_eqdsk, rhop**2, profiles.pres.values),
    )
    np.testing.assert_allclose(
        _profile_values(truncated.density),
        np.interp(x_eqdsk, x, _profile_values(density)),
    )
    np.testing.assert_allclose(truncated.Rlcfs.max() - 3.0, 0.8)
    assert truncated.rhop_boundary == 0.8
    assert truncated.psipol_boundary == 0.8**2


def test_from_eqdsk_rejects_noncanonical_input_before_profile_access():
    density = _profile(
        np.linspace(0.0, 1.0, 5),
        np.ones(5),
        "m**-3",
        "density",
    )
    noncanonical = SimpleNamespace(
        cocos_internal=2,
        phiclockwise_internal=False,
        flux_normalization_internal="Wb",
    )

    with pytest.raises(ValueError, match="internal COCOS 1"):
        DESCSolver.from_eqdsk(
            noncanonical,
            density,
            use_surface_guess=False,
        )


@pytest.fixture(scope="module")
def desc_equilibrium():
    x = np.linspace(0.0, 1.0, 5)
    q = _profile(x, np.full_like(x, 2.0), "1", "q")
    density = _profile(x, np.full_like(x, 1.0e19), "m**-3", "density")
    pressure = _profile(x, 1.0e3 * (1.0 - x), "Pa", "pressure")
    theta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)

    solver = DESCSolver(
        LCFS={
            "R": 3.0 + np.cos(theta),
            "Z": np.sin(theta),
            "theta": theta,
        },
        Baxis=1.0,
        q=q,
        density=density,
        pressure=pressure,
        m_max=4,
        boundary_m_max=1,
        desc_resolution=(2, 2),
        mag_resolution=(7, 32),
        psi_domain=(0.05, 0.95),
    )
    return solver.solve(
        psi_tor_guess=np.pi,
        max_psi_iterations=1,
        rtol=0.1,
        fast=True,
        maxiter=5,
        verbose=0,
    )


def test_desc_solver_matches_axis_field(desc_equilibrium):
    relative_error = abs(desc_equilibrium.Baxis - desc_equilibrium.Baxis_target)
    relative_error /= desc_equilibrium.Baxis_target
    assert relative_error < 0.1


def test_desc_solver_qprof_covers_full_domain(desc_equilibrium):
    q = desc_equilibrium.qprof.data.sel(mpol=0, ntor=0).real
    np.testing.assert_allclose(q.x.values[[0, -1]], [0.0, 1.0])
    assert np.all(np.isfinite(q.interp(x=[0.0, 1.0]).values))


def test_desc_solver_hprof_covers_full_domain(desc_equilibrium):
    h = desc_equilibrium.hprof.data.sel(mpol=0, ntor=0).real
    np.testing.assert_allclose(h.x.values[[0, -1]], [0.0, 1.0])
    assert np.all(np.isfinite(h.interp(x=[0.0, 1.0]).values))


def test_desc_solver_builds_complete_boozer_magspace(desc_equilibrium):
    required = [
        "R", "z", "B0", "jac", "jac_from_desc_sqrtg", "q", "G", "I",
        "h", "rho_tor", "psi_pol", "theta_B", "theta_DESC",
        "zeta_B", "gpsi_psi", "gpsi_theta", "gpsi_zeta", "gtheta_theta",
        "gtheta_zeta", "gzeta_zeta", "kappa_Psi", "kappa_s", "J_B",
        "J_diag", "shear", "divPsiVec", "pressure", "density",
    ]
    for name in required:
        assert name in desc_equilibrium.magspace
        assert np.all(np.isfinite(desc_equilibrium.magspace[name].values)), name

    theta = desc_equilibrium.magspace.theta.values
    expected_theta = np.broadcast_to(theta[None, :], desc_equilibrium.magspace.theta_B.shape)
    np.testing.assert_allclose(desc_equilibrium.magspace.theta_B.values, expected_theta)
    assert np.all(np.diff(desc_equilibrium.magspace.rho_tor.values) > 0.0)


def test_desc_solver_metric_and_flux_identities(desc_equilibrium):
    mag = desc_equilibrium.magspace
    np.testing.assert_allclose(mag.gpsi_theta, mag.gtheta_psi, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(mag.gpsi_zeta, mag.gzeta_psi, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(mag.gtheta_zeta, mag.gzeta_theta, rtol=1.0e-12, atol=1.0e-12)
    assert "I_boozer" not in mag
    assert "I_legacy_2pi" not in mag
    np.testing.assert_allclose(mag.h, mag.q * mag.G + mag.I)
    np.testing.assert_allclose(mag.jac, mag.h / mag.B0**2)
    np.testing.assert_allclose(
        mag.jac_from_desc_sqrtg,
        mag.jac,
        rtol=3.0e-2,
        atol=1.0e-8,
    )


def test_desc_negative_q_reflection_reindexes_fields_and_cross_metrics():
    base = np.arange(8, dtype=float).reshape(2, 4)
    data = {"scalar": base.copy(), "q": -np.ones_like(base)}
    metric = np.zeros((2, 4, 3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            metric[..., i, j] = base + 10.0 * i + j
    shear = base + 100.0
    div_psi = base + 200.0
    indices = np.array([0, 3, 2, 1])

    reflected, reflected_metric, reflected_shear, reflected_div = (
        _reflect_boozer_surface_arrays(
            data,
            metric,
            shear,
            div_psi,
            -1,
        )
    )

    np.testing.assert_allclose(reflected["scalar"], base[:, indices])
    np.testing.assert_allclose(reflected_shear, shear[:, indices])
    np.testing.assert_allclose(reflected_div, div_psi[:, indices])
    np.testing.assert_allclose(
        reflected_metric[..., 0, 0], metric[:, indices, 0, 0]
    )
    np.testing.assert_allclose(
        reflected_metric[..., 0, 1], -metric[:, indices, 0, 1]
    )
    np.testing.assert_allclose(
        reflected_metric[..., 1, 2], -metric[:, indices, 1, 2]
    )
    np.testing.assert_allclose(
        reflected_metric[..., 1, 1], metric[:, indices, 1, 1]
    )


def test_desc_solver_local_shear_identity(desc_equilibrium):
    rho = desc_equilibrium.magspace.rho_tor.values
    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        desc_equilibrium.magspace.sizes["theta"],
        endpoint=False,
    )
    grid = LinearGrid(rho=rho, theta=theta, zeta=[0.0], NFP=1)
    eq = desc_equilibrium.desc_equilibrium
    data = eq.compute(
        [
            "B",
            "e^rho",
            "g^rr",
            "(B*grad) grad(rho)",
            "chi_r",
        ],
        grid=grid,
        basis="xyz",
    )
    # DESC 0.17.2 cannot convert tensor outputs to an xyz basis.
    grad_B = eq.compute("grad(B)", grid=grid)["grad(B)"]

    f_r = data["chi_r"]
    grad_f = f_r[:, None] * data["e^rho"]
    grad_f_dot_grad_B = np.einsum("...i,...ij->...j", grad_f, grad_B)
    B_dot_grad_grad_f = f_r[:, None] * data["(B*grad) grad(rho)"]
    A = np.cross(data["B"], grad_f)
    direct = np.einsum(
        "...i,...i->...",
        A,
        grad_f_dot_grad_B - B_dot_grad_grad_f,
    ) / (f_r**2 * data["g^rr"]) ** 2
    shear = _local_magnetic_shear_desc(eq, grid)

    # NOVA's local-shear convention has the opposite sign to this direct
    # Cartesian s dot curl(s) identity.
    np.testing.assert_allclose(shear, -direct, rtol=1.0e-6, atol=1.0e-10)


def test_desc_solver_kappa_s_boozer_identity(desc_equilibrium):
    mag = desc_equilibrium.magspace
    expected = (
        mag.G.values[:, None]
        * mag.B0.values
        / mag.h.values[:, None]
        * _dtheta(mag.B0.values)
    )
    scale = max(float(np.max(np.abs(mag.kappa_s.values))), 1.0e-12)
    relative_error = np.max(np.abs(mag.kappa_s.values - expected)) / scale
    assert relative_error < 0.2


def test_desc_solver_inherited_profile_interface(desc_equilibrium):
    names = [
        "q", "h", "gpsi_psi", "kappa_Psi", "kappa_s", "J_B", "J_diag",
        "shear", "divPsiVec",
    ]
    profiles = desc_equilibrium(*names)
    assert len(profiles) == len(names)
    for profile in profiles:
        assert isinstance(profile, Profile)
        assert np.all(np.isfinite(profile.data.values))
    np.testing.assert_array_equal(profiles[0].m_pol, [0])
    assert profiles[2].m_pol.size > 1


def test_desc_solver_constructs_continuum_and_full_solver_profiles(desc_equilibrium):
    continuum = superContinuum(desc_equilibrium, dense=True)
    eigenvalues = continuum.solve(
        ntor=1,
        mmax=2,
        psi_vals=np.array([0.0, 0.001, 0.5, 1.0]),
        which="slow_sound",
    )
    assert np.all(np.isfinite(eigenvalues))
    radial = BSplineFE(
        knots=np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]),
        order=3,
        domain=(0.0, 1.0),
        quad_order=3,
    )
    solver = superSolver(desc_equilibrium, continuum, radial)
    for profile in [solver.C11, solver.C22, solver.D11, solver.D21_1, solver.F22]:
        assert np.all(np.isfinite(profile.data.values))


def test_desc_solver_summary(desc_equilibrium):
    summary = desc_equilibrium.solution_summary()
    assert summary["kind"] == "DESCSolver"
    assert summary["magnetic_grid"]["theta_coordinate"] == "theta_B"
    assert "kappa_s" in summary["available_magspace_variables"]
    assert summary["jacobian_check"]["max_relative_error"] < 3.0e-2
