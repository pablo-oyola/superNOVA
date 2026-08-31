"""End-to-end acceptance on the analytical ITPA TAE benchmark.

The unit tests elsewhere pin pieces: kernel hashes, projection checksums,
boundary ranks.  None of them proves the assembled physics is unchanged.  This
one runs the whole pipeline -- analytical equilibrium, continuum, block
assembly, mixed projection, boundary reduction, eigensolve -- and compares the
result against values measured on superNOVA_2 at f8b246e with the same inputs.

The reference values below were produced by running this configuration in both
trees and comparing: frequencies agreed to 6e-16 relative and the continuum
checksums were bit-identical.  ITPA is a good choice for this because it is
analytical (no eqdsk, about a second end to end) and because ``gamma_s = 0``
exercises the generalized-linear backend rather than the polynomial one.
"""

from __future__ import annotations

import numpy as np
import pytest
import unyt

import supernova as sn
from supernova.cases.itpa import ITPAEq

pytest.importorskip("slepc4py")

# The published benchmark configuration.
SPECIES, GAMMA_S, NTOR, MMAX = "H", 0.0, 6, 20
NFE, ORDER, QUAD_ORDER = 100, 3, 6
NS, NTHETA = 256, 256
MAX_DM, REGULARIZER, BC_KIND = 5, 1.0e-5, "full"
NEV, TOL, MAXIT = 4, 1.0e-7, 80
FREQUENCY_WINDOW_KHZ = (60.0, 76.0)

# Measured on superNOVA_2 at f8b246e.
REFERENCE_FREQUENCY_KHZ = (
    69.95411566599043,
    70.14986464493438,
    70.33584493960305,
    70.65894649445084,
)
REFERENCE_CONTINUUM_CHECKSUM = 683994.8618455227
REFERENCE_XI_PSI_ABS_SUM = 0.2967674146631284

# MUMPS runs out of its estimated real workspace on this pencil under SLEPc
# 3.25 (INFOG(1) = -9).  ICNTL(14) is the percentage headroom and the default
# 20 is not enough.  superNOVA_2 fails identically here, so this is the
# environment rather than anything the consolidation changed.
PETSC_OPTIONS = {
    "st_ksp_type": "preonly",
    "st_pc_type": "lu",
    "st_pc_factor_mat_solver_type": "mumps",
    "st_mat_mumps_icntl_14": "200",
}


@pytest.fixture(scope="module")
def itpa_case():
    equilibrium = ITPAEq(aminor=1.0 * unyt.m, Rmajor=10.0 * unyt.m,
                         q0=1.71, qedge=1.87, B0=3.0 * unyt.T,
                         ns=NS, ntheta=NTHETA, m_max=MMAX, Bt_sign=1)
    continuum = sn.superContinuum(equilibrium, gamma_s=GAMMA_S,
                                  species=SPECIES, dense=True)
    return equilibrium, continuum


@pytest.fixture(scope="module")
def itpa_solution(itpa_case):
    equilibrium, continuum = itpa_case
    grid = sn.BSplineFE(knots=np.linspace(0.0, 1.0, NFE + 1), order=ORDER,
                        domain=(0.0, 1.0), quad_order=QUAD_ORDER)
    solver = sn.StabilitySolver(equilibrium, continuum, radialfe=grid)

    frequency = unyt.unyt_array(FREQUENCY_WINDOW_KHZ, "kHz",
                                registry=equilibrium.registry)
    omega_min, omega_max = 2.0 * np.pi * frequency.to("omegaA").v
    s_min, s_max = omega_min**2, omega_max**2
    center, radius = 0.5 * (s_min + s_max), 0.5 * (s_max - s_min)

    return solver.solve_pep(
        ntor=NTOR, target_omega=np.sqrt(center),
        region={"type": "ellipse", "center": center, "radius": radius,
                "vscale": 0.25},
        mmax=MMAX, max_dm=MAX_DM, regularizer=REGULARIZER,
        apply_bcs=True, bc_kind=BC_KIND, nev=NEV, tol=TOL, maxit=MAXIT,
        max_eigenpairs=NEV, compute_residuals=True, options=PETSC_OPTIONS,
    )


def _frequencies_khz(dataset, equilibrium):
    omega = np.asarray(dataset["omega"].values)
    khz = unyt.unyt_array(omega.real / (2.0 * np.pi), "omegaA",
                          registry=equilibrium.registry).to("kHz").v
    return np.sort(np.asarray(khz))


def test_gamma_zero_takes_the_generalized_linear_backend(itpa_solution):
    """gamma_s == 0 must not be solved as a degenerate quadratic."""
    assert itpa_solution.attrs["problem_type"] == "GENERALIZED_LINEAR"
    assert itpa_solution.attrs["gamma_zero_reduction"] is True


def test_itpa_frequencies_match_the_supernova_2_reference(itpa_solution, itpa_case):
    equilibrium, _ = itpa_case
    assert itpa_solution.sizes["eigenpair"] == NEV
    np.testing.assert_allclose(
        _frequencies_khz(itpa_solution, equilibrium),
        REFERENCE_FREQUENCY_KHZ, rtol=1.0e-10,
        err_msg="The assembled physics moved. The unit checksums only prove "
                "nothing changed by accident; this is the test that proves "
                "the eigenvalues did not.",
    )


def test_itpa_roots_are_real_to_solver_tolerance(itpa_solution):
    """An ideal-MHD stable case should return real frequencies."""
    omega = np.asarray(itpa_solution["omega"].values)
    np.testing.assert_array_less(np.abs(omega.imag) / np.abs(omega.real), 1.0e-4)


def test_itpa_eigenvectors_match_the_reference(itpa_solution):
    total = float(np.abs(np.asarray(
        itpa_solution["xi_psi_coefficient"].values)).sum())
    assert total == pytest.approx(REFERENCE_XI_PSI_ABS_SUM, rel=1.0e-9)


def test_itpa_continuum_matches_the_reference(itpa_case):
    """The continuum is assembled independently of the solver pipeline."""
    equilibrium, continuum = itpa_case
    eigenvalues = continuum.solve(
        ntor=NTOR, which="slow_sound", normalize_eigvals=True,
        psi_vals=np.linspace(0.0, 1.0, 256), classify=False, mmax=MMAX,
    )
    khz = np.sqrt(eigenvalues.real).to("kHz").v / (2.0 * np.pi)
    assert float(np.nansum(np.abs(khz))) == pytest.approx(
        REFERENCE_CONTINUUM_CHECKSUM, rel=1.0e-12)
