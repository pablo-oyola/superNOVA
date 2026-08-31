"""Dense references for the NOVA angular-operator convention."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import coo_array, csr_matrix

from supernova.core._solver_kernel2 import _build_static_blocks
from supernova.core._solver_kernel2 import build_M_at, solver_blocks2
from supernova.core.solver3 import _build_local_M_split
from supernova.core.solver4 import _project_local_tensor_mixed


def _amplitudes(a0, a1):
    """Return one radial slice of one-sided m=0,1 amplitudes."""
    out = np.zeros((1, 2, 1), dtype=np.complex128)
    out[0, 0, 0] = a0
    out[0, 1, 0] = a1
    return out


def _multiplication_matrix(a0, a1):
    """Exact two-mode Toeplitz matrix implied by one-sided amplitudes."""
    return np.array(
        [
            [a0, np.conjugate(a1) / 2.0],
            [a1 / 2.0, a0],
        ],
        dtype=np.complex128,
    )


def test_static_blocks_match_dense_g_gh_and_sigma_placements():
    m_range = np.array([2, 3], dtype=np.int32)
    ntor = 2
    q = 1.4
    current_I = 0.4
    covariant_g = 2.3
    h = 1.7
    pressure = 0.5
    gamma_s = 5.0 / 3.0
    zeros = _amplitudes(0.0, 0.0)

    c12_factor = _amplitudes(1.1, -0.30 + 0.10j)
    operator_O = _amplitudes(0.7, 0.20 + 0.15j)
    d11_profile = _amplitudes(0.9, 0.25 - 0.35j)
    kappa_s = _amplitudes(0.15, 0.12 + 0.08j)
    curvature_K = _amplitudes(0.4, -0.18 + 0.11j)
    compressibility_M = _amplitudes(0.6, 0.21 - 0.09j)
    inverse_B2 = _amplitudes(0.8, 0.13 + 0.17j)
    f12_left = _amplitudes(0.5, -0.14 + 0.19j)
    f12_right = _amplitudes(-0.2, 0.16 + 0.07j)
    metric_theta = _amplitudes(0.3, 0.11 - 0.05j)
    metric_zeta = _amplitudes(-0.4, 0.06 + 0.09j)

    C0, _, D0, D1, F, phase_theta, phase_zeta = _build_static_blocks(
        C11=zeros,
        O=operator_O,
        C12_2=c12_factor,
        C12_3=zeros,
        C22=zeros,
        D11=d11_profile,
        D12=zeros,
        kappa_s=kappa_s,
        K=curvature_K,
        M=compressibility_M,
        F11_2=inverse_B2,
        F12_1=f12_left,
        F12_2=f12_right,
        F21=zeros,
        F22=zeros,
        hprof=np.array([h], dtype=np.complex128),
        Iprof=np.array([current_I], dtype=np.complex128),
        gprof=np.array([covariant_g], dtype=np.complex128),
        qprof=np.array([q], dtype=np.complex128),
        pressure=np.array([pressure], dtype=np.complex128),
        pressure_deriv=np.zeros(1, dtype=np.complex128),
        density=np.ones(1, dtype=np.complex128),
        gamma_s=gamma_s,
        one_gpp=zeros,
        gtp_over_gpp=metric_theta,
        gpz_over_gpp=metric_zeta,
        ntor=ntor,
        m_range=m_range,
        dm_max=1,
        dn_max=0,
    )

    nmode = m_range.size
    G = np.diag(1j * (ntor * q - m_range))
    Sigma = np.diag(-1j * (covariant_g * m_range + current_I * ntor))
    C2 = _multiplication_matrix(1.1, -0.30 + 0.10j)
    O = _multiplication_matrix(0.7, 0.20 + 0.15j)
    D11 = _multiplication_matrix(0.9, 0.25 - 0.35j)
    Kappa = _multiplication_matrix(0.15, 0.12 + 0.08j)
    K = _multiplication_matrix(0.4, -0.18 + 0.11j)
    M = _multiplication_matrix(0.6, 0.21 - 0.09j)
    InvB2 = _multiplication_matrix(0.8, 0.13 + 0.17j)
    F12_left = _multiplication_matrix(0.5, -0.14 + 0.19j)
    F12_right = _multiplication_matrix(-0.2, 0.16 + 0.07j)
    MetricTheta = _multiplication_matrix(0.3, 0.11 - 0.05j)
    MetricZeta = _multiplication_matrix(-0.4, 0.06 + 0.09j)

    np.testing.assert_allclose(
        C0[0, :nmode, nmode:],
        -C2 @ G.conj().T @ O @ G,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(D0[0, :nmode, :nmode], D11 @ G, atol=2.0e-13)
    np.testing.assert_allclose(
        D0[0, nmode:, :nmode],
        2.0 * Kappa - Sigma / h,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        F[0, :nmode, :nmode],
        -K + InvB2 @ Sigma,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        F[0, :nmode, nmode:],
        G @ F12_left + F12_right @ G,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        D1[0],
        -C2 @ (gamma_s * pressure * G.conj().T @ M @ G),
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        phase_theta[0],
        MetricTheta @ np.diag(-1j * m_range),
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        phase_zeta[0],
        MetricZeta @ np.diag(1j * np.full_like(m_range, ntor)),
        atol=2.0e-13,
    )


def _zero_tensor(shape):
    return coo_array(shape, dtype=np.complex128)


def test_solver5_local_split_subtracts_direct_phase_and_recomposes_M_at():
    """The solver3 split used by solver5 must reproduce the direct sparse M(s)."""
    npsi = 1
    nmode = 2
    nxip1 = 2 * nmode
    harmonic_shape = (npsi, nmode, nmode)
    xip1_shape = (npsi, nxip1, nxip1)
    phase_theta = np.array(
        [[[0.2j, 0.1 - 0.05j], [-0.1 - 0.05j, 0.3j]]],
        dtype=np.complex128,
    )
    phase_zeta = np.array(
        [[[-0.4j, 0.03 + 0.02j], [-0.03 + 0.02j, -0.5j]]],
        dtype=np.complex128,
    )
    phase = phase_theta + phase_zeta
    harmonic_identity = coo_array(
        np.eye(nmode, dtype=np.complex128)[None, :, :]
    )
    xip1_identity = coo_array(
        np.eye(nxip1, dtype=np.complex128)[None, :, :]
    )

    sb = SimpleNamespace(
        m_range=np.array([2, 3], dtype=np.int32),
        density=np.array([2.0], dtype=np.complex128),
        C0=_zero_tensor(xip1_shape),
        C1=0.4 * harmonic_identity,
        D0=_zero_tensor(xip1_shape),
        D1=0.6 * harmonic_identity,
        F=0.8 * xip1_identity,
        A=1.2 * xip1_identity,
        B=1.4 * xip1_identity,
        gtp_over_gpp=coo_array(phase_theta),
        gpz_over_gpp=coo_array(phase_zeta),
        dpsi_hat_dx=0.25,
        phi_q=np.ones((1, 1), dtype=np.float64),
        weights_q=np.ones(1, dtype=np.float64),
        WD=csr_matrix([[7.0]]),
    )

    M0, M1, Mminus1 = _build_local_M_split(sb, regularizer=0.0)
    local = M0.toarray()
    np.testing.assert_allclose(local[0, :nmode, :nmode], -phase[0])
    np.testing.assert_allclose(local[0, nmode:nxip1, nmode:nxip1], -phase[0])

    s = 0.8
    direct = build_M_at(sb, omega=np.sqrt(s), regularizer=0.0).toarray()
    split = (
        M0.toarray()
        + s * M1.toarray()
        + (1.0 / s) * Mminus1.toarray()
    )
    np.testing.assert_allclose(direct, split, atol=1.0e-14)


def _radial_projection_blocks(dpsi_hat_dx):
    return SimpleNamespace(
        m_range=np.array([2], dtype=np.int32),
        dpsi_hat_dx=dpsi_hat_dx,
    )


def _unit_mixed_bases():
    """One quadrature point, one active basis function in each space.

    The mixed layout needs at least two high-order degrees of freedom, so the
    second one is present but everywhere zero.  Trial columns then land at
    ``_TRIAL_COLUMN`` per field: p1 and xi^psi in the high space, xi^s and
    div_xi in the low space.
    """
    return (
        np.array([[1.0, 0.0]], dtype=np.float64),   # phi_high_q, n_high = 2
        np.ones((1, 1), dtype=np.float64),          # phi_low_q,  n_low  = 1
        np.ones(1, dtype=np.float64),               # weights_q
        csr_matrix([[7.0, 0.0]]),                   # G_low_high
    )


# field -> trial column, for n_high = 2, n_low = 1, one poloidal mode
_TRIAL_COLUMN = (0, 2, 4, 5)


def test_radial_chain_factor_scales_only_differential_equation_rows():
    beta = 0.25
    local = np.arange(1.0, 17.0, dtype=np.complex128).reshape(1, 4, 4)
    tensor = coo_array(local)
    blocks = _radial_projection_blocks(beta)

    scaled = -local[0].copy()
    scaled[:2, :] *= beta          # only the two differential-equation rows

    expected = np.zeros((4, 6), dtype=np.complex128)
    for column, trial in enumerate(_TRIAL_COLUMN):
        expected[:, trial] = scaled[:, column]
    expected[0, _TRIAL_COLUMN[0]] += 7.0    # weak d/dx on p1
    expected[1, _TRIAL_COLUMN[1]] += 7.0    # weak d/dx on xi^psi

    projected = _project_local_tensor_mixed(
        blocks, tensor, *_unit_mixed_bases(), include_derivative=True
    )

    np.testing.assert_allclose(projected.toarray(), expected)


@pytest.mark.parametrize(
    "bad_factor",
    [0.0, -1.0, np.nan, np.inf, 1.0j, np.array([1.0, 2.0])],
)
def test_radial_chain_factor_rejects_invalid_values(bad_factor):
    blocks = _radial_projection_blocks(bad_factor)
    tensor = coo_array(np.eye(4, dtype=np.complex128)[None, :, :])

    with pytest.raises(ValueError, match="dpsi_hat_dx"):
        _project_local_tensor_mixed(
            blocks, tensor, *_unit_mixed_bases(), include_derivative=True
        )


def test_solver_blocks_roundtrip_preserves_radial_chain_factor():
    harmonic = _zero_tensor((1, 1, 1))
    xip1 = _zero_tensor((1, 2, 2))
    blocks = solver_blocks2(
        m_range=np.array([2], dtype=np.int32),
        max_dm=0,
        ntor=1,
        C0=xip1,
        C1=harmonic,
        D0=xip1,
        D1=harmonic,
        F=xip1,
        gtp_over_gpp=harmonic,
        gpz_over_gpp=harmonic,
        hprof=np.ones(1),
        Iprof=np.ones(1),
        gprof=np.ones(1),
        qprof=np.ones(1),
        pressure=np.ones(1),
        pressure_deriv=np.ones(1),
        density=np.ones(1),
        A=xip1,
        B=xip1,
        weights_q=np.ones(1),
        points_q=np.array([0.5]),
        phi_q=np.ones((1, 1)),
        WD=csr_matrix([[1.0]]),
        dpsi_hat_dx=0.25,
    )

    with pytest.warns(UserWarning, match="Duplicate dimension names"):
        dataset = blocks.to_xarray()
    restored = solver_blocks2.from_xarray(dataset)
    assert restored.dpsi_hat_dx == pytest.approx(0.25)

    dataset.attrs.pop("dpsi_hat_dx")
    legacy_restored = solver_blocks2.from_xarray(dataset)
    assert legacy_restored.dpsi_hat_dx == pytest.approx(1.0)
