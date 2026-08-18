import numpy as np
import pytest
from scipy.constants import mu_0

from supernova.core.itpa_benchmark import ITPAEq


@pytest.mark.parametrize("Bt_sign", [1, -1])
def test_itpa_cocos1_and_positive_q_boozer_identities(Bt_sign):
    equ = ITPAEq(ns=48, ntheta=128, Bt_sign=Bt_sign)
    mag = equ.magspace

    assert "I_boozer" not in mag
    assert "I_legacy_2pi" not in mag

    theta_geom = mag.theta_geom.values
    radius = mag.R.values
    r_minor = equ._rminor[:, None]
    r_safe = np.where(r_minor == 0.0, equ._rminor[1], r_minor)
    psi_prime = equ._dpsi_dr(equ._rminor)[:, None]

    grad_psi = np.stack(
        (
            psi_prime * np.cos(theta_geom),
            np.zeros_like(radius),
            psi_prime * np.sin(theta_geom),
        ),
        axis=-1,
    )
    grad_theta_geom = np.stack(
        (
            -np.sin(theta_geom) / r_safe,
            np.zeros_like(radius),
            np.cos(theta_geom) / r_safe,
        ),
        axis=-1,
    )
    grad_theta = (
        mag.dtheta_boozer_dpsi_geom.values[..., None] * grad_psi
        + mag.dtheta_boozer_dtheta_geom.values[..., None] * grad_theta_geom
    )
    grad_zeta = np.stack(
        (np.zeros_like(radius), 1.0 / radius, np.zeros_like(radius)),
        axis=-1,
    )
    field = np.stack((mag.BR.values, mag.Bphi.values, mag.Bz.values), axis=-1)
    diamagnetic = np.cross(field, grad_psi)
    jacobian = mag.jac_magnetic.values

    # Reversing Bt reverses raw geometric q relative to the fixed poloidal
    # flux.  The solver-facing coordinate orientation keeps q positive.
    assert mag.attrs["poloidal_coordinate_orientation"] == -Bt_sign
    assert np.all(mag.jac_current.values[2:] < 0.0)
    assert np.all(Bt_sign * mag.Bphi.values[2:] > 0.0)
    assert np.all(Bt_sign * mag.F.values[2:] > 0.0)
    assert np.all(Bt_sign * mag.g.values[2:] > 0.0)
    assert np.all(Bt_sign * mag.I.values[2:] > 0.0)
    assert np.all(Bt_sign * mag.h.values[2:] > 0.0)
    # The analytic poloidal field, and therefore Ip, is unchanged; J is purely
    # toroidal here, so sign(J dot B) follows the selected Bt direction.
    assert np.all(Bt_sign * mag.J_B.values[2:] > 0.0)
    assert np.all(Bt_sign * jacobian[2:] > 0.0)
    assert np.all(
        -Bt_sign * mag.dtheta_boozer_dtheta_geom.values[2:] > 0.0
    )

    expected_br = psi_prime * np.sin(theta_geom) / radius
    expected_bz = -psi_prime * np.cos(theta_geom) / radius
    for actual, expected in (
        (mag.BR.values[2:], expected_br[2:]),
        (mag.Bz.values[2:], expected_bz[2:]),
    ):
        # Surface fields have been periodically interpolated onto reflected
        # Theta, so use a scale-aware interpolation-error bound near zeros.
        relative_max = np.max(np.abs(actual - expected)) / np.max(np.abs(expected))
        assert relative_max < 1.0e-3

    interior = np.s_[2:-2, :]

    def assert_identity(actual, expected, *, atol=5.0e-4):
        scale = max(1.0, float(np.max(np.abs(expected[interior]))))
        error = np.max(np.abs((actual - expected)[interior])) / scale
        assert error < atol

    assert_identity(
        jacobian * np.sum(field * grad_theta, axis=-1),
        np.ones_like(jacobian),
    )
    assert_identity(
        jacobian * np.sum(field * grad_zeta, axis=-1),
        np.broadcast_to(mag.q.values[:, None], jacobian.shape),
    )
    assert_identity(
        jacobian * np.sum(field * field, axis=-1),
        np.broadcast_to(mag.h.values[:, None], jacobian.shape),
    )
    assert_identity(
        jacobian * np.sum(diamagnetic * grad_theta, axis=-1),
        np.broadcast_to(mag.g.values[:, None], jacobian.shape),
    )
    assert_identity(
        jacobian * np.sum(diamagnetic * grad_zeta, axis=-1),
        -np.broadcast_to(mag.I.values[:, None], jacobian.shape),
    )

    sin_theta = np.sin(theta_geom)
    cos_theta = np.cos(theta_geom)
    Btheta = -psi_prime / radius
    btheta = Btheta / mag.B0.values
    bphi = mag.Bphi.values / mag.B0.values
    kappa_r = -(btheta**2) / r_safe - bphi**2 * cos_theta / radius
    kappa_theta = bphi**2 * sin_theta / radius
    kappa_phi = -(bphi * btheta) * sin_theta / radius
    curvature = np.stack(
        (
            kappa_r * cos_theta - kappa_theta * sin_theta,
            kappa_phi,
            kappa_r * sin_theta + kappa_theta * cos_theta,
        ),
        axis=-1,
    )
    assert_identity(
        mag.kappa_s.values,
        np.sum(curvature * diamagnetic, axis=-1),
        atol=1.0e-3,
    )


@pytest.mark.parametrize("Bt_sign", [0, 2, -2, True])
def test_itpa_rejects_invalid_toroidal_field_sign(Bt_sign):
    with pytest.raises(ValueError, match="Bt_sign must be either"):
        ITPAEq(ns=8, ntheta=16, Bt_sign=Bt_sign)


def test_itpa_local_shear_matches_current_geometry_split():
    equ = ITPAEq(ns=256, ntheta=512)
    mag = equ.magspace
    theta = mag.theta_geom.values
    r = equ._rminor[:, None]
    r_safe = np.where(r == 0.0, equ._rminor[1], r)
    psi_prime = equ._dpsi_dr(equ._rminor)[:, None]
    psi_prime_safe = np.where(
        psi_prime == 0.0,
        equ._dpsi_dr(equ._rminor)[1],
        psi_prime,
    )

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    B_R = mag.BR.values
    B_phi = mag.Bphi.values
    B_Z = mag.Bz.values
    R = mag.R.values
    B_theta = -B_R * sin_theta + B_Z * cos_theta

    B = np.stack((B_R, B_phi, B_Z), axis=-1)
    grad_psi = np.stack(
        (
            psi_prime * cos_theta,
            np.zeros_like(R),
            psi_prime * sin_theta,
        ),
        axis=-1,
    )
    B_grad_grad_psi = np.stack(
        (
            -B_theta * psi_prime / r_safe * sin_theta,
            B_phi * psi_prime / R * cos_theta,
            B_theta * psi_prime / r_safe * cos_theta,
        ),
        axis=-1,
    )
    grad_psi_sq = np.where(
        psi_prime == 0.0,
        psi_prime_safe**2,
        psi_prime**2,
    )

    # NOVA uses the negative-local-shear convention S=A.curl(A). Relative to
    # the conventional local shear, both the current and geometric pieces have
    # the opposite sign.
    shear_current = mu_0 * mag.J_B.values / grad_psi_sq
    shear_geometry = (
        -2.0
        * np.sum(np.cross(B, grad_psi) * B_grad_grad_psi, axis=-1)
        / grad_psi_sq**2
    )
    expected = shear_current + shear_geometry
    expected[0] = expected[1]

    interior = (mag.psi.values >= 0.05) & (mag.psi.values <= 0.95)
    error = mag.shear.values[interior] - expected[interior]
    relative_l2 = np.linalg.norm(error) / np.linalg.norm(expected[interior])

    assert relative_l2 < 0.02
