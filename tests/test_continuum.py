from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import supernova.continuum as continuum_module
from supernova.continuum import _eigh_signed_definite


def test_eigh_signed_definite_preserves_common_sign_reversal():
    A = np.array(
        [[3.0, 0.25 - 0.1j], [0.25 + 0.1j, 1.0]],
        dtype=np.complex128,
    )
    B = np.array(
        [[2.0, 0.1j], [-0.1j, 1.5]],
        dtype=np.complex128,
    )

    positive = _eigh_signed_definite(A, B)
    negative = _eigh_signed_definite(-A, -B)

    np.testing.assert_allclose(negative, positive, rtol=1.0e-13, atol=1.0e-13)


def test_eigh_signed_definite_rejects_indefinite_mass_matrix():
    A = np.diag([1.0, 2.0])
    B = np.diag([1.0, -1.0])

    with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
        _eigh_signed_definite(A, B)


def test_solve_loads_harmonics_for_full_shifted_mode_difference(monkeypatch):
    radial = np.array([0.0, 1.0])
    harmonics = np.arange(31)
    profile_values = np.zeros((radial.size, harmonics.size, 1), dtype=complex)
    profile_values[:, 0, 0] = 1.0
    profile_data = xr.DataArray(
        profile_values,
        dims=("x", "mpol", "ntor"),
        coords={"x": radial, "mpol": harmonics, "ntor": [0]},
    )

    def profile():
        return SimpleNamespace(data=profile_data.copy(), m_pol=harmonics)

    continuum = continuum_module.superContinuum.__new__(
        continuum_module.superContinuum
    )
    for name in ("H", "J", "K", "L", "N", "M"):
        setattr(continuum, name, profile())
    continuum.equ = SimpleNamespace(
        qprof=SimpleNamespace(
            data=xr.DataArray([1.8, 15.7], dims="x", coords={"x": radial})
        )
    )
    continuum.p0 = xr.DataArray([1.0, 1.0], dims="x", coords={"x": radial})
    continuum.n0 = xr.DataArray([1.0, 1.0], dims="x", coords={"x": radial})
    continuum.gamma_s = 5.0 / 3.0

    captured = {}

    def fake_slow_sound(*, H_mn, J_mn, N_mn, q, m_range, ntor):
        captured["harmonic_count"] = H_mn.shape[0]
        captured["mode_range"] = m_range.copy()
        identity = np.eye(m_range.size, dtype=complex)
        return identity, identity

    monkeypatch.setattr(
        continuum_module,
        "assembly_slow_sound",
        fake_slow_sound,
    )

    continuum.solve(
        ntor=1,
        mmax=17,
        psi_vals=np.array([0.5]),
        which="slow_sound",
    )

    np.testing.assert_array_equal(captured["mode_range"], np.arange(-2, 18))
    assert captured["harmonic_count"] == 20

    def fake_full(*, H_mn, m_range, **kwargs):
        captured["build_harmonic_count"] = H_mn.shape[0]
        captured["build_mode_range"] = m_range.copy()
        identity = np.eye(2 * m_range.size, dtype=complex)
        return identity, identity

    monkeypatch.setattr(
        continuum_module,
        "assemble_full_matrices",
        fake_full,
    )
    continuum.build_AB(ntor=1, mmax=3, psi_vals=np.array([0.5]))

    np.testing.assert_array_equal(captured["build_mode_range"], np.arange(-3, 4))
    assert captured["build_harmonic_count"] == 7
