import numpy as np
import pytest
import xarray as xr

from supernova.core._continuum_kernel import (
    _fill_block_kernel,
    _fill_convolution_block_kernel,
    get_harmonic_value,
)
from supernova.radial._harmonics_kernels import decompose2d
from supernova.radial.profile import ProfileFactoryPsiThetaZeta


def test_decompose2d_fft_and_profile_factory_recover_same_components():
    psi = np.array([0.2, 0.7])
    theta0 = 0.173
    theta = theta0 + 2.0 * np.pi * np.arange(64) / 64
    fields = np.stack(
        [
            (
                1.2
                + 2.5 * np.cos(theta)
                - 0.75 * np.sin(theta)
                + 4.0 * np.cos(3.0 * theta)
                + 1.25 * np.sin(3.0 * theta)
            ),
            -0.4 + 1.1 * np.cos(2.0 * theta) + 2.2 * np.sin(2.0 * theta),
        ]
    )
    expected_cos = np.array(
        [
            [1.2, 2.5, 0.0, 4.0],
            [-0.4, 0.0, 1.1, 0.0],
        ]
    )
    expected_sin = np.array(
        [
            [0.0, -0.75, 0.0, 1.25],
            [0.0, 0.0, 2.2, 0.0],
        ]
    )

    coeffs_cos, coeffs_sin = decompose2d(fields, m_max=3, angle=theta)
    np.testing.assert_allclose(coeffs_cos, expected_cos, atol=2.0e-14)
    np.testing.assert_allclose(coeffs_sin, expected_sin, atol=2.0e-14)

    field = xr.DataArray(
        fields,
        dims=("psi", "theta"),
        coords={"psi": psi, "theta": theta},
    )
    (profile,) = ProfileFactoryPsiThetaZeta(
        field,
        ntormax=0,
        mpolmax=3,
    )
    expected_profile = expected_cos + 1j * expected_sin
    np.testing.assert_allclose(
        profile.data.sel(ntor=0).values,
        expected_profile,
        atol=2.0e-14,
    )


def test_decompose2d_fft_accepts_duplicated_periodic_endpoint():
    psi = np.array([0.35])
    theta = np.linspace(0.0, 2.0 * np.pi, 64)
    field_values = (
        0.75
        + 2.25 * np.cos(2.0 * theta)
        - 1.5 * np.sin(2.0 * theta)
    )[None, :]

    coeffs_cos, coeffs_sin = decompose2d(
        field_values,
        m_max=2,
        angle=theta,
    )
    np.testing.assert_allclose(coeffs_cos, [[0.75, 0.0, 2.25]], atol=2.0e-14)
    np.testing.assert_allclose(coeffs_sin, [[0.0, 0.0, -1.5]], atol=2.0e-14)

    field = xr.DataArray(
        field_values,
        dims=("psi", "theta"),
        coords={"psi": psi, "theta": theta},
    )
    (profile,) = ProfileFactoryPsiThetaZeta(
        field,
        ntormax=0,
    )
    assert profile.m_pol[-1] == 31
    np.testing.assert_allclose(
        profile.data.sel(ntor=0, mpol=[0, 1, 2]).values,
        [[0.75 + 0.0j, 0.0 + 0.0j, 2.25 - 1.5j]],
        atol=2.0e-14,
    )


def test_decompose2d_keeps_nonuniform_grid_compatibility():
    uniform_parameter = 2.0 * np.pi * np.arange(257) / 257
    theta = uniform_parameter + 0.08 * np.sin(uniform_parameter)
    field = (
        1.1
        + 2.0 * np.cos(theta)
        + 0.5 * np.sin(theta)
        - 1.25 * np.cos(3.0 * theta)
        + 0.75 * np.sin(3.0 * theta)
    )[None, :]

    coeffs_cos, coeffs_sin = decompose2d(field, m_max=3, angle=theta)

    np.testing.assert_allclose(
        coeffs_cos,
        [[1.1, 2.0, 0.0, -1.25]],
        atol=5.0e-4,
    )
    np.testing.assert_allclose(
        coeffs_sin,
        [[0.0, 0.5, 0.0, 0.75]],
        atol=5.0e-4,
    )


def test_profile_factory_accepts_radial_only_psi_fields():
    psi = np.linspace(0.0, 1.0, 5)
    q = xr.DataArray(
        1.0 + psi,
        dims=["psi"],
        coords={"psi": psi},
        attrs={"units": "1", "description": "safety factor"},
    )

    (q_prof,) = ProfileFactoryPsiThetaZeta(q, ntormax=3, mpolmax=4)

    np.testing.assert_array_equal(q_prof.n_tor, [0])
    np.testing.assert_array_equal(q_prof.m_pol, [0])
    np.testing.assert_allclose(q_prof.x, psi)
    np.testing.assert_allclose(q_prof.data[:, 0, 0].values.real, q.values)
    assert q_prof.data.attrs["units"] == "1"


def test_profile_factory_mixes_radial_only_and_angular_fields_in_order():
    psi = np.linspace(0.0, 1.0, 5)
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    q = xr.DataArray(1.0 + psi, dims=["psi"], coords={"psi": psi})
    metric = xr.DataArray(
        2.0 + psi[:, None] + 4.0 * np.cos(theta)[None, :],
        dims=["psi", "theta"],
        coords={"psi": psi, "theta": theta},
    )
    data = xr.Dataset({"q": q, "metric": metric})

    q_prof, metric_prof = ProfileFactoryPsiThetaZeta(data, ntormax=0, mpolmax=1)

    np.testing.assert_array_equal(q_prof.n_tor, [0])
    np.testing.assert_array_equal(q_prof.m_pol, [0])
    np.testing.assert_allclose(q_prof.data[:, 0, 0].values.real, q.values)

    np.testing.assert_array_equal(metric_prof.n_tor, [0])
    np.testing.assert_array_equal(metric_prof.m_pol, [0, 1])
    np.testing.assert_allclose(
        metric_prof.data.sel(mpol=0, ntor=0).values.real,
        2.0 + psi,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        metric_prof.data.sel(mpol=1, ntor=0).values,
        4.0 + 0.0j,
        atol=1e-12,
    )


def test_poloidal_sine_profile_has_required_phase_and_toeplitz_orientation():
    """Store a+ib but use half-amplitudes in the two-sided Toeplitz matrix."""
    psi = np.array([0.4])
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    field = xr.DataArray(
        1.5
        + 4.0 * np.cos(theta)[None, :]
        + 6.0 * np.sin(theta)[None, :],
        dims=("psi", "theta"),
        coords={"psi": psi, "theta": theta},
    )

    (profile,) = ProfileFactoryPsiThetaZeta(
        field,
        ntormax=0,
        mpolmax=1,
    )

    amplitude = profile.data.sel(x=psi[0], mpol=1, ntor=0).item()
    np.testing.assert_allclose(amplitude, 4.0 + 6.0j, atol=1.0e-13)
    reconstructed = 1.5 + np.real(amplitude * np.exp(-1j * theta))
    np.testing.assert_allclose(reconstructed, field.values[0], atol=1.0e-13)

    modes = np.array([9, 10, 11], dtype=np.int32)
    toroidal_modes = np.zeros_like(modes)
    matrix = np.zeros((modes.size, modes.size), dtype=np.complex128)
    _fill_block_kernel(
        np.asarray(profile.data.isel(x=0)),
        modes,
        toroidal_modes,
        matrix,
    )

    np.testing.assert_allclose(np.diag(matrix), 1.5, atol=1.0e-13)
    np.testing.assert_allclose(np.diag(matrix, k=-1), amplitude / 2.0, atol=1.0e-13)
    np.testing.assert_allclose(
        np.diag(matrix, k=1),
        np.conjugate(amplitude) / 2.0,
        atol=1.0e-13,
    )


def test_toroidal_only_sine_uses_positive_zeta_phase():
    psi = np.array([0.3])
    zeta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    field = xr.DataArray(
        2.0 + 3.0 * np.cos(zeta)[None, :] + 5.0 * np.sin(zeta)[None, :],
        dims=("psi", "zeta"),
        coords={"psi": psi, "zeta": zeta},
    )

    (profile,) = ProfileFactoryPsiThetaZeta(field, mpolmax=0, ntormax=1)

    amplitude = profile.data.sel(x=psi[0], mpol=0, ntor=1).item()
    np.testing.assert_allclose(amplitude, 3.0 - 5.0j, atol=1.0e-13)
    reconstructed = 2.0 + np.real(amplitude * np.exp(1j * zeta))
    np.testing.assert_allclose(reconstructed, field.values[0], atol=1.0e-13)


def test_even_grid_poloidal_nyquist_mode_is_not_doubled():
    psi = np.array([0.3])
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    nyquist = theta.size // 2
    field = xr.DataArray(
        1.25 + 2.5 * np.cos(nyquist * theta)[None, :],
        dims=("psi", "theta"),
        coords={"psi": psi, "theta": theta},
    )

    (profile,) = ProfileFactoryPsiThetaZeta(
        field,
        ntormax=0,
        mpolmax=nyquist,
    )
    amplitude = profile.data.sel(x=psi[0], mpol=nyquist, ntor=0).item()
    np.testing.assert_allclose(amplitude, 2.5 + 0.0j, atol=1.0e-13)
    reconstructed = 1.25 + np.real(amplitude * np.exp(-1j * nyquist * theta))
    np.testing.assert_allclose(reconstructed, field.values[0], atol=1.0e-13)


def test_three_dimensional_corner_nyquist_mode_is_not_doubled():
    psi = np.array([0.6])
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False)
    mpol = theta.size // 2
    ntor = zeta.size // 2
    phase = mpol * theta[:, None] - ntor * zeta[None, :]
    field = xr.DataArray(
        0.75 + 3.5 * np.cos(phase)[None, :, :],
        dims=("psi", "theta", "zeta"),
        coords={"psi": psi, "theta": theta, "zeta": zeta},
    )

    (profile,) = ProfileFactoryPsiThetaZeta(
        field,
        mpolmax=mpol,
        ntormax=ntor,
    )
    amplitude = profile.data.sel(x=psi[0], mpol=mpol, ntor=ntor).item()
    np.testing.assert_allclose(amplitude, 3.5 + 0.0j, atol=2.0e-13)
    reconstructed = 0.75 + np.real(amplitude * np.exp(-1j * phase))
    np.testing.assert_allclose(reconstructed, field.values[0], atol=2.0e-13)


def test_three_dimensional_helical_mode_round_trip():
    psi = np.array([0.2])
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    mpol = 2
    ntor = 3
    phase = mpol * theta[:, None] - ntor * zeta[None, :]
    field = xr.DataArray(
        1.25 + 3.0 * np.cos(phase)[None, :, :] + 5.0 * np.sin(phase)[None, :, :],
        dims=("psi", "theta", "zeta"),
        coords={"psi": psi, "theta": theta, "zeta": zeta},
    )

    (profile,) = ProfileFactoryPsiThetaZeta(
        field,
        mpolmax=mpol,
        ntormax=ntor,
    )

    amplitude = profile.data.sel(x=psi[0], mpol=mpol, ntor=ntor).item()
    np.testing.assert_allclose(amplitude, 3.0 + 5.0j, atol=2.0e-13)
    reconstructed = 1.25 + np.real(amplitude * np.exp(-1j * phase))
    np.testing.assert_allclose(reconstructed, field.values[0], atol=2.0e-13)


def test_three_dimensional_mixed_sign_helicity_is_rejected():
    psi = np.array([0.2])
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    phase = 2 * theta[:, None] + 3 * zeta[None, :]
    field = xr.DataArray(
        1.0 + 0.4 * np.cos(phase)[None, :, :] + 0.7 * np.sin(phase)[None, :, :],
        dims=("psi", "theta", "zeta"),
        coords={"psi": psi, "theta": theta, "zeta": zeta},
    )

    with pytest.raises(
        ValueError,
        match=r"unsupported mixed-sign Fourier content.*\(m, n\)=\(2, 3\)",
    ):
        ProfileFactoryPsiThetaZeta(field, mpolmax=2, ntormax=3)


def test_harmonic_lookup_and_convolution_convert_amplitudes_once():
    table = np.zeros((3, 1), dtype=np.complex128)
    table[0, 0] = 1.5
    table[1, 0] = 4.0 + 6.0j

    np.testing.assert_allclose(get_harmonic_value(table, 0, 0), 1.5)
    np.testing.assert_allclose(get_harmonic_value(table, 1, 0), 2.0 + 3.0j)
    np.testing.assert_allclose(get_harmonic_value(table, -1, 0), 2.0 - 3.0j)

    cosine = np.zeros((3, 1), dtype=np.complex128)
    sine = np.zeros((3, 1), dtype=np.complex128)
    cosine[1, 0] = 2.0
    sine[1, 0] = 3.0j
    modes = np.arange(3, dtype=np.int32)
    toroidal_modes = np.zeros_like(modes)
    product_matrix = np.zeros((modes.size, modes.size), dtype=np.complex128)
    _fill_convolution_block_kernel(
        cosine,
        sine,
        modes,
        toroidal_modes,
        product_matrix,
    )

    # 2*cos(theta) * 3*sin(theta) = 3*sin(2*theta), whose stored
    # one-sided amplitude is 3j and whose two-sided +m coefficient is 1.5j.
    np.testing.assert_allclose(np.diag(product_matrix), 0.0, atol=1.0e-14)
    np.testing.assert_allclose(product_matrix[2, 0], 1.5j, atol=1.0e-14)
    np.testing.assert_allclose(product_matrix[0, 2], -1.5j, atol=1.0e-14)
