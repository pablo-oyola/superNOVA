"""Focused tests for the solver4 compatible mixed projection."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import coo_array, csr_matrix, isspmatrix_csr

from supernova.core.solver4 import (
    MixedFieldLayout,
    _build_mixed_radial_data,
    _project_local_tensor_mixed,
)
from supernova.radial import BSplineFE


def _high_space():
    return BSplineFE(
        knots=np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]),
        order=3,
        domain=(0.0, 1.0),
        quad_order=5,
    )


def _dense_oracle(M, mixed, n_modes, include_derivative):
    n_high = mixed.pair.high.get_ndofs()
    n_low = mixed.pair.low.get_ndofs()
    layout = MixedFieldLayout(n_high, n_low, n_modes)
    out = np.zeros((layout.n_rows, layout.n_trial), dtype=np.complex128)

    for q in range(M.shape[0]):
        for local_row in range(4 * n_modes):
            row_field, row_mode = divmod(local_row, n_modes)
            for local_col in range(4 * n_modes):
                value = M[q, local_row, local_col]
                if value == 0.0:
                    continue
                col_field, col_mode = divmod(local_col, n_modes)
                trial_basis = (
                    mixed.phi_high_q if col_field < 2 else mixed.phi_low_q
                )
                for radial_row in range(n_low):
                    for radial_col in range(trial_basis.shape[1]):
                        out[
                            layout.row_index(row_field, radial_row, row_mode),
                            layout.trial_index(col_field, radial_col, col_mode),
                        ] -= (
                            mixed.weights_q[q]
                            * mixed.phi_low_q[q, radial_row]
                            * value
                            * trial_basis[q, radial_col]
                        )

    if include_derivative:
        G = mixed.G_low_high.toarray()
        for field in (0, 1):
            for mode in range(n_modes):
                for radial_row in range(n_low):
                    for radial_col in range(n_high):
                        out[
                            layout.row_index(field, radial_row, mode),
                            layout.trial_index(field, radial_col, mode),
                        ] += G[radial_row, radial_col]
    return out


def test_mixed_layout_is_field_major_and_harmonic_fast():
    layout = MixedFieldLayout(n_high=6, n_low=5, n_modes=3)

    assert layout.n_rows == 60
    assert layout.n_trial == 66
    assert layout.trial_offsets == (0, 18, 36, 51)
    assert layout.row_index(2, 4, 1) == 2 * 15 + 4 * 3 + 1
    assert layout.trial_index(0, 5, 2) == 17
    assert layout.trial_index(1, 0, 1) == 19
    assert layout.trial_index(2, 4, 2) == 50
    assert layout.trial_index(3, 4, 2) == 65


def test_mixed_layout_rejects_incompatible_dimensions():
    with pytest.raises(ValueError, match="n_low=n_high-1"):
        MixedFieldLayout(n_high=6, n_low=4, n_modes=2)


def test_mixed_radial_data_builds_exact_strong_derivative():
    mixed = _build_mixed_radial_data(_high_space())
    direct = mixed.phi_low_q.T @ (
        mixed.weights_q[:, None]
        * mixed.pair.high.eval_basis_derivative(mixed.points_q).T
    )

    assert mixed.pair.high.get_ndofs() == 4
    assert mixed.pair.low.get_ndofs() == 3
    assert isspmatrix_csr(mixed.mass_low)
    assert isspmatrix_csr(mixed.G_low_high)
    np.testing.assert_allclose(mixed.G_low_high.toarray(), direct, atol=1e-13)


@pytest.mark.parametrize("include_derivative", [False, True])
def test_mixed_projection_matches_direct_dense_quadrature(include_derivative):
    mixed = _build_mixed_radial_data(_high_space())
    n_modes = 2
    n_state = 4 * n_modes
    rng = np.random.default_rng(314159)
    dense_M = rng.normal(size=(mixed.points_q.size, n_state, n_state))
    dense_M = dense_M + 1.0j * rng.normal(size=dense_M.shape)
    dense_M[np.abs(dense_M) < 0.7] = 0.0
    sb = SimpleNamespace(m_range=np.array([-1, 2]))

    projected = _project_local_tensor_mixed(
        sb,
        coo_array(dense_M),
        mixed.phi_high_q,
        mixed.phi_low_q,
        mixed.weights_q,
        mixed.G_low_high,
        include_derivative=include_derivative,
    )
    expected = _dense_oracle(dense_M, mixed, n_modes, include_derivative)

    assert isinstance(projected, csr_matrix)
    assert projected.shape == (24, 28)
    np.testing.assert_allclose(projected.toarray(), expected, rtol=1e-13, atol=1e-13)


def test_derivative_blocks_have_positive_sign_and_only_high_trial_fields():
    mixed = _build_mixed_radial_data(_high_space())
    n_modes = 2
    n_state = 4 * n_modes
    zero_M = coo_array((mixed.points_q.size, n_state, n_state))
    sb = SimpleNamespace(m_range=np.arange(n_modes))
    layout = MixedFieldLayout(4, 3, n_modes)

    projected = _project_local_tensor_mixed(
        sb,
        zero_M,
        mixed.phi_high_q,
        mixed.phi_low_q,
        mixed.weights_q,
        mixed.G_low_high,
        include_derivative=True,
    ).toarray()
    expected = _dense_oracle(
        np.zeros(zero_M.shape, dtype=np.complex128),
        mixed,
        n_modes,
        True,
    )

    np.testing.assert_allclose(projected, expected, atol=1e-14)
    assert np.count_nonzero(projected[2 * layout.n_low * n_modes :, :]) == 0
    assert np.count_nonzero(projected[:, layout.trial_offsets[2] :]) == 0


def test_single_local_entry_uses_negative_galerkin_sign_and_mixed_offsets():
    mixed = _build_mixed_radial_data(_high_space())
    n_modes = 2
    n_state = 4 * n_modes
    dense_M = np.zeros((mixed.points_q.size, n_state, n_state), dtype=np.complex128)
    dense_M[1, 2 * n_modes + 1, 0] = 2.0 - 3.0j
    sb = SimpleNamespace(m_range=np.arange(n_modes))

    projected = _project_local_tensor_mixed(
        sb,
        coo_array(dense_M),
        mixed.phi_high_q,
        mixed.phi_low_q,
        mixed.weights_q,
        mixed.G_low_high,
    ).toarray()
    expected = _dense_oracle(dense_M, mixed, n_modes, False)

    np.testing.assert_allclose(projected, expected, atol=1e-14)
    nonzero_rows, nonzero_cols = np.nonzero(projected)
    assert np.all(nonzero_rows >= 2 * mixed.pair.low.get_ndofs() * n_modes)
    assert np.all(nonzero_cols < mixed.pair.high.get_ndofs() * n_modes)


def test_projector_validates_local_state_and_derivative_shapes():
    mixed = _build_mixed_radial_data(_high_space())
    sb = SimpleNamespace(m_range=np.array([0, 1]))

    with pytest.raises(ValueError, match="state dimensions"):
        _project_local_tensor_mixed(
            sb,
            coo_array((mixed.points_q.size, 7, 7)),
            mixed.phi_high_q,
            mixed.phi_low_q,
            mixed.weights_q,
            mixed.G_low_high,
        )

    with pytest.raises(ValueError, match="G_low_high"):
        _project_local_tensor_mixed(
            sb,
            coo_array((mixed.points_q.size, 8, 8)),
            mixed.phi_high_q,
            mixed.phi_low_q,
            mixed.weights_q,
            np.zeros((2, 2)),
        )
