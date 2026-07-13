"""Structural tests for solver4 mixed-space boundary conditions."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye

from supernova.core._boundary4 import (
    BoundaryCondition,
    idealSeparatrixBC,
    regularityAxisBC,
)
from supernova.radial import BSplineFE


@pytest.fixture
def mixed_spaces():
    high = BSplineFE(
        knots=np.linspace(0.0, 1.0, 7),
        order=4,
        domain=(0.0, 1.0),
        quad_order=6,
    )
    low = BSplineFE(
        knots=high.knots[1:-1],
        order=high.order - 1,
        domain=(high.knots[0], high.knots[-1]),
        quad_order=high.quadrature_order,
    )
    return high, low


@pytest.fixture
def mixed_bcs(mixed_spaces):
    high, low = mixed_spaces
    m_numbers = np.array([-2, 0, 3])
    axis = regularityAxisBC(high, low, m_numbers)
    edge = idealSeparatrixBC(high, low, m_numbers.size)
    return high, low, m_numbers, axis, edge


def test_axis_and_edge_select_field_major_xipsi_columns(mixed_bcs):
    high, low, m_numbers, axis, edge = mixed_bcs
    n_high = high.get_ndofs()
    n_low = low.get_ndofs()
    n_modes = m_numbers.size
    n_trial = 2 * n_modes * (n_high + n_low)
    xipsi_offset = n_high * n_modes

    expected_axis = xipsi_offset + np.arange(n_modes)
    expected_edge = xipsi_offset + (n_high - 1) * n_modes + np.arange(n_modes)

    assert axis.C.shape == (n_modes, n_trial)
    assert edge.C.shape == (n_modes, n_trial)
    assert np.linalg.matrix_rank(axis.C) == n_modes
    assert np.linalg.matrix_rank(edge.C) == n_modes
    np.testing.assert_array_equal(axis.endpoint_columns, expected_axis)
    np.testing.assert_array_equal(edge.endpoint_columns, expected_edge)

    for boundary, expected_columns in ((axis, expected_axis), (edge, expected_edge)):
        for row, column in enumerate(expected_columns):
            nonzero = np.flatnonzero(boundary.C[row])
            np.testing.assert_array_equal(nonzero, [column])
            assert boundary.C[row, column] == pytest.approx(1.0)

    sentinel = np.zeros(n_trial, dtype=np.complex128)
    axis_values = np.array([11.0 + 2.0j, 12.0 + 3.0j, 13.0 + 4.0j])
    edge_values = np.array([21.0 - 2.0j, 22.0 - 3.0j, 23.0 - 4.0j])
    sentinel[expected_axis] = axis_values
    sentinel[expected_edge] = edge_values
    np.testing.assert_allclose(axis.C @ sentinel, axis_values)
    np.testing.assert_allclose(edge.C @ sentinel, edge_values)


def test_combined_bc_has_required_rank_and_sparse_nullspace(mixed_bcs):
    high, low, m_numbers, axis, edge = mixed_bcs
    n_high = high.get_ndofs()
    n_low = low.get_ndofs()
    n_modes = m_numbers.size
    n_trial = 2 * n_modes * (n_high + n_low)
    n_rows = 4 * n_modes * n_low

    combined = axis * edge
    assert isinstance(combined, BoundaryCondition)
    assert combined.C.shape == (2 * n_modes, n_trial)
    assert np.linalg.matrix_rank(combined.C) == 2 * n_modes

    P = combined.get_sparse_prolongation()
    assert isinstance(P, csr_matrix)
    assert P.shape == (n_trial, n_rows)
    np.testing.assert_allclose(combined.C @ P.toarray(), 0.0, atol=1e-13)

    reverse = edge * axis
    assert isinstance(reverse, BoundaryCondition)
    np.testing.assert_allclose(reverse.C, np.vstack([edge.C, axis.C]))


def test_one_sided_trial_reduction_makes_mixed_operator_square(mixed_bcs):
    high, low, m_numbers, axis, edge = mixed_bcs
    n_high = high.get_ndofs()
    n_low = low.get_ndofs()
    n_modes = m_numbers.size
    n_trial = 2 * n_modes * (n_high + n_low)
    n_rows = 4 * n_modes * n_low

    combined = axis * edge
    T = eye(n_rows, n_trial, dtype=np.complex128, format="csr")
    T_reduced = combined.reduce_sparse_trial_matrix(T)

    assert isinstance(T_reduced, csr_matrix)
    assert T_reduced.shape == (n_rows, n_rows)
    np.testing.assert_allclose(
        T_reduced.toarray(),
        (T @ combined.get_sparse_prolongation()).toarray(),
    )


def test_trial_reduction_validates_trial_dimension(mixed_bcs):
    _, _, _, axis, edge = mixed_bcs
    combined = axis * edge
    wrong = eye(5, combined.C.shape[1] - 1, format="csr")

    with pytest.raises(ValueError, match="trial columns"):
        combined.reduce_sparse_trial_matrix(wrong)


def test_boundary4_rejects_incompatible_radial_dimensions(mixed_spaces):
    high, _ = mixed_spaces
    wrong_low = BSplineFE(
        knots=high.knots,
        order=high.order,
        domain=(high.knots[0], high.knots[-1]),
        quad_order=high.quadrature_order,
    )

    with pytest.raises(ValueError, match="n_low=n_high-1"):
        idealSeparatrixBC(high, wrong_low, n_modes=2)
