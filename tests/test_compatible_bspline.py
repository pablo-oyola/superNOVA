import numpy as np
import pytest
from scipy.sparse import isspmatrix_csr

from supernova.radial import BSplineFE, CompatibleBSplinePair


def _make_high(interior_knots, *, order=4, quad_order=6):
    knots = np.concatenate(
        (
            np.zeros(order),
            np.asarray(interior_knots, dtype=np.float64),
            np.ones(order),
        )
    )
    return BSplineFE(
        knots=knots,
        order=order,
        domain=(0.0, 1.0),
        quad_order=quad_order,
    )


def test_pair_uses_trimmed_knots_and_shared_quadrature():
    high = _make_high([0.15, 0.4, 0.8])

    pair = CompatibleBSplinePair.from_high(high)

    np.testing.assert_array_equal(pair.low.knots, high.knots[1:-1])
    assert pair.low.order == high.order - 1
    assert pair.low.degree == high.degree - 1
    assert pair.low.quadrature_order == high.quadrature_order
    assert pair.low.get_ndofs() == high.get_ndofs() - 1

    high_points, high_weights = high.get_quadrature_points()
    low_points, low_weights = pair.low.get_quadrature_points()
    np.testing.assert_array_equal(low_points, high_points)
    np.testing.assert_array_equal(low_weights, high_weights)


@pytest.mark.parametrize(
    "interior_knots",
    (
        [0.2, 0.4, 0.6, 0.8],
        [0.07, 0.25, 0.61, 0.9],
    ),
)
def test_derivative_matrix_has_expected_coefficients(interior_knots):
    high = _make_high(interior_knots)
    pair = CompatibleBSplinePair.from_high(high)

    derivative = pair.derivative
    n_low, n_high = derivative.shape
    indices = np.arange(n_low)
    alpha = high.degree / (
        high.knots[indices + high.degree + 1]
        - high.knots[indices + 1]
    )
    expected = np.zeros((n_low, n_high))
    expected[indices, indices] = -alpha
    expected[indices, indices + 1] = alpha

    assert isspmatrix_csr(derivative)
    assert derivative.shape == (high.get_ndofs() - 1, high.get_ndofs())
    np.testing.assert_allclose(derivative.toarray(), expected)


@pytest.mark.parametrize(
    "interior_knots",
    (
        [0.25, 0.5, 0.75],
        [0.04, 0.19, 0.52, 0.86],
    ),
)
def test_compatible_basis_reproduces_high_basis_derivatives(interior_knots):
    pair = CompatibleBSplinePair.from_high(_make_high(interior_knots))
    points, _ = pair.high.get_quadrature_points()

    phi_high_prime = pair.high.eval_basis_derivative(points).T
    phi_low = pair.low.eval_basis(points).T

    np.testing.assert_allclose(
        phi_high_prime,
        phi_low @ pair.derivative.toarray(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_low_mass_times_derivative_matches_direct_mixed_quadrature():
    pair = CompatibleBSplinePair.from_high(
        _make_high([0.03, 0.21, 0.58, 0.91], quad_order=7)
    )
    points, weights = pair.high.get_quadrature_points()
    phi_low = pair.low.eval_basis(points).T
    phi_high_prime = pair.high.eval_basis_derivative(points).T

    direct = phi_low.T @ (weights[:, None] * phi_high_prime)
    compatible = pair.low.mass_matrix @ pair.derivative

    np.testing.assert_allclose(
        compatible.toarray(),
        direct,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_pair_rejects_high_space_below_quadratic_degree():
    high = _make_high([0.3, 0.7], order=2, quad_order=4)

    with pytest.raises(ValueError, match="degree >= 2"):
        CompatibleBSplinePair.from_high(high)


def test_pair_rejects_nonpositive_derivative_denominator():
    high = _make_high([0.25, 0.5, 0.75])
    high.knots = high.knots.copy()
    high.knots[high.degree + 1] = high.knots[1]

    with pytest.raises(ValueError, match="positive derivative denominators"):
        CompatibleBSplinePair.from_high(high)


def test_evaluation_only_pair_uses_external_points_without_native_quadrature():
    assembled = _make_high([0.2, 0.5, 0.8])
    points, _ = assembled.get_quadrature_points()
    high = BSplineFE(
        knots=assembled.knots,
        order=assembled.order,
        domain=(0.0, 1.0),
        quad_order=assembled.quadrature_order,
        build_matrices=False,
    )

    pair = CompatibleBSplinePair.from_high(
        high,
        build_low_matrices=False,
        verification_points=points,
    )

    assert high.mass_matrix is None
    assert pair.low.mass_matrix is None
    with pytest.raises(RuntimeError, match="no native quadrature"):
        high.get_quadrature_points()
    with pytest.raises(RuntimeError, match="no native quadrature"):
        pair.low.get_quadrature_points()
    np.testing.assert_allclose(
        high.eval_basis_derivative(points).T,
        pair.low.eval_basis(points).T @ pair.derivative.toarray(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
