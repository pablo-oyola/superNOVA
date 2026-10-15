"""Global complex radial path and its local least-squares continuation."""

import numpy as np
from scipy.sparse import coo_array

from supernova.operators.contour import GlobalPath, LocalPolynomialContinuation, continue_tensor


def test_global_path_vanishes_at_the_walls_and_peaks_at_the_centre():
    path = GlobalPath(0.08, domain=(0.1, 0.9))
    assert np.allclose(path.y([0.1, 0.9]), 0.0)
    assert np.isclose(path.y(0.5), path.amplitude)
    x = np.linspace(0.1, 0.9, 401)
    assert np.allclose(np.gradient(path.y(x), x)[1:-1], path.dy(x)[1:-1], atol=1e-10)
    assert np.allclose(path.z(x), x + 1j * path.y(x))
    assert np.isclose(path.alpha_for_height(0.3, path.y(0.3)), 0.08)


def test_local_polynomial_continuation_is_exact_for_low_degree_polynomials():
    x = np.linspace(0.0, 1.0, 60)
    points = np.linspace(0.05, 0.95, 17)
    iy = 1j * 0.03 * np.sin(np.pi * points)
    f = lambda z: 1.0 - 2.0 * z + 0.5 * z**3 - 0.7 * z**5
    W = LocalPolynomialContinuation(degree=5, neighbors=14).matrix(x, points, iy)
    assert np.allclose(W @ f(x), f(points + iy), atol=1e-10)


def test_continue_tensor_scales_only_the_differential_rows_by_the_tangent():
    x = np.linspace(0.0, 1.0, 40)
    q = np.repeat(np.arange(x.size), 2)
    rows = np.tile([0, 1], x.size)
    cols = np.zeros_like(rows)
    data = np.stack([x**2, 1.0 + x], axis=1).ravel().astype(np.complex128)
    tensor = coo_array((data, (q, rows, cols)), shape=(x.size, 2, 1))
    path = GlobalPath(0.1, domain=(0.0, 1.0))
    points = np.array([0.2, 0.5, 0.8])
    out = continue_tensor(x, tensor, points, path, LocalPolynomialContinuation(),
                          scaled_rows=1).todense()
    z = path.z(points)
    assert np.allclose(out[:, 0, 0], z**2 * path.dz(points), atol=1e-10)
    assert np.allclose(out[:, 1, 0], 1.0 + z, atol=1e-10)
