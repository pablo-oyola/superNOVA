r"""Global complex radial path for classifying eigenvalues near the continuum.

On the real radial axis the ideal-MHD operator is singular wherever a local
continuum branch ``s_k(x)`` meets the eigenvalue ``s = omega**2``, and a
discretisation returns a dense set of continuum approximants there.  Solving
the same equations along the complex path

    z(x) = x + i*alpha*(x - x0)*(x1 - x)

leaves every physical (discrete) eigenvalue where it is, by Cauchy's theorem,
while the continuum approximants rotate onto the image ``s_k(z(x))`` and move
with ``alpha``.  Path invariance, not the damping a path produces, is the
discriminator: run the real axis and a few ``alpha`` of both signs and keep the
eigenvalues that do not move.

The invariance only holds while the path does not reach a singular point.  A
discrete eigenvalue at distance ``Delta s`` from a continuum extremum (a gap
edge) has a complex pair of singular points at

    d = sqrt(2 |Delta s| / |s''|)

from the real axis.  Keep the path height at that extremum below about
``d / 2``; at ``alpha*(x_e - x0)*(x1 - x_e) = d`` the eigenvalue breaks.  Modes
close to a gap edge therefore need small ``alpha``, and the mode must be
resolved by the radial mesh before the test means anything.

For time dependence ``exp(-i*omega*t)`` and ``Re(omega) > 0``, ``alpha > 0``
is causal at crossings of falling branches (``ds/dx < 0``) and ``alpha < 0``
at rising ones; a single sign therefore yields the continuum-damped resonance
only when all relevant crossings share one slope sign.

Along the path the first-order rows ``du/dx = beta*M(x)*state`` become
``du/dx = z'(x)*beta*M(z(x))*state`` and the algebraic rows only need
``M(z(x))``.  The continuation acts on the assembled local tensors, not on the
profile harmonics (the harmonic kernels build negative-``dm`` couplings as
``conj(f_m)``, valid on the real axis only).  Tensors sampled on the native
equilibrium surfaces are continued by local least-squares polynomials, which
smooth the sample noise and have no poles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_array, csr_matrix


@dataclass(frozen=True)
class GlobalPath:
    r"""Global parabolic path ``z = x + i*alpha*(x - x0)*(x1 - x)``.

    One smooth deformation of the whole radial domain, zero at both walls so
    the boundary conditions stay on the real axis.  ``alpha = 0`` is the real
    axis.
    """

    alpha: float
    domain: tuple[float, float] = (0.0, 1.0)

    @property
    def amplitude(self) -> float:
        """Largest ``|y|``, reached at the centre of the domain."""
        lo, hi = self.domain
        return abs(self.alpha) * 0.25 * (hi - lo) ** 2

    def y(self, x):
        """Imaginary displacement ``y(x)`` of the path."""
        x = np.asarray(x, dtype=np.float64)
        lo, hi = self.domain
        return self.alpha * (x - lo) * (hi - x)

    def dy(self, x):
        """Derivative ``y'(x)``."""
        x = np.asarray(x, dtype=np.float64)
        lo, hi = self.domain
        return self.alpha * (lo + hi - 2.0 * x)

    def z(self, x):
        """Complex path ``x + i*y(x)``."""
        x = np.asarray(x, dtype=np.float64)
        return x + 1.0j * self.y(x)

    def dz(self, x):
        """Path tangent ``1 + i*y'(x)``."""
        return 1.0 + 1.0j * self.dy(x)

    def alpha_for_height(self, x: float, height: float) -> float:
        """The ``|alpha|`` whose path reaches ``height`` at ``x``."""
        lo, hi = self.domain
        return float(height) / ((float(x) - lo) * (hi - float(x)))


@dataclass
class LocalPolynomialContinuation:
    r"""Continue by a least-squares polynomial fitted around each point.

    For a point ``x_q`` the ``neighbors`` nearest samples are fitted, in the
    least-squares sense, by a polynomial of ``degree`` in ``(x - x_q)/s_q``
    (``s_q`` the half-span of those samples), which is then evaluated at the
    complex offset ``i*y_q``.  The result is linear in the samples, so one
    sparse operator continues every tensor entry.  It smooths sample noise
    instead of differentiating it and has no poles; :meth:`noise_gain` gives
    the amplification of independent sample noise at each point.
    """

    degree: int = 5
    neighbors: int = 14

    def _rows(self, sample_x, points, iy):
        sample_x = np.asarray(sample_x, dtype=np.float64)
        points = np.asarray(points, dtype=np.float64)
        iy = np.broadcast_to(np.asarray(iy, dtype=np.complex128), points.shape)
        k = min(self.neighbors, sample_x.size)
        p = min(self.degree, k - 1)
        idx_all, w_all = [], []
        for xq, offset in zip(points, iy):
            idx = np.argsort(np.abs(sample_x - xq))[:k]
            idx.sort()
            local = sample_x[idx] - xq
            span = max(np.max(np.abs(local)), 1e-300)
            V = np.vander(local / span, p + 1, increasing=True)
            # Weights e(t)^T pinv(V): the fitted polynomial evaluated at t.
            t = offset / span
            e = t ** np.arange(p + 1)
            idx_all.append(idx)
            w_all.append(e @ np.linalg.pinv(V))
        return idx_all, w_all

    def matrix(self, sample_x, points, iy) -> csr_matrix:
        """Sparse ``(npoint, nsample)`` operator: continued values = matrix @ samples."""
        idx_all, w_all = self._rows(sample_x, points, iy)
        rows = np.repeat(np.arange(len(idx_all)), [len(i) for i in idx_all])
        return csr_matrix((np.concatenate(w_all), (rows, np.concatenate(idx_all))),
                          shape=(len(idx_all), np.size(sample_x)))

    def noise_gain(self, sample_x, points, iy) -> np.ndarray:
        """Per-point amplification of i.i.d. sample noise (1-norm of the weights)."""
        _, w_all = self._rows(sample_x, points, iy)
        return np.array([np.sum(np.abs(w)) for w in w_all])


def continue_tensor(
    sample_x: np.ndarray,
    tensor,
    points: np.ndarray,
    path: GlobalPath,
    continuation: LocalPolynomialContinuation,
    scaled_rows: int = 0,
    chunk: int = 4096,
) -> coo_array:
    r"""Continue one local COO tensor from real samples onto the path.

    Every stored ``(row, col)`` entry, sampled at ``sample_x``, is evaluated at
    ``z = points + i*y(points)`` through ``continuation``.  Rows below
    ``scaled_rows`` are the first-order differential rows and are multiplied
    by the path tangent ``z'``.
    """
    tensor = coo_array(tensor) if not isinstance(tensor, coo_array) else tensor
    sample_x = np.asarray(sample_x, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    nsample, nrow, ncol = tensor.shape
    if nsample != sample_x.size:
        raise ValueError("tensor and sample_x disagree on the sample count.")
    q, rows, cols = tensor.coords
    key = rows.astype(np.int64) * ncol + cols.astype(np.int64)
    keys, inverse = np.unique(key, return_inverse=True)

    weights = continuation.matrix(sample_x, points, 1.0j * path.y(points))
    tangent = path.dz(points)

    out_q, out_r, out_c, out_v = [], [], [], []
    for start in range(0, keys.size, chunk):
        stop = min(start + chunk, keys.size)
        mask = (inverse >= start) & (inverse < stop)
        dense = np.zeros((nsample, stop - start), dtype=np.complex128)
        np.add.at(dense, (q[mask], inverse[mask] - start), tensor.data[mask])
        values = np.asarray(weights @ dense)
        chunk_rows = keys[start:stop] // ncol
        chunk_cols = keys[start:stop] % ncol
        values[:, chunk_rows < scaled_rows] *= tangent[:, None]
        qq, kk = np.nonzero(values)
        out_q.append(qq)
        out_r.append(chunk_rows[kk])
        out_c.append(chunk_cols[kk])
        out_v.append(values[qq, kk])
    if not out_v:
        return coo_array((points.size, nrow, ncol), dtype=np.complex128)
    return coo_array(
        (
            np.concatenate(out_v),
            (np.concatenate(out_q), np.concatenate(out_r), np.concatenate(out_c)),
        ),
        shape=(points.size, nrow, ncol),
        dtype=np.complex128,
    )
