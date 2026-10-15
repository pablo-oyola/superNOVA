r"""Complex radial contour through the continuum resonances.

On the real radial axis the ideal-MHD operator is singular wherever a local
continuum branch ``s_k(x)`` meets the eigenvalue ``s = omega**2``.  Ideal MHD
does not fix the connection across those points, which is why every ``s`` in
a continuum band is a spectral point and why a discretisation returns a dense
set of element-scale continuum approximants there.

For time dependence ``exp(-i*omega*t)`` the causal (Landau) prescription is
``s -> s + i0``.  A singular point ``x_c(s)`` then sits on the side
``sign(Im x_c) = sign(ds_k/dx)`` of the real axis, and the analytic
continuation of the solution to damped ``Im(s) < 0`` must keep it there.  The
radial path ``z(x) = x + i*y(x)`` does so when, for ``Re(omega) > 0``,

    sign(y(x_c)) = -sign(ds_k/dx)   and   |y(x_c)| > |Im(s)| / |ds_k/dx|.

On such a path the damped resonances become isolated eigenvalues of a
non-Hermitian pencil, while the discretised continuum moves onto the image
curve ``s_k(z(x))``.  Physical eigenvalues do not depend on the path; the
rotated continuum does.  That is the classification test this module serves.

Along the path the first-order rows ``du/dx = beta*M(x)*state`` become
``du/dx = z'(x)*beta*M(z(x))*state`` and the algebraic rows only need
``M(z(x))``.  The continuation is applied to the assembled local tensors, not
to the profile harmonics: the harmonic kernels build negative-``dm`` couplings
by conjugating positive harmonics, ``f_{-m} = conj(f_m)``, which holds on the
real axis only.  The tensors are sampled on the native equilibrium surfaces,
fitted by a cubic spline in ``x``, and each cubic piece is continued exactly
by its Taylor series ``f(x+iy) = sum_k f^(k)(x) (iy)^k / k!``, ``k <= 3``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import factorial

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import eigvals
from scipy.sparse import coo_array


@dataclass(frozen=True)
class RadialContour:
    r"""Radial path ``z(x) = x + i*y(x)`` on the solver interval ``[0, 1]``.

    ``y`` is a sum of smooth plateau bumps, one per continuum crossing,
    each with the causal sign ``-sign(ds_k/dx)``.  An endpoint taper keeps
    ``y(0) = y(1) = 0`` so the boundary conditions stay on the real axis.

    Parameters
    ----------
    centers
        Crossing positions ``x_c`` in the solver coordinate.
    signs
        ``+1`` or ``-1``: the deformation direction at each crossing.
    amplitude
        Plateau height ``delta`` of every bump.
    width
        Plateau half-width ``w`` of every bump.
    edge
        Sharpness of the bump edges, in units of ``w``.
    slopes
        Optional ``ds_k/dx`` at each crossing, used only by :meth:`coverage`.
    """

    centers: np.ndarray
    signs: np.ndarray
    amplitude: float
    width: float
    edge: float = 0.35
    slopes: np.ndarray | None = None
    domain: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self):
        centers = np.atleast_1d(np.asarray(self.centers, dtype=np.float64))
        signs = np.atleast_1d(np.asarray(self.signs, dtype=np.float64))
        if centers.shape != signs.shape:
            raise ValueError("centers and signs must have the same shape.")
        if np.any(np.abs(signs) != 1.0):
            raise ValueError("signs must be +1 or -1.")
        if self.width <= 0.0 or self.edge <= 0.0:
            raise ValueError("width and edge must be positive.")
        if self.slopes is not None:
            slopes = np.atleast_1d(np.asarray(self.slopes, dtype=np.float64))
            if slopes.shape != centers.shape:
                raise ValueError("slopes must match centers.")
            object.__setattr__(self, "slopes", slopes)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "signs", signs)

    def with_amplitude(self, amplitude: float) -> "RadialContour":
        """Return the same path shape with a different plateau height."""
        return replace(self, amplitude=float(amplitude))

    def _bump(self, x: np.ndarray):
        """Return the unsigned plateau profile and its x-derivative."""
        x = np.asarray(x, dtype=np.float64)
        tau = self.edge
        norm = np.tanh(1.0 / tau)
        value = np.zeros((self.centers.size,) + x.shape)
        slope = np.zeros_like(value)
        for ii, center in enumerate(self.centers):
            u = (x - center) / self.width
            a = (u + 1.0) / tau
            b = (u - 1.0) / tau
            value[ii] = 0.5 * (np.tanh(a) - np.tanh(b)) / norm
            slope[ii] = (
                0.5 / (tau * self.width * norm)
                * ((1.0 - np.tanh(a) ** 2) - (1.0 - np.tanh(b) ** 2))
            )
        return value, slope

    def _taper(self, x: np.ndarray):
        """Return the endpoint taper and its x-derivative."""
        lo, hi = self.domain
        eps = self.width
        a = np.exp(-(((x - lo) / eps) ** 2))
        b = np.exp(-(((hi - x) / eps) ** 2))
        value = (1.0 - a) * (1.0 - b)
        slope = (
            (2.0 * (x - lo) / eps**2) * a * (1.0 - b)
            - (1.0 - a) * (2.0 * (hi - x) / eps**2) * b
        )
        return value, slope

    def y(self, x: np.ndarray) -> np.ndarray:
        """Imaginary displacement ``y(x)`` of the path."""
        x = np.asarray(x, dtype=np.float64)
        if self.centers.size == 0 or self.amplitude == 0.0:
            return np.zeros_like(x)
        bump, _ = self._bump(x)
        taper, _ = self._taper(x)
        return self.amplitude * taper * np.tensordot(self.signs, bump, axes=1)

    def dy(self, x: np.ndarray) -> np.ndarray:
        """Derivative ``y'(x)`` of the imaginary displacement."""
        x = np.asarray(x, dtype=np.float64)
        if self.centers.size == 0 or self.amplitude == 0.0:
            return np.zeros_like(x)
        bump, dbump = self._bump(x)
        taper, dtaper = self._taper(x)
        total = np.tensordot(self.signs, bump, axes=1)
        dtotal = np.tensordot(self.signs, dbump, axes=1)
        return self.amplitude * (dtaper * total + taper * dtotal)

    def z(self, x: np.ndarray) -> np.ndarray:
        """Complex path ``x + i*y(x)``."""
        x = np.asarray(x, dtype=np.float64)
        return x + 1.0j * self.y(x)

    def dz(self, x: np.ndarray) -> np.ndarray:
        """Path tangent ``1 + i*y'(x)``."""
        return 1.0 + 1.0j * self.dy(x)

    def coverage(self) -> dict:
        r"""Report, per crossing, whether the path passes on the causal side.

        ``max_damping`` is the largest ``|Im s|`` whose singular point the
        path still uncovers at that crossing, to first order
        ``|y(x_c)| * |ds_k/dx|``.  A crossing with the wrong sign of
        ``y(x_c)`` (a neighbour of opposite sign dominates) is a conflict.
        """
        y_c = self.y(self.centers)
        wrong_side = np.sign(y_c) != self.signs
        out = {
            "y_at_crossing": y_c,
            "conflict": wrong_side | (y_c == 0.0),
        }
        if self.slopes is not None:
            out["max_damping"] = np.where(
                out["conflict"], 0.0, np.abs(y_c) * np.abs(self.slopes)
            )
        return out


@dataclass
class ContinuumCrossings:
    """Continuum crossings of one target eigenvalue along the real axis."""

    s_target: complex
    x: np.ndarray
    slope_sign: np.ndarray
    slope: np.ndarray
    counts_below: np.ndarray = field(repr=False)
    sample_x: np.ndarray = field(repr=False)

    def contour(
        self,
        amplitude: float,
        width: float,
        edge: float = 0.35,
        omega_sign: float = 1.0,
    ) -> RadialContour:
        r"""Return the causal path for these crossings.

        ``omega_sign`` is the sign of ``Re(omega)``; the causal side flips for
        negative frequencies.
        """
        signs = -np.sign(omega_sign) * self.slope_sign
        return RadialContour(
            centers=self.x,
            signs=signs,
            amplitude=amplitude,
            width=width,
            edge=edge,
            slopes=self.slope,
        )


def _dense_local(tensor: coo_array, index: int, shape: tuple[int, int]) -> np.ndarray:
    """Return one point of a COO tensor as a dense matrix."""
    q, rows, cols = tensor.coords
    mask = q == index
    out = np.zeros(shape, dtype=np.complex128)
    np.add.at(out, (rows[mask], cols[mask]), tensor.data[mask])
    return out


def local_continuum_eigenvalues(A, B, rho_mu0: np.ndarray) -> np.ndarray:
    r"""Return the local continuum ``s`` at every sample point.

    The singular algebraic block is ``A - lambda*B`` with
    ``lambda = mu_0*rho*s``, so the local continuum is the generalised
    spectrum of ``(A, mu_0*rho*B)``.  Infinite eigenvalues are returned as
    ``nan``.  The result has shape ``(npoints, 2*nmode)``.
    """
    A = coo_array(A) if not isinstance(A, coo_array) else A
    B = coo_array(B) if not isinstance(B, coo_array) else B
    npts, n, _ = A.shape
    out = np.full((npts, n), np.nan + 0.0j)
    for ii in range(npts):
        a = _dense_local(A, ii, (n, n))
        b = _dense_local(B, ii, (n, n)) * rho_mu0[ii]
        w = eigvals(a, b)
        w[~np.isfinite(w)] = np.nan
        out[ii, : w.size] = w
    return out


def find_continuum_crossings(
    sample_x: np.ndarray,
    continuum_s: np.ndarray,
    s_target: complex,
    max_magnitude: float | None = None,
) -> ContinuumCrossings:
    r"""Locate where local continuum branches cross ``Re(s_target)``.

    Branches are not tracked.  Instead the number of branches below the
    target is counted at every sample point; a fall by one between two
    neighbours is a rising branch (``ds/dx > 0``), a rise by one a falling
    branch.  Two opposite crossings inside one sample interval cancel and are
    missed, so the sample grid must be finer than the crossing spacing.
    """
    sample_x = np.asarray(sample_x, dtype=np.float64)
    s0 = complex(s_target).real
    values = np.asarray(continuum_s)
    real = values.real.copy()
    bad = ~np.isfinite(real)
    if max_magnitude is None:
        max_magnitude = 1.0e6 * max(abs(s0), 1.0)
    bad |= np.abs(values) > max_magnitude
    real[bad] = np.inf
    below = np.sum(real < s0, axis=1)

    xs, signs, slopes = [], [], []
    for jj in range(sample_x.size - 1):
        change = int(below[jj + 1] - below[jj])
        if change == 0:
            continue
        x0, x1 = sample_x[jj], sample_x[jj + 1]
        # The branch nearest the target on each side places the crossing.
        d0 = real[jj] - s0
        d1 = real[jj + 1] - s0
        i0 = np.nanargmin(np.where(np.isfinite(d0), np.abs(d0), np.inf))
        i1 = np.nanargmin(np.where(np.isfinite(d1), np.abs(d1), np.inf))
        a, b = d0[i0], d1[i1]
        if np.isfinite(a) and np.isfinite(b) and a * b <= 0.0 and a != b and abs(change) == 1:
            frac = a / (a - b)
            slope = (b - a) / (x1 - x0)
        else:
            frac = 0.5
            slope = np.nan
        sign = -np.sign(change)
        for _ in range(abs(change)):
            xs.append(x0 + frac * (x1 - x0))
            signs.append(sign)
            slopes.append(slope if np.isfinite(slope) else sign * np.nan)
    return ContinuumCrossings(
        s_target=complex(s_target),
        x=np.asarray(xs),
        slope_sign=np.asarray(signs, dtype=np.float64),
        slope=np.asarray(slopes, dtype=np.float64),
        counts_below=below,
        sample_x=sample_x,
    )


def continue_tensor(
    sample_x: np.ndarray,
    tensor,
    points: np.ndarray,
    contour,
    scaled_rows: int = 0,
    order: int = 3,
    chunk: int = 4096,
    rational: "RationalContinuation | None" = None,
) -> coo_array:
    r"""Continue one local COO tensor from real samples onto the path.

    By default every stored ``(row, col)`` entry is fitted by a not-a-knot
    cubic spline over ``sample_x`` and evaluated at ``z = points + i*y(points)``
    by the Taylor series of its cubic piece, truncated at ``order`` (3 is exact
    for the spline).  With ``rational`` the entries are instead evaluated at
    ``z`` through that shared barycentric rational.  Rows below
    ``scaled_rows`` are the first-order differential rows and are multiplied
    by the path tangent ``z'``.
    """
    if order < 0 or order > 3:
        raise ValueError("order must be between 0 and 3 for a cubic spline.")
    tensor = coo_array(tensor) if not isinstance(tensor, coo_array) else tensor
    sample_x = np.asarray(sample_x, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    nsample, nrow, ncol = tensor.shape
    if nsample != sample_x.size:
        raise ValueError("tensor and sample_x disagree on the sample count.")
    q, rows, cols = tensor.coords
    key = rows.astype(np.int64) * ncol + cols.astype(np.int64)
    keys, inverse = np.unique(key, return_inverse=True)

    iy = 1.0j * contour.y(points)
    tangent = contour.dz(points)
    taylor = [iy**k / factorial(k) for k in range(order + 1)]
    if rational is not None:
        weights = rational.matrix(sample_x, points, iy)

    out_q, out_r, out_c, out_v = [], [], [], []
    for start in range(0, keys.size, chunk):
        stop = min(start + chunk, keys.size)
        mask = (inverse >= start) & (inverse < stop)
        dense = np.zeros((nsample, stop - start), dtype=np.complex128)
        np.add.at(dense, (q[mask], inverse[mask] - start), tensor.data[mask])
        if rational is not None:
            values = np.asarray(weights @ dense)
        else:
            spline = CubicSpline(sample_x, dense, axis=0, bc_type="not-a-knot")
            values = taylor[0][:, None] * spline(points)
            for k in range(1, order + 1):
                values += taylor[k][:, None] * spline(points, k)
        chunk_rows = keys[start:stop] // ncol
        chunk_cols = keys[start:stop] % ncol
        differential = chunk_rows < scaled_rows
        values[:, differential] *= tangent[:, None]
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


def continue_scalar(
    sample_x: np.ndarray,
    values: np.ndarray,
    points: np.ndarray,
    contour,
    order: int = 3,
    rational: "RationalContinuation | None" = None,
) -> np.ndarray:
    """Continue a sampled scalar profile onto the path."""
    if rational is not None:
        iy = 1.0j * contour.y(points)
        return np.asarray(rational.matrix(sample_x, points, iy) @ np.asarray(values))
    spline = CubicSpline(np.asarray(sample_x, dtype=np.float64), values)
    iy = 1.0j * contour.y(points)
    out = np.asarray(spline(points), dtype=np.complex128)
    for k in range(1, order + 1):
        out = out + iy**k / factorial(k) * spline(points, k)
    return out


def continuum_image(
    sample_x: np.ndarray,
    A,
    B,
    rho_mu0: np.ndarray,
    contour,
    points: np.ndarray,
    rational: "RationalContinuation | None" = None,
) -> np.ndarray:
    r"""Local continuum along the path: where the rotated continuum must lie.

    Discrete eigenvalues near this image are continuum approximants on the
    deformed path.  Eigenvalues that are neither on it nor invariant under a
    change of path are numerical artefacts.
    """
    A_z = continue_tensor(sample_x, A, points, contour, rational=rational)
    B_z = continue_tensor(sample_x, B, points, contour, rational=rational)
    rho_z = continue_scalar(sample_x, np.asarray(rho_mu0), points, contour,
                            rational=rational)
    return local_continuum_eigenvalues(A_z, B_z, rho_z)


# --------------------------------------------------------------------------
# A path shaped by the sign pattern of the crossings.

@dataclass(frozen=True)
class CausalPath:
    r"""Causal path whose bumps adapt to their opposite-sign neighbours.

    Each crossing ``c`` gets a plateau bump of width ``w_c`` and height
    ``a_c`` with the causal sign.  Next to a crossing of the opposite sign the
    path must cross zero in between, so ``w_c`` is capped at a fraction of
    that gap and ``a_c`` at ``edge * w_c``, which keeps ``|y'|`` of order one.
    Same-sign neighbours merge: the bumps are summed and saturated smoothly,
    ``y = A * tanh(Y / A)``, so a run of same-sign crossings becomes one broad
    deformation instead of a comb.
    """

    centers: np.ndarray
    signs: np.ndarray
    amplitudes: np.ndarray
    widths: np.ndarray
    amplitude: float
    edge: float = 0.35
    slopes: np.ndarray | None = None
    domain: tuple[float, float] = (0.0, 1.0)
    taper_width: float = 0.01

    @property
    def width(self) -> float:
        return float(np.max(self.widths)) if self.widths.size else 0.0

    def _sum(self, x):
        x = np.asarray(x, dtype=np.float64)
        Y = np.zeros_like(x)
        dY = np.zeros_like(x)
        tau = self.edge
        norm = np.tanh(1.0 / tau)
        for center, sign, a, w in zip(self.centers, self.signs, self.amplitudes, self.widths):
            u = (x - center) / w
            p = (u + 1.0) / tau
            m = (u - 1.0) / tau
            Y += sign * a * 0.5 * (np.tanh(p) - np.tanh(m)) / norm
            dY += sign * a * 0.5 / (tau * w * norm) * ((1.0 - np.tanh(p) ** 2) - (1.0 - np.tanh(m) ** 2))
        return Y, dY

    def _taper(self, x):
        lo, hi = self.domain
        eps = self.taper_width
        a = np.exp(-(((x - lo) / eps) ** 2))
        b = np.exp(-(((hi - x) / eps) ** 2))
        value = (1.0 - a) * (1.0 - b)
        slope = ((2.0 * (x - lo) / eps**2) * a * (1.0 - b)
                 - (1.0 - a) * (2.0 * (hi - x) / eps**2) * b)
        return value, slope

    def y(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.centers.size == 0 or self.amplitude == 0.0:
            return np.zeros_like(x)
        Y, _ = self._sum(x)
        taper, _ = self._taper(x)
        return self.amplitude * np.tanh(Y / self.amplitude) * taper

    def dy(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.centers.size == 0 or self.amplitude == 0.0:
            return np.zeros_like(x)
        Y, dY = self._sum(x)
        taper, dtaper = self._taper(x)
        t = np.tanh(Y / self.amplitude)
        return (1.0 - t**2) * dY * taper + self.amplitude * t * dtaper

    def z(self, x):
        x = np.asarray(x, dtype=np.float64)
        return x + 1.0j * self.y(x)

    def dz(self, x):
        return 1.0 + 1.0j * self.dy(x)

    def coverage(self) -> dict:
        """Per-crossing side and uncovered damping, as for RadialContour."""
        y_c = self.y(self.centers)
        conflict = (np.sign(y_c) != self.signs) | (y_c == 0.0)
        out = {"y_at_crossing": y_c, "conflict": conflict}
        if self.slopes is not None:
            out["max_damping"] = np.where(conflict, 0.0, np.abs(y_c) * np.abs(self.slopes))
        return out

    def resolution(self, edge_elements: int = 4, distance_elements: float = 3.0) -> np.ndarray:
        """Element size each crossing needs: bump edge and path distance resolved."""
        return np.minimum(2.0 * self.edge * self.widths / edge_elements,
                          self.amplitudes / distance_elements)


def causal_path(crossings: ContinuumCrossings, amplitude: float, width: float,
                domain: tuple[float, float], edge: float = 0.35,
                gap_fraction: float = 0.35, omega_sign: float = 1.0,
                taper_width: float | None = None) -> CausalPath:
    r"""Build a :class:`CausalPath` for one target eigenvalue's crossings."""
    order = np.argsort(crossings.x)
    x = crossings.x[order]
    signs = -np.sign(omega_sign) * crossings.slope_sign[order]
    slopes = crossings.slope[order]
    gaps = np.full(x.size, np.inf)
    for ii in range(x.size):
        opposite = np.abs(x[signs != signs[ii]] - x[ii])
        if opposite.size:
            gaps[ii] = opposite.min()
    widths = np.minimum(width, gap_fraction * gaps)
    amplitudes = np.minimum(amplitude, edge * widths)
    if taper_width is None:
        taper_width = width
    return CausalPath(centers=x, signs=signs, amplitudes=amplitudes, widths=widths,
                      amplitude=float(amplitude), edge=edge, slopes=slopes,
                      domain=tuple(domain), taper_width=float(taper_width))


def graded_breakpoints(domain: tuple[float, float], base: float,
                       centers: np.ndarray, local: np.ndarray,
                       growth: float = 0.25, max_elements: int = 20000) -> np.ndarray:
    r"""Breakpoints whose size is ``min(base, local_c + growth*|x - x_c|)``.

    The linear growth away from each refined centre keeps neighbouring
    elements within a factor ``1 + growth`` of each other.
    """
    lo, hi = domain
    centers = np.asarray(centers, dtype=np.float64)
    local = np.asarray(local, dtype=np.float64)

    def size(x):
        if centers.size == 0:
            return base
        return float(min(base, np.min(local + growth * np.abs(x - centers))))

    points = [lo]
    while points[-1] < hi:
        if len(points) > max_elements:
            raise RuntimeError("graded mesh exceeds max_elements; coarsen the path.")
        points.append(points[-1] + size(points[-1]))
    points = np.asarray(points)
    # Stretch the overshooting last step back onto the wall.
    points = lo + (points - lo) * (hi - lo) / (points[-1] - lo)
    points[0], points[-1] = lo, hi   # exact walls: the B-spline space checks them
    return points


# --------------------------------------------------------------------------
# Vector AAA: one barycentric rational for every tensor entry.

@dataclass
class RationalContinuation:
    r"""Shared-denominator AAA rational fitted on real samples.

    ``r(z) = sum_j w_j F_j / (z - x_j) / sum_j w_j / (z - x_j)`` is linear in
    the sampled values ``F_j``, so one set of support points and weights
    continues every column of the local operator at once.  Its poles are the
    only places the continuation can blow up; a pole between the real axis and
    the path is exactly how analytic continuation manufactures spurious modes,
    so :meth:`poles_in_strip` must be empty for a trusted solve.
    """

    sample_x: np.ndarray
    support: np.ndarray
    weights: np.ndarray
    errors: np.ndarray

    @property
    def support_x(self) -> np.ndarray:
        return self.sample_x[self.support]

    def cauchy_weights(self, z: np.ndarray) -> np.ndarray:
        """Rows ``C(z)`` with ``r(z) = C(z) @ F[support]``."""
        z = np.asarray(z, dtype=np.complex128)
        diff = z[:, None] - self.support_x[None, :]
        exact = diff == 0.0
        diff[exact] = 1.0
        C = self.weights[None, :] / diff
        C = C / np.sum(C, axis=1, keepdims=True)
        rows = np.flatnonzero(np.any(exact, axis=1))
        for r in rows:
            C[r] = exact[r].astype(np.complex128)
        return C

    def matrix(self, sample_x: np.ndarray, points: np.ndarray, iy: np.ndarray) -> np.ndarray:
        """Dense ``(npoint, nsample)`` operator: continued values = matrix @ samples."""
        if not np.allclose(self.sample_x, sample_x):
            raise ValueError("The rational continuation was fitted on other samples.")
        out = np.zeros((np.size(points), self.sample_x.size), dtype=np.complex128)
        out[:, self.support] = self.cauchy_weights(np.asarray(points) + iy)
        return out

    def poles(self) -> np.ndarray:
        """Zeros of the denominator, from the arrowhead pencil."""
        from scipy.linalg import eig

        m = self.support.size
        E = np.zeros((m + 1, m + 1), dtype=np.complex128)
        E[0, 1:] = self.weights
        E[1:, 0] = 1.0
        E[1:, 1:] = np.diag(self.support_x)
        Bm = np.eye(m + 1, dtype=np.complex128)
        Bm[0, 0] = 0.0
        values = eig(E, Bm, right=False)
        return values[np.isfinite(values)]

    def poles_in_strip(self, domain: tuple[float, float], height: float) -> np.ndarray:
        """Poles with ``Re`` in the domain and ``|Im| <= height``."""
        p = self.poles()
        lo, hi = domain
        return p[(p.real >= lo) & (p.real <= hi) & (np.abs(p.imag) <= height)]


def fit_rational(sample_x: np.ndarray, columns: np.ndarray, rtol: float = 1e-10,
                 max_support: int = 120, rank_tol: float = 1e-13) -> RationalContinuation:
    r"""Vector AAA on the real samples ``columns`` (``nsample x ncolumn``).

    The columns are first compressed by an SVD to the rank that matters at
    ``rank_tol``; the greedy step then picks the sample with the largest error
    over all retained components.  Stops at ``rtol`` relative to the largest
    singular value.
    """
    x = np.asarray(sample_x, dtype=np.float64)
    F = np.asarray(columns, dtype=np.complex128)
    U, S, _ = np.linalg.svd(F, full_matrices=False)
    keep = S > rank_tol * S[0]
    G = U[:, keep] * S[keep]            # nsample x r, absolute scale S[0]
    scale = S[0]
    n = x.size
    support: list[int] = []
    approx = np.tile(G.mean(axis=0), (n, 1))
    errors = []
    weights = np.zeros(0, dtype=np.complex128)
    for _ in range(min(max_support, n - 1)):
        err = np.max(np.abs(G - approx), axis=1)
        if support:
            err[support] = 0.0
        errors.append(err.max() / scale)
        if errors[-1] <= rtol:
            break
        support.append(int(np.argmax(err)))
        rest = np.setdiff1d(np.arange(n), support)
        J = np.asarray(support)
        cauchy = 1.0 / (x[rest][:, None] - x[J][None, :])           # (n-m, m)
        # Loewner blocks stacked over components: (F_i - F_j)/(x_i - x_j).
        loewner = (G[rest][:, None, :] - G[J][None, :, :]) * cauchy[:, :, None]
        L = loewner.transpose(0, 2, 1).reshape(-1, J.size)
        _, _, Vh = np.linalg.svd(L, full_matrices=False)
        weights = Vh[-1].conj()
        num = cauchy @ (weights[:, None] * G[J])
        den = cauchy @ weights
        approx = G.copy()
        approx[rest] = num / den[:, None]
    return RationalContinuation(sample_x=x, support=np.asarray(support),
                                weights=weights, errors=np.asarray(errors))



# --------------------------------------------------------------------------
# Local least-squares polynomials: smoothing, pole-free continuation.

@dataclass
class LocalPolynomialContinuation:
    r"""Continue by a least-squares polynomial fitted around each point.

    For a point ``x_q`` the ``neighbors`` nearest samples are fitted, in the
    least-squares sense, by a polynomial of ``degree`` in ``(x - x_q)/s_q``
    (``s_q`` the half-span of those samples), which is then evaluated at the
    complex offset ``i*y_q``.  The result is linear in the samples, so one
    sparse operator continues every tensor entry.

    Unlike an interpolating spline it smooths sample noise instead of
    differentiating it, and unlike a global rational it has no poles at all.
    Its noise gain at each point is the 1-norm of that point's weights,
    available from :meth:`noise_gain`.
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
            w = e @ np.linalg.pinv(V)
            idx_all.append(idx)
            w_all.append(w)
        return idx_all, w_all

    def matrix(self, sample_x, points, iy):
        from scipy.sparse import csr_matrix

        idx_all, w_all = self._rows(sample_x, points, iy)
        rows = np.repeat(np.arange(len(idx_all)), [len(i) for i in idx_all])
        return csr_matrix((np.concatenate(w_all), (rows, np.concatenate(idx_all))),
                          shape=(len(idx_all), np.size(sample_x)))

    def noise_gain(self, sample_x, points, iy) -> np.ndarray:
        """Per-point amplification of i.i.d. sample noise (1-norm of the weights)."""
        _, w_all = self._rows(sample_x, points, iy)
        return np.array([np.sum(np.abs(w)) for w in w_all])


# --------------------------------------------------------------------------
# One global path for every crossing.

@dataclass(frozen=True)
class GlobalPath:
    r"""Global parabolic path ``z = x + i*alpha*(x - x0)*(x1 - x)``.

    One smooth deformation for the whole domain, zero at both walls.  Changing
    ``alpha`` moves the path everywhere at once, which makes the
    path-invariance test clean.  The price: a single sign is causal only at
    crossings whose branch slope has the matching sign -- ``alpha > 0`` passes
    falling branches (ds/dx < 0) on the causal side, ``alpha < 0`` rising
    ones -- and takes the anti-causal side at the others.
    """

    alpha: float
    domain: tuple[float, float] = (0.0, 1.0)

    # Attributes the reporting code expects from a path.
    centers: np.ndarray = field(default_factory=lambda: np.zeros(0))
    signs: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def amplitude(self) -> float:
        lo, hi = self.domain
        return abs(self.alpha) * 0.25 * (hi - lo) ** 2

    @property
    def width(self) -> float:
        return self.domain[1] - self.domain[0]

    def y(self, x):
        x = np.asarray(x, dtype=np.float64)
        lo, hi = self.domain
        return self.alpha * (x - lo) * (hi - x)

    def dy(self, x):
        x = np.asarray(x, dtype=np.float64)
        lo, hi = self.domain
        return self.alpha * (lo + hi - 2.0 * x)

    def z(self, x):
        x = np.asarray(x, dtype=np.float64)
        return x + 1.0j * self.y(x)

    def dz(self, x):
        return 1.0 + 1.0j * self.dy(x)

    def coverage(self) -> dict:
        return {"y_at_crossing": np.zeros(0), "conflict": np.zeros(0, dtype=bool)}
