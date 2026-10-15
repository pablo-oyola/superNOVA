r"""The stability pencil assembled on a complex radial contour.

:mod:`supernova.operators.contour` holds the path and the continuation; this
module feeds it superNOVA's own local operator.  The operator is sampled on the
native equilibrium surfaces, where the profiles need no interpolation, by a
second :class:`StabilitySolver` whose "quadrature points" are those surfaces.
The production solver keeps its own blocks, its own mixed spaces and its own
boundary reduction, so a contour solve differs from a production solve only in
the local tensors handed to the projection (and, with :meth:`solve_window`,
in a radial mesh refined around the crossings).

At zero amplitude the contour pencil is not bit-identical to production: the
tensors reach the quadrature points through a continuation of the native
samples instead of linear interpolation of the profiles.  Compare contour runs
with the zero-amplitude contour run, not with production.

Sound models
------------
``sound="ideal"`` keeps the full ideal-MHD compressibility.  ``sound="nova"``
is legacy NOVA's default ``ksound=1``: the parallel-acoustic operator
``mu_0*gamma_s*p_0*G^H M G`` is dropped wherever it appears, which in the
split operator is exactly the ``1/s`` tensor ``Mminus1`` (the ``A22`` row of
the div-xi equation and the ``1/lambda`` part of ``D22``).  ``gamma_s*p_0``
survives in the fast-wave compressibility ``L``, the geodesic coupling
``A12`` and ``D12``, as NOVA keeps it in ``ad1``, ``sd``, ``gd`` and ``dd``.
The slow-sound continuum disappears and the pencil becomes linear in ``s``.
Physically this stands in for ion Landau damping, which suppresses the
ion-acoustic response that ideal MHD would otherwise resonate with.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_array
import xarray as xr

from supernova.operators.blocks import mu_0
from supernova.operators.contour import (
    CausalPath,
    ContinuumCrossings,
    RadialContour,
    RationalContinuation,
    causal_path,
    continue_tensor,
    continuum_image,
    find_continuum_crossings,
    fit_rational,
    graded_breakpoints,
    local_continuum_eigenvalues,
    LocalPolynomialContinuation,
)
from supernova.operators.local import _build_local_M_split
from supernova.operators.mixed import MixedFieldLayout, _build_mixed_radial_data
from supernova.radial import BSplineFE
from supernova.eigen.dispatch import build_eigensolver
from supernova.eigen.region import _as_scalar_complex, _ellipse_region_dict

SOUND_MODELS = ("ideal", "nova")


class _SampleGrid:
    """Stand-in radial space whose quadrature points are given samples.

    A :class:`StabilitySolver` built on it evaluates every profile exactly at
    ``x``: the solver appends the last mesh point as its edge point, so the
    first ``n - 1`` samples are offered as quadrature points and the edge
    completes the set without a duplicate (a repeated point would break the
    pressure-derivative finite difference).  Only the hooks the solver
    touches before projection are provided.
    """

    def __init__(self, x: np.ndarray, high):
        self.x = np.asarray(x, dtype=np.float64)
        self.high = high

    def get_quadrature_points(self):
        return self.x[:-1], np.ones(self.x.size - 1)

    def get_mesh(self):
        return self.x

    def eval_basis(self, x):
        return self.high.eval_basis(x)

    @property
    def weak_derivative_matrix(self):
        return self.high.weak_derivative_matrix


def native_sample_points(solver) -> np.ndarray:
    """The equilibrium's own radial surfaces in the solver coordinate."""
    magspace = getattr(solver.equ, "magspace", None)
    if magspace is not None and "psi" in magspace.coords:
        x = np.asarray(magspace.coords["psi"].values, dtype=np.float64)
    else:
        x = np.asarray(solver.equ("q", normalize=True).data.x, dtype=np.float64)
    return np.unique(x)


def _drop_rows(tensor: coo_array, row_start: int) -> coo_array:
    """Zero every entry whose local row is ``>= row_start``."""
    q, rows, cols = tensor.coords
    keep = rows < row_start
    return coo_array((tensor.data[keep], (q[keep], rows[keep], cols[keep])),
                     shape=tensor.shape, dtype=np.complex128)


@dataclass
class ContourSolution:
    """Eigenpairs of one window, with what is needed to rebuild the fields."""

    s: np.ndarray
    backward_error: np.ndarray
    vectors: np.ndarray            # (eigenpair, reduced dof)
    path: object
    mixed: object
    prolongation: object
    layout: MixedFieldLayout
    m_pol: np.ndarray
    breakpoints: np.ndarray

    def accepted(self, threshold: float = 1e-8) -> np.ndarray:
        return self.backward_error <= threshold

    def xi_psi(self, x: np.ndarray, index: int | None = None) -> np.ndarray:
        """``xi^psi`` harmonics on the path parameter ``x``: (eigenpair, x, m)."""
        x = np.asarray(x, dtype=np.float64)
        basis = np.asarray(self.mixed.pair.high.eval_basis(x))  # (n_high, nx)
        rows = range(self.s.size) if index is None else [index]
        out = []
        for k in rows:
            full = self.prolongation @ self.vectors[k]
            coeff = full[self.layout.trial_slice(1)].reshape(self.layout.n_high,
                                                              self.layout.n_modes)
            out.append(basis.T @ coeff)
        return np.asarray(out)


class ContourProblem:
    r"""One harmonic band of one solver, ready to assemble on any path.

    Parameters
    ----------
    solver
        A built :class:`~supernova.solve.stability.StabilitySolver`.
    ntor, max_dm, mmax
        As for :meth:`StabilitySolver.solve_pep`.
    regularizer
        The production Landau-like shift; zero for a contour solve.
    sample_x
        Real sample points for the continuation; the native surfaces by
        default.  They must cover the radial domain.
    sound
        ``"ideal"`` or ``"nova"`` (legacy ``ksound=1``); see the module notes.
    continuation
        ``"spline"`` (cubic spline, Taylor series of each piece) or ``"aaa"``
        (one shared-denominator rational for every entry, with pole report).
    """

    def __init__(self, solver, ntor: int, max_dm: int = 10, mmax: int | None = None,
                 regularizer: float = 0.0, sample_x: np.ndarray | None = None,
                 sound: str = "ideal", continuation: str = "spline",
                 aaa_rtol: float = 1e-10):
        from supernova.solve.stability import StabilitySolver

        if sound not in SOUND_MODELS:
            raise ValueError(f"sound must be one of {SOUND_MODELS}, got {sound!r}.")
        if continuation not in ("spline", "aaa", "local"):
            raise ValueError("continuation must be 'spline', 'aaa' or 'local'.")
        self.solver = solver
        self.ntor = int(ntor)
        self.max_dm = int(max_dm)
        self.mmax = mmax
        self.regularizer = float(regularizer)
        self.sound = sound
        self.continuation = continuation
        self.sb = solver.build_solver_blocks(ntor=ntor, mmax=mmax, max_dm=max_dm)
        self.mixed = solver._get_mixed_radial_data()
        mesh = np.asarray(solver.radialfe.get_mesh(), dtype=np.float64)
        self.domain = (float(mesh[0]), float(mesh[-1]))
        self.order = int(solver.radialfe.order)
        self.quad_order = int(solver.radialfe.quadrature_order)

        if sample_x is None:
            sample_x = native_sample_points(solver)
        sample_x = np.asarray(sample_x, dtype=np.float64)
        if sample_x[0] > self.domain[0] or sample_x[-1] < self.domain[1]:
            raise ValueError("The continuation samples must cover the radial domain.")
        self.sample_x = sample_x

        sampler = StabilitySolver(solver.equ, solver.cont,
                                  _SampleGrid(sample_x, solver.radialfe))
        sbs = sampler._assemble_solver_blocks_for_m_range(
            ntor=self.ntor, m_range=np.asarray(self.sb.m_range), max_dm=self.max_dm,
        )
        n = sample_x.size
        self.n_modes = int(np.asarray(self.sb.m_range).size)
        M0, M1, Mm1 = _build_local_M_split(sbs, regularizer=self.regularizer)
        M0, M1, Mm1 = M0[:n], M1[:n], Mm1[:n]
        A, B = sbs.A[:n], sbs.B[:n]
        if sound == "nova":
            Mm1 = coo_array(Mm1.shape, dtype=np.complex128)
            A = _drop_rows(A, self.n_modes)     # A22 row: the acoustic operator
        self.local_samples = (M0, M1, Mm1)
        self.A, self.B = A, B
        self.rho_mu0 = mu_0 * np.asarray(sbs.density[:n], dtype=np.complex128)
        self.continuum_s = local_continuum_eigenvalues(self.A, self.B, self.rho_mu0)

        self.rational = None
        if continuation == "aaa":
            self.rational = fit_rational(sample_x, self._stacked_columns(), rtol=aaa_rtol)
        elif continuation == "local":
            self.rational = LocalPolynomialContinuation()

    # ------------------------------------------------------------ continuation

    def _stacked_columns(self) -> np.ndarray:
        """Every sampled entry, each tensor scaled by its own maximum."""
        n = self.sample_x.size
        blocks = []
        for tensor in (*self.local_samples, self.A, self.B):
            if tensor.nnz == 0:
                continue
            q, rows, cols = tensor.coords
            key = rows.astype(np.int64) * tensor.shape[2] + cols
            keys, inverse = np.unique(key, return_inverse=True)
            dense = np.zeros((n, keys.size), dtype=np.complex128)
            np.add.at(dense, (q, inverse), tensor.data)
            blocks.append(dense / max(np.max(np.abs(dense)), 1e-300))
        rho = self.rho_mu0[:, None]
        blocks.append(rho / np.max(np.abs(rho)))
        return np.concatenate(blocks, axis=1)

    def continuation_report(self, path) -> dict:
        """AAA support size, fit error, and poles the path could reach."""
        if self.rational is None:
            return {"method": "spline"}
        if isinstance(self.rational, LocalPolynomialContinuation):
            points = self.mixed.points_q
            iy = 1.0j * path.y(points) if path is not None else np.zeros(points.size)
            gain = self.rational.noise_gain(self.sample_x, points, iy)
            return {"method": "local", "poles_in_strip": np.zeros(0),
                    "max_noise_gain": float(gain.max()),
                    "median_noise_gain": float(np.median(gain))}
        xs = np.linspace(*self.domain, 4000)
        height = 1.5 * float(np.max(np.abs(path.y(xs)))) if path is not None else 0.0
        strip = self.rational.poles_in_strip(self.domain, max(height, 1e-12))
        poles = self.rational.poles()
        inside = poles[(poles.real >= self.domain[0]) & (poles.real <= self.domain[1])]
        return {
            "method": "aaa",
            "support": int(self.rational.support.size),
            "fit_error": float(self.rational.errors[-1]),
            "poles_in_strip": strip,
            "nearest_pole_height": float(np.min(np.abs(inside.imag))) if inside.size else np.inf,
        }

    # ------------------------------------------------------------------ path

    def crossings(self, s_target: complex) -> ContinuumCrossings:
        """Continuum crossings of ``Re(s_target)`` inside the radial domain."""
        found = find_continuum_crossings(self.sample_x, self.continuum_s, s_target)
        inside = (found.x > self.domain[0]) & (found.x < self.domain[1])
        return ContinuumCrossings(
            s_target=found.s_target,
            x=found.x[inside],
            slope_sign=found.slope_sign[inside],
            slope=found.slope[inside],
            counts_below=found.counts_below,
            sample_x=found.sample_x,
        )

    def contour(self, s_target: complex, amplitude: float, width: float,
                edge: float = 0.35) -> RadialContour:
        """Uniform-bump causal path (first experiments)."""
        crossings = self.crossings(s_target)
        path = crossings.contour(amplitude, width, edge=edge,
                                 omega_sign=np.sign(np.sqrt(complex(s_target)).real) or 1.0)
        return RadialContour(path.centers, path.signs, path.amplitude, path.width,
                             edge=path.edge, slopes=path.slopes, domain=self.domain)

    def causal_path(self, s_target: complex, amplitude: float, width: float,
                    edge: float = 0.35, gap_fraction: float = 0.35) -> CausalPath:
        """Gap-aware causal path for ``s_target``'s crossings."""
        return causal_path(self.crossings(s_target), amplitude, width, self.domain,
                           edge=edge, gap_fraction=gap_fraction,
                           omega_sign=np.sign(np.sqrt(complex(s_target)).real) or 1.0)

    def refined_breakpoints(self, path, base_spans: int, growth: float = 0.25,
                            min_size: float = 0.0) -> np.ndarray:
        """Base mesh refined where ``path`` needs it (see CausalPath.resolution)."""
        lo, hi = self.domain
        base = (hi - lo) / base_spans
        if not isinstance(path, CausalPath) or path.centers.size == 0 or path.amplitude == 0.0:
            return np.linspace(lo, hi, base_spans + 1)
        local = np.maximum(path.resolution(), min_size)
        return graded_breakpoints(self.domain, base, path.centers, local, growth=growth)

    # -------------------------------------------------------------- assembly

    def local(self, contour, points: np.ndarray | None = None, order: int = 3):
        """Continued ``(M0, M1, Mminus1)`` at ``points`` (production quadrature)."""
        points = self.mixed.points_q if points is None else points
        scaled = 2 * self.n_modes
        return tuple(
            continue_tensor(self.sample_x, tensor, points, contour,
                            scaled_rows=scaled, order=order, rational=self.rational)
            for tensor in self.local_samples
        )

    def matrices(self, contour, order: int = 3, breakpoints: np.ndarray | None = None):
        """BC-reduced ``(T0, T1, Tminus1)`` on the path, optionally on a new mesh."""
        sb, mixed = self._space(breakpoints)
        return self.solver._assemble_split_matrices_from_local(
            sb, mixed, self.local(contour, points=mixed.points_q, order=order),
        ), sb, mixed

    def _space(self, breakpoints):
        if breakpoints is None:
            return self.sb, self.mixed
        high = BSplineFE(knots=np.asarray(breakpoints, dtype=np.float64), order=self.order,
                         domain=self.domain, quad_order=self.quad_order)
        mixed = _build_mixed_radial_data(high)
        sb = copy.copy(self.sb)
        sb.points_q = mixed.points_q
        sb.weights_q = mixed.weights_q
        return sb, mixed

    def image(self, contour, n: int = 1200) -> tuple[np.ndarray, np.ndarray]:
        """Points along the path and the local continuum there."""
        x = np.linspace(self.domain[0], self.domain[1], n)
        return x, continuum_image(self.sample_x, self.A, self.B, self.rho_mu0, contour, x,
                                  rational=self.rational)

    def _eigensolver(self, T0, T1, Tminus1, target_s, region, nev, tol, maxit, options, verbose):
        gamma = 0.0 if self.sound == "nova" else self.solver.gamma_s
        return build_eigensolver(
            gamma_s=gamma, T0=T0, T1=T1, Tminus1=Tminus1,
            target_s=target_s, region=region, prefix="nova_contour_pep",
            pep_type="TOAR", st_type="SINVERT", which="TARGET_MAGNITUDE",
            nev=nev, tol=tol, maxit=maxit, options=options, verbose=verbose,
            compute_residuals=True,
        )

    def solve_window(self, path, target_omega: complex, region: dict | None,
                     breakpoints: np.ndarray | None = None, nev: int = 20,
                     tol: float = 1e-8, maxit: int = 300, options: dict | None = None,
                     region_vscale: float | None = 1.0, order: int = 3,
                     verbose: bool = False) -> ContourSolution:
        """Solve one window on ``path``; ``breakpoints`` select a refined mesh."""
        target_omega = _as_scalar_complex(target_omega, "target_omega")
        target_s = target_omega**2
        region = _ellipse_region_dict(region, target_s)
        if region_vscale is not None:
            region = dict(region, vscale=float(region_vscale))
        (T0, T1, Tminus1), sb, mixed = self.matrices(path, order=order, breakpoints=breakpoints)
        bc = self.solver._build_nep_boundary_condition(sb, bc_kind="full", mixed=mixed)
        prolongation = bc.get_sparse_prolongation()
        layout = self.solver._mixed_layout(sb, mixed=mixed)
        eigensolver = self._eigensolver(T0, T1, Tminus1, target_s, region, nev, tol,
                                        maxit, options, verbose)
        try:
            result = eigensolver.solve(max_eigenpairs=None, allow_empty=True)
            s = np.asarray(result.eigenvalues, dtype=np.complex128)
            berr = np.asarray(eigensolver.last_backward_errors if s.size else [], dtype=float)
            vectors = np.asarray(result.eigenvectors, dtype=np.complex128)
        finally:
            eigensolver.destroy()
        bp = (np.asarray(breakpoints) if breakpoints is not None
              else np.asarray(self.solver.radialfe.get_mesh()))
        return ContourSolution(s=s, backward_error=berr, vectors=vectors, path=path,
                               mixed=mixed, prolongation=prolongation, layout=layout,
                               m_pol=np.asarray(self.sb.m_range), breakpoints=bp)

    def solve(self, contour, target_omega: complex, region: dict | None,
              nev: int = 20, tol: float = 1e-8, maxit: int = 100,
              options: dict | None = None, order: int = 3,
              region_vscale: float | None = None,
              verbose: bool = False) -> xr.Dataset:
        """Solve on the production mesh and pack the result like ``solve_pep``.

        ``region_vscale`` overrides the ellipse's vertical scale: on a deformed
        path the rotated continuum and the resonances sit off the real axis,
        outside the flat default ellipse.  An empty result is returned, not
        raised.
        """
        target_omega = _as_scalar_complex(target_omega, "target_omega")
        target_s = target_omega**2
        region = _ellipse_region_dict(region, target_s)
        if region_vscale is not None:
            region = dict(region, vscale=float(region_vscale))
        (T0, T1, Tminus1), _, _ = self.matrices(contour, order=order)
        eigensolver = self._eigensolver(T0, T1, Tminus1, target_s, region, nev, tol,
                                        maxit, options, verbose)
        try:
            result = eigensolver.solve(max_eigenpairs=None, allow_empty=True)
            if np.asarray(result.eigenvalues).size == 0:
                return xr.Dataset(
                    {"s": ("eigenpair", np.zeros(0, dtype=np.complex128))},
                    attrs={"contour_amplitude": float(contour.amplitude)},
                )
            dataset = self.solver._split_result_dataset(
                result=result, nep=eigensolver, ntor=self.ntor,
                target_omega=target_omega, target_s=target_s, region=region,
                mmax=self.mmax, max_dm=self.max_dm, regularizer=self.regularizer,
                apply_bcs=True, bc_kind="full", bc_tol=None, verbose=verbose,
                compute_residuals=True, which="TARGET_MAGNITUDE",
                nep_type=getattr(eigensolver, "backend", "PEP"),
                description="Contour-deformed pencil on the production mesh.",
                extra_attrs={
                    "contour_amplitude": float(contour.amplitude),
                    "contour_width": float(contour.width),
                    "contour_centers": np.asarray(contour.centers),
                    "contour_signs": np.asarray(contour.signs),
                    "continuation": self.continuation,
                    "sound_model": self.sound,
                    "time_dependence": "exp(-i*omega*t); damped means Im(s) < 0",
                },
            )
        finally:
            eigensolver.destroy()
        return dataset
