r"""The stability pencil assembled on a global complex radial path.

:mod:`supernova.operators.contour` holds the path and the continuation; this
module feeds it superNOVA's own local operator.  The operator is sampled on the
native equilibrium surfaces, where the profiles need no interpolation, by a
second :class:`StabilitySolver` whose "quadrature points" are those surfaces.
The production solver keeps its own blocks, mixed spaces and boundary
reduction, so a path solve differs from a production solve only in the local
tensors handed to the projection and, optionally, in the radial mesh.

At ``alpha = 0`` the path pencil is not bit-identical to production: the
tensors reach the quadrature points through the continuation of the native
samples instead of linear interpolation of the profiles.  Compare path runs
with the ``alpha = 0`` path run, not with production.

Typical use: solve one window on the real axis and on ``alpha = +-a`` (and
``+-a/2``), with the target on the candidate and a small ``nev``; a discrete
eigenvalue keeps its frequency and shape (harmonic correlation) on every path,
continuum approximants move.  See :mod:`supernova.operators.contour` for the
limit on ``alpha`` near gap edges.

``sound="nova"`` is legacy NOVA's default ``ksound=1``, as
``StabilitySolver.solve_pep(sound_model="nova")``: the ``1/s`` parallel-acoustic
tensor ``Mminus1`` is dropped and the pencil becomes linear in ``s``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_array

from supernova.eigen.dispatch import build_eigensolver
from supernova.eigen.region import _as_scalar_complex, _ellipse_region_dict
from supernova.operators.contour import GlobalPath, LocalPolynomialContinuation, continue_tensor
from supernova.operators.local import _build_local_M_split
from supernova.operators.mixed import MixedFieldLayout, _build_mixed_radial_data
from supernova.radial import BSplineFE
from supernova.solve.stability import StabilitySolver, _validate_sound_model


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


@dataclass
class ContourSolution:
    """Eigenpairs of one window, with what is needed to rebuild the fields."""

    s: np.ndarray
    backward_error: np.ndarray
    vectors: np.ndarray            # (eigenpair, reduced dof)
    path: GlobalPath
    mixed: object
    prolongation: object
    layout: MixedFieldLayout
    m_pol: np.ndarray
    breakpoints: np.ndarray

    def accepted(self, threshold: float = 1e-8) -> np.ndarray:
        """Eigenpairs within the production backward-error gate."""
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
    r"""One harmonic band of one solver, ready to assemble on any global path.

    Parameters
    ----------
    solver
        A built :class:`~supernova.solve.stability.StabilitySolver`.
    ntor, max_dm, mmax
        As for :meth:`StabilitySolver.solve_pep`.
    regularizer
        The production Landau-like shift; zero for a path solve.
    sample_x
        Real sample points for the continuation; the native surfaces by
        default.  They must cover the radial domain.
    sound
        ``"ideal"`` or ``"nova"`` (legacy ``ksound=1``).
    continuation
        Defaults to :class:`LocalPolynomialContinuation` (degree 5, 14
        neighbours).
    """

    def __init__(self, solver, ntor: int, max_dm: int = 10, mmax: int | None = None,
                 regularizer: float = 0.0, sample_x: np.ndarray | None = None,
                 sound: str = "ideal",
                 continuation: LocalPolynomialContinuation | None = None):
        self.solver = solver
        self.ntor = int(ntor)
        self.max_dm = int(max_dm)
        self.mmax = mmax
        self.regularizer = float(regularizer)
        self.sound = _validate_sound_model(sound)
        self.continuation = LocalPolynomialContinuation() if continuation is None else continuation
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
        if self.sound == "nova":
            Mm1 = coo_array(Mm1.shape, dtype=np.complex128)
        self.local_samples = (M0, M1, Mm1)

    def path(self, alpha: float) -> GlobalPath:
        """The global path of amplitude ``alpha`` on this problem's domain."""
        return GlobalPath(float(alpha), domain=self.domain)

    def local(self, path: GlobalPath, points: np.ndarray | None = None):
        """Continued ``(M0, M1, Mminus1)`` at ``points`` (production quadrature)."""
        points = self.mixed.points_q if points is None else points
        scaled = 2 * self.n_modes
        return tuple(
            continue_tensor(self.sample_x, tensor, points, path, self.continuation,
                            scaled_rows=scaled)
            for tensor in self.local_samples
        )

    def matrices(self, path: GlobalPath, breakpoints: np.ndarray | None = None):
        """BC-reduced ``(T0, T1, Tminus1)`` on the path, optionally on a new mesh."""
        sb, mixed = self._space(breakpoints)
        return self.solver._assemble_split_matrices_from_local(
            sb, mixed, self.local(path, points=mixed.points_q),
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

    def _eigensolver(self, T0, T1, Tminus1, target_s, region, nev, tol, maxit, options, verbose):
        gamma = 0.0 if self.sound == "nova" else self.solver.gamma_s
        return build_eigensolver(
            gamma_s=gamma, T0=T0, T1=T1, Tminus1=Tminus1,
            target_s=target_s, region=region, prefix="nova_contour_pep",
            pep_type="TOAR", st_type="SINVERT", which="TARGET_MAGNITUDE",
            nev=nev, tol=tol, maxit=maxit, options=options, verbose=verbose,
            compute_residuals=True,
        )

    def solve_window(self, path: GlobalPath, target_omega: complex, region: dict | None,
                     breakpoints: np.ndarray | None = None, nev: int = 8,
                     tol: float = 1e-9, maxit: int = 300, options: dict | None = None,
                     region_vscale: float | None = 1.0,
                     verbose: bool = False) -> ContourSolution:
        """Solve one window on ``path``; ``breakpoints`` select another radial mesh.

        Put the target on the candidate and keep ``nev`` small: in a wide
        region the eigenvalues nearest the target are returned quickly, while
        a large ``nev`` in a sparse region iterates to ``maxit``.
        """
        target_omega = _as_scalar_complex(target_omega, "target_omega")
        target_s = target_omega**2
        region = _ellipse_region_dict(region, target_s)
        if region_vscale is not None:
            region = dict(region, vscale=float(region_vscale))
        (T0, T1, Tminus1), sb, mixed = self.matrices(path, breakpoints=breakpoints)
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
