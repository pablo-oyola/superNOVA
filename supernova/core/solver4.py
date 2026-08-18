r"""Compatible mixed-spline split-rational NEP solver.

``superSolver4`` keeps solver3's pointwise MHD tensors and SLEPc wrappers, but
uses the compatible radial trial/test pair

    [p1_H, xi^psi_H, xi^s_L, div_xi_L],

with all four equations tested in the lower-order space.  The global split
matrices are rectangular until the two endpoint constraints on ``xi^psi`` are
applied on the trial side.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import xarray as xr
from scipy.sparse import coo_matrix, csr_matrix

from ..radial import BSplineFE, CompatibleBSplinePair
from ._boundary4 import BoundaryCondition, idealSeparatrixBC, regularityAxisBC
from ._solver_kernel2 import (
    _as_coo_tensor,
    _validate_dpsi_hat_dx,
    solver_blocks2,
)
from .solver3 import (
    SplitRationalCISSNEPSolver,
    SplitRationalNEPSolver,
    _build_local_M_split,
    _matrix_stats,
    _slice_to_quadrature,
    superSolver3,
)


@dataclass(frozen=True)
class MixedFieldLayout:
    """Dimensions and field-major index maps for the solver4 operator."""

    n_high: int
    n_low: int
    n_modes: int

    field_names = ("p1", "xi_psi", "xi_s", "div_xi")

    def __post_init__(self):
        if self.n_high < 2:
            raise ValueError("n_high must be at least 2.")
        if self.n_low != self.n_high - 1:
            raise ValueError(
                "Compatible mixed spaces require n_low=n_high-1, got "
                f"n_high={self.n_high}, n_low={self.n_low}."
            )
        if self.n_modes < 1:
            raise ValueError("n_modes must be positive.")

    @property
    def n_rows(self) -> int:
        return 4 * self.n_low * self.n_modes

    @property
    def n_trial(self) -> int:
        return 2 * self.n_modes * (self.n_high + self.n_low)

    @property
    def trial_offsets(self) -> tuple[int, int, int, int]:
        nm = self.n_modes
        return (
            0,
            self.n_high * nm,
            2 * self.n_high * nm,
            (2 * self.n_high + self.n_low) * nm,
        )

    def row_index(self, field: int, radial: int, mode: int) -> int:
        """Return the low-test row for one field/radial/harmonic index."""
        self._validate_field_mode(field, mode)
        if not 0 <= radial < self.n_low:
            raise IndexError(f"Low-space radial index {radial} is out of range.")
        return field * self.n_low * self.n_modes + radial * self.n_modes + mode

    def trial_index(self, field: int, radial: int, mode: int) -> int:
        """Return the mixed trial column for one field/radial/harmonic index."""
        self._validate_field_mode(field, mode)
        n_radial = self.n_high if field < 2 else self.n_low
        if not 0 <= radial < n_radial:
            raise IndexError(
                f"Trial radial index {radial} is out of range for field {field}."
            )
        return self.trial_offsets[field] + radial * self.n_modes + mode

    def trial_slice(self, field: int) -> slice:
        """Return the full coefficient slice for one trial field."""
        if not 0 <= field < 4:
            raise IndexError(f"Field index {field} is out of range.")
        start = self.trial_offsets[field]
        n_radial = self.n_high if field < 2 else self.n_low
        return slice(start, start + n_radial * self.n_modes)

    def _validate_field_mode(self, field: int, mode: int):
        if not 0 <= field < 4:
            raise IndexError(f"Field index {field} is out of range.")
        if not 0 <= mode < self.n_modes:
            raise IndexError(f"Mode index {mode} is out of range.")


@dataclass(frozen=True)
class MixedRadialData:
    """Cached compatible bases and their shared quadrature representation."""

    pair: CompatibleBSplinePair
    points_q: np.ndarray
    weights_q: np.ndarray
    phi_high_q: np.ndarray
    phi_low_q: np.ndarray
    mass_low: csr_matrix
    G_low_high: csr_matrix


def _build_mixed_radial_data(high: BSplineFE) -> MixedRadialData:
    """Build and verify the shared high/low quadrature data."""
    pair = CompatibleBSplinePair.from_high(high)
    points_q, weights_q = pair.high.get_quadrature_points()
    points_low, weights_low = pair.low.get_quadrature_points()
    if not np.array_equal(points_q, points_low) or not np.array_equal(
        weights_q, weights_low
    ):
        raise RuntimeError("Compatible high and low spaces must share quadrature.")

    phi_high_q = pair.high.eval_basis(points_q).T
    phi_low_q = pair.low.eval_basis(points_q).T
    dphi_high_q = pair.high.eval_basis_derivative(points_q).T
    compatible_derivative = phi_low_q @ pair.derivative.toarray()
    if not np.allclose(
        dphi_high_q,
        compatible_derivative,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError("The compatible high/low derivative identity failed.")

    mass_low = csr_matrix(phi_low_q.T @ (weights_q[:, None] * phi_low_q))
    G_low_high = (mass_low @ pair.derivative).tocsr()
    direct_G = phi_low_q.T @ (weights_q[:, None] * dphi_high_q)
    if not np.allclose(
        G_low_high.toarray(),
        direct_G,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError("The mixed strong-derivative matrix identity failed.")

    return MixedRadialData(
        pair=pair,
        points_q=np.asarray(points_q),
        weights_q=np.asarray(weights_q),
        phi_high_q=np.asarray(phi_high_q),
        phi_low_q=np.asarray(phi_low_q),
        mass_low=mass_low,
        G_low_high=G_low_high,
    )


def _project_local_tensor_mixed(
    sb: solver_blocks2,
    M,
    phi_high_q: np.ndarray,
    phi_low_q: np.ndarray,
    weights_q: np.ndarray,
    G_low_high,
    include_derivative: bool = False,
) -> csr_matrix:
    r"""Project one local split tensor into the mixed field-major spaces.

    The Galerkin contribution is ``-<Phi_L, M Phi_trial>`` after multiplying
    its first two differential-equation rows by ``sb.dpsi_hat_dx``. For the
    first two state groups, the optional derivative contribution is
    ``+<Phi_L, d(Phi_H)/dx> = +G_low_high``.
    """
    M = _as_coo_tensor(M, "M")
    phi_high_q = np.asarray(phi_high_q)
    phi_low_q = np.asarray(phi_low_q)
    weights_q = np.asarray(weights_q)
    G_low_high = csr_matrix(G_low_high)

    if phi_high_q.ndim != 2:
        raise ValueError("phi_high_q must have shape (n_quad, n_high).")
    if phi_low_q.ndim != 2:
        raise ValueError("phi_low_q must have shape (n_quad, n_low).")

    nq = M.shape[0]
    n_modes = np.asarray(sb.m_range).size
    dpsi_hat_dx = _validate_dpsi_hat_dx(
        getattr(sb, "dpsi_hat_dx", 1.0)
    )
    n_state = 4 * n_modes
    if M.shape[1:] != (n_state, n_state):
        raise ValueError(
            f"M state dimensions must be {(n_state, n_state)}, got {M.shape[1:]}."
        )
    if phi_high_q.shape[0] != nq or phi_low_q.shape[0] != nq:
        raise ValueError("M and both radial bases must use the same quadrature points.")
    if weights_q.shape != (nq,):
        raise ValueError(f"weights_q must have shape {(nq,)}, got {weights_q.shape}.")

    n_high = phi_high_q.shape[1]
    n_low = phi_low_q.shape[1]
    layout = MixedFieldLayout(n_high=n_high, n_low=n_low, n_modes=n_modes)
    if G_low_high.shape != (n_low, n_high):
        raise ValueError(
            f"G_low_high must have shape {(n_low, n_high)}, "
            f"got {G_low_high.shape}."
        )

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []

    if include_derivative:
        derivative = coo_matrix(G_low_high)
        if derivative.nnz:
            for local_state in range(2 * n_modes):
                field, mode = divmod(local_state, n_modes)
                row_parts.append(
                    field * n_low * n_modes
                    + derivative.row.astype(np.int64, copy=False) * n_modes
                    + mode
                )
                col_parts.append(
                    layout.trial_offsets[field]
                    + derivative.col.astype(np.int64, copy=False) * n_modes
                    + mode
                )
                data_parts.append(derivative.data.astype(np.complex128, copy=False))

    q_coords, local_rows, local_cols = M.coords
    local_data = M.data.astype(np.complex128, copy=True)
    local_data[local_rows < 2 * n_modes] *= dpsi_hat_dx
    nonzero = local_data != 0.0
    q_coords = q_coords[nonzero]
    local_rows = local_rows[nonzero]
    local_cols = local_cols[nonzero]
    local_data = local_data[nonzero]

    for q in np.unique(q_coords):
        test_support = np.flatnonzero(phi_low_q[q] != 0.0)
        if test_support.size == 0:
            continue
        q_entries = np.flatnonzero(q_coords == q)
        for entry in q_entries:
            local_row = int(local_rows[entry])
            local_col = int(local_cols[entry])
            row_field, row_mode = divmod(local_row, n_modes)
            col_field, col_mode = divmod(local_col, n_modes)
            trial_basis = phi_high_q if col_field < 2 else phi_low_q
            trial_support = np.flatnonzero(trial_basis[q] != 0.0)
            if trial_support.size == 0:
                continue

            radial_rows = np.repeat(test_support, trial_support.size)
            radial_cols = np.tile(trial_support, test_support.size)
            radial_factor = (
                weights_q[q]
                * phi_low_q[q, radial_rows]
                * trial_basis[q, radial_cols]
            )
            row_parts.append(
                row_field * n_low * n_modes + radial_rows * n_modes + row_mode
            )
            col_parts.append(
                layout.trial_offsets[col_field]
                + radial_cols * n_modes
                + col_mode
            )
            data_parts.append(-radial_factor * local_data[entry])

    if not data_parts:
        return csr_matrix((layout.n_rows, layout.n_trial), dtype=np.complex128)

    projected = coo_matrix(
        (
            np.concatenate(data_parts),
            (np.concatenate(row_parts), np.concatenate(col_parts)),
        ),
        shape=(layout.n_rows, layout.n_trial),
        dtype=np.complex128,
    ).tocsr()
    projected.sum_duplicates()
    projected.eliminate_zeros()
    return projected


class superSolver4(superSolver3):
    """Solver3's split NEP infrastructure with compatible mixed splines."""

    def _get_mixed_radial_data(self) -> MixedRadialData:
        """Return compatible basis data cached for the current high space."""
        if not isinstance(self.radialfe, BSplineFE):
            raise TypeError("superSolver4 requires a BSplineFE high radial space.")
        cached = getattr(self, "_mixed_radial_data", None)
        if cached is None or cached.pair.high is not self.radialfe:
            cached = _build_mixed_radial_data(self.radialfe)
            self._mixed_radial_data = cached
        return cached

    def _mixed_layout(self, sb: solver_blocks2) -> MixedFieldLayout:
        mixed = self._get_mixed_radial_data()
        return MixedFieldLayout(
            n_high=mixed.pair.high.get_ndofs(),
            n_low=mixed.pair.low.get_ndofs(),
            n_modes=np.asarray(sb.m_range).size,
        )

    def _build_nep_boundary_condition(
        self,
        sb: solver_blocks2,
        bc_kind: str | None = "full",
    ) -> BoundaryCondition:
        """Build endpoint constraints in the mixed field-major trial space."""
        mixed = self._get_mixed_radial_data()
        layout = self._mixed_layout(sb)
        kind = self._normalize_bc_kind(bc_kind)

        if kind == "none":
            return BoundaryCondition(
                np.zeros((0, layout.n_trial), dtype=np.complex128)
            )

        axis = regularityAxisBC(mixed.pair.high, mixed.pair.low, sb.m_range)
        if kind == "axis":
            return axis

        edge = idealSeparatrixBC(
            mixed.pair.high,
            mixed.pair.low,
            layout.n_modes,
        )
        if kind == "edge":
            return edge
        return axis * edge

    def assemble_split_matrices(
        self,
        ntor: int,
        mmax: int | None = None,
        max_dm: int = 10,
        regularizer: float = 0.0,
        apply_bcs: bool = True,
        bc_kind: str | None = "full",
        bc_tol: float | None = None,
        verbose: bool = False,
        m_values: np.ndarray | None = None,
    ) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
        r"""Assemble the mixed ``T0``, ``T1`` and ``Tminus1`` matrices.

        ``m_values`` is an opt-in verification interface for an explicit,
        possibly noncontiguous signed harmonic list.  It bypasses the
        production q-band selector without changing or populating the
        production solver-block cache.
        """
        if m_values is None:
            sb = self.build_solver_blocks(ntor=ntor, mmax=mmax, max_dm=max_dm)
        else:
            if mmax is not None:
                raise ValueError("Specify either mmax or m_values, not both.")
            sb = self.build_solver_blocks_for_m_values(
                ntor=ntor,
                m_values=m_values,
                max_dm=max_dm,
            )
        mixed = self._get_mixed_radial_data()
        layout = self._mixed_layout(sb)

        if not np.allclose(
            np.asarray(sb.points_q),
            mixed.points_q,
            rtol=0.0,
            atol=1.0e-14,
        ):
            raise ValueError("Solver blocks and mixed bases use different quadrature points.")
        if not np.allclose(
            np.asarray(sb.weights_q),
            mixed.weights_q,
            rtol=0.0,
            atol=1.0e-14,
        ):
            raise ValueError("Solver blocks and mixed bases use different quadrature weights.")

        M0, M1, Mminus1 = _build_local_M_split(sb, regularizer=regularizer)
        nq = mixed.weights_q.size
        M0_q = _slice_to_quadrature(M0, nq, "M0")
        M1_q = _slice_to_quadrature(M1, nq, "M1")
        Mminus1_q = _slice_to_quadrature(Mminus1, nq, "Mminus1")

        projector_args = (
            sb,
            mixed.phi_high_q,
            mixed.phi_low_q,
            mixed.weights_q,
            mixed.G_low_high,
        )
        T0 = _project_local_tensor_mixed(
            projector_args[0],
            M0_q,
            *projector_args[1:],
            include_derivative=True,
        )
        T1 = _project_local_tensor_mixed(
            projector_args[0],
            M1_q,
            *projector_args[1:],
            include_derivative=False,
        )
        Tminus1 = _project_local_tensor_mixed(
            projector_args[0],
            Mminus1_q,
            *projector_args[1:],
            include_derivative=False,
        )

        expected_shape = (layout.n_rows, layout.n_trial)
        for name, matrix in (
            ("T0", T0),
            ("T1", T1),
            ("Tminus1", Tminus1),
        ):
            if matrix.shape != expected_shape:
                raise RuntimeError(
                    f"{name} mixed projection has shape {matrix.shape}; "
                    f"expected {expected_shape}."
                )

        if verbose:
            print("[superSolver4] assembled mixed split matrices before BC reduction")
            print(
                f"  ntor={ntor}, n_modes={layout.n_modes}, "
                f"n_high={layout.n_high}, n_low={layout.n_low}"
            )
            print(f"  {_matrix_stats('T0', T0)}")
            print(f"  {_matrix_stats('T1', T1)}")
            print(f"  {_matrix_stats('Tminus1', Tminus1)}")

        kind = self._normalize_bc_kind(bc_kind)
        if not apply_bcs or kind == "none":
            return T0, T1, Tminus1

        bc = self._build_nep_boundary_condition(sb, bc_kind=kind)
        bc_rank = int(np.linalg.matrix_rank(bc.C))
        expected_rank = 2 * layout.n_modes if kind == "full" else layout.n_modes
        if bc_rank != expected_rank:
            raise ValueError(
                f"solver4 {kind!r} boundary rank must be {expected_rank}, got {bc_rank}."
            )

        reduced = tuple(
            bc.reduce_sparse_trial_matrix(matrix, tol=bc_tol)
            for matrix in (T0, T1, Tminus1)
        )
        reduced_shape = (layout.n_rows, layout.n_trial - bc_rank)
        for name, matrix in zip(("T0", "T1", "Tminus1"), reduced):
            if matrix.shape != reduced_shape:
                raise RuntimeError(
                    f"Reduced {name} has shape {matrix.shape}; expected {reduced_shape}."
                )

        if kind == "full" and reduced_shape != (layout.n_rows, layout.n_rows):
            raise ValueError(
                "Full mixed boundary reduction must produce a square operator, "
                f"got {reduced_shape}."
            )

        if verbose:
            print("[superSolver4] split matrices after one-sided BC reduction")
            print(
                f"  bc_kind={kind}, bc_rank={bc_rank}, "
                f"full_shape={expected_shape}, reduced_shape={reduced_shape}"
            )
            for name, matrix in zip(("T0", "T1", "Tminus1"), reduced):
                print(f"  {_matrix_stats(name + '_reduced', matrix)}")
        return reduced

    def _require_full_bc_nep(self, apply_bcs: bool, bc_kind: str | None) -> None:
        """Reject rectangular diagnostic configurations before PETSc is loaded."""
        kind = self._normalize_bc_kind(bc_kind)
        if not apply_bcs or kind != "full":
            raise ValueError(
                "superSolver4 NEP solves require apply_bcs=True and bc_kind='full' "
                "so the compatible mixed operator is square. Partial/no-BC "
                "assemblies are available only through assemble_split_matrices()."
            )

    def build_split_nep_solver(
        self,
        ntor: int,
        target_omega: complex,
        region: dict | None,
        mmax: int | None = None,
        max_dm: int = 10,
        regularizer: float = 0.0,
        apply_bcs: bool = True,
        bc_kind: str | None = "full",
        bc_tol: float | None = None,
        prefix: str = "nova_split_nep",
        which: str = "TARGET_MAGNITUDE",
        nev: int = 6,
        tol: float = 1e-8,
        maxit: int = 100,
        options: dict | None = None,
        spectral_shift: complex | None = None,
        verbose: bool = False,
        compute_residuals: bool = True,
    ) -> SplitRationalNEPSolver:
        """Build NLEIGS only for the square full-BC mixed operator."""
        self._require_full_bc_nep(apply_bcs, bc_kind)
        nep = super().build_split_nep_solver(
            ntor=ntor,
            target_omega=target_omega,
            region=region,
            mmax=mmax,
            max_dm=max_dm,
            regularizer=regularizer,
            apply_bcs=apply_bcs,
            bc_kind=bc_kind,
            bc_tol=bc_tol,
            prefix=prefix,
            which=which,
            nev=nev,
            tol=tol,
            maxit=maxit,
            options=options,
            spectral_shift=spectral_shift,
            verbose=verbose,
            compute_residuals=compute_residuals,
        )
        self._configure_nleigs_shift_solvers(nep)
        return nep

    @staticmethod
    def _configure_nleigs_shift_solvers(nep: SplitRationalNEPSolver) -> None:
        """Use QR for NLEIGS shift solves unless PETSc options override it.

        One-sided mixed reduction does not guarantee a nonzero in every
        natural-order diagonal position even when the square matrix has a
        valid structural matching.  PETSc's natural-order LU rejects that
        pattern before numerical factorization, whereas QR supports the mixed
        ordering directly.  Calling ``setFromOptions`` last preserves an
        explicit user choice for the internal KSP/PC.
        """
        for ksp in nep.nep.getNLEIGSKSPs():
            ksp.getPC().setType("qr")
            ksp.setFromOptions()
            nep._vprint(
                "[superSolver4] configured NLEIGS mixed shift solve: "
                f"prefix={ksp.getOptionsPrefix()!r}, ksp={ksp.getType()}, "
                f"pc={ksp.getPC().getType()}"
            )

    def build_split_ciss_solver(
        self,
        ntor: int,
        region: dict,
        target_omega: complex | None = None,
        mmax: int | None = None,
        max_dm: int = 10,
        regularizer: float = 0.0,
        apply_bcs: bool = True,
        bc_kind: str | None = "full",
        bc_tol: float | None = None,
        prefix: str = "nova_split_ciss",
        which: str = "ALL",
        nev: int = 32,
        tol: float = 1e-8,
        maxit: int = 100,
        ciss_sizes: dict | None = None,
        ciss_extraction: str | None = None,
        ciss_refinement: dict | None = None,
        ciss_threshold: dict | None = None,
        options: dict | None = None,
        verbose: bool = False,
        compute_residuals: bool = True,
    ) -> SplitRationalCISSNEPSolver:
        """Build CISS only for the square full-BC mixed operator."""
        self._require_full_bc_nep(apply_bcs, bc_kind)
        return super().build_split_ciss_solver(
            ntor=ntor,
            region=region,
            target_omega=target_omega,
            mmax=mmax,
            max_dm=max_dm,
            regularizer=regularizer,
            apply_bcs=apply_bcs,
            bc_kind=bc_kind,
            bc_tol=bc_tol,
            prefix=prefix,
            which=which,
            nev=nev,
            tol=tol,
            maxit=maxit,
            ciss_sizes=ciss_sizes,
            ciss_extraction=ciss_extraction,
            ciss_refinement=ciss_refinement,
            ciss_threshold=ciss_threshold,
            options=options,
            verbose=verbose,
            compute_residuals=compute_residuals,
        )

    def _split_result_dataset(
        self,
        result,
        nep,
        ntor: int,
        target_omega: complex | None,
        target_s: complex,
        region: dict,
        mmax: int | None,
        max_dm: int,
        regularizer: float,
        apply_bcs: bool,
        bc_kind: str | None,
        bc_tol: float | None,
        verbose: bool,
        compute_residuals: bool,
        which: str,
        nep_type: str,
        description: str,
        extra_attrs: dict | None = None,
        m_values: np.ndarray | None = None,
    ) -> xr.Dataset:
        """Pack full mixed vectors and named high/low coefficient arrays."""
        if m_values is None:
            sb = self.build_solver_blocks(ntor=ntor, mmax=mmax, max_dm=max_dm)
            explicit_m_values = None
        else:
            if mmax is not None:
                raise ValueError("Specify either mmax or m_values, not both.")
            available_mmax = int(np.abs(self.M.m_pol).max())
            explicit_m_values = self._validate_explicit_m_values(
                m_values,
                available_mmax,
            )
            # Result packing needs only the harmonic labels. Reassembling the
            # uncached diagnostic blocks here would duplicate the expensive
            # signed-m assembly already performed for the eigensolve.
            sb = SimpleNamespace(m_range=explicit_m_values)
        layout = self._mixed_layout(sb)
        species_mass_amu = self._species_mass_amu()
        omega = np.sqrt(result.eigenvalues)

        solver_eigenvectors = np.asarray(result.eigenvectors, dtype=np.complex128)
        if solver_eigenvectors.ndim != 2:
            raise ValueError("NEP eigenvectors must be a two-dimensional array.")
        slepc_dofs = solver_eigenvectors.shape[1]
        eigenvectors = solver_eigenvectors
        bc_expanded_eigenvectors = False
        kind = self._normalize_bc_kind(bc_kind)
        slepc_eigenvector_space = "full"

        if apply_bcs and kind != "none":
            bc = self._build_nep_boundary_condition(sb, bc_kind=kind)
            if slepc_dofs == layout.n_trial:
                pass
            elif slepc_dofs == bc.get_sparse_free_dofs(tol=bc_tol):
                slepc_eigenvector_space = "reduced"
                eigenvectors = np.vstack(
                    [bc.expand_sparse(vector, tol=bc_tol) for vector in solver_eigenvectors]
                )
                bc_expanded_eigenvectors = True
            else:
                raise ValueError(
                    "SLEPc eigenvector width does not match either the full or "
                    f"reduced mixed space: got {slepc_dofs}, full={layout.n_trial}, "
                    f"reduced={bc.get_sparse_free_dofs(tol=bc_tol)}."
                )
        elif slepc_dofs != layout.n_trial:
            raise ValueError(
                f"Expected {layout.n_trial} full mixed DoFs, got {slepc_dofs}."
            )

        if eigenvectors.shape[1] != layout.n_trial:
            raise RuntimeError(
                f"Expanded mixed eigenvectors have width {eigenvectors.shape[1]}; "
                f"expected {layout.n_trial}."
            )

        n_eigenpairs = eigenvectors.shape[0]

        def field_coefficients(field: int, n_radial: int) -> np.ndarray:
            return eigenvectors[:, layout.trial_slice(field)].reshape(
                n_eigenpairs,
                n_radial,
                layout.n_modes,
            )

        data_vars = {
            "s": ("eigenpair", result.eigenvalues),
            "omega": ("eigenpair", omega),
            "eigenvector": (("eigenpair", "dof"), eigenvectors),
            "p1_coefficient": (
                ("eigenpair", "radial_dof_high", "m_pol"),
                field_coefficients(0, layout.n_high),
            ),
            "xi_psi_coefficient": (
                ("eigenpair", "radial_dof_high", "m_pol"),
                field_coefficients(1, layout.n_high),
            ),
            "xi_s_coefficient": (
                ("eigenpair", "radial_dof_low", "m_pol"),
                field_coefficients(2, layout.n_low),
            ),
            "div_xi_coefficient": (
                ("eigenpair", "radial_dof_low", "m_pol"),
                field_coefficients(3, layout.n_low),
            ),
        }
        if getattr(nep, "last_residual_norms", None) is not None:
            data_vars["residual_norm"] = (
                "eigenpair",
                nep.last_residual_norms,
            )
            data_vars["absolute_residual_norm"] = (
                "eigenpair",
                nep.last_absolute_residual_norms,
            )

        attrs = {
            "ntor": int(ntor),
            "split_variable": "omega2",
            "omega_normalization": "omegaA",
            "omega_is_species_mass_normalized": True,
            "density_includes_species_mass": True,
            "species_mass_amu": species_mass_amu,
            "target_omega": target_omega,
            "target_s": target_s,
            "mmax": None if mmax is None else int(mmax),
            "harmonic_selection": (
                "production_q_band"
                if explicit_m_values is None
                else "explicit_diagnostic"
            ),
            "explicit_m_values": (
                ""
                if explicit_m_values is None
                else np.array2string(explicit_m_values, separator=",")
            ),
            "max_dm": int(max_dm),
            "regularizer": regularizer,
            "apply_bcs": bool(apply_bcs),
            "bc_kind": kind,
            "region": str(region),
            "region_type": region["type"],
            "region_center": region["center"],
            "region_radius": region["radius"],
            "region_vscale": region["vscale"],
            "verbose": bool(verbose),
            "compute_residuals": bool(compute_residuals),
            "eigenvector_space": "full",
            "slepc_eigenvector_space": slepc_eigenvector_space,
            "bc_expanded_eigenvectors": bool(bc_expanded_eigenvectors),
            "slepc_dofs": int(slepc_dofs),
            "full_dofs": int(layout.n_trial),
            "equation_dofs": int(layout.n_rows),
            "n_radial_high": int(layout.n_high),
            "n_radial_low": int(layout.n_low),
            "n_modes": int(layout.n_modes),
            "radial_discretization": "compatible_mixed_bspline",
            "trial_layout": "field-major [p1_H, xi_psi_H, xi_s_L, div_xi_L]",
            "residual_space": "reduced" if apply_bcs and kind != "none" else "full",
            "nep_type": nep_type,
            "problem_type": "RATIONAL",
            "which": which,
            "nconv": int(result.nconv),
            "iterations": int(result.iterations),
            "reason": str(result.reason),
            "description": description,
        }
        if hasattr(nep, "spectral_shift"):
            attrs["spectral_shift"] = nep.spectral_shift
            attrs["slepc_variable"] = "s_minus_spectral_shift"
        else:
            attrs["slepc_variable"] = "s"
        if extra_attrs:
            attrs.update(extra_attrs)

        return xr.Dataset(
            data_vars=data_vars,
            coords={
                "eigenpair": np.arange(n_eigenpairs, dtype=np.int64),
                "dof": np.arange(layout.n_trial, dtype=np.int64),
                "radial_dof_high": np.arange(layout.n_high, dtype=np.int64),
                "radial_dof_low": np.arange(layout.n_low, dtype=np.int64),
                "m_pol": np.asarray(sb.m_range),
            },
            attrs=attrs,
        )


__all__ = [
    "MixedFieldLayout",
    "MixedRadialData",
    "_build_mixed_radial_data",
    "_project_local_tensor_mixed",
    "superSolver4",
]
