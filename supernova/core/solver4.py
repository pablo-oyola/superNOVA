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


def _build_mixed_radial_data(
    high: BSplineFE,
    points_q: np.ndarray | None = None,
    weights_q: np.ndarray | None = None,
    validate: bool = True,
) -> MixedRadialData:
    """Build compatible basis data on native or externally supplied quadrature.

    Supplying ``points_q`` and ``weights_q`` avoids generating a native
    quadrature for the compatible low space.  This is the multigrid path: all
    nested levels are evaluated on the identical finest-grid rule.

    ``validate`` runs two dense identity checks on the compatible derivative.
    They are worth their cost once per basis but not once per multigrid level,
    so the nested-level construction turns them off after the first level has
    proved the construction sound.
    """
    external_quadrature = points_q is not None or weights_q is not None
    if external_quadrature:
        if points_q is None or weights_q is None:
            raise ValueError("points_q and weights_q must be supplied together.")
        points_q = np.asarray(points_q, dtype=np.float64)
        weights_q = np.asarray(weights_q, dtype=np.float64)
        if points_q.ndim != 1 or weights_q.shape != points_q.shape:
            raise ValueError(
                "External quadrature points and weights must be equal-length "
                "one-dimensional arrays."
            )
        if points_q.size == 0:
            raise ValueError("External quadrature must contain at least one point.")
        if np.any(~np.isfinite(points_q)) or np.any(~np.isfinite(weights_q)):
            raise ValueError("External quadrature points and weights must be finite.")
        if np.any(np.diff(points_q) <= 0.0) or np.any(weights_q <= 0.0):
            raise ValueError(
                "External quadrature points must be strictly increasing and "
                "weights must be positive."
            )
        pair = CompatibleBSplinePair.from_high(
            high,
            build_low_matrices=False,
            verification_points=points_q,
        )
    else:
        pair = CompatibleBSplinePair.from_high(high)
        points_q, weights_q = pair.high.get_quadrature_points()
        points_low, weights_low = pair.low.get_quadrature_points()
        if not np.array_equal(points_q, points_low) or not np.array_equal(
            weights_q, weights_low
        ):
            raise RuntimeError("Compatible high and low spaces must share quadrature.")

    phi_high_q = pair.high.eval_basis(points_q).T
    phi_low_q = pair.low.eval_basis(points_q).T
    mass_low = csr_matrix(phi_low_q.T @ (weights_q[:, None] * phi_low_q))
    G_low_high = (mass_low @ pair.derivative).tocsr()

    if validate:
        dphi_high_q = pair.high.eval_basis_derivative(points_q).T
        compatible_derivative = phi_low_q @ pair.derivative.toarray()
        if not np.allclose(
            dphi_high_q,
            compatible_derivative,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise RuntimeError("The compatible high/low derivative identity failed.")

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
            fields, modes = np.divmod(
                np.arange(2 * n_modes, dtype=np.int64), n_modes
            )
            offsets = np.asarray(layout.trial_offsets, dtype=np.int64)
            row_parts.append(
                (
                    fields[:, None] * n_low * n_modes
                    + derivative.row.astype(np.int64, copy=False)[None, :] * n_modes
                    + modes[:, None]
                ).ravel()
            )
            col_parts.append(
                (
                    offsets[fields][:, None]
                    + derivative.col.astype(np.int64, copy=False)[None, :] * n_modes
                    + modes[:, None]
                ).ravel()
            )
            data_parts.append(
                np.broadcast_to(
                    derivative.data.astype(np.complex128, copy=False),
                    (fields.size, derivative.nnz),
                ).ravel()
            )

    q_coords, local_rows, local_cols = M.coords
    local_data = M.data.astype(np.complex128, copy=True)
    local_data[local_rows < 2 * n_modes] *= dpsi_hat_dx
    nonzero = local_data != 0.0
    q_coords = q_coords[nonzero].astype(np.int64, copy=False)
    local_rows = local_rows[nonzero].astype(np.int64, copy=False)
    local_cols = local_cols[nonzero].astype(np.int64, copy=False)
    local_data = local_data[nonzero]

    if local_data.size:
        row_field, row_mode = np.divmod(local_rows, n_modes)
        col_field, col_mode = np.divmod(local_cols, n_modes)
        trial_offsets = np.asarray(layout.trial_offsets, dtype=np.int64)

        # Quadrature points that see the same basis functions form one element.
        # Grouping by the support signature rather than by position keeps this
        # correct for any point ordering, and the element blocks below then sum
        # over quadrature before any global index is materialised.
        support_signature = np.concatenate(
            [phi_low_q != 0.0, phi_high_q != 0.0], axis=1
        )
        _, element_of_q = np.unique(support_signature, axis=0, return_inverse=True)
        element_of_q = np.ravel(element_of_q)
        order = np.argsort(element_of_q[q_coords], kind="stable")
        entry_bounds = np.searchsorted(
            element_of_q[q_coords][order], np.arange(element_of_q.max() + 2)
        )

        for element in range(entry_bounds.size - 1):
            first, last = entry_bounds[element], entry_bounds[element + 1]
            if first == last:
                continue
            entries = order[first:last]
            quad = np.flatnonzero(element_of_q == element)

            # One support per element: the union over its quadrature points.
            # Any spurious index contributes an exact zero and is dropped by
            # eliminate_zeros below, so a slightly wide support stays correct.
            test_support = np.flatnonzero(np.any(phi_low_q[quad] != 0.0, axis=0))
            if test_support.size == 0:
                continue
            test_values = phi_low_q[np.ix_(quad, test_support)]

            local_q = np.searchsorted(quad, q_coords[entries])
            weighted = local_data[entries] * weights_q[q_coords[entries]]

            for low_trial in (False, True):
                selected = (col_field[entries] >= 2) == low_trial
                if not selected.any():
                    continue
                trial_basis = phi_low_q if low_trial else phi_high_q
                trial_support = np.flatnonzero(
                    np.any(trial_basis[quad] != 0.0, axis=0)
                )
                if trial_support.size == 0:
                    continue
                trial_values = trial_basis[np.ix_(quad, trial_support)]

                chosen = entries[selected]
                # Distinct (row, col) pairs in this element and trial space.
                pair_key = local_rows[chosen] * n_state + local_cols[chosen]
                pairs, pair_of_entry = np.unique(pair_key, return_inverse=True)
                accumulated = np.zeros(
                    (pairs.size, quad.size), dtype=np.complex128
                )
                np.add.at(
                    accumulated,
                    (pair_of_entry, local_q[selected]),
                    weighted[selected],
                )

                block = -np.einsum(
                    "gq,qi,qj->gij",
                    accumulated,
                    test_values,
                    trial_values,
                    optimize=True,
                )
                pair_rows, pair_cols = np.divmod(pairs, n_state)
                pair_row_field, pair_row_mode = np.divmod(pair_rows, n_modes)
                pair_col_field, pair_col_mode = np.divmod(pair_cols, n_modes)
                row_parts.append(
                    np.broadcast_to(
                        pair_row_field[:, None, None] * n_low * n_modes
                        + test_support[None, :, None] * n_modes
                        + pair_row_mode[:, None, None],
                        block.shape,
                    ).ravel()
                )
                col_parts.append(
                    np.broadcast_to(
                        trial_offsets[pair_col_field][:, None, None]
                        + trial_support[None, None, :] * n_modes
                        + pair_col_mode[:, None, None],
                        block.shape,
                    ).ravel()
                )
                data_parts.append(block.ravel())

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

    def _mixed_layout(
        self,
        sb: solver_blocks2,
        mixed: MixedRadialData | None = None,
    ) -> MixedFieldLayout:
        mixed = self._get_mixed_radial_data() if mixed is None else mixed
        return MixedFieldLayout(
            n_high=mixed.pair.high.get_ndofs(),
            n_low=mixed.pair.low.get_ndofs(),
            n_modes=np.asarray(sb.m_range).size,
        )

    def _build_nep_boundary_condition(
        self,
        sb: solver_blocks2,
        bc_kind: str | None = "full",
        mixed: MixedRadialData | None = None,
    ) -> BoundaryCondition:
        """Build endpoint constraints in the mixed field-major trial space."""
        mixed = self._get_mixed_radial_data() if mixed is None else mixed
        layout = self._mixed_layout(sb, mixed=mixed)
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
        M0, M1, Mminus1 = _build_local_M_split(sb, regularizer=regularizer)
        return self._assemble_split_matrices_from_local(
            sb,
            mixed,
            (M0, M1, Mminus1),
            apply_bcs=apply_bcs,
            bc_kind=bc_kind,
            bc_tol=bc_tol,
            verbose=verbose,
        )

    def _assemble_split_matrices_from_local(
        self,
        sb: solver_blocks2,
        mixed: MixedRadialData,
        local_split: tuple,
        *,
        apply_bcs: bool = True,
        bc_kind: str | None = "full",
        bc_tol: float | None = None,
        verbose: bool = False,
    ) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
        """Project already-built local tensors into one compatible radial level.

        This is intentionally separate from local block construction so nested
        Galerkin spaces can reuse one finest-grid point-local MHD operator.
        """
        layout = self._mixed_layout(sb, mixed=mixed)

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

        if len(local_split) != 3:
            raise ValueError("local_split must contain (M0, M1, Mminus1).")
        M0, M1, Mminus1 = local_split
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
                f"  ntor={sb.ntor}, n_modes={layout.n_modes}, "
                f"n_high={layout.n_high}, n_low={layout.n_low}"
            )
            print(f"  {_matrix_stats('T0', T0)}")
            print(f"  {_matrix_stats('T1', T1)}")
            print(f"  {_matrix_stats('Tminus1', Tminus1)}")

        kind = self._normalize_bc_kind(bc_kind)
        if not apply_bcs or kind == "none":
            return T0, T1, Tminus1

        bc = self._build_nep_boundary_condition(
            sb,
            bc_kind=kind,
            mixed=mixed,
        )
        expected_rank = 2 * layout.n_modes if kind == "full" else layout.n_modes
        # get_sparse_prolongation already resolves the constraint rank by
        # pivoted QR on C.T, so its column count is the free-dof count.  Read
        # it instead of paying for a second, dense SVD of the same matrix.
        prolongation = bc.get_sparse_prolongation(tol=bc_tol)
        bc_rank = int(layout.n_trial - prolongation.shape[1])
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
        if getattr(nep, "last_backward_errors", None) is not None:
            data_vars["backward_error"] = (
                "eigenpair",
                nep.last_backward_errors,
            )
        if getattr(nep, "last_slepc_error_estimates", None) is not None:
            data_vars["slepc_error_estimate"] = (
                "eigenpair",
                nep.last_slepc_error_estimates,
            )
        if getattr(nep, "last_slepc_backward_errors", None) is not None:
            data_vars["slepc_backward_error"] = (
                "eigenpair",
                nep.last_slepc_backward_errors,
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
