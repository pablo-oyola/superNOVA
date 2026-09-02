r"""Boundary conditions for the four-field mixed compatible-spline state.

The trial vector is field-major,

    [p1_H, xi^psi_H, xi^s_L, div_xi_L],

with radial-major, harmonic-fast ordering inside each field.  Every equation is
tested in the lower-order space, so the assembled operator is rectangular
before boundary conditions.  :class:`MixedBoundaryCondition` therefore reduces
only the trial side, as ``T @ P``, and the constraints below act on the
high-space ``xi^psi`` coefficients.

Current conditions:

- ``xi^psi_m = 0`` at the plasma boundary. @TODO: the wall condition.
- ``p_1``, ``xi^psi``, ``xi^s`` and ``div.xi`` all vanish at the axis.

This module merges the former ``_boundary3`` base and ``_boundary4`` mixed
subclasses; the monolithic-space subclasses went with the retired monolithic-space boundary
builder.
"""

from __future__ import annotations

import operator

import numpy as np
from scipy.linalg import null_space, solve, qr
from scipy.sparse import csr_matrix, issparse

from supernova.radial.base import RadialFE

def boundary_basis_values(fe: RadialFE, side: str="right"):
    """
    Evalautes the basis functions at the boundary point specified by side ("right" or "left").
    """
    if side == "right":
        psi_b = fe.get_mesh()[-1]
    elif side == "left":
        psi_b = fe.get_mesh()[0]
    else:
        raise ValueError("side must be 'left' or 'right'")

    phi_b = fe.eval_basis(psi_b)

    return psi_b, phi_b

def endpoint_is_single_dof(fe: RadialFE, side: str="right", 
                           tol: float=1e-12):
    """
    Checks whether the endpoint specified by side ("right" or "left") 
    is a single DoF in the finite element basis.
    """
    psi_b, phi_b = boundary_basis_values(fe, side=side)

    if side == "right":
        expected = np.zeros_like(phi_b)
        expected[-1] = 1.0
    else:
        expected = np.zeros_like(phi_b)
        expected[0] = 1.0

    return np.allclose(phi_b, expected, atol=tol, rtol=0.0)

class BoundaryCondition:
    """
    Generic implementation of the boundary conditions for the superNOVA solver.
    """
    C: np.ndarray=None # The matrix of the linear system to be solved: Cx = 0
    Z: np.ndarray=None # Basis of the transformation to apply the boundary conditions.
    Z_sparse: csr_matrix=None # Sparse, non-orthonormal prolongation basis.
    sparse_free: np.ndarray=None
    sparse_dependent: np.ndarray=None
    sparse_constraint_rows: np.ndarray=None

    def __init__(self, C: np.ndarray):
        self.C = C

    def make_subspace(self):
        """
        Constructs the subspace of the transformation E and R matrices,
        such that we can apply the boundary conditions.
        """
        # We compute the null-space.
        self.Z = null_space(self.C, check_finite=False)
    
    def expand(self, z: np.ndarray) -> np.ndarray:
        """
        Transforms the restricted space into the full space taking into 
        account the boundary conditions.
        """
        if self.Z is None:
            self.make_subspace()

        # We check the shape of z and apply the expansion matrix to get the full space vector.
        assert z.shape[0] == self.Z.shape[1], f"Expected reduced vector length {self.Z.shape[1]}, got {z.shape[0]}"

        return self.Z @ z

    def restrict(self, x: np.ndarray) -> np.ndarray:
        """
        restricts the full space into the restricted space taking into account
        the boundary conditions.
        """
        if self.Z is None:
            self.make_subspace()
        assert x.shape[0] == self.Z.shape[0], f"Expected full vector length {self.Z.shape[0]}, got {x.shape[0]}"
        # We apply the restriction matrix to the full space vector.
        return self.Z.conj().T @ x

    @staticmethod
    def _qr_rank(R: np.ndarray, matrix_shape: tuple[int, int], tol: float=None) -> int:
        """
        Estimate numerical rank from the diagonal of a QR ``R`` factor.
        """
        diag = np.abs(np.diag(R))
        if diag.size == 0:
            return 0
        if tol is None:
            tol = np.finfo(diag.dtype).eps * max(matrix_shape) * np.max(diag)
        return int(np.sum(diag > tol))

    def make_sparse_subspace(self, tol: float=None) -> csr_matrix:
        r"""
        Constructs a sparse prolongation matrix for the constrained subspace.

        The dense implementation stores an orthonormal null-space basis ``Z``
        from an SVD. This method keeps that path untouched and instead builds a
        sparse, generally non-orthonormal basis ``P`` by eliminating dependent
        variables from

            C x = 0.

        With the split ``x = (x_free, x_dep)`` chosen by pivoted QR,

            C_dep x_dep + C_free x_free = 0,
            x_dep = -C_dep^{-1} C_free x_free.

        The returned sparse matrix maps reduced coordinates to full
        coordinates, ``x = P z``. It can be used to reduce a sparse operator as
        ``P.conj().T @ T @ P`` without constructing the dense SVD nullspace.
        """
        if self.C is None:
            raise ValueError("Boundary condition matrix C must be defined.")

        C = np.asarray(self.C, dtype=np.complex128)
        if C.ndim != 2:
            raise ValueError("Boundary condition matrix C must be two-dimensional.")

        n_constraints, n_full = C.shape
        identity_rows = np.arange(n_full, dtype=np.int64)
        if n_constraints == 0:
            self.sparse_free = np.arange(n_full, dtype=np.int64)
            self.sparse_dependent = np.empty(0, dtype=np.int64)
            self.sparse_constraint_rows = np.empty(0, dtype=np.int64)
            self.Z_sparse = csr_matrix(
                (
                    np.ones(n_full, dtype=np.complex128),
                    (identity_rows, identity_rows),
                ),
                shape=(n_full, n_full),
            )
            return self.Z_sparse

        _, row_R, row_piv = qr(C.T, mode="economic", pivoting=True, check_finite=False)
        rank = self._qr_rank(row_R, C.shape, tol=tol)

        if rank == 0:
            self.sparse_free = np.arange(n_full, dtype=np.int64)
            self.sparse_dependent = np.empty(0, dtype=np.int64)
            self.sparse_constraint_rows = np.empty(0, dtype=np.int64)
            self.Z_sparse = csr_matrix(
                (
                    np.ones(n_full, dtype=np.complex128),
                    (identity_rows, identity_rows),
                ),
                shape=(n_full, n_full),
            )
            return self.Z_sparse

        independent_rows = np.asarray(row_piv[:rank], dtype=np.int64)
        C_independent = C[independent_rows, :]

        _, col_R, col_piv = qr(C_independent, mode="economic", pivoting=True, check_finite=False)
        col_rank = self._qr_rank(col_R, C_independent.shape, tol=tol)
        if col_rank != rank:
            raise ValueError("Could not identify a full-rank dependent column set for boundary constraints.")

        dependent = np.asarray(col_piv[:rank], dtype=np.int64)
        free = np.setdiff1d(np.arange(n_full, dtype=np.int64), dependent, assume_unique=False)

        if free.size == 0:
            self.sparse_free = free
            self.sparse_dependent = dependent
            self.sparse_constraint_rows = independent_rows
            self.Z_sparse = csr_matrix((n_full, 0), dtype=np.complex128)
            return self.Z_sparse

        C_dep = C_independent[:, dependent]
        C_free = C_independent[:, free]
        dep_from_free = -solve(C_dep, C_free, check_finite=False)

        rows = [free]
        cols = [np.arange(free.size, dtype=np.int64)]
        data = [np.ones(free.size, dtype=np.complex128)]

        dep_rows = np.repeat(dependent, free.size)
        dep_cols = np.tile(np.arange(free.size, dtype=np.int64), dependent.size)
        dep_data = dep_from_free.reshape(-1)
        if tol is None:
            keep = dep_data != 0.0
        else:
            keep = np.abs(dep_data) > tol

        rows.append(dep_rows[keep])
        cols.append(dep_cols[keep])
        data.append(dep_data[keep])

        self.sparse_free = free
        self.sparse_dependent = dependent
        self.sparse_constraint_rows = independent_rows
        self.Z_sparse = csr_matrix(
            (
                np.concatenate(data),
                (np.concatenate(rows), np.concatenate(cols)),
            ),
            shape=(n_full, free.size),
            dtype=np.complex128,
        )
        return self.Z_sparse

    def get_sparse_prolongation(self, tol: float=None) -> csr_matrix:
        """
        Returns the sparse prolongation matrix ``P`` for ``x = P z``.
        """
        if self.Z_sparse is None:
            self.make_sparse_subspace(tol=tol)
        return self.Z_sparse

    def get_sparse_free_dofs(self, tol: float=None) -> int:
        """
        Returns the reduced dimension implied by the sparse prolongation path.
        """
        return self.get_sparse_prolongation(tol=tol).shape[1]

    def expand_sparse(self, z: np.ndarray, tol: float=None) -> np.ndarray:
        """
        Expands reduced coordinates with the sparse prolongation matrix.
        """
        P = self.get_sparse_prolongation(tol=tol)
        assert z.shape[0] == P.shape[1], f"Expected reduced vector length {P.shape[1]}, got {z.shape[0]}"
        return P @ z

    def project_sparse(self, x: np.ndarray, tol: float=None) -> np.ndarray:
        """
        Projects a full vector onto the sparse constrained basis.
        """
        P = self.get_sparse_prolongation(tol=tol)
        assert x.shape[0] == P.shape[0], f"Expected full vector length {P.shape[0]}, got {x.shape[0]}"
        return P.conj().T @ x

    def reduce_sparse_matrix(self, matrix, tol: float=None) -> csr_matrix:
        """
        Applies the sparse boundary-condition reduction to a matrix.

        If ``matrix`` is the full-space operator ``T`` and ``P`` is the sparse
        prolongation from :meth:`get_sparse_prolongation`, this returns

            P.conj().T @ T @ P.
        """
        P = self.get_sparse_prolongation(tol=tol)
        mat = matrix.tocsr() if issparse(matrix) else csr_matrix(matrix)
        if mat.shape != (P.shape[0], P.shape[0]):
            raise ValueError(f"Expected matrix shape {(P.shape[0], P.shape[0])}, got {mat.shape}.")
        return (P.conj().T @ mat @ P).tocsr()
    
    def __mul__(self, other: "BoundaryCondition") -> "BoundaryCondition":
        """
        Combines two boundary conditions by multiplying their transformation matrices.
        """
        if self.C is None or other.C is None:
            raise ValueError("Both boundary conditions must have their C matrices defined.")
        
        if not isinstance(other, BoundaryCondition):
            raise ValueError("Can only combine with another BoundaryCondition.")
        
        # We combine the transformation matrices.
        C_combined = np.vstack([self.C, other.C])
        combined = BoundaryCondition(
            C=C_combined
        )

        return combined
    
    def __rmul__(self, other: "BoundaryCondition") -> "BoundaryCondition":
        """
        Combines two boundary conditions by multiplying their transformation matrices.
        """
        if self.C is None or other.C is None:
            raise ValueError("Both boundary conditions must have their C matrices defined.")
        
        if not isinstance(other, BoundaryCondition):
            raise ValueError("Can only combine with another BoundaryCondition.")
        
        # We combine the transformation matrices.
        C_combined = np.vstack([other.C, self.C])
        combined = BoundaryCondition(
            C=C_combined
        )

        return combined

    def get_free_dofs(self) -> np.ndarray:
        """
        Returns the number of actually free degress of freedom.
        """
        if self.Z is None:
            self.make_subspace()
        return self.Z.shape[1]


# ---------------------------------------------------------------------------
# Mixed compatible-space boundary conditions.
#
# Merged from the former _boundary4.py, which was a 209-line subclass module
# with nothing to separate it from its base.  The four monolithic-space
# subclasses that used to live above went with the retired monolithic-space boundary
# builder; these are the only ones the solver reaches.
# ---------------------------------------------------------------------------

class MixedBoundaryCondition(BoundaryCondition):
    """Sparse-nullspace BC that reduces only the trial side.

    The mixed operator is rectangular: its equation rows already live in
    the lower test space, so only its columns are projected.
    """

    def reduce_sparse_trial_matrix(self, matrix, tol: float | None = None) -> csr_matrix:
        r"""Apply a boundary-condition prolongation on the trial side.

        For a rectangular mixed operator ``T`` and a sparse prolongation ``P``
        satisfying ``C @ P = 0``, return ``T @ P``.  Equation rows are not
        projected because they already live in the lower-order test space.
        """
        P = self.get_sparse_prolongation(tol=tol)
        mat = matrix.tocsr() if issparse(matrix) else csr_matrix(matrix)
        if mat.shape[1] != P.shape[0]:
            raise ValueError(
                f"Expected matrix with {P.shape[0]} trial columns, got shape {mat.shape}."
            )
        return (mat @ P).tocsr()

    def __mul__(self, other: "BoundaryCondition") -> "MixedBoundaryCondition":
        """Stack two compatible constraint matrices in left-to-right order."""
        if not isinstance(other, BoundaryCondition):
            raise ValueError("Can only combine with another BoundaryCondition.")
        if self.C is None or other.C is None:
            raise ValueError("Both boundary conditions must have their C matrices defined.")
        if self.C.shape[1] != other.C.shape[1]:
            raise ValueError(
                "Boundary conditions must act on the same trial space, got "
                f"{self.C.shape[1]} and {other.C.shape[1]} columns."
            )
        return MixedBoundaryCondition(C=np.vstack([self.C, other.C]))

    def __rmul__(self, other: "BoundaryCondition") -> "MixedBoundaryCondition":
        """Stack two compatible constraint matrices in right-to-left order."""
        if not isinstance(other, BoundaryCondition):
            raise ValueError("Can only combine with another BoundaryCondition.")
        if self.C is None or other.C is None:
            raise ValueError("Both boundary conditions must have their C matrices defined.")
        if self.C.shape[1] != other.C.shape[1]:
            raise ValueError(
                "Boundary conditions must act on the same trial space, got "
                f"{other.C.shape[1]} and {self.C.shape[1]} columns."
            )
        return MixedBoundaryCondition(C=np.vstack([other.C, self.C]))


def _validate_mixed_spaces(
    high_radialfe: RadialFE,
    low_radialfe: RadialFE,
) -> tuple[int, int]:
    """Return compatible high/low dimensions for the mixed field-major layout."""
    n_high = high_radialfe.get_ndofs()
    n_low = low_radialfe.get_ndofs()
    if n_high < 2:
        raise ValueError(f"The high radial space must have at least 2 DoFs, got {n_high}.")
    if n_low != n_high - 1:
        raise ValueError(
            "Compatible mixed spaces require n_low=n_high-1, got "
            f"n_high={n_high}, n_low={n_low}."
        )
    return n_high, n_low


class _XiPsiEndpointBC(MixedBoundaryCondition):
    """Common high-space ``xi^psi`` endpoint trace constraint."""

    def __init__(
        self,
        high_radialfe: RadialFE,
        low_radialfe: RadialFE,
        n_modes: int,
        side: str,
    ):
        try:
            n_modes = operator.index(n_modes)
        except TypeError as exc:
            raise ValueError("n_modes must be an integer.") from exc
        if n_modes < 1:
            raise ValueError("At least one mode is required.")

        n_high, n_low = _validate_mixed_spaces(high_radialfe, low_radialfe)
        _, endpoint_trace = boundary_basis_values(high_radialfe, side=side)
        endpoint_trace = np.asarray(endpoint_trace, dtype=np.complex128)
        if endpoint_trace.shape != (n_high,):
            raise ValueError(
                f"Expected high-space endpoint trace shape {(n_high,)}, "
                f"got {endpoint_trace.shape}."
            )
        if not np.all(np.isfinite(endpoint_trace)):
            raise ValueError("High-space endpoint trace contains non-finite values.")
        if not np.any(endpoint_trace != 0.0):
            raise ValueError(f"High-space {side} endpoint trace is identically zero.")

        self.high_radialfe = high_radialfe
        self.low_radialfe = low_radialfe
        self.n_high = n_high
        self.n_low = n_low
        self.n_modes = n_modes
        self.n_rows = 4 * n_low * n_modes
        self.n_trial = 2 * n_modes * (n_high + n_low)

        self.p1_offset = 0
        self.xipsi_offset = n_high * n_modes
        self.xis_offset = 2 * n_high * n_modes
        self.divxi_offset = (2 * n_high + n_low) * n_modes
        self.field_offsets = (
            self.p1_offset,
            self.xipsi_offset,
            self.xis_offset,
            self.divxi_offset,
        )

        self.side = side
        self.endpoint_trace = endpoint_trace
        self.endpoint_support = np.flatnonzero(endpoint_trace != 0.0)

        C = np.zeros((n_modes, self.n_trial), dtype=np.complex128)
        harmonic_rows = np.arange(n_modes)
        for radial_index, value in enumerate(endpoint_trace):
            if value == 0.0:
                continue
            columns = self.xipsi_offset + radial_index * n_modes + harmonic_rows
            C[harmonic_rows, columns] = value

        # For clamped B-splines this contains exactly one column per harmonic.
        self.endpoint_columns = np.concatenate(
            [
                self.xipsi_offset
                + radial_index * n_modes
                + np.arange(n_modes, dtype=np.int64)
                for radial_index in self.endpoint_support
            ]
        )
        super().__init__(C=C)


class regularityAxisBC(_XiPsiEndpointBC):
    r"""Enforce ``xi^psi_m(0)=0`` in the high trial space."""

    def __init__(
        self,
        high_radialfe: RadialFE,
        low_radialfe: RadialFE,
        m_numbers: np.ndarray,
    ):
        m_numbers = np.asarray(m_numbers)
        if m_numbers.ndim != 1:
            raise ValueError("m_numbers must be a 1D array of poloidal mode numbers.")
        if m_numbers.size < 1:
            raise ValueError("At least one mode is required.")

        self.m_numbers = m_numbers
        super().__init__(
            high_radialfe=high_radialfe,
            low_radialfe=low_radialfe,
            n_modes=m_numbers.size,
            side="left",
        )


class idealSeparatrixBC(_XiPsiEndpointBC):
    r"""Enforce ``xi^psi_m(1)=0`` in the high trial space."""

    def __init__(
        self,
        high_radialfe: RadialFE,
        low_radialfe: RadialFE,
        n_modes: int,
    ):
        super().__init__(
            high_radialfe=high_radialfe,
            low_radialfe=low_radialfe,
            n_modes=n_modes,
            side="right",
        )


__all__ = [
    "BoundaryCondition",
    "idealSeparatrixBC",
    "regularityAxisBC",
]
