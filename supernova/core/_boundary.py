r"""
Implements the boundary conditions for the superNOVA solver. 

Current boundary conditions implemented are:
- \xi_m = 0 at the plasma boundary --> @TODO: need to implement the wall boundary condition.
- \xi_m = 0 |m|!=1 at the plasma center.
- d\xi_m_dx = 0 |m|=1 at the plasma center.

The regularity conditions at the plasma center are imposed by
mixing the basis functions together at the superSolver level, and
reducing the DoF by 1. In this object we will focus on the wall boundary
condition.
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.linalg import null_space, solve, qr
from scipy.sparse import csr_matrix, issparse
from ..radial.base import RadialFE

from abc import ABC, abstractmethod

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
    
class regularityAxisBC(BoundaryCondition):
    r"""
    Enforces axis regularity for xi_m at x=0:

        xi_m(0) = 0          for |m| != 1
        d xi_m / dx (0) = 0  for |m| == 1

    Assumes state ordering:

        [p1_m0, ..., p1_mN, xi_m0, ..., xi_mN]

    and flattened ordering:

        x_full = X.ravel(order="C")

    where X.shape == (nq, n_state).
    """

    def __init__(
        self,
        radialfe: RadialFE,
        m_numbers: np.ndarray
    ):
        """
        Initializes the regularity boundary condition at the plasma axis.
        Parameters
        ----------
        radialfe : RadialFE
            The radial finite element basis used for the discretization of the problem.
        m_numbers : np.ndarray
            The array of poloidal mode numbers corresponding to the state variables.
        """
        m_numbers = np.asarray(m_numbers)

        if m_numbers.ndim != 1:
            raise ValueError("m_numbers must be a 1D array of poloidal mode numbers.")

        self.m_numbers = m_numbers
        self.n_modes = len(m_numbers)

        if self.n_modes < 1:
            raise ValueError("At least one mode is required.")

        self.nq = radialfe.get_ndofs()
        self.n_state = 2 * self.n_modes
        self.ndof_full = self.nq * self.n_state

        # State columns:
        #
        #   [p1_m0, ..., p1_mN, xi_m0, ..., xi_mN]
        #
        self.p_cols = np.arange(0, self.n_modes)
        self.xi_cols = np.arange(self.n_modes, 2 * self.n_modes)

        # Axis value trace:
        #
        #   e0[k] = phi_k(0)
        #
        psi_axis, e0 = boundary_basis_values(radialfe, side="left")
        e0 = np.asarray(e0, dtype=np.complex128)

        if e0.shape != (self.nq,):
            raise ValueError(f"Expected e0 shape {(self.nq,)}, got {e0.shape}.")

        # Axis derivative trace:
        #
        #   d0[k] = d phi_k / dx at x = 0
        #
        # This should be the pointwise derivative trace in coefficient space.
        d0 = radialfe.eval_basis_derivative(psi_axis)
        d0 = np.asarray(d0, dtype=np.complex128)

        if d0.shape != (self.nq,):
            raise ValueError(f"Expected d0 shape {(self.nq,)}, got {d0.shape}.")

        # We impose one pressure regularity row and one xi regularity row
        # per retained m harmonic.
        #
        # Rows:
        #
        #   0 ... n_modes - 1:
        #       pressure regularity
        #
        #   n_modes ... 2*n_modes - 1:
        #       xi regularity
        #
        C = np.zeros((2 * self.n_modes, self.ndof_full), dtype=np.complex128)

        for j, m in enumerate(self.m_numbers):
            p_col = self.p_cols[j]
            xi_col = self.xi_cols[j]

            # ------------------------------------------------------------
            # Pressure scalar regularity
            # ------------------------------------------------------------
            #
            #   m = 0:
            #       d p1_0 / dx (0) = 0
            #
            #   |m| != 0:
            #       p1_m(0) = 0
            #
            p_row = j

            if m == 0:
                p_trace = d0
            else:
                p_trace = e0

            for k in range(self.nq):
                col = k * self.n_state + p_col
                C[p_row, col] = p_trace[k]

            # ------------------------------------------------------------
            # Displacement vector-component regularity
            # ------------------------------------------------------------
            #
            #   |m| = 1:
            #       d xi_m / dx (0) = 0
            #
            #   |m| != 1:
            #       xi_m(0) = 0
            #
            xi_row = self.n_modes + j

            if abs(m) == 1:
                xi_trace = d0
            else:
                xi_trace = e0

            for k in range(self.nq):
                col = k * self.n_state + xi_col
                C[xi_row, col] = xi_trace[k]

        self.C = C

class idealSeparatrixBC(BoundaryCondition):
    """
    Implements the ideal wall boundary condition at the plasma boundary, 
    and the separatrix boundary condition at the plasma center.
    """
    def __init__(self, radialfe: RadialFE, n_state: int):
        # We will enforce for the moment that some of the
        # collocation/knots points are actually at the plasma boundary.
        ok = endpoint_is_single_dof(radialfe, side="right")

        if not ok:
            raise NotImplementedError("The finite element object does not contain" \
            " the Psi=1.0 as one of the nodes, which is required for the ideal" \
            " wall boundary condition current version.")
        
        if n_state < 2:
            raise ValueError("At least 2 state variables are required to initialize the class")
        
        # Saving the shapes.
        self.nq = radialfe.get_ndofs()
        self.nstate = n_state

        self.xi_cols = np.arange(self.nstate, 2 * self.nstate)

        free_mask = np.ones((self.nq, 2*self.nstate), dtype=bool)
        free_mask[-1, self.xi_cols] = False

        self.free_mask_2d = free_mask
        self.free = np.flatnonzero(free_mask.ravel(order="C"))
        self.constrained = np.flatnonzero((np.logical_not(free_mask)).ravel(order="C"))

        self.ndof_free = self.free.size
        self.ndof_full = self.nq * 2 * self.nstate

        # We build the C matrix for the ideal wall boundary condition.
        n_constraints = self.constrained.size

        self.C = np.zeros((n_constraints, self.ndof_full), dtype=np.float64)

        self.C[
            np.arange(n_constraints),
            self.constrained
        ] = 1.0

class couplingPressureDisplacementBC(BoundaryCondition):
    r"""
    This implements the coupling between the pressure and the plasma
    displacement at the plasma boundary, which is required to fulfill the ideal
    MHD equations.

    Using the Schur decomposition, we can write the system as:

    M21 @ M11^{-1} (\partial_psi) p_1 = \partial_\psi\xi^\psi + 
                       (M21 @ M11^{-1} M12 - M22) \xi^\psi

    where the previous expression is evaluated at the plasma boundary.
    Mij are the ideal MHD matrices.

    In the case of ideal wall boundary conditions, the second line
    vanishes identically.    
    """
    def __init__(self, radialfe: RadialFE, n_modes: int, M: np.ndarray):
        """
        Initializes the coupling between the pressure and the plasma
        """
        # Checking the M array shape.
        if M.shape[0] != 2*n_modes or M.shape[1] != 2*n_modes:
            raise ValueError(f"M matrix must have shape ({2*n_modes}, {2*n_modes}), got {M.shape}")
        
        self.M = M
        self.n_modes = n_modes
        self.radialfe = radialfe
        self.n_state = 2 * n_modes
        self.ndof_full = radialfe.get_ndofs() * self.n_state

        # Builds the C matrix for the coupling between the pressure and the
        # plasma.
        p1_cols = np.arange(0, self.n_modes)
        xi_cols = np.arange(self.n_modes, 2 * self.n_modes)

        # We verify the edge point is a single DoF.
        ok= boundary_basis_values(radialfe, side="right")
        if not ok:
            raise NotImplementedError("The finite element object does not contain" \
            " the Psi=1.0 as one of the nodes, which is " \
            "required for the coupling pressure-displacement boundary" \
            " condition current version.")
        

        # We evaluate the basis functions at the plasma boundary.
        self.nq = radialfe.get_ndofs()
        phi_b = np.asarray(self.radialfe.eval_basis([1.0])).reshape(-1)
        dphi_b = np.asarray(self.radialfe.eval_basis_derivative([1.0])).reshape(-1)

        # Checking that the basis functions are the same shape.
        if phi_b.shape != (self.nq,):
            raise ValueError(f"Expected phi_b to have shape ({self.nq},), got {phi_b.shape}")
        if dphi_b.shape != (self.nq,):
            raise ValueError(f"Expected dphi_b to have shape ({self.nq},), got {dphi_b.shape}")

        # Separating the matrix into the pressure and displacement parts.
        M11 = self.M[:self.n_modes, :self.n_modes]
        M12 = self.M[:self.n_modes, self.n_modes:2*self.n_modes]
        M21 = self.M[self.n_modes:2*self.n_modes, :self.n_modes]
        M22 = self.M[self.n_modes:2*self.n_modes, self.n_modes:2*self.n_modes]

        # Building the operators for the coupling
        A = M21 @ solve(M11, M12, check_finite=False) - M22
        B = M21 @ solve(M11, np.eye(self.n_modes, dtype=np.complex128), check_finite=False)

        Kd = np.zeros((self.n_modes, self.n_state), dtype=np.complex128)
        Kv = np.zeros((self.n_modes, self.n_state), dtype=np.complex128)

        # Derivative terms:
        #
        #     B p1'(1) - xi'(1)
        #
        Kd[:, p1_cols] = B
        Kd[:, xi_cols] = -np.eye(self.n_modes)
        Kv[:, xi_cols] = -A

        C = np.zeros((self.n_modes, self.ndof_full), dtype=np.complex128)

        for k in range(self.nq):
            cols_k = slice(k * self.n_state, (k + 1) * self.n_state)

            C[:, cols_k] = dphi_b[k] * Kd + phi_b[k] * Kv

        self.C = C

class idealWallBC(BoundaryCondition):
    """
    Implements the ideal wall boundary condition. The user should
    provide here the plasma boundary and wall position to
    compute the relevant Green functions for the ideal wall boundary condition.
    """
    def __init__(self, radialfe: RadialFE, sep_R: np.ndarray, sep_z: np.ndarray,
                 wall_R: np.ndarray, wall_z: np.ndarray):
        """
        Initializes the ideal wall boundary condition with the 
        plasma boundary and wall position.

        Will compute the relevant Green functions to be casted inside the 
        """
        super().__init__(radialfe)


    def find_green_function(self):
        """
        Computes the Green function for the ideal wall boundary condition.
        """
        raise NotImplementedError("This method is not implemented yet.")
    
    @classmethod
    def load_from_file(cls, fn: str):
        """
        Loads the Green function from a file.
        """
        raise NotImplementedError("This method is not implemented yet.")
