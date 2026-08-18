"""
Implements the B-Spline as a possible radial finite element.
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Optional, Tuple
from .base import RadialFE
from scipy.interpolate import BSpline
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
from numpy.polynomial.legendre import leggauss
# from scipy.sparse import linalg
from scipy.sparse.linalg import spsolve


class BSplineFE(RadialFE):
    """
    B-Spline finite element with non-uniform knots.
    """

    def __init__(self, knots: np.ndarray, order: int, domain: Tuple[float, float],
                 quad_order: int=None, build_matrices: bool=True):
        """
        Define a B-Spline basis based on the knots position and order. Domains are
        usually clamped, so the elements at the edge have to handled properly if needed.

        Parameters
        ----------
        knots : np.ndarray
            Knot vector (1D array, strictly increasing). For open B-splines on [a, b],
            use clamped knots: [a]*order + interior_knots + [b]*order.
        order : int
            B-spline order (degree = order - 1). Must be >= 1.
        domain : tuple (a, b),
            Domain bounds. Ensures knots cover [a, b] with proper multiplicity 
            at boundaries.
        """
        knots = np.asarray(knots, dtype=np.float64).ravel()
        if len(knots) < 2:
            raise ValueError("knots must have at least 2 points")
        if np.any(np.diff(knots) < 0):
            raise ValueError("knots must be non-decreasing")
        if order < 1:
            raise ValueError("order must be >= 1")
        if len(knots) < order + 1:
            raise ValueError(f"knots length ({len(knots)}) must be >= order + 1 ({order + 1})")

        self.order = order
        self.degree = order - 1

        a, b = domain
        if knots[0] > a or knots[-1] < b:
            raise ValueError(f"knots must cover domain [{a}, {b}]")
        # Ensure clamped boundaries
        if knots[0] != a or np.any(knots[:order] != a):
            knots = np.concatenate([[a] * order, knots[knots > a]])
        if knots[-1] != b or np.any(knots[-order:] != b):
            knots = np.concatenate([knots[knots < b], [b] * order])

        # Store knots before calling super (which expects strictly increasing mesh)
        self.knots = knots

        # For base class, use unique knots as "mesh" (element boundaries)
        unique_knots = np.unique(knots)
        super().__init__(unique_knots, nelements=len(unique_knots) - 1)

        # We set up now the quadrature order. If not provided,
        # we will use order + 1, which is sufficient for integrating 
        # polynomials of degree 2*degree.
        if quad_order is None:
            self.quadrature_order = self.degree + 1
        else:
            if quad_order < 1:
                raise ValueError("quad_order must be >= 1")
            self.quadrature_order = quad_order

        # We now evaluate the grid.
        self.dofs = len(self.knots) - self.order
        self.grid = np.empty(self.dofs)
        for i in range(self.dofs):
            self.grid[i] = np.mean(self.knots[i+1 : i + self.order + 1])

        # Evaluation-only spaces are used by nested multigrid levels that are
        # projected on one externally supplied finest-grid quadrature.  The
        # default remains the fully assembled single-grid behavior.
        self.build_matrices = bool(build_matrices)

        # Building the basis functions and, when requested, the native
        # quadrature matrices and cache.
        self._build_design_matrices()

    def _build_design_matrices(self):
        """
        Builds the design matrices for evaluation and pre-computes the mass,
        weak-derivative and stiffness matrices.

        This version keeps the cache indexed by basis function i, but stores
        all active-span quadrature data for each basis.
        """

        # Full B-spline space. Axis regularity should be applied later.
        n_basis = self.dofs

        self.basis_functions = np.empty(n_basis, dtype=object)
        self.dbasis_functions = np.empty(n_basis, dtype=object)

        for ii in range(n_basis):
            coeffs = np.zeros(n_basis)
            coeffs[ii] = 1.0

            self.basis_functions[ii] = BSpline(
                self.knots,
                coeffs,
                self.degree,
                extrapolate=False,
            )

            self.dbasis_functions[ii] = self.basis_functions[ii].derivative(1)

        if not self.build_matrices:
            self.M = None
            self.WD = None
            self.S = None
            self.active_span_index = None
            self._quadrature_cache = None
            return

        self.M = lil_matrix((n_basis, n_basis), dtype=np.float64)
        self.WD = lil_matrix((n_basis, n_basis), dtype=np.float64)
        self.S = lil_matrix((n_basis, n_basis), dtype=np.float64)

        unique_spans = [
            (e, self.knots[e], self.knots[e + 1])
            for e in range(len(self.knots) - 1)
            if self.knots[e] < self.knots[e + 1]
        ]

        self.active_span_index = np.empty((n_basis, 2), dtype=int)

        for i in range(n_basis):
            support_start = self.knots[i]
            support_end = self.knots[i + self.order]

            start_span = next(
                e for e, k_start, k_end in unique_spans
                if k_start <= support_start < k_end
            )

            end_span = next(
                e for e, k_start, k_end in unique_spans
                if k_start < support_end <= k_end
            )

            self.active_span_index[i] = (start_span, end_span)

        # Same cache names as before, but now each basis function stores
        # a list of quadrature blocks, one block per active span.
        self._quadrature_cache = {
            'spans': unique_spans,
            'span_ids': np.empty(n_basis, dtype=object),
            'points': np.empty(n_basis, dtype=object),
            'weights': np.empty(n_basis, dtype=object),
            'phi_i': np.empty(n_basis, dtype=object),
            'dphi_i': np.empty(n_basis, dtype=object),
        }

        for i in range(n_basis):
            start_span, end_span = self.active_span_index[i]

            span_ids_i = []
            points_i = []
            weights_i = []
            phi_i_cache = []
            dphi_i_cache = []

            for e in range(start_span, end_span + 1):
                k_start, k_end = self.knots[e], self.knots[e + 1]

                if k_start >= k_end:
                    continue

                quad_points, quad_weights = leggauss(self.quadrature_order)

                mapped_points = (
                    0.5 * (quad_points + 1.0) * (k_end - k_start) + k_start
                )

                mapped_weights = 0.5 * (k_end - k_start) * quad_weights

                phi_i = self.basis_functions[i](mapped_points)
                dphi_i = self.dbasis_functions[i](mapped_points)

                # Save this span's quadrature data for basis i.
                span_ids_i.append(e)
                points_i.append(mapped_points)
                weights_i.append(mapped_weights)
                phi_i_cache.append(phi_i)
                dphi_i_cache.append(dphi_i)

                # Matrix assembly, keeping your basis-function loop style.
                j_min = max(0, i - self.degree)
                j_max = min(n_basis - 1, i + self.degree)

                for j in range(j_min, j_max + 1):
                    phi_j = self.basis_functions[j](mapped_points)
                    dphi_j = self.dbasis_functions[j](mapped_points)

                    mass_contrib = np.sum(mapped_weights * phi_i * phi_j)
                    self.M[i, j] += mass_contrib

                    weak_deriv_contrib = np.sum(mapped_weights * dphi_i * phi_j)
                    self.WD[i, j] += weak_deriv_contrib

                    stiffness_contrib = np.sum(mapped_weights * dphi_i * dphi_j)
                    self.S[i, j] += stiffness_contrib

            self._quadrature_cache['span_ids'][i] = np.asarray(span_ids_i, dtype=int)
            self._quadrature_cache['points'][i] = points_i
            self._quadrature_cache['weights'][i] = weights_i
            self._quadrature_cache['phi_i'][i] = phi_i_cache
            self._quadrature_cache['dphi_i'][i] = dphi_i_cache

        # More efficient for later solves/matvecs than keeping LIL.
        self.M = self.M.tocsr()
        self.WD = self.WD.tocsr()
        self.S = self.S.tocsr()

    # We only return the matrices, since all the work is done at initialization time.
    @property
    def mass_matrix(self) -> np.ndarray:
        return self.M
    
    @property
    def weak_derivative_matrix(self) -> np.ndarray:
        return self.WD
    
    @property
    def stiffness_matrix(self) -> np.ndarray:
        return self.S
    
    def get_mesh(self) -> np.ndarray:
        return self.knots

    # We return the cached quadrature points.
    def get_integrating_points(self, ielem: List[int]=None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Return cached quadrature points and weights for the requested basis indices.

        Note
        ----
        Each requested basis index may contain multiple active span blocks.
        Therefore the returned objects are lists of lists.
        """
        if self._quadrature_cache is None:
            raise RuntimeError(
                "This evaluation-only BSplineFE has no native quadrature cache."
            )
        if ielem is None:
            ielem = list(range(self.dofs))

        points = []
        weights = []

        for i in ielem:
            points.append(self._quadrature_cache['points'][i])
            weights.append(self._quadrature_cache['weights'][i])

        return points, weights

    def get_quadrature_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return one quadrature rule per nonzero knot span.

        This is the solver-facing quadrature grid: each physical quadrature
        point appears once, while basis overlap is represented by evaluating
        all basis functions on these points.
        """
        if self._quadrature_cache is None:
            raise RuntimeError(
                "This evaluation-only BSplineFE has no native quadrature rule."
            )
        points = []
        weights = []
        quad_points, quad_weights = leggauss(self.quadrature_order)

        for _, k_start, k_end in self._quadrature_cache['spans']:
            mapped_points = (
                0.5 * (quad_points + 1.0) * (k_end - k_start) + k_start
            )
            mapped_weights = 0.5 * (k_end - k_start) * quad_weights
            points.append(mapped_points)
            weights.append(mapped_weights)

        return np.concatenate(points), np.concatenate(weights)
    
    def evaluate(self, a_k: np.ndarray, x: np.ndarray=None) -> np.ndarray:
        """
        Evaluates the B-spline expansion at the given points x.

        Parameters
        ----------
        a_k : np.ndarray
            Coefficients of the B-spline expansion (length should match number of basis functions).
        x : np.ndarray
            Points at which to evaluate the expansion.

        Returns
        -------
        np.ndarray
            Evaluated values of the B-spline expansion at points x.
        """
        if len(a_k) != self.dofs:
            raise ValueError(f"Coefficient array length ({len(a_k)}) must match number of basis functions ({self.dofs})")
        
        if x is None:
            x = self.knots  # Default to evaluating at the knots if no points provided
        
        result = np.zeros_like(x)
        for i in range(len(a_k)):
            result += a_k[i] * self.basis_functions[i](x)
        
        return result
    
    def project(self, prof: np.ndarray, x: np.ndarray):
        """
        Projects a given profile onto the B-spline basis by computing the coefficients.

        Parameters
        ----------
        prof : np.ndarray
            Profile values evaluated at points x.
        x : np.ndarray
            Points at which the profile is evaluated.

        Returns
        -------
        np.ndarray
            Coefficients of the B-spline expansion.
        """

        n_basis = self.dofs

        x = np.asarray(x, dtype=np.float64)
        prof = np.asarray(prof)

        if x.ndim != 1:
            raise ValueError("x must be a 1D array")

        if prof.ndim != 1:
            raise ValueError("prof must be a 1D array")

        if len(x) != len(prof):
            raise ValueError("x and prof must have the same length")

        if np.any(np.diff(x) <= 0.0):
            raise ValueError("x must be strictly increasing for np.interp")

        int_prof_phi = np.zeros(n_basis, dtype=np.dtype(prof.dtype))

        for i in range(n_basis):
            points_i = self._quadrature_cache['points'][i]
            weights_i = self._quadrature_cache['weights'][i]
            phi_i_cache = self._quadrature_cache['phi_i'][i]

            for points, weights, phi_i in zip(points_i, weights_i, phi_i_cache):
                prof_at_points = np.interp(points, x, prof)
                int_prof_phi[i] += np.sum(weights * prof_at_points * phi_i)

        coeffs = spsolve(self.mass_matrix, int_prof_phi)

        return coeffs
    
    def evaluate(self, a_k: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Evaluates the B-spline expansion at the given points x.
        """

        n_basis = self.dofs

        a_k = np.asarray(a_k, dtype=np.float64)

        if len(a_k) != n_basis:
            raise ValueError(
                f"Coefficient array length ({len(a_k)}) must match "
                f"number of basis functions ({n_basis})"
            )

        if x is None:
            x = self.grid

        x = np.asarray(x, dtype=np.float64)

        result = np.zeros(x.shape, dtype=np.float64)

        for i in range(n_basis):
            result += a_k[i] * self.basis_functions[i](x)

        return result
    
    def eval_basis(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the B-spline basis functions at the given points x.

        Returns an array of shape (n_basis, len(x)) where each row corresponds
        to a basis function evaluated at all points in x.
        """
        n_basis = self.dofs
        return np.array([self.basis_functions[i](x) for i in range(n_basis)], dtype=np.float64)
    
    def eval_basis_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the derivatives of the basis functions at the given points x.
        """
        n_basis = self.dofs
        return np.array([self.dbasis_functions[i](x) for i in range(n_basis)], dtype=np.float64)

    def get_ndofs(self) -> int:
        return self.dofs


@dataclass(frozen=True)
class CompatibleBSplinePair:
    """Compatible high/low B-spline spaces for a first derivative.

    The lower space has degree one below ``high`` and uses the trimmed knot
    vector ``high.knots[1:-1]``.  ``derivative`` maps coefficients in the
    high space to coefficients of their exact derivative in the low space.
    """

    high: BSplineFE
    low: BSplineFE
    derivative: csr_matrix

    @classmethod
    def from_high(
        cls,
        high: BSplineFE,
        *,
        build_low_matrices: bool | None = None,
        verification_points: np.ndarray | None = None,
    ) -> "CompatibleBSplinePair":
        """Construct the compatible lower space and coefficient derivative."""
        if not isinstance(high, BSplineFE):
            raise TypeError("high must be a BSplineFE instance")
        if high.degree < 2:
            raise ValueError("The high-order space must have degree >= 2.")

        n_high = high.get_ndofs()
        n_low_expected = n_high - 1
        degree = high.degree
        knots = high.knots
        indices = np.arange(n_low_expected)
        denominators = knots[indices + degree + 1] - knots[indices + 1]
        if np.any(~np.isfinite(denominators)) or np.any(denominators <= 0.0):
            raise ValueError(
                "The high-space knots do not define positive derivative "
                "denominators."
            )

        if build_low_matrices is None:
            build_low_matrices = high.build_matrices
        low = BSplineFE(
            knots=knots[1:-1],
            order=high.order - 1,
            domain=(knots[0], knots[-1]),
            quad_order=high.quadrature_order,
            build_matrices=build_low_matrices,
        )

        n_low = low.get_ndofs()
        if n_low != n_low_expected:
            raise RuntimeError(
                "Expected n_low=n_high-1, got "
                f"n_high={n_high}, n_low={n_low}."
            )

        alpha = degree / denominators
        rows = np.repeat(np.arange(n_low), 2)
        cols = np.column_stack(
            (np.arange(n_low), np.arange(1, n_low + 1))
        ).ravel()
        data = np.column_stack((-alpha, alpha)).ravel()
        derivative = coo_matrix(
            (data, (rows, cols)),
            shape=(n_low, n_high),
        ).tocsr()

        if verification_points is None:
            points, _ = high.get_quadrature_points()
        else:
            points = np.asarray(verification_points, dtype=np.float64)
            if points.ndim != 1 or points.size == 0:
                raise ValueError(
                    "verification_points must be a nonempty one-dimensional array."
                )
            if np.any(~np.isfinite(points)):
                raise ValueError("verification_points must be finite.")
        high_derivative = high.eval_basis_derivative(points)
        compatible_derivative = derivative.T @ low.eval_basis(points)
        if not np.allclose(
            high_derivative,
            compatible_derivative,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            max_error = np.max(
                np.abs(high_derivative - compatible_derivative)
            )
            raise RuntimeError(
                "Compatible B-spline derivative identity failed: "
                f"max error={max_error:.3e}."
            )

        return cls(high=high, low=low, derivative=derivative)
