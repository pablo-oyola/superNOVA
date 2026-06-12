
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple
from .base import RadialFE
from scipy.special import chebyt, chebyu
from scipy.interpolate import BarycentricInterpolator
from scipy.linalg import solve

def clenshaw_curtis_weights(nq: int) -> np.ndarray:
    """
    Clenshaw-Curtis quadrature weights for integral over [-1,1].

    Returns weights w such that sum_q w[q] f(x[q])
    approximates int_{-1}^{1} f(x) dx.
    """
    if nq == 1:
        return np.array([2.0])

    n = nq - 1
    theta = np.pi * np.arange(nq) / n
    w = np.zeros(nq, dtype=np.float64)

    ii = np.arange(1, n)

    if n % 2 == 0:
        w[0] = 1.0 / (n**2 - 1.0)
        w[-1] = w[0]

        for i in ii:
            s = 0.0
            for k in range(1, n // 2):
                s += 2.0 * np.cos(2.0 * k * theta[i]) / (4.0 * k**2 - 1.0)
            s += np.cos(n * theta[i]) / (n**2 - 1.0)
            w[i] = 2.0 * (1.0 - s) / n

    else:
        w[0] = 1.0 / n**2
        w[-1] = w[0]

        for i in ii:
            s = 0.0
            for k in range(1, (n + 1) // 2):
                s += 2.0 * np.cos(2.0 * k * theta[i]) / (4.0 * k**2 - 1.0)
            w[i] = 2.0 * (1.0 - s) / n

    return w

class ChebyshevFE(RadialFE):
    def __init__(self, domain: List[Tuple[float, float]], n_elem: int):
        """
        Initialize the class to be used a finite-element basis with
        Chebyshev elements.
        """
        a, b = domain
        if a >= b:
            raise ValueError("Domain should finite: a >= b -- (%.2e, %.2e)" % (a, b))
        
        if n_elem < 2:
            raise ValueError("At least 2 elements are required to initialize the class")
        
        self.domain = (a, b)
        self.n_elem = n_elem

        # We get here the collocation nodes.
        self._get_collocation_nodes()
        self._build_basis()
        self._build_design_matrices()

        # We call the parent and initialize with the collocation nodes.
        super().__init__(self.collocation, nelements=n_elem)

    def eval_basis(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the basis functions at the given points x.
        """
        return np.array([self.basis[i](x) for i in range(self.n_elem)])
    
    def eval_dbasis(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the derivatives of the basis functions at the given points x.
        """
        return np.array([self.dbasis[i](x) for i in range(self.n_elem)])

    def eval_basis_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the derivatives of the basis functions at the given points x.
        """
        return self.eval_dbasis(x)
    
    def get_ndof(self) -> int:
        return self.n_elem

    def get_ndofs(self) -> int:
        return self.get_ndof()
    
    def get_mesh(self) -> np.ndarray:
        return self.grid

    def _get_collocation_nodes(self):
        """
        Using the internal domain for the Chebyshev, it will
        compute the optimal collocation nodes using the 
        Chebyshev-Gauss-Lobatto points. These nodes happen to be the
        same ones as the one for the Clenshaw-Curtis quadrature, so they 
        are optimal for interpolation and integration.
        """
        a, b = self.domain
        N = self.n_elem

        if N < 2:
            raise ValueError("At least 2 Chebyshev points are required.")

        # Chebyshev-Gauss-Lobatto points in [-1, 1]
        jj = np.arange(N)
        x_cheb = np.cos(np.pi * jj / (N - 1))

        # Sort physical coordinate increasingly
        perm = np.argsort(x_cheb)

        self._x_chebyshev = x_cheb[perm]

        # Store weights with the same ordering as the sorted points
        self._cc_weights = clenshaw_curtis_weights(N)[perm]

        # Map from [-1, 1] to [a, b]
        self.collocation = 0.5 * (a + b) + 0.5 * (b - a) * self._x_chebyshev
    
    def _build_basis(self):
        """
        We will store the basis functions ready to be evaluated. 
        """
        midsize = 0.5 * (self.domain[1] - self.domain[0])
        midpoint = 0.5 * (self.domain[0] + self.domain[1])

        def to_chebyshev_coordinate(x):
            x = np.asarray(x, dtype=np.float64)
            return (x - midpoint) / midsize

        self._to_chebyshev_coordinate = to_chebyshev_coordinate

        self.basis = []
        self.dbasis = []

        for i in range(self.n_elem):
            Ti = chebyt(i)

            # Bind Ti into the lambda.
            self.basis.append(
                lambda x, Ti=Ti: Ti(self._to_chebyshev_coordinate(x))
            )

            if i == 0:
                self.dbasis.append(
                    lambda x: np.zeros_like(np.asarray(x, dtype=np.float64))
                )
            else:
                Ui_minus_1 = chebyu(i - 1)

                # d/dx T_i(eta(x)) = i U_{i-1}(eta) / midsize
                self.dbasis.append(
                    lambda x, i=i, Ui_minus_1=Ui_minus_1:
                        (i / midsize) * Ui_minus_1(self._to_chebyshev_coordinate(x))
                )
    
    def _build_design_matrices(self):
        """
        Builds the design matrices for evaluation and pre-computes the mass,
        weak derivative and stiffness matrices. For Chebyshev polynomials, we can
        use the properties of the basis to compute these matrices efficiently.
        """
        # The first step we need is to evaluate the Clenshaw-Curtis quadrature weights for
        # our current collocation nodes.
        self._quadrature_cache = {
            'points': self.collocation,
            'weights': clenshaw_curtis_weights(len(self.collocation)),
            'phi_i': np.empty((self.n_elem, len(self.collocation)), dtype=np.float64),
            'dphi_i': np.empty((self.n_elem, len(self.collocation)), dtype=np.float64),
        }

        self.M = np.empty((self.n_elem, self.n_elem), dtype=np.float64)
        self.WD = np.empty((self.n_elem, self.n_elem), dtype=np.float64)
        self.S = np.empty((self.n_elem, self.n_elem), dtype=np.float64)

        # Computing the basis functions and their derivatives at the 
        # collocation points, and then using them to compute the matrices.
        for i in range(self.n_elem):
            self._quadrature_cache['phi_i'][i, :] = self.basis[i](self.collocation)
            self._quadrature_cache['dphi_i'][i, :] = self.dbasis[i](self.collocation)

        # Building the mass matrix.
        midsize = 0.5 * (self.domain[1] - self.domain[0])
        self.M = midsize * np.einsum("q,iq,jq->ij", 
                                    self._quadrature_cache['weights'], 
                                    self._quadrature_cache['phi_i'], 
                                    self._quadrature_cache['phi_i'])
        # Building the weak derivative matrix.
        self.WD = midsize * np.einsum("q,iq,jq->ij", 
                                        self._quadrature_cache['weights'], 
                                        self._quadrature_cache['dphi_i'], 
                                        self._quadrature_cache['phi_i'])
        
        # Building the stiffness matrix.
        self.S = midsize * np.einsum("q,iq,jq->ij", 
                                    self._quadrature_cache['weights'], 
                                    self._quadrature_cache['dphi_i'], 
                                    self._quadrature_cache['dphi_i'])
        
    def evaluate(self, a_k: np.ndarray, x: np.ndarray=None) -> np.ndarray:
        """
        Evaluates the basis functions at the given points x.
        """
        a_k = np.asarray(a_k, dtype=np.float64)
        if a_k.size != self.n_elem:
            raise ValueError("Coefficient array a_k should have the same size as the number of elements.")
        if x is None:
            x = self.collocation

        x = np.asarray(x, dtype=np.float64)
        return np.dot(a_k, self.eval_basis(x))
    
    def project(self, prof: np.ndarray, x: np.ndarray):
        """
        L2-project a profile onto the Chebyshev basis.

        This returns coefficients a_k such that

            sum_k a_k phi_k(x)

        is the Galerkin/L2 projection of the profile.
        """

        prof = np.asarray(prof, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)

        if prof.size != x.size:
            raise ValueError("Profile and points x should have the same size.")

        # Interpolate the input profile onto the quadrature/collocation points if needed.
        if (
            x.size != self._quadrature_cache['points'].size
            or not np.allclose(x, self._quadrature_cache['points'])
        ):
            y = np.interp(self._quadrature_cache['points'], x, prof)
        else:
            y = prof

        midsize = 0.5 * (self.domain[1] - self.domain[0])

        b = midsize * np.einsum(
            "q,q,jq->j",
                self._quadrature_cache['weights'],
            y,
            self._quadrature_cache['phi_i'],
        )
        
        a_k = solve(self.M, b)

        return a_k  
    
    @property
    def mass_matrix(self):
        """
        Returns the mass matrix for the Chebyshev basis.
        """
        return self.M

    @property
    def stiffness_matrix(self):
        """
        Returns the stiffness matrix for the Chebyshev basis.
        """
        return self.S

    @property
    def weak_derivative_matrix(self):
        """
        Returns the weak derivative matrix for the Chebyshev basis.
        """
        return self.WD

    def get_integrating_points(self):
        """
        Returns the collocation points and weights for integration.
        """
        return self._quadrature_cache['points'], self._quadrature_cache['weights']

    def get_quadrature_points(self):
        """
        Returns the physical quadrature points and weights for integration.
        """
        midsize = 0.5 * (self.domain[1] - self.domain[0])
        return self._quadrature_cache['points'], midsize * self._quadrature_cache['weights']
