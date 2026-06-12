"""
Utilities for the radial finite-element implementation.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

class RadialFE(ABC):
    """
    Abstract implementation of the radial Finite Element, delegating the basic implementation
    to the specifics of each basis.
    """

    def __init__(self, mesh: float, nelements: int):
        """
        Initializes the base class by validating the mesh is 1D and ascending.
        """

        mesh = np.asarray(mesh, dtype=np.float64) # We enforce the type.
        if mesh.ndim != 1 or len(mesh) < 2:
            raise ValueError("mesh input should be a 1D array with at least 2 points")
        if np.any(np.diff(mesh) < 0):
            raise ValueError("mesh should be a monotonically increasing grid")
        if nelements < 2:
            raise ValueError("At least 2 finite elements are required to initialize the class")
        
        self.mesh = mesh

    @abstractmethod
    def mass_matrix(self) -> np.ndarray:
        r"""
        Computes and returns the mass matrix.

        M_ij = \int \phi_i(x)\phi_j(x) dx in the mesh interval.

        This method is left for implementation for the actual radial basis, to
        use some precalculated basis or cached values computed at start.
        """
        pass

    @abstractmethod
    def weak_derivative_matrix(self) -> np.ndarray:
        r"""
        Computes the weak derivative matrix.

        D_ij = \int d\phi_i(x)/dx \phi_j(x) dx in the mesh interval.

        This method is left for implementation for the actual radial basis, to
        use some precalculated basis or cached values computed at start.
        """
        pass

    @abstractmethod
    def stiffness_matrix(self) -> np.ndarray:
        r"""
        Computes the stiffness matrix.

        M_ij = \int d\phi_i(x)/dx d\phi_j(x)/dx dx in the mesh interval.

        This method is left for implementation for the actual radial basis, to
        use some precalculated basis or cached values computed at start.
        """
        pass

    @abstractmethod
    def get_integrating_points(self, ielem: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        For the list of elements, this should return the value of the points
        where to evaluate a function for integration.

        The function will return for each `ielem` in the list the set of points
        to discretize the quadrature of a given function, f(x) on the mesh for the
        \phi_{ielem}, and the weights for the integral, w_k.

        L_i = \int f(x) \phi_i(x) dx \approx \sum_k w_k f(x_k) \phi_i(x_k)
        """
        pass

    @abstractmethod
    def get_quadrature_points(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Return the physical quadrature points and weights for integrals over
        the full radial domain.

        Unlike ``get_integrating_points``, which may be organized around the
        support of individual basis functions, this method returns each
        physical quadrature point once. Solver matrix-vector products should
        evaluate radial profiles and local physics matrices on this grid.
        """
        pass

    @abstractmethod
    def evaluate(self, a_k: np.ndarray, x: Optional[np.ndarray]=None):
        """
        For a given set of input coefficients {a_k}, this will return the
        evaluated function on the input grid `x` or default to the 
        collocation points, `self.mesh`.
        """
        pass

    @abstractmethod
    def project(self, prof: np.ndarray, x: np.ndarray):
        """
        For a given profile, this will return the coefficients of the expansion
        in the basis, by evaluating the profile on the input grid `x` or default to the
        collocation points, `self.mesh`.
        """
        pass

    @abstractmethod
    def get_mesh(self) -> np.ndarray:
        """
        Returns the mesh of the finite element.
        """
        pass

    @abstractmethod
    def eval_basis(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the basis functions at the given points x.
        """
        pass

    @abstractmethod
    def eval_basis_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the derivatives of the basis functions at the given points x.
        """
        pass

    @abstractmethod
    def get_ndofs(self) -> int:
        """
        Returns the number of degrees of freedom of the finite element.
        """
        pass
