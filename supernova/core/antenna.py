r"""
We will implement here the main class for the antenna perturbation.

In ideal MHD eigenvalue problems, the problem to solve to obtain the 
eigenvalues is a nonlinear egenvalue problem of the form:

M(\omega) x = 0

We can instead pose an equivalent "antenna" problem, where we add
a source term to the right hand side, and solve for a given frequency:
M(\omega) x = s(\omega)

such that the response to the system would diverge at the eigenvalues,
accelerating the nonlinear eigenvalue solver. 

We offer different approaches to impose such antenna pertubation.
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from ..radial.base import RadialFE


class antenna:
    def __init__(self, psi: np.ndarray, ampl: np.ndarray, radialfe: RadialFE,
                 m: int=None, n: int=None, which: str='xi'):
        """
        Generates the antenna perturbation for a given poloidal and toroidal mode number, 
        and a given amplitude.

        If m and n are not provided, the antenna will be generated for all mode numbers.
        """
        self.psi = np.asarray(psi, dtype=np.float64)
        self.ampl = np.asarray(ampl, dtype=np.complex128)
        self.radialfe = radialfe
        self.which = which

        if self.which.lower() not in ['xi', 'p1']:
            raise ValueError("which must be either 'xi' or 'p1'")
        if self.m is not None and self.n is not None:
            self.m = np.asarray(self.m, dtype=np.int32)
            self.n = np.asarray(self.n, dtype=np.int32)
        else:
            self.m = None
            self.n = None

        if self.psi.size != self.ampl.size:
            raise ValueError("psi and ampl must have the same size")
        
        # Let's get the radial coeffiecients to the antenna perturbation
        self.coeffs = self.radialfe.project(self.ampl, x=self.psi)
    
    def evaluate(self, nstate: int, m: np.ndarray=None, n: np.ndarray=None):
        """
        Evaluates the antenna perturbation for a given poloidal and toroidal mode number.

        If m and n are not provided, the antenna will be evaluated for all mode numbers.
        """
        if m is None and self.m is None:
            raise ValueError("m must be provided if it was not provided in the constructor")
        if n is None and self.n is None:
            raise ValueError("n must be provided if it was not provided in the constructor")
        
        if m is not None:
            # If the user provides m, we will take the intersection of
            # the provided m and the m provided in the constructor (if any)
            # When not provided in the constructor, we will take all m provided by the user
            if self.m is None:
                m = np.asarray(m, dtype=np.int32)
            else:
                m = np.asarray(m, dtype=np.int32)
                m = np.intersect1d(m, self.m)
        else:
            m = self.m

        if n is not None:
            # If the user provides n, we will take the intersection of
            # the provided n and the n provided in the constructor (if any)
            # When not provided in the constructor, we will take all n provided by the user
            if self.n is None:
                n = np.asarray(n, dtype=np.int32)
            else:
                n = np.asarray(n, dtype=np.int32)
                n = np.intersect1d(n, self.n)
        else:
            n = self.n

        # nstate = m.size * n.size
        if nstate % 2 != 1:
            raise ValueError("nstate must be odd, to have a well defined center point")
        m_range = np.arange(-nstate//2, nstate//2 + 1)
        antenna = np.zeros((self.coeffs.shape[0], 2*nstate), dtype=np.complex128)

        if self.which.lower() == 'xi':
            state_slice = slice(nstate, 2*nstate)
        elif self.which.lower() == 'p1':
            state_slice = slice(0, nstate)

        # We take the intersection of the m and the m_range for slicing.
        idx_intersection = np.intersect1d(m_range, m, return_indices=True)[1]
        antenna[:, state_slice][..., idx_intersection] = np.tile(self.coeffs, (1, idx_intersection.size))

        return antenna


