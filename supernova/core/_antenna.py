"""
This class will contain a set of functions used to generate and project the antenna
pattern on the model finite elements.
"""

import numpy as np
from ..radial.base import RadialFE
from ._boundary import BoundaryCondition


class Antenna:
    r"""
    This represents a forcing term in the MHD equation, that is written as:

    T(\omega)x = b,

    where x = (p_1, \xi^\psi), T(\omega) is the whole ideal MHD operator
    and b is the antenna term.
    """
    def __init__(self, rho: np.ndarray,
                 profile: np.ndarray,
                 radial_fe: RadialFE, bc: BoundaryCondition,
                 which: str='p1', parity: str='none'):
        r"""
        Generates the profile of the antenna and projects it to the finite element space,
        reducing it to the non-deficient degrees of freedom.
        
        """
        self.radial_fe = radial_fe
        self.bc = bc
        self.rho = np.asarray(rho)
        self.profile = np.asarray(profile).squeeze()

        if self.profile.ndim != 1:
            raise ValueError("The profile must be a 1D array.")

        if self.profile.shape[0] != self.rho.shape[0]:
            raise ValueError("The profile and the density must have the same shape.")

        if which.lower() not in ['p1', 'xi', 'both']:
            raise ValueError("which must be one of 'p1', 'xi' or 'both'.")
        if parity.lower() not in ['even', 'odd', 'none']:
            raise ValueError("parity must be one of 'even', 'odd' or 'none'.")
        
        self.which = which.lower()
        self.parity = parity.lower()

        # We will obtain the coefficients of the antenna profile in the 
        # radial finite element basis.
        self.a_k = self.radial_fe.project(self.profile, x=self.rho)
        self.full_antenna = None
        

    def get_full_pattern(self, mmax: int, force_gen: bool=False) -> np.ndarray:
        r"""
        Get the full antenna pattern on the finite element space, without reducing to the
        non-deficient degrees of freedom.
        """
        if self.full_antenna is not None and not force_gen:
            return self.full_antenna

        nstates = (2*mmax + 1)
        m_range = np.arange(-mmax, mmax+1)

        out = np.zeros((self.radial_fe.get_ndofs(), 2*nstates), dtype=complex)

        # Checking on what slices we need to add the antenna pattern.
        p1_slice = slice(0, nstates)
        xi_slice = slice(nstates, 2*nstates)
        if self.which == 'p1':
            slice_size = nstates
            antenna_slice = p1_slice
        elif self.which == 'xi':
            slice_size = nstates
            antenna_slice = xi_slice
        else:
            slice_size = 2*nstates
            antenna_slice = slice(0, 2*nstates)
        
        if self.parity == 'even':
            # We will generate modes that are even, we will only generate the 
            # modes with m >= 0 and use the same ones for the m < 0.
            phases = np.random.random(size=m_range[m_range >= 0].shape) * 2.0 - 1.0
            phases = np.concatenate([phases[:0:-1]/2.0, [phases[0], ], phases[1:]/2.0])
        elif self.parity == 'odd':
            # We will generate modes that are odd, we will only generate the 
            # modes with m >= 0 and use the opposite ones for the m < 0.
            phases = np.random.random(size=m_range[m_range >= 0].shape) * 2.0 - 1.0
            phases = np.concatenate([-phases[:0:-1]/(2.0j), [0.0,], phases[1:]/(2.0j)])
        else: # self.parity == 'none'
            # We will generate modes with no particular parity, we will generate
            # all the modes independently.
            phases = np.random.random(size=m_range.shape) * 2.0 - 1.0

        # Now we will fill the antenna pattern with the coefficients of the antenna
        # profile in the radial finite element basis, multiplied by the random phases.
        out[:, antenna_slice] += self.a_k[:, None] * phases[None, :]

        self.full_antenna = out.ravel(order="C")

        return self.full_antenna
    

    def get_reduced_antenna(self, mmax: int) -> np.ndarray:
        r"""
        Get the antenna pattern projected on the finite element space, reduced to the
        non-deficient degrees of freedom.
        """
        # Get the full antenna pattern on the finite element space
        full_pattern = self.get_full_pattern(mmax)

        # Reduce to the non-deficient degrees of freedom
        reduced_pattern = self.bc.restrict(full_pattern)

        return reduced_pattern