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
        

    def get_full_pattern(self, m_range: np.ndarray, force_gen: bool=False) -> np.ndarray:
        r"""
        Get the full antenna pattern on the finite element space, without reducing to the
        non-deficient degrees of freedom.
        """
        if self.full_antenna is not None and not force_gen:
            return self.full_antenna

        # nstates = (2*mmax + 1)
        # m_range = np.arange(-mmax, mmax+1)
        nstates = m_range.size

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

        # Generate phases taking into account possible asymmetric m_range
        phases = np.empty(m_range.shape, dtype=complex)
        if self.parity in ('even', 'odd'):
            # build a lookup for non-negative m phases
            m_nonneg = np.unique(m_range[m_range >= 0])
            phases_nonneg = np.random.random(size=m_nonneg.shape) * 2.0 - 1.0
            phase_map = {int(m): ph for m, ph in zip(m_nonneg, phases_nonneg)}

            for idx, m in enumerate(m_range):
                if m >= 0:
                    phases[idx] = phase_map[int(m)]
                else:
                    abs_m = int(abs(m))
                    if abs_m in phase_map:
                        if self.parity == 'even':
                            phases[idx] = phase_map[abs_m]
                        else:  # odd
                            phases[idx] = 1j*phase_map[abs_m]
                    else:
                        # fallback: generate independent phase when counterpart missing
                        phases[idx] = np.random.random() * 2.0 - 1.0
        else:
            # no parity constraint: independent phases for all m
            phases = np.random.random(size=m_range.shape) * 2.0 - 1.0

        # Now we will fill the antenna pattern with the coefficients of the antenna
        # profile in the radial finite element basis, multiplied by the random phases.
        out[:, antenna_slice] += self.a_k[:, None] * phases[None, :]

        self.full_antenna = out.ravel(order="C")

        return self.full_antenna
    

    def get_reduced_antenna(self, m_range: np.ndarray) -> np.ndarray:
        r"""
        Get the antenna pattern projected on the finite element space, reduced to the
        non-deficient degrees of freedom.
        """
        # Get the full antenna pattern on the finite element space
        full_pattern = self.get_full_pattern(m_range)

        # Reduce to the non-deficient degrees of freedom
        reduced_pattern = self.bc.restrict(full_pattern)

        return reduced_pattern