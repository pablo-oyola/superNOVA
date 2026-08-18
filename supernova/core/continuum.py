"""
Implementation of a solver for the ideal MHD continuum spectra for
a tokamak equilibrium.
"""

import unyt

from .equilibrium import superEquilibrium
from ._continuum_kernel import assemble_full_matrices, assembly_slow_sound, assembly_only_saw
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from typing import Optional
from .isotopes import isotopes_mass
from scipy.linalg import eig, eigh
from scipy.constants import mu_0
from ._classify_mode import ContinuumClassifier
from scipy.sparse import coo_matrix
from scipy.sparse import linalg as spla

from tqdm import tqdm


def _biorthogonal_normalizations(vl: np.ndarray, B: np.ndarray,
                                 vr: np.ndarray) -> np.ndarray:
    """
    Return the per-mode generalized-eigenvector factors vl_k^H B vr_k.
    """
    nstate = vr.shape[1]
    normalizations = np.zeros(nstate, dtype=np.complex128)
    for k in range(nstate):
        normalizations[k] = np.vdot(vl[:, k], B @ vr[:, k])
    return normalizations


def _eigh_signed_definite(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve a Hermitian pencil whose mass matrix has either orientation.

    The Boozer Jacobian, and therefore both matrices in the reduced Alfvén
    and slow-sound pencils, may be globally negative under a valid coordinate
    convention.  ``scipy.linalg.eigh`` requires a positive-definite second
    matrix, so reverse both matrices when the zero Fourier harmonic of ``B``
    has the negative orientation.  A genuinely indefinite ``B`` is still
    rejected by ``eigh``.
    """
    pencil_sign = -1.0 if np.real(B[0, 0]) < 0.0 else 1.0
    return eigh(pencil_sign * A, pencil_sign * B, eigvals_only=True)


class superContinuum:
    """
    """
    def __init__(self, equ: superEquilibrium, gamma_s: float=5/3, species: str='D',
                 dense: bool=True):
        """
        Initializes the class by calling the superEquilibrium constructor and
        setting the parameters for the continuum solver.

        Parameters:
        -----------
        dense: bool
            Whether to use a dense grid in the radial direction for the continuum solver.
        gamma_s: float
            The adiabatic index for the plasma, typically 5/3 for an ideal monoatomic
            atomic gas.
        species: str
            The ion species to consider for the mass density, which will 
            affect the Alfvén speed and thus the continuum spectrum.
        equ: superEquilibrium
            The equilibrium object containing the profiles and geometry of the plasma.  
        """
        self.equ = equ
        self.dense = dense
        
        if not dense:
            raise NotImplementedError("Currently, only the dense option is implemented. The sparse option will be added in future releases.")

        if gamma_s < 0.0:
            raise ValueError("The adiabatic index gamma_s should be greater than 0.")
        self.gamma_s = gamma_s

        if species not in isotopes_mass:
            raise ValueError(f"Species {species} not recognized. Available species: {list(isotopes_mass.keys())}")
        
        self.species = species
        self.mass = isotopes_mass[species].to('amu').value * unyt.Unit("amu", registry=self.equ.registry)

        # Building the internal profiles for the continuum solver.
        self._build_profiles()

    def _build_profiles(self):
        r"""
        We build now the main profiles used to compute the continuum spectrum.
        These correspond to (using the Boozer Jacobian)
        H = g^{\psi\psi} / (Jacobian * B0^2) = gpsi_psi / h(\Psi)
        J = g^{\psi\psi} * Jacobian / B0^2 = gpsi_psi * h(\Psi) / B0^4
        K = 2 * kappa_s * Jacobian / B0^2 = 2 * kappa_s * h(\Psi) / B0^4
        L = (gamma_s * mu_0 * pressure + B0^2) / (Jacobian * B0^2) = (gamma_s * mu_0 * pressure + B0^2) / h(\Psi)
        For the slow-sound case:
        N = h/B0^4 * (gamma_s * mu_0 * pressure * kappa_s^2) / (gamma_s * mu_0 * pressure + B0^2)
        M = 1/(Jacobian * B0^2) = 1/h(\Psi)
        """

        names = ['gpsi_psi / h',
                 'gpsi_psi / B0**4 * h',
                 '2.0 * kappa_s/B0**4 * h',
                 f'({self.gamma_s} * pressure + B0**2) * h / B0**4',
                 f'h/B0**4 * (4 * {self.gamma_s} * pressure * kappa_s**2)/({self.gamma_s} * pressure + B0**2)',
                 '1.0 / h',
                 'pressure',
                 'density']
        
        tmp = self.equ(*names, normalize=True) # This will return profiles.
        self.H = tmp[0]
        self.J = tmp[1]
        self.K = tmp[2]
        self.L = tmp[3]
        self.N = tmp[4]
        self.M = tmp[5]
        self.p0 = tmp[6].data.squeeze().real
        self.n0 = tmp[7].data.squeeze().real

        # Use the first finite-radius operator as the temporary axis limit.
        for profile in [self.H, self.J, self.N]:
            if profile.x.size > 1 and np.isclose(profile.x[0], 0.0):
                profile.data.loc[dict(x=0.0)] = profile.data.isel(x=1).values

        # Normalization density, which we take as the density at the axis.
        densitynorm = (1 * unyt.Unit("naxis", registry=self.equ.registry)).to("1/aminor**3")
        self.n0 /= densitynorm.v

        # We will pad all the profiles to have the same (n, m)-range.
        mpol = np.empty(0, dtype=np.int32)
        for iprof in [self.H, self.J, self.K, self.L, self.N, self.M]:
            mpol = np.union1d(mpol, iprof.m_pol)

        for iprof in [self.H, self.J, self.K, self.L, self.N, self.M]:
            iprof.data = iprof.data.reindex(mpol=mpol, fill_value=0)

    def solve(self, ntor: int, mmax: int=None, psi_vals: np.ndarray=None, 
              which: str='full', return_eigvecs: bool=False,
              normalize_eigvals: bool=False, classify: bool=False, 
              max_dm: int=1000, max_dn: int=1000,
              **classify_kwargs):
        """
        Solves the continuum spectrum for the given equilibrium and parameters.

        Parameters:
        -----------
        ntor: int
            The toroidal mode number to consider for the continuum spectrum.
        psi_vals: np.ndarray, optional
            The specific values of the poloidal flux to evaluate the continuum spectrum at. If None, it will be evaluated at all available radial points.
        which: str
            Which part of the spectrum to return. Options are 'full', 'alfven', 'slow_sound'.
        return_eigvecs: bool
            Whether to return the eigenvectors along with the eigenvectors (left, right and normalizations).
        normalize_eigvals: bool
            Whether to normalize the eigenvalues.
        classify: bool
            Whether to classify the modes.
        **classify_kwargs
            Additional keyword arguments for the classification.

        Returns:
        --------
        frequencies: np.ndarray
            The computed continuum frequencies at the specified psi values.
        eigvecs: Optional[np.ndarray]
            The corresponding eigenvectors if return_eigvecs is True, otherwise None.
        """
        # Setting the defaults
        if psi_vals is None:
            psi_vals = self.H.data.x

        if classify and not return_eigvecs:
            return_eigvecs = True # We enforce this.

        if classify and not normalize_eigvals:
            normalize_eigvals = True # We enforce this, as the classification relies on the normalized eigenvalues.

        if classify and which.lower() != 'full':
            raise ValueError("Mode classification is currently only implemented for the full spectrum. Please set which='full' to use the classification.")

        # If the user provides no mode limit, we will take the maximum poloidal mode
        # number from the H profile.
        H_mmax = np.abs(self.H.m_pol).max()
        if mmax is None:
            mmax = int(H_mmax)
        else:
            mmax = min(mmax, int(H_mmax))

        # From the internal q-profile, we can determine for the given ntor the maximum
        # and minimum poloidal mode numbers that can be excited at all.
        qmin = self.equ.qprof.data.squeeze().real.min()
        qmax = self.equ.qprof.data.squeeze().real.max()

        mmin_q = int(np.floor(ntor * qmin)) - 3
        mmax_q = min(mmax, int(np.ceil(ntor * qmax)) + 3)

        if which.lower() not in ['full', 'alfven', 'slow_sound']:
            raise ValueError(f"Invalid option for 'which': {which}. Valid options are 'full', 'alfven', 'slow_sound'.")
        
        if which.lower() == 'full':
            slow_sound = False
        else:
            slow_sound = True

        psi_vals = np.asarray(psi_vals, dtype=np.float64).reshape(-1)
        m_range = np.arange(mmin_q, mmax_q + 1)
        m_range2sel = np.arange(mmax_q+1)
        nstate = m_range.size if slow_sound else 2 * m_range.size

        eigenvalues = np.zeros((psi_vals.size, nstate), dtype=np.complex128)

        if return_eigvecs:
            right_eigenvectors = np.zeros((psi_vals.size, nstate, nstate), dtype=np.complex128)
            left_eigenvectors = np.zeros((psi_vals.size, nstate, nstate), dtype=np.complex128)
            normalizations = np.zeros((psi_vals.size, nstate), dtype=np.complex128)

        # We interpolate the profiles at the position we are requested.
        H_nm = self.H.data.sel(mpol=m_range2sel).interp(x=psi_vals)
        J_nm = self.J.data.sel(mpol=m_range2sel).interp(x=psi_vals)
        qval = self.equ.qprof.data.squeeze().interp(x=psi_vals).real
        N_nm = self.N.data.sel(mpol=m_range2sel).interp(x=psi_vals)
        K_nm = self.K.data.sel(mpol=m_range2sel).interp(x=psi_vals)
        L_nm = self.L.data.sel(mpol=m_range2sel).interp(x=psi_vals)
        M_nm = self.M.data.sel(mpol=m_range2sel).interp(x=psi_vals)
        p0 = self.p0.interp(x=psi_vals)
        n0 = self.n0.interp(x=psi_vals)

        # We create a short-cut for the solver.
        if which.lower() == 'full':
            fun = lambda **kwargs: assemble_full_matrices(H_mn=kwargs['H_nm'], 
                                                            K_mn=kwargs['K_mn'],
                                                            J_mn=kwargs['J_nm'], 
                                                            L_mn=kwargs['L_mn'],
                                                            M_mn=kwargs['M_mn'],
                                                            p0=kwargs['p0'],
                                                            q=kwargs['q'], 
                                                            m_range=kwargs['m_range'], 
                                                            ntor=kwargs['ntor'],
                                                            gammas=self.gamma_s,
                                                            max_dm=kwargs.get('max_dm', 1000),
                                                            max_dn=kwargs.get('max_dn', 1000))
        elif which.lower() == 'slow_sound':
            fun = lambda **kwargs: assembly_slow_sound(H_mn=kwargs['H_nm'], 
                                                        J_mn=kwargs['J_nm'], 
                                                        N_mn=kwargs['N_mn'],
                                                        q=kwargs['q'],
                                                        m_range=kwargs['m_range'],
                                                        ntor=kwargs['ntor'])
                                                              
        else:
            fun = lambda **kwargs: assembly_only_saw(H_mn=kwargs['H_nm'], 
                                                    J_mn=kwargs['J_nm'], 
                                                    q=kwargs['q'],
                                                    m_range=kwargs['m_range'],
                                                    ntor=kwargs['ntor'])
            
        def _solve_(A, B):
            if slow_sound: # We can use eigh.
                if return_eigvecs:
                    raise NotImplementedError("Currently, the return of eigenvectors is only implemented for the full case. This will be added for the slow_sound case in future releases.")
                else:
                    w = _eigh_signed_definite(A, B)
                    order = np.argsort(np.real(w))
                    w = w[order]
                    vl = None
                    vr = None

            else:
                if return_eigvecs:
                    w, vl, vr = eig(A, B, left=True, right=True)
                    order = np.argsort(np.real(w))
                    w = w[order]
                    vl = vl[:, order]
                    vr = vr[:, order]
                else:
                    w = eig(A, B, left=False, right=False)
                    order = np.argsort(np.real(w))
                    w = w[order]
                    vl = None
                    vr = None
            return w, vl, vr

        # Solving for the x-points.
        for ii, _ in enumerate(tqdm(psi_vals)):
            iH_mn = H_nm.isel(x=ii).values
            iJ_mn = J_nm.isel(x=ii).values
            iqval = qval.isel(x=ii).values.item()
            iK_mn = K_nm.isel(x=ii).values
            iL_mn = L_nm.isel(x=ii).values
            iM_mn = M_nm.isel(x=ii).values
            iN_mn = N_nm.isel(x=ii).values
            ip0 = p0.isel(x=ii).values.item()

            A, B = fun(H_nm=iH_mn, J_nm=iJ_mn, 
                       K_mn=iK_mn, L_mn=iL_mn, M_mn=iM_mn, N_mn=iN_mn,
                       q=iqval, m_range=m_range, ntor=ntor, p0=ip0)
            
            # We obtain the eigenvalues and eigenvectors of the generalized eigenvalue problem A v = lambda B v.
            w, vl, vr = _solve_(A, B)

            eigenvalues[ii, :w.size] = w
        
            if return_eigvecs:
                right_eigenvectors[ii] = vr
                left_eigenvectors[ii] = vl
                normalizations[ii] = _biorthogonal_normalizations(vl, B, vr)

        if normalize_eigvals:
            normfrequency = n0.data
            eigenvalues /= normfrequency[..., None]
            mass_scale = (1 * unyt.Unit("amu", registry=self.equ.registry) / self.mass)
            eigenvalues *= mass_scale * unyt.Unit("omegaA", registry=self.equ.registry)**2
            
        if return_eigvecs:
            eigvecs = {
                'right': right_eigenvectors,
                'left': left_eigenvectors,
                'normalizations': normalizations
            }

            # We need to return the classification if requested.
            if classify:
                classifier = ContinuumClassifier(vr=right_eigenvectors, vl=left_eigenvectors, w=eigenvalues.to('Hz**2').v,
                                                 J=J_nm.values, L=L_nm.values, m_range=m_range, **classify_kwargs)
                return eigenvalues, eigvecs, classifier

            return eigenvalues, eigvecs
        else:             
            return eigenvalues

    def build_AB(self, ntor: int, mmax: int=None, psi_vals: np.ndarray=None):
        """
        Constructs the A and B matrices without solving the eigenvalue problem.

        Parameters:
        -----------
        ntor: int
            The toroidal mode number to consider for the continuum spectrum.
        psi_vals: np.ndarray, optional
            The specific values of the poloidal flux to evaluate the continuum spectrum at. If None, it will be evaluated at all available radial points.

        Returns:
        --------
        A: np.ndarray
            The A matrix of the generalized eigenvalue problem.
        B: np.ndarray
            The B matrix of the generalized eigenvalue problem.
        """
        # Setting the defaults
        if psi_vals is None:
            psi_vals = self.H.data.x

        # If the user provides no mode limit, we will take the maximum poloidal mode
        # number from the H profile.
        H_mmax = np.abs(self.H.m_pol).max()
        if mmax is None:
            mmax = int(H_mmax)
        else:
            mmax = min(mmax, int(H_mmax))


        psi_vals = np.asarray(psi_vals, dtype=np.float64).reshape(-1)
        m_range = np.arange(-mmax, mmax + 1)
        m_range2sel = np.arange(mmax+1)
        nstate = 2 * m_range.size

        A = np.zeros((psi_vals.size, nstate, nstate), dtype=np.complex128)
        B = np.zeros((psi_vals.size, nstate, nstate), dtype=np.complex128)

        # We interpolate the profiles at the position we are requested.
        H_nm = self.H.data.sel(mpol=m_range2sel).interp(x=psi_vals)
        J_nm = self.J.data.sel(mpol=m_range2sel).interp(x=psi_vals)
        qval = self.equ.qprof.data.squeeze().interp(x=psi_vals).real
        N_nm = self.N.data.sel(mpol=m_range2sel).interp(x=psi_vals)
        K_nm = self.K.data.sel(mpol=m_range2sel).interp(x=psi_vals)
        L_nm = self.L.data.sel(mpol=m_range2sel).interp(x=psi_vals)
        M_nm = self.M.data.sel(mpol=m_range2sel).interp(x=psi_vals)
        p0 = self.p0.interp(x=psi_vals)
        n0 = self.n0.interp(x=psi_vals)

        # We create a short-cut for the solver.
        fun = lambda **kwargs: assemble_full_matrices(H_mn=kwargs['H_nm'], 
                                                        K_mn=kwargs['K_mn'],
                                                        J_mn=kwargs['J_nm'], 
                                                        L_mn=kwargs['L_mn'],
                                                        M_mn=kwargs['M_mn'],
                                                        p0=kwargs['p0'],
                                                        q=kwargs['q'], 
                                                        m_range=kwargs['m_range'], 
                                                        ntor=kwargs['ntor'],
                                                        gammas=self.gamma_s)
            

        # Solving for the x-points.
        for ii, _ in enumerate(tqdm(psi_vals)):
            iH_mn = H_nm.isel(x=ii).values
            iJ_mn = J_nm.isel(x=ii).values
            iqval = qval.isel(x=ii).values.item()
            iK_mn = K_nm.isel(x=ii).values
            iL_mn = L_nm.isel(x=ii).values
            iM_mn = M_nm.isel(x=ii).values
            iN_mn = N_nm.isel(x=ii).values
            ip0 = p0.isel(x=ii).values.item()

            Aii, Bii = fun(H_nm=iH_mn, J_nm=iJ_mn, 
                       K_mn=iK_mn, L_mn=iL_mn, M_mn=iM_mn, N_mn=iN_mn,
                       q=iqval, m_range=m_range, ntor=ntor, p0=ip0)
            
            A[ii] = Aii
            B[ii] = Bii

        return A, B

def continuum_crossings(x: float, w: np.ndarray, freq: float) -> np.ndarray:
    """
    Returns the points where the continuum crosses a given frequency.

    Parameters:
    -----------
    x: np.ndarray
        The radial positions (normalized poloidal flux).
    w: np.ndarray
        The continuum frequencies at the given x positions.
    freq: float
        The frequency to find crossings with.

    Returns:
    --------
    crossings: np.ndarray
        The x positions where the continuum crosses the given frequency.
    """
    dwsaw_si = np.sqrt(w.real).to('kHz').v/(2*np.pi) - freq
    dwsaw_si = np.sign(dwsaw_si[1:, :] * dwsaw_si[:-1, :])

    # We mark with an 'x' the points where the continuum is cut by the frequency.
    # aka, those points where dwsaw_si is negative.
    crossings = []
    crossing_branches = []
    
    for ibranch in range(w.shape[1]):
        cut_points = np.where(dwsaw_si[:, ibranch] < 0)[0]

        # These crossings are brute-force approximation. We take a line that goes from the x to the following point and
        # evaluate where they cross by zero.
        for cut in cut_points:
            x1, x2 = x[cut], x[cut+1]
            y1= np.sqrt(w.real).to('kHz').v[cut, ibranch]/(2*np.pi) - freq
            y2 = np.sqrt(w.real).to('kHz').v[cut+1, ibranch]/(2*np.pi) - freq
            # Linearly interpolate to find the crossing point.
            if y2 != y1:  # Avoid division by zero
                x_cross = x1 - y1 * (x2 - x1) / (y2 - y1)
                crossings.append(x_cross)
                crossing_branches.append(ibranch*np.ones_like(x_cross, dtype=np.int16))
    
    return np.array(crossings), np.array(crossing_branches)

def get_knots_clustered_at(x: np.ndarray, nknots: int, eta: float=1e-5,
                           threshold: float=None) -> np.ndarray:
    """
    Returns a set of knots clustered at the points in x.

    Parameters:
    -----------
    x: np.ndarray
        The points to cluster the knots around.
    nknots: int
        The number of knots to generate.
    threshold: int, optional
        If provided, this will provide a way to distribute better the 
        knots.

    Returns:
    --------
    knots: np.ndarray
        The generated knots clustered around the points in x.
    """
    if threshold is None:
        threshold = 0.01 # Default threshold.

    eta = np.minimum(1e-5, eta)

    # We compute the cumulative distribution for the resonant layer.
    xnew = np.linspace(0.0, 1.0, 10000) # Midpoints between the original x points, where dwsaw_si is defined.
    singular_layer = np.zeros(xnew.shape, dtype=np.float64)
    eta = 1e-3 # Regularizer. We will use a Lorentzian to approximate the delta function at the singular layer.
    for x_cross in x:
        singular_layer += eta / ((xnew - x_cross)**2 + eta**2)

    cumulative = np.cumsum(np.maximum(threshold*singular_layer.max(), singular_layer))
    cumulative -= cumulative[0]  # Start from zero
    cumulative /= cumulative[-1]  # Normalize to [0, 1]

    new_knots = np.interp(np.linspace(0.0, 1.0, nknots), cumulative, xnew)

    return new_knots
