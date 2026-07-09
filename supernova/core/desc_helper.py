"""
DESC-backed equilibrium wrapper for superNOVA.

This module is built on top of the equilibrium object from the pyCOCOS and
the superEquilibrium to provide a smoother interface to re-solve the equilibrium.
"""

import numpy as np
from typing import Optional
import xarray as xr

from ..radial.profile import Profile
from ..radial.profile import decompose2d
from .equilibrium import superEquilibrium
from pycocos.core.equilibrium import equilibrium
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import UnivariateSpline
import unyt

# Importing DESC.
from .desc_loader import *


def _check_units(val, unit: str, registry: unyt.UnitRegistry=None):
    """
    Check if the value has the correct units.
    """
    if hasattr(val, 'to'):
        return float(val.to(unit).value) * unyt.Unit(unit, registry=registry)
    
    return float(val) * unyt.Unit(unit, registry=registry)

def _enforce_1d_profile(profile: Profile, name: str) -> Profile:
    """
    Enforce that the profile is 1D.
    """
    if not isinstance(profile, Profile):
        raise TypeError(f"{name} must be a Profile object.")

    if profile.m_pol.size != 1 or profile.m_pol[0] != 0:
            raise NotImplementedError(f"{name} must be a flux-function Profile with mpol=0.")
    if profile.n_tor.size != 1 or profile.n_tor[0] != 0:
        raise NotImplementedError(f"{name} must be axisymmetric with ntor=0.")
    return profile.copy()

def _create_zero_profile(name: str, x: np.ndarray, **attrs) -> Profile:
    """
    Create a zero profile with the given name and x values.
    """
    return Profile(
            x=x,
            F_nm=np.zeros_like(x),
            attrs=attrs,
        )

def _optional_flux_function(name: str, x: np.ndarray, profile: Optional[Profile]=None, **attrs) -> Profile:
    """
    Create a zero profile with the given name and x values if the profile is None.
    """
    if profile is None:
        return _create_zero_profile(name, x, **attrs)
    
    return _enforce_1d_profile(profile, name)

def _build_mapping_psipol_psitor(psitor: np.ndarray, qprof: np.ndarray):
    """
    From the psipol and the q-profile, build a mapping from psipol to psitor.

    dpsitor/dpsipol = q
    """
    psitor = cumulative_trapezoid(1/qprof, psitor, initial=0)
    psitor /= psitor[-1]  # Normalize to 1 at the edge.
    return UnivariateSpline(psitor, psitor)

def _all_on_the_same_grid(profiles: list[Profile]) -> bool:
    """
    Check if all profiles are on the same grid.
    """
    if len(profiles) < 2:
        return True

    x0 = profiles[0].x
    for profile in profiles[1:]:
        if not np.allclose(x0, profile.x):
            return False
    return True

def _check_lcfs(R: np.ndarray, z: np.ndarray, theta: np.ndarray):
    """
    Check if the LCFS is valid.
    """
    if R.ndim != 1 or z.ndim != 1 or theta.ndim != 1:
        raise ValueError("R, z, and theta must be 1D arrays.")
    if R.size != z.size or R.size != theta.size:
        raise ValueError("R, z, and theta must have the same size.")
    if not np.all(np.diff(theta) > 0):
        raise ValueError("theta must be monotonically increasing.")
    
def _fit_lcfs_to_fourier(R: np.ndarray, z: np.ndarray, theta: np.ndarray, mmax: int=10):
    """
    Fit the LCFS to a Fourier series.
    """
    Rmn = decompose2d(R, m_max=mmax, angle=theta)
    Zmn = decompose2d(z, m_max=mmax, angle=theta)

    # These are the modes in the complex exponential basis, but
    # for DESC we need the modes in the sin/cos basis. The conversion is:
    # Rmn = R_lmn[0] + sum_{m=1}^{M} R_lmn[m] * cos(m*theta) + sum_{m=1}^{M} R_lmn[m+M] * sin(m*theta)
    # Zmn = Z_lmn[0] + sum_{m=1}^{M} Z_lmn[m] * cos(m*theta) + sum_{m=1}^{M} Z_lmn[m+M] * sin(m*theta)
    R_lmn = np.zeros(2*mmax+1, dtype=np.float64)
    Z_lmn = np.zeros(2*mmax+1, dtype=np.float64)
    modes = [(0, 0)] + [(m, 0) for m in range(1, mmax+1)] + [(m, 0) for m in range(1, mmax+1)]
    R_lmn[0] = Rmn[0]
    Z_lmn[0] = Zmn[0]
    for m in range(1, mmax+1):
        R_lmn[m] = 0.5 * (Rmn[m] + Rmn[-m])
        R_lmn[m+mmax] = -0.5j * (Rmn[m] - Rmn[-m])
        Z_lmn[m] = 0.5 * (Zmn[m] + Zmn[-m])
        Z_lmn[m+mmax] = -0.5j * (Zmn[m] - Zmn[-m])
    return FourierRZToroidalSurface(
        R_lmn=R_lmn,
        Z_lmn=Z_lmn,
        modes_R=modes,
        modes_Z=modes,
        NFP=1,
        sym=False,
        M=mmax,
        N=0,
    )

def _make_spline_profiles_desc(profiles: list[Profile]) -> list[SplineProfile]:
    """
    Convert a list of Profile objects to a list of SplineProfile objects.
    """
    spline_profiles = []
    for profile in profiles:
        if profile is None:
            spline_profiles.append(None)
            continue
        spline_profiles.append(SplineProfile(profile.F_nm.values.squeeze().real,
                                             rho=np.sqrt(profile.x.values),))
    return spline_profiles

def _make_guesses(equ: equilibrium, 
                  rhopol: np.ndarray, 
                  rhotor: np.ndarray):
    """
    From a given equilibrium, it will return a list 
    of points to initialize the DESC solver.
    """
    nodes_list = []
    R_list = []
    Z_list = []

    # Axis location used only to define a smooth geometric theta label.
    Raxis = float(equ.fluxdata.Raxis)
    Zaxis = float(getattr(equ.fluxdata, "Zaxis", 0.0))

    for s, rho in zip(rhopol, rhotor):
        R_path, Z_path = equ.flux_surface(s)

        R_path = np.asarray(R_path, dtype=float)
        Z_path = np.asarray(Z_path, dtype=float)

        # Remove repeated closure point if present.
        if R_path.size > 1 and np.allclose(
            [R_path[0], Z_path[0]],
            [R_path[-1], Z_path[-1]],
            rtol=1e-10,
            atol=1e-12,
        ):
            R_path = R_path[:-1]
            Z_path = Z_path[:-1]

        # DESC's theta convention is clockwise in the R-Z plane.
        # Standard arctan2 increases counter-clockwise, so use minus sign.
        theta = -np.unwrap(np.arctan2(Z_path - Zaxis, R_path - Raxis))

        # Shift and sort for cleanliness. The fit itself does not require
        # uniformly spaced theta.
        theta = np.mod(theta - theta[0], 2 * np.pi)
        order = np.argsort(theta)

        theta = theta[order]
        R_path = R_path[order]
        Z_path = Z_path[order]

        nodes_i = np.column_stack(
            [
                np.full_like(theta, rho),
                theta,
                np.full_like(theta, 0.0),  # phi = 0 for axisymmetric surfaces
            ]
        )

        nodes_list.append(nodes_i)
        R_list.append(R_path)
        Z_list.append(Z_path)

    nodes = np.vstack(nodes_list)
    R_values = np.concatenate(R_list)
    Z_values = np.concatenate(Z_list)

    grid = Grid(nodes, NFP=1, sort=False)

    return (
        grid,
        R_values,
        Z_values
    )



class DESCSolver:
    """
    """

    def __init__(self, LCFS: dict, Baxis: float, q: Profile, 
                 density: Profile, pressure: Profile, resistiviy: Profile, 
                 vtor: Optional[Profile] = None, vpol: Optional[Profile] = None,
                 **kwargs):
        """
        Builds the DESCSolver object with the LCFS and q-profile as inputs.

        All the input profiles must be flux functions (i.e., mpol=0, ntor=0) and the
        x basis must be the "normalized toroidal flux" (i.e., x=psi_tor/psi_lcfs).
        The LCFS must contain the keys 'R', 'Z', and 'theta'.

        Parameters
        ----------
        LCFS : dict
            Dictionary containing the LCFS information. Must contain the keys 'R', 'Z', and
            'theta'.
        Baxis : float
            Magnetic field at the magnetic axis in Tesla.
        q : Profile
            Safety factor profile as a flux function (mpol=0, ntor=0).
        density : Profile
            Density profile as a flux function (mpol=0, ntor=0).
        pressure : Profile
            Pressure profile as a flux function (mpol=0, ntor=0).
        resistiviy : Profile
            Resistivity profile as a flux function (mpol=0, ntor=0).
        vtor : Profile, optional
            Toroidal velocity profile as a flux function (mpol=0, ntor=0
        vpol : Profile, optional
            Poloidal velocity profile as a flux function (mpol=0, ntor=0)
        desc_resolution : tuple[int, int], optional
            Resolution of the DESC solver (L, M) where L is the number of radial basis 
            functions and M is the number of poloidal basis functions. Axisymmetry is assumed.
        """

        # Checking the LCFS has the required keys.
        if not all(key in LCFS for key in ["R", "Z", "theta"]):
            raise ValueError("LCFS must contain keys 'R', 'Z', and 'theta'.")
        
        # Checking the units in R, z, theta:
        self.Rlcfs = _check_units(LCFS["R"], "m")
        self.zlcfs = _check_units(LCFS["Z"], "m")
        self.theta = _check_units(LCFS["theta"], "rad")
        self.Baxis = _check_units(Baxis, "T")
        
        # Raise error if the LCFS is not valid.
        _check_lcfs(self.Rlcfs, self.zlcfs, self.theta)

        if self.Baxis.v == 0:
            raise ValueError("Baxis must be non-zero.")
        
        self.qin = _enforce_1d_profile(q, "q")
        self.density = _enforce_1d_profile(density, "density")
        self.pressure = _enforce_1d_profile(pressure, "pressure")
        self.resistivity = _optional_flux_function("resistivity", self.density.x, resistiviy, units="Ohm*m")
        self.vtor = _optional_flux_function("vtor", self.density.x, vtor, units="m/s")
        self.vpol = _optional_flux_function("vpol", self.density.x, vpol, units="m/s")

        # Checking all profiles are on the same grid.
        if not _all_on_the_same_grid([self.qin, self.density, self.pressure, self.resistivity, self.vtor, self.vpol]):
            raise ValueError("All profiles must be on the same grid.")
        
        # We first fit the LCFS to a Fourier series as a boundary condition for DESC.
        self.lcfs_fit = _fit_lcfs_to_fourier(self.Rlcfs, self.zlcfs, self.theta)
        self.pressure = _make_spline_profiles_desc([self.pressure])[0]
        self.density = _make_spline_profiles_desc([self.density])[0]
        self.iota = _make_spline_profiles_desc([1./self.q_in])[0]


    def solve(self, initial_guess: Optional[equilibrium]=None,
              max_psi_iterations: int=4, 
              desc_resolution: tuple[int, int]=(6, 6), 
              rtol: float=1e-2, fast: bool=False,
              **kwargs) -> superEquilibrium:
        """
        Solve the equilibrium using DESC.

        Parameters
        ----------
        max_psi_iterations : int, optional
            Maximum number of iterations for the psi solver. Default is 4.
        desc_resolution : tuple[int, int], optional
            Resolution of the DESC solver (L, M) where L is the number of radial basis 
            functions and M is the number of poloidal basis functions. Axisymmetry is assumed.
        """
        # We will iteratively solve for the equilibrium, updating the psi profile each time and 
        # the psitor until we get the desired Btor convergence at the axis.
        # 1. We get an initial guess for the Psitor:
        aminor = 0.5 * (np.max(self.Rlcfs) - np.min(self.Rlcfs))
        area   = np.pi * aminor**2
        psitor_guess = (self.Baxis * area).to('Wb').value

        # We take from the initial_guess some flux surfaces to initialize the DESC solver.
        if initial_guess is not None:
            psitor = self.qin.x
            psipol = _build_mapping_psipol_psitor(psitor, 
                                                  self.qin.data.values.squeeze())(psitor)
            grid, R, z = _make_guesses(initial_guess,
                                       self.qin.x)
        else:
            guesses = None

        L, M = desc_resolution
        last_eq = None
        last_baxis = np.nan
        converged = False
        history = []
        for ii in range(max(max_psi_iterations, 1)):
            # 2. We build the DESC equilibrium object.
            desc_equilibrium = DESC_Equilibrium(
                L = L,
                M = M,
                N = 0,
                L_grid = max(2*L, L + 3),
                M_grid = max(2*M, M + 3),
                N_grid = 0,
                surface=self.lcfs_fit,
                pressure=self.pressure,
                iota=self.iota,
                Psi=psitor_guess,
                electron_density=self.density,
                NFP=1,
                sym=False
            )
            
            if guesses is not None:
                desc_equilibrium.set_initial_guess(guesses)

            # 3. We solve the DESC equilibrium.
            if kwargs.get("verbose", False):
                print("--- Solving DESC equilibrium. Iteration #{} ---".format(ii+1))

            if fast:
                family, _ = desc_equilibrium.solve(verbose=0, copy=True)
            else:
                if 'verbose' in kwargs:
                    del kwargs['verbose']
                family = solve_continuation_automatic(desc_equilibrium, verbose=0, **kwargs)
                try:
                    ii_equ = family[-1]
                except TypeError:
                    ii_equ = family

            # We compute the new magnetic field on axis:
            B0axis = float(np.asarray(ii_equ.compute("<|B|>_axis")['<|B|>_axis']))

            # Computing the error in the magnetic field on axis:
            error = np.abs(B0axis - self.Baxis.to('T').value) / self.Baxis.to('T').value

            last_eq = ii_equ
            last_baxis = B0axis

            tmp = { 'psitor': psitor_guess,
                    'Baxis': last_baxis,
                    'error': error,}

            if error < rtol:
                converged = True
                break

            history.append(tmp)

            psitor_guess *= self.Baxis.to('T').value / B0axis
        
        if not converged:
            raise RuntimeError(f"DESC solver did not converge after {max_psi_iterations} " +
                               f"iterations. Last Baxis: {last_baxis}, Target Baxis: " +
                               f"{self.Baxis.to('T').value}, Relative error: {error}")
        
        self.last_equilibrium = last_eq
        self.last_baxis = last_baxis
        self.history = history
        
        return last_eq, last_baxis, history
    
    def build_magnetic_coordinates(self):
        r"""
        Using the smooth magnetic field from DESC we can build the magnetic coordinates
        object.

        For the magnetic coordinates object we need:

        - Flux functions:
            - I: plasma current.
            - q: safety factor profile.
            - g: R*B_tor
        - Metric coefficients: g_ij and g^ij
            - where i = {\psi_{pol}, \theta_{Boozer}, \zeta_{Boozer}}
        - Derivatives of the transformation:
            - dR/d\psi_{pol}, dR/d\theta_{Boozer}, dR/d\zeta_{Boozer}
            - dz/d\psi_{pol}, dz/d\theta_{Boozer}, dz/d\zeta_{Boozer}
            - d\phi/d\psi_{pol}, d\phi/d\theta_{Boozer}, d\phi/d\zeta_{Boozer}
        - Jacobian of the transformation: \sqrt{g} = h/B^2
        - Mapping between magnetic and cylindrical coordinates.
            - R=R(\psi_{pol}, \theta_{Boozer}, \zeta_{Boozer})
            - z=z(\psi_{pol}, \theta_{Boozer}, \zeta_{Boozer})
            - Psi_pol = Psi_pol(R, z)
            - Theta_Boozer = Theta_Boozer(R, z)
            - Zeta_Boozer = Zeta_Boozer(R, z)
        """
        list2eval =  ['R_r', 'R_t', 'R_z', # Derivative of R with to rhotor, theta and phi.
                      'Z_r', 'Z_t', 'Z_z', # Derivative of z with to rhotor, theta and phi.
                      'phi_r', 'phi_t', 'phi_z', # Derivative of phi with to rhotor, theta and phi.
                      'nu',                # Streaming function: nu = \zeta_Boozer - \phi
                      'nu_r', 'nu_t', 'nu_z',  # Derivatives of nu with respect to rhotor, theta and phi.
                      'G', 'I',            # Boozer flux functions.
                      'psi', 'psi_r', 'psi_t', 'psi_z', # Poloidal flux and its derivatives.
                      'theta_B', 'zeta_B', '|B|', 
                      'g^rr', 'g^rt', 'g^rz', 'g^tt', 'g^tz', 'g^zz', # Contravariant metric coefficients.
                      'g_rr', 'g_rt', 'g_rz', 'g_tt', 'g_tz', 'g_zz', # Covariant metric coefficients.
                      'R', 'Z' # The cylindrical coordinates.
                      ]
        
        if not hasattr(self, 'last_equilibrium'):
            raise RuntimeError("You must call the solve() method before calling build_magnetic_coordinates().")
        
        # Computing the elements on a grid of points in the (psi, theta) plane.
        grid = LinearGrid(rho=np.linspace(0, 1, 512), 
                          theta=np.linspace(0, 2*np.pi, 256), 
                          zeta=[0.0,], NFP=1)
        
        # We query all the elements we require to build the magnetic coordinates object.
        data = self.last_equilibrium.compute(list2eval, grid=grid)

        # All derivatives or values of 'r' that we computed are the sqrt of the normalized toroidal flux,
        # whereas we instead need the normalized poloidal flux.
        rhotor = np.linspace(0, 1, 512)
        q_at = self.qin.data.interp(x=rhotor**2, method='linear').values.squeeze()

        gpsi_psi = data['g^rr'] * 4 * rhotor[:, None]**2/q_at[:, None]**2
        gpsi_theta = data['g^rt'] * 2 * rhotor[:, None]/q_at[:, None]
        gpsi_zeta = data['g^rz'] * 2 * rhotor[:, None]/q_at[:, None]
        gtheta_theta = data['g^tt']
        gtheta_zeta = data['g^tz']
        gzeta_zeta = data['g^zz']

        # We still one extra step: in the previous metric elements, the
        # contravariant metric elements are defined in Theta and zeta that are not
        # the Boozer coordinates.
        # We compute the theta_B and zeta_B coordinates, 
    
    def to_pycocos(self, nr: int=101, nz: int=102) -> equilibrium:
        """
        It will transform the contents to a pycocos equilibrium object.

        This will compute the magnetic field on a given (R, z) grid and call the constructor
        of the object with the magnetic field evaluated at those points.
        """
        assert nr > 1 and nz > 1, "nr and nz must be greater than 1."
        assert hasattr(self, 'last_equilibrium'), "You must call the solve() method before calling to_pycocos()."

        # Getting the limits of the LCFS to build the grid.
        Rmin, Rmax = np.min(self.Rlcfs), np.max(self.Rlcfs)
        zmin, zmax = np.min(self.zlcfs), np.max(self.zlcfs)

        R = np.linspace(Rmin, Rmax, nr)
        z = np.linspace(zmin, zmax, nz)



        