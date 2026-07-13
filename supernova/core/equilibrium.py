"""
Local implementation of the equilibrium solver.

This will extend functionalities to the Equilibrium/EQDSK class
of the pyCOCOS library to make it more usable in the context of
the superNOVA code.

@TODO: some kind of the equilibrium processing would be appreciated, 
like "resolving" the equilibrium on the magnetic coordinates (maybe DESC 
would usable?).
"""

import numpy as np
import xarray as xr
from pycocos import Equilibrium
from ..radial.profile import Profile
from findiff import Diff
from .._logger import get_logger
from ..radial.profile import ProfileFactoryPsiThetaZeta

logger = get_logger(__name__)

from typing import Optional, List, Any, Tuple
import unyt
from ._mhd_equivalence import (
    _has_unyt_units,
    _sync_xarray_units_attr,
    _unytfy_my_xarray,
    _in_unit_system,
    _deunytfy_my_xarray,
    _to_normalized_expression_units,
    _expression_binary_op,
    create_nova_registry
)

# --- Used to interprete user-provided operations.
import ast
import operator

_ALLOWED_EXPRESSION_FUNCTIONS = {
    "abs": np.abs,
    "arccos": np.arccos,
    "arcsin": np.arcsin,
    "arctan": np.arctan,
    "cos": np.cos,
    "cosh": np.cosh,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "sign": np.sign,
    "sin": np.sin,
    "sinh": np.sinh,
    "sqrt": np.sqrt,
    "tan": np.tan,
    "tanh": np.tanh,
}
_ALLOWED_EXPRESSION_CONSTANTS = {"pi": np.pi, "e": np.e, "mu_0": 1*unyt.mu_0}


def _axisymmetric_cylindrical_local_shear(A, R, d_dr, d_dz):
    """Return NOVA local shear from an axisymmetric cylindrical vector A."""
    radius = R[:, None] if np.ndim(R) == 1 else R
    curl_A = np.array(
        [
            -d_dz(A[1]),
            d_dz(A[0]) - d_dr(A[2]),
            d_dr(radius * A[1]) / radius,
        ]
    )
    return np.sum(A * curl_A, axis=0)



class superEquilibrium:
    """
    This extends the functionalities of the pyCOCOS equilibrium class.

    It will also contain important equilibrium profiles like the background
    flows, density, pressur and resistivity profiles, which are important for the stability analysis.
    """
    def __init__(self, equ: Equilibrium, density: Profile, pressure: Profile,
                 m_max: int=128,
                 resistivity: Profile=None, vtor: Profile=None, vpol: Profile=None,
                 rhodomain_boozer: Optional[Tuple[float, float]]=None,
                 res_boozer: Optional[Tuple[int, int]]=None):
        """
        Initializes the class with the equilbrium and the profiles. From here,
        we will compute the Boozer coordinates and compute the metric coefficients.

        Parameters
        ----------
        equ : Equilibrium
            The equilibrium object from pyCOCOS.
        density : Profile
            The density profile.
        pressure : Profile
            The pressure profile.
        vtor : Optional[Profile], optional
            The toroidal flow velocity profile.
        vpol : Optional[Profile], optional
            The poloidal flow velocity profile.
        resistivity : Optional[Profile], optional
            The resistivity profile.
        """
         # We store internally the equilibrium data
        self.equ: Equilibrium = equ
        self.density: Profile = density
        self.pressure: Profile = pressure
        self.vtor: Profile = vtor
        self.vpol: Profile = vpol
        self.eta: Profile = resistivity


        # We validate the inputs:.
        # We assume for the moment only m=0 and n=0 are allowed:
        # Let's raise some temporal errors if the profiles have toroidal
        # or poloidal modes different from 0.
        if self.density.m_pol.size > 1 or self.density.m_pol[0] != 0:
            raise NotImplementedError("Currently the implementation is only valid for flux-surface dependent profiles.")
        if self.pressure.m_pol.size > 1 or self.pressure.m_pol[0] != 0:
            raise NotImplementedError("Currently the implementation is only valid for flux-surface dependent profiles.")
        if self.vtor is None:
            self.vtor = Profile(x=self.density.data.x, mpol=np.array([0]), ntor=np.array([0]), F_nm=np.zeros_like(self.density.data.x))
        else:
            if self.vtor is not None and (self.vtor.m_pol.size > 1 or self.vtor.m_pol[0] != 0):
                raise NotImplementedError("Currently the implementation is only valid for flux-surface dependent profiles.")
        if self.vpol is None:
            self.vpol = Profile(x=self.density.data.x, mpol=np.array([0]), ntor=np.array([0]), F_nm=np.zeros_like(self.density.data.x))
        else:
            if self.vpol is not None and (self.vpol.m_pol.size > 1 or self.vpol.m_pol[0] != 0):
                raise NotImplementedError("Currently the implementation is only valid for flux-surface dependent profiles.")
        if self.eta is None:
            self.eta = Profile(x=self.density.data.x, mpol=np.array([0]), ntor=np.array([0]), F_nm=np.zeros_like(self.density.data.x))
        else:
            if self.eta is not None and (self.eta.m_pol.size > 1 or self.eta.m_pol[0] != 0):
                raise NotImplementedError("Currently the implementation is only valid for flux-surface dependent profiles.")
            
        # Verifying all the profiles are defined in the same x grid.
        if not (np.array_equal(self.density.x, self.pressure.x) and
                np.array_equal(self.density.x, self.vtor.x) and
                np.array_equal(self.density.x, self.vpol.x) and
                np.array_equal(self.density.x, self.eta.x)):
            raise ValueError("All profiles need to be defined in the same x grid.")


        # -----
        # Building the Boozer coordinates and metric.
        # -----
        # First, we validate the inputs.
        if rhodomain_boozer is None:
            rhodomain_boozer = [0.005, 1.0-0.005]
        if res_boozer is None:
            res_boozer = [511, 512]

        a, b = rhodomain_boozer
        npsi, ntheta = res_boozer

        if a < 0 or b > 1 or a >= b:
            raise ValueError(f"Invalid rhodomain_boozer. Should be in the range [0, 1] and a < b ({a}, {b}).")
        if npsi < 2 or ntheta < 2:
            raise ValueError(f"Invalid res_boozer. Should be at least [2, 2], got instead {npsi}, {ntheta}.")

        # Lets get some notable values from the equilibrium, that will be useful
        # for normalization and sanity checks later.
        # We will compute: major radius at the axis, minor radius,
        # magnetic field at the axis.
        self.Rmajor = self.equ.fluxdata.Raxis
        self.aminor = self.equ.fluxdata.aminor
        self.Baxis  = self.equ.Bdata.Baxis.values

        # Preparing the normalization.
        self._prepare_unit_system()

        self.boozer = self.equ.compute_coordinates(coordinate_system="boozer",
                                                   lpsi=npsi, ltheta=ntheta, 
                                                   rhopol_min=a, rhopol_max=b)
        self._eval_boozer_profiles()

        # We prepare some pre-calculated quantities.
        self._prepare_quantities()

        # We will establish here a simple af cache.
        self._cache: dict[str, Profile] = {}

        # Saving the maximum poloidal mode number for the profiles, which will be useful later.
        self.m_max = abs(m_max)
        self.n_max = 0 # Current limitations: tokamaks.

    def _eval_boozer_profiles(self):
        r"""
        It gets and regularizes some important profiles from the Boozer coordinates that 
        are commonly used across the problem.

        I = \int B \cdot dl, the toroidal current function (divided by 2\pi).
        g = RB_Tor
        q = \frac{1}{2\pi} \int \frac{B \cdot dl}{RB_Tor} (Global safety factor)
        h = gq + I --> This is the factor to make the toroidal angle straight in the Boozer coordinates.
        """
         # Evaluate the function h(\Psi):
        psiN = (self.boozer.deriv.h.psi0 - self.boozer.deriv.h.psi0.min()) / \
               (self.boozer.deriv.h.psi0.max() - self.boozer.deriv.h.psi0.min())
        psi0min = self.boozer.deriv.h.psi0.min().item()
        psi0max = self.boozer.deriv.h.psi0.max().item()
        psiNorm = (psi0max - psi0min)
        psigrid = self.density.x * psiNorm + psi0min

        h = self.boozer.deriv.h.squeeze()
        q = self.boozer.deriv.q.squeeze()
        F = self.boozer.deriv.F.squeeze()
        I = self.boozer.deriv.I.squeeze()

        # Let's check if the innermost point show a divergence.
        if np.abs(h[0]) < 1e-6:
            # If so, we set it to the value of the second point. This is a common issue in the calculation of the magnetic coordinates and it is not physical.
            h.values[0] = h.values[1]
            q.values[0] = q.values[1]
            F.values[0] = F.values[1]
            I.values[0] = I.values[1]
        if np.abs(h[-1]) < 1e-6:
            h.values[-1] = h.values[-2]
            q.values[-1] = q.values[-2]
            F.values[-1] = F.values[-2]
            I.values[-1] = I.values[-2]

        h = np.abs(h.interp(psi0=psigrid, method='cubic').values)
        q = np.abs(q.interp(psi0=psigrid, method='cubic').values)
        F = np.abs(F.interp(psi0=psigrid, method='cubic').values)
        I = np.abs(I.interp(psi0=psigrid, method='cubic').values)


        self.hprof = Profile(F_nm=h, x=self.density.x, attrs={'units': 'T*m', 'name': 'h'})
        self.qprof = Profile(F_nm=q, x=self.density.x, attrs={'units': '1', 'name': 'q'})
        self.Fprof = Profile(F_nm=F, x=self.density.x, attrs={'units': 'T*m', 'name': 'F'})
        self.Iprof = Profile(F_nm=I, x=self.density.x, attrs={'units': 'T*m', 'name': 'I'})

    def _prepare_quantities(self):
        r"""
        It will prepare and evaluate quantities that are commonly used across the problem,
        and cache them.

        This will prepare:
            - All contravariant metric coefficients (g^{ij}).
            - Un-normalized diamagnetic vector: \hat{s} = \nabla \Psi \times \mathbf{B}.
            - Normalized diamagnetic vector: \hat{s}_N = \hat{s} / g^\psi\psi
            - Magnetic field components.
            - Curvature: \kappa = \hat{b} \cdot \nabla \hat{b}.
            - Current and parallel current.
            - Local shear, \mathcal{S} = \hat{s}_N\cdot\nabla\times\hat{s}_N.
            - Other variables like \nabla\cdot(\nabla\Psi/g^\psi\psi)

        This will prepare all the variables in real space (R, z) from the equilibrium and 
        in the Boozer magnetic coordinates (Psi, Theta), so they'll be cached for later.

        @TODO: we are doing this in a very naive way, by evaluating the quantities in the (R, z) 
        this will be problematic when the equilibrium are coarse or close to O or X-points,
        where the magnetic coordinates can be fluky. We should consider getting around some 
        of these calculations using a re-solver of the equilibrium in magnetic coordinates.
        """
        self.realspace = xr.Dataset() # Will store the database.

        units_i = ['Wb', 'rad', 'rad']

        for ii in ['psi', 'theta', 'zeta']:
            for jj in ['psi', 'theta', 'zeta']:
                tmp = self.boozer.metric(ii, jj, 
                                            return_psi_norm=True
                ).interp(R=self.equ.B.R.values, z=self.equ.B.z.values, method='linear')
                self.realspace[f'g{ii}_{jj}'] = tmp

                # Computing the units:
                unit_ii = units_i[['psi', 'theta', 'zeta'].index(ii)]
                unit_jj = units_i[['psi', 'theta', 'zeta'].index(jj)]
                unit_str = str(unyt.Unit(f"{unit_ii}*{unit_jj}/m**2").simplify())
                self.realspace[f'g{ii}_{jj}'].attrs['units'] = unit_str

        # 2. Curvature vector in real space.
        curvature = self.equ.make_curvature()
        curvb_r = curvature.kappa_R
        curvb_z = curvature.kappa_z
        curvb_phi = curvature.kappa_phi

        # From the magnetic coordinates, we have the derivatives:
        # dPsi/dR, dPsi/dZ, dPsi/dphi, so we can compute the vector 
        # \nabla \Psi = (dPsi/dR, dPsi/dZ, dPsi/dphi).
        dPsi_dr = self.boozer.deriv.dPsi_dr.interp(R=self.equ.B.R, z=self.equ.B.z, method='cubic')
        dPsi_dphi = self.boozer.deriv.dPsi_dphi.interp(R=self.equ.B.R, z=self.equ.B.z, method='cubic')
        dPsi_dz = self.boozer.deriv.dPsi_dz.interp(R=self.equ.B.R, z=self.equ.B.z, method='cubic')
        radius = self.equ.B.R.values[:, None]
        nabla_Psi = np.array([dPsi_dr.values, dPsi_dphi.values/radius, dPsi_dz.values])
        B0 = np.array([self.equ.Br.values, self.equ.Bphi.values, self.equ.Bz.values])
        B0_norm = np.sqrt(B0[0]**2 + B0[1]**2 + B0[2]**2)

        # We compute the diamagnetic vector. In the derivation, we heavily
        # relied on the 1/||\nabla\Psi||^2 factor, but computationally along
        # a valley of no Bpol, this becomes a complication.
        # Since typically, these terms will later cancel with 
        # the ones in the equilibrium, it is safer to avoid the division
        # altogether.
        s = np.cross(B0, nabla_Psi, axis=0)

        # Projecting the curvature over the perpendicular directions.
        kappa_Psi = curvb_r * dPsi_dr.values + curvb_phi * dPsi_dphi.values / radius + curvb_z * dPsi_dz.values
        kappa_s = curvb_r * s[0] + curvb_phi * s[1] + curvb_z * s[2]

        self.realspace['kappa_Psi'] = xr.DataArray(kappa_Psi, 
                                                    coords={'R': self.equ.B.R, 
                                                            'z': self.equ.B.z},
                                                    dims=['R', 'z'],
                                                    attrs={'description': 'Curvature vector projected along the gradient of the poloidal flux.',
                                                           'units': 'Wb/m**2'})

        self.realspace['kappa_s'] = xr.DataArray(kappa_s, 
                                                coords={'R': self.equ.B.R, 
                                                        'z': self.equ.B.z},
                                                dims=['R', 'z'],
                                                attrs={'description': 'Curvature vector projected along the binormal direction.',
                                                       'units': 'T**2'})
        jac = self.boozer.deriv.jacobian.interp(R=self.equ.B.R,
                                                   z=self.equ.B.z,
                                                   method='linear')
        self.realspace['jac'] = xr.DataArray(
            np.abs(jac.values),
            coords={'R': self.equ.B.R, 'z': self.equ.B.z},
            dims=['R', 'z'],
            attrs={'description': 'Coordinate transformation Jacobian.',
                   'units': 'm**3/Wb'}
        )
        self.realspace['B0'] = xr.DataArray(B0_norm,
                                            coords={'R': self.equ.B.R, 
                                                    'z': self.equ.B.z},
                                            dims=['R', 'z'],
                                            attrs={'description': 'Norm of the equilibrium magnetic field.',
                                                   'units': 'T'})
        
        # We will also add quantities related to the current density.
        Jvec = np.array((self.equ.Jdata.Jr.values,
                         self.equ.Jdata.Jphi.values,
                         self.equ.Jdata.Jz.values))
        
        JB = np.sum(Jvec*B0, axis=0) # Parallel component of the current
        Jdiag = np.sum(Jvec*s, axis=0) # Diamagnetic component of the current
        self.realspace['J_B'] = xr.DataArray(JB, dims = ('R', 'z'),
                                             attrs={'description': r'$\vec{J}\cdot\vec{B}$',
                                                    'units': 'A*T/m**2'})
        self.realspace['J_diag'] = xr.DataArray(Jdiag, dims = ('R', 'z'),
                                                attrs={'description': r'$\vec{J}\cdot\hat{s}$',
                                                       'units': 'A*T**2/m'})

        # Computing the magnetic shear.
        # The shear appearing in the equations is the following:
        # S = s · ∇ × (s/gpsi_psi) / gpsipsi
        R = self.realspace[f'gpsi_psi'].R.values
        z = self.realspace[f'gpsi_psi'].z.values
        gpsipsi = self.realspace[f'gpsi_psi'].values
        s_hat = s / gpsipsi[None, ...]
        d_dr = Diff(axis=0, acc=4)
        d_dz = Diff(axis=1, acc=4)
        d_dr.set_grid({0: R[1] - R[0]})
        d_dz.set_grid({1: z[1] - z[0]})
        shear = _axisymmetric_cylindrical_local_shear(
            s_hat,
            R,
            d_dr,
            d_dz,
        )
        self.realspace['shear'] = xr.DataArray(shear, 
                                                dims=('R', 'z'),
                                                attrs={'description': 'Local magnetic shear.',
                                                       'units': '1/m**3'})

        # We also compute the following quantity:
        # gpsipsi * div(\nabla\Psi/gpsipsi)
        gradpsi_norm = nabla_Psi / gpsipsi[None, ...]
        divPsiVec = d_dr(gradpsi_norm[0]) + d_dz(gradpsi_norm[2]) + gradpsi_norm[0]/R[:, None]
        self.realspace['divPsiVec'] = xr.DataArray(divPsiVec, 
                                                    dims=('R', 'z'),
                                                    attrs={'description': r'div(\nabla\Psi/||\nabla\Psi||^2).',
                                                           'units': '1/m'})

        # We can now transform all the (R, z) quantities to the (Psi, Theta) coordinates.
        self.magspace = self.boozer.cyl2mag_scalar(field=self.realspace, return_psi_norm=True)

        # We will add to the magnetic space the h, pressure, density, q, I and the flows.
        prof = [self.hprof, self.qprof, self.Fprof, self.Iprof,
                self.density, self.pressure,
                self.vtor, self.vpol, self.eta]
        names = ['h', 'q', 'F', 'I', 'density', 'pressure', 'vtor', 'vpol', 'eta']
        desc = ['h profile', 'q profile', 'F profile', 'I profile', 
                'Density profile', 'Pressure profile', 
                'Toroidal flow velocity profile', 
                'Poloidal flow velocity profile', 
                'Resistivity profile']
        units = ['T*m', '1', 'T*m', 'T*m', 'm**-3', 'Pa', 'm/s', 'm/s', 'ohm*m']
        
        # We interpolate all profiles to the same grid as the magnetic space.
        for iprof in prof:
            iprof.data = iprof.data.interp(x=self.magspace.psi.values, method='cubic')

        for iname, iprof, idesc in zip(names, prof, desc):
            tmp = iprof.data.interp(x=self.magspace.psi.values,
                                    method='cubic').values.squeeze()
            self.magspace[iname] = xr.DataArray(tmp.real.astype(np.float64),
                                                dims = ['psi'],
                                                attrs = {'description': idesc,
                                                         'units': units[names.index(iname)]})
        self.magspace['g'] = self.magspace['F'].copy()
        self.magspace['g'].attrs.update({'description': 'g profile', 'units': 'T*m'})

    def _prepare_unit_system(self):
        """
        This will prepare the system of units for the normalization of the problem.

        The normalization table that we will follow is.
        | Quantity | Symbol | Normalization |
        |------------------------|------------------|----------------------------|
        | Length                 | L                | Minor radius               |
        | Magnetic field         | B0               | Magnetic field at the axis |
        | Density                | n0               | Density at the axis        |
        | Pressure               | 1/(Baxis^2/mu_0) | On-axis magnetic pressure  |
        """
        self.reg_name = f"nova_{id(self)}"
        n0 = self.density.data.real.interp(x=0.0).values.squeeze()
        self.my_norm, self.registry = create_nova_registry(self.reg_name,
                                                           self.aminor * unyt.m, 
                                                           self.Baxis * unyt.T, 
                                                           n0 * unyt.m**(-3))

    # --------------------
    # Handling new additions to the real space and magnetic space databases.
    # --------------------
    def add_real(self, name: str, data: xr.DataArray, units: str,
                 description: Optional[str] = None):
        """
        Adds a new quantity to the real space database.

        Parameters
        ----------
        name : str
            The name of the quantity to be added.
        data : xr.DataArray
            The data array containing the values of the quantity. It should have dimensions (R, z).
        description : Optional[str], optional
            A description of the quantity being added.
        """
        if not isinstance(data, xr.DataArray):
            raise ValueError("Data should be an xarray DataArray.")
        if 'R' not in data.dims or 'z' not in data.dims:
            raise ValueError("Data should have dimensions (R, z).")
        
        self.realspace[name] = xr.DataArray(data.values,
                                            coords={'R': self.equ.B.R, 'z': self.equ.B.z},
                                            dims=['R', 'z'],
                                            attrs={'description': description if description is not None else '',
                                                   'units': units})
        
        # Transforming the new quantity to magnetic coordinates and adding it to the magspace database.
        mag_data = self.magspace.cyl2mag_scalar(field=self.realspace[name], return_psi_norm=True)
        self.magspace[name] = mag_data
    
    def add_mag(self, name: str, data: xr.DataArray, units: str, description: Optional[str] = None):
        """
        Adds a new quantity to the magnetic space database.

        WARNING: this will not be added to the real space database.

        Parameters
        ----------
        name : str
            The name of the quantity to be added.
        data : xr.DataArray
            The data array containing the values of the quantity. It should have dimensions (Psi, Theta).
        description : Optional[str], optional
            A description of the quantity being added.
        """
        if not isinstance(data, xr.DataArray):
            raise ValueError("Data should be an xarray DataArray.")
        if 'Psi' not in data.dims or 'Theta' not in data.dims:
            raise ValueError("Data should have dimensions (Psi, Theta).")
        
        self.magspace[name] = xr.DataArray(data.values,
                                            coords={'Psi': self.magspace.Psi, 'Theta': self.magspace.Theta},
                                            dims=['Psi', 'Theta'],
                                            attrs={'description': description if description is not None else '',
                                                   'units': units})
        logger.debug("Added {name} in the magnetic database, not in the real space database.")

    # --------------------
    # Internal equilibrium interpreter.
    # ---------------------
    def _resolve_symbol(self, name: str, expression: str, normalize: bool=False):
        """
        Locates and returns the object.

        We will look in the following way:
        1. Stuff on the magnetic space database.
        2. Stuff on the real space database that needs to be transformed to magnetic coordinates.
        3. Stuff in valid operations.
        """
        if name in self.magspace:
            out = _unytfy_my_xarray(self.magspace[name], reg=self.registry)
        elif name in self.realspace:
            # We need to transform it to magnetic coordinates.
            mag_data = self.magspace.cyl2mag_scalar(field=self.realspace[name], return_psi_norm=True)
            self.magspace[name] = mag_data
            out =  _unytfy_my_xarray(mag_data, reg=self.registry)
        elif name in _ALLOWED_EXPRESSION_CONSTANTS:
            out =  _ALLOWED_EXPRESSION_CONSTANTS[name]
        else:
            available = ", ".join(list(self._cache.keys()) + list(self.magspace.data_vars) + list(self.realspace.data_vars) + list(_ALLOWED_EXPRESSION_CONSTANTS.keys()))
            raise ValueError(f"Symbol {name} not found in the cache, magnetic space, or real space databases. "+
                             f"Available symbols are: {available}. Expression was: {expression}.")

        unit_system = self.my_norm if normalize else "mks"
        out = _in_unit_system(out, unit_system, reg=self.registry)
        if normalize:
            out = _to_normalized_expression_units(name, out, self.registry)
        return out
    
    def _evaluate_ast_node(self, node: ast.AST, expression: str, normalize: bool=False):
        if isinstance(node, ast.BinOp):
            left = self._evaluate_ast_node(node.left, expression, normalize)
            right = self._evaluate_ast_node(node.right, expression, normalize)

            if isinstance(node.op, ast.Add):
                return _expression_binary_op(left, right, operator.add)
            if isinstance(node.op, ast.Sub):
                return _expression_binary_op(left, right, operator.sub)
            if isinstance(node.op, ast.Mult):
                return _expression_binary_op(left, right, operator.mul)
            if isinstance(node.op, ast.Div):
                return _expression_binary_op(left, right, operator.truediv)
            if isinstance(node.op, ast.Pow):
                return _expression_binary_op(left, right, operator.pow)

            raise ValueError(
                f"Operator '{type(node.op).__name__}' is not allowed in expression '{expression}'."
            )

        if isinstance(node, ast.UnaryOp):
            operand = self._evaluate_ast_node(node.operand, expression, normalize)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError(
                f"Unary operator '{type(node.op).__name__}' is not allowed in expression '{expression}'."
            )

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError(
                    f"Only direct function calls are allowed in expression '{expression}'."
                )
            if node.keywords:
                raise ValueError(
                    f"Keyword arguments are not allowed in expression '{expression}'."
                )

            func_name = node.func.id
            if func_name not in _ALLOWED_EXPRESSION_FUNCTIONS:
                raise ValueError(
                    f"Function '{func_name}' is not allowed in expression '{expression}'."
                )
            func = _ALLOWED_EXPRESSION_FUNCTIONS[func_name]
            args = [self._evaluate_ast_node(arg, expression, normalize) for arg in node.args]
            try:
                return func(*args)
            except Exception as exc:
                raise ValueError(
                    f"Failed evaluating function '{func_name}' in expression '{expression}': {exc}"
                ) from exc

        if isinstance(node, ast.Name):
            return self._resolve_symbol(node.id, expression, normalize)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, complex)):
                return node.value
            raise ValueError(
                f"Constant '{node.value}' is not allowed in expression '{expression}'."
            )

        if isinstance(node, ast.Attribute):
            raise ValueError(
                f"Attribute access is not allowed in expression '{expression}'."
            )
        if isinstance(node, ast.Subscript):
            raise ValueError(f"Indexing is not allowed in expression '{expression}'.")

        raise ValueError(
            f"Unsupported syntax in expression '{expression}': "
            f"{ast.dump(node, include_attributes=False)}"
        )
    
    def _evaluate_expression(self, expression: str, normalize: bool=False) -> xr.DataArray:
        expression = expression.strip()
        if not expression:
            raise ValueError("Empty expressions are not allowed.")

        try:
            parsed = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(
                f"Invalid expression '{expression}': {exc.msg}."
            ) from exc

        result = self._evaluate_ast_node(parsed.body, expression, normalize)
        if isinstance(result, xr.DataArray):
            return _sync_xarray_units_attr(result)

        if not self.magspace.data_vars:
            raise ValueError(
                "No base requirements available to broadcast scalar expression results."
            )
        template_name = next(iter(self.magspace.data_vars))
        template = self.magspace[template_name]
        if _has_unyt_units(result):
            scalar_field = xr.full_like(template, fill_value=result.value)
            scalar_field.attrs["units"] = str(result.units)
        else:
            scalar_field = xr.full_like(template, fill_value=result)
            scalar_field.attrs["units"] = "dimensionless"
        return scalar_field

    def _profile_in_unit_system(self, prof: Profile, normalize: bool=False) -> Profile:
        """
        Return a copy of a Profile converted to the requested unit system.
        """
        out = prof.copy()
        unit_system = self.my_norm if normalize else "mks"
        out.data = _deunytfy_my_xarray(
            _in_unit_system(out.data, unit_system, reg=self.registry),
            reg=self.registry,
        )
        return out

    def __call__(self, *args, which: Optional[str] = 'magnetic', normalize: bool=False, **kwargs) -> List[Profile]:
        """
        This will allow to call the class to get the equilibrium quantities in a more direct way.

        For example, we can do:
        eq('kappa_s', which='magnetic') to get the curvature in magnetic coordinates.
        eq('kappa_s', which='profile') to get the curvature as a profile in the radial coordinate
        with a decomposition in (n, m) Fourier modes.

        Parameters
        ----------
        args : str
            The names of the quantities to be retrieved, or expressions to be evaluated.
        which : str, optional
            The space in which the quantities should be retrieved. Allowed values are 'magnetic' and
            'profile'. The default is 'magnetic'.
        normalize : bool, optional
            Whether to normalize the quantities using the normalization table defined in the class.
            The default is False
        **kwargs
            Additional keyword arguments for future extensions.

        Returns
        -------
        List[Profile]
            A list of profiles corresponding to the requested quantities.
            If only one quantity is requested, a single Profile is returned
            instead of a list.
        """

        if len(args) < 1:
            raise ValueError("At least one argument is required, which is the name of the quantity to be retrieved.")
        if which.lower() not in ['magnetic', 'profile']:
            raise ValueError("Invalid value for 'which'. Allowed values are: 'magnetic', 'profile', " \
                            f"got instead {which}.")

        normalized_args: list[str] = []
        for arg in args:
            if not isinstance(arg, str):
                raise TypeError(
                    "All positional arguments must be strings naming requirements "
                    "or expressions."
                )
            normalized = arg.strip()
            if not normalized:
                raise ValueError("Empty requirement names or expressions are not allowed.")
            normalized_args.append(normalized)

        # Let's try to get all the possible outputs from the cache, magnetic space, or real space databases.
        outputs = np.empty(len(normalized_args), dtype=Profile)
        real2mag: List[Tuple[int, Any]] = list() # Things we need to transform from real space to magnetic space.
        mag2prof: List[Tuple[int, Any]] = list() # Things we need to transform from magnetic space to profiles.
        to_evaluate: List[Tuple[int, Any]] = list() # Things we need to evaluate from expressions.
        for ii, arg in enumerate(normalized_args):
            # if arg in self._cache:
            #     outputs[ii] = self._profile_in_unit_system(self._cache[arg], normalize=normalize)
                
            # el
            if arg in self.magspace:
                mag2prof.append((ii, arg))
            elif arg in self.realspace:
                real2mag.append((ii, arg))
            else:
                to_evaluate.append((ii, arg))
        
        # First thing, we transform from the real space to the magnetic space.
        for ii, name in real2mag:
            mag_data = self.magspace.cyl2mag_scalar(field=self.realspace[name], return_psi_norm=True)
            self.magspace[name] = mag_data # We cache the magnetic coordinates version of the quantity for later.
            mag2prof.append((ii, name))

        # We need evaluate the expressions now to get the (Psi, Theta) fields of the operations.
        for ii, arg in to_evaluate:
            mag_data = self._evaluate_expression(arg, normalize=normalize)
            name = f"_expr_{arg}"
            self.magspace[name] = mag_data
            mag2prof.append((ii, name))

        # We transform from the magnetic space to profiles.
        # To make a bulk transformation we build a xr.Dataset.
        if mag2prof:
            # We transform the xarray.Dataset into a non-quantity dataset set.
            unit_system = self.my_norm if normalize else "mks"
            mag_dataset = xr.Dataset({
                name: _deunytfy_my_xarray(
                    _in_unit_system(self.magspace[name], unit_system, reg=self.registry),
                    reg=self.registry,
                )
                for _, name in mag2prof
            })

            # mag_dataset = xr.Dataset({name: self.magspace[name] for _, name in mag2prof})
            prof_data = ProfileFactoryPsiThetaZeta(mag_dataset, ntormax=0,
                                                   mpolmax=self.m_max)
        for idx, (ii, name) in enumerate(mag2prof):
            prof_data[idx].data.attrs.update(mag_dataset[name].attrs)
            outputs[ii] = prof_data[idx]

        # We add the stuff to the outputs and cache the profiles.
        for idx, (ii, name) in enumerate(mag2prof):
            if not name.startswith("_expr_"):
                self._cache[name] = self._profile_in_unit_system(prof_data[idx], normalize=False)

        return outputs if len(outputs) > 1 else outputs[0]

    def normalize(self, *args):
        """
        Transforms the requested inputs into the NOVA normalized unit system.

        Parameters
        ----------
        args : List[Union[xr.DataArray, unyt.unyt_array]]
            The quantities to be normalized: either a unyt.unyt_array
             or an xarray.DataArray with unyt units or units in the
             attributes.
        """
        # Checking all elements are valid.
        if len(args) < 1:
            raise ValueError("At least one argument is required, which is the quantity to be normalized.")
        
        for iargs in args:
            if not isinstance(iargs, (xr.DataArray, unyt.unyt_array)):
                raise TypeError(
                    "All arguments must be either xarray.DataArray or unyt.unyt_array."
                )
            
            # In the case of xarray.DataArray, we use the routines to enforce the 
            # units in the xarray.
            iargs = _unytfy_my_xarray(iargs, reg=self.registry)
        
        # We normalize all the arguments and return them.
        normalized_args = []
        for iargs in args:
            if isinstance(iargs, xr.DataArray):
                normalized = _in_unit_system(iargs, self.my_norm, reg=self.registry)
                normalized = _to_normalized_expression_units(iargs.name, normalized, self.registry)
                normalized_args.append(normalized)
            else:
                normalized = _in_unit_system(iargs, self.my_norm, reg=self.registry)
                normalized = _to_normalized_expression_units("", normalized, self.registry)
                normalized_args.append(normalized)

        # We detach the units if the input was an xarray, to avoid issues with the profiles.
        normalized_args = [
            _deunytfy_my_xarray(arg, reg=self.registry) if isinstance(arg, xr.DataArray) else arg
            for arg in normalized_args
        ]
        return normalized_args if len(normalized_args) > 1 else normalized_args[0]
    
    def to_mks(self, *args):
        """
        Transforms the requested inputs into the MKS unit system.

        Parameters
        ----------
        args : List[Union[xr.DataArray, unyt.unyt_array]]
            The quantities to be transformed: either a unyt.unyt_array
             or an xarray.DataArray with unyt units or units in the
             attributes.
        """
        # Checking all elements are valid.
        if len(args) < 1:
            raise ValueError("At least one argument is required, which is the quantity to be transformed.")
        
        for iargs in args:
            if not isinstance(iargs, (xr.DataArray, unyt.unyt_array)):
                raise TypeError(
                    "All arguments must be either xarray.DataArray or unyt.unyt_array."
                )
            
            # In the case of xarray.DataArray, we use the routines to enforce the 
            # units in the xarray.
            if isinstance(iargs, xr.DataArray):
                iargs = _unytfy_my_xarray(iargs, reg=self.registry)
        
        # We transform all the arguments and return them.
        transformed_args = []
        for iargs in args:
            if isinstance(iargs, xr.DataArray):
                transformed = _in_unit_system(iargs, "mks", reg=self.registry)
                transformed_args.append(transformed)
            else:
                transformed = _in_unit_system(iargs, "mks", reg=self.registry)
                transformed_args.append(transformed)
        # We detach the units if the input was an xarray, to avoid issues with the profiles.
        transformed_args = [
            _deunytfy_my_xarray(arg, reg=self.registry) if isinstance(arg, xr.DataArray) else arg
            for arg in transformed_args
        ]
        return transformed_args if len(transformed_args) > 1 else transformed_args[0]
