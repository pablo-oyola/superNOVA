r"""One solver for the ideal-MHD stability problem.

This was once four classes in a linear inheritance chain, each level bolting on
a different eigensolver strategy.  Only one strategy is used, so there is one
class.  Nothing in the chain called ``super()``, so this is the same code with
the indirection removed.

The pipeline, top to bottom:

1. resample the equilibrium and continuum profiles onto the quadrature grid,
2. build the point-local blocks for a q-selected band of poloidal harmonics,
3. split them into ``M0``, ``M1`` and ``Mminus1``,
4. project onto the compatible mixed B-spline spaces,
5. reduce on the trial side with the boundary conditions,
6. hand the result to the PEP, or to the GEVP at exact ``gamma_s == 0``.

Steps 1-2 are cached on the instance; a frequency scan repeats only 6.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import unyt
import xarray as xr
from scipy.sparse import csr_matrix

from supernova.equilibrium import superEquilibrium, _active_poloidal_flux_span
from supernova.continuum import superContinuum
from supernova.radial.base import RadialFE
from supernova.radial.profile import Profile
from supernova.radial import BSplineFE

from supernova.operators.blocks import SolverBlocks, build_solver_blocks
from supernova.operators.boundary import (
    MixedBoundaryCondition as BoundaryCondition,
    idealSeparatrixBC,
    regularityAxisBC,
)
from supernova.operators.local import (
    _build_local_M_split,
    _matrix_stats,
    _slice_to_quadrature,
)
from supernova.operators.mixed import (
    MixedFieldLayout,
    MixedRadialData,
    _build_mixed_radial_data,
)
from supernova.operators.project import _project_local_tensor_mixed
from supernova.eigen.dispatch import build_eigensolver
from supernova.eigen.gevp import LinearGeneralizedEPSSolver
from supernova.eigen.pep import QuadraticPEPSolver
from supernova.eigen.petsc import NEPResult
from supernova.eigen.region import _as_scalar_complex, _ellipse_region_dict


@dataclass(frozen=True)
class WindowSolution:
    """One window's eigenpairs as plain arrays.

    Exactly what :meth:`StabilitySolver.pack_window` needs from an eigensolver,
    and nothing that holds PETSc state, so a window can be solved in another
    process -- one that never saw the solver -- and packed where it lives.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    nconv: int
    reason: int
    iterations: int
    backend: str
    eps_type: str
    residual_norms: np.ndarray | None = None
    absolute_residual_norms: np.ndarray | None = None
    backward_errors: np.ndarray | None = None
    slepc_error_estimates: np.ndarray | None = None
    slepc_backward_errors: np.ndarray | None = None

    @classmethod
    def from_solver(cls, eigensolver, result) -> "WindowSolution":
        return cls(
            eigenvalues=np.asarray(result.eigenvalues),
            eigenvectors=np.asarray(result.eigenvectors),
            nconv=int(result.nconv),
            reason=int(result.reason),
            iterations=int(result.iterations),
            backend=getattr(eigensolver, "backend", "PEP"),
            eps_type=getattr(eigensolver, "eps_type", "KRYLOVSCHUR"),
            residual_norms=getattr(eigensolver, "last_residual_norms", None),
            absolute_residual_norms=getattr(
                eigensolver, "last_absolute_residual_norms", None
            ),
            backward_errors=getattr(eigensolver, "last_backward_errors", None),
            slepc_error_estimates=getattr(
                eigensolver, "last_slepc_error_estimates", None
            ),
            slepc_backward_errors=getattr(
                eigensolver, "last_slepc_backward_errors", None
            ),
        )


class StabilitySolver:
    """Assemble and solve the ideal-MHD pencil on one radial grid."""

    
    def __init__(self, equ: superEquilibrium, 
                 cont: superContinuum,
                 radialfe: RadialFE):
        """
        Initialize the solver with the equilibrium, continuum and radial FE basis.

        Parameters
        ----------
        equ : Equilibrium
            The equilibrium object containing the plasma equilibrium data.
        cont : superContinuum
            The continuum object containing the assembled continuum matrices.
        radialfe : RadialFE
            The radial finite element basis used for the discretization of the problem.
        """
        self.equ: superEquilibrium = equ
        self.cont: superContinuum = cont
        self.radialfe: RadialFE = radialfe

        # From the continuum object, we read the gamma_s
        self.gamma_s = cont.gamma_s

        # We build the the relevant profiles at initialization.
        self._build_at_init()

        # We will set up the points where the radial finite elements basis
        # will be need to be evaluated to perform the integrals.
        self._setup_radial_points()

    def _build_at_init(self):
        r"""
        Sub-routine to generate the relevant matrices and pieces required to later
        assemble and solve the full MHD problem.

        This will build the following objects:

              (C11  C12)
        C   = (0    C22), 

        where C11 = 2 * K_\Psi, 
        C22 = - div (\nabla\Psi / g^{\psi\psi})
        C12 = \lambda + 2\mu_0p_0'K_\Psi - sqrt(g) * g^{\psi\psi} G^H O G + \
              (J · B - Sg^{\psi\psi}) * S*g^{\psi\psi}/B^2,
        where O = 1/(sqrt(g)||\nabla\Psi||^2) and G = i(nq - m)\delta_{m,m'}\delta_{n,n'} and
        \lambda is the eigenvalue: \lambda = \omega^2 * \rho_0 * \mu_0.


        D =  (D11  D12)
             (D21  D22)

        where D11 = - (J · B - Sg^{\psi\psi}) * g^{\psi\psi}/(sqrt(g)*B^2) * G
              D12 = 2 \gamma_s * p_0 * \mu_0 K_\Psi
              D21 = g^{\psi\psi} * (2K_s - \sigma/(sqrt(g)*B^2))
              D22 = g^{\psi\psi} * (1 - \mu_0 p_0 \gamma_s / \lambda G^H M G),
              where
              \sigma = -i(gm + In)\delta_{m,m'}\delta_{n,n'}, and
              M = 1/(sqrt(g)B_0^2).

        F = (F11  F12)
            (F21  F22)

            where F11 = - K + \sigma/B^2
                  F12 = G(g^{\psi\psi}/B^2 * S) - (J · B) / B^2 * G - \mu_0 p_0' K
                  F21 = - sqrt(g)/B^2
                  F22 = - 2\kappa_\psi*sqrt(g)/g^{\psi\psi}.

        The eigenvalue solver will later solver for the following problem:
        \nabla\Psi·\nabla (p_1, \xi_\psi) = (C + DE^-1F) (p_1, \xi_\psi)

        and \nabla\Psi·\nabla = g^{\psi\psi}\partial_\Psi
        + g^{\psi\Theta}\partial_\Theta + g^{\psi\zeta}\partial_\zeta.

        Since the objective is to separate the radial from the
        radially-local operators to apply Krylov-based solvers, we need to move the
        g^{\Psi\Psi} = ||\nabla\Psi||^2 from the LHS to the RHS. After recalculation:

        C11 = 2*kappa_\Psi/g^{\psi\psi}
        C12 = \lambda/g^{\psi\psi} + \mu_0 * p_0' * C11 - 1/sqrt(g) G^H O G + \
              (J · B - Sg^{\psi\psi}) * S/ B^2
        C22 = - \nabla·(\nabla\Psi/g^{\psi\psi}) --> Precomputed as divPsiVec

        D11 = - (J · B - Sg^{\psi\psi}) /(sqrt(g)*B^2) * G
        D12 = gamma_s * p_0 * mu_0 * C11
        D21 = 2*kappa_s - \sigma/(sqrt(g)*B^2)
        D22 = 1 - \mu_0 p_0 \gamma_s / \lambda G^H M G

        For the stored basis ``exp(i*(n*zeta-m*Theta))``, the final problem is:
        \partial_\Psi (p_1, \xi_\psi) =
        (C + DE^-1F + i m g^{\psi\Theta}/g^{\psi\psi}
        - i n g^{\psi\zeta}/g^{\psi\psi}) (p_1, \xi_\psi)

        We will change the Jacobian, assuming the coordinates are the
        Boozer coordinates by using

        sqrt(g) = Jacobian = h / B0**2

        """

        _to_compute = ['kappa_Psi/gpsi_psi',             # C11, also for D12                      - 0
                       'B0**2/h',   # C12 - second part                                           - 1
                       '(J_B - shear*gpsi_psi) * shear / B0**2',  # C12 - third part              - 2
                       'B0**2/(gpsi_psi*h)',    # The O operator                                    - 3
                       '- divPsiVec', # C22 - Precomputed.                                        - 4
                       '(shear*gpsi_psi - J_B)/h',  # D11                                         - 5
                       'kappa_s / B0**2',  # D21 - first part; distinct from the continuum K operator.    - 6
                       '1/B0**2', # F11 second part.                                              - 7
                       'gpsi_psi/B0**2 * shear', # F12 - first part                               - 8
                       '- J_B/B0**2', # F12 - second part                                         - 9
                       '-h/B0**4', # F21                                                        - 10
                       '-2*kappa_Psi/(gpsi_psi*B0**2)*h', # F22                                     - 11
                       '1/gpsi_psi', # Harmonic operator needed by lambda/gpsi_psi.               - 12
                       'gpsi_theta/gpsi_psi', # Surface phase term for e^{i(n*zeta-m*theta)}.     - 13
                       'gpsi_zeta/gpsi_psi', # Surface phase term for e^{i(n*zeta-m*theta)}.      - 14
                       ]
        
        # We only build the elements at the collocation nodes
        # and using the user-provided FE basis.
        tmp = self.equ(*_to_compute, normalize=True)

        # tmp = list()
        # for ii in _to_compute:
        #     try:
        #         tmp.append(self.equ(ii, normalize=True))
        #     except:
        #         breakpoint()

        self.C11 = tmp[0]
        self.C12_2 = tmp[1]
        self.C12_3 = tmp[2]
        self.O = tmp[3]
        self.C22 = tmp[4]
        self.D11 = tmp[5]
        self.D21_1 = tmp[6]
        self.F11_2 = tmp[7]
        self.F12_1 = tmp[8]
        self.F12_2 = tmp[9]
        self.F21 = tmp[10]
        self.F22 = tmp[11]
        self.inv_gpsi_psi = tmp[12]
        self.gpsi_theta_over_gpsi_psi = tmp[13]
        self.gpsi_zeta_over_gpsi_psi = tmp[14]

        # The rest of the operators come from the continuum.  Solvers may use
        # different radial quadrature grids while sharing one continuum, so
        # keep private copies before _setup_radial_points resamples them.
        self.H = self.cont.H.copy(deepcopy=True)
        self.J = self.cont.J.copy(deepcopy=True)
        self.K = self.cont.K.copy(deepcopy=True)
        self.L = self.cont.L.copy(deepcopy=True)
        self.M = self.cont.M.copy(deepcopy=True)
        self.D21_2 = self.M  # = h(\psi)

    def _setup_radial_points(self):
        """
        From a given prepared Radial finite element basis, we
        get the points where the whole problem needs to be evaluated
        to perform the integrals in Galerkin scheme.

        We will also resample the profiles to these points and
        store the integration weights for later use.
        """
        self.points, self.weights = self.radialfe.get_quadrature_points()

        # Solver quadrature points are physical radial points. The local MHD
        # operator also needs an edge value for the boundary condition, but that
        # edge point is not part of the quadrature rule.
        self.edge_points = np.asarray([self.radialfe.get_mesh()[-1]], dtype=np.float64)
        self.operator_points = np.concatenate([self.points, self.edge_points])
        self.unique_points = self.operator_points

        # Re-evaluating the profiles.
        self.C11.data = self.C11.data.interp(x=self.unique_points)
        self.C12_2.data = self.C12_2.data.interp(x=self.unique_points)
        self.C12_3.data = self.C12_3.data.interp(x=self.unique_points)
        self.O.data = self.O.data.interp(x=self.unique_points)
        self.C22.data = self.C22.data.interp(x=self.unique_points)
        self.D11.data = self.D11.data.interp(x=self.unique_points)
        self.D21_1.data = self.D21_1.data.interp(x=self.unique_points)
        self.F11_2.data = self.F11_2.data.interp(x=self.unique_points)
        self.F12_1.data = self.F12_1.data.interp(x=self.unique_points)
        self.F12_2.data = self.F12_2.data.interp(x=self.unique_points)
        self.F21.data = self.F21.data.interp(x=self.unique_points)
        self.F22.data = self.F22.data.interp(x=self.unique_points)
        self.inv_gpsi_psi.data = self.inv_gpsi_psi.data.interp(x=self.unique_points)
        self.gpsi_theta_over_gpsi_psi.data = self.gpsi_theta_over_gpsi_psi.data.interp(x=self.unique_points)
        self.gpsi_zeta_over_gpsi_psi.data = self.gpsi_zeta_over_gpsi_psi.data.interp(x=self.unique_points)

        # The rest of the operators come from the continuum:
        self.H.data = self.H.data.interp(x=self.unique_points)
        self.J.data = self.J.data.interp(x=self.unique_points)
        self.K.data = self.K.data.interp(x=self.unique_points)
        self.L.data = self.L.data.interp(x=self.unique_points)
        self.M.data = self.M.data.interp(x=self.unique_points)

        # We need to get for the radial derivative the underlying unit and scale
        # of the flux. Use the active-domain span so this derivative is
        # invariant to the arbitrary additive offset of poloidal flux.
        psiN = _active_poloidal_flux_span(
            self.equ.boozer.coords.psi0
        ) * unyt.Unit('Wb/rad', registry=self.equ.registry)
        # COCOS 1 stores poloidal flux in Wb/rad, while the NOVA magnetic-flux
        # scale is psi0 = Baxis*aminor**2. Keep the angle unit explicit until
        # the normalized value enters the numerical operator.
        self.dpsi_hat_dx = float(
            psiN.in_base(self.equ.my_norm).to("psi0/rad").v
        )

        # We need also to divide the density by the maximum density.
        n0 = (1 * unyt.Unit("naxis", registry=self.equ.registry)).to("1/aminor**3")

        # We keep too a copy of the density, pressure and the derivative.
        self.density = self.equ('density', normalize=True)
        self.pressure = self.equ('pressure', normalize=True)
        self.pressure_deriv = self.equ('pressure', normalize=True)
        self.density.data = self.density.data.interp(x=self.unique_points) / n0.v
        self.pressure.data = self.pressure.data.interp(x=self.unique_points)
        self.pressure_deriv.data = (
            self.pressure.data.differentiate('x').interp(x=self.unique_points)
            / self.dpsi_hat_dx
        )

        # We evaluate as well hprof, qprof, Iprof and gprof at the solver points for later use in the integrals.
        self.hprof = self.equ('h', normalize=True)
        self.qprof = self.equ('q', normalize=True)
        self.Iprof = self.equ('I', normalize=True)
        self.gprof = self.equ('g', normalize=True)

        self.hprof.data = self.hprof.data.interp(x=self.unique_points)
        self.qprof.data = self.qprof.data.interp(x=self.unique_points)
        self.Iprof.data = self.Iprof.data.interp(x=self.unique_points)
        self.gprof.data = self.gprof.data.interp(x=self.unique_points)

    def _solver_mmax(self, mmax: int=None) -> int:
        """
        Return the effective poloidal harmonic limit used by both the static
        solver blocks and the continuum eigensystem.
        """
        available = int(np.abs(self.H.m_pol).max())
        if mmax is None:
            return available
        if mmax < 0:
            raise ValueError("mmax must be non-negative.")
        return min(int(mmax), available)

    @staticmethod
    def _profile_harmonics(profile: Profile, mmax: int, nmax: int=0) -> np.ndarray:
        """
        Convert a Profile into the harmonic array expected by the kernels.

        Output shape is (x, mpol, ntor), and missing harmonics are explicitly
        filled with zero. The dtype is always complex for homogeneous kernel
        inputs.
        """
        data = profile.data.reindex(
            mpol=np.arange(mmax + 1, dtype=np.int32),
            ntor=np.arange(nmax + 1, dtype=np.int32),
            fill_value=0.0 + 0.0j,
        ).transpose("x", "mpol", "ntor")

        return np.ascontiguousarray(data.values, dtype=np.complex128)

    @staticmethod
    def _profile_scalar(profile: Profile) -> np.ndarray:
        """
        Convert the (m=0, n=0) component of a Profile into a radial array.

        Output shape is (x,), and the dtype is kept complex even for real-valued
        scalar profiles.
        """
        data = profile.data.reindex(
            mpol=np.array([0], dtype=np.int32),
            ntor=np.array([0], dtype=np.int32),
            fill_value=0.0 + 0.0j,
        ).transpose("x", "mpol", "ntor")

        return np.ascontiguousarray(data.values[:, 0, 0], dtype=np.complex128)

    @staticmethod
    def _validate_explicit_m_values(
        m_values: np.ndarray,
        available_mmax: int,
    ) -> np.ndarray:
        """Validate an explicit diagnostic perturbation-harmonic list.

        The production path continues to select a contiguous positive-q band.
        This helper exists for verification problems whose harmonic space must
        be closed under a specified transformation, such as ``m -> -m``.
        """
        try:
            values = np.asarray(m_values)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "m_values must be a numeric array of integer mode numbers."
            ) from error
        if values.ndim != 1 or values.size == 0:
            raise ValueError("m_values must be a nonempty one-dimensional array.")
        if np.issubdtype(values.dtype, np.bool_):
            raise ValueError("m_values must contain integer mode numbers, not booleans.")
        if np.issubdtype(values.dtype, np.integer):
            int64_info = np.iinfo(np.int64)
            if np.issubdtype(values.dtype, np.unsignedinteger):
                if np.any(values > int64_info.max):
                    raise ValueError("m_values exceed the supported int64 range.")
            elif values.dtype.itemsize > np.dtype(np.int64).itemsize:
                if np.any(values < int64_info.min) or np.any(values > int64_info.max):
                    raise ValueError("m_values exceed the supported int64 range.")
        elif np.issubdtype(values.dtype, np.floating):
            if not np.all(np.isfinite(values)) or not np.all(values == np.rint(values)):
                raise ValueError("m_values must contain finite integer mode numbers.")
            # ``float64(int64.max)`` rounds to 2**63, so compare against the
            # exactly representable half-open int64 interval rather than
            # converting ``int64.max`` to float.
            if np.any(values < -float(2**63)) or np.any(values >= float(2**63)):
                raise ValueError("m_values exceed the supported int64 range.")
        else:
            raise ValueError(
                "m_values must be a real numeric array of integer mode numbers."
            )
        try:
            values = values.astype(np.int64, copy=False)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError("m_values exceed the supported int64 range.") from error
        if np.unique(values).size != values.size:
            raise ValueError("m_values must not contain duplicate mode numbers.")
        if np.any(np.diff(values) <= 0):
            raise ValueError("m_values must be strictly increasing.")
        available_mmax = int(available_mmax)
        if available_mmax < 0:
            raise ValueError("available_mmax must be non-negative.")
        if np.any(values < -available_mmax) or np.any(values > available_mmax):
            maximum_absolute_mode = max(abs(int(value)) for value in values)
            raise ValueError(
                "m_values exceed the available equilibrium harmonic range: "
                f"max(abs(m_values))={maximum_absolute_mode}, "
                f"available={available_mmax}."
            )
        return values

    def _assemble_solver_blocks_for_m_range(
        self,
        *,
        ntor: int,
        m_range: np.ndarray,
        max_dm: int,
    ) -> SolverBlocks:
        """Assemble solver blocks for an already validated harmonic list."""
        m_range = np.asarray(m_range, dtype=np.int64)
        if int(max_dm) < 0:
            raise ValueError("max_dm must be non-negative.")
        max_dm = int(max_dm)
        perturbation_mmax = int(np.max(np.abs(m_range)))
        largest_retained_difference = min(
            int(max_dm),
            int(np.max(m_range) - np.min(m_range)),
        )
        available_mmax = int(np.abs(self.M.m_pol).max())
        coefficient_mmax = min(
            available_mmax,
            max(perturbation_mmax, largest_retained_difference),
        )
        nmax_eff = 0

        H = self._profile_harmonics
        S = self._profile_scalar
        # Solver blocks use lambda = mu_0 * rho_mass * omega**2.  The
        # equilibrium density profile is normalized number density, while
        # omegaA is referenced to 1 amu, so include the species mass in amu
        # here rather than rescaling the requested frequency.
        mass_amu = float(self.cont.mass.to("amu").v)
        mass_density = S(self.density) * mass_amu

        blocks = build_solver_blocks(
            C11=H(self.C11, coefficient_mmax, nmax_eff),
            O=H(self.O, coefficient_mmax, nmax_eff),
            C12_2=H(self.C12_2, coefficient_mmax, nmax_eff),
            C12_3=H(self.C12_3, coefficient_mmax, nmax_eff),
            C22=H(self.C22, coefficient_mmax, nmax_eff),
            D11=H(self.D11, coefficient_mmax, nmax_eff),
            D12=H(self.C11, coefficient_mmax, nmax_eff),
            kappa_s=H(self.D21_1, coefficient_mmax, nmax_eff),
            K=H(self.K, coefficient_mmax, nmax_eff),
            M=H(self.M, coefficient_mmax, nmax_eff),
            F11_2=H(self.F11_2, coefficient_mmax, nmax_eff),
            F12_1=H(self.F12_1, coefficient_mmax, nmax_eff),
            F12_2=H(self.F12_2, coefficient_mmax, nmax_eff),
            F21=H(self.F21, coefficient_mmax, nmax_eff),
            F22=H(self.F22, coefficient_mmax, nmax_eff),
            hprof=S(self.hprof),
            Iprof=S(self.Iprof),
            gprof=S(self.gprof),
            qprof=S(self.qprof),
            pressure=S(self.pressure),
            pressure_deriv=S(self.pressure_deriv),
            density=mass_density,
            gamma_s=self.gamma_s,
            dpsi_hat_dx=self.dpsi_hat_dx,
            one_gpp=H(self.inv_gpsi_psi, coefficient_mmax, nmax_eff),
            gtp_over_gpp=H(
                self.gpsi_theta_over_gpsi_psi,
                coefficient_mmax,
                nmax_eff,
            ),
            gpz_over_gpp=H(
                self.gpsi_zeta_over_gpsi_psi,
                coefficient_mmax,
                nmax_eff,
            ),
            H=H(self.H, coefficient_mmax, nmax_eff),
            J=H(self.J, coefficient_mmax, nmax_eff),
            L=H(self.L, coefficient_mmax, nmax_eff),
            ntor=ntor,
            m_range=m_range,
            max_dm=max_dm,
        )
        blocks.mmax = perturbation_mmax
        blocks.coefficient_mmax = coefficient_mmax
        blocks.weights_q = self.weights
        blocks.points_q = self.points
        blocks.phi_q = self.radialfe.eval_basis(self.points).T
        blocks.WD = self.radialfe.weak_derivative_matrix.T
        return blocks

    def build_solver_blocks_for_m_values(
        self,
        ntor: int,
        m_values: np.ndarray,
        max_dm: int = 10,
    ) -> SolverBlocks:
        """Build uncached blocks for an explicit diagnostic harmonic list.

        This opt-in path does not alter ``self.solver_blocks`` and therefore
        cannot change the production positive-band cache or defaults.
        """
        available_mmax = int(np.abs(self.M.m_pol).max())
        m_range = self._validate_explicit_m_values(m_values, available_mmax)
        return self._assemble_solver_blocks_for_m_range(
            ntor=ntor,
            m_range=m_range,
            max_dm=max_dm,
        )

    def build_solver_blocks(self, ntor: int, max_dm: int=10, mmax: int=None) -> SolverBlocks:
        """
        We build the solver blocks using the pre-evaluated profiles already
        evaluated at the radial points and fill the class to be used later 
        in a faster assembly.
        """
        # If the user provides no mode limit, we will take the maximum poloidal mode
        # number from the H profile.
        H_mmax = np.abs(self.M.m_pol).max()
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
        # mmin_q = -mmax_q

        mmax = min(mmax, mmax_q)

        mmax_eff = mmax#self._solver_mmax(mmax)

        m_range = np.arange(mmin_q, mmax + 1)

        # Let's check if we can reuse the previous solver blocks.
        if hasattr(self, 'solver_blocks'):
            if (
                self.solver_blocks.ntor == ntor
                and self.solver_blocks.mmax == mmax_eff
                and self.solver_blocks.max_dm == max_dm
            ):
                return self.solver_blocks
            else:
                print("Building new solver blocks for ntor =", ntor, "and mmax =", mmax_eff)

        self.solver_blocks = self._assemble_solver_blocks_for_m_range(
            ntor=ntor,
            m_range=m_range,
            max_dm=max_dm,
        )
        
        # We will now solve the continuum problem for the first time to 
        # get the SVD decomposition for the E^-1 operator for the ntor given.
        # w, vdict = self.cont.solve(ntor=ntor, mmax=mmax_eff, 
        #                            psi_vals=self.unique_points,
        #                            which='full', return_eigvecs=True, 
        #                            normalize_eigvals=False)
        # self.solver_blocks.vr = vdict['right']
        # self.solver_blocks.vl = vdict['left']
        # self.solver_blocks.norm = vdict['normalizations']
        # self.solver_blocks.cont_eigvals = w

        return self.solver_blocks

    
    @staticmethod
    def _normalize_bc_kind(bc_kind: str | None) -> str:
        """
        Normalize the boundary-condition selector used for diagnostics.
        """
        if bc_kind is None:
            return "full"

        aliases = {
            "both": "full",
            "all": "full",
            "separatrix": "edge",
            "edge_only": "edge",
            "axis_only": "axis",
            "false": "none",
            "off": "none",
            "no": "none",
        }
        kind = str(bc_kind).strip().lower()
        kind = aliases.get(kind, kind)
        valid = {"full", "axis", "edge", "none"}
        if kind not in valid:
            raise ValueError(f"bc_kind must be one of {sorted(valid)}, got {bc_kind!r}.")
        return kind

    def _species_mass_amu(self) -> float:
        """
        Species mass factor used in the inherited solver-block density.
        """
        return float(self.cont.mass.to("amu").v)

    
    def _get_mixed_radial_data(self) -> MixedRadialData:
        """Return compatible basis data cached for the current high space."""
        if not isinstance(self.radialfe, BSplineFE):
            raise TypeError("The mixed compatible spaces require a BSplineFE high radial space.")
        cached = getattr(self, "_mixed_radial_data", None)
        if cached is None or cached.pair.high is not self.radialfe:
            cached = _build_mixed_radial_data(self.radialfe)
            self._mixed_radial_data = cached
        return cached

    def _mixed_layout(
        self,
        sb: SolverBlocks,
        mixed: MixedRadialData | None = None,
    ) -> MixedFieldLayout:
        mixed = self._get_mixed_radial_data() if mixed is None else mixed
        return MixedFieldLayout(
            n_high=mixed.pair.high.get_ndofs(),
            n_low=mixed.pair.low.get_ndofs(),
            n_modes=np.asarray(sb.m_range).size,
        )

    def _build_nep_boundary_condition(
        self,
        sb: SolverBlocks,
        bc_kind: str | None = "full",
        mixed: MixedRadialData | None = None,
    ) -> BoundaryCondition:
        """Build endpoint constraints in the mixed field-major trial space."""
        mixed = self._get_mixed_radial_data() if mixed is None else mixed
        layout = self._mixed_layout(sb, mixed=mixed)
        kind = self._normalize_bc_kind(bc_kind)

        if kind == "none":
            return BoundaryCondition(
                np.zeros((0, layout.n_trial), dtype=np.complex128)
            )

        axis = regularityAxisBC(mixed.pair.high, mixed.pair.low, sb.m_range)
        if kind == "axis":
            return axis

        edge = idealSeparatrixBC(
            mixed.pair.high,
            mixed.pair.low,
            layout.n_modes,
        )
        if kind == "edge":
            return edge
        return axis * edge

    def assemble_split_matrices(
        self,
        ntor: int,
        mmax: int | None = None,
        max_dm: int = 10,
        regularizer: float = 0.0,
        apply_bcs: bool = True,
        bc_kind: str | None = "full",
        bc_tol: float | None = None,
        verbose: bool = False,
        m_values: np.ndarray | None = None,
    ) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
        r"""Assemble the mixed ``T0``, ``T1`` and ``Tminus1`` matrices.

        ``m_values`` is an opt-in verification interface for an explicit,
        possibly noncontiguous signed harmonic list.  It bypasses the
        production q-band selector without changing or populating the
        production solver-block cache.
        """
        if m_values is None:
            sb = self.build_solver_blocks(ntor=ntor, mmax=mmax, max_dm=max_dm)
        else:
            if mmax is not None:
                raise ValueError("Specify either mmax or m_values, not both.")
            sb = self.build_solver_blocks_for_m_values(
                ntor=ntor,
                m_values=m_values,
                max_dm=max_dm,
            )
        mixed = self._get_mixed_radial_data()
        M0, M1, Mminus1 = _build_local_M_split(sb, regularizer=regularizer)
        return self._assemble_split_matrices_from_local(
            sb,
            mixed,
            (M0, M1, Mminus1),
            apply_bcs=apply_bcs,
            bc_kind=bc_kind,
            bc_tol=bc_tol,
            verbose=verbose,
        )

    def _assemble_split_matrices_from_local(
        self,
        sb: SolverBlocks,
        mixed: MixedRadialData,
        local_split: tuple,
        *,
        apply_bcs: bool = True,
        bc_kind: str | None = "full",
        bc_tol: float | None = None,
        verbose: bool = False,
    ) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
        """Project already-built local tensors into one compatible radial level.

        This is intentionally separate from local block construction so nested
        Galerkin spaces can reuse one finest-grid point-local MHD operator.
        """
        layout = self._mixed_layout(sb, mixed=mixed)

        if not np.allclose(
            np.asarray(sb.points_q),
            mixed.points_q,
            rtol=0.0,
            atol=1.0e-14,
        ):
            raise ValueError("Solver blocks and mixed bases use different quadrature points.")
        if not np.allclose(
            np.asarray(sb.weights_q),
            mixed.weights_q,
            rtol=0.0,
            atol=1.0e-14,
        ):
            raise ValueError("Solver blocks and mixed bases use different quadrature weights.")

        if len(local_split) != 3:
            raise ValueError("local_split must contain (M0, M1, Mminus1).")
        M0, M1, Mminus1 = local_split
        nq = mixed.weights_q.size
        M0_q = _slice_to_quadrature(M0, nq, "M0")
        M1_q = _slice_to_quadrature(M1, nq, "M1")
        Mminus1_q = _slice_to_quadrature(Mminus1, nq, "Mminus1")

        projector_args = (
            sb,
            mixed.phi_high_q,
            mixed.phi_low_q,
            mixed.weights_q,
            mixed.G_low_high,
        )
        T0 = _project_local_tensor_mixed(
            projector_args[0],
            M0_q,
            *projector_args[1:],
            include_derivative=True,
        )
        T1 = _project_local_tensor_mixed(
            projector_args[0],
            M1_q,
            *projector_args[1:],
            include_derivative=False,
        )
        Tminus1 = _project_local_tensor_mixed(
            projector_args[0],
            Mminus1_q,
            *projector_args[1:],
            include_derivative=False,
        )

        expected_shape = (layout.n_rows, layout.n_trial)
        for name, matrix in (
            ("T0", T0),
            ("T1", T1),
            ("Tminus1", Tminus1),
        ):
            if matrix.shape != expected_shape:
                raise RuntimeError(
                    f"{name} mixed projection has shape {matrix.shape}; "
                    f"expected {expected_shape}."
                )

        if verbose:
            print("[assembly] assembled mixed split matrices before BC reduction")
            print(
                f"  ntor={sb.ntor}, n_modes={layout.n_modes}, "
                f"n_high={layout.n_high}, n_low={layout.n_low}"
            )
            print(f"  {_matrix_stats('T0', T0)}")
            print(f"  {_matrix_stats('T1', T1)}")
            print(f"  {_matrix_stats('Tminus1', Tminus1)}")

        kind = self._normalize_bc_kind(bc_kind)
        if not apply_bcs or kind == "none":
            return T0, T1, Tminus1

        bc = self._build_nep_boundary_condition(
            sb,
            bc_kind=kind,
            mixed=mixed,
        )
        expected_rank = 2 * layout.n_modes if kind == "full" else layout.n_modes
        # get_sparse_prolongation already resolves the constraint rank by
        # pivoted QR on C.T, so its column count is the free-dof count.  Read
        # it instead of paying for a second, dense SVD of the same matrix.
        prolongation = bc.get_sparse_prolongation(tol=bc_tol)
        bc_rank = int(layout.n_trial - prolongation.shape[1])
        if bc_rank != expected_rank:
            raise ValueError(
                f"{kind!r} boundary rank must be {expected_rank}, got {bc_rank}."
            )

        reduced = tuple(
            bc.reduce_sparse_trial_matrix(matrix, tol=bc_tol)
            for matrix in (T0, T1, Tminus1)
        )
        reduced_shape = (layout.n_rows, layout.n_trial - bc_rank)
        for name, matrix in zip(("T0", "T1", "Tminus1"), reduced):
            if matrix.shape != reduced_shape:
                raise RuntimeError(
                    f"Reduced {name} has shape {matrix.shape}; expected {reduced_shape}."
                )

        if kind == "full" and reduced_shape != (layout.n_rows, layout.n_rows):
            raise ValueError(
                "Full mixed boundary reduction must produce a square operator, "
                f"got {reduced_shape}."
            )

        if verbose:
            print("[assembly] split matrices after one-sided BC reduction")
            print(
                f"  bc_kind={kind}, bc_rank={bc_rank}, "
                f"full_shape={expected_shape}, reduced_shape={reduced_shape}"
            )
            for name, matrix in zip(("T0", "T1", "Tminus1"), reduced):
                print(f"  {_matrix_stats(name + '_reduced', matrix)}")
        return reduced

    def _require_full_bc_nep(self, apply_bcs: bool, bc_kind: str | None) -> None:
        """Reject rectangular diagnostic configurations before PETSc is loaded."""
        kind = self._normalize_bc_kind(bc_kind)
        if not apply_bcs or kind != "full":
            raise ValueError(
                "Mixed-space solves require apply_bcs=True and bc_kind='full' "
                "so the compatible mixed operator is square. Partial/no-BC "
                "assemblies are available only through assemble_split_matrices()."
            )

    def _split_result_dataset(
        self,
        result,
        nep,
        ntor: int,
        target_omega: complex | None,
        target_s: complex,
        region: dict,
        mmax: int | None,
        max_dm: int,
        regularizer: float,
        apply_bcs: bool,
        bc_kind: str | None,
        bc_tol: float | None,
        verbose: bool,
        compute_residuals: bool,
        which: str,
        nep_type: str,
        description: str,
        extra_attrs: dict | None = None,
        m_values: np.ndarray | None = None,
    ) -> xr.Dataset:
        """Pack full mixed vectors and named high/low coefficient arrays."""
        if m_values is None:
            sb = self.build_solver_blocks(ntor=ntor, mmax=mmax, max_dm=max_dm)
            explicit_m_values = None
        else:
            if mmax is not None:
                raise ValueError("Specify either mmax or m_values, not both.")
            available_mmax = int(np.abs(self.M.m_pol).max())
            explicit_m_values = self._validate_explicit_m_values(
                m_values,
                available_mmax,
            )
            # Result packing needs only the harmonic labels. Reassembling the
            # uncached diagnostic blocks here would duplicate the expensive
            # signed-m assembly already performed for the eigensolve.
            sb = SimpleNamespace(m_range=explicit_m_values)
        layout = self._mixed_layout(sb)
        species_mass_amu = self._species_mass_amu()
        omega = np.sqrt(result.eigenvalues)

        solver_eigenvectors = np.asarray(result.eigenvectors, dtype=np.complex128)
        if solver_eigenvectors.ndim != 2:
            raise ValueError("NEP eigenvectors must be a two-dimensional array.")
        slepc_dofs = solver_eigenvectors.shape[1]
        eigenvectors = solver_eigenvectors
        bc_expanded_eigenvectors = False
        kind = self._normalize_bc_kind(bc_kind)
        slepc_eigenvector_space = "full"

        if apply_bcs and kind != "none":
            bc = self._build_nep_boundary_condition(sb, bc_kind=kind)
            if slepc_dofs == layout.n_trial:
                pass
            elif slepc_dofs == bc.get_sparse_free_dofs(tol=bc_tol):
                slepc_eigenvector_space = "reduced"
                eigenvectors = (
                    np.vstack(
                        [bc.expand_sparse(vector, tol=bc_tol)
                         for vector in solver_eigenvectors]
                    )
                    if len(solver_eigenvectors)
                    else np.empty((0, layout.n_trial), dtype=np.complex128)
                )
                bc_expanded_eigenvectors = True
            else:
                raise ValueError(
                    "SLEPc eigenvector width does not match either the full or "
                    f"reduced mixed space: got {slepc_dofs}, full={layout.n_trial}, "
                    f"reduced={bc.get_sparse_free_dofs(tol=bc_tol)}."
                )
        elif slepc_dofs != layout.n_trial:
            raise ValueError(
                f"Expected {layout.n_trial} full mixed DoFs, got {slepc_dofs}."
            )

        if eigenvectors.shape[1] != layout.n_trial:
            raise RuntimeError(
                f"Expanded mixed eigenvectors have width {eigenvectors.shape[1]}; "
                f"expected {layout.n_trial}."
            )

        n_eigenpairs = eigenvectors.shape[0]

        def field_coefficients(field: int, n_radial: int) -> np.ndarray:
            return eigenvectors[:, layout.trial_slice(field)].reshape(
                n_eigenpairs,
                n_radial,
                layout.n_modes,
            )

        data_vars = {
            "s": ("eigenpair", result.eigenvalues),
            "omega": ("eigenpair", omega),
            "eigenvector": (("eigenpair", "dof"), eigenvectors),
            "p1_coefficient": (
                ("eigenpair", "radial_dof_high", "m_pol"),
                field_coefficients(0, layout.n_high),
            ),
            "xi_psi_coefficient": (
                ("eigenpair", "radial_dof_high", "m_pol"),
                field_coefficients(1, layout.n_high),
            ),
            "xi_s_coefficient": (
                ("eigenpair", "radial_dof_low", "m_pol"),
                field_coefficients(2, layout.n_low),
            ),
            "div_xi_coefficient": (
                ("eigenpair", "radial_dof_low", "m_pol"),
                field_coefficients(3, layout.n_low),
            ),
        }
        if getattr(nep, "last_residual_norms", None) is not None:
            data_vars["residual_norm"] = (
                "eigenpair",
                nep.last_residual_norms,
            )
            data_vars["absolute_residual_norm"] = (
                "eigenpair",
                nep.last_absolute_residual_norms,
            )
        if getattr(nep, "last_backward_errors", None) is not None:
            data_vars["backward_error"] = (
                "eigenpair",
                nep.last_backward_errors,
            )
        if getattr(nep, "last_slepc_error_estimates", None) is not None:
            data_vars["slepc_error_estimate"] = (
                "eigenpair",
                nep.last_slepc_error_estimates,
            )
        if getattr(nep, "last_slepc_backward_errors", None) is not None:
            data_vars["slepc_backward_error"] = (
                "eigenpair",
                nep.last_slepc_backward_errors,
            )

        attrs = {
            "ntor": int(ntor),
            "split_variable": "omega2",
            "omega_normalization": "omegaA",
            "omega_is_species_mass_normalized": True,
            "density_includes_species_mass": True,
            "species_mass_amu": species_mass_amu,
            "target_omega": target_omega,
            "target_s": target_s,
            "mmax": None if mmax is None else int(mmax),
            "harmonic_selection": (
                "production_q_band"
                if explicit_m_values is None
                else "explicit_diagnostic"
            ),
            "explicit_m_values": (
                ""
                if explicit_m_values is None
                else np.array2string(explicit_m_values, separator=",")
            ),
            "max_dm": int(max_dm),
            "regularizer": regularizer,
            "apply_bcs": bool(apply_bcs),
            "bc_kind": kind,
            "region": str(region),
            "region_type": region["type"],
            "region_center": region["center"],
            "region_radius": region["radius"],
            "region_vscale": region["vscale"],
            "verbose": bool(verbose),
            "compute_residuals": bool(compute_residuals),
            "eigenvector_space": "full",
            "slepc_eigenvector_space": slepc_eigenvector_space,
            "bc_expanded_eigenvectors": bool(bc_expanded_eigenvectors),
            "slepc_dofs": int(slepc_dofs),
            "full_dofs": int(layout.n_trial),
            "equation_dofs": int(layout.n_rows),
            "n_radial_high": int(layout.n_high),
            "n_radial_low": int(layout.n_low),
            "n_modes": int(layout.n_modes),
            "radial_discretization": "compatible_mixed_bspline",
            "trial_layout": "field-major [p1_H, xi_psi_H, xi_s_L, div_xi_L]",
            "residual_space": "reduced" if apply_bcs and kind != "none" else "full",
            "nep_type": nep_type,
            "problem_type": "RATIONAL",
            "which": which,
            "nconv": int(result.nconv),
            "iterations": int(result.iterations),
            "reason": str(result.reason),
            "description": description,
        }
        if hasattr(nep, "spectral_shift"):
            attrs["spectral_shift"] = nep.spectral_shift
            attrs["slepc_variable"] = "s_minus_spectral_shift"
        else:
            attrs["slepc_variable"] = "s"
        if extra_attrs:
            attrs.update(extra_attrs)

        return xr.Dataset(
            data_vars=data_vars,
            coords={
                "eigenpair": np.arange(n_eigenpairs, dtype=np.int64),
                "dof": np.arange(layout.n_trial, dtype=np.int64),
                "radial_dof_high": np.arange(layout.n_high, dtype=np.int64),
                "radial_dof_low": np.arange(layout.n_low, dtype=np.int64),
                "m_pol": np.asarray(sb.m_range),
            },
            attrs=attrs,
        )

    
    def build_pep_solver(
        self,
        ntor: int,
        target_omega: complex,
        region: dict | None,
        mmax: int | None = None,
        max_dm: int = 10,
        regularizer: float = 0.0,
        apply_bcs: bool = True,
        bc_kind: str | None = "full",
        bc_tol: float | None = None,
        prefix: str = "nova_quadratic_pep",
        pep_type: str = "TOAR",
        st_type: str = "SINVERT",
        which: str = "TARGET_MAGNITUDE",
        nev: int = 6,
        tol: float = 1e-8,
        maxit: int = 100,
        options: dict | None = None,
        verbose: bool = False,
        compute_residuals: bool = True,
        m_values: np.ndarray | None = None,
    ) -> QuadraticPEPSolver | LinearGeneralizedEPSSolver:
        """Assemble and build the appropriate eigensolver.

        The historical method name remains unchanged.  Exact ``gamma_s == 0``
        returns a generalized EPS solver for ``(-T0)x = s*T1*x``; otherwise it
        returns the quadratic PEP with ``[Tminus1, T0, T1]``.

        ``m_values`` is an opt-in diagnostic harmonic list. The production
        q-selected positive-m band remains the default.
        """
        self._require_full_bc_nep(apply_bcs, bc_kind)
        if m_values is not None and mmax is not None:
            raise ValueError("Specify either mmax or m_values, not both.")
        target_omega = _as_scalar_complex(target_omega, "target_omega")
        target_s = target_omega**2
        region = _ellipse_region_dict(region, target_s)
        T0, T1, Tminus1 = self.assemble_split_matrices(
            ntor=ntor,
            mmax=mmax,
            max_dm=max_dm,
            regularizer=regularizer,
            apply_bcs=apply_bcs,
            bc_kind=bc_kind,
            bc_tol=bc_tol,
            verbose=verbose,
            m_values=m_values,
        )
        return build_eigensolver(
            gamma_s=self.gamma_s,
            T0=T0,
            T1=T1,
            Tminus1=Tminus1,
            target_s=target_s,
            region=region,
            prefix=prefix,
            pep_type=pep_type,
            st_type=st_type,
            which=which,
            nev=nev,
            tol=tol,
            maxit=maxit,
            options=options,
            verbose=verbose,
            compute_residuals=compute_residuals,
        )

    def solve_pep(
        self,
        ntor: int,
        target_omega: complex,
        region: dict | None,
        mmax: int | None = None,
        max_dm: int = 10,
        regularizer: float = 0.0,
        apply_bcs: bool = True,
        bc_kind: str | None = "full",
        bc_tol: float | None = None,
        prefix: str = "nova_quadratic_pep",
        pep_type: str = "TOAR",
        st_type: str = "SINVERT",
        which: str = "TARGET_MAGNITUDE",
        nev: int = 6,
        tol: float = 1e-8,
        maxit: int = 100,
        options: dict | None = None,
        max_eigenpairs: int | None = None,
        verbose: bool = False,
        compute_residuals: bool = True,
        m_values: np.ndarray | None = None,
    ) -> xr.Dataset:
        """Solve with transparent exact-zero EPS dispatch.

        Pass an explicit signed ``m_values`` array only for diagnostics that
        require a harmonic space closed under ``m -> -m``.
        """
        if m_values is not None and mmax is not None:
            raise ValueError("Specify either mmax or m_values, not both.")
        target_omega = _as_scalar_complex(target_omega, "target_omega")
        target_s = target_omega**2
        region = _ellipse_region_dict(region, target_s)
        eigensolver = self.build_pep_solver(
            ntor=ntor,
            target_omega=target_omega,
            region=region,
            mmax=mmax,
            max_dm=max_dm,
            regularizer=regularizer,
            apply_bcs=apply_bcs,
            bc_kind=bc_kind,
            bc_tol=bc_tol,
            prefix=prefix,
            pep_type=pep_type,
            st_type=st_type,
            which=which,
            nev=nev,
            tol=tol,
            maxit=maxit,
            options=options,
            verbose=verbose,
            compute_residuals=compute_residuals,
            m_values=m_values,
        )
        backend, description, extra_attrs = self._backend_attrs(
            eigensolver, pep_type, st_type
        )
        try:
            result = eigensolver.solve(max_eigenpairs=max_eigenpairs)
            output = self._split_result_dataset(
                result=result,
                nep=eigensolver,
                ntor=ntor,
                target_omega=target_omega,
                target_s=target_s,
                region=region,
                mmax=mmax,
                max_dm=max_dm,
                regularizer=regularizer,
                apply_bcs=apply_bcs,
                bc_kind=bc_kind,
                bc_tol=bc_tol,
                verbose=verbose,
                compute_residuals=compute_residuals,
                which=which,
                nep_type=backend,
                description=description,
                m_values=m_values,
                extra_attrs=extra_attrs,
            )
        except BaseException as primary_error:
            try:
                eigensolver.destroy()
            except BaseException as cleanup_error:
                # Keep the solve/packing failure as the exception presented to
                # the caller.  The cleanup failure remains available as its
                # explicit cause instead of masking the actionable traceback.
                raise primary_error from cleanup_error
            raise
        else:
            eigensolver.destroy()
            return output

    def solve_pep_windows(
        self,
        ntor: int,
        windows,
        mmax: int | None = None,
        max_dm: int = 10,
        regularizer: float = 0.0,
        apply_bcs: bool = True,
        bc_kind: str | None = "full",
        bc_tol: float | None = None,
        prefix: str = "nova_quadratic_pep",
        pep_type: str = "TOAR",
        st_type: str = "SINVERT",
        which: str = "TARGET_MAGNITUDE",
        nev: int = 6,
        tol: float = 1e-8,
        maxit: int = 100,
        options: dict | None = None,
        max_eigenpairs: int | None = None,
        verbose: bool = False,
        compute_residuals: bool = True,
        m_values: np.ndarray | None = None,
    ):
        """Solve several frequency windows on this radial grid, assembling once.

        ``windows`` yields ``(target_omega, region)`` pairs, each as
        :meth:`solve_pep` takes them. The split matrices are projected and the
        eigensolver is built for the first window only; every later window
        retargets it, so only the shift-and-invert factorization is redone.
        Re-projecting the same pencil per window is what dominates calling
        :meth:`solve_pep` in a loop.

        Yields one dataset per window, in order. A window in which nothing
        converges yields a dataset with no eigenpairs instead of raising; its
        ``nconv``, ``iterations`` and ``reason`` attributes say why. As a
        generator it lets the caller report progress or stop between windows;
        the eigensolver is destroyed when it is exhausted, closed or raises.
        """
        if m_values is not None and mmax is not None:
            raise ValueError("Specify either mmax or m_values, not both.")
        eigensolver = None
        try:
            for target_omega, region in windows:
                target_omega = _as_scalar_complex(target_omega, "target_omega")
                target_s = target_omega**2
                region = _ellipse_region_dict(region, target_s)
                if eigensolver is None:
                    eigensolver = self.build_pep_solver(
                        ntor=ntor,
                        target_omega=target_omega,
                        region=region,
                        mmax=mmax,
                        max_dm=max_dm,
                        regularizer=regularizer,
                        apply_bcs=apply_bcs,
                        bc_kind=bc_kind,
                        bc_tol=bc_tol,
                        prefix=prefix,
                        pep_type=pep_type,
                        st_type=st_type,
                        which=which,
                        nev=nev,
                        tol=tol,
                        maxit=maxit,
                        options=options,
                        verbose=verbose,
                        compute_residuals=compute_residuals,
                        m_values=m_values,
                    )
                else:
                    eigensolver.retarget(target_s, region)
                result = eigensolver.solve(
                    max_eigenpairs=max_eigenpairs, allow_empty=True
                )
                yield self.pack_window(
                    WindowSolution.from_solver(eigensolver, result),
                    ntor=ntor,
                    target_omega=target_omega,
                    region=region,
                    mmax=mmax,
                    max_dm=max_dm,
                    regularizer=regularizer,
                    apply_bcs=apply_bcs,
                    bc_kind=bc_kind,
                    bc_tol=bc_tol,
                    pep_type=pep_type,
                    st_type=st_type,
                    which=which,
                    verbose=verbose,
                    compute_residuals=compute_residuals,
                    m_values=m_values,
                )
        finally:
            if eigensolver is not None:
                eigensolver.destroy()

    def pack_window(
        self,
        solution: WindowSolution,
        *,
        ntor: int,
        target_omega: complex,
        region: dict | None,
        mmax: int | None = None,
        max_dm: int = 10,
        regularizer: float = 0.0,
        apply_bcs: bool = True,
        bc_kind: str | None = "full",
        bc_tol: float | None = None,
        pep_type: str = "TOAR",
        st_type: str = "SINVERT",
        which: str = "TARGET_MAGNITUDE",
        verbose: bool = False,
        compute_residuals: bool = True,
        m_values: np.ndarray | None = None,
    ) -> xr.Dataset:
        """Pack one window's eigenpairs into the dataset :meth:`solve_pep` returns.

        The arguments are the ones the window was solved with. Packing needs
        this solver -- its harmonic band, mixed layout and boundary reduction
        expand the eigenvectors -- but not the eigensolver, which may have run
        in another process; see :class:`WindowSolution`.
        """
        target_omega = _as_scalar_complex(target_omega, "target_omega")
        target_s = target_omega**2
        region = _ellipse_region_dict(region, target_s)
        nep = SimpleNamespace(
            backend=solution.backend,
            eps_type=solution.eps_type,
            last_residual_norms=solution.residual_norms,
            last_absolute_residual_norms=solution.absolute_residual_norms,
            last_backward_errors=solution.backward_errors,
            last_slepc_error_estimates=solution.slepc_error_estimates,
            last_slepc_backward_errors=solution.slepc_backward_errors,
        )
        backend, description, extra_attrs = self._backend_attrs(nep, pep_type, st_type)
        result = NEPResult(
            solution.eigenvalues,
            solution.eigenvectors,
            solution.nconv,
            solution.reason,
            solution.iterations,
        )
        return self._split_result_dataset(
            result=result,
            nep=nep,
            ntor=ntor,
            target_omega=target_omega,
            target_s=target_s,
            region=region,
            mmax=mmax,
            max_dm=max_dm,
            regularizer=regularizer,
            apply_bcs=apply_bcs,
            bc_kind=bc_kind,
            bc_tol=bc_tol,
            verbose=verbose,
            compute_residuals=compute_residuals,
            which=which,
            nep_type=backend,
            description=description,
            m_values=m_values,
            extra_attrs=extra_attrs,
        )

    @staticmethod
    def _backend_attrs(eigensolver, pep_type: str, st_type: str):
        """``(backend, description, extra_attrs)`` for a built eigensolver."""
        backend = getattr(eigensolver, "backend", "PEP")
        if backend == "EPS":
            description = (
                "SLEPc generalized EPS solution of the exact gamma_s=0 "
                "pencil (-T0) x = s*T1 x."
            )
            extra_attrs = {
                "problem_type": "GENERALIZED_LINEAR",
                "eps_type": getattr(eigensolver, "eps_type", "KRYLOVSCHUR"),
                "st_type": st_type,
                "linear_coefficients": "[-T0, T1]",
                "gamma_zero_reduction": True,
                "requested_pep_type": pep_type,
            }
        else:
            description = (
                "SLEPc quadratic PEP solution of the sparse pencil "
                "(Tminus1 + s*T0 + s**2*T1) x = 0."
            )
            extra_attrs = {
                "problem_type": "POLYNOMIAL",
                "pep_type": pep_type,
                "st_type": st_type,
                "polynomial_degree": 2,
                "polynomial_coefficients": "[Tminus1, T0, T1]",
                "gamma_zero_reduction": False,
                "zero_root_caveat": (
                    "Multiplication by s can introduce roots at s=0; "
                    "the configured search region excludes zero."
                ),
            }
        return backend, description, extra_attrs

