r"""
Main solver for the linear ideal MHD eigenvalue problem.

The problem is a nonlinear eigenvalue problem on the form:

T(\omega) x = 0

In this implementation, we will solve for the big problem in equations
(24) and (25) instead of using the Schur's complement. This should get
rid of the annoying "divide by zero" problem that arises when the continuum is not properly
regularized. The downside is that we will have to solve a bigger problem, but we will
use sparse methods.


T = (\partial_\psi, \partial_\psi, 0, 0)(p_1, \xi_\psi, \xi^s, \nabla\cdot\xi) - ((C D)(F - E)) (p_1, \xi_\psi, \xi^s, \nabla\cdot\xi)
"""

import numpy as np
import unyt
from supernova.equilibrium import superEquilibrium, _active_poloidal_flux_span
from supernova.continuum import superContinuum
from supernova.continuum import continuum_crossings, get_knots_clustered_at
from supernova.radial.base import RadialFE
from supernova.radial.profile import Profile

# Let's import the kernels
from supernova.operators.blocks import solver_blocks2, build_solver_blocks2
# Let's import the PETSc interface for the Krylov inversion.


class superSolver2:
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

        Since the objective of the superSolver is to separate the radial from the 
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
    ) -> solver_blocks2:
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

        blocks = build_solver_blocks2(
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
    ) -> solver_blocks2:
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

    def build_solver_blocks(self, ntor: int, max_dm: int=10, mmax: int=None) -> solver_blocks2:
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

