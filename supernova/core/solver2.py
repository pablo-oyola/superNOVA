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
import matplotlib.pyplot as plt
import xarray as xr
import unyt
from scipy.sparse.linalg import splu
from .equilibrium import superEquilibrium
from .continuum import superContinuum
from .continuum import continuum_crossings, get_knots_clustered_at
from ..radial.base import RadialFE
from ..radial.profile import Profile

# Let's import the kernels
from ._solver_kernel2 import solver_blocks2, build_solver_blocks2, build_M_at, get_T_sparse
from ._antenna import Antenna
from ._boundary import BoundaryCondition, idealSeparatrixBC, couplingPressureDisplacementBC, regularityAxisBC
from ._petsc_helpers import KrylovSolver, NEPSolver
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
        C22 = - g^{\psi\psi}* div (\nabla\Psi / g^{\psi\psi})
        C12 = \lambda + 2\mu_0p_0'K_\Psi + sqrt(g) * g^{\psi\psi} G^H O G + \
              (J · B - Sg^{\psi\psi}) * S*g^{\psi\psi}/B^2,
        where O = 1/(sqrt(g)||\nabla\Psi||^2) and G = i(nq - m)\delta_{m,m'}\delta_{n,n'} and
        \lambda is the eigenvalue: \lambda = \omega^2 * \rho_0 * \mu_0.


        D =  (D11  D12)
             (D21  D22)

        where D11 = - (J · B - Sg^{\psi\psi}) * g^{\psi\psi}/(sqrt(g)*B^2) * G
              D12 = 2 \gamma_s * p_0 * \mu_0 K_\Psi
              D21 = g^{\psi\psi} * (2K_s - \sigma/(sqrt(g)*B^2))
              D22 = g^{\psi\psi} * (1 + \mu_0 p_0 \gamma_s / \lambda G^H M G),
              where
              \sigma = i(In - gm)\delta_{m,m'}\delta_{n,n'}, and
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
        C12 = \lambda/g^{\psi\psi} + \mu_0 * p_0' * C11 + 1/sqrt(g) G^H O G + \
              (J · B - Sg^{\psi\psi}) * S/ B^2
        C22 = - \nabla·(\nabla\Psi/g^{\psi\psi}) --> Precomputed as divPsiVec

        D11 = - (J · B - Sg^{\psi\psi}) /(sqrt(g)*B^2) * G
        D12 = gamma_s * p_0 * mu_0 * C11
        D21 = 2*kappa_s - \sigma/(sqrt(g)*B^2)
        D22 = 1 + \mu_0 p_0 \gamma_s / \lambda G^H M G

        Hence the final problem is:
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
                       'B0**2/gpsi_psi/h',    # The O operator                                    - 3
                       '- divPsiVec', # C22 - Precomputed.                                        - 4
                       '(shear*gpsi_psi - J_B)/h',  # D11                                         - 5
                       'kappa_s',  # D21 - first part; distinct from the continuum K operator.    - 6
                       '1/B0**2', # F11 second part.                                              - 7
                       'gpsi_psi/B0**2 * shear', # F12 - first part                               - 8
                       '- J_B/B0**2', # F12 - second part                                         - 9
                       '-1/B0**4*h', # F21                                                        - 10
                       '-2*kappa_Psi/(gpsi_psi*B0**2)*h', # F22                                     - 11
                       '1/gpsi_psi', # Harmonic operator needed by lambda/gpsi_psi.               - 12
                       'gpsi_theta/gpsi_psi', # Surface phase term for e^{i(m*theta-n*zeta)}.     - 13
                       'gpsi_zeta/gpsi_psi', # Surface phase term for e^{i(m*theta-n*zeta)}.      - 14
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
        self.D21_2 = self.cont.M # = h(\psi)
        self.F11_2 = tmp[7]
        self.F12_1 = tmp[8]
        self.F12_2 = tmp[9]
        self.F21 = tmp[10]
        self.F22 = tmp[11]
        self.inv_gpsi_psi = tmp[12]
        self.gpsi_theta_over_gpsi_psi = tmp[13]
        self.gpsi_zeta_over_gpsi_psi = tmp[14]

        # The rest of the operators come from the continuum:
        self.M = self.cont.M
        self.K = self.cont.K

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
        self.D21_2.data = self.D21_2.data.interp(x=self.unique_points)
        self.F11_2.data = self.F11_2.data.interp(x=self.unique_points)
        self.F12_1.data = self.F12_1.data.interp(x=self.unique_points)
        self.F12_2.data = self.F12_2.data.interp(x=self.unique_points)
        self.F21.data = self.F21.data.interp(x=self.unique_points)
        self.F22.data = self.F22.data.interp(x=self.unique_points)
        self.cont.H.data = self.cont.H.data.interp(x=self.unique_points)
        self.cont.M.data = self.cont.M.data.interp(x=self.unique_points)
        self.cont.L.data = self.cont.L.data.interp(x=self.unique_points)
        self.cont.J.data = self.cont.J.data.interp(x=self.unique_points)
        self.inv_gpsi_psi.data = self.inv_gpsi_psi.data.interp(x=self.unique_points)
        self.gpsi_theta_over_gpsi_psi.data = self.gpsi_theta_over_gpsi_psi.data.interp(x=self.unique_points)
        self.gpsi_zeta_over_gpsi_psi.data = self.gpsi_zeta_over_gpsi_psi.data.interp(x=self.unique_points)

        # The rest of the operators come from the continuum:
        self.M.data = self.M.data.interp(x=self.unique_points)
        self.K.data = self.K.data.interp(x=self.unique_points)

        # We need to get for the radial derivative the underlying unit and scale
        # of the flux. For the absolute value of the flux at the edge, psiN, we
        # need to inquire the boozer coordinates.
        psiN = np.abs(self.equ.boozer.coords.psi0).max() * unyt.Unit('Wb', registry=self.equ.registry)
        psiN = psiN.data.in_base(self.equ.my_norm).to('psi0').v

        # We need also to divide the density by the maximum density.
        n0 = (1 * unyt.Unit("naxis", registry=self.equ.registry)).to("1/aminor**3")

        # We keep too a copy of the density, pressure and the derivative.
        self.density = self.equ('density', normalize=True)
        self.pressure = self.equ('pressure', normalize=True)
        self.pressure_deriv = self.equ('pressure', normalize=True)
        self.density.data = self.density.data.interp(x=self.unique_points) / n0.v
        self.pressure.data = self.pressure.data.interp(x=self.unique_points)
        self.pressure_deriv.data = self.pressure.data.differentiate('x').interp(x=self.unique_points) / psiN

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
        available = int(np.abs(self.cont.H.m_pol).max())
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

        mmax = min(mmax, mmax_q)

        mmax_eff = mmax#self._solver_mmax(mmax)
        nmax_eff = 0

        m_range = np.arange(mmin_q, mmax + 1)

        # Let's check if we can reuse the previous solver blocks.
        if hasattr(self, 'solver_blocks'):
            if self.solver_blocks.ntor == ntor and self.solver_blocks.mmax == mmax_eff:
                print("Reusing previous solver blocks.")
                return self.solver_blocks
            else:
                print("Building new solver blocks for ntor =", ntor, "and mmax =", mmax_eff)

        H = self._profile_harmonics
        S = self._profile_scalar
        # Solver blocks use lambda = mu_0 * rho_mass * omega**2.  The
        # equilibrium density profile is normalized number density, while
        # omegaA is referenced to 1 amu, so include the species mass in amu
        # here rather than rescaling the requested frequency.
        mass_amu = float(self.cont.mass.to("amu").v)
        mass_density = S(self.density) * mass_amu

        self.solver_blocks = build_solver_blocks2(C11=H(self.C11, mmax_eff, nmax_eff),
                                                 O=H(self.O, mmax_eff, nmax_eff),
                                                 C12_2=H(self.C12_2, mmax_eff, nmax_eff),
                                                 C12_3=H(self.C12_3, mmax_eff, nmax_eff),
                                                 C22=H(self.C22, mmax_eff, nmax_eff),
                                                 D11=H(self.D11, mmax_eff, nmax_eff),
                                                 D12=H(self.C11, mmax_eff, nmax_eff),
                                                 kappa_s=H(self.D21_1, mmax_eff, nmax_eff),
                                                 K=H(self.K, mmax_eff, nmax_eff),
                                                 M=H(self.M, mmax_eff, nmax_eff),
                                                 F11_2=H(self.F11_2, mmax_eff, nmax_eff),
                                                 F12_1=H(self.F12_1, mmax_eff, nmax_eff),
                                                 F12_2=H(self.F12_2, mmax_eff, nmax_eff),
                                                 F21=H(self.F21, mmax_eff, nmax_eff),
                                                 F22=H(self.F22, mmax_eff, nmax_eff),
                                                 hprof=S(self.hprof),
                                                 Iprof=S(self.Iprof),
                                                 gprof=S(self.gprof),
                                                 qprof=S(self.qprof),
                                                 pressure=S(self.pressure),
                                                 pressure_deriv=S(self.pressure_deriv),
                                                 density=mass_density,
                                                 gamma_s=self.gamma_s,
                                                 one_gpp=H(self.inv_gpsi_psi, mmax_eff, nmax_eff),
                                                 gtp_over_gpp=H(self.gpsi_theta_over_gpsi_psi, mmax_eff, nmax_eff),
                                                 gpz_over_gpp=H(self.gpsi_zeta_over_gpsi_psi, mmax_eff, nmax_eff),
                                                 H=H(self.cont.H, mmax_eff, nmax_eff),
                                                 J=H(self.cont.J, mmax_eff, nmax_eff),
                                                 L=H(self.cont.L, mmax_eff, nmax_eff),
                                                 ntor=ntor,
                                                 m_range=m_range,
                                                 max_dm=max_dm)
        self.solver_blocks.mmax = mmax_eff
        
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

        # We fill the elements related to the radial operations.
        self.solver_blocks.weights_q = self.weights
        self.solver_blocks.points_q = self.points
        self.solver_blocks.phi_q = self.radialfe.eval_basis(self.points).T
        self.solver_blocks.WD = self.radialfe.weak_derivative_matrix.T

        return self.solver_blocks

    def _lift_xip1_boundary_condition(
        self,
        bc: BoundaryCondition,
        n_modes: int,
    ) -> BoundaryCondition:
        """
        Lift a two-field ``[p1, xi^psi]`` boundary condition to the four-field
        solver2 state ``[p1, xi^psi, xi^s, div_xi]``.
        """
        n_radial = self.radialfe.get_ndofs()
        two_state = 2 * n_modes
        four_state = 4 * n_modes
        expected = (n_radial * two_state,)
        if bc.C.shape[1:] != expected:
            raise ValueError(f"Expected two-field BC width {expected[0]}, got {bc.C.shape[1]}.")

        C = np.zeros((bc.C.shape[0], n_radial * four_state), dtype=np.complex128)
        for irad in range(n_radial):
            cols2 = slice(irad * two_state, (irad + 1) * two_state)
            cols4 = slice(irad * four_state, irad * four_state + two_state)
            C[:, cols4] = bc.C[:, cols2]
        return BoundaryCondition(C)

    def _build_nep_boundary_condition(self, sb: solver_blocks2) -> BoundaryCondition:
        """
        Build the sparse-compatible boundary condition for the current NEP test path.

        This applies the existing axis regularity and ideal-separatrix constraints
        to the ``[p1, xi^psi]`` part of the solver2 state. The algebraic pressure-
        displacement coupling at the edge is not lifted here because the new
        four-field operator keeps ``xi^s`` and ``div_xi`` explicitly.
        """
        n_modes = sb.m_range.size
        bc_axis = regularityAxisBC(self.radialfe, sb.m_range)
        bc_sep = idealSeparatrixBC(self.radialfe, n_modes)
        bc_xip1 = bc_axis * bc_sep
        return self._lift_xip1_boundary_condition(bc_xip1, n_modes)

    def assemble_nep_matrix(
        self,
        omega: complex,
        ntor: int,
        mmax: int=None,
        max_dm: int=10,
        regularizer: float=0.0,
        apply_bcs: bool=True,
        bc_tol: float=None,
    ):
        """
        Assemble the sparse global NEP matrix ``T(omega)`` for solver2.

        The returned matrix is a SciPy CSR matrix. When ``apply_bcs`` is true,
        the sparse boundary-condition prolongation path is used, so the reduced
        matrix remains sparse before it is copied into PETSc.
        """
        sb = self.build_solver_blocks(ntor=ntor, mmax=mmax, max_dm=max_dm)
        M = build_M_at(sb=sb, omega=omega, regularizer=regularizer)

        nq = np.asarray(sb.weights_q).shape[0]
        if M.shape[0] == nq + 1:
            M_q = M[:nq]
        elif M.shape[0] == nq:
            M_q = M
        else:
            raise ValueError(
                "M must be evaluated at the quadrature points, optionally plus "
                f"one edge point. Got {M.shape[0]} local points and {nq} quadrature points."
            )

        T = get_T_sparse(sb=sb, M=M_q, omega=omega, regularizer=regularizer)
        if not apply_bcs:
            return T

        bc = self._build_nep_boundary_condition(sb)
        return bc.reduce_sparse_matrix(T, tol=bc_tol)

    def build_nep_solver(
        self,
        ntor: int,
        target_omega: complex,
        mmax: int=None,
        max_dm: int=10,
        regularizer: float=0.0,
        apply_bcs: bool=True,
        bc_tol: float=None,
        prefix: str="nova_nep",
        nep_type: str="SLP",
        which: str="TARGET_MAGNITUDE",
        nev: int=6,
        tol: float=1e-8,
        maxit: int=100,
        finite_difference_jacobian: bool=True,
        fd_step: float=None,
        options: dict=None,
    ) -> NEPSolver:
        """
        Create a SLEPc NEP solver around the sparse solver2 matrix assembly.
        """
        def operator(omega):
            return self.assemble_nep_matrix(
                omega=omega,
                ntor=ntor,
                mmax=mmax,
                max_dm=max_dm,
                regularizer=regularizer,
                apply_bcs=apply_bcs,
                bc_tol=bc_tol,
            )

        initial_matrix = operator(target_omega)
        return NEPSolver(
            operator=operator,
            target=target_omega,
            initial_matrix=initial_matrix,
            prefix=prefix,
            nep_type=nep_type,
            which=which,
            nev=nev,
            tol=tol,
            maxit=maxit,
            finite_difference_jacobian=finite_difference_jacobian,
            fd_step=fd_step,
            options=options,
        )

    def solve_nep(
        self,
        ntor: int,
        target_omega: complex,
        mmax: int=None,
        max_dm: int=10,
        regularizer: float=0.0,
        apply_bcs: bool=True,
        bc_tol: float=None,
        prefix: str="nova_nep",
        nep_type: str="SLP",
        which: str="TARGET_MAGNITUDE",
        nev: int=6,
        tol: float=1e-8,
        maxit: int=100,
        finite_difference_jacobian: bool=True,
        fd_step: float=None,
        options: dict=None,
        max_eigenpairs: int=None,
    ) -> xr.Dataset:
        """
        Solve ``T(omega) x = 0`` with SLEPc using sparse PETSc AIJ matrices.
        """
        nep = self.build_nep_solver(
            ntor=ntor,
            target_omega=target_omega,
            mmax=mmax,
            max_dm=max_dm,
            regularizer=regularizer,
            apply_bcs=apply_bcs,
            bc_tol=bc_tol,
            prefix=prefix,
            nep_type=nep_type,
            which=which,
            nev=nev,
            tol=tol,
            maxit=maxit,
            finite_difference_jacobian=finite_difference_jacobian,
            fd_step=fd_step,
            options=options,
        )
        result = nep.solve(max_eigenpairs=max_eigenpairs)

        eigenpair = np.arange(result.eigenvalues.size, dtype=np.int64)
        dof = np.arange(result.eigenvectors.shape[1], dtype=np.int64)
        return xr.Dataset(
            data_vars={
                "omega": ("eigenpair", result.eigenvalues),
                "eigenvector": (("eigenpair", "dof"), result.eigenvectors),
            },
            coords={
                "eigenpair": eigenpair,
                "dof": dof,
            },
            attrs={
                "ntor": int(ntor),
                "target_omega": target_omega,
                "mmax": None if mmax is None else int(mmax),
                "max_dm": int(max_dm),
                "regularizer": regularizer,
                "apply_bcs": bool(apply_bcs),
                "nep_type": nep_type,
                "which": which,
                "nconv": int(result.nconv),
                "iterations": int(result.iterations),
                "reason": str(result.reason),
                "description": "SLEPc NEP solution of sparse solver2 T(omega) x = 0.",
            },
        )
