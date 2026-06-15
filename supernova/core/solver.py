r"""
Main solver for the linear ideal MHD eigenvalue problem.

The problem is a nonlinear eigenvalue problem on the form:

T(\omega) x = 0
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import unyt
from .equilibrium import superEquilibrium
from .continuum import superContinuum
from ..radial.base import RadialFE
from ..radial.profile import Profile

# Let's import the kernels
from ._solver_kernel import apply_to_reduced, solver_blocks, eval_M_at, build_solver_blocks
from ._antenna import Antenna
from ._boundary import idealSeparatrixBC, couplingPressureDisplacementBC, regularityAxisBC
from ._petsc_helpers import KrylovSolver
# Let's import the PETSc interface for the Krylov inversion.


class superSolver:
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
                       '(mu_0*J_B - shear*gpsi_psi) * shear / B0**2',  # C12 - third part              - 2
                       'B0**2/gpsi_psi/h',    # The O operator                                    - 3
                       '- divPsiVec', # C22 - Precomputed.                                        - 4
                       '(shear*gpsi_psi - J_B)/h',  # D11                                         - 5
                       'kappa_s',  # D21 - first part; distinct from the continuum K operator.    - 6
                       '1/B0**2', # F11 second part.                                              - 7
                       'gpsi_psi/B0**2 * shear', # F12 - first part                               - 8
                       '- J_B/B0**2', # F12 - second part                                         - 9
                       '-1/B0**4*h', # F21                                                        - 10
                       '-kappa_Psi/(gpsi_psi*B0**2)*h', # F22                                     - 11
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

    def build_solver_blocks(self, ntor: int, mmax: int=None):
        """
        We build the solver blocks using the pre-evaluated profiles already
        evaluated at the radial points and fill the class to be used later 
        in a faster assembly.
        """
        mmax_eff = self._solver_mmax(mmax)
        nmax_eff = 0

        # Let's check if we can reuse the previous solver blocks.
        if hasattr(self, 'solver_blocks'):
            if self.solver_blocks.ntor == ntor and self.solver_blocks.mmax == mmax_eff:
                print("Reusing previous solver blocks.")
                return self.solver_blocks
            else:
                print("Building new solver blocks for ntor =", ntor, "and mmax =", mmax_eff)

        H = self._profile_harmonics
        S = self._profile_scalar
        mass_density = S(self.density)# * self.cont.mass

        self.solver_blocks = build_solver_blocks(H(self.C11, mmax_eff, nmax_eff),
                                                 H(self.O, mmax_eff, nmax_eff),
                                                 H(self.C12_2, mmax_eff, nmax_eff),
                                                 H(self.C12_3, mmax_eff, nmax_eff),
                                                 H(self.C22, mmax_eff, nmax_eff),
                                                 H(self.D11, mmax_eff, nmax_eff),
                                                 H(self.C11, mmax_eff, nmax_eff),
                                                 H(self.D21_1, mmax_eff, nmax_eff),
                                                 H(self.K, mmax_eff, nmax_eff),
                                                 H(self.M, mmax_eff, nmax_eff),
                                                 H(self.F11_2, mmax_eff, nmax_eff),
                                                 H(self.F12_1, mmax_eff, nmax_eff),
                                                 H(self.F12_2, mmax_eff, nmax_eff),
                                                 H(self.F21, mmax_eff, nmax_eff),
                                                 H(self.F22, mmax_eff, nmax_eff),
                                                 S(self.hprof),
                                                 S(self.Iprof),
                                                 S(self.gprof),
                                                 S(self.qprof),
                                                 S(self.pressure),
                                                 S(self.pressure_deriv),
                                                 mass_density,
                                                 self.gamma_s,
                                                 H(self.inv_gpsi_psi, mmax_eff, nmax_eff),
                                                 H(self.gpsi_theta_over_gpsi_psi, mmax_eff, nmax_eff), 
                                                 H(self.gpsi_zeta_over_gpsi_psi, mmax_eff, nmax_eff),
                                                 ntor=ntor)
        self.solver_blocks.mmax = mmax_eff
        
        # We will now solve the continuum problem for the first time to 
        # get the SVD decomposition for the E^-1 operator for the ntor given.
        w, vdict = self.cont.solve(ntor=ntor, mmax=mmax_eff, 
                                   psi_vals=self.unique_points,
                                   which='full', return_eigvecs=True, 
                                   normalize_eigvals=False)
        self.solver_blocks.vr = vdict['right']
        self.solver_blocks.vl = vdict['left']
        self.solver_blocks.norm = vdict['normalizations']
        self.solver_blocks.cont_eigvals = w

        # We fill the elements related to the radial operations.
        self.solver_blocks.weights_q = self.weights
        self.solver_blocks.points_q = self.points
        self.solver_blocks.phi_q = self.radialfe.eval_basis(self.points).T
        self.solver_blocks.WD = self.radialfe.weak_derivative_matrix.T

        return self.solver_blocks

    def get_response(self, antenna_profile: Profile, 
                     frequency: np.ndarray[float],
                     ntor: int,  mmax: int=None,
                     regularizer: float=None) -> xr.Dataset:
        """
        For a given antenna profile and frequency, we compute the response of the 
        ideal MHD system to the antenna drive.

        Parameters
        ----------
        antenna_profile : Profile
            The radial profile of the antenna drive.
        frequency : np.ndarray[float]
            The frequencies at which to compute the response.
        mmax : int, optional
            The maximum poloidal mode number to consider in the response. 
            If None, all modes will be considered.
        regularizer : float, optional
            Lambda-space continuum broadening. With the exp(-i omega t)
            convention, positive values damp the continuum pole.
        """
        omega = np.asarray(frequency) * 2 * np.pi
        if np.any(np.isclose(frequency, 0)):
            raise ValueError("Frequency cannot be zero for the response solver.")
        if np.any(np.iscomplex(frequency)):
            raise ValueError("Frequency cannot be complex for the response solver.")
        
        # We will force the antenna profile to be only m=0, n=0.
        if antenna_profile.m_pol.size > 1 or antenna_profile.n_tor.size > 1:
            raise ValueError("Antenna profile must be only m=0, n=0 for the response solver.")
        if antenna_profile.m_pol[0] != 0 or antenna_profile.n_tor[0] != 0:
            raise ValueError("Antenna profile must be only m=0, n=0 for the response solver.")
        
        # Generating the mpol range.
        mmax = self._solver_mmax(mmax)
        m_range = np.arange(-mmax, mmax+1)
        n_range = np.ones_like(m_range) * ntor
        nmpol = len(m_range)
        nstates = 2 * nmpol # p1 and xi for each mpol.
        nq = self.radialfe.get_ndofs()
        n_response_points = self.points.shape[0]

        # Preparing the results arrays.
        response_x = np.zeros((len(omega), n_response_points, nstates, 4), dtype=np.complex128)

        # We will integrate the response for every omega, for every antenna drive.
        # There will be two approaches:
        # Measuring the response to the antenna drive using perpendicular displacement
        # terms and the pressure perturbation.
        integrated_response = np.zeros((len(omega), 4, 2), dtype=np.complex128)

        # The integrated response for the xi^\psi component is:
        #
        # I(\omega) = \omega^2/2\sum_{nm} a_{nm}^H @ M @ a_{nm}  
        #
        # where a_{nm} are the coefficients of the xi^\psi, M is the 
        # mass matrix (identical for all m-modes) and the sum is performed
        # over all modes. For the pressure perturbation, we will use 
        # RMS version of the response (the energy integral will give 0 for n!=0 and m!=0 modes :( )
        # I(\omega) = \sqrt{\sum_{nm} a_{nm}^H @ M @ a_{nm}}.
        # We will add some normalization factors to these integrated responses later 
        # to make them more interpretable and comparable across different equilibria 
        # and parameters.

        # Let's generate the solver_blocks for our ntor and mmax.
        sb = self.build_solver_blocks(ntor=ntor, mmax=mmax)

        # We also prebuild the frequency-independent boundary conditions.
        bc1 = regularityAxisBC(self.radialfe, m_range)
        bc2 = idealSeparatrixBC(self.radialfe, m_range.size)
        bc3 = bc1 * bc2

        # For the antenna driven case, we will generate four independent antenna,
        # two for the p1 component and two for the xi component, with different parities,
        # i.e., odd and even. This will allow us to capture the response of the system 
        # to different antenna drives and to check the effect of the parity on the response.
        antP1_even = Antenna(rho=antenna_profile.x, 
                            profile=antenna_profile.data.values, 
                            radial_fe=self.radialfe,
                            bc=bc3, which='p1', parity='even')
        antP1_odd = Antenna(rho=antenna_profile.x, 
                            profile=antenna_profile.data.values, 
                            radial_fe=self.radialfe,
                            bc=bc3, which='p1', parity='odd')
        antXi_even = Antenna(rho=antenna_profile.x, 
                            profile=antenna_profile.data.values, 
                            radial_fe=self.radialfe,
                            bc=bc3, which='xi', parity='even')
        antXi_odd = Antenna(rho=antenna_profile.x, 
                            profile=antenna_profile.data.values, 
                            radial_fe=self.radialfe,
                            bc=bc3, which='xi', parity='odd')
        b_elements = [antP1_even.get_full_pattern(mmax=mmax),
                      antP1_odd.get_full_pattern(mmax=mmax),
                      antXi_even.get_full_pattern(mmax=mmax),
                      antXi_odd.get_full_pattern(mmax=mmax)]
        
        krylov = None # Dummy.

        # Getting a lazy reference to the mass matrix.
        massmatrix = self.radialfe.mass_matrix
        xicols = np.arange(nmpol, 2*nmpol)
        p1cols = np.arange(0, nmpol)

        for ii, iomega in enumerate(omega):
            print("Solving for {ii} frequency: {freq:.3f} Hz".format(ii=ii, freq=frequency[ii]))
            # We build the pressure coupling operator for the given frequency.
            M = eval_M_at(sb=sb, omega=iomega, regularizer=regularizer)
            nq_operator = sb.weights_q.shape[0]
            if M.shape[0] != nq_operator + 1:
                raise ValueError(
                    "M must be evaluated at the quadrature points plus one edge point "
                    f"for the boundary condition. Got {M.shape[0]} points and "
                    f"{nq_operator} quadrature weights."
                )
            M_q = M[:nq_operator]
            M_edge = M[-1]

            # From the local M matrix at the edge, we build the last boundary condition.
            bc4 = couplingPressureDisplacementBC(self.radialfe, m_range.size, M=M_edge)
            bc = bc3 * bc4 # Total boundary condition.

            # We reduced the b_elements to the boundary conditions.
            b_reduced = [bc.restrict(be) for be in b_elements]
            context = {'sb': sb, 'M': M_q, 'bc': bc}

            if krylov is None:
                krylov = KrylovSolver(size=bc.get_free_dofs(), 
                                      operator=apply_to_reduced,
                                      context=context, rtol=1e-3, atol=1e-5, maxit=100)
            else:
                krylov.swap_context(operator=apply_to_reduced, 
                                    context=context,
                                    reuse_solution=True)
                
            # We will now solve the linear system for each of the four independent antenna drives.
            for jj, bre in enumerate(b_reduced):
                print("  Solving for antenna drive {jj}: {drive}".format(jj=jj, drive=['p1_even', 'p1_odd', 'xi_even', 'xi_odd'][jj]))
                response_reduced = krylov.solve(bre)
                response_full = bc.expand(response_reduced)
                response_coeff = response_full.reshape((nq, nstates))
                response_x[ii, :, :, jj] = sb.phi_q @ response_coeff

                # We also compute the integrated response for each case.
                # dotproduct = np.vdot(response_full.flatten(), response_full.flatten())
                # integrated_response[ii, jj] = np.sqrt(np.sum(dotproduct * self.weights[:, None], axis=0))
                xi = response_coeff[:, xicols]
                p1 = response_coeff[:, p1cols]
                xisum = np.vdot(xi, massmatrix @ xi)
                p1sum = np.vdot(p1, massmatrix @ p1)
                integrated_response[ii, jj, 0] = 0.5 * iomega**2 * np.sum(xisum)
                integrated_response[ii, jj, 1] = np.sqrt(np.sum(p1sum))

        # We will add some units to the integrated response to make it interpretable:
        xinorm = self.equ.hprof.data.values[0] * self.equ.qprof.data.values[0]**2 \
                * self.equ.density.data.values[0] / self.equ.Baxis**2 / self.equ.aminor**2
        p1norm = np.sqrt(self.equ.hprof.data.values[0] / self.equ.Baxis**2) / (self.gamma_s - 1)

        integrated_response[:, :, 0] *= xinorm
        integrated_response[:, :, 1] /= p1norm

        # We will return a dataset with all the relevant information.
        dset = xr.Dataset()
        dset['response'] = xr.DataArray(response_x, dims=['frequency', 'radial_point', 'state', 'antenna_drive'],
                                        coords={'frequency': frequency, 
                                                'radial_point': self.points,
                                                'state': np.arange(nstates), 
                                                'antenna_drive': ['p1_even', 'p1_odd', 'xi_even', 'xi_odd']},
                                        attrs={'units': 'arbitrary',
                                               'description': 'Response of the system to the antenna drive at each frequency, evaluated at the radial quadrature points.'})
        dset['integrated_response'] = xr.DataArray(integrated_response, dims=['frequency', 'antenna_drive', 'quantity'],
                                        coords={'frequency': frequency,
                                                'antenna_drive': ['p1_even', 'p1_odd', 'xi_even', 'xi_odd'],
                                                'quantity': ['xi_response', 'p1_response']},
                                        attrs={'units': 'arbitrary',
                                               'description': 'Integrated response of the system to the antenna drive at each frequency.'})
        return dset
