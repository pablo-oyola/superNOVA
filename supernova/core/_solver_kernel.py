r"""
@TODO: We need to introduce sparse-matrix support -
    - Limiting a diagonal coupling, our matrices will diagonal-sparse, which do
      occupy significant less space than the full dense matrices. We can use sparse-matrix 
      libraries to store and operate with them, which will allow us to reach higher mmax and nmax.
    - We can also introduce a cutoff in the coupling, which will make the matrices even more
        sparse, but we need to be careful to not introduce unphysical results by cutting important couplings.
    - Allowing for different toroidal mode number couplings, i.e., stellarator support (long term).
"""

import numpy as np
from .._utils import njit, prange
# from scipy.constants import mu_0
mu_0 = 1.0 # Normalized units.
from dataclasses import dataclass
import xarray as xr

# Some of the kernels are already implemented.
from ._continuum_kernel import (_fill_block_kernel, 
                                _apply_ghg_scaling,
                                get_active_bounds,
                                _fill_convolution_block_kernel,
                                get_harmonic_value,
                                )

from ..radial.base import RadialFE
from ._boundary import BoundaryCondition

@dataclass
class solver_blocks:
    """
    Contains the block matrices of the problem, once they are filled and prepared
    at initialization.
    """
    # The X0 contain the parts of the matrices that are not dependent on 
    # the eigenvalue, while the X1 are the parts that are related to the eigenvalue
    # operations. The X1 blocks are only Nstate*Nstate, whereas the 
    # X0 blocks are the full (2*Nstate)*(2*Nstate) blocks.
    m_range: np.ndarray
    ntor: int
    C0: np.ndarray
    C1: np.ndarray # C = C0 + C1 * lambda

    D0: np.ndarray
    D1: np.ndarray # D = D0 + D1 / lambda

    F: np.ndarray

    # Metric parts for the phase.
    gtp_over_gpp: np.ndarray # g^{\psi\theta}/g^{\psi\psi}
    gpz_over_gpp: np.ndarray # g^{\psi\zeta}/g^{\psi\psi}

    # Profiles.
    hprof: np.ndarray
    Iprof: np.ndarray
    gprof: np.ndarray
    qprof: np.ndarray
    pressure: np.ndarray
    pressure_deriv: np.ndarray
    density: np.ndarray # Mass density used in lambda = mu_0 * rho_mass * omega**2.

    # The following are the blocks related to the continuum.
    vr: np.ndarray # Right eigenvectors of the continuum.
    vl: np.ndarray # Left eigenvectors of the continuum.
    norm: np.ndarray # Eigenvalue normalization vl^H @ B @ vr.
    cont_eigvals: np.ndarray # Unnormalized continuum lambdas.

    # And finally some elements related to the radialFE.
    weights_q: np.ndarray # Quadrature weights.
    points_q: np.ndarray # Quadrature points.
    phi_q: np.ndarray # Radial basis functions evaluated at the quadrature points.
    WD: np.ndarray # Weak derivative matrix evaluated at the quadrature points.

    def to_xarray(self):
        """
        Builds an xarray dataset with the blocks and profiles, that can be used for
        analysis and plotting, and maybe for loading.
        """

        n_x = self.C0.shape[0]
        n_harm = self.m_range.size
        n_full = 2 * n_harm
        m_range = np.asarray(self.m_range)
        full_state_m = np.concatenate([m_range, m_range])
        full_state_component = np.array(["p1"] * n_harm + ["xi"] * n_harm)
        full_state = np.array(
            [f"{component}_m={m}" for component, m in zip(full_state_component, full_state_m)]
        )

        x = np.arange(n_x, dtype=np.int64)
        point_kind = np.array(["operator"] * n_x)
        if self.points_q is not None:
            points_q = np.asarray(self.points_q)
            if points_q.ndim != 1:
                raise ValueError("points_q must be a one-dimensional array.")
            if points_q.size == n_x:
                x = points_q
                point_kind = np.array(["quadrature"] * n_x)
            elif points_q.size + 1 == n_x:
                x = np.concatenate([points_q, np.array([1.0], dtype=points_q.dtype)])
                point_kind = np.array(["quadrature"] * points_q.size + ["boundary"])

        coords = {
            "x": ("x", x),
            "point_kind": ("x", point_kind),
            "m_pol": ("m_pol", m_range),
            "m_pol_row": ("m_pol_row", m_range),
            "m_pol_col": ("m_pol_col", m_range),
            "full_state": ("full_state", full_state),
            "full_state_row": ("full_state_row", full_state),
            "full_state_col": ("full_state_col", full_state),
            "full_state_m": ("full_state", full_state_m),
            "full_state_component": ("full_state", full_state_component),
            "full_state_row_m": ("full_state_row", full_state_m),
            "full_state_row_component": ("full_state_row", full_state_component),
            "full_state_col_m": ("full_state_col", full_state_m),
            "full_state_col_component": ("full_state_col", full_state_component),
            "continuum_mode": ("continuum_mode", np.arange(n_full, dtype=np.int64)),
        }

        if self.points_q is not None:
            coords["x_quad"] = ("x_quad", np.asarray(self.points_q))
        elif self.weights_q is not None:
            coords["x_quad"] = ("x_quad", np.arange(np.asarray(self.weights_q).shape[0], dtype=np.int64))
        elif self.phi_q is not None:
            coords["x_quad"] = ("x_quad", np.arange(np.asarray(self.phi_q).shape[0], dtype=np.int64))

        if self.phi_q is not None:
            n_radial = np.asarray(self.phi_q).shape[1]
        elif self.WD is not None:
            n_radial = np.asarray(self.WD).shape[0]
        else:
            n_radial = None
        if n_radial is not None:
            radial_dof = np.arange(n_radial, dtype=np.int64)
            coords["radial_dof"] = ("radial_dof", radial_dof)
            coords["radial_dof_row"] = ("radial_dof_row", radial_dof)
            coords["radial_dof_col"] = ("radial_dof_col", radial_dof)

        sizes = {
            "x": n_x,
            "m_pol": n_harm,
            "m_pol_row": n_harm,
            "m_pol_col": n_harm,
            "full_state": n_full,
            "full_state_row": n_full,
            "full_state_col": n_full,
            "continuum_mode": n_full,
        }
        if "x_quad" in coords:
            sizes["x_quad"] = coords["x_quad"][1].size
        if n_radial is not None:
            sizes["radial_dof"] = n_radial
            sizes["radial_dof_row"] = n_radial
            sizes["radial_dof_col"] = n_radial

        data_vars = {}

        def add_var(name, value, dims, attrs=None):
            if value is None:
                return
            data = np.asarray(value)
            expected_shape = tuple(sizes[dim] for dim in dims)
            if data.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {data.shape}.")
            data_vars[name] = (dims, data, attrs or {})

        block_attrs = {"description": "Frequency-independent local solver block."}
        add_var("C0", self.C0, ("x", "full_state_row", "full_state_col"), block_attrs)
        add_var("D0", self.D0, ("x", "full_state_row", "full_state_col"), block_attrs)
        add_var("F", self.F, ("x", "full_state_row", "full_state_col"), block_attrs)
        add_var("C1", self.C1, ("x", "m_pol_row", "m_pol_col"),
                {"description": "Lambda-dependent C12 harmonic block."})
        add_var("D1", self.D1, ("x", "m_pol_row", "m_pol_col"),
                {"description": "Inverse-lambda-dependent D22 harmonic block."})
        add_var("gtp_over_gpp", self.gtp_over_gpp, ("x", "m_pol_row", "m_pol_col"))
        add_var("gpz_over_gpp", self.gpz_over_gpp, ("x", "m_pol_row", "m_pol_col"))

        profile_dims = ("x",)
        add_var("hprof", self.hprof, profile_dims)
        add_var("Iprof", self.Iprof, profile_dims)
        add_var("gprof", self.gprof, profile_dims)
        add_var("qprof", self.qprof, profile_dims)
        add_var("pressure", self.pressure, profile_dims)
        add_var("pressure_deriv", self.pressure_deriv, profile_dims)
        add_var("density", self.density, profile_dims,
                {"description": "Mass density used in lambda = mu_0 * rho_mass * omega**2."})

        add_var("vr", self.vr, ("x", "full_state", "continuum_mode"),
                {"description": "Right continuum eigenvectors, stored by column/mode."})
        add_var("vl", self.vl, ("x", "full_state", "continuum_mode"),
                {"description": "Left continuum eigenvectors, stored by column/mode."})
        add_var("norm", self.norm, ("x", "continuum_mode"),
                {"description": "Biorthogonal normalization vl^H B vr per continuum mode."})
        add_var("cont_eigvals", self.cont_eigvals, ("x", "continuum_mode"),
                {"description": "Unnormalized continuum generalized eigenvalues."})

        add_var("weights_q", self.weights_q, ("x_quad",),
                {"description": "Quadrature weights for radial integrals."})
        add_var("phi_q", self.phi_q, ("x_quad", "radial_dof"),
                {"description": "Radial basis functions evaluated at quadrature points."})
        add_var("WD", self.WD, ("radial_dof_row", "radial_dof_col"),
                {"description": "Weak derivative matrix."})
        add_var("points_q", self.points_q, ("x_quad",),
                {"description": "Quadrature points for radial integrals."})

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                "ntor": int(self.ntor),
                "mmax": int(getattr(self, "mmax", np.max(np.abs(m_range)))),
                "state_order": "full_state is ordered as p1(m_range) followed by xi(m_range).",
                "description": "Serialized solver_blocks data for solver debugging and persistence.",
            },
        )
    
    @staticmethod
    def from_xarray(dataset: xr.Dataset) -> "solver_blocks":
        """
        Reconstructs a solver_blocks instance from an xarray dataset created by to_xarray.

        Parameters
        ----------
        dataset : xr.Dataset
            The xarray dataset containing the serialized solver_blocks data.

        Returns
        -------
        solver_blocks
            The reconstructed solver_blocks instance.
        """
        # We will extract the variables and coordinates from the dataset and use them to construct the solver_blocks instance.
        C0 = dataset["C0"].values if "C0" in dataset else None
        D0 = dataset["D0"].values if "D0" in dataset else None
        F = dataset["F"].values if "F" in dataset else None
        C1 = dataset["C1"].values if "C1" in dataset else None
        D1 = dataset["D1"].values if "D1" in dataset else None
        gtp_over_gpp = dataset["gtp_over_gpp"].values if "gtp_over_gpp" in dataset else None
        gpz_over_gpp = dataset["gpz_over_gpp"].values if "gpz_over_gpp" in dataset else None

        hprof = dataset["hprof"].values if "hprof" in dataset else None
        Iprof = dataset["Iprof"].values if "Iprof" in dataset else None
        gprof = dataset["gprof"].values if "gprof" in dataset else None
        qprof = dataset["qprof"].values if "qprof" in dataset else None
        pressure = dataset["pressure"].values if "pressure" in dataset else None
        pressure_deriv = dataset["pressure_deriv"].values if "pressure_deriv" in dataset else None
        density = dataset["density"].values if "density" in dataset else None

        vr = dataset["vr"].values if "vr" in dataset else None
        vl = dataset["vl"].values if "vl" in dataset else None
        norm = dataset["norm"].values if "norm" in dataset else None
        cont_eigvals = dataset["cont_eigvals"].values if "cont_eigvals" in dataset else None

        weights_q = dataset["weights_q"].values if "weights_q" in dataset else None
        phi_q = dataset["phi_q"].values if "phi_q" in dataset else None
        WD = dataset["WD"].values if "WD" in dataset else None
        m_range = dataset.coords["m_pol"].values if "m_pol" in dataset.coords else None
        ntor = int(dataset.attrs["ntor"]) if "ntor" in dataset.attrs else None

        return solver_blocks(
            C0=C0,
            D0=D0,
            F=F,
            C1=C1,
            D1=D1,
            gtp_over_gpp=gtp_over_gpp,
            gpz_over_gpp=gpz_over_gpp,
            hprof=hprof,
            Iprof=Iprof,
            gprof=gprof,
            qprof=qprof,
            pressure=pressure,
            pressure_deriv=pressure_deriv,
            density=density,
            vr=vr,
            vl=vl,
            norm=norm,
            cont_eigvals=cont_eigvals,
            weights_q=weights_q,
            phi_q=phi_q,
            WD=WD,
            m_range=m_range,
            ntor=ntor,
            points_q=dataset.coords["x_quad"].values if "x_quad" in dataset.coords else None,
        )

# -----------------------------------------------------------------------------
# General blocks building kernels.
# -----------------------------------------------------------------------------
# We recompile the continuum kernels here to make them serial.
fill_block_kernel = njit(parallel=False, cache=True, nogil=True)(_fill_block_kernel)
apply_ghg_scaling = njit(parallel=False, cache=True, nogil=True, inline='always')(_apply_ghg_scaling)
fill_convolution_block_kernel = njit(parallel=False, cache=True, nogil=True, inline='always')(_fill_convolution_block_kernel)

@njit(cache=True, nogil=True, inline="always")
def scale_rows(matrix: np.ndarray, diag: np.ndarray, conj: bool=False):
    """
    Multiply each matrix row by the corresponding diagonal factor in place.

    This is a small Numba helper for left multiplication by a diagonal matrix.
    After the call, ``matrix[i, j]`` has been multiplied by ``diag[i]`` for all
    columns ``j``. The operation mutates ``matrix`` and returns nothing.

    It will be used to apply the following operation:
    G @ H, where G = diag(g) and H is the original matrix.

    The `conj` flag can be set to evaluate:

    G^H @ H, where G^H = diag(g^*) is the Hermitian conjugate of G. 
    In this case, each row will be multiplied by the complex 
    conjugate of the corresponding diagonal factor.

    Parameters
    ----------
    matrix : ndarray
        Two-dimensional complex matrix to mutate.
    diag : ndarray
        One-dimensional row-scaling factors with length ``matrix.shape[0]``.
    conj : bool, optional
        If True, use the complex conjugate of the diagonal factors. 
        Default is False.
    """
    for i in range(matrix.shape[0]):
        factor = diag[i]
        if conj:
            factor = np.conj(factor)
        for j in range(matrix.shape[1]):
            matrix[i, j] *= factor

@njit(cache=True, nogil=True, inline="always")
def scale_cols(matrix: np.ndarray, diag: np.ndarray, conj: bool=False):
    """
    Multiply each matrix row by the corresponding diagonal factor in place.

    This is a small Numba helper for left multiplication by a diagonal matrix.
    After the call, ``matrix[i, j]`` has been multiplied by ``diag[i]`` for all
    columns ``j``. The operation mutates ``matrix`` and returns nothing.

    It will be used to apply the following operation:
    H @ G, where G = diag(g) and H is the original matrix.

    The `conj` flag can be set to evaluate:

    H @ G^H, where G^H = diag(g^*) is the Hermitian conjugate of G. 
    In this case, each row will be multiplied by the complex 
    conjugate of the corresponding diagonal factor.

    Parameters
    ----------
    matrix : ndarray
        Two-dimensional complex matrix to mutate.
    diag : ndarray
        One-dimensional row-scaling factors with length ``matrix.shape[0]``.
    conj : bool, optional
        If True, use the complex conjugate of the diagonal factors. 
        Default is False.
    """
    for j in range(matrix.shape[1]):
        factor = diag[j]
        if conj:
            factor = np.conj(factor)
        for i in range(matrix.shape[0]):
            matrix[i, j] *= factor

# -----------------------------------------------------------------------------
# Static blocks building kernels.
# -----------------------------------------------------------------------------

@njit(parallel=True, cache=True, nogil=True)
def _build_static_blocks(
    C11: np.ndarray, O: np.ndarray, C12_2: np.ndarray, C12_3: np.ndarray, C22: np.ndarray,
    D11: np.ndarray, D12: np.ndarray, kappa_s: np.ndarray,
    K: np.ndarray, M: np.ndarray, 
    F11_2: np.ndarray, F12_1: np.ndarray, F12_2: np.ndarray, F21: np.ndarray, F22: np.ndarray,
    hprof: np.ndarray, Iprof: np.ndarray, gprof: np.ndarray, qprof: np.ndarray,
    pressure: np.ndarray, pressure_deriv: np.ndarray, density: np.ndarray, gamma_s: float,
    one_gpp: np.ndarray, gtp_over_gpp: np.ndarray, gpp_over_gpp: np.ndarray,
    ntor: int
):
    r"""
    Build the static blocks of the C, D and F matrices.

    Parameters
    ----------
    C11 : ndarray [npsi, mpol, ntor]
        The C11 profile of the C matrix, which is static and does not depend on the
        eigenvalue. 2K_\Psi/g^{\psi\psi}.
    """
    # Dimensions of the problem. We will need these to allocate the relevant matrices and blocks.
    assert C11.ndim == 3, "C11 must have shape (npsi, mpol, ntor)"
    nstate = (C11.shape[1] - 1) * 2 + 1 # m=0 is only one state, m>0 are two states.
    mpol_max = C11.shape[1] - 1
    npsi = C11.shape[0]
    m_range = np.arange(-mpol_max, mpol_max + 1)
    n_range = np.zeros_like(m_range) + ntor

    # We will check the shape of all inputs.
    assert O.shape == C11.shape, "O must have shape (npsi, mpol, ntor)"
    assert C12_2.shape == C11.shape, "C12_2 must have shape (npsi, mpol, ntor)"
    assert C12_3.shape == C11.shape, "C12_3 must have shape (npsi, mpol, ntor)"
    assert C22.shape == C11.shape, "C22 must have shape (npsi, mpol, ntor)"
    assert D11.shape == C11.shape, "D11 must have shape (npsi, mpol, ntor)"
    assert D12.shape == C11.shape, "D12 must have shape (npsi, mpol, ntor)"
    assert kappa_s.shape == C11.shape, "kappa_s must have shape (npsi, mpol, ntor)"
    assert K.shape == C11.shape, "K must have shape (npsi, mpol, ntor)"
    assert M.shape == C11.shape, "M must have shape (npsi, mpol, ntor)"
    assert F11_2.shape == C11.shape, "F11_2 must have shape (npsi, mpol, ntor)"
    assert F12_1.shape == C11.shape, "F12_1 must have shape (npsi, mpol, ntor)"
    assert F12_2.shape == C11.shape, "F12_2 must have shape (npsi, mpol, ntor)"
    assert F21.shape == C11.shape, "F21 must have shape (npsi, mpol, ntor)"
    assert F22.shape == C11.shape, "F22 must have shape (npsi, mpol, ntor)"
    assert hprof.shape == (npsi,), "hprof must have shape (npsi,)"
    assert Iprof.shape == (npsi,), "Iprof must have shape (npsi,)"
    assert gprof.shape == (npsi,), "gprof must have shape (npsi,)"
    assert pressure.shape == (npsi,), "pressure must have shape (npsi,)"
    assert pressure_deriv.shape == (npsi,), "pressure_deriv must have shape (npsi,)"
    assert density.shape == (npsi,), "density must have shape (npsi,)"
    assert one_gpp.shape == C11.shape, "one_gpp must have shape (npsi, mpol, ntor)"
    assert gtp_over_gpp.shape == C11.shape, r"g^{\theta\psi} must have shape (npsi, mpol, ntor)"
    assert gpp_over_gpp.shape == C11.shape, r"g_{\zeta\psi} must have shape (npsi, mpol, ntor)"
    assert gamma_s > 0, "gamma_s must be positive."

    # Allocating the space. 
    C0 = np.zeros((npsi, 2*nstate, 2*nstate), dtype=np.complex128)
    D0 = np.zeros((npsi, 2*nstate, 2*nstate), dtype=np.complex128)
    F  = np.zeros((npsi, 2*nstate, 2*nstate), dtype=np.complex128)
    C1 = np.zeros((npsi, nstate, nstate), dtype=np.complex128)
    D1 = np.zeros((npsi, nstate, nstate), dtype=np.complex128)
    gtp_over_gpp_mat = np.zeros((npsi, nstate, nstate), dtype=np.complex128)
    gzp_over_gpp_mat = np.zeros((npsi, nstate, nstate), dtype=np.complex128)

    # Some preparatory elements.
    diagonal_ones = np.eye(nstate, dtype=np.complex128)

    for ipsi in prange(npsi):
        # We place first into the C0, D0 and F matrices the blocks that do
        # not require convolution or extra operations.
        # C11 and C22 elements go directly into the matrix without further ado.
        fill_block_kernel(2.0*C11[ipsi], m_range, n_range, C0[ipsi, :nstate, :nstate])
        fill_block_kernel(C22[ipsi], m_range, n_range, C0[ipsi, nstate:, nstate:])

        # On the element C12, we only place the C12_3 part and then we add up the 
        # scaled C11 by mu_0 p_0'.
        fill_block_kernel(C12_3[ipsi], m_range, n_range, C0[ipsi, :nstate, nstate:])
        C0[ipsi, :nstate, nstate:] += mu_0 * pressure_deriv[ipsi] * C0[ipsi, :nstate, :nstate]

        # We fill D11 with the D11 profile, and we will add later the contribution of the convolution with the G operator.
        fill_block_kernel(D11[ipsi], m_range, n_range, D0[ipsi, :nstate, :nstate])

        # D12 element only needs to be multiplied later by 2\mu_0\gamma_s p_0 
        # which is a local value.
        fill_block_kernel(D12[ipsi], m_range, n_range, D0[ipsi, :nstate, nstate:])
        D0[ipsi, :nstate, nstate:] *= 2 * mu_0 * gamma_s * pressure[ipsi]

        # D21 has two pieces: one that is directly the kappa_s and the other one is 
        # the operator \hat\sigma/h(\Psi) applied to the eigenvector.
        fill_block_kernel(2.0*kappa_s[ipsi], m_range, n_range, D0[ipsi, nstate:, :nstate])

        # D22 has a 1 + \beta elements on it. The beta elements depend on the
        # eigenvalue, so we will place the 1 part here and the rest later.
        D0[ipsi, nstate:, nstate:] = diagonal_ones

        # F11 element is composed by -K + \hat\sigma/B^2. We just place now the
        # K part, and we will add the rest later.
        kmat = np.zeros((nstate, nstate), dtype=np.complex128)
        fill_block_kernel(K[ipsi], m_range, n_range, kmat)
        F[ipsi, :nstate, :nstate] = -kmat
        one_over_B2 = np.zeros((nstate, nstate), dtype=np.complex128)
        fill_block_kernel(F11_2[ipsi], m_range, n_range, one_over_B2)

        # F21 and F22 are directly the F21 and F22 profiles, so we can place them directly.
        fill_block_kernel(F21[ipsi], m_range, n_range, F[ipsi, nstate:, :nstate])
        fill_block_kernel(F22[ipsi], m_range, n_range, F[ipsi, nstate:, nstate:])

        # We create the diagonal representations of the G and sigma operators.
        g_diag = np.zeros(nstate, dtype=np.complex128)
        sigma_diag = np.zeros(nstate, dtype=np.complex128)
        for i in range(nstate):
            g_diag[i] = i*(qprof[ipsi] * n_range[i] - m_range[i])
            sigma_diag[i] = i * (Iprof[ipsi] * n_range[i] - gprof[ipsi] * m_range[i])

        # Now we create and fill other matrices.
        c12_2mat = np.zeros((nstate, nstate), dtype=np.complex128)
        fill_block_kernel(C12_2[ipsi], m_range, n_range, c12_2mat)
        omat = np.zeros((nstate, nstate), dtype=np.complex128)
        fill_block_kernel(O[ipsi], m_range, n_range, omat)
        apply_ghg_scaling(omat, qprof[ipsi], n_range, m_range) # G^H @ O @ G.

        # Now, we can get C12 contribution by convoluting the G^H @ O @ G with the C12_2 factor on the left.
        C0[ipsi, :nstate, nstate:] += c12_2mat @ omat # C12 is done!.

        # D11 is the D11 input convoluted with the G operator on the right.
        # D11 = D11 @ G. Since G is a diagonal operator, we can just scale
        # the columns of D11
        scale_cols(D0[ipsi, :nstate, :nstate], g_diag)

        # We now scale the sigma for the D21 contribution.
        D21_2mat = np.zeros((nstate, nstate), dtype=np.complex128)
        one_over_h = np.zeros((C11.shape[1], C11.shape[2]), dtype=np.complex128)
        one_over_h[0, 0] = -1.0 / hprof[ipsi]
        fill_block_kernel(one_over_h, m_range, n_range, D21_2mat) 
        # We fill D21_2 with a scalar zero-mode, since we will just scale it by the sigma_diag.
        scale_cols(D21_2mat, sigma_diag) # D21_2 = D
        D0[ipsi, nstate:, :nstate] += D21_2mat # We add the D21_2 contribution to D21.

        # -sigma/B²
        scale_cols(one_over_B2, sigma_diag)
        F[ipsi, :nstate, :nstate] += one_over_B2 # We add the 1/B^2 part to F11, which is the part that does

        # We build now F12 contributions.
        f12_1mat = np.zeros((nstate, nstate), dtype=np.complex128)
        f12_2mat = np.zeros((nstate, nstate), dtype=np.complex128)
        fill_block_kernel(F12_1[ipsi], m_range, n_range, f12_1mat)
        fill_block_kernel(F12_2[ipsi], m_range, n_range, f12_2mat)
        scale_rows(f12_1mat, g_diag) # G @ F12_1
        scale_cols(f12_2mat, g_diag) # F12_2 @ G

        # We add the F12 contributions, and we also add the contribution of 
        # the convolution of K with the 2\mu_0\gamma_s p_0 factor.
        F[ipsi, :nstate, nstate:] += f12_1mat + f12_2mat \
            - kmat * mu_0 * pressure_deriv[ipsi] 

        # Filling the C1 and D1 matrices, that will have a dependence on the eigenvalue.
        fill_block_kernel(one_gpp[ipsi], m_range, n_range, C1[ipsi])
        fill_block_kernel(M[ipsi], m_range, n_range, D1[ipsi])
        apply_ghg_scaling(D1[ipsi], qprof[ipsi], n_range, m_range) # D1 = G^H @ M @ G.
        D1[ipsi] *= mu_0 * gamma_s * pressure[ipsi] # We scale D1 by the 2\mu_0\gamma_s p_0 factor.

        # And finally we fill the gpt_over_gpp and gpz_over_gpp matrices, that will be used to apply the convolution with the G operator in the solver.+
        fill_block_kernel(gtp_over_gpp[ipsi], m_range, n_range, gtp_over_gpp_mat[ipsi])
        fill_block_kernel(gpp_over_gpp[ipsi], m_range, n_range, gzp_over_gpp_mat[ipsi])

        # We need now to scale the gtp,gzp_over_gpp by the appropriate factors, 
        # im and in to get the correct scaling.
        scale_cols(gtp_over_gpp_mat[ipsi], -1j*m_range) # g^{\theta\psi}/g^{\psi\psi} has an im factor.
        scale_cols(gzp_over_gpp_mat[ipsi],  1j*n_range) # g^{\zeta\psi}/g^{\psi\psi} has an in factor.

    return C0, C1, D0, D1, F, gtp_over_gpp_mat, gzp_over_gpp_mat

def build_solver_blocks(
    C11: np.ndarray, O: np.ndarray, C12_2: np.ndarray, C12_3: np.ndarray, C22: np.ndarray,
    D11: np.ndarray, D12: np.ndarray, kappa_s: np.ndarray,
    K: np.ndarray, M: np.ndarray, 
    F11_2: np.ndarray, F12_1: np.ndarray, F12_2: np.ndarray, F21: np.ndarray, F22: np.ndarray,
    hprof: np.ndarray, Iprof: np.ndarray, gprof: np.ndarray, qprof: np.ndarray,
    pressure: np.ndarray, pressure_deriv: np.ndarray, density: np.ndarray, gamma_s: float,
    one_gpp: np.ndarray, gtp_over_gpp: np.ndarray, gpp_over_gpp: np.ndarray,
    ntor: int
) -> solver_blocks:
    r"""
    Intermediary function that will call the kernel to build the blocks,
    and then return the dataclass with the filled frozen blocks.
    """
    tmp = _build_static_blocks(C11=C11, O=O, C12_2=C12_2, C12_3=C12_3,
                               C22=C22, D11=D11, D12=D12, kappa_s=kappa_s,
                               K=K, M=M, F11_2=F11_2, F12_1=F12_1, F12_2=F12_2, 
                               F21=F21, F22=F22, 
                               hprof=hprof, Iprof=Iprof, gprof=gprof, qprof=qprof,
                               pressure=pressure, pressure_deriv=pressure_deriv, density=density, gamma_s=gamma_s,
                               one_gpp=one_gpp, gtp_over_gpp=gtp_over_gpp, gpp_over_gpp=gpp_over_gpp, ntor=ntor)
    C0, C1, D0, D1, F, gtp_over_gpp_mat, gzp_over_gpp_mat = tmp

    # We create the m_range array, that will be used in the solver to 
    # identify the m values of each state.
    mpol_max = C11.shape[1] - 1
    m_range = np.arange(-mpol_max, mpol_max + 1)

    out = solver_blocks(m_range=m_range, ntor=ntor, C0=C0, C1=C1, D0=D0, D1=D1, F=F,
                        gtp_over_gpp=gtp_over_gpp_mat, gpz_over_gpp=gzp_over_gpp_mat,
                        hprof=hprof, Iprof=Iprof, gprof=gprof, qprof=qprof, 
                        pressure=pressure, pressure_deriv=pressure_deriv, density=density,
                        vr=None, vl=None, norm=None, cont_eigvals=None,
                        weights_q=None, points_q=None, phi_q=None, WD=None)
    return out

# -----------------------------------------------------------------------------
# Continuum inversion kernels.
# -----------------------------------------------------------------------------
# @njit(parallel=True, cache=True, nogil=True)
def _build_inverse_E(vr: np.ndarray, vl: np.ndarray, norm: np.ndarray,
                     omega_cont: np.ndarray, omega: float, density: np.ndarray,
                     regularizer: float=None):
    r"""
    Build the E^-1 operator (the continuum operator) using spectral inversion
    and the right-and-left eigenvectors of the continuum.

    E^-1 = \sum_k (1/(\lambda_k - \lambda)) * (vr_k @ vl_k^H) / norm_k, where

    - vr_k is the k-th right eigenvector of the continuum.
    - vl_k is the k-th left eigenvector of the continuum.
    - norm_k is the normalization factor vl_k^H @ B @ vr_k, where B is the 
    orthogonality matrix of the continuum.

    The continuum eigenvalues passed by ``superContinuum.solve`` with
    ``normalize_eigvals=False`` are already the generalized lambdas. Only the
    target frequency is converted to ``\lambda = \mu_0 \rho_m \omega^2`` here.

    For the code convention ``y(x, t) = y(x) exp(-i omega t)``, a positive
    regularizer is damping. It applies the denominator
    ``\lambda_k - \lambda - i eta``, which places the pole below the real
    omega axis. The regularizer is therefore a lambda-space broadening.

    The A and B matrices actually correspond to the following problem

    E' = A - lambda B,

    but the E operator that we have in our problem is the following instead:


    E' = (1         0)
         (0   \lambda) @ E.

    E^-1 = E'^-1 @ (1         0)
                   (0   \lambda)

    which is equivalent to scaling the whole block-column by lambda.

    Parameters
    ----------
    vr : ndarray
        Right eigenvectors of the continuum, with shape (n_psi, n_state, n_state).
    vl : ndarray
        Left eigenvectors of the continuum, with shape (n_psi, n_state, n_state).
    norm : ndarray
        Normalization factors for the eigenvectors, with shape (n_psi, n_state).
    omega_cont : ndarray
        Unnormalized continuum generalized eigenvalues, with shape
        (n_psi, n_state). The name is kept for backward compatibility.
    omega : float
        Target angular frequency for which to build the E^-1 operator.
    density : ndarray
        Mass density profile, with shape (n_psi,).
    regularizer : float, optional
        Nonnegative lambda-space damping width.
    """
    assert vr.shape == vl.shape, "vr and vl must have the same shape"
    assert vr.shape[0] == norm.shape[0], "The first dimension of vr and norm must match"
    assert vr.shape[1] == norm.shape[1], "The second dimension of vr and norm must match"
    assert vr.shape[0] == omega_cont.shape[0], "The first dimension of vr and omega_cont must match"
    assert vr.shape[1] == omega_cont.shape[1], "The second dimension of vr and omega_cont must match"
    assert density.shape[0] == vr.shape[0], "The first dimension of density and vr must match"

    n_psi, n_state, _ = vr.shape
    n_harm = n_state // 2
    E_inv = np.zeros((n_psi, n_state, n_state), dtype=np.complex128)

    lambda_cont = np.asarray(omega_cont, dtype=np.complex128)
    lambda_target = mu_0 * density * omega**2

    if regularizer is None:
        eta = 0.0
    else:
        eta = np.abs(np.real(regularizer))

    for ipsi in prange(n_psi):
        for k in range(n_state):
            # We compute the contribution of the k-th eigenmode to E^-1.
            denominator = lambda_cont[ipsi, k] - lambda_target[ipsi] - 1j * eta
            contribution = (1 / denominator) * np.outer(vr[ipsi, :, k], np.conj(vl[ipsi, :, k])) / norm[ipsi, k]
            E_inv[ipsi] += contribution

        # We scale the second block-column by lambda, since E^-1 = E'^-1 @ (1 0; 0 lambda).
        E_inv[ipsi, :, n_harm:] *= lambda_target[ipsi]

    return E_inv

@njit(parallel=True, cache=True, nogil=True)
def _apply_E_inv(vr: np.ndarray, vl: np.ndarray, norm: np.ndarray,
                 omega_cont: np.ndarray, omega: float, density: np.ndarray,
                 D: np.ndarray, F: np.ndarray,
                 regularizer: float = None):
    """
    Compute

        D @ E^{-1} @ F

    without explicitly building E^{-1}.

    This preserves the convention

        denominator = lambda_k - lambda_target - i eta

    and applies

        E^{-1} = E'^{-1} @ diag(I, lambda I)

    by scaling the second block rows of F before applying E'^{-1}.
    """
    n_psi = vr.shape[0]
    n_state = vr.shape[1]
    n_mode = vr.shape[2]

    n_harm = n_state // 2

    eta = 0.0
    if regularizer is not None:
        eta = abs(regularizer)

    out = np.zeros((n_psi, n_state, n_state), dtype=np.complex128)

    for ipsi in prange(n_psi):
        lambda_target = mu_0 * density[ipsi] * omega * omega

        D_vr_k = np.empty(n_state, dtype=np.complex128)
        vlh_TF_k = np.empty(n_state, dtype=np.complex128)

        for k in range(n_mode):
            # D_vr_k = D @ vr_k
            for i in range(n_state):
                acc = 0.0 + 0.0j
                for a in range(n_state):
                    acc += D[ipsi, i, a] * vr[ipsi, a, k]
                D_vr_k[i] = acc

            # vlh_TF_k = vl_k^H @ T @ F
            #
            # Since T @ F scales the second block rows of F by lambda_target.
            for j in range(n_state):
                acc = 0.0 + 0.0j
                for b in range(n_state):
                    factor = 1.0
                    if b >= n_harm:
                        factor = lambda_target

                    acc += np.conj(vl[ipsi, b, k]) * factor * F[ipsi, b, j]

                vlh_TF_k[j] = acc

            denominator = (
                omega_cont[ipsi, k]
                - lambda_target
                - 1j * eta
            )

            coeff = 1.0 / (denominator * norm[ipsi, k])

            for i in range(n_state):
                left = coeff * D_vr_k[i]
                for j in range(n_state):
                    out[ipsi, i, j] += left * vlh_TF_k[j]

    return out

# -----------------------------------------------------------------------------
# Evaluating the matrix blocks at a given frequency.
# -----------------------------------------------------------------------------
def eval_M_at(sb: solver_blocks, omega: float, regularizer: float=None):
    """
    Using the information in the solver_blocks, we evaluate the MHD operator
    at a given frequency and return the evaluated full-state blocks M(x), that
    will be used to perform the matrix-vector products in the solver.
    """
    assert np.isclose(omega, 0.0) == False, "Omega cannot be zero for the evaluation of the MHD operator."
    n_harm = sb.C1.shape[1]
    n_full = sb.C0.shape[1]
    n_psi = sb.C0.shape[0]
    lambda_target = mu_0 * sb.density * omega**2

    if sb.C0.shape != (n_psi, n_full, n_full):
        raise ValueError("C0 must have shape (n_psi, n_full, n_full).")
    if sb.D0.shape != (n_psi, n_full, n_full):
        raise ValueError("D0 must have shape (n_psi, n_full, n_full).")
    if sb.F.shape != (n_psi, n_full, n_full):
        raise ValueError("F must have shape (n_psi, n_full, n_full).")
    if n_full != 2 * n_harm:
        raise ValueError("Full state size must be twice the harmonic state size.")
    if sb.D1.shape != (n_psi, n_harm, n_harm):
        raise ValueError("D1 must have shape (n_psi, n_harm, n_harm).")
    if sb.gtp_over_gpp.shape != (n_psi, n_harm, n_harm):
        raise ValueError("gpt_over_gpp must have shape (n_psi, n_harm, n_harm).")
    if sb.gpz_over_gpp.shape != (n_psi, n_harm, n_harm):
        raise ValueError("gpz_over_gpp must have shape (n_psi, n_harm, n_harm).")

    # We first obtain the inversion of the continuum operator at the target frequency.
    # E_inv = _build_inverse_E(vr=sb.vr, vl=sb.vl, norm=sb.norm, 
    #                          omega_cont=sb.cont_eigvals, omega=omega,
    #                          density=sb.density, regularizer=regularizer)
    
    # We will build now the E^-1 @ F blocks.
    # M = np.einsum('ilm,imk->ilk', E_inv, sb.F) # E^-1 @ F, shape (n_psi, n_full, n_full)
    D = sb.D0.astype(np.complex128).copy()
    D[:, n_harm:, n_harm:] += sb.D1 / lambda_target[:, None, None]
    M = _apply_E_inv(
                        vr=sb.vr,
                        vl=sb.vl,
                        norm=sb.norm,
                        omega_cont=sb.cont_eigvals,
                        omega=omega,
                        density=sb.density,
                        D=D,
                        F=sb.F,
                        regularizer=regularizer,
                    )

    # We add D0 @ E^-1 @ F to M, since D0 is the part of D that is not 
    # dependent on the eigenvalue.
    for ipsi in range(n_psi):
        # # D1 is the lambda-dependent D22 block, not a full-state matrix.
        # D = sb.D0[ipsi].copy()
        # D[n_harm:, n_harm:] += sb.D1[ipsi] / lambda_target[ipsi] # D = D0 + D1/lambda
        # M[ipsi] = D @ M[ipsi] # D @ E^-1 @ F

        # C1 is the lambda-dependent C12 block, not a full-state matrix.
        C = sb.C0[ipsi].copy()
        C[:n_harm, n_harm:] += sb.C1[ipsi] * lambda_target[ipsi] # C = C0 + C1*lambda
        M[ipsi] += C # C + D @ E^-1 @ F

        # Finally, add the radial-surface phase to both p1 and xi equations.
        phase = sb.gtp_over_gpp[ipsi] + sb.gpz_over_gpp[ipsi]
        M[ipsi, :n_harm, :n_harm] += phase
        M[ipsi, n_harm:, n_harm:] += phase

    return M

# -----------------------------------------------------------------------------
# Applying the ideal MHD matrix to a vector.
# -----------------------------------------------------------------------------
def apply_to(sb: solver_blocks, M: np.ndarray[np.complex128], x: np.ndarray[np.complex128]):
    r"""
    Given the solver blocks and the evaluated M matrix at a given frequency, 
    we apply the ideal MHD operator to a vector x.

    The vector X is the full state: Radial \otimes Poloidal vector.

    y_l = \int phi_l (\partial_\psi - M) x d\psi.

    For the radial derivative, we will use the convolution with the weak
    derivative operator, which is stored in the solver_blocks. Instead for the
    M matrix application:

    M_part^l = \sum_k \int phi_l \phi_k M(\Psi) x_k d\psi \approx
                \sum_{kq} w_q phi_l(\psi_q) phi_k(\psi_q) M(\psi_q) x_k

    where the sum over q is the quadrature sum, and the sum over k is the sum
    over the states. We will use the quadrature points and weights stored in the
    solver_blocks to perform the quadrature.
    """
    # Let's check for the sizes.
    nq, n_state, _ = M.shape
    n_radial = sb.phi_q.shape[1]
    x = np.asarray(x, dtype=np.complex128)

    if x.ndim == 1:
        assert x.shape[0] == n_radial * n_state, "x must have shape (n_radial * n_state,)"
        x = x.reshape((n_radial, n_state))
    elif x.ndim == 2:
        assert x.shape == (n_radial, n_state), "x must have shape (n_radial, n_state)"
    else:
        raise ValueError("x must be a 1D or 2D array with shape (n_radial * n_state,) or (n_radial, n_state)")

    if sb.phi_q.shape[0] != nq:
        raise ValueError("M and phi_q must use the same quadrature points.")
    if sb.weights_q.shape[0] != nq:
        raise ValueError("M and weights_q must use the same quadrature points.")

    # Evaluate the radial expansion at quadrature points, apply the local
    # physics matrix, then project back against the test basis.
    x_q = np.einsum('qk,kj->qj', sb.phi_q, x, optimize=True)
    mx_q = np.einsum('qij,qj->qi', M, x_q, optimize=True)
    y_m = np.einsum('q,ql,qi->li', sb.weights_q, sb.phi_q, mx_q, optimize=True)

    # Apply the radial derivative part.
    y = sb.WD @ x - y_m
    return y.flatten()

def apply_to_reduced(xin: np.ndarray[np.complex128], 
                     xout: np.ndarray[np.complex128], 
                     sb: solver_blocks, M: np.ndarray[np.complex128], 
                     bc: BoundaryCondition):
    """
    The input here is the reduced vector z, which only contains the real
    degrees of freedom of the problem, i.e., those not affected by the boundary
    conditions. 

    This will transform z into the full vector x, using the corresponding subspace
    transformation, and then apply the M operator to x, and finally return the result
    in the reduced space, applying the corresponding projection to the result.
    """
    # We first transform z into the full vector x.
    x = bc.expand(xin)
    y = apply_to(sb, M, x)
    xout[:] = bc.restrict(y)

    return xout
