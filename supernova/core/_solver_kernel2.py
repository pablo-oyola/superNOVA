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
from scipy.sparse import csr_matrix, coo_matrix, coo_array

# Some of the kernels are already implemented.
from ._continuum_kernel import (_fill_block_kernel, 
                                _apply_ghg_scaling,
                                _fill_convolution_block_kernel,
                                )
from ._solver_kernel import _build_static_blocks

from ..radial.base import RadialFE
from ._boundary import BoundaryCondition


def _build_sparse(C: np.ndarray, dm: int=None) -> coo_array:
    """
    Store a dense 2D or 3D array as a sparse COO array.

    The harmonic coupling cutoff is applied by the block-building kernels, so
    this helper only removes exact zeros. This preserves off-block entries such
    as C12/D12/F12, whose row-column offsets are larger than the harmonic
    coupling width in the flattened local-state matrix.
    """
    C = np.asarray(C)
    coords = np.nonzero(C)
    if len(coords) == 0 or coords[0].size == 0:
        return coo_array(C.shape, dtype=C.dtype)
    return coo_array((C[coords], coords), shape=C.shape)


def _as_coo_tensor(value, name: str) -> coo_array:
    """
    Return ``value`` as a 3D COO sparse array.
    """
    tensor = value if isinstance(value, coo_array) else coo_array(value)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be a 3D sparse array, got shape {tensor.shape}.")
    return tensor


def _filter_zero_data(data: np.ndarray, coords: tuple[np.ndarray, ...]):
    keep = data != 0.0
    if np.all(keep):
        return data, coords
    return data[keep], tuple(coord[keep] for coord in coords)


def _coo_tensor_from_parts(
    shape: tuple[int, int, int],
    data_parts: list[np.ndarray],
    coord_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> coo_array:
    """
    Build a 3D COO tensor from coordinate chunks.
    """
    if not data_parts:
        return coo_array(shape, dtype=np.complex128)

    data = np.concatenate(data_parts).astype(np.complex128, copy=False)
    coords = tuple(
        np.concatenate([coord_part[axis] for coord_part in coord_parts])
        for axis in range(3)
    )
    data, coords = _filter_zero_data(data, coords)
    out = coo_array((data, coords), shape=shape, dtype=np.complex128)
    out.sum_duplicates()
    out.eliminate_zeros()
    return out


def _scale_coo_axis0(tensor, factors: np.ndarray, name: str, scale=1.0) -> coo_array:
    """
    Scale a 3D COO tensor by factors indexed by its first coordinate.
    """
    tensor = _as_coo_tensor(tensor, name)
    factors = np.asarray(factors, dtype=np.complex128)
    if factors.shape != (tensor.shape[0],):
        raise ValueError(f"{name} scale factors must have shape {(tensor.shape[0],)}, got {factors.shape}.")

    q = tensor.coords[0]
    data = tensor.data.astype(np.complex128, copy=True) * factors[q] * scale
    data, coords = _filter_zero_data(data, tensor.coords)
    return coo_array((data, coords), shape=tensor.shape, dtype=np.complex128)


def _scale_coo_rows_by_axis0(
    tensor,
    row_start: int,
    factors: np.ndarray,
    name: str,
) -> coo_array:
    """
    Scale rows ``row_start:`` of each first-axis slice by slice-dependent factors.
    """
    tensor = _as_coo_tensor(tensor, name)
    factors = np.asarray(factors, dtype=np.complex128)
    if factors.shape != (tensor.shape[0],):
        raise ValueError(f"{name} scale factors must have shape {(tensor.shape[0],)}, got {factors.shape}.")

    q, rows, _ = tensor.coords
    data = tensor.data.astype(np.complex128, copy=True)
    row_mask = rows >= row_start
    data[row_mask] *= factors[q[row_mask]]
    data, coords = _filter_zero_data(data, tensor.coords)
    return coo_array((data, coords), shape=tensor.shape, dtype=np.complex128)


def _offset_coo_tensor(
    tensor,
    row_offset: int,
    col_offset: int,
    shape: tuple[int, int, int],
    name: str,
    scale=1.0,
) -> coo_array:
    """
    Place a 3D local tensor into a larger row/column block.
    """
    tensor = _as_coo_tensor(tensor, name)
    if tensor.shape[0] != shape[0]:
        raise ValueError(f"{name} first dimension must be {shape[0]}, got {tensor.shape[0]}.")
    if row_offset < 0 or col_offset < 0:
        raise ValueError("row_offset and col_offset must be nonnegative.")
    if row_offset + tensor.shape[1] > shape[1] or col_offset + tensor.shape[2] > shape[2]:
        raise ValueError(f"{name} with offsets does not fit inside shape {shape}.")

    q, rows, cols = tensor.coords
    data = tensor.data.astype(np.complex128, copy=True) * scale
    coords = (q.copy(), rows + row_offset, cols + col_offset)
    data, coords = _filter_zero_data(data, coords)
    return coo_array((data, coords), shape=shape, dtype=np.complex128)


def _combine_coo_tensors(shape: tuple[int, int, int], *tensors) -> coo_array:
    """
    Sum 3D COO tensors with duplicate-coordinate coalescing.
    """
    data_parts = []
    coord_parts = []
    for ii, tensor in enumerate(tensors):
        tensor = _as_coo_tensor(tensor, f"tensor[{ii}]")
        if tensor.shape != shape:
            raise ValueError(f"Expected tensor shape {shape}, got {tensor.shape}.")
        if tensor.nnz == 0:
            continue
        data_parts.append(tensor.data)
        coord_parts.append(tensor.coords)
    return _coo_tensor_from_parts(shape, data_parts, coord_parts)

@dataclass
class solver_blocks2:
    """
    Contains the block matrices of the problem, once they are filled and prepared
    at initialization.
    """
    # The X0 contain the parts of the matrices that are not dependent on 
    # the eigenvalue, while the X1 are the parts that are related to the eigenvalue
    # operations. The X1 blocks are only Nstate*Nstate, whereas the 
    # X0 blocks are the full (2*Nstate)*(2*Nstate) blocks.
    m_range: np.ndarray
    max_dm: int # Maximum coupling between m-modes in the system.
    ntor: int
    C0: coo_array # C = C0 + C1 * lambda
    C1: coo_array # C = C0 + C1 * lambda

    D0: coo_array
    D1: coo_array # D = D0 + D1 / lambda

    F: coo_array

    # Metric parts for the phase.
    gtp_over_gpp: coo_array # g^{\psi\theta}/g^{\psi\psi}
    gpz_over_gpp: coo_array # g^{\psi\zeta}/g^{\psi\psi}

    # Profiles.
    hprof: np.ndarray
    Iprof: np.ndarray
    gprof: np.ndarray
    qprof: np.ndarray
    pressure: np.ndarray
    pressure_deriv: np.ndarray
    density: np.ndarray # Mass density used in lambda = mu_0 * rho_mass * omega**2.

    # The following are the blocks related to the continuum.
    A: coo_array # The A matrix from the continuum solver.
    B: coo_array # The B matrix from the continuum solver.

    # And finally some elements related to the radialFE.
    weights_q: np.ndarray # Quadrature weights.
    points_q: np.ndarray # Quadrature points.
    phi_q: np.ndarray # Radial basis functions evaluated at the quadrature points.
    WD: csr_matrix # Weak derivative matrix evaluated at the quadrature points.

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
        add_var("C0", self.C0.toarray(), ("x", "full_state_row", "full_state_col"), block_attrs)
        add_var("D0", self.D0.toarray(), ("x", "full_state_row", "full_state_col"), block_attrs)
        add_var("F", self.F.toarray(), ("x", "full_state_row", "full_state_col"), block_attrs)
        add_var("C1", self.C1.toarray(), ("x", "m_pol_row", "m_pol_col"),
                {"description": "Lambda-dependent C12 harmonic block."})
        add_var("D1", self.D1.toarray(), ("x", "m_pol_row", "m_pol_col"),
                {"description": "Inverse-lambda-dependent D22 harmonic block."})
        add_var("A", self.A.toarray(), ("x", "full_state_row", "full_state_row"),
                {"description": "Continuum A matrix."})
        add_var("B", self.B.toarray(), ("x", "full_state_row", "full_state_row"),
                {"description": "Continuum B matrix."})
        add_var("gtp_over_gpp", self.gtp_over_gpp.toarray(), ("x", "m_pol_row", "m_pol_col"))
        add_var("gpz_over_gpp", self.gpz_over_gpp.toarray(), ("x", "m_pol_row", "m_pol_col"))

        profile_dims = ("x",)
        add_var("hprof", self.hprof, profile_dims)
        add_var("Iprof", self.Iprof, profile_dims)
        add_var("gprof", self.gprof, profile_dims)
        add_var("qprof", self.qprof, profile_dims)
        add_var("pressure", self.pressure, profile_dims)
        add_var("pressure_deriv", self.pressure_deriv, profile_dims)
        add_var("density", self.density, profile_dims,
                {"description": "Mass density used in lambda = mu_0 * rho_mass * omega**2."})

        add_var("weights_q", self.weights_q, ("x_quad",),
                {"description": "Quadrature weights for radial integrals."})
        add_var("phi_q", self.phi_q, ("x_quad", "radial_dof"),
                {"description": "Radial basis functions evaluated at quadrature points."})
        add_var("WD", self.WD.toarray(), ("radial_dof_row", "radial_dof_col"),
                {"description": "Weak derivative matrix."})
        add_var("points_q", self.points_q, ("x_quad",),
                {"description": "Quadrature points for radial integrals."})

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                "ntor": int(self.ntor),
                "max_dm": int(getattr(self, "max_dm", 100)),
                "mmax": int(getattr(self, "mmax", np.max(np.abs(m_range)))),
                "state_order": "full_state is ordered as p1(m_range) followed by xi(m_range).",
                "description": "Serialized solver_blocks2 data for solver debugging and persistence.",
            },
        )
    
    @staticmethod
    def from_xarray(dataset: xr.Dataset) -> "solver_blocks2":
        """
        Reconstructs a solver_blocks2 instance from an xarray dataset created by to_xarray.

        Parameters
        ----------
        dataset : xr.Dataset
            The xarray dataset containing the serialized solver_blocks2 data.

        Returns
        -------
        solver_blocks2
            The reconstructed solver_blocks2 instance.
        """
        # We get the maximum coupling.
        dm_max = dataset.attrs.get("max_dm", 100)

        # We will extract the variables and coordinates from the dataset and use them to construct the solver_blocks2 instance.
        C0 = _build_sparse(dataset["C0"].values, dm_max) if "C0" in dataset else None
        D0 = _build_sparse(dataset["D0"].values, dm_max) if "D0" in dataset else None
        F = _build_sparse(dataset["F"].values, dm_max) if "F" in dataset else None
        C1 = _build_sparse(dataset["C1"].values, dm_max) if "C1" in dataset else None
        D1 = _build_sparse(dataset["D1"].values, dm_max) if "D1" in dataset else None
        gtp_over_gpp = _build_sparse(dataset["gtp_over_gpp"].values, dm_max) if "gtp_over_gpp" in dataset else None
        gpz_over_gpp = _build_sparse(dataset["gpz_over_gpp"].values, dm_max) if "gpz_over_gpp" in dataset else None

        hprof = dataset["hprof"].values if "hprof" in dataset else None
        Iprof = dataset["Iprof"].values if "Iprof" in dataset else None
        gprof = dataset["gprof"].values if "gprof" in dataset else None
        qprof = dataset["qprof"].values if "qprof" in dataset else None
        pressure = dataset["pressure"].values if "pressure" in dataset else None
        pressure_deriv = dataset["pressure_deriv"].values if "pressure_deriv" in dataset else None
        density = dataset["density"].values if "density" in dataset else None

        A = _build_sparse(dataset["A"].values, dm_max) if "A" in dataset else None
        B = _build_sparse(dataset["B"].values, dm_max) if "B" in dataset else None

        weights_q = dataset["weights_q"].values if "weights_q" in dataset else None
        phi_q = dataset["phi_q"].values if "phi_q" in dataset else None
        WD = _build_sparse(dataset["WD"].values, dm_max) if "WD" in dataset else None
        m_range = dataset.coords["m_pol"].values if "m_pol" in dataset.coords else None
        ntor = int(dataset.attrs["ntor"]) if "ntor" in dataset.attrs else None

        return solver_blocks2(
            max_dm=dm_max,
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
            A=A, B=B,
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
# -----------------------------------------------------------------------------
# Static blocks building kernels.
# -----------------------------------------------------------------------------

@njit(parallel=True, cache=True, nogil=True)
def assemble_continuum(H: np.ndarray, K: np.ndarray, J: np.ndarray, 
                       L: np.ndarray, M: np.ndarray, m_range: np.ndarray,
                       ntor: int, gammas: float, q: np.ndarray, p0: np.ndarray,
                       max_dm: int = 1000, max_dn: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds the full A and B matrices by filling sub-blocks directly.
    
    A = [[A11, A12],    B = [[B11, 0  ],
         [0  , A22]]         [B21, B22]]

    where A11 = G* H G,
          A12 = gamma_s p_0 K
          A21 = 0
          A22 = gamma_s p_0 G* M G
          B11 = J
          B12 = 0
          B21 = K
          B22 = L
    """
    # Checking all the inputs have the same shape (they should be 
    # profiles evaluated at the same radial point with the same number of
    # harmonics).
    assert np.all(H.shape == K.shape), "H and K must have the same shape"
    assert np.all(H.shape == J.shape), "H and J must have the same shape"
    assert np.all(H.shape == L.shape), "H and L must have the same shape"
    assert np.all(H.shape == M.shape), "H and M must have the same shape"
    # assert np.all(H.size == m_range.size//2), "H and m_range must have the same size"

    n_range = ntor + np.zeros_like(m_range) # n_range will assume no toroidal coupling.
    nmode = m_range.size
    two_n = 2 * nmode

    # 2. Pre-allocate the full matrices (one-time allocation)
    A_out = np.zeros((H.shape[0], two_n, two_n), dtype=np.complex128)
    B_out = np.zeros((H.shape[0], two_n, two_n), dtype=np.complex128)

    for ipsi in prange(H.shape[0]):
        # 3. Fill A blocks via views (no copies made)
        # A11 (Top-Left)
        fill_block_kernel(H[ipsi], m_range, n_range, A_out[ipsi, :nmode, :nmode], 
                          max_dm=max_dm, max_dn=max_dn)
        apply_ghg_scaling(A_out[ipsi, :nmode, :nmode], q[ipsi], n_range, 
                         m_range)  # Apply G^H * A11 * G scaling
    
        # A12 (Top-Right)
        # A12 = - gamma_s * (pressure_mn * K_mn)
        fill_block_kernel(K[ipsi], m_range, n_range, A_out[ipsi, :nmode, nmode:], 
                          conjugate_mode=False,  max_dm=max_dm, max_dn=max_dn)
        A_out[ipsi, :nmode, nmode:] *= - (gammas * mu_0 * p0[ipsi])
        
        # A22 (Bottom-Right)
        fill_block_kernel(M[ipsi], m_range, n_range, A_out[ipsi, nmode:, nmode:], 
                        max_dm=max_dm, max_dn=max_dn)
        apply_ghg_scaling(A_out[ipsi, nmode:, nmode:], q[ipsi], n_range,
                          m_range, scale=p0[ipsi]*gammas*mu_0)  # Apply G^H * M * G scaling

        # 4. Fill B blocks via views
        # B11 (Top-Left)
        fill_block_kernel(J[ipsi], m_range, n_range, B_out[ipsi, :nmode, :nmode], 
                        max_dm=max_dm, max_dn=max_dn)
        # B21 (Bottom-Left)
        fill_block_kernel(K[ipsi], m_range, n_range, B_out[ipsi, nmode:, :nmode], 
                        max_dm=max_dm, max_dn=max_dn)
        # B22 (Bottom-Right)
        fill_block_kernel(L[ipsi], m_range, n_range, B_out[ipsi, nmode:, nmode:], 
                        max_dm=max_dm, max_dn=max_dn)

    return A_out, B_out

def build_solver_blocks2(
    C11: np.ndarray, O: np.ndarray, C12_2: np.ndarray, C12_3: np.ndarray, C22: np.ndarray,
    D11: np.ndarray, D12: np.ndarray, kappa_s: np.ndarray,
    K: np.ndarray, M: np.ndarray, 
    F11_2: np.ndarray, F12_1: np.ndarray, F12_2: np.ndarray, F21: np.ndarray, F22: np.ndarray,
    hprof: np.ndarray, Iprof: np.ndarray, gprof: np.ndarray, qprof: np.ndarray,
    pressure: np.ndarray, pressure_deriv: np.ndarray, density: np.ndarray, gamma_s: float,
    one_gpp: np.ndarray, gtp_over_gpp: np.ndarray, gpz_over_gpp: np.ndarray,
    H: np.ndarray, J: np.ndarray, L: np.ndarray,
    ntor: int, m_range: np.ndarray, max_dm: int=1000, max_dn: int=1000
) -> solver_blocks2:
    r"""
    Intermediary function that will call the kernel to build the blocks,
    and then return the dataclass with the filled frozen blocks.
    """
    tmp = _build_static_blocks(C11=C11, O=O, C12_2=C12_2, C12_3=C12_3,
                               C22=C22, D11=D11, D12=D12, kappa_s=kappa_s,
                               K=K, M=M, F11_2=F11_2, F12_1=F12_1, F12_2=F12_2, 
                               F21=F21, F22=F22, 
                               hprof=hprof, Iprof=Iprof, gprof=gprof, qprof=qprof,
                               pressure=pressure, pressure_deriv=pressure_deriv, 
                               density=density, gamma_s=gamma_s,
                               one_gpp=one_gpp, gtp_over_gpp=gtp_over_gpp, 
                               gpz_over_gpp=gpz_over_gpp, ntor=ntor,
                               m_range=m_range, dm_max=max_dm, dn_max=max_dn)
    C0, C1, D0, D1, F, gtp_over_gpp_mat, gpz_over_gpp_mat = tmp

    # We now assemble the continuum matrices A and B, that will be used in the solver to apply the continuum operator.
    A, B = assemble_continuum(H=H, K=K, J=J, L=L, M=M, m_range=m_range, 
                              ntor=ntor, gammas=gamma_s, q=qprof, p0=pressure, max_dm=max_dm, max_dn=max_dn)
    
    # Compressing the matrices to sparse format to save memory and speed up operations.
    C0 = _build_sparse(C0, max_dm)
    C1 = _build_sparse(C1, max_dm)
    D0 = _build_sparse(D0, max_dm)
    D1 = _build_sparse(D1, max_dm)
    F = _build_sparse(F, max_dm)
    gtp_over_gpp_mat = _build_sparse(gtp_over_gpp_mat, max_dm)
    gpz_over_gpp_mat = _build_sparse(gpz_over_gpp_mat, max_dm)
    A = _build_sparse(A, max_dm)
    B = _build_sparse(B, max_dm)

    # We create the m_range array, that will be used in the solver to 
    # identify the m values of each state.
    out = solver_blocks2(m_range=m_range, max_dm=max_dm, ntor=ntor,
                        C0=C0, C1=C1, D0=D0, D1=D1, F=F,
                        gtp_over_gpp=gtp_over_gpp_mat, gpz_over_gpp=gpz_over_gpp_mat,
                        hprof=hprof, Iprof=Iprof, gprof=gprof, qprof=qprof, 
                        pressure=pressure, pressure_deriv=pressure_deriv, density=density,
                        A=A, B=B,
                        weights_q=None, points_q=None, phi_q=None, WD=None)
    out.max_dm = max_dm
    out.max_dn = max_dn
    return out

# -- Building the operator for a given frequency:
def build_M_at(sb: solver_blocks2, omega: float, regularizer: float=1e-3) -> coo_array:
    r"""
    Builds the full operator M in the space of (p1, xi^\psi, xi^s, \nabla\cdot\xi) for a given
    frequency omega.

    The operator is built as:
    M = [[C'     D]
         [F   - E]],

    where C' = C - (im(g^{\psi\theta}/g^{\psi\psi}) + in(g^{\psi\zeta}/g^{\psi\psi})) * ((1 0)(0 1))

    this forms part of the ideal MHD problem:

    ((\partial_\psi,\partial_\psi,0, 0) - M )(p1, xi^\psi, xi^s, \nabla\cdot\xi) = 0.
    """
    nmode = sb.m_range.size
    nstates_xip1 = 2 * nmode
    nstates = 4 * nmode
    lambda_target = mu_0 * sb.density * omega**2
    if np.any(lambda_target == 0.0):
        raise ValueError("lambda_target cannot contain zero values.")

    npsi = sb.C0.shape[0]
    if lambda_target.shape != (npsi,):
        raise ValueError(f"density must have shape {(npsi,)}, got {sb.density.shape}.")

    xip1_shape = (npsi, nstates_xip1, nstates_xip1)
    harmonic_shape = (npsi, nmode, nmode)
    full_shape = (npsi, nstates, nstates)

    expected_shapes = {
        "C0": xip1_shape,
        "D0": xip1_shape,
        "F": xip1_shape,
        "C1": harmonic_shape,
        "D1": harmonic_shape,
        "gtp_over_gpp": harmonic_shape,
        "gpz_over_gpp": harmonic_shape,
        "A": xip1_shape,
        "B": xip1_shape,
    }
    for name, expected_shape in expected_shapes.items():
        tensor = _as_coo_tensor(getattr(sb, name), name)
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {tensor.shape}.")

    phase = _combine_coo_tensors(
        harmonic_shape,
        sb.gtp_over_gpp,
        sb.gpz_over_gpp,
    )
    C = _combine_coo_tensors(
        xip1_shape,
        sb.C0,
        _offset_coo_tensor(
            _scale_coo_axis0(sb.C1, lambda_target, "C1"),
            0,
            nmode,
            xip1_shape,
            "C1",
        ),
        _offset_coo_tensor(phase, 0, 0, xip1_shape, "phase"),
        _offset_coo_tensor(phase, nmode, nmode, xip1_shape, "phase"),
    )
    D = _combine_coo_tensors(
        xip1_shape,
        sb.D0,
        _offset_coo_tensor(
            _scale_coo_axis0(sb.D1, 1.0 / lambda_target, "D1"),
            nmode,
            nmode,
            xip1_shape,
            "D1",
        ),
    )
    F = _as_coo_tensor(sb.F, "F")

    r = (1+1j*regularizer)**2
    E = _combine_coo_tensors(
        xip1_shape,
        _scale_coo_axis0(sb.A, np.ones(npsi, dtype=np.complex128), "A", scale=-1.0),
        _scale_coo_axis0(sb.B, lambda_target * r, "B"),
    )

    # We need to scale the E matrix: 
    # E' = T @ E, where T = ((1, 0)(0 \lambda))
    # E  = T^{-1} @ E' = ((1, 0)(0 1/\lambda)) @ E'
    # So we need to scale the E matrix by 1/lambda for the xi part
    E = _scale_coo_rows_by_axis0(E, nmode, 1.0 / lambda_target, "E")

    # We build the M matrix as:
    #
    # M = [(C'    D)
    #      (F   - E)]
    #
    #
    return _combine_coo_tensors(
        full_shape,
        _offset_coo_tensor(C, 0, 0, full_shape, "C"),
        _offset_coo_tensor(D, 0, nstates_xip1, full_shape, "D"),
        _offset_coo_tensor(F, nstates_xip1, 0, full_shape, "F"),
        _offset_coo_tensor(E, nstates_xip1, nstates_xip1, full_shape, "E", scale=-1.0),
    )

def get_T_sparse(sb: solver_blocks2, M: coo_array, 
                 omega: float=None, regularizer: float=1e-3) -> csr_matrix:
    r"""
    Builds the full ideal MHD operator T in the space of 
    (p1, xi^\psi, xi^s, \nabla\cdot\xi) for a given
    frequency omega.

    T = ((\partial_\psi,\partial_\psi,0, 0)\otimes I_M - M )
    """
    M = _as_coo_tensor(M, "M")
    if M.shape[1] != M.shape[2]:
        raise ValueError("M must be square in its state dimensions.")

    nq, n_state, _ = M.shape
    nmode = sb.m_range.size
    expected_state = 4 * nmode
    if n_state != expected_state:
        raise ValueError(f"M state size must be {expected_state}, got {n_state}.")

    if sb.phi_q is None:
        raise ValueError("sb.phi_q must be set before assembling T.")
    if sb.weights_q is None:
        raise ValueError("sb.weights_q must be set before assembling T.")
    if sb.WD is None:
        raise ValueError("sb.WD must be set before assembling T.")

    phi_q = np.asarray(sb.phi_q)
    weights_q = np.asarray(sb.weights_q)
    if phi_q.ndim != 2:
        raise ValueError("phi_q must have shape (n_quad, n_radial).")
    if phi_q.shape[0] != nq:
        raise ValueError("M and phi_q must use the same quadrature points.")
    if weights_q.shape != (nq,):
        raise ValueError(f"weights_q must have shape {(nq,)}, got {weights_q.shape}.")

    n_radial = phi_q.shape[1]
    if sb.WD.shape != (n_radial, n_radial):
        raise ValueError(f"WD must have shape {(n_radial, n_radial)}, got {sb.WD.shape}.")

    rows = []
    cols = []
    data = []

    # Radial derivative part. Only p1 and xi^psi carry a radial derivative in
    # ((partial_psi, partial_psi, 0, 0) - M).
    wd = coo_matrix(sb.WD)
    if wd.nnz:
        derivative_states = np.arange(2 * nmode, dtype=np.int64)
        rows.append(
            np.repeat(wd.row, derivative_states.size) * n_state
            + np.tile(derivative_states, wd.nnz)
        )
        cols.append(
            np.repeat(wd.col, derivative_states.size) * n_state
            + np.tile(derivative_states, wd.nnz)
        )
        data.append(np.repeat(wd.data.astype(np.complex128), derivative_states.size))

    q_coords, local_rows, local_cols = M.coords
    local_data = M.data.astype(np.complex128, copy=False)
    local_keep = local_data != 0.0
    if np.any(local_keep):
        q_coords = q_coords[local_keep]
        local_rows = local_rows[local_keep]
        local_cols = local_cols[local_keep]
        local_data = local_data[local_keep]

        for q in np.unique(q_coords):
            radial_support = np.flatnonzero(phi_q[q] != 0.0)
            if radial_support.size == 0:
                continue

            q_mask = q_coords == q
            local_q_rows = local_rows[q_mask]
            local_q_cols = local_cols[q_mask]
            local_q_data = local_data[q_mask]
            if local_q_data.size == 0:
                continue

            radial_rows = np.repeat(radial_support, radial_support.size)
            radial_cols = np.tile(radial_support, radial_support.size)
            radial_data = (
                weights_q[q]
                * phi_q[q, radial_rows]
                * phi_q[q, radial_cols]
            ).astype(np.complex128, copy=False)

            rows.append(
                (
                    radial_rows[:, None] * n_state
                    + local_q_rows[None, :]
                ).ravel()
            )
            cols.append(
                (
                    radial_cols[:, None] * n_state
                    + local_q_cols[None, :]
                ).ravel()
            )
            data.append(-(radial_data[:, None] * local_q_data[None, :]).ravel())

    size = n_radial * n_state
    if not data:
        return csr_matrix((size, size), dtype=np.complex128)

    return coo_matrix(
        (
            np.concatenate(data),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(size, size),
        dtype=np.complex128,
    ).tocsr()
