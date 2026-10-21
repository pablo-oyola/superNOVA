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
from supernova._utils import njit, prange
# from scipy.constants import mu_0
mu_0 = 1.0 # Normalized units.
from dataclasses import dataclass
import xarray as xr
from scipy.sparse import csr_matrix, coo_array

# Some of the kernels are already implemented.
from supernova.operators.continuum_kernel import (_fill_block_kernel, 
                                _apply_ghg_scaling,
                                _fill_convolution_block_kernel,
                                )
# Moved verbatim from the retired _solver_kernel.py.  These four kernels
# carry the static block construction and are frozen by
# tests/test_physics_freeze.py: relocate them, never rewrite them.
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
    Multiply each matrix column by the corresponding diagonal factor in place.

    This is a small Numba helper for right multiplication by a diagonal matrix.
    After the call, ``matrix[i, j]`` has been multiplied by ``diag[j]`` for all
    rows ``i``. The operation mutates ``matrix`` and returns nothing.

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
        One-dimensional column-scaling factors with length ``matrix.shape[1]``.
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

@njit(cache=True, nogil=True, inline="always")
def apply_dm_cutoff(matrix: np.ndarray, m_range: np.ndarray, dm_max: int):
    """
    Zero entries outside the requested physical poloidal-mode coupling width.
    """
    for i in range(matrix.shape[0]):
        mi = m_range[i]
        for j in range(matrix.shape[1]):
            if abs(mi - m_range[j]) > dm_max:
                matrix[i, j] = 0.0

@njit(parallel=True, cache=True, nogil=True)
def _build_static_blocks(
    C11: np.ndarray, O: np.ndarray, C12_2: np.ndarray, C12_3: np.ndarray, C22: np.ndarray,
    D11: np.ndarray, D12: np.ndarray, kappa_s: np.ndarray,
    K: np.ndarray, M: np.ndarray, 
    F11_2: np.ndarray, F12_1: np.ndarray, F12_2: np.ndarray, F21: np.ndarray, F22: np.ndarray,
    hprof: np.ndarray, Iprof: np.ndarray, gprof: np.ndarray, qprof: np.ndarray,
    pressure: np.ndarray, pressure_deriv: np.ndarray, density: np.ndarray, gamma_s: float,
    one_gpp: np.ndarray, gtp_over_gpp: np.ndarray, gpz_over_gpp: np.ndarray,
    ntor: int, m_range: int,
    dm_max: int=100, dn_max: int=100):
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
    # nstate = (C11.shape[1] - 1) * 2 + 1 # m=0 is only one state, m>0 are two states.
    # mpol_max = C11.shape[1] - 1
    npsi = C11.shape[0]
    # m_range = np.arange(-mpol_max, mpol_max + 1)
    n_range = np.zeros_like(m_range) + ntor
    mpol_max = np.max(np.abs(m_range))
    nstate = m_range.size

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
    assert gpz_over_gpp.shape == C11.shape, r"g_{\zeta\psi} must have shape (npsi, mpol, ntor)"
    assert gamma_s >= 0, "gamma_s must be positive."

    # Allocating the space. 
    C0 = np.zeros((npsi, 2*nstate, 2*nstate), dtype=np.complex128)
    D0 = np.zeros((npsi, 2*nstate, 2*nstate), dtype=np.complex128)
    F  = np.zeros((npsi, 2*nstate, 2*nstate), dtype=np.complex128)
    C1 = np.zeros((npsi, nstate, nstate), dtype=np.complex128)
    D1 = np.zeros((npsi, nstate, nstate), dtype=np.complex128)
    gtp_over_gpp_mat = np.zeros((npsi, nstate, nstate), dtype=np.complex128)
    gpz_over_gpp_mat = np.zeros((npsi, nstate, nstate), dtype=np.complex128)

    # Some preparatory elements.
    diagonal_ones = np.eye(nstate, dtype=np.complex128)

    for ipsi in prange(npsi):
        # We place first into the C0, D0 and F matrices the blocks that do
        # not require convolution or extra operations.
        # C11 and C22 elements go directly into the matrix without further ado.
        fill_block_kernel(2.0*C11[ipsi], m_range, n_range, 
                          C0[ipsi, :nstate, :nstate], 
                          max_dm=dm_max, max_dn=dn_max)
        fill_block_kernel(C22[ipsi], m_range, n_range, 
                          C0[ipsi, nstate:, nstate:], 
                          max_dm=dm_max, max_dn=dn_max)

        # On the element C12, we only place the C12_3 part and then we add up the 
        # scaled C11 by mu_0 p_0'.
        fill_block_kernel(C12_3[ipsi], m_range, n_range, 
                          C0[ipsi, :nstate, nstate:], 
                          max_dm=dm_max, max_dn=dn_max)
        C0[ipsi, :nstate, nstate:] += mu_0 * pressure_deriv[ipsi] * C0[ipsi, :nstate, :nstate]

        # We fill D11 with the D11 profile, and we will add later the contribution of the convolution with the G operator.
        fill_block_kernel(D11[ipsi], m_range, n_range, 
                          D0[ipsi, :nstate, :nstate], 
                          max_dm=dm_max, max_dn=dn_max)

        # D12 element only needs to be multiplied later by 2\mu_0\gamma_s p_0 
        # which is a local value: D12 is an alias for C11.
        fill_block_kernel(C11[ipsi], m_range, n_range, 
                          D0[ipsi, :nstate, nstate:], 
                          max_dm=dm_max, max_dn=dn_max)
        D0[ipsi, :nstate, nstate:] *= 2 * mu_0 * gamma_s * pressure[ipsi]

        # D21 has two pieces: one that is directly the kappa_s and the other one is 
        # the operator \hat\sigma/h(\Psi) applied to the eigenvector.
        fill_block_kernel(2.0*kappa_s[ipsi], m_range, n_range, 
                          D0[ipsi, nstate:, :nstate], 
                          max_dm=dm_max, max_dn=dn_max)

        # D22 has a 1 - \beta contribution. The beta term depends on the
        # eigenvalue, so place the 1 part here and the signed term in D1 below.
        D0[ipsi, nstate:, nstate:] = diagonal_ones

        # F11 element is composed by -K + \hat\sigma/B^2. We just place now the
        # K part, and we will add the rest later.
        kmat = np.zeros((nstate, nstate), dtype=np.complex128)
        fill_block_kernel(K[ipsi], m_range, n_range, kmat, max_dm=dm_max, max_dn=dn_max)
        F[ipsi, :nstate, :nstate] = - kmat
        one_over_B2 = np.zeros((nstate, nstate), dtype=np.complex128) 
        fill_block_kernel(F11_2[ipsi], m_range, n_range, one_over_B2, max_dm=dm_max, max_dn=dn_max)

        # F21 and F22 are directly the F21 and F22 profiles, so we can place them directly.
        fill_block_kernel(F21[ipsi], m_range, n_range, F[ipsi, nstate:, :nstate], max_dm=dm_max, max_dn=dn_max)
        fill_block_kernel(F22[ipsi], m_range, n_range, F[ipsi, nstate:, nstate:], max_dm=dm_max, max_dn=dn_max)

        # We create the diagonal representations of the G and sigma operators.
        g_diag = np.zeros(nstate, dtype=np.complex128)
        sigma_diag = np.zeros(nstate, dtype=np.complex128)
        for i in range(nstate):
            g_diag[i] = 1j*(qprof[ipsi] * n_range[i] - m_range[i])
            sigma_diag[i] = - 1j * (Iprof[ipsi] * n_range[i] + gprof[ipsi] * m_range[i])

        # Now we create and fill other matrices.
        c12_2mat = np.zeros((nstate, nstate), dtype=np.complex128)
        fill_block_kernel(C12_2[ipsi], m_range, n_range, c12_2mat, max_dm=dm_max, max_dn=dn_max)
        omat = np.zeros((nstate, nstate), dtype=np.complex128)
        fill_block_kernel(O[ipsi], m_range, n_range, omat, max_dm=dm_max, max_dn=dn_max)
        apply_ghg_scaling(omat, qprof[ipsi], n_range, m_range) # G^H @ O @ G.

        # Now, we can get C12 contribution by convoluting the G^H @ O @ G with
        # the C12_2 factor on the left. The product of two dm-limited factors
        # can otherwise broaden the final C12 band, so cap this contribution
        # before adding it to C0.
        c12_conv = c12_2mat @ omat
        apply_dm_cutoff(c12_conv, m_range, dm_max)
        C0[ipsi, :nstate, nstate:] -= c12_conv # C12 is done!.

        # D11 is the D11 input convoluted with the G operator on the right.
        # D11 = D11 @ G. Since G is a diagonal operator, we can just scale
        # the columns of D11
        scale_cols(D0[ipsi, :nstate, :nstate], g_diag)

        # We now scale the sigma for the D21 contribution.
        D21_2mat = np.zeros((nstate, nstate), dtype=np.complex128)
        one_over_h = np.zeros((C11.shape[1], C11.shape[2]), dtype=np.complex128)
        one_over_h[0, 0] = -1.0 / hprof[ipsi]
        fill_block_kernel(one_over_h, m_range, n_range, D21_2mat, max_dm=dm_max, max_dn=dn_max) 
        # We fill D21_2 with a scalar zero-mode, since we will just scale it by the sigma_diag.
        scale_cols(D21_2mat, sigma_diag) # D21_2 = D
        D0[ipsi, nstate:, :nstate] += D21_2mat # We add the D21_2 contribution to D21.

        # sigma/B²
        scale_cols(one_over_B2, sigma_diag)
        F[ipsi, :nstate, :nstate] += one_over_B2 # We add the 1/B^2 part to F11, which is the part that does

        # We build now F12 contributions.
        f12_1mat = np.zeros((nstate, nstate), dtype=np.complex128)
        f12_2mat = np.zeros((nstate, nstate), dtype=np.complex128)
        fill_block_kernel(F12_1[ipsi], m_range, n_range, f12_1mat, max_dm=dm_max, max_dn=dn_max)
        fill_block_kernel(F12_2[ipsi], m_range, n_range, f12_2mat, max_dm=dm_max, max_dn=dn_max)
        scale_rows(f12_1mat, g_diag) # G @ F12_1
        scale_cols(f12_2mat, g_diag) # F12_2 @ G

        # We add the F12 contributions and the pressure-curvature term
        # -mu_0 * p_0' * K. The continuum profile K already contains 2*K_s.
        F[ipsi, :nstate, nstate:] += f12_1mat + f12_2mat \
            - kmat * mu_0 * pressure_deriv[ipsi] 

        # Filling the C1 and D1 matrices, that will have a dependence on the eigenvalue.
        fill_block_kernel(one_gpp[ipsi], m_range, n_range, C1[ipsi], max_dm=dm_max, max_dn=dn_max)
        fill_block_kernel(M[ipsi], m_range, n_range, D1[ipsi], max_dm=dm_max, max_dn=dn_max)
        apply_ghg_scaling(D1[ipsi], qprof[ipsi], n_range, m_range) # Compressibility core: G^H @ M @ G.
        D1[ipsi] *= -mu_0 * gamma_s * pressure[ipsi]
        # Stored inverse-lambda coefficient: D1 = -M[B0**2/h] @ A22.
        D1[ipsi] = c12_2mat @ D1[ipsi]
        apply_dm_cutoff(D1[ipsi], m_range, dm_max)

        # And finally we fill the gpt_over_gpp and gpz_over_gpp matrices, that will be used to apply the convolution with the G operator in the solver.+
        fill_block_kernel(gtp_over_gpp[ipsi], m_range, n_range, gtp_over_gpp_mat[ipsi], max_dm=dm_max, max_dn=dn_max)
        fill_block_kernel(gpz_over_gpp[ipsi], m_range, n_range, gpz_over_gpp_mat[ipsi], max_dm=dm_max, max_dn=dn_max)

        # We need now to scale the gtp,gzp_over_gpp by the appropriate factors, 
        # im and in to get the correct scaling.
        scale_cols(gtp_over_gpp_mat[ipsi], -1j*m_range) # g^{\theta\psi}/g^{\psi\psi} has an im factor.
        scale_cols(gpz_over_gpp_mat[ipsi],  1j*n_range) # g^{\zeta\psi}/g^{\psi\psi} has an in factor.

    return C0, C1, D0, D1, F, gtp_over_gpp_mat, gpz_over_gpp_mat




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


def _parts_are_disjoint(
    coord_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> bool:
    """True when no two parts can share a ``(row, col)`` position.

    Each part is summarised by its row/column bounding box.  Parts placed by
    :func:`_offset_coo_tensor` at different block offsets have disjoint boxes,
    and disjoint boxes cannot produce duplicate coordinates, so the combined
    tensor needs no coalescing.  The test is conservative: overlapping boxes
    only mean that duplicates are *possible*.
    """
    boxes = [
        (int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max()))
        for _, rows, cols in coord_parts
        if rows.size
    ]
    for ii, a in enumerate(boxes):
        for b in boxes[ii + 1:]:
            if a[0] <= b[1] and b[0] <= a[1] and a[2] <= b[3] and b[2] <= a[3]:
                return False
    return True


def _coo_tensor_from_parts(
    shape: tuple[int, int, int],
    data_parts: list[np.ndarray],
    coord_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> coo_array:
    """
    Build a 3D COO tensor from coordinate chunks.

    The assembled operators are sums of blocks placed at disjoint offsets, so
    most calls have no duplicate coordinates at all and are returned as a plain
    concatenation, unsorted.  Every consumer (the mixed projection, the contour
    continuation, the CSR conversions) groups or sums entries by coordinate
    itself and does not rely on canonical COO order.  When parts overlap, the
    duplicates are coalesced with one integer key sort, which is several times
    cheaper than scipy's three-key lexsort over the same entries.  This replaced
    an unconditional ``sum_duplicates`` that cost about a third of the assembly
    time at full harmonic coupling.
    """
    if not data_parts:
        return coo_array(shape, dtype=np.complex128)

    data = np.concatenate(data_parts).astype(np.complex128, copy=False)
    coords = tuple(
        np.concatenate(
            [coord_part[axis] for coord_part in coord_parts]
        ).astype(np.int64, copy=False)
        for axis in range(3)
    )
    data, coords = _filter_zero_data(data, coords)
    if data.size == 0:
        return coo_array(shape, dtype=np.complex128)

    if _parts_are_disjoint(coord_parts):
        return coo_array((data, coords), shape=shape, dtype=np.complex128)

    key = (coords[0] * shape[1] + coords[1]) * shape[2] + coords[2]
    order = np.argsort(key, kind="stable")
    key = key[order]
    data = data[order]
    first = np.empty(key.size, dtype=bool)
    first[0] = True
    np.not_equal(key[1:], key[:-1], out=first[1:])
    starts = np.flatnonzero(first)
    summed = np.add.reduceat(data, starts)
    key = key[starts]
    keep = summed != 0.0
    summed = summed[keep]
    key = key[keep]
    q, remainder = np.divmod(key, shape[1] * shape[2])
    rows, cols = np.divmod(remainder, shape[2])
    out = coo_array((summed, (q, rows, cols)), shape=shape, dtype=np.complex128)
    # Sorted by (q, row, col) with unique coordinates: scipy's canonical order.
    out.has_canonical_format = True
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


def _validate_dpsi_hat_dx(value) -> float:
    """Return the positive scalar ``d(psi_hat)/dx`` radial chain factor."""
    raw = np.asarray(value)
    if raw.ndim != 0 or np.iscomplexobj(raw):
        raise ValueError("dpsi_hat_dx must be a finite positive real scalar.")
    try:
        factor = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "dpsi_hat_dx must be a finite positive real scalar."
        ) from error
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("dpsi_hat_dx must be a finite positive real scalar.")
    return factor


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
class SolverBlocks:
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
    # Radial chain factor d(Psi/(Baxis*aminor**2))/dx for solver coordinate x.
    dpsi_hat_dx: float = 1.0

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
                "dpsi_hat_dx": _validate_dpsi_hat_dx(self.dpsi_hat_dx),
                "dpsi_hat_dx_definition": (
                    "d(Psi/(Baxis*aminor**2))/dx for the solver radial coordinate"
                ),
                "state_order": "full_state is ordered as p1(m_range) followed by xi(m_range).",
                "description": "Serialized SolverBlocks data for solver debugging and persistence.",
            },
        )
    
    @staticmethod
    def from_xarray(dataset: xr.Dataset) -> "SolverBlocks":
        """
        Reconstructs a SolverBlocks instance from an xarray dataset created by to_xarray.

        Parameters
        ----------
        dataset : xr.Dataset
            The xarray dataset containing the serialized SolverBlocks data.

        Returns
        -------
        SolverBlocks
            The reconstructed SolverBlocks instance.
        """
        # We get the maximum coupling.
        dm_max = dataset.attrs.get("max_dm", 100)

        # We will extract the variables and coordinates from the dataset and use them to construct the SolverBlocks instance.
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
        dpsi_hat_dx = _validate_dpsi_hat_dx(
            dataset.attrs.get("dpsi_hat_dx", 1.0)
        )

        return SolverBlocks(
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
            dpsi_hat_dx=dpsi_hat_dx,
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

    where A11 = G^H H G,
          A12 = -mu_0 gamma_s p_0 K
          A21 = 0
          A22 = mu_0 gamma_s p_0 G^H M G
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

def build_solver_blocks(
    C11: np.ndarray, O: np.ndarray, C12_2: np.ndarray, C12_3: np.ndarray, C22: np.ndarray,
    D11: np.ndarray, D12: np.ndarray, kappa_s: np.ndarray,
    K: np.ndarray, M: np.ndarray, 
    F11_2: np.ndarray, F12_1: np.ndarray, F12_2: np.ndarray, F21: np.ndarray, F22: np.ndarray,
    hprof: np.ndarray, Iprof: np.ndarray, gprof: np.ndarray, qprof: np.ndarray,
    pressure: np.ndarray, pressure_deriv: np.ndarray, density: np.ndarray, gamma_s: float,
    one_gpp: np.ndarray, gtp_over_gpp: np.ndarray, gpz_over_gpp: np.ndarray,
    H: np.ndarray, J: np.ndarray, L: np.ndarray,
    ntor: int, m_range: np.ndarray, max_dm: int=1000, max_dn: int=1000,
    dpsi_hat_dx: float = 1.0,
) -> SolverBlocks:
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
    dpsi_hat_dx = _validate_dpsi_hat_dx(dpsi_hat_dx)

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
    out = SolverBlocks(m_range=m_range, max_dm=max_dm, ntor=ntor,
                        C0=C0, C1=C1, D0=D0, D1=D1, F=F,
                        gtp_over_gpp=gtp_over_gpp_mat, gpz_over_gpp=gpz_over_gpp_mat,
                        hprof=hprof, Iprof=Iprof, gprof=gprof, qprof=qprof, 
                        pressure=pressure, pressure_deriv=pressure_deriv, density=density,
                        dpsi_hat_dx=dpsi_hat_dx,
                        A=A, B=B,
                        weights_q=None, points_q=None, phi_q=None, WD=None)
    out.max_dm = max_dm
    out.max_dn = max_dn
    return out

# -- Building the operator for a given frequency:
def build_M_at(sb: SolverBlocks, omega: float, regularizer: float=1e-3) -> coo_array:
    r"""
    Builds the full operator M in the space of (p1, xi^\psi, xi^s, \nabla\cdot\xi) for a given
    frequency omega.

    The operator is built as:
    M = [[C'     D]
         [F   - E]],

    where the direct angular phase is

    ``P = (g^{psi,theta}/g^{psi,psi})(-i*m)
          + (g^{psi,zeta}/g^{psi,psi})(+i*n)``

    for ``exp(i*(n*zeta-m*theta))``, and ``C' = C - P*I`` because this
    angular derivative is moved from the left-hand side to the radial RHS.

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
        _offset_coo_tensor(phase, 0, 0, xip1_shape, "phase", scale=-1.0),
        _offset_coo_tensor(
            phase,
            nmode,
            nmode,
            xip1_shape,
            "phase",
            scale=-1.0,
        ),
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

