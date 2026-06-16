"""
We build using numba/cupy/accelerator the matrices to solve the continuum spectrum.

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

# -------------------------------------
# General kernels.  
# -------------------------------------

# @njit(parallel=True, cache=True, nogil=True)
def _fill_block_kernel(
    harmonics: np.ndarray,
    mode_m: np.ndarray,
    mode_n: np.ndarray,
    out_view: np.ndarray,
    conjugate_mode: bool = False
):
    """
    Core kernel to fill a specific block of the result matrix.
    Calculates differences on-the-fly to save memory.

    With the paper basis ``exp(i*(m*theta - n*zeta))`` and the Galerkin
    bracket ``<row|A|col>``, entry ``(i, j)`` uses the stored coefficient at
    ``(m_i - m_j, n_i - n_j)``.
    """
    n_i = mode_m.size
    n_j = mode_m.size
    mmax = harmonics.shape[0] - 1
    nmax = harmonics.shape[1] - 1

    for i in prange(n_i):
        mi, ni = mode_m[i], mode_n[i]
        for j in range(n_j):
            # dm[i, j] = m_i - m_j
            dmi = mi - mode_m[j]
            dni = ni - mode_n[j]
            
            val = 0.0 + 0.0j
            if dmi >= 0 and dni >= 0:
                if dmi <= mmax and dni <= nmax:
                    val = harmonics[dmi, dni]
            elif dmi <= 0 and dni <= 0:
                dmp, dnp = -dmi, -dni
                if dmp <= mmax and dnp <= nmax:
                    val = np.conjugate(harmonics[dmp, dnp])
            
            if conjugate_mode:
                out_view[i, j] = np.conjugate(val)
            else:
                out_view[i, j] = val

# @njit(parallel=True, cache=True, nogil=True, inline='always')
def _apply_ghg_scaling(matrix: np.ndarray, qval: float, 
                      n_modes: np.ndarray, 
                      m_modes: np.ndarray, scale: float=1.0):
    """
    In-place G^H * Matrix * G scaling.
    """
    n = m_modes.size
    g_diag = qval * n_modes - m_modes
    for i in prange(n):
        for j in range(n):
            matrix[i, j] *= (g_diag[i] * g_diag[j]) * scale

@njit(cache=True, nogil=True)
def get_active_bounds(table, epsilon=1e-15):
    """Finds the min/max m and n indices that are non-zero."""
    m_max_idx, n_max_idx = table.shape
    
    m_min, m_max = m_max_idx, 0
    n_min, n_max = n_max_idx, 0
    
    # We only need to scan the stored (nonnegative) table
    for m in range(m_max_idx):
        for n in range(n_max_idx):
            if abs(table[m, n]) > epsilon:
                if m < m_min: m_min = m
                if m > m_max: m_max = m
                if n < n_min: n_min = n
                if n > n_max: n_max = n
                
    # If the table is all zeros
    if m_min > m_max:
        return 0, 0, 0, 0
    return m_min, m_max, n_min, n_max

# @njit(parallel=True, cache=True, nogil=True)
def _fill_convolution_block_kernel(
    table_A, table_B, mode_m, mode_n, out_view, scale=1.0
):
    """Convolution with pre-calculated active index bounds."""
    n_i = mode_m.size
    n_j = mode_m.size
    
    # 1. Get bounds for both tables
    # Note: Because of conjugate symmetry, if (m,n) is active, (-m,-n) is also active.
    # We use the symmetry to define the symm_ range.
    _, m_max_a, _, n_max_a = get_active_bounds(table_A)
    _, m_max_b, _, n_max_b = get_active_bounds(table_B)

    for i in prange(n_i):
        mi, ni = mode_m[i], mode_n[i]
        for j in range(n_j):
            dmi = mi - mode_m[j]
            dni = ni - mode_n[j]
            
            conv_val = 0.0 + 0.0j
            
            # 2. Iterate only over the active range of table_B
            # We must check both positive and negative frequencies
            for mb in range(-m_max_b, m_max_b + 1):
                # Optimization: check if 'ma' could even exist in table_A
                ma = dmi - mb
                if abs(ma) > m_max_a:
                    continue
                    
                for nb in range(-n_max_b, n_max_b + 1):
                    na = dni - nb
                    if abs(na) > n_max_a:
                        continue
                    
                    # 3. Perform lookup
                    val_b = get_harmonic_value(table_B, mb, nb)
                    if val_b == 0: continue
                    
                    val_a = get_harmonic_value(table_A, ma, na)
                    if val_a == 0: continue
                    
                    conv_val += val_a * val_b
            
            out_view[i, j] = conv_val * scale

# -------------------------------------
# Compiled general kernels for parallel execution.  
# -------------------------------------
fill_block_kernel = njit(parallel=True, cache=True, nogil=True)(_fill_block_kernel)
apply_ghg_scaling = njit(parallel=True, cache=True, nogil=True, inline='always')(_apply_ghg_scaling)
fill_convolution_block_kernel = njit(parallel=True, cache=True, nogil=True, inline='always')(_fill_convolution_block_kernel)

@njit(inline='always')
def get_harmonic_value(table, m, n):
    """Helper to handle the nonnegative storage lookup."""
    m_max, n_max = table.shape[0] - 1, table.shape[1] - 1
    if m >= 0 and n >= 0:
        if m <= m_max and n <= n_max:
            return table[m, n]
    elif m <= 0 and n <= 0:
        if -m <= m_max and -n <= n_max:
            return np.conjugate(table[-m, -n])
    return 0.0 + 0.0j

@njit(cache=True, nogil=True)
def assemble_full_matrices(
    H_mn: np.ndarray, 
    K_mn: np.ndarray,
    J_mn: np.ndarray, 
    L_mn: np.ndarray,
    M_mn: np.ndarray,
    m_range: np.ndarray,
    ntor: int,
    gammas: float,
    q: float,
    p0: float
):
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
    assert np.all(H_mn.shape == K_mn.shape), "H and K must have the same shape"
    assert np.all(H_mn.shape == J_mn.shape), "H and J must have the same shape"
    assert np.all(H_mn.shape == L_mn.shape), "H and L must have the same shape"
    assert np.all(H_mn.shape == M_mn.shape), "H and M must have the same shape"
    # assert np.all(H_mn.size == m_range.size//2), "H and m_range must have the same size"

    n_range = ntor + np.zeros_like(m_range) # n_range will assume no toroidal coupling.
    nmode = m_range.size
    two_n = 2 * nmode

    # 2. Pre-allocate the full matrices (one-time allocation)
    A_out = np.zeros((two_n, two_n), dtype=np.complex128)
    B_out = np.zeros((two_n, two_n), dtype=np.complex128)

    # 3. Fill A blocks via views (no copies made)
    # A11 (Top-Left)
    fill_block_kernel(H_mn, m_range, n_range, A_out[:nmode, :nmode])
    apply_ghg_scaling(A_out[:nmode, :nmode], q, n_range, 
                      m_range)  # Apply G^H * A11 * G scaling
    
    # A12 (Top-Right)
    # A12 = - gamma_s * (pressure_mn * K_mn)
    fill_block_kernel(K_mn, m_range, n_range, A_out[:nmode, nmode:], conjugate_mode=False)
    A_out[:nmode, nmode:] *= - (gammas * mu_0 * p0)
    
    # A22 (Bottom-Right)
    fill_block_kernel(M_mn, m_range, n_range, A_out[nmode:, nmode:])
    apply_ghg_scaling(A_out[nmode:, nmode:], q, n_range,
                      m_range, scale=p0*gammas*mu_0)  # Apply G^H * M * G scaling

    # 4. Fill B blocks via views
    # B11 (Top-Left)
    fill_block_kernel(J_mn, m_range, n_range, B_out[:nmode, :nmode])
    # B21 (Bottom-Left)
    fill_block_kernel(K_mn, m_range, n_range, B_out[nmode:, :nmode])
    # B22 (Bottom-Right)
    fill_block_kernel(L_mn, m_range, n_range, B_out[nmode:, nmode:])

    return A_out, B_out

@njit(cache=True, nogil=True)
def assembly_slow_sound(H_mn: np.ndarray,  J_mn: np.ndarray, 
                        N_mn: np.ndarray,  q: float,
                        m_range: np.ndarray,
                        ntor: int):
    """
    Assembles the slow sound approximation matrices to determine the
    Shear-Alfven continuum. In this approximation:

    A = G*HG + N
    B = J

    where G = (nq - m)\\delta_{mm'}\\delta_{nn'}
          H = gpsi_psi/(jac*B0**2)
          N = jac/B0**2 * (gamma_s*p0*mu_0*kappa_s**2/(gamma_s*p0*mu_0 + B0**2))
          J = gpsi_psi*jac/B0**2
    """
    # Checking all the inputs have the same shape (they should be 
    # profiles evaluated at the same radial point with the same number of
    # harmonics).
    assert H_mn.shape == J_mn.shape, "H and J must have the same shape"
    assert H_mn.shape == N_mn.shape, "H and N must have the same shape"
    # assert np.all(H_mn.size == m_range.size), "H and m_range must have the same size"

    # 1. Setup mode vectors
    # n_range is assuming n is 0 which is valid only for tokamaks
    n_range = np.zeros_like(m_range) + ntor
    
    nmode = m_range.size
    two_n = nmode

    # 2. Pre-allocate the full matrices (one-time allocation)
    A_out = np.zeros((two_n, two_n), dtype=np.complex128)
    B_out = np.zeros((two_n, two_n), dtype=np.complex128)

    # 3. Fill the A matrix with the GHG
    fill_block_kernel(H_mn, m_range, n_range, A_out)
    apply_ghg_scaling(A_out, q, n_range, m_range)

    # 4. Adding the N-matrix.
    fill_block_kernel(N_mn, m_range, n_range, B_out)
    A_out += B_out

    # 5. Fill the B matrix with J
    fill_block_kernel(J_mn, m_range, n_range, B_out)

    return A_out, B_out

def assembly_only_saw(H_mn: np.ndarray,  J_mn: np.ndarray, 
                      q: float, m_range: np.ndarray, ntor: int):
    """
    Assembles the slow sound approximation matrices to determine the
    Shear-Alfven continuum. In this approximation:

    A = G*HG
    B = J

    where G = (nq - m)\\delta_{mm'}\\delta_{nn'}
          H = gpsi_psi/(jac*B0**2)
          J = gpsi_psi*jac/B0**2
    """
    # Checking all the inputs have the same shape (they should be 
    # profiles evaluated at the same radial point with the same number of
    # harmonics).
    assert H_mn.shape == J_mn.shape, "H and J must have the same shape"
    # assert np.all(H_mn.size == m_range.size), "H and m_range must have the same size"

    # 1. Setup mode vectors
    # n_range is assuming n is 0 which is valid only for tokamaks
    n_range = np.zeros_like(m_range) + ntor
    
    nmode = m_range.size
    two_n = nmode

    # 2. Pre-allocate the full matrices (one-time allocation)
    A_out = np.zeros((two_n, two_n), dtype=np.complex128)
    B_out = np.zeros((two_n, two_n), dtype=np.complex128)

    # 3. Fill the A matrix with the GHG
    fill_block_kernel(H_mn, m_range, n_range, A_out)
    apply_ghg_scaling(A_out, q, n_range, m_range)

    # 5. Fill the B matrix with J
    fill_block_kernel(J_mn, m_range, n_range, B_out)

    return A_out, B_out
