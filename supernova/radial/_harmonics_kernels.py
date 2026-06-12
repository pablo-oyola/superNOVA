"""
Implements using numba the kernels for the 2D and 3D Fourier decomposition of the profiles in the (Theta, Zeta) plane.
"""
"""
Accelerated kernels to decompose fields into (n, m) harmonics.
"""

import numpy as np

# Importing the accelerators.
from .._utils import njit, prange


@njit(cache=True, nogil=True)
def _periodic_weights(angle: np.ndarray) -> np.ndarray:
    """Quadrature weights for a periodic grid, with or without a duplicated endpoint."""
    npts = angle.size
    weights = np.empty(npts, dtype=np.float64)

    if npts == 1:
        weights[0] = 2.0 * np.pi
        return weights

    forward = np.empty(npts, dtype=np.float64)
    for i in range(npts - 1):
        forward[i] = angle[i + 1] - angle[i]
    forward[-1] = angle[0] + 2.0 * np.pi - angle[-1]

    previous = forward[-1]
    for i in range(npts):
        weights[i] = 0.5 * (previous + forward[i])
        previous = forward[i]

    return weights


@njit(cache=True, nogil=True)
def _periodic_integral_1d(y: np.ndarray, weights: np.ndarray) -> float:
    total = 0.0
    for i in range(y.size):
        total += y[i] * weights[i]
    return total


@njit(parallel=True, cache=True, nogil=True)
def _decompose2d_impl(field: np.ndarray, m_max: int, angle: np.ndarray, weights: np.ndarray):
    coeffs_real = np.zeros((field.shape[0], m_max + 1))
    coeffs_imag = np.zeros((field.shape[0], m_max + 1))

    for i in prange(field.shape[0]):
        for m in range(m_max + 1):
            cos_m_theta = np.cos(m * angle)
            sin_m_theta = np.sin(m * angle)
            coeffs_real[i, m] = _periodic_integral_1d(field[i] * cos_m_theta, weights)
            coeffs_imag[i, m] = _periodic_integral_1d(field[i] * sin_m_theta, weights)

    for m in range(m_max + 1):
        if m == 0:
            coeffs_real[:, m] /= (2.0 * np.pi)
            coeffs_imag[:, m] /= (2.0 * np.pi)
        else:
            coeffs_real[:, m] /= np.pi
            coeffs_imag[:, m] /= np.pi

    return coeffs_real, coeffs_imag


@njit(parallel=True, cache=True, nogil=True)
def _decompose3d_impl(
    field: np.ndarray,
    m_max: int,
    n_max: int,
    theta: np.ndarray,
    zeta: np.ndarray,
    theta_weights: np.ndarray,
    zeta_weights: np.ndarray,
):
    coeffs_real = np.zeros((field.shape[0], m_max + 1, n_max + 1))
    coeffs_imag = np.zeros((field.shape[0], m_max + 1, n_max + 1))

    for i in prange(field.shape[0]):
        for m in range(m_max + 1):
            cos_m_theta = np.cos(m * theta)
            sin_m_theta = np.sin(m * theta)
            for n in range(n_max + 1):
                cos_n_zeta = np.cos(n * zeta)
                sin_n_zeta = np.sin(n * zeta)
                real_total = 0.0
                imag_total = 0.0

                # Paper convention: exp(i*(m*theta - n*zeta)).
                # Decompose in cos(m*theta - n*zeta) and sin(m*theta - n*zeta)
                # using cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
                # and sin(a - b) = sin(a)cos(b) - cos(a)sin(b).
                for itheta in range(field.shape[1]):
                    cos_mt = cos_m_theta[itheta]
                    sin_mt = sin_m_theta[itheta]
                    theta_weight = theta_weights[itheta]
                    for izeta in range(field.shape[2]):
                        cos_nz = cos_n_zeta[izeta]
                        sin_nz = sin_n_zeta[izeta]
                        weight = theta_weight * zeta_weights[izeta]
                        value = field[i, itheta, izeta]

                        cospart = cos_mt * cos_nz + sin_mt * sin_nz
                        sinpart = sin_mt * cos_nz - cos_mt * sin_nz
                        real_total += value * cospart * weight
                        imag_total += value * sinpart * weight

                coeffs_real[i, m, n] = real_total
                coeffs_imag[i, m, n] = imag_total

    for m in range(m_max + 1):
        for n in range(n_max + 1):
            if m == 0 and n == 0:
                coeffs_real[:, m, n] /= (4.0 * np.pi**2)
                coeffs_imag[:, m, n] /= (4.0 * np.pi**2)
            else:
                coeffs_real[:, m, n] /= (2.0 * np.pi**2)
                coeffs_imag[:, m, n] /= (2.0 * np.pi**2)

    return coeffs_real, coeffs_imag


def _prepare_periodic_angle(angle: np.ndarray | None, size: int, name: str) -> np.ndarray:
    """Validate or construct a strictly increasing periodic angle grid."""
    if angle is None:
        return np.linspace(0.0, 2.0 * np.pi, size, endpoint=False, dtype=np.float64)

    angle = np.asarray(angle, dtype=np.float64)
    if angle.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if angle.size != size:
        raise ValueError(f"{name} size must match the field dimension ({size}).")
    if angle.size == 0:
        raise ValueError(f"{name} must contain at least one point.")
    if angle.size == 1:
        return angle

    diffs = np.diff(angle)
    if np.any(diffs <= 0.0):
        raise ValueError(f"{name} must be strictly increasing and must not repeat the periodic endpoint.")

    wrap = angle[0] + 2.0 * np.pi - angle[-1]
    # Accept both endpoint-exclusive grids and endpoint-inclusive grids that
    # duplicate the periodic endpoint. The latter gives wrap ~= 0 and is
    # handled naturally by the periodic quadrature weights.
    tol = 1e-12
    if wrap < -tol:
        raise ValueError(f"{name} must not span more than one periodic interval of length 2*pi.")

    return angle


def decompose2d(field: np.ndarray, m_max: int = None, angle: np.ndarray | None = None):
    """
    Computes using the trapezoidal rule the Fourier decomposition along the
    poloidal angle.

    The input field is assumed to have  (Nx, Ntheta) shape and span over
    [0, 2pi) in the poloidal angle. This automatically computes the Fourier modes
    in the poloidal direction, assuming a periodic domain. The output is a 2D array of 
    shape (Nx, m_max+1) containing the Fourier coefficients for each mode m, where m ranges 
    from 0 to m_max. If m_max is None, it defaults to Ntheta//2, which corresponds to 
    the Nyquist frequency.

    Parameters
    ----------
    field : np.ndarray
        2D array of shape (Nx, Ntheta) representing the field to be decomposed.
    m_max : int, optional
        Maximum mode number to compute. If None, defaults to Ntheta//2.
    
    Returns
    -------
    np.ndarray
        2D array of shape (Nx, m_max+1) containing the Fourier coefficients for each mode m - Real part.
    np.ndarray
        2D array of shape (Nx, m_max+1) containing the Fourier coefficients for each mode m - Imaginary part.
    """
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError("field must have shape (Nx, Nangle).")

    if m_max is None:
        m_max = field.shape[1] // 2

    angle = _prepare_periodic_angle(angle, field.shape[1], "angle")
    weights = _periodic_weights(angle)
    return _decompose2d_impl(field, m_max, angle, weights)


def decompose3d(
    field: np.ndarray,
    m_max: int = None,
    n_max: int = None,
    theta: np.ndarray | None = None,
    zeta: np.ndarray | None = None,
):
    """
    For an input F(Nx, Ntheta, Nzeta) this computes for each (n, m) the Fourier decomposition along the poloidal and toroidal angles.
    The output is a 3D array of shape (Nx, m_max+1, n_max+1) containing the Fourier coefficients for each mode (n, m), where m ranges from 0 to m_max and n ranges from 0 to n_max. If m_max or n_max are None, they default to Ntheta//2 and Nzeta//2, respectively, which correspond to the Nyquist frequencies.
    
    Parameters
    ----------
    field : np.ndarray
        3D array of shape (Nx, Ntheta, Nzeta) representing the field to be decomposed.
    m_max : int, optional
        Maximum mode number in the poloidal direction. If None, defaults to Ntheta//2.
    n_max : int, optional
        Maximum mode number in the toroidal direction. If None, defaults to Nzeta//2.

    Returns
    -------
    np.ndarray
        3D array of shape (Nx, m_max+1, n_max+1) containing the Fourier coefficients for each mode (n, m) - Real part.
    np.ndarray
        3D array of shape (Nx, m_max+1, n_max+1) containing the Fourier coefficients for each mode (n, m) - Imaginary part.
    """
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 3:
        raise ValueError("field must have shape (Nx, Ntheta, Nzeta).")

    if m_max is None:
        m_max = field.shape[1] // 2
    if n_max is None:
        n_max = field.shape[2] // 2

    theta = _prepare_periodic_angle(theta, field.shape[1], "theta")
    zeta = _prepare_periodic_angle(zeta, field.shape[2], "zeta")
    theta_weights = _periodic_weights(theta)
    zeta_weights = _periodic_weights(zeta)
    return _decompose3d_impl(field, m_max, n_max, theta, zeta, theta_weights, zeta_weights)

@njit(parallel=True, cache=True, nogil=True)
def recompose2d(theta: float, coeffs_real: np.ndarray, coeffs_imag: np.ndarray) -> np.ndarray:
    """
    Reconstructs the original 2D field from its Fourier coefficients.
    
    Parameters
    ----------
    coeffs_real : np.ndarray
        2D array of shape (Nx, m_max+1) containing the real parts of the Fourier coefficients.
    coeffs_imag : np.ndarray
        2D array of shape (Nx, m_max+1) containing the imaginary parts of the Fourier coefficients.
    
    Returns
    -------
    np.ndarray
        2D array of shape (Nx, Ntheta) representing the reconstructed field.
    """
    field = np.zeros((coeffs_real.shape[0], theta.size))

    for i in prange(coeffs_real.shape[0]):
        for m in range(coeffs_real.shape[1]):
            field[i, :] += coeffs_real[i, m] * np.cos(m * theta) + coeffs_imag[i, m] * np.sin(m * theta)

    return field

@njit(parallel=True, cache=True, nogil=True)
def recompose3d(theta: float, zeta: float, coeffs_real: np.ndarray, coeffs_imag: np.ndarray) -> np.ndarray:
    """
    Reconstructs the original 3D field from its Fourier coefficients.
    
    Parameters
    ----------
    coeffs_real : np.ndarray
        3D array of shape (Nx, m_max+1, n_max+1) containing the real parts of the Fourier coefficients.
    coeffs_imag : np.ndarray
        3D array of shape (Nx, m_max+1, n_max+1) containing the imaginary parts of the Fourier coefficients.
    """
    field = np.zeros((coeffs_real.shape[0], theta.size, zeta.size))

    gtht, gzeta = np.meshgrid(theta, zeta, indexing='ij')

    for i in prange(coeffs_real.shape[0]):
        for m in range(coeffs_real.shape[1]):
            for n in range(coeffs_real.shape[2]):
                cospart = np.cos(m * gtht) * np.cos(n * gzeta) + np.sin(m * gtht) * np.sin(n * gzeta)
                sinpart = np.sin(m * gtht) * np.cos(n * gzeta) - np.cos(m * gtht) * np.sin(n * gzeta)
                field[i, :, :] += coeffs_real[i, m, n] * cospart + coeffs_imag[i, m, n] * sinpart

    return field
