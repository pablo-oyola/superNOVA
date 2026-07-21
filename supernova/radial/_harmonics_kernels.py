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


@njit(cache=True, nogil=True, inline="always")
def _solve_trigonometric_gram(
    field_cos: float,
    field_sin: float,
    cos_cos: float,
    sin_sin: float,
    cos_sin: float,
):
    """Return cosine/sine amplitudes, including self-conjugate grid modes."""
    determinant = cos_cos * sin_sin - cos_sin * cos_sin
    gram_scale = (cos_cos + sin_sin) * (cos_cos + sin_sin)
    tolerance = 100.0 * np.finfo(np.float64).eps * max(gram_scale, 1.0)
    if np.abs(determinant) > tolerance:
        cosine = (field_cos * sin_sin - field_sin * cos_sin) / determinant
        sine = (field_sin * cos_cos - field_cos * cos_sin) / determinant
        return cosine, sine

    # The constant and even-grid Nyquist modes have only one independent
    # sampled basis vector. Choose that vector explicitly instead of applying
    # the ordinary pi normalization and doubling its amplitude.
    if cos_cos >= sin_sin and cos_cos > tolerance:
        return field_cos / cos_cos, 0.0
    if sin_sin > tolerance:
        return 0.0, field_sin / sin_sin
    return 0.0, 0.0


@njit(parallel=True, cache=True, nogil=True)
def _decompose2d_impl(field: np.ndarray, m_max: int, angle: np.ndarray, weights: np.ndarray):
    coeffs_real = np.zeros((field.shape[0], m_max + 1))
    coeffs_imag = np.zeros((field.shape[0], m_max + 1))

    for i in prange(field.shape[0]):
        for m in range(m_max + 1):
            cos_m_theta = np.cos(m * angle)
            sin_m_theta = np.sin(m * angle)
            field_cos = _periodic_integral_1d(
                field[i] * cos_m_theta,
                weights,
            )
            field_sin = _periodic_integral_1d(
                field[i] * sin_m_theta,
                weights,
            )
            cos_cos = _periodic_integral_1d(
                cos_m_theta * cos_m_theta,
                weights,
            )
            sin_sin = _periodic_integral_1d(
                sin_m_theta * sin_m_theta,
                weights,
            )
            cos_sin = _periodic_integral_1d(
                cos_m_theta * sin_m_theta,
                weights,
            )
            coeffs_real[i, m], coeffs_imag[i, m] = _solve_trigonometric_gram(
                field_cos,
                field_sin,
                cos_cos,
                sin_sin,
                cos_sin,
            )

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
    toroidal_sign: int,
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
                cos_cos = 0.0
                sin_sin = 0.0
                cos_sin = 0.0

                # ``toroidal_sign=-1`` is the represented solver convention
                # x=m*theta-n*zeta. ``toroidal_sign=+1`` is used only to
                # detect the unsupported mixed-sign family x=m*theta+n*zeta.
                for itheta in range(field.shape[1]):
                    cos_mt = cos_m_theta[itheta]
                    sin_mt = sin_m_theta[itheta]
                    theta_weight = theta_weights[itheta]
                    for izeta in range(field.shape[2]):
                        cos_nz = cos_n_zeta[izeta]
                        sin_nz = sin_n_zeta[izeta]
                        weight = theta_weight * zeta_weights[izeta]
                        value = field[i, itheta, izeta]

                        cospart = (
                            cos_mt * cos_nz
                            - toroidal_sign * sin_mt * sin_nz
                        )
                        sinpart = (
                            sin_mt * cos_nz
                            + toroidal_sign * cos_mt * sin_nz
                        )
                        real_total += value * cospart * weight
                        imag_total += value * sinpart * weight
                        cos_cos += cospart * cospart * weight
                        sin_sin += sinpart * sinpart * weight
                        cos_sin += cospart * sinpart * weight

                (
                    coeffs_real[i, m, n],
                    coeffs_imag[i, m, n],
                ) = _solve_trigonometric_gram(
                    real_total,
                    imag_total,
                    cos_cos,
                    sin_sin,
                    cos_sin,
                )

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


def _decompose3d_with_toroidal_sign(
    field: np.ndarray,
    m_max: int = None,
    n_max: int = None,
    theta: np.ndarray | None = None,
    zeta: np.ndarray | None = None,
    toroidal_sign: int = -1,
):
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 3:
        raise ValueError("field must have shape (Nx, Ntheta, Nzeta).")
    if toroidal_sign not in (-1, 1):
        raise ValueError("toroidal_sign must be either -1 or +1.")

    if m_max is None:
        m_max = field.shape[1] // 2
    if n_max is None:
        n_max = field.shape[2] // 2

    theta = _prepare_periodic_angle(theta, field.shape[1], "theta")
    zeta = _prepare_periodic_angle(zeta, field.shape[2], "zeta")
    theta_weights = _periodic_weights(theta)
    zeta_weights = _periodic_weights(zeta)
    return _decompose3d_impl(
        field,
        m_max,
        n_max,
        theta,
        zeta,
        theta_weights,
        zeta_weights,
        toroidal_sign,
    )


def decompose3d(
    field: np.ndarray,
    m_max: int = None,
    n_max: int = None,
    theta: np.ndarray | None = None,
    zeta: np.ndarray | None = None,
):
    """
    Decompose ``F(Nx, Ntheta, Nzeta)`` in the represented family
    ``m*theta - n*zeta`` for non-negative ``m`` and ``n``.

    The returned cosine and sine arrays have shape
    ``(Nx, m_max + 1, n_max + 1)`` and contain one-sided real-field
    amplitudes, not two-sided complex Fourier coefficients.
    """
    return _decompose3d_with_toroidal_sign(
        field,
        m_max,
        n_max,
        theta,
        zeta,
        toroidal_sign=-1,
    )


def decompose3d_mixed_sign(
    field: np.ndarray,
    m_max: int = None,
    n_max: int = None,
    theta: np.ndarray | None = None,
    zeta: np.ndarray | None = None,
):
    """Project onto ``m*theta + n*zeta`` to detect unsupported helicities."""
    return _decompose3d_with_toroidal_sign(
        field,
        m_max,
        n_max,
        theta,
        zeta,
        toroidal_sign=1,
    )

@njit(parallel=True, cache=True, nogil=True)
def recompose2d(theta: float, coeffs_real: np.ndarray, coeffs_imag: np.ndarray) -> np.ndarray:
    """
    Reconstructs the original 2D field from its real cosine/sine amplitudes.
    
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
    Reconstructs the original 3D field from its real cosine/sine amplitudes.

    The phase is ``x=m*theta-n*zeta``.  Consequently this is equivalent to
    ``Re[(coeffs_real + 1j*coeffs_imag)*exp(i*(n*zeta-m*theta))]`` for each
    represented nonzero mode.
    
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
