"""Targets, search regions and residual reporting.

The eigensolvers work in ``s = omega**2``, so a frequency window becomes an
ellipse in the complex ``s`` plane.  These helpers build that region, coerce
scalar targets, and format residual norms for the result datasets.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def _as_scalar_complex(value, name: str) -> complex:
    """
    Convert a scalar or one-element array to a Python complex scalar.
    """
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"{name} must be a scalar or one-element array, got shape {arr.shape}.")
    return complex(arr.reshape(-1)[0])


def _format_complex(value: complex) -> str:
    """
    Compact scalar formatting for verbose solver diagnostics.
    """
    value = complex(value)
    if abs(value.imag) < 1e-14:
        return f"{value.real:.16g}"
    return f"{value.real:.16g}{value.imag:+.16g}j"


def _ellipse_region_dict(region: dict | None, target_s: complex) -> dict:
    """
    Normalize and validate the v1 ellipse region description.
    """
    if target_s == 0.0:
        raise ValueError("target_omega must be nonzero because the split NEP contains 1/s.")

    if region is None:
        region = {
            "type": "ellipse",
            "center": target_s,
            "radius": 0.25 * abs(target_s),
            "vscale": 0.25,
        }

    if not isinstance(region, dict):
        raise ValueError("region must be a dictionary.")
    region_type = region.get("type", "ellipse")
    if str(region_type).lower() != "ellipse":
        raise ValueError("solver3 v1 supports only ellipse regions.")

    center = _as_scalar_complex(region.get("center", target_s), "region['center']")
    radius = float(region.get("radius", 0.25 * abs(target_s)))
    vscale = float(region.get("vscale", 1.0))
    if radius <= 0.0:
        raise ValueError("ellipse region radius must be positive.")
    if vscale <= 0.0:
        raise ValueError("ellipse region vscale must be positive.")

    scaled_zero_distance = (center.real / radius) ** 2 + (center.imag / (radius * vscale)) ** 2
    if scaled_zero_distance <= 1.0:
        raise ValueError("ellipse region must not include the pole at s=0.")

    return {
        "type": "ellipse",
        "center": center,
        "radius": radius,
        "vscale": vscale,
    }


def _split_residual_norm(
    T0: csr_matrix,
    T1: csr_matrix,
    Tminus1: csr_matrix,
    eigenvalue: complex,
    eigenvector: np.ndarray,
) -> tuple[float, float]:
    """
    Return absolute and vector-normalized residual norms for ``T(s) x``.
    """
    if eigenvalue == 0.0:
        return np.inf, np.inf

    residual = T0 @ eigenvector
    residual = residual + eigenvalue * (T1 @ eigenvector)
    residual = residual + (1.0 / eigenvalue) * (Tminus1 @ eigenvector)
    abs_norm = float(np.linalg.norm(residual))
    rel_norm = abs_norm / max(1.0, float(np.linalg.norm(eigenvector)))
    return abs_norm, rel_norm
