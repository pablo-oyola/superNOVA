r"""Radial breakpoints graded towards chosen radii.

A mode that lives in a continuum gap can be far narrower than a uniform grid's
elements -- a core TAE at low shear is a few millimetres wide on a grid of
several -- and its frequency only converges once the elements around it are a
fraction of its width.  Refining everywhere is wasteful; these breakpoints are
``base`` wide far from the centres and shrink linearly to ``local`` at each of
them, so neighbouring elements never differ by more than a factor
``1 / (1 - growth)``.
"""

from __future__ import annotations

import numpy as np


def graded_breakpoints(domain: tuple[float, float], base: float,
                       centers, local, growth: float = 0.15,
                       max_elements: int = 20000) -> np.ndarray:
    r"""Breakpoints whose spacing is ``min(base, local_c + growth*|x - x_c|)``.

    Parameters
    ----------
    domain
        ``(x0, x1)``; the first and last breakpoints are exactly these.
    base
        Element size away from every centre.
    centers, local
        Where to refine, and the element size wanted at each.
    growth
        How fast the element size relaxes back to ``base`` with distance, in
        ``(0, 1)``; neighbours differ by at most ``1 / (1 - growth)``.
    """
    lo, hi = (float(domain[0]), float(domain[1]))
    if not hi > lo:
        raise ValueError("The domain must satisfy x0 < x1.")
    if not base > 0.0 or not 0.0 < growth < 1.0:
        raise ValueError("base must be positive and growth in (0, 1).")
    centers = np.atleast_1d(np.asarray(centers, dtype=float))
    local = np.broadcast_to(np.asarray(local, dtype=float), centers.shape)
    if np.any(local <= 0.0):
        raise ValueError("The local element sizes must be positive.")

    def size(x: float) -> float:
        if centers.size == 0:
            return base
        return float(min(base, np.min(local + growth * np.abs(x - centers))))

    points = [lo]
    while points[-1] < hi:
        if len(points) > max_elements:
            raise ValueError(f"The graded mesh needs more than {max_elements} elements; "
                             "ask for a coarser local size.")
        points.append(points[-1] + size(points[-1]))
    points = np.asarray(points)
    # stretch the overshooting last step back onto the wall
    points = lo + (points - lo) * (hi - lo) / (points[-1] - lo)
    points[0], points[-1] = lo, hi
    return points


__all__ = ["graded_breakpoints"]
