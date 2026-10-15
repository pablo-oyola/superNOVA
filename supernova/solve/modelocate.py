r"""Gerrit Kramer's MODELOCATE mode filter.

A port of ``classify_mode`` from GJK's ``modelocate`` (version 6.0, June 2022),
the post-processor that sorted NOVA eigenmodes into gap modes and rejects
before anyone plotted them.  It needs only the radial displacement harmonics
and the radii where the mode frequency crosses the continuum, which makes it
the cheapest filter in the package: one pass over an envelope, with no
refinement levels and no tracking.

The envelope is ``sum_m |xi^psi_m|`` normalized to a peak of one.  From where
that peak sits, a mode is flagged as

``SPIKE``
    the peak is a single grid point: both neighbours are below
    ``spike_level`` of it;
``CORE``
    the peak lies at or inside ``core_edge``;
``EDGE``
    the peak lies at or outside ``boundary_edge``;
``CONTINUUM``
    the envelope exceeds ``continuum_level`` on either grid point bracketing
    a continuum crossing.

Flags combine as a bitmask, except that ``CONTINUUM`` replaces the others.  A
mode with no flag is a ``GAP`` mode.  A mode with ``Re(omega**2) < 0`` is a
purely growing ideal mode, ``IDEAL``, and is not tested further.

The radial thresholds are in ``rho_pol = sqrt(psi_N)``, NOVA's radial
coordinate.  ``SPIKE`` and ``CONTINUUM`` compare neighbouring grid points, so
they only mean something on the grid the mode was resolved on, not on an
arbitrarily fine resampling of it.

One deliberate difference from the C: ``modelocate`` advances to the next
crossing at most once per grid interval, so a second crossing in the same
interval is tested one point late.  Here every crossing is tested against its
own bracket.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import xarray as xr


# Bit flags, as in ``include.h``.
GAP = 0
SPIKE = 1
CORE = 2
EDGE = 4
CONTINUUM = 8
IDEAL = 16

_FLAG_NAMES = (
    (SPIKE, "spike"),
    (CORE, "core"),
    (EDGE, "edge"),
    (CONTINUUM, "continuum"),
    (IDEAL, "ideal"),
)


@dataclass(frozen=True)
class ModeLocateThresholds:
    """The constants of ``include.h``, under their names there."""

    core_edge: float = 0.05        # LOWEDGE, in rho_pol
    boundary_edge: float = 0.96    # HIGEDGE, in rho_pol
    continuum_level: float = 0.9   # REJECT_AMPL
    spike_level: float = 0.7       # REJECT_SPIKE


def mode_label(code: int) -> str:
    """``"gap"``, or the flags that are set joined by ``|``."""
    names = [name for flag, name in _FLAG_NAMES if code & flag]
    return "|".join(names) if names else "gap"


def branch_crossings(psi: np.ndarray, branches: np.ndarray,
                     omega: complex) -> np.ndarray:
    """Sorted radii where a continuum branch passes through ``|Re(omega)|``.

    ``branches`` is ``(npsi, nbranch)``, each column one tracked continuum
    curve, as NOVA's are.  Crossings are found along each branch rather than
    by pairing the sorted frequency sets of neighbouring radii, which goes
    wrong whenever the number of finite roots changes from one radius to the
    next -- as the degenerate sound roots of a ``gamma_s = 0`` pencil do.
    """
    psi = np.asarray(psi, dtype=float)
    offset = np.abs(np.real(np.asarray(branches))) - abs(np.real(omega))
    left, right = offset[:-1], offset[1:]
    with np.errstate(invalid="ignore"):
        cut = np.isfinite(left) & np.isfinite(right) & (left * right <= 0.0) & (left != right)
    index, branch = np.nonzero(cut)
    fraction = left[index, branch] / (left[index, branch] - right[index, branch])
    return np.sort(psi[index] + fraction * (psi[index + 1] - psi[index]))


def envelope(harmonics: np.ndarray) -> np.ndarray:
    """``sum_m |xi_m|`` over the last axis, normalized to a peak of one."""
    total = np.sum(np.abs(np.asarray(harmonics)), axis=-1)
    peak = np.max(total, axis=-1, keepdims=True)
    return np.divide(total, peak, out=np.zeros_like(total), where=peak > 0.0)


def classify_mode(
    rho: np.ndarray,
    profile: np.ndarray,
    crossings: np.ndarray,
    omega: complex,
    thresholds: ModeLocateThresholds | None = None,
) -> int:
    """Classify one mode from its envelope on an increasing ``rho`` grid."""
    thresholds = ModeLocateThresholds() if thresholds is None else thresholds
    if np.real(complex(omega) ** 2) < 0.0:
        return IDEAL

    rho = np.asarray(rho, dtype=float)
    profile = np.asarray(profile, dtype=float)
    imax = int(np.argmax(profile))
    peak = profile[imax]

    code = GAP
    if 0 < imax < profile.size - 1 and max(
        profile[imax - 1], profile[imax + 1]
    ) < thresholds.spike_level * peak:
        code |= SPIKE
    if rho[imax] <= thresholds.core_edge:
        code |= CORE
    if rho[imax] >= thresholds.boundary_edge:
        code |= EDGE

    # A crossing at or past the last grid point is never bracketed.
    crossings = np.asarray(crossings, dtype=float)
    crossings = crossings[crossings < rho[-1]]
    if crossings.size:
        right = np.clip(np.searchsorted(rho, crossings, side="right"),
                        1, rho.size - 1)
        bracket = np.maximum(profile[right - 1], profile[right])
        if np.any(bracket > thresholds.continuum_level * peak):
            code = CONTINUUM
    return code


def locate_modes(
    x: np.ndarray,
    harmonics: np.ndarray,
    omega: np.ndarray,
    crossings: Sequence[np.ndarray],
    *,
    rho: np.ndarray | None = None,
    thresholds: ModeLocateThresholds | None = None,
) -> xr.Dataset:
    """Classify a set of modes sampled on one radial grid.

    Parameters
    ----------
    x:
        Increasing radial grid, shape ``(nx,)``, in the coordinate the
        ``crossings`` are given in.
    harmonics:
        ``xi^psi`` per poloidal harmonic, shape ``(nmode, nx, nm)``.
    omega:
        Mode frequencies, shape ``(nmode,)``.
    crossings:
        For each mode, the ``x`` where its frequency crosses the continuum.
    rho:
        ``rho_pol`` at ``x``, where the radial thresholds apply.  Defaults to
        ``sqrt(x)``, which is right when ``x`` is ``psi_N``.
    """
    thresholds = ModeLocateThresholds() if thresholds is None else thresholds
    x = np.asarray(x, dtype=float)
    rho = np.sqrt(np.clip(x, 0.0, None)) if rho is None else np.asarray(rho, dtype=float)
    omega = np.atleast_1d(np.asarray(omega, dtype=np.complex128))
    profiles = envelope(harmonics)
    if profiles.shape != (omega.size, x.size) or len(crossings) != omega.size:
        raise ValueError(
            f"Expected harmonics (nmode, nx, nm) with nmode={omega.size} and "
            f"nx={x.size}, and one crossing array per mode; got harmonics "
            f"{np.shape(harmonics)} and {len(crossings)} crossing arrays."
        )

    codes = np.array([
        classify_mode(rho, profile, np.interp(np.asarray(cut, dtype=float), x, rho),
                      w, thresholds)
        for profile, cut, w in zip(profiles, crossings, omega)
    ], dtype=np.int32)
    imax = np.argmax(profiles, axis=1)

    located = xr.Dataset(
        data_vars={
            "mode_class": ("eigenpair", codes),
            "mode_label": ("eigenpair", np.array([mode_label(c) for c in codes])),
            "peak_x": ("eigenpair", x[imax]),
            "peak_rho": ("eigenpair", rho[imax]),
            "envelope": (("eigenpair", "x"), profiles),
        },
        coords={"eigenpair": np.arange(omega.size), "x": x, "rho": ("x", rho)},
        attrs={"method": "modelocate", **asdict(thresholds)},
    )
    located["mode_class"].attrs["legend"] = (
        "0 gap, 1 spike, 2 core, 4 edge, 8 continuum, 16 ideal; bits combine "
        "except continuum, which replaces the rest"
    )
    located["envelope"].attrs["description"] = (
        "sum_m |xi^psi_m| normalized to a peak of one"
    )
    return located


__all__ = [
    "CONTINUUM",
    "CORE",
    "EDGE",
    "GAP",
    "IDEAL",
    "SPIKE",
    "ModeLocateThresholds",
    "branch_crossings",
    "classify_mode",
    "envelope",
    "locate_modes",
    "mode_label",
]
