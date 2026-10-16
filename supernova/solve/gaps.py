r"""Continuum gaps, and how wide the modes that live in them are.

A gap mode is resolved only once the radial elements are a fraction of two
lengths, and both can be read off before any eigenvalue solve:

* the mode's width.  With ``r0 = rho_tor * a`` at the gap, ``m`` its poloidal
  number and ``eps = r0/R0``, a TAE-like mode is about

      w = (r0/m) * eps/|s|            at finite shear,
      w = (r0/m) * sqrt(eps/|s|)      once |s| < eps (low shear),

  i.e. ``w = (r0/m) * min(eps/|s|, sqrt(eps/|s|))``;
* the distance of the singular points of the nearest continuum extremum.  A
  frequency ``f`` on the open side of an extremum ``f_tip`` (above a maximum,
  below a minimum) meets that branch at a complex pair a distance

      d(f) = sqrt(2 |f**2 - f_tip**2| / |(f**2)''|)

  from the real radial axis; it goes to zero at the gap edges, and a mode near
  an edge has radial structure on that scale.

On the ITPA n=6 and DIII-D n=3 cases, ``h <= min(w, d)`` gave about 1e-4 in
frequency and ``h <= min(w/4, d/2)`` about 1e-5.

The same distance limits the global complex path of
:mod:`supernova.operators.contour`: an eigenvalue stays invariant while the
path height at every extremum it sits next to is below about ``d/2``.
:func:`path_alpha` turns that into the largest safe ``alpha``.

Everything here works on the solver's own radial coordinate ``x``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GAP_NAMES = {1: "TAE", 2: "EAE", 3: "NAE"}


# ----------------------------------------------------------------------------
# Geometry along the solver's radial coordinate
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class RadialGeometry:
    """``q``, ``r = rho_tor * a``, shear and inverse aspect ratio along ``x``."""

    x: np.ndarray            # solver coordinate, increasing
    psi_n: np.ndarray        # physical normalized poloidal flux at x
    r: np.ndarray            # rho_tor * a [m]
    q: np.ndarray
    aminor: float            # [m]
    rmajor: float            # [m]

    @classmethod
    def from_equilibrium(cls, equilibrium) -> "RadialGeometry":
        data = equilibrium.qprof.data
        for dim in data.dims:
            if dim != "x":
                data = data.isel({dim: 0})
        x = np.asarray(data.coords["x"].values, dtype=float)
        q = np.abs(np.real(np.asarray(data.values)))
        magspace = equilibrium.magspace
        grid = np.asarray(magspace.coords["psi"].values, dtype=float)
        if "psi_physical_norm" in magspace.coords:
            psi_n = np.interp(x, grid, np.asarray(magspace.coords["psi_physical_norm"].values,
                                                  dtype=float))
        else:
            psi_n = x.copy()
        # toroidal flux: dPhi = q dpsi, the core and edge outside the Boozer
        # domain carried at their end values of q
        phi = q[0] * psi_n[0] + np.concatenate(
            [[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(psi_n))])
        phi_edge = phi[-1] + q[-1] * max(1.0 - psi_n[-1], 0.0)
        aminor = float(np.asarray(equilibrium.aminor).squeeze())
        r = np.sqrt(np.clip(phi / phi_edge, 0.0, None)) * aminor
        return cls(x=x, psi_n=psi_n, r=r, q=q, aminor=aminor,
                   rmajor=float(np.asarray(equilibrium.Rmajor).squeeze()))

    def at(self, name: str, x):
        return np.interp(np.asarray(x, dtype=float), self.x, getattr(self, name))

    def shear(self, x):
        """``s = (r/q) dq/dr``."""
        with np.errstate(divide="ignore", invalid="ignore"):
            values = self.r / self.q * np.gradient(self.q, self.r)
        return np.interp(np.asarray(x, dtype=float), self.x, values)

    def epsilon(self, x):
        return self.at("r", x) / self.rmajor

    def dr_dx(self, x):
        """Metres of ``r`` per unit of ``x``."""
        return np.interp(np.asarray(x, dtype=float), self.x, np.gradient(self.r, self.x))

    def x_of_psi_n(self, psi_n):
        return np.interp(np.asarray(psi_n, dtype=float), self.psi_n, self.x)


# ----------------------------------------------------------------------------
# Extrema of the continuum, and the gaps they bound
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ContinuumTip:
    """A smooth extremum of one continuum level, in ``s = f**2``."""

    x: float
    f_khz: float
    s2: float                # d^2(f^2)/dx^2 at the tip [kHz^2 per unit x^2]
    level: int               # index among the sorted continuum frequencies

    @property
    def is_maximum(self) -> bool:
        return self.s2 < 0.0

    def faces(self, f_khz: float) -> bool:
        """True if ``f`` is on the open side: above a maximum, below a minimum."""
        return (f_khz > self.f_khz) if self.is_maximum else (f_khz < self.f_khz)

    def distance(self, f_khz: float) -> float:
        """Distance in ``x`` of the complex singular pair at ``f``; 0 if real."""
        if not self.faces(f_khz) or self.s2 == 0.0:
            return 0.0
        return float(np.sqrt(2.0 * abs(f_khz**2 - self.f_khz**2) / abs(self.s2)))


def _sorted_levels(f_khz: np.ndarray) -> np.ndarray:
    values = np.array(f_khz, dtype=float)
    values[~np.isfinite(values)] = np.inf
    values = np.sort(values, axis=1)
    values[~np.isfinite(values)] = np.nan
    return values


def continuum_tips(x, f_khz, kink_factor: float = 1.0) -> list[ContinuumTip]:
    """Every smooth local extremum of the sorted continuum levels.

    ``f_khz`` is ``(nx, nbranch)``.  Sorting each radius turns every crossing
    of two uncoupled branches into a kink of two neighbouring levels; a kink is
    told from a real extremum by the spacing to the neighbouring level, which
    at a kink is no larger than what the branches' slopes cover in one sample.
    """
    x = np.asarray(x, dtype=float)
    levels = _sorted_levels(f_khz)
    tips = []
    for k in range(levels.shape[1]):
        f = levels[:, k]
        for i in range(1, x.size - 1):
            left, mid, right = f[i - 1], f[i], f[i + 1]
            if not np.all(np.isfinite((left, mid, right))):
                continue
            maximum = mid > left and mid >= right
            minimum = mid < left and mid <= right
            if not (maximum or minimum):
                continue
            neighbour = k + 1 if maximum else k - 1
            if not 0 <= neighbour < levels.shape[1] or not np.isfinite(levels[i, neighbour]):
                continue
            spacing = abs(levels[i, neighbour] - mid)
            if spacing <= kink_factor * (abs(mid - left) + abs(right - mid)):
                continue                      # a crossing, sorted into a kink
            # parabola through the three samples, in s = f^2
            xs, ss = x[i - 1:i + 2], f[i - 1:i + 2] ** 2
            c2, c1, c0 = np.polyfit(xs - x[i], ss, 2)
            if c2 == 0.0 or (c2 < 0.0) != maximum:
                continue
            offset = float(np.clip(-c1 / (2.0 * c2), xs[0] - x[i], xs[-1] - x[i]))
            s_tip = c0 + c1 * offset + c2 * offset**2
            tips.append(ContinuumTip(x=float(x[i] + offset), f_khz=float(np.sqrt(max(s_tip, 0.0))),
                                     s2=float(2.0 * c2), level=k))
    return tips


@dataclass(frozen=True)
class Gap:
    """A lower-branch maximum facing an upper-branch minimum at about one radius."""

    lower: ContinuumTip
    upper: ContinuumTip

    @property
    def x(self) -> float:
        return 0.5 * (self.lower.x + self.upper.x)

    @property
    def f_low(self) -> float:
        return self.lower.f_khz

    @property
    def f_high(self) -> float:
        return self.upper.f_khz

    @property
    def f_center(self) -> float:
        return 0.5 * (self.f_low + self.f_high)

    def distance(self, f_khz: float) -> float:
        """The singular-pair distance at ``f`` inside the gap; 0 outside it."""
        if not self.f_low < f_khz < self.f_high:
            return 0.0
        return min(self.lower.distance(f_khz), self.upper.distance(f_khz))


def find_gaps(tips: list[ContinuumTip], max_offset: float = 0.05,
              min_relative_width: float = 1.0e-3) -> list[Gap]:
    """Pair each maximum of level ``k`` with the nearest minimum of level ``k+1``.

    The two edges of a gap need not sit at exactly one radius; they are paired
    when within ``max_offset`` in ``x`` and the gap is wider than
    ``min_relative_width`` of its frequency.
    """
    maxima = [tip for tip in tips if tip.is_maximum]
    minima = [tip for tip in tips if not tip.is_maximum]
    gaps = []
    for low in maxima:
        candidates = [tip for tip in minima if tip.level == low.level + 1
                      and abs(tip.x - low.x) <= max_offset and tip.f_khz > low.f_khz]
        if not candidates:
            continue
        high = min(candidates, key=lambda tip: abs(tip.x - low.x))
        if (high.f_khz - low.f_khz) > min_relative_width * high.f_khz:
            gaps.append(Gap(low, high))
    # one gap per facing pair
    unique = {}
    for gap in gaps:
        key = (gap.upper.level, round(gap.upper.x, 6))
        if key not in unique or gap.lower.f_khz > unique[key].lower.f_khz:
            unique[key] = gap
    return sorted(unique.values(), key=lambda gap: (gap.f_center, gap.x))


def gap_name(delta_m: int | None, family: str = "Alfvenic") -> str:
    if family == "sound":
        return "sound"
    if delta_m is None:
        return "gap"
    return GAP_NAMES.get(int(delta_m), f"dm={int(delta_m)} gap")


# ----------------------------------------------------------------------------
# Size estimates
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class GapEstimate:
    """What it takes to resolve a mode in one gap; plain values, picklable."""

    kind: str                    # TAE, EAE, NAE, ...
    m: tuple                     # the two harmonics the gap edge couples
    family: str                  # of the gap-edge continuum mode
    x: float                     # solver coordinate of the gap centre
    psi_n: float
    f_low: float                 # gap edges [kHz]
    f_high: float
    s2_low: float                # (f^2)'' at the edges [kHz^2 per x^2]
    s2_high: float
    r_mm: float                  # rho_tor * a
    q: float
    shear: float
    epsilon: float
    m_eff: float                 # |n| q at the gap
    width_mm: float              # the estimate in its regime
    width_shear_mm: float        # (r0/m) eps/|s|
    width_low_shear_mm: float    # (r0/m) sqrt(eps/|s|)
    dr_dx_mm: float              # mm of r per unit x

    @property
    def f_center(self) -> float:
        return 0.5 * (self.f_low + self.f_high)

    @property
    def regime(self) -> str:
        if not np.isfinite(self.shear) or abs(self.shear) < self.epsilon:
            return "low shear: (r0/m) sqrt(eps/|s|)"
        return "finite shear: (r0/m) eps/|s|"

    def edge_distance_mm(self, f_khz: float) -> float:
        """``d`` at ``f`` inside the gap [mm]; 0 outside it."""
        if not self.f_low < f_khz < self.f_high:
            return 0.0
        d = [np.sqrt(2.0 * abs(f_khz**2 - f_edge**2) / abs(s2))
             for f_edge, s2 in ((self.f_low, self.s2_low), (self.f_high, self.s2_high))
             if s2 != 0.0]
        return float(min(d) * self.dr_dx_mm) if d else np.inf

    def element_mm(self, accuracy: str = "1e-5", f_khz: float | None = None) -> float:
        """Largest element at the gap for about this relative frequency error.

        ``"1e-4"``: ``min(w, d)``; ``"1e-5"``: ``min(w/4, d/2)``, with ``d`` at
        ``f`` (the gap centre by default).
        """
        d = self.edge_distance_mm(self.f_center if f_khz is None else f_khz)
        if accuracy == "1e-4":
            return float(min(self.width_mm, d))
        return float(min(self.width_mm / 4.0, d / 2.0))


def mode_width_m(geometry: RadialGeometry, ntor: int, x: float) -> tuple[float, float, float]:
    """``(w, w_shear, w_low_shear)`` in metres at ``x``; ``w`` is the one whose
    regime applies, ``min`` of the two."""
    r0 = float(geometry.at("r", x))
    m_eff = abs(int(ntor)) * float(geometry.at("q", x))
    shear = float(geometry.shear(x))
    epsilon = r0 / geometry.rmajor
    if not (m_eff > 0.0 and np.isfinite(shear)) or shear == 0.0:
        return np.nan, np.nan, np.nan
    ratio = epsilon / abs(shear)
    w_shear = r0 / m_eff * ratio
    w_low = r0 / m_eff * np.sqrt(ratio)
    return float(min(w_shear, w_low)), float(w_shear), float(w_low)


def estimate_gap(gap: Gap, geometry: RadialGeometry, ntor: int, *,
                 kind: str = "gap", m: tuple = (), family: str = "Alfvenic") -> GapEstimate:
    x = gap.x
    r0 = float(geometry.at("r", x))
    q0 = float(geometry.at("q", x))
    width, width_shear, width_low = mode_width_m(geometry, ntor, x)
    return GapEstimate(
        kind=kind, m=tuple(int(v) for v in m), family=family, x=float(x),
        psi_n=float(geometry.at("psi_n", x)), f_low=gap.f_low, f_high=gap.f_high,
        s2_low=gap.lower.s2, s2_high=gap.upper.s2, r_mm=r0 * 1e3, q=q0,
        shear=float(geometry.shear(x)), epsilon=r0 / geometry.rmajor,
        m_eff=abs(int(ntor)) * q0, width_mm=width * 1e3,
        width_shear_mm=width_shear * 1e3, width_low_shear_mm=width_low * 1e3,
        dr_dx_mm=float(geometry.dr_dx(x)) * 1e3)


def local_scales_mm(geometry: RadialGeometry, tips: list[ContinuumTip], ntor: int,
                    x: float, f_khz: float, reach: float = 0.1) -> tuple[float, float]:
    """``(w, d)`` in mm for a mode at ``x`` and ``f``: the width estimate there,
    and the nearest singular-pair distance among the extrema within ``reach``
    of ``x`` that ``f`` faces (``inf`` if none)."""
    w = mode_width_m(geometry, ntor, x)[0] * 1e3
    distances = [tip.distance(f_khz) * float(geometry.dr_dx(tip.x)) * 1e3
                 for tip in tips if tip.faces(f_khz) and abs(tip.x - x) < reach]
    return w, (min(distances) if distances else np.inf)


# ----------------------------------------------------------------------------
# How far the global path may go
# ----------------------------------------------------------------------------

def path_alpha(f_khz: float, tips: list[ContinuumTip], domain: tuple[float, float],
               kappa: float = 0.5, alpha_max: float = 0.05) -> tuple[float, ContinuumTip | None]:
    """The largest ``alpha`` whose path height stays below ``kappa * d`` at every
    extremum that ``f`` faces inside ``domain``; capped at ``alpha_max``.

    Returns ``(alpha, limiting tip or None)``.
    """
    lo, hi = domain
    best, limiting = float(alpha_max), None
    for tip in tips:
        if not lo < tip.x < hi or not tip.faces(f_khz):
            continue
        d = tip.distance(f_khz)
        alpha = kappa * d / ((tip.x - lo) * (hi - tip.x))
        if alpha < best:
            best, limiting = float(alpha), tip
    return best, limiting


def harmonic_correlation(a, b) -> float:
    """``|<a, b>| / (|a| |b|)`` of two harmonic fields, phase-free."""
    a = np.ravel(np.asarray(a, dtype=complex))
    b = np.ravel(np.asarray(b, dtype=complex))
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(abs(np.vdot(a, b)) / norm) if norm > 0.0 else 0.0


__all__ = [
    "GAP_NAMES",
    "ContinuumTip",
    "Gap",
    "GapEstimate",
    "RadialGeometry",
    "continuum_tips",
    "estimate_gap",
    "find_gaps",
    "gap_name",
    "harmonic_correlation",
    "local_scales_mm",
    "mode_width_m",
    "path_alpha",
]
