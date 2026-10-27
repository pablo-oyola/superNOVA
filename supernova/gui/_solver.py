"""The solver tab without Qt: its state, the solve, and what gets drawn.

The tab runs the simple pass: one radial grid, the band cut into windows,
one shift-and-invert solve per window. Every root it finds then goes through a
mode filter. Gerrit Kramer's MODELOCATE (:meth:`supernova.api.ModeSet.locate`)
calls a mode a gap mode unless its envelope peaks on a single grid point, in the
core, at the edge, or on a continuum crossing. The global-path filter keeps
MODELOCATE's spike, core and edge tests but replaces its continuum test: each
remaining mode is re-solved on the complex radial paths ``alpha`` and
``-alpha`` (:mod:`supernova.operators.contour`), with ``alpha`` kept below the
singular points of the nearest continuum extrema (:func:`supernova.solve.gaps.
path_alpha`); a discrete mode does not move, a continuum approximant does.

Before a solve the continuum preview also locates the gaps (TAE, EAE, ...) and
estimates how wide their modes are and what element size resolves them
(:mod:`supernova.solve.gaps`). After one, selected modes can be refined: re-solved
on meshes graded towards them, finer each time, to see the frequency converge.

The sweep goes through :meth:`~supernova.solve.stability.StabilitySolver.
solve_pep_windows` rather than :meth:`~supernova.api.Scan.run`: the pencil is
projected once and the eigensolver retargeted for each window, where calling
the solver per window re-projected the same matrices every time (about three
quarters of the cost); a window in which nothing converges comes back as a row
in the window table, not the end of the sweep; and stepping it window by
window is where progress is reported and Stop takes effect.

Everything the window draws is computed on the worker thread and handed over
as plain arrays in :class:`SolverResult`, so the GUI thread never touches the
solver objects a later solve may be using. Radii in it are physical psi_N:
the solver's own coordinate x spans only the equilibrium's Boozer domain, and
the two differ whenever that domain is not (0, 1).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
import time

import numpy as np

from supernova import defaults
from supernova.gui._session import JsonState

SPECIES = ("H", "D", "T", "He3", "He4")
FACTORIZATIONS = ("mumps", "spqr", "petsc_qr")
PEP_TYPES = ("TOAR", "STOAR", "QARNOLDI", "LINEAR", "JD")
ST_TYPES = ("SINVERT", "CAYLEY", "SHIFT")
WHICH = ("TARGET_MAGNITUDE", "TARGET_REAL", "TARGET_IMAGINARY")
FIELDS = ("xi_psi", "p1", "xi_s", "div_xi")
SOUND_MODELS = ("ideal", "nova")
SOUND_TITLES = {"ideal": "ideal MHD", "nova": "NOVA ksound=1 (no parallel sound)"}
MODE_FILTERS = ("modelocate", "path")
FILTER_TITLES = {"modelocate": "MODELOCATE (Kramer)",
                 "path": "global path alpha"}
BACKWARD_ERROR_GATE = 1.0e-8      # the production acceptance of an eigenpair
FOLLOW_UP_TOL = 1.0e-10           # SLEPc tol of the path and refinement solves
# Path and refinement solves aim at a known root. Shift-and-invert exactly on
# it factorizes a singular operator; aim this far off instead (relative, in omega).
TARGET_OFFSET = 1.0e-4
# Some shifts give a factorization too inaccurate for the gate; such a solve is
# repeated with the target moved by these (relative, in omega) before giving up.
RETRY_OFFSETS = (3.0e-3, -3.0e-3)
CONVERGED = 1.0e-4                # relative change between the two finest meshes
TRACK_FIRST = 0.5                 # shape correlation to follow a mode onto the first refined mesh
INCONCLUSIVE = 32                 # mode_class code of a path test that could not decide
PATH_LEGEND = ("0 gap (path-invariant at the thresholds), 8 continuum (moves on a path), "
               f"16 ideal (Re omega^2 < 0, not tested), {INCONCLUSIVE} path test inconclusive")
PARTS = ("real", "imag", "abs")
# how the mode-structure page draws a mode: over the continuum with xi = 0 on
# the mode's frequency, or its harmonics alone
MODE_VIEWS = ("continuum", "harmonics")
MODE_VIEW_TITLES = {"harmonics": "harmonics", "continuum": "on the continuum"}
MAX_WINDOWS = 2000
DUPLICATE_RELATIVE = 1.0e-8       # two windows returning one root

# What a window's outcome means for the person choosing nev, widths and maxit.
STATUS_COLOURS = {
    "ok": "#159A8A",
    "saturated": "#E39A31",
    "empty": "#78879D",
    "unconverged": "#B45309",
    "failed": "#D64B5F",
    "skipped": "#B8C2CF",
}
STATUS_HINTS = {
    "ok": "fewer roots than nev converged; the window is probably exhausted",
    "saturated": "nev or more roots converged -- there may be more: raise nev "
                 "or narrow the windows",
    "empty": "SLEPc converged and found no root in the window",
    "unconverged": "nothing converged by maxit. SLEPc ends an empty window the "
                   "same way, so this may be a window with no root, or one "
                   "that needs more iterations",
    "failed": "the solve raised; the message is in the table and the log",
    "skipped": "not searched: the solve was stopped",
}
# SLEPc ConvergedReason codes
REASONS = {"1": "converged", "2": "converged (user test)", "0": "iterating",
           "-1": "max iterations", "-2": "breakdown", "-3": "symmetry lost"}

# MODELOCATE outcomes, as drawn; a combination of flags takes its first one's
LOCATE_STYLE = {
    "gap": {"title": "gap mode", "color": "#159A8A", "marker": "o"},
    "continuum": {"title": "on a continuum crossing", "color": "#D64B5F", "marker": "D"},
    "spike": {"title": "single-point spike", "color": "#475569", "marker": "x"},
    "core": {"title": "peaks in the core", "color": "#758295", "marker": "<"},
    "edge": {"title": "peaks at the edge", "color": "#758295", "marker": ">"},
    "ideal": {"title": "ideal (omega^2 < 0)", "color": "#7C3AED", "marker": "*"},
    "inconclusive": {"title": "path test inconclusive", "color": "#B45309", "marker": "s"},
}
LOCATE_ORDER = tuple(LOCATE_STYLE)
LOCATE_WHY = {
    "gap": "no flag fired: its peak is wider than one grid point, lies between "
           "the core and boundary edges, and its envelope stays below the "
           "continuum level at every continuum crossing. MODELOCATE does not "
           "test how global the mode is",
    "continuum": "its envelope is above the continuum level on the grid points "
                 "bracketing a continuum crossing",
    "spike": "both neighbours of the envelope's peak are below the spike level",
    "core": "the envelope peaks inside the core edge",
    "edge": "the envelope peaks outside the boundary edge",
    "ideal": "Re(omega^2) < 0: a purely growing ideal mode, not tested further",
    "inconclusive": "the path test could not decide",
}
# the same verdicts, as the global-path filter reaches them
PATH_WHY = {
    "gap": "it stays put on the complex paths +alpha and -alpha: same frequency "
           "within the tolerance and the same shape. A discrete eigenvalue does "
           "not depend on the path",
    "continuum": "it moves on the complex paths by more than the tolerance, or "
                 "changes shape below the correlation threshold: a continuum "
                 "approximant, which the path rotates into the complex plane",
    "inconclusive": "either nothing converged on a path, or the singular points of "
                    "a continuum extremum next to it are so close that the path "
                    "cannot rise a quarter element there. Refine the mode first",
}

# fields that never change what is solved: what is drawn, and how the
# windows are spread over worker processes
# The path test's thresholds are here too: every mode keeps its drift and
# correlation, so moving a threshold relabels the pass without solving it again.
_VIEW_FIELDS = frozenset({"field", "part", "mode_view", "continuum_preview", "gap_only",
                          "workers", "threads_per_worker", "resolution_target",
                          "refine_h_mm", "refine_levels", "refine_growth",
                          "refine_path_check", "path_tolerance", "path_correlation"})


def locate_family(label: str) -> str:
    """The style key for a MODELOCATE label such as ``"spike|core"``."""
    return str(label).split("|")[0]


def options_from_overrides(cls, overrides: dict, *, skip=()):
    """Build an options dataclass from the fields that differ from default.

    JSON has no tuples, so a list is turned back into one wherever the
    default is a tuple.
    """
    defaults = {f.name: f.default for f in fields(cls)}
    unknown = sorted(set(overrides) - set(defaults) - set(skip))
    if unknown:
        raise ValueError(f"{cls.__name__} has no option {', '.join(unknown)}.")
    kwargs = {}
    for name, value in overrides.items():
        if name in skip:
            continue
        kwargs[name] = tuple(value) if isinstance(defaults[name], tuple) else value
    return cls(**kwargs)


def parse_petsc_options(text: str) -> dict[str, str]:
    """``key value`` or ``key=value`` per line; a leading ``-`` is optional."""
    options = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" in line:
            key, _, value = line.partition("=")
        else:
            key, _, value = line.partition(" ")
        key = key.strip().lstrip("-")
        if not key:
            raise ValueError(f"PETSc option line {number} has no name.")
        options[key] = value.strip()
    return options


@dataclass
class SolverState(JsonState):
    """Every control in the solver tab, and nothing else."""

    # the problem; every default is the script API's, from supernova.defaults
    ntor: int = 4
    mmax: int = 0                     # 0 is auto: ceil(n q) at rho_pol = 0.95
    max_dm: int = defaults.MAX_DM
    regularizer: float = defaults.REGULARIZER
    gamma_s: float = defaults.GAMMA_S
    species: str = defaults.SPECIES
    sound_model: str = defaults.SOUND_MODEL  # "nova": legacy ksound=1, no parallel sound

    # the frequency search
    f_low_khz: float = 50.0
    f_high_khz: float = 150.0
    window_width_khz: float = 2.5     # the size of each subrange
    window_overlap: float = 0.0       # fraction of a window shared with the next
    nev: int = defaults.NEV           # roots requested per window
    region_vscale: float = defaults.REGION_VSCALE  # imaginary semi-axis of the ellipse
    tol: float = defaults.TOL
    maxit: int = defaults.MAXIT

    # the radial discretization, between walls at these rho_pol (clipped to
    # the equilibrium's Boozer domain)
    spans: int = 150
    order: int = defaults.ORDER
    quadrature_order: int = defaults.QUAD_ORDER
    rho_min: float = 0.0
    rho_max: float = 1.0

    # the eigensolver and its linear solves
    pep_type: str = "TOAR"
    st_type: str = "SINVERT"
    which: str = "TARGET_MAGNITUDE"
    factorization: str = defaults.FACTORIZATION
    mumps_icntl_14: int = defaults.MUMPS_ICNTL_14  # % workspace headroom; 0 keeps the preset's
    extra_options: str = ""

    # the mode filter, and MODELOCATE as overrides of ModeLocateThresholds
    mode_filter: str = "modelocate"
    modelocate: dict = field(default_factory=dict)
    path_alpha_max: float = 0.05      # the global path's largest amplitude
    path_kappa: float = 0.4           # path height, as a fraction of d
    path_tolerance: float = 3.0e-4    # relative frequency change still invariant
    path_correlation: float = 0.9     # harmonic correlation still the same mode
    path_nev: int = 4                 # roots per path solve, nearest the mode

    # refinement of selected modes, and the resolution guide
    resolution_target: str = "1e-5"   # "1e-4" or "1e-5": the guide's accuracy
    refine_h_mm: float = 0.0          # finest element at a mode; 0 picks it
    refine_levels: int = 3            # meshes, each halving the element
    refine_growth: float = 0.15       # how fast the elements relax to the base
    refine_path_check: bool = True    # path-test the modes on the finest mesh

    # parallel windows: worker processes, each with its own OpenMP threads
    workers: int = 2
    threads_per_worker: int = 4

    # output
    reconstruction_points: int = 512
    verbose: bool = False

    # view
    continuum_preview: bool = True
    gap_only: bool = False
    field: str = "xi_psi"
    part: str = "real"
    mode_view: str = "continuum"

    # ------------------------------------------------------------ derived

    @property
    def band(self) -> tuple[float, float]:
        return (float(self.f_low_khz), float(self.f_high_khz))

    def windows_khz(self) -> list[tuple[float, float]]:
        """The subranges, exactly as :meth:`supernova.api.Band.windows` cuts them."""
        from supernova.api import Band

        return Band(*self.band).windows(float(self.window_width_khz),
                                        float(self.window_overlap))

    def window_count(self) -> int:
        """How many windows, without building the list."""
        step = float(self.window_width_khz) * (1.0 - float(self.window_overlap))
        return max(1, int(np.ceil((self.f_high_khz - self.f_low_khz) / step)))

    @property
    def mmax_limit(self) -> int | None:
        """mmax as the solvers take it: None, the q-based default, when it reads auto (0)."""
        return int(self.mmax) or None

    def petsc_options(self):
        from supernova.api import PetscOptions

        extra = {}
        if self.factorization == "mumps" and int(self.mumps_icntl_14) > 0:
            extra["st_mat_mumps_icntl_14"] = str(int(self.mumps_icntl_14))
        extra.update(parse_petsc_options(self.extra_options))
        return PetscOptions(self.factorization, extra)

    def locate_thresholds(self):
        from supernova.solve.modelocate import ModeLocateThresholds

        return options_from_overrides(ModeLocateThresholds, self.modelocate)

    # ---------------------------------------------------------- validation

    def validate(self) -> list[str]:
        """Return the reasons this state cannot be solved, in reading order."""
        problems = []
        if self.ntor == 0:
            problems.append("n = 0 is not a toroidal eigenmode problem here; choose n != 0.")
        if self.mmax < 0:
            problems.append("mmax must be auto (0) or at least 1.")
        if self.max_dm < 1:
            problems.append("max_dm must be at least 1.")
        if not np.isfinite(self.regularizer) or self.regularizer < 0.0:
            problems.append("The regularizer must be finite and non-negative.")
        if not np.isfinite(self.gamma_s) or self.gamma_s < 0.0:
            problems.append("gamma_s must be finite and non-negative.")
        if self.species not in SPECIES:
            problems.append(f"Species must be one of {', '.join(SPECIES)}.")
        if self.sound_model not in SOUND_MODELS:
            problems.append(f"The sound model must be one of {', '.join(SOUND_MODELS)}.")

        band_ok = (np.isfinite(self.f_low_khz) and np.isfinite(self.f_high_khz)
                   and 0.0 < self.f_low_khz < self.f_high_khz)
        if not band_ok:
            problems.append("The frequency band must satisfy 0 < low < high.")
        if not self.window_width_khz > 0.0:
            problems.append("The window width must be positive.")
        elif not 0.0 <= self.window_overlap < 1.0:
            problems.append("The window overlap must satisfy 0 <= overlap < 1.")
        elif band_ok and self.window_count() > MAX_WINDOWS:
            problems.append(
                f"{self.window_count()} windows is too many; widen them "
                f"(at most {MAX_WINDOWS}).")
        if self.nev < 1:
            problems.append("Ask for at least one root per window.")
        if not self.region_vscale > 0.0:
            problems.append("The region's imaginary scale must be positive.")
        if not self.tol > 0.0:
            problems.append("The tolerance must be positive.")
        if self.maxit < 1:
            problems.append("maxit must be at least 1.")

        if self.spans < 2:
            problems.append("The radial grid needs at least 2 spans.")
        if not 0.0 <= self.rho_min < self.rho_max <= 1.0:
            problems.append("The radial walls must satisfy 0 <= rho_min < rho_max <= 1.")
        if self.order < 1:
            problems.append("The B-spline order must be positive.")
        if self.quadrature_order < 1:
            problems.append("The quadrature order must be positive.")

        for name, allowed in (("pep_type", PEP_TYPES), ("st_type", ST_TYPES),
                              ("which", WHICH), ("factorization", FACTORIZATIONS)):
            if getattr(self, name) not in allowed:
                problems.append(f"{name} must be one of {', '.join(allowed)}.")
        if not 1 <= self.workers <= 256:
            problems.append("Use between 1 and 256 worker processes.")
        if not 1 <= self.threads_per_worker <= 256:
            problems.append("Use between 1 and 256 threads per worker.")
        if self.reconstruction_points < 16:
            problems.append("Reconstruct on at least 16 points.")
        if self.field not in FIELDS or self.part not in PARTS:
            problems.append("Unknown field or part to draw.")
        if self.mode_view not in MODE_VIEWS:
            problems.append(f"The mode view must be one of {', '.join(MODE_VIEWS)}.")
        if self.mode_filter not in MODE_FILTERS:
            problems.append(f"The mode filter must be one of {', '.join(MODE_FILTERS)}.")
        if not 0.0 < self.path_alpha_max <= 0.5:
            problems.append("The path's largest alpha must lie in (0, 0.5].")
        if not 0.0 < self.path_kappa < 1.0:
            problems.append("The path height fraction kappa must lie in (0, 1).")
        if not 0.0 < self.path_tolerance < 1.0:
            problems.append("The path tolerance must lie in (0, 1).")
        if not 0.0 < self.path_correlation <= 1.0:
            problems.append("The path correlation must lie in (0, 1].")
        if self.path_nev < 1:
            problems.append("Ask the path solves for at least one root.")
        if self.resolution_target not in ("1e-4", "1e-5"):
            problems.append("The resolution target must be 1e-4 or 1e-5.")
        if not np.isfinite(self.refine_h_mm) or self.refine_h_mm < 0.0:
            problems.append("The refined element size must be non-negative (0 picks it).")
        if not 1 <= self.refine_levels <= 8:
            problems.append("Refine on 1 to 8 meshes.")
        if not 0.0 < self.refine_growth < 1.0:
            problems.append("The refinement growth must lie in (0, 1).")

        if not problems:
            try:
                self.petsc_options()
                self.locate_thresholds()
            except (TypeError, ValueError) as error:
                problems.append(str(error))
        return problems

    def solve_key(self) -> tuple:
        """Everything the result depends on; the view settings are not here."""
        items = []
        for name, value in sorted(self.to_dict().items()):
            if name in _VIEW_FIELDS:
                continue
            if isinstance(value, dict):
                value = tuple(sorted((k, repr(v)) for k, v in value.items()))
            items.append((name, value))
        return tuple(items)

    def preview_key(self) -> tuple:
        return (self.ntor, self.mmax, self.max_dm, self.gamma_s, self.species,
                self.sound_model)

    @property
    def drops_sound(self) -> bool:
        """No slow-sound continuum: gamma_s = 0, or the NOVA sound model."""
        return float(self.gamma_s) == 0.0 or self.sound_model == "nova"


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------

def _flux_maps(equilibrium):
    """``(x, psi_N)`` on the equilibrium's surfaces, for converting radii."""
    magspace = equilibrium.magspace
    x = np.asarray(magspace.coords["psi"].values, dtype=float)
    if "psi_physical_norm" in magspace.coords:
        return x, np.asarray(magspace.coords["psi_physical_norm"].values, dtype=float)
    return x, x


def radial_domain(equilibrium, rho_min: float, rho_max: float) -> tuple[float, float]:
    """The solver-x interval between walls at ``rho_min`` and ``rho_max``.

    Walls outside the Boozer domain are moved to its edge; x is only defined
    there.
    """
    x, psi_n = _flux_maps(equilibrium)
    low, high = (float(np.interp(rho * rho, psi_n, x)) for rho in (rho_min, rho_max))
    low, high = max(low, 0.0), min(high, 1.0)
    if rho_min <= 0.0:
        low = 0.0
    if rho_max >= 1.0:
        high = 1.0
    if not high > low:
        raise ValueError("The radial walls enclose no part of the Boozer domain.")
    return low, high


@dataclass(frozen=True)
class ContinuumPreview:
    x: np.ndarray                     # (npsi,) physical psi_N
    frequency_khz: np.ndarray         # (npsi, nbranch)
    which: str
    key: tuple
    seconds: float
    gaps: tuple = ()                  # GapEstimate per Alfvenic gap found
    solver_x: np.ndarray | None = None  # the solver's x at each of ``x``

    def x_of_psi_n(self, psi_n):
        """The solver coordinate at ``psi_n``, for element sizes."""
        grid = self.x if self.solver_x is None else self.solver_x
        return np.interp(np.asarray(psi_n, dtype=float), self.x, grid)

    def element_mm(self, gap, state) -> float:
        """The current grid's element size at ``gap``, in mm of rho_tor * a."""
        # as radial_domain places the walls
        low = 0.0 if state.rho_min <= 0.0 else float(self.x_of_psi_n(state.rho_min ** 2))
        high = 1.0 if state.rho_max >= 1.0 else float(self.x_of_psi_n(state.rho_max ** 2))
        return max(high - low, 0.0) / max(int(state.spans), 1) * gap.dr_dx_mm


@dataclass(frozen=True)
class WindowRow:
    window: int
    range_khz: tuple
    nev: int
    nconv: int = 0
    kept: int = 0
    dropped: int = 0                  # converged, but outside the backward-error gate
    iterations: int = 0
    reason: str = ""
    status: str = "ok"
    message: str = ""
    seconds: float = 0.0

    @property
    def reason_text(self) -> str:
        if not self.reason:
            return ""
        return f"{REASONS.get(self.reason, 'reason')} ({self.reason})"


@dataclass(frozen=True)
class SolverResult:
    """One simple pass and its MODELOCATE verdicts, as plain arrays."""

    state: SolverState                # a copy, as solved
    title: str                        # the equilibrium it was solved on
    windows: tuple                    # WindowRow per window
    dataset: object                   # merged eigenpair Dataset, or None
    frequency_khz: np.ndarray         # (nmode,)
    omega: np.ndarray                 # (nmode,) normalized
    center: np.ndarray                # power-weighted centre, psi_N
    width: np.ndarray                 # RMS radial width, in psi_N
    dominant_m: np.ndarray
    backward_error: np.ndarray
    window_of: np.ndarray             # which window found each mode
    located: object                   # MODELOCATE Dataset, or None
    crossings: tuple                  # per mode, continuum crossings in psi_N
    x: np.ndarray                     # reconstruction grid, psi_N
    m_pol: np.ndarray
    fields: dict                      # name -> (nmode, nx, nm) complex
    envelope_x: np.ndarray            # MODELOCATE's breakpoints, psi_N
    continuum_x: np.ndarray           # psi_N
    continuum_khz: np.ndarray
    psi_core: float                   # MODELOCATE's core and boundary edges
    psi_edge: float
    walls: tuple                      # the radial domain solved, psi_N
    seconds: float
    equilibrium: object               # None once sent out of the compute process
    handle: int | None = None         # the compute process's name for it

    @property
    def filter_method(self) -> str:
        if self.located is None:
            return self.state.mode_filter
        return "path" if self.located.attrs.get("method") == "global_path" else "modelocate"

    def why(self, label: str) -> str:
        """Why the filter gave ``label``, as the filter that ran would say it."""
        texts = PATH_WHY if self.filter_method == "path" else {}
        return " and ".join(texts.get(flag, LOCATE_WHY.get(flag, flag))
                            for flag in str(label).split("|"))

    def with_path_thresholds(self, tolerance: float, correlation: float) -> "SolverResult":
        """This pass with its path verdicts drawn again at these thresholds.

        Nothing is solved: each mode keeps the drift and correlation the paths
        gave it, and only the cut moves. A MODELOCATE pass comes back as it is.
        """
        if self.filter_method != "path":
            return self
        state = self.state.copy()
        state.path_tolerance, state.path_correlation = float(tolerance), float(correlation)
        return replace(self, state=state,
                       located=_apply_path_thresholds(self.located, tolerance, correlation))

    @property
    def mode_count(self) -> int:
        return int(self.frequency_khz.size)

    @property
    def labels(self) -> np.ndarray:
        if self.located is None:
            return np.empty(0, dtype=str)
        return np.asarray(self.located["mode_label"].values, dtype=str)

    def label_counts(self) -> dict[str, int]:
        labels, counts = np.unique(self.labels, return_counts=True)
        return {str(a): int(b) for a, b in zip(labels, counts)}

    def mode_rows(self) -> list[dict]:
        """One row per mode, with the columns the table shows."""
        rows = []
        labels = self.labels
        peak = (np.asarray(self.located["peak_rho"].values, dtype=float)
                if self.located is not None else np.full(self.mode_count, np.nan))
        for index in range(self.mode_count):
            omega = self.omega[index]
            rows.append({
                "mode": index,
                "frequency_khz": float(self.frequency_khz[index]),
                "growth_ratio": float(omega.imag / omega.real) if omega.real else np.nan,
                "label": str(labels[index]),
                "peak_rho": float(peak[index]),
                "center": float(self.center[index]),
                "width": float(self.width[index]),
                "dominant_m": int(self.dominant_m[index]),
                "backward_error": float(self.backward_error[index]),
                "window": int(self.window_of[index]),
                "path_alpha": self._path_value("path_alpha", index),
                "path_drift": self._path_value("path_drift", index),
                "path_correlation": self._path_value("path_correlation", index),
            })
        return rows

    def _path_value(self, name: str, index: int) -> float:
        if self.located is None or name not in self.located:
            return np.nan
        return float(self.located[name].values[index])

    def field(self, mode: int, name: str, part: str) -> np.ndarray:
        values = self.fields[name][int(mode)]
        return {"real": values.real, "imag": values.imag, "abs": np.abs(values)}[part]

    def envelope(self, mode: int):
        """MODELOCATE's envelope for one mode, on the grid it was judged on."""
        return (self.envelope_x,
                np.asarray(self.located["envelope"].values[int(mode)], dtype=float))


@dataclass(frozen=True)
class RefineResult:
    """Selected modes re-solved on meshes graded towards them, coarse to fine."""

    state: SolverState                # as refined
    title: str
    modes: tuple                      # the original mode indices
    rows: tuple                       # one dict per (mode, level); see SolverModel.refine
    x: np.ndarray                     # psi_N of the envelopes
    envelopes: dict                   # mode -> (nlevel, nx) |xi^psi| envelopes
    path: dict                        # mode -> path verdict on the finest mesh, or {}
    seconds: float
    handle: int | None = None

    def rows_for(self, mode: int) -> list[dict]:
        return [row for row in self.rows if row["mode"] == int(mode)]

    def change(self, mode: int) -> float:
        """Relative frequency change between the two finest meshes it was found on."""
        f = [row["frequency_khz"] for row in self.rows_for(mode)
             if np.isfinite(row["frequency_khz"])]
        if len(f) < 2:
            return np.nan
        return float(abs(f[-1] - f[-2]) / abs(f[-1]))

    def same_root_as(self, mode: int):
        """Another refined mode that ended on the same root, or None."""
        mine = self.rows_for(mode)
        if not mine or not np.isfinite(mine[-1]["frequency_khz"]):
            return None
        for other in self.modes:
            theirs = self.rows_for(other)
            if (other != mode and theirs and np.isfinite(theirs[-1]["frequency_khz"])
                    and abs(theirs[-1]["frequency_khz"] - mine[-1]["frequency_khz"])
                    <= 1.0e-7 * abs(mine[-1]["frequency_khz"])):
                return other
        return None

    def verdict(self, mode: int) -> tuple[str, str]:
        """``(label, why)`` from convergence and the path test on the finest mesh.

        Lost on refinement, or still moving between the two finest meshes:
        continuum. Converged and
        path-invariant: a gap mode. Converged but moving on the +-alpha path:
        continuum-coupled -- its frequency meets the continuum where it has
        amplitude (its damping is not computed here). Otherwise inconclusive.
        """
        change = self.change(mode)
        test = self.path.get(mode, {})
        if not np.isfinite(change):
            return "continuum", ("it does not survive refinement: no root of its shape on "
                                 "the refined meshes")
        if change > CONVERGED:
            return "continuum", f"the frequency still moves by {change:.1e} between the two finest meshes"
        label = test.get("label")
        if label == "gap":
            return "gap", (f"converged (change {change:.1e}) and path-invariant (alpha = "
                           f"{test['alpha']:.3g}, change {test['drift']:.1e})")
        if label == "continuum":
            return "continuum-coupled", (
                f"converged (change {change:.1e}) but moves by {test['drift']:.1e} on the "
                f"+-alpha path (alpha = {test['alpha']:.3g}): it reaches the continuum")
        return "inconclusive", (f"converged (change {change:.1e}); the path test on the "
                                "finest mesh could not decide")


class SolverModel:
    """Owns the continuum operators for the current equilibrium and solves."""

    def __init__(self):
        self._continuum = None
        self._continuum_key = None
        self._equilibrium = None          # held so its id() is not reused

    def continuum_for(self, equilibrium, state: SolverState, progress=None):
        from supernova.continuum import superContinuum

        key = (id(equilibrium), float(state.gamma_s), state.species)
        if key != self._continuum_key:
            if progress:
                progress("Building the continuum operators for this equilibrium...")
            self._continuum = superContinuum(equilibrium, gamma_s=float(state.gamma_s),
                                             species=state.species, dense=True)
            self._continuum_key = key
            self._equilibrium = equilibrium
        return self._continuum

    def continuum_khz(self, equilibrium, state: SolverState, x, progress=None):
        """``(which, (nx, nbranch) kHz)`` of the continuum the solve sees.

        The full sound-Alfven continuum; at gamma_s = 0 the Alfven one; with the
        NOVA sound model the solver's own continuum with the parallel-acoustic
        rows dropped, exactly as the pencil drops them.
        """
        continuum = self.continuum_for(equilibrium, state, progress)
        if state.sound_model == "nova" and float(state.gamma_s) != 0.0:
            omega = _nova_continuum(equilibrium, continuum, state, np.asarray(x, dtype=float))
            return "nova", _khz(equilibrium, omega)
        which = "alfven" if state.drops_sound else "full"
        eigenvalues = continuum.solve(ntor=int(state.ntor), mmax=state.mmax_limit,
                                      psi_vals=np.asarray(x, dtype=float), which=which,
                                      normalize_eigvals=True, max_dm=int(state.max_dm))
        with np.errstate(invalid="ignore"):
            khz = np.sqrt(eigenvalues.real).to("kHz").v / (2.0 * np.pi)
        return which, np.asarray(khz, dtype=float)

    def preview(self, equilibrium, state: SolverState, *, points: int = 256,
                progress=None) -> ContinuumPreview:
        """The continuum for this n, and the gaps in it, to choose a band and a
        grid against before solving."""
        started = time.perf_counter()
        x = np.linspace(0.0, 1.0, int(points))
        if progress:
            progress(f"Solving the continuum for n = {state.ntor}...")
        which, khz = self.continuum_khz(equilibrium, state, x, progress)
        if progress:
            progress("Locating the gaps and estimating their mode widths...")
        try:
            gaps = tuple(self.gap_estimates(equilibrium, state, x, khz))
        except Exception:              # the guide is a help, never a blocker
            gaps = ()
        grid, psi_n = _flux_maps(equilibrium)
        return ContinuumPreview(x=np.interp(x, grid, psi_n),
                                frequency_khz=khz, which=which, key=state.preview_key(),
                                seconds=time.perf_counter() - started, gaps=gaps,
                                solver_x=x)

    def gap_estimates(self, equilibrium, state: SolverState, x, khz) -> list:
        """Every Alfvenic gap in this continuum, typed and sized.

        A gap is typed by the two harmonics that dominate its lower edge's
        continuum eigenvector (TAE: m and m+1, EAE: m and m+2, ...); an edge
        made of one harmonic is a lone extremum, not a gap, and a sound-like
        edge is not an Alfven gap -- neither is listed.
        """
        from supernova.solve.gaps import (
            RadialGeometry, continuum_tips, estimate_gap, find_gaps, gap_name)

        geometry = RadialGeometry.from_equilibrium(equilibrium)
        continuum = self.continuum_for(equilibrium, state)
        estimates = []
        for gap in find_gaps(continuum_tips(x, khz)):
            w, _, classifier = continuum.solve(
                ntor=int(state.ntor), mmax=state.mmax_limit, psi_vals=np.array([gap.x]),
                which="full", classify=True, max_dm=int(state.max_dm))
            with np.errstate(invalid="ignore"):
                f = np.asarray(np.sqrt(w.real).to("kHz").v / (2.0 * np.pi), dtype=float)[0]
            data = classifier.data
            edges = [int(np.nanargmin(np.abs(f - value))) for value in (gap.f_low, gap.f_high)]
            families = [str(data["family"].values[0][k]) for k in edges]
            if "Alfvenic" not in families:
                continue
            k = edges[0] if families[0] == "Alfvenic" else edges[1]
            top_m = np.asarray(data["top_m"].values[0][k][:2], dtype=int)
            weight = np.asarray(data["top_weight"].values[0][k][:2], dtype=float)
            if not weight[1] >= 0.2 * weight[0]:
                continue                       # one harmonic: an extremum, not a gap
            delta_m = abs(int(top_m[1]) - int(top_m[0]))
            estimates.append(estimate_gap(gap, geometry, int(state.ntor),
                                          kind=gap_name(delta_m), m=tuple(sorted(top_m)),
                                          family="Alfvenic"))
        return estimates

    def continuum_tips(self, equilibrium, state: SolverState, points: int = 512):
        """The smooth extrema of the continuum the solve sees, in solver x."""
        from supernova.solve.gaps import continuum_tips

        x = np.linspace(0.0, 1.0, int(points))
        _, khz = self.continuum_khz(equilibrium, state, x)
        return continuum_tips(x, khz)

    def solve(self, equilibrium, state: SolverState, *, title: str = "",
              progress=None, log=None, cancel=None, pool=None) -> SolverResult:
        """Search every window, then reconstruct and filter what was found.

        ``log`` receives the solver's verbose output when ``state.verbose``;
        ``cancel`` is a ``threading.Event``. With a :class:`~supernova.gui.
        _workers.WindowPool` the windows are solved side by side in its worker
        processes and a cancel stops them mid-solve; without one they are
        solved here, one after another, and a cancel takes effect between
        windows.
        """
        problems = state.validate()
        if problems:
            raise ValueError(problems[0])

        from supernova.api import Band, Case, RadialGrid, Scan

        state = state.copy()
        started = time.perf_counter()
        continuum = self.continuum_for(equilibrium, state, progress)
        case = Case(equilibrium, continuum,
                    metadata={"gamma_s": float(state.gamma_s), "species": state.species})
        domain = radial_domain(equilibrium, float(state.rho_min), float(state.rho_max))
        if progress:
            progress(f"Building the radial space: {state.spans} spans of order "
                     f"{state.order} on x = {domain[0]:.4f} - {domain[1]:.4f}...")
        scan = Scan(case, RadialGrid.uniform(int(state.spans), int(state.order),
                                             int(state.quadrature_order),
                                             domain=domain),
                    ntor=int(state.ntor), mmax=state.mmax_limit,
                    max_dm=int(state.max_dm), regularizer=float(state.regularizer))
        band = Band(*state.band, units="kHz",
                    vscale=float(state.region_vscale)).for_case(case)
        found = []
        redirect = (contextlib.redirect_stdout(log) if state.verbose and log is not None
                    else contextlib.nullcontext())
        with redirect:
            if pool is None:
                rows = self._solve_windows(scan, band, state, found, progress, cancel)
            else:
                rows = self._solve_windows_pooled(scan, band, state, found, progress,
                                                  cancel, pool, log)

        crossing_continuum = self._crossing_continuum(equilibrium, scan, state)
        modes = _merge_windows(found, scan)
        if modes is None:
            return self._pack(state, title, rows, None, None, scan, started, equilibrium,
                              crossing_continuum)
        thresholds = state.locate_thresholds()
        if state.mode_filter == "path":
            if progress:
                progress(f"Reconstructing {len(modes)} modes for the path test...")
            # for its envelopes and peaks only: the path test judges every mode
            located = modes.locate(thresholds=thresholds, continuum=_NO_CONTINUUM)
            located = self._path_filter(equilibrium, scan, modes, located, state, band,
                                        progress, cancel, pool)
        else:
            if progress:
                progress(f"Reconstructing {len(modes)} modes and running MODELOCATE...")
            located = modes.locate(thresholds=thresholds, continuum=crossing_continuum)
        return self._pack(state, title, rows, modes, located, scan, started, equilibrium,
                          crossing_continuum)

    def _crossing_continuum(self, equilibrium, scan, state):
        """The continuum MODELOCATE's crossings come from; None for the scan's own.

        The scan's continuum is the full sound-Alfven one. With the NOVA sound
        model the slow sound is gone from the pencil, so its crossings must go
        too: the Alfven continuum on the same radii is used instead.
        """
        if state.sound_model != "nova" or float(state.gamma_s) == 0.0:
            return None
        from supernova.solve.multigrid import _solve_master_continuum

        blocks = scan.solver.build_solver_blocks(ntor=int(state.ntor), mmax=state.mmax_limit,
                                                 max_dm=int(state.max_dm))
        dataset = _solve_master_continuum(
            _without_parallel_sound(blocks),
            np.asarray(scan.solver.unique_points, dtype=float))
        return (np.asarray(dataset.coords["psi"].values, dtype=float),
                np.asarray(dataset["omega"].values, dtype=complex))

    # ------------------------------------------------------- the path test

    def _path_filter(self, equilibrium, scan, modes, located, state, band, progress,
                     cancel, pool):
        """Every mode put to the global-path test; MODELOCATE judges none of them.

        ``located`` lends its envelopes and peaks, not its labels. A mode with
        Re(omega^2) < 0 grows without oscillating, so it has no frequency to
        follow on a path: it is "ideal", untested. Every other mode keeps the
        drift and correlation the paths gave it, and its label follows from
        the thresholds (:func:`_apply_path_thresholds`), which can therefore
        change after the solve.
        """
        omega = np.asarray(modes.omega, dtype=complex)
        count = omega.size
        ideal = np.real(omega ** 2) < 0.0
        alpha = np.full(count, np.nan)
        drift = np.full(count, np.nan)
        correlation = np.full(count, np.nan)
        candidates = np.flatnonzero(~ideal)
        if candidates.size:
            reference = modes.field("xi_psi", n=int(state.reconstruction_points))
            targets = [(int(index), complex(omega[index]),
                        float(located["peak_x"].values[index]),
                        np.asarray(reference.values[index]))
                       for index in candidates]
            verdicts = self.path_test(equilibrium, scan, state, band, targets,
                                      np.asarray(reference.coords["x"].values),
                                      progress=progress, cancel=cancel, pool=pool)
            for index, verdict in verdicts.items():
                alpha[index] = verdict["alpha"]
                drift[index] = verdict["drift"]
                correlation[index] = verdict["correlation"]
        located = located.copy()
        located["path_alpha"] = ("eigenpair", alpha)
        located["path_drift"] = ("eigenpair", drift)
        located["path_correlation"] = ("eigenpair", correlation)
        located["path_ideal"] = ("eigenpair", ideal)
        located.attrs.update({
            "method": "global_path", "path_alpha_max": float(state.path_alpha_max),
            "path_kappa": float(state.path_kappa)})
        return _apply_path_thresholds(located, state.path_tolerance, state.path_correlation)

    def path_test(self, equilibrium, scan, state, band, targets, x, *, progress=None,
                  cancel=None, pool=None, tips=None) -> dict:
        """Re-solve each target on the paths ``+alpha`` and ``-alpha``.

        ``targets`` are ``(key, omega, peak x, xi^psi on x)``. Each gets the
        largest alpha whose path stays below ``path_kappa`` of the singular-
        pair distance of every continuum extremum its frequency faces
        (:func:`supernova.solve.gaps.path_alpha`), rounded down to a shared
        level so modes can share pencils. A path that cannot rise a quarter of
        the local element at the mode's peak is no test: inconclusive.
        Returns ``{key: {"label", "alpha", "drift", "correlation"}}``.
        """
        from supernova.eigen.region import _ellipse_region_dict
        from supernova.operators.contour import GlobalPath
        from supernova.solve.contour import ContourProblem
        from supernova.solve.gaps import path_alpha

        if not targets:
            return {}
        if progress:
            progress(f"Path test: the complex-path operator for {len(targets)} mode(s)...")
        problem = ContourProblem(scan.solver, int(state.ntor), max_dm=int(state.max_dm),
                                 mmax=state.mmax_limit, regularizer=float(state.regularizer),
                                 sound="nova" if state.sound_model == "nova" else "ideal")
        if tips is None:
            tips = self.continuum_tips(equilibrium, state)
        low, high = problem.domain
        mesh = np.asarray(scan.grid.knots, dtype=float)
        verdicts, levels, regions = {}, {}, {}
        for key, omega, peak, _ in targets:
            frequency = float(scan.band_for(omega.real, units="kHz"))
            wanted, _ = path_alpha(frequency, tips, problem.domain,
                                   kappa=float(state.path_kappa),
                                   alpha_max=float(state.path_alpha_max))
            level = _alpha_level(wanted, float(state.path_alpha_max))
            span = np.clip(np.searchsorted(mesh, peak) - 1, 0, mesh.size - 2)
            height = level * (peak - low) * (high - peak)
            if height < 0.25 * (mesh[span + 1] - mesh[span]):
                verdicts[key] = {"label": "inconclusive", "alpha": level,
                                 "drift": np.nan, "correlation": np.nan}
                continue
            half = max(float(state.window_width_khz), 0.02 * frequency)
            _, region = band.region(max(frequency - half, 1.0e-6), frequency + half)
            target_s = complex(omega.real * (1.0 + TARGET_OFFSET)) ** 2
            regions[key] = (target_s, dict(_ellipse_region_dict(region, target_s), vscale=1.0))
            levels.setdefault(level, []).append(key)

        found = {key: [] for members in levels.values() for key in members}
        reference = {key: (omega, fields) for key, omega, _, fields in targets}
        # converge well inside the acceptance gate: at tol = gate a true root
        # can land just outside it
        settings = dict(prefix="nova_path_pep", pep_type=state.pep_type,
                        st_type=state.st_type, which=state.which, nev=int(state.path_nev),
                        tol=min(float(state.tol), FOLLOW_UP_TOL), maxit=int(state.maxit),
                        options=state.petsc_options().as_dict(), verbose=False,
                        compute_residuals=True)
        for level, members in sorted(levels.items(), reverse=True):
            for sign in (1.0, -1.0):
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("Stopped during the path test.")
                if progress:
                    progress(f"Path test: alpha = {sign * level:+.4g} for {len(members)} "
                             "mode(s)...")
                pencil = problem.pencil(GlobalPath(sign * level, domain=problem.domain))
                solutions = _solve_gated(
                    {"gamma_s": pencil.gamma_s, "T0": pencil.T0, "T1": pencil.T1,
                     "Tminus1": pencil.Tminus1, "settings": settings},
                    [(key, *regions[key]) for key in members], pool, cancel)
                for key in members:
                    found[key].append(_best_match(solutions.get(key), pencil, x,
                                                  reference[key][1], reference[key][0]))
        for level, members in levels.items():
            for key in members:
                matches = found[key]
                if any(match is None for match in matches):
                    label, worst_drift, worst_correlation = "inconclusive", np.nan, np.nan
                else:
                    worst_drift = max(match[0] for match in matches)
                    worst_correlation = min(match[1] for match in matches)
                    invariant = (worst_drift <= float(state.path_tolerance)
                                 and worst_correlation >= float(state.path_correlation))
                    label = "gap" if invariant else "continuum"
                verdicts[key] = {"label": label, "alpha": level, "drift": worst_drift,
                                 "correlation": worst_correlation}
        return verdicts

    # ------------------------------------------------------- refinement

    def refine(self, equilibrium, state: SolverState, request: dict, *, title: str = "",
               progress=None, cancel=None, pool=None) -> RefineResult:
        """Re-solve selected modes on meshes graded towards them, coarse to fine.

        ``request`` holds, per mode, ``modes`` (indices), ``omega`` (normalized),
        ``frequency_khz``, ``peak_psi`` (where its envelope peaks, psi_N) and
        ``fields`` -- ``xi^psi`` on the solve's reconstruction grid -- to
        follow it by shape from mesh to mesh.

        The finest element at a mode is ``refine_h_mm``, or when that is 0 the
        smaller of the resolution guide's element for its width and edge
        distance and a quarter of the current element. Each coarser mesh
        doubles it, never above the current element. With
        ``refine_path_check`` the modes are path-tested on the finest mesh.
        Each row: ``mode, level, elements, h_mm, frequency_khz, growth_ratio,
        correlation, backward_error, fwhm_mm`` (``frequency_khz`` nan when
        nothing converged).
        """
        from supernova.api import Band, Case, ModeSet, RadialGrid, Scan
        from supernova.eigen.region import _ellipse_region_dict
        from supernova.radial.grading import graded_breakpoints
        from supernova.solve.gaps import RadialGeometry, harmonic_correlation, local_scales_mm

        state = state.copy()
        started = time.perf_counter()
        modes = [int(mode) for mode in request["modes"]]
        omega = np.asarray(request["omega"], dtype=complex)
        frequency = np.asarray(request["frequency_khz"], dtype=float)
        references = np.asarray(request["fields"], dtype=complex)
        points = references.shape[1]
        continuum = self.continuum_for(equilibrium, state, progress)
        case = Case(equilibrium, continuum,
                    metadata={"gamma_s": float(state.gamma_s), "species": state.species})
        domain = radial_domain(equilibrium, float(state.rho_min), float(state.rho_max))
        geometry = RadialGeometry.from_equilibrium(equilibrium)
        tips = self.continuum_tips(equilibrium, state)
        centers = geometry.x_of_psi_n(np.asarray(request["peak_psi"], dtype=float))
        dr_dx_mm = geometry.dr_dx(centers) * 1.0e3
        base = (domain[1] - domain[0]) / int(state.spans)
        current_mm = base * dr_dx_mm
        if float(state.refine_h_mm) > 0.0:
            finest = np.full(len(modes), float(state.refine_h_mm))
        else:
            finest = []
            for center, f_khz, current in zip(centers, frequency, current_mm):
                w, d = local_scales_mm(geometry, tips, int(state.ntor), center, f_khz)
                guide = (min(w, d) if state.resolution_target == "1e-4"
                         else min(w / 4.0, d / 2.0))
                finest.append(min(guide if np.isfinite(guide) else current, current / 4.0))
            finest = np.maximum(np.asarray(finest), 0.01)
        band = Band(*state.band, units="kHz", vscale=float(state.region_vscale)).for_case(case)
        settings = dict(prefix="nova_refine_pep", pep_type=state.pep_type,
                        st_type=state.st_type, which=state.which,
                        nev=max(2 * int(state.path_nev), 8),
                        tol=min(float(state.tol), FOLLOW_UP_TOL),
                        maxit=int(state.maxit), options=state.petsc_options().as_dict(),
                        verbose=False, compute_residuals=True)

        rows, envelopes, path = [], {mode: [] for mode in modes}, {}
        tracked = references.copy()
        levels = int(state.refine_levels)
        x_psi = None
        for level in range(levels):
            if cancel is not None and cancel.is_set():
                raise RuntimeError("Stopped during the refinement.")
            h_mm = np.minimum(finest * 2.0 ** (levels - 1 - level), current_mm)
            breakpoints = graded_breakpoints(domain, base, centers, h_mm / dr_dx_mm,
                                             growth=float(state.refine_growth))
            elements = int(breakpoints.size - 1)
            if progress:
                progress(f"Refinement mesh {level + 1} of {levels}: {elements} elements, "
                         f"{h_mm.min():.3g} mm at the finest...")
            scan = Scan(case, RadialGrid(breakpoints, int(state.order),
                                         int(state.quadrature_order), domain),
                        ntor=int(state.ntor), mmax=state.mmax_limit,
                        max_dm=int(state.max_dm), regularizer=float(state.regularizer))
            T0, T1, Tminus1 = scan.solver.assemble_split_matrices(
                ntor=int(state.ntor), mmax=state.mmax_limit, max_dm=int(state.max_dm),
                regularizer=float(state.regularizer), sound_model=state.sound_model)
            jobs, regions = [], {}
            for index, f_khz in enumerate(frequency):
                half = max(float(state.window_width_khz), 0.02 * f_khz)
                target_omega, region = band.region(max(f_khz - half, 1.0e-6), f_khz + half)
                target_s = complex(omega[index].real * (1.0 + TARGET_OFFSET)) ** 2
                region = _ellipse_region_dict(region, target_s)
                regions[index] = (target_s, region)
                jobs.append((index, target_s, region))
            solutions = _solve_gated({"gamma_s": _dispatch_gamma(scan, state), "T0": T0,
                                      "T1": T1, "Tminus1": Tminus1, "settings": settings},
                                     jobs, pool, cancel)
            for index, mode in enumerate(modes):
                row = {"mode": mode, "level": level, "elements": elements,
                       "h_mm": float(h_mm[index]), "frequency_khz": np.nan,
                       "growth_ratio": np.nan, "correlation": np.nan,
                       "backward_error": np.nan, "fwhm_mm": np.nan}
                solution = solutions.get(index)
                if solution is not None and np.size(solution.eigenvalues):
                    target_s, region = regions[index]
                    dataset = scan.solver.pack_window(
                        solution, ntor=int(state.ntor), target_omega=np.sqrt(target_s),
                        region=region, mmax=state.mmax_limit, max_dm=int(state.max_dm),
                        regularizer=float(state.regularizer), pep_type=state.pep_type,
                        st_type=state.st_type, which=state.which,
                        sound_model=state.sound_model)
                    found = ModeSet(dataset, scan)
                    xi = found.field("xi_psi", n=points)
                    if x_psi is None:
                        x_psi = geometry.at("psi_n", np.asarray(xi.coords["x"].values))
                    fields = np.asarray(xi.values)
                    errors = (np.asarray(dataset["backward_error"].values, dtype=float)
                              if "backward_error" in dataset else np.zeros(len(found)))
                    correlations = np.array([
                        harmonic_correlation(field, tracked[index]) if error <= BACKWARD_ERROR_GATE
                        else -1.0 for field, error in zip(fields, errors)])
                    best = int(np.argmax(correlations))
                    # below the floor it is another root: the mode is lost on this
                    # mesh. On the first refined mesh an unresolved mode may change
                    # shape a lot, so it only has to be recognizably the same one.
                    floor = TRACK_FIRST if level == 0 else float(state.path_correlation)
                    if correlations[best] >= floor:
                        w = complex(found.omega[best])
                        envelope = np.sqrt(np.sum(np.abs(fields[best]) ** 2, axis=1))
                        envelope = envelope / max(envelope.max(), 1e-300)
                        x_grid = np.asarray(xi.coords["x"].values)
                        r_mm = geometry.at("r", x_grid) * 1.0e3
                        above = np.flatnonzero(envelope >= 0.5)
                        if above[-1] - above[0] <= 3:
                            # narrower than the drawing grid: measure it on a fine one
                            peak = int(np.argmax(envelope))
                            dense = np.linspace(x_grid[max(peak - 4, 0)],
                                                x_grid[min(peak + 4, x_grid.size - 1)], 2001)
                            local = found._evaluate("xi_psi", dense)[best]
                            fine = np.sqrt(np.sum(np.abs(local) ** 2, axis=1))
                            r_mm = geometry.at("r", dense) * 1.0e3
                            above = np.flatnonzero(fine >= 0.5 * fine.max())
                        row.update(frequency_khz=float(found.frequency_khz[best]),
                                   growth_ratio=float(w.imag / w.real) if w.real else np.nan,
                                   correlation=float(correlations[best]),
                                   backward_error=float(errors[best]),
                                   fwhm_mm=float(r_mm[above[-1]] - r_mm[above[0]]))
                        tracked[index] = fields[best]
                        omega[index] = w
                        frequency[index] = row["frequency_khz"]
                        envelopes[mode].append(envelope)
                rows.append(row)
            if level == levels - 1 and state.refine_path_check:
                targets = [(mode, complex(omega[index]), float(centers[index]),
                            tracked[index]) for index, mode in enumerate(modes)
                           if np.isfinite(rows[-len(modes) + index]["frequency_khz"])]
                if targets:
                    path = self.path_test(
                        equilibrium, scan, state, band, targets,
                        np.linspace(domain[0], domain[1], points), progress=progress,
                        cancel=cancel, pool=pool, tips=tips)
        return RefineResult(
            state=state, title=title, modes=tuple(modes), rows=tuple(rows),
            x=np.asarray(x_psi if x_psi is not None else np.empty(0), dtype=float),
            envelopes={mode: np.asarray(values) for mode, values in envelopes.items()},
            path=path, seconds=time.perf_counter() - started)

    @staticmethod
    def _solve_windows(scan, band, state, found, progress, cancel) -> list:
        """Every window on one assembled pencil; datasets with roots go to ``found``.

        A window that raises costs that window only: the sweep carries on with
        a freshly assembled pencil for the windows after it.
        """
        nev = int(state.nev)
        settings = dict(
            ntor=int(state.ntor), mmax=state.mmax_limit, max_dm=int(state.max_dm),
            regularizer=float(state.regularizer), pep_type=state.pep_type,
            st_type=state.st_type, which=state.which, nev=nev, tol=float(state.tol),
            maxit=int(state.maxit), options=state.petsc_options().as_dict(),
            verbose=bool(state.verbose), sound_model=state.sound_model,
        )
        windows = state.windows_khz()
        remaining = list(enumerate(windows))
        rows = []

        def stopped():
            return cancel is not None and cancel.is_set()

        while remaining and not stopped():
            targets = [band.region(low, high) for _, (low, high) in remaining]
            solves = scan.solver.solve_pep_windows(windows=targets, **settings)
            try:
                while remaining and not stopped():
                    index, (low, high) = remaining.pop(0)
                    if progress:
                        progress(f"Window {index + 1} of {len(windows)}: "
                                 f"{low:.3f} - {high:.3f} kHz...")
                    started = time.perf_counter()
                    try:
                        dataset = next(solves)
                    except Exception as error:
                        rows.append(WindowRow(index, (low, high), nev, status="failed",
                                              message=f"{type(error).__name__}: {error}",
                                              seconds=time.perf_counter() - started))
                        break
                    dataset, dropped = _gate(dataset)
                    rows.append(_window_row(index, (low, high), nev, dataset,
                                            time.perf_counter() - started, dropped))
                    if dataset.sizes.get("eigenpair", 0):
                        found.append(dataset.assign_coords(
                            window=("eigenpair", np.full(dataset.sizes["eigenpair"], index))))
            finally:
                solves.close()
        rows.extend(WindowRow(index, window, nev, status="skipped")
                    for index, window in remaining)
        return rows

    @staticmethod
    def _solve_windows_pooled(scan, band, state, found, progress, cancel, pool,
                              log) -> list:
        """Assemble the pencil here once, solve the windows in ``pool``.

        Each finished window is packed here while the workers carry on, so
        reconstruction overlaps the remaining solves.
        """
        from supernova.eigen.region import _ellipse_region_dict
        from supernova.gui._workers import new_pencil_key

        nev = int(state.nev)
        windows = state.windows_khz()
        if progress:
            progress(f"Assembling the pencil once for {len(windows)} windows...")
        T0, T1, Tminus1 = scan.solver.assemble_split_matrices(
            ntor=int(state.ntor), mmax=state.mmax_limit, max_dm=int(state.max_dm),
            regularizer=float(state.regularizer), verbose=bool(state.verbose),
            sound_model=state.sound_model)
        pencil = {
            "gamma_s": _dispatch_gamma(scan, state), "T0": T0, "T1": T1,
            "Tminus1": Tminus1,
            "settings": dict(
                prefix="nova_quadratic_pep", pep_type=state.pep_type,
                st_type=state.st_type, which=state.which, nev=nev,
                tol=float(state.tol), maxit=int(state.maxit),
                options=state.petsc_options().as_dict(),
                verbose=bool(state.verbose), compute_residuals=True),
        }
        targets, jobs = {}, []
        for index, (low, high) in enumerate(windows):
            target_omega, region = band.region(low, high)
            target_s = complex(target_omega) ** 2
            region = _ellipse_region_dict(region, target_s)
            targets[index] = (target_omega, region)
            jobs.append((index, target_s, region))

        rows = {}

        def on_window(index, solution, error, text, seconds):
            window = windows[index]
            if text and state.verbose and log is not None:
                log.write(text)
            if solution is not None:
                target_omega, region = targets[index]
                try:
                    dataset = scan.solver.pack_window(
                        solution, ntor=int(state.ntor), target_omega=target_omega,
                        region=region, mmax=state.mmax_limit, max_dm=int(state.max_dm),
                        regularizer=float(state.regularizer), pep_type=state.pep_type,
                        st_type=state.st_type, which=state.which,
                        verbose=bool(state.verbose), sound_model=state.sound_model)
                except Exception as packing:
                    solution, error = None, f"{type(packing).__name__}: {packing}"
            if solution is None:
                rows[index] = WindowRow(index, window, nev, status="failed",
                                        message=error, seconds=seconds)
            else:
                dataset, dropped = _gate(dataset)
                rows[index] = _window_row(index, window, nev, dataset, seconds, dropped)
                if dataset.sizes.get("eigenpair", 0):
                    found.append(dataset.assign_coords(
                        window=("eigenpair", np.full(dataset.sizes["eigenpair"], index))))
            if progress:
                progress(f"{len(rows)} of {len(windows)} windows solved on "
                         f"{pool.workers} worker(s) x {pool.threads} thread(s)...")

        if progress:
            progress(f"Solving {len(windows)} windows on {pool.workers} worker(s) x "
                     f"{pool.threads} thread(s)...")
        pool.solve(new_pencil_key(), pencil, jobs, on_window, cancel)
        return [rows.get(index, WindowRow(index, window, nev, status="skipped"))
                for index, window in enumerate(windows)]

    @staticmethod
    def _pack(state, title, rows, modes, located, scan, started, equilibrium,
              crossing_continuum=None):
        """Copy out everything the window draws, on this (the worker) thread."""
        from supernova.solve.modelocate import branch_crossings

        if crossing_continuum is None:
            continuum = scan.continuum
            psi = np.asarray(continuum.coords["psi"].values, dtype=float)
            branches = np.asarray(continuum["omega"].values, dtype=complex)
        else:
            psi, branches = crossing_continuum
        continuum_khz = scan.band_for(branches.real, units="kHz")
        # a root with s < 0 has an imaginary omega; it is not on a kHz axis
        continuum_khz = np.where(np.abs(branches.imag) > 1.0e-6 * np.abs(branches),
                                 np.nan, continuum_khz)

        grid = np.linspace(0.0, 1.0, 2001)
        psi_grid = scan.rho_pol(grid) ** 2

        def to_psi(x):
            return np.interp(np.asarray(x, dtype=float), grid, psi_grid)

        thresholds = state.locate_thresholds()
        mesh = np.asarray(scan.grid.knots, dtype=float)
        walls = (float(to_psi(mesh[0])), float(to_psi(mesh[-1])))

        empty = np.empty(0)
        if modes is None:
            arrays = dict(frequency_khz=empty, omega=empty.astype(complex), center=empty,
                          width=empty, dominant_m=empty.astype(int),
                          backward_error=empty, window_of=empty.astype(int),
                          crossings=(), x=empty, m_pol=empty.astype(int), fields={},
                          envelope_x=empty)
            dataset = None
        else:
            points = int(state.reconstruction_points)
            fields = {name: np.asarray(modes.field(name, n=points).values)
                      for name in FIELDS}
            xi = modes.field("xi_psi", n=points)
            power = np.trapezoid(np.abs(fields["xi_psi"]) ** 2,
                                 np.asarray(xi.coords["x"].values), axis=1)
            m_pol = np.asarray(xi.coords["m_pol"].values, dtype=int)
            dataset = modes.dataset
            backward = (np.asarray(dataset["backward_error"].values, dtype=float)
                        if "backward_error" in dataset else np.full(len(modes), np.nan))
            center = np.asarray(modes.centroid, dtype=float)
            # a width is a length along x; carry it over with the local slope
            slope = np.interp(center, grid, np.gradient(psi_grid, grid))
            arrays = dict(
                frequency_khz=np.asarray(modes.frequency_khz, dtype=float),
                omega=np.asarray(modes.omega, dtype=complex),
                center=to_psi(center),
                width=np.asarray(modes.width, dtype=float) * slope,
                dominant_m=m_pol[np.argmax(power, axis=1)],
                backward_error=backward,
                window_of=np.asarray(dataset.coords["window"].values, dtype=int),
                crossings=tuple(to_psi(branch_crossings(psi, branches, omega))
                                for omega in modes.omega),
                x=to_psi(xi.coords["x"].values), m_pol=m_pol,
                fields=fields,
                envelope_x=to_psi(located.coords["x"].values),
            )
        return SolverResult(
            state=state, title=title, windows=tuple(rows), dataset=dataset,
            located=located, continuum_x=to_psi(psi),
            continuum_khz=np.asarray(continuum_khz, dtype=float),
            # MODELOCATE's core and edge, where it judged anything
            psi_core=0.0 if state.mode_filter == "path" else float(thresholds.core_edge) ** 2,
            psi_edge=1.0 if state.mode_filter == "path" else float(thresholds.boundary_edge) ** 2,
            walls=walls, seconds=time.perf_counter() - started,
            equilibrium=equilibrium, **arrays)

    @staticmethod
    def export(result: SolverResult, path) -> Path:
        """Write the modes, their MODELOCATE verdicts and the window table.

        One netCDF file: the eigenpair dataset as :meth:`supernova.api.ModeSet.
        to_netcdf` writes it, plus per-mode ``frequency_khz``, ``center``,
        ``width``, ``mode_class``, ``mode_label`` and ``peak_rho``, plus
        ``window_*`` variables along a ``search_window`` dimension (each
        mode's own ``window`` coordinate indexes it). The settings go in the
        ``gui_settings`` attribute as JSON.
        """
        import json

        from supernova.api import ModeSet

        if result.dataset is None:
            raise ValueError("There are no modes to export.")
        dataset = result.dataset.copy()
        located = result.located
        dataset["frequency_khz"] = ("eigenpair", result.frequency_khz)
        dataset["center"] = ("eigenpair", result.center)
        dataset["center"].attrs["units"] = "psi_N"
        dataset["width"] = ("eigenpair", result.width)
        dataset["width"].attrs["units"] = "psi_N"
        for name in ("mode_class", "mode_label", "peak_x", "peak_rho"):
            dataset[name] = ("eigenpair", np.asarray(located[name].values))
        dataset["mode_class"].attrs = dict(located["mode_class"].attrs)
        rows = result.windows
        dimension = "search_window"
        dataset["window_low_khz"] = (dimension, [row.range_khz[0] for row in rows])
        dataset["window_high_khz"] = (dimension, [row.range_khz[1] for row in rows])
        for name in ("nconv", "kept", "dropped", "iterations"):
            dataset[f"window_{name}"] = (dimension, [getattr(row, name) for row in rows])
        dataset["window_status"] = (dimension, [row.status for row in rows])
        dataset.attrs = {key: value for key, value in dataset.attrs.items()
                         if isinstance(value, (str, int, float))
                         and not isinstance(value, bool)}
        dataset.attrs.update({
            "equilibrium": result.title,
            "modelocate": json.dumps({k: located.attrs[k] for k in located.attrs
                                      if k != "method"}),
            "gui_settings": json.dumps(result.state.to_dict()),
            "written_by": "supernova.gui.solver",
        })
        path = Path(path)
        ModeSet(dataset, None).to_netcdf(path)
        return path


def _gate(dataset):
    """``(dataset, dropped)``: the window's roots inside BACKWARD_ERROR_GATE only.

    A root outside it is not an eigenpair of the pencil to any useful accuracy
    -- an inaccurate factorization returns a pile of them on the shift -- so it
    never reaches the filter, which would judge its envelope like any other.
    """
    if "backward_error" not in dataset or not dataset.sizes.get("eigenpair", 0):
        return dataset, 0
    errors = np.asarray(dataset["backward_error"].values, dtype=float)
    keep = np.flatnonzero(~(errors > BACKWARD_ERROR_GATE))     # nan: not measured, kept
    return dataset.isel(eigenpair=keep), int(errors.size - keep.size)


def _window_row(index, window, nev, dataset, seconds, dropped=0) -> WindowRow:
    """What one window did, read from its dataset's SLEPc attributes."""
    attrs = dataset.attrs
    nconv = int(attrs.get("nconv", 0))
    kept = int(dataset.sizes.get("eigenpair", 0))
    reason = str(attrs.get("reason", "")).strip()
    if nconv == 0:
        status = "empty" if reason in ("1", "2") else "unconverged"
    else:
        status = "saturated" if nconv >= nev else "ok"
    return WindowRow(index, window, nev, nconv=nconv, kept=kept, dropped=int(dropped),
                     iterations=int(attrs.get("iterations", 0)), reason=reason,
                     status=status, seconds=seconds)


def _merge_windows(found: list, scan):
    """Concatenate the windows' roots, sorted by frequency, duplicates dropped.

    A root on the boundary of two windows, or inside an overlap, is returned
    by both; the two copies agree to solver tolerance.
    """
    import xarray as xr

    from supernova.api import ModeSet

    if not found:
        return None
    merged = xr.concat(found, dim="eigenpair", data_vars="minimal",
                       coords="minimal", compat="override",
                       combine_attrs="drop_conflicts")
    omega = np.asarray(merged["omega"].values, dtype=complex)
    keep = []
    for index in np.argsort(omega.real, kind="stable"):
        if keep and abs(omega[index] - omega[keep[-1]]) <= (
                DUPLICATE_RELATIVE * abs(omega[index])):
            continue
        keep.append(index)
    merged = merged.isel(eigenpair=np.asarray(keep))
    merged = merged.assign_coords(eigenpair=np.arange(merged.sizes["eigenpair"]))
    return ModeSet(merged, scan)


def _without_parallel_sound(blocks):
    """The solver blocks with the parallel-acoustic rows of ``A`` dropped.

    In the continuum pencil ``A - mu_0 rho s B`` the rows past the Alfven ones
    are the parallel sound; without them the slow-sound roots fall to s = 0,
    which is the continuum the NOVA sound model (``ksound=1``) leaves.
    """
    import copy

    from scipy.sparse import coo_array

    A = coo_array(blocks.A)
    q, rows, cols = A.coords
    keep = rows < len(blocks.m_range)
    out = copy.copy(blocks)
    out.A = coo_array((A.data[keep], (q[keep], rows[keep], cols[keep])), shape=A.shape)
    return out


def _nova_continuum(equilibrium, continuum, state, x) -> np.ndarray:
    """``(nx, nbranch)`` normalized omega of the NOVA-sound continuum at ``x``."""
    from supernova.api import RadialGrid
    from supernova.solve.contour import _SampleGrid
    from supernova.solve.multigrid import _solve_master_continuum
    from supernova.solve.stability import StabilitySolver

    # a solver whose "quadrature points" are x evaluates every block there
    sampler = StabilitySolver(equilibrium, continuum,
                              _SampleGrid(x, RadialGrid.uniform(8).build()))
    blocks = sampler.build_solver_blocks(ntor=int(state.ntor), mmax=state.mmax_limit,
                                         max_dm=int(state.max_dm))
    dataset = _solve_master_continuum(_without_parallel_sound(blocks), x)
    return np.asarray(dataset["omega"].values, dtype=complex)


def _khz(equilibrium, omega) -> np.ndarray:
    """Normalized omega to kHz; nan where omega is not real (s < 0)."""
    import unyt

    omega = np.asarray(omega, dtype=complex)
    real = np.where(np.abs(omega.imag) > 1.0e-6 * np.abs(omega), np.nan, omega.real)
    return np.asarray(unyt.unyt_array(real / (2.0 * np.pi), "omegaA",
                                      registry=equilibrium.registry).to("kHz").v, dtype=float)


# no continuum at all: MODELOCATE then runs its spike, core and edge tests only
_NO_CONTINUUM = (np.array([0.0, 1.0]), np.full((2, 0), np.nan, dtype=complex))


def _apply_path_thresholds(located, tolerance: float, min_correlation: float):
    """The path verdicts at these thresholds, on a copy of ``located``.

    Gap if a mode moved by no more than ``tolerance`` (relative) on either path
    and kept a harmonic correlation of at least ``min_correlation``; continuum
    otherwise. What the paths could not judge -- an ideal mode, or a test that
    found nothing or could not rise -- stays so whatever the thresholds.
    """
    drift = np.asarray(located["path_drift"].values, dtype=float)
    correlation = np.asarray(located["path_correlation"].values, dtype=float)
    ideal = (np.asarray(located["path_ideal"].values, dtype=bool) if "path_ideal" in located
             else np.asarray(located["mode_class"].values) == 16)
    judged = np.isfinite(drift) & np.isfinite(correlation)
    gap = judged & (drift <= float(tolerance)) & (correlation >= float(min_correlation))
    labels = np.where(ideal, "ideal", np.where(~judged, "inconclusive",
                                               np.where(gap, "gap", "continuum")))
    codes = np.where(ideal, 16, np.where(~judged, INCONCLUSIVE, np.where(gap, 0, 8)))
    located = located.copy()
    located["mode_label"] = ("eigenpair", labels.astype(str))
    located["mode_class"] = ("eigenpair", codes.astype(np.int32))
    located["mode_class"].attrs["legend"] = PATH_LEGEND
    located.attrs.update(path_tolerance=float(tolerance),
                         path_correlation=float(min_correlation))
    return located


def _dispatch_gamma(scan, state) -> float:
    """What the eigensolver dispatches on: without the 1/s term it is linear."""
    return 0.0 if state.sound_model == "nova" else float(scan.solver.gamma_s)


def _alpha_level(alpha: float, alpha_max: float) -> float:
    """``alpha_max * 2**(-k/2)``: the largest such level not above ``alpha``."""
    if alpha >= alpha_max:
        return float(alpha_max)
    if not alpha > 0.0:
        return 0.0
    k = int(np.ceil(2.0 * np.log2(alpha_max / alpha) - 1.0e-12))
    return float(alpha_max * 2.0 ** (-k / 2.0))


def _solve_jobs(pencil: dict, jobs, pool, cancel) -> dict:
    """``{key: WindowSolution or None}`` for ``(key, target_s, region)`` jobs.

    In the pool's worker processes when there is one -- the compute process
    never runs an eigensolver itself -- else here, one after another.
    """
    out = {}
    if pool is not None:
        from supernova.gui._workers import new_pencil_key

        def on_window(index, solution, error, text, seconds):
            out[index] = solution

        pool.solve(new_pencil_key(), pencil, list(jobs), on_window, cancel)
        return out

    from supernova.eigen.dispatch import build_eigensolver
    from supernova.solve.stability import WindowSolution

    eigensolver = None
    try:
        for key, target_s, region in jobs:
            if cancel is not None and cancel.is_set():
                break
            try:
                if eigensolver is None:
                    eigensolver = build_eigensolver(
                        gamma_s=pencil["gamma_s"], T0=pencil["T0"], T1=pencil["T1"],
                        Tminus1=pencil["Tminus1"], target_s=target_s, region=region,
                        **pencil["settings"])
                else:
                    eigensolver.retarget(target_s, region)
                result = eigensolver.solve(allow_empty=True)
                out[key] = WindowSolution.from_solver(eigensolver, result)
            except Exception:
                out[key] = None
                if eigensolver is not None:
                    with contextlib.suppress(Exception):
                        eigensolver.destroy()
                eigensolver = None
    finally:
        if eigensolver is not None:
            with contextlib.suppress(Exception):
                eigensolver.destroy()
    return out


def _gated(solution) -> bool:
    """True if the solution holds a root inside the production backward-error gate."""
    if solution is None or np.size(solution.eigenvalues) == 0:
        return False
    errors = solution.backward_errors
    return errors is None or bool(np.any(np.asarray(errors, dtype=float) <= BACKWARD_ERROR_GATE))


def _solve_gated(pencil: dict, jobs, pool, cancel) -> dict:
    """:func:`_solve_jobs`, repeating with a moved target every job that
    returned no root inside the gate (see ``RETRY_OFFSETS``)."""
    jobs = list(jobs)
    out = _solve_jobs(pencil, jobs, pool, cancel)
    for offset in RETRY_OFFSETS:
        again = [(key, target_s * (1.0 + offset) ** 2, region)
                 for key, target_s, region in jobs if not _gated(out.get(key))]
        if not again or (cancel is not None and cancel.is_set()):
            break
        for key, solution in _solve_jobs(pencil, again, pool, cancel).items():
            if _gated(solution) or out.get(key) is None:
                out[key] = solution
    return out


def _best_match(solution, pencil, x, reference, omega_reference):
    """``(drift, correlation, omega)`` of the root most like ``reference``.

    Only roots inside the production backward-error gate count; None when
    there are none.
    """
    from supernova.solve.gaps import harmonic_correlation

    if solution is None or np.size(solution.eigenvalues) == 0:
        return None
    s = np.asarray(solution.eigenvalues, dtype=complex)
    errors = solution.backward_errors
    keep = (np.ones(s.size, dtype=bool) if errors is None
            else np.asarray(errors, dtype=float) <= BACKWARD_ERROR_GATE)
    if not keep.any():
        return None
    vectors = np.asarray(solution.eigenvectors)[keep]
    fields = pencil.xi_psi(vectors, x)
    correlations = np.array([harmonic_correlation(field, reference) for field in fields])
    best = int(np.argmax(correlations))
    omega = np.sqrt(s[keep][best])
    if omega.real < 0.0:
        omega = -omega
    return (float(abs(omega - omega_reference) / abs(omega_reference)),
            float(correlations[best]), complex(omega))
