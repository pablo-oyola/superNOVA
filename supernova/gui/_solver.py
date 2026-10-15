"""The solver tab without Qt: its state, the solve, and what gets drawn.

The tab runs the simple pass: one radial grid, the band cut into windows,
one shift-and-invert solve per window. Every root it finds then goes through Gerrit Kramer's
MODELOCATE filter (:meth:`supernova.api.ModeSet.locate`), which calls a mode a
gap mode unless its envelope peaks on a single grid point, in the core, at the
edge, or on a continuum crossing.

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
from dataclasses import dataclass, field, fields
from pathlib import Path
import time

import numpy as np

from supernova.gui._session import JsonState

SPECIES = ("H", "D", "T", "He3", "He4")
FACTORIZATIONS = ("mumps", "spqr", "petsc_qr")
PEP_TYPES = ("TOAR", "STOAR", "QARNOLDI", "LINEAR", "JD")
ST_TYPES = ("SINVERT", "CAYLEY", "SHIFT")
WHICH = ("TARGET_MAGNITUDE", "TARGET_REAL", "TARGET_IMAGINARY")
FIELDS = ("xi_psi", "p1", "xi_s", "div_xi")
PARTS = ("real", "imag", "abs")
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
}

# fields that never change what is solved: what is drawn, and how the
# windows are spread over worker processes
_VIEW_FIELDS = frozenset({"field", "part", "continuum_preview", "gap_only",
                          "workers", "threads_per_worker"})


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

    # the problem
    ntor: int = 4
    mmax: int = 40
    max_dm: int = 5
    regularizer: float = 0.0
    gamma_s: float = 5.0 / 3.0
    species: str = "D"

    # the frequency search
    f_low_khz: float = 50.0
    f_high_khz: float = 150.0
    window_width_khz: float = 2.5     # the size of each subrange
    window_overlap: float = 0.0       # fraction of a window shared with the next
    nev: int = 8                      # roots requested per window
    region_vscale: float = 0.25       # imaginary semi-axis of the ellipse
    tol: float = 1.0e-8
    maxit: int = 200

    # the radial discretization, between walls at these rho_pol (clipped to
    # the equilibrium's Boozer domain)
    spans: int = 150
    order: int = 3
    quadrature_order: int = 6
    rho_min: float = 0.0
    rho_max: float = 1.0

    # the eigensolver and its linear solves
    pep_type: str = "TOAR"
    st_type: str = "SINVERT"
    which: str = "TARGET_MAGNITUDE"
    factorization: str = "mumps"
    mumps_icntl_14: int = 200         # % workspace headroom; 20 fails on SLEPc 3.25
    extra_options: str = ""

    # MODELOCATE, as overrides of ModeLocateThresholds
    modelocate: dict = field(default_factory=dict)

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
        if self.mmax < 1:
            problems.append("mmax must be at least 1.")
        if self.max_dm < 1:
            problems.append("max_dm must be at least 1.")
        if not np.isfinite(self.regularizer) or self.regularizer < 0.0:
            problems.append("The regularizer must be finite and non-negative.")
        if not np.isfinite(self.gamma_s) or self.gamma_s < 0.0:
            problems.append("gamma_s must be finite and non-negative.")
        if self.species not in SPECIES:
            problems.append(f"Species must be one of {', '.join(SPECIES)}.")

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
        return (self.ntor, self.mmax, self.max_dm, self.gamma_s, self.species)


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


@dataclass(frozen=True)
class WindowRow:
    window: int
    range_khz: tuple
    nev: int
    nconv: int = 0
    kept: int = 0
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
            })
        return rows

    def field(self, mode: int, name: str, part: str) -> np.ndarray:
        values = self.fields[name][int(mode)]
        return {"real": values.real, "imag": values.imag, "abs": np.abs(values)}[part]

    def envelope(self, mode: int):
        """MODELOCATE's envelope for one mode, on the grid it was judged on."""
        return (self.envelope_x,
                np.asarray(self.located["envelope"].values[int(mode)], dtype=float))


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

    def preview(self, equilibrium, state: SolverState, *, points: int = 256,
                progress=None) -> ContinuumPreview:
        """The continuum for this n, to choose a band against before solving."""
        started = time.perf_counter()
        continuum = self.continuum_for(equilibrium, state, progress)
        which = "alfven" if float(state.gamma_s) == 0.0 else "full"
        x = np.linspace(0.0, 1.0, int(points))
        if progress:
            progress(f"Solving the {which} continuum for n = {state.ntor}...")
        eigenvalues = continuum.solve(ntor=int(state.ntor), mmax=int(state.mmax),
                                      psi_vals=x, which=which, normalize_eigvals=True,
                                      max_dm=int(state.max_dm))
        with np.errstate(invalid="ignore"):
            khz = np.sqrt(eigenvalues.real).to("kHz").v / (2.0 * np.pi)
        grid, psi_n = _flux_maps(equilibrium)
        return ContinuumPreview(x=np.interp(x, grid, psi_n),
                                frequency_khz=np.asarray(khz, dtype=float),
                                which=which, key=state.preview_key(),
                                seconds=time.perf_counter() - started)

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
                    ntor=int(state.ntor), mmax=int(state.mmax),
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

        modes = _merge_windows(found, scan)
        if modes is None:
            return self._pack(state, title, rows, None, None, scan, started, equilibrium)
        if progress:
            progress(f"Reconstructing {len(modes)} modes and running MODELOCATE...")
        located = modes.locate(thresholds=state.locate_thresholds())
        return self._pack(state, title, rows, modes, located, scan, started, equilibrium)

    @staticmethod
    def _solve_windows(scan, band, state, found, progress, cancel) -> list:
        """Every window on one assembled pencil; datasets with roots go to ``found``.

        A window that raises costs that window only: the sweep carries on with
        a freshly assembled pencil for the windows after it.
        """
        nev = int(state.nev)
        settings = dict(
            ntor=int(state.ntor), mmax=int(state.mmax), max_dm=int(state.max_dm),
            regularizer=float(state.regularizer), pep_type=state.pep_type,
            st_type=state.st_type, which=state.which, nev=nev, tol=float(state.tol),
            maxit=int(state.maxit), options=state.petsc_options().as_dict(),
            verbose=bool(state.verbose),
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
                    rows.append(_window_row(index, (low, high), nev, dataset,
                                            time.perf_counter() - started))
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
            ntor=int(state.ntor), mmax=int(state.mmax), max_dm=int(state.max_dm),
            regularizer=float(state.regularizer), verbose=bool(state.verbose))
        pencil = {
            "gamma_s": float(scan.solver.gamma_s), "T0": T0, "T1": T1,
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
                        region=region, mmax=int(state.mmax), max_dm=int(state.max_dm),
                        regularizer=float(state.regularizer), pep_type=state.pep_type,
                        st_type=state.st_type, which=state.which,
                        verbose=bool(state.verbose))
                except Exception as packing:
                    solution, error = None, f"{type(packing).__name__}: {packing}"
            if solution is None:
                rows[index] = WindowRow(index, window, nev, status="failed",
                                        message=error, seconds=seconds)
            else:
                rows[index] = _window_row(index, window, nev, dataset, seconds)
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
    def _pack(state, title, rows, modes, located, scan, started, equilibrium):
        """Copy out everything the window draws, on this (the worker) thread."""
        from supernova.solve.modelocate import branch_crossings

        continuum = scan.continuum
        psi = np.asarray(continuum.coords["psi"].values, dtype=float)
        branches = np.asarray(continuum["omega"].values, dtype=complex)
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
            psi_core=float(thresholds.core_edge) ** 2,
            psi_edge=float(thresholds.boundary_edge) ** 2,
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
        for name in ("nconv", "kept", "iterations"):
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


def _window_row(index, window, nev, dataset, seconds) -> WindowRow:
    """What one window did, read from its dataset's SLEPc attributes."""
    attrs = dataset.attrs
    nconv = int(attrs.get("nconv", 0))
    kept = int(dataset.sizes.get("eigenpair", 0))
    reason = str(attrs.get("reason", "")).strip()
    if nconv == 0:
        status = "empty" if reason in ("1", "2") else "unconverged"
    else:
        status = "saturated" if nconv >= nev else "ok"
    return WindowRow(index, window, nev, nconv=nconv, kept=kept,
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
