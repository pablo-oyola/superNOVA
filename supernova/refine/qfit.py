"""Reusable fixed-boundary TokaMaker q-profile fitting workflow.

Moved verbatim from ``no_sync/tokamaker_test.py``, which was neither a test nor
version controlled.  The module is deliberately side-effect free so the fitter
can be driven from a script, from :mod:`supernova.refine.fixed_boundary`, or
from an interactive session.

OpenFUSIONToolkit is imported at module scope, so ``import supernova`` never
reaches it -- only ``import supernova.refine`` does.  Install it in the
environment that runs the refinement; the solver environment does not need it.

The default TokaMaker source mode is made explicit here: ``mode=1`` means the
``ffp_profile`` input is the Grad--Shafranov source :math:`F F'`, not
:math:`F'` alone.  Pressure is kept fixed while the shape of :math:`F F'` and,
optionally, the total plasma current are fitted to a target q profile.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from OpenFUSIONToolkit import OFT_env
    from OpenFUSIONToolkit.TokaMaker import TokaMaker
    from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain
except ImportError as error:  # pragma: no cover - depends on the environment
    raise ImportError(
        "supernova.refine needs OpenFUSIONToolkit, which is not installed in "
        "this environment. Equilibrium refinement runs separately from the "
        "stability solve and hands it a refined eqdsk plus a diagnostics "
        "sidecar, so the two do not have to share an environment."
    ) from error


FloatArray = NDArray[np.float64]


def cumulative_trapezoid(y: ArrayLike, x: ArrayLike) -> FloatArray:
    """Return a cumulative trapezoidal integral with a zero first entry."""
    y_array = np.asarray(y, dtype=float)
    x_array = np.asarray(x, dtype=float)
    if y_array.ndim != 1 or x_array.ndim != 1 or y_array.shape != x_array.shape:
        raise ValueError("x and y must be one-dimensional arrays of equal length")
    if y_array.size < 2:
        raise ValueError("at least two samples are required")
    result = np.zeros_like(y_array)
    result[1:] = np.cumsum(
        0.5 * (y_array[1:] + y_array[:-1]) * np.diff(x_array)
    )
    return result


def normalized_toroidal_flux(psi: ArrayLike, q: ArrayLike) -> FloatArray:
    r"""Compute :math:`s(\psi)=\int_0^\psi q\,d\psi/\int_0^1q\,d\psi`.

    ``psi`` need not contain the endpoints.  The closest sampled q value is
    used to pad the axis and LCFS, matching the convention used by the
    original ITPA fitting script.
    """
    psi_array = np.asarray(psi, dtype=float)
    q_array = np.asarray(q, dtype=float)
    if psi_array.ndim != 1 or q_array.shape != psi_array.shape:
        raise ValueError("psi and q must be one-dimensional arrays of equal length")
    if psi_array.size < 2 or np.any(np.diff(psi_array) <= 0.0):
        raise ValueError("psi must contain at least two strictly increasing samples")
    if not np.all(np.isfinite(q_array)):
        raise ValueError("q contains non-finite values")

    psi_augmented = np.r_[0.0, psi_array, 1.0]
    q_augmented = np.r_[q_array[0], q_array, q_array[-1]]
    toroidal_flux = cumulative_trapezoid(q_augmented, psi_augmented)
    if not np.isfinite(toroidal_flux[-1]) or toroidal_flux[-1] == 0.0:
        raise ValueError("the integrated q profile has zero or non-finite toroidal flux")
    s_augmented = toroidal_flux / toroidal_flux[-1]
    return np.interp(psi_array, psi_augmented, s_augmented)


@dataclass(frozen=True)
class QFitConfig:
    """Controls for the damped finite-difference Gauss--Newton fit."""

    fit_range: tuple[float, float] = (0.03, 0.96)
    fit_points: int = 22
    q_sample_range: tuple[float, float] = (0.01, 0.99)
    q_sample_points: int = 280
    max_iterations: int = 12
    target_rms_error: float = 5.0e-5
    profile_fd_step_abs: float = 0.025
    log_current_fd_step: float = 0.012
    regularization_initial: float = 3.0e-3
    regularization_floor: float = 3.0e-4
    regularization_decay: float = 0.6
    ridge: float = 2.0e-4
    max_profile_step: float = 0.18
    max_log_current_step: float = 0.06
    damping_factors: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)
    ffp_bounds: tuple[float | None, float | None] = (0.02, 2.0)
    fit_plasma_current: bool = True
    verbose: bool = True


@dataclass(frozen=True)
class QFitIteration:
    """One accepted (or terminal) outer fit iteration."""

    iteration: int
    rms_before: float
    max_before: float
    plasma_current_before: float
    regularization: float
    accepted_damping: float | None
    rms_after: float
    max_after: float
    plasma_current_after: float
    nonlinear_solves: int
    failed_trials: int


@dataclass(frozen=True)
class QFitResult:
    """Final coarse-grid q-fit result."""

    converged: bool
    reason: str
    target_coordinate: str
    target_locations: FloatArray
    target_q: FloatArray
    initial_rms_error: float
    initial_max_error: float
    final_rms_error: float
    final_max_error: float
    plasma_current: float
    ffp_x: FloatArray
    ffp_y: FloatArray
    history: tuple[QFitIteration, ...]

    @property
    def ffp_profile(self) -> dict[str, Any]:
        """Return a fresh TokaMaker ``linterp`` profile dictionary."""
        return {
            "type": "linterp",
            "x": self.ffp_x.copy(),
            "y": self.ffp_y.copy(),
        }


@dataclass(frozen=True)
class EquilibriumSamples:
    """Lightweight arrays and global values retained for diagnostics."""

    label: str
    mesh_dx: float
    mesh_points: int
    mesh_cells: int
    nonlinear_iterations: int
    psi: FloatArray
    s: FloatArray
    q: FloatArray
    F: FloatArray
    Fprime: FloatArray
    FFprime: FloatArray
    pressure: FloatArray
    Pprime: FloatArray
    magnetic_axis: FloatArray
    plasma_current: float
    ffp_scale: float
    pressure_scale: float


class FixedBoundaryQFitter:
    r"""Fit a fixed-LCFS TokaMaker equilibrium to an input q profile.

    The workflow is intentionally split into the three requested phases:

    1. :meth:`solve_initial` solves the supplied pressure and :math:`F F'`
       profiles on the coarse mesh without changing either profile.
    2. :meth:`fit_coarse` performs a damped Gauss--Newton fit of the
       :math:`F F'` shape (and optionally ``Ip``) to q on that mesh.
    3. :meth:`refine` continues the accepted *absolute* poloidal flux through
       one or more separately constructed meshes and solves with the fitted
       inputs at each stage.

    Parameters
    ----------
    lcfs:
        Fixed LCFS polygon with columns ``(R, Z)`` in metres.
    F0:
        Vacuum toroidal-field flux function, normally ``B0 * R0`` [T m].
    pressure_profile:
        Initial TokaMaker ``P'`` shape dictionary versus normalized poloidal
        flux.  It is held fixed throughout the q fit.
    ffp_profile:
        Initial TokaMaker :math:`F F'` shape dictionary versus normalized
        poloidal flux.  Its existing knots are the fit parameters.
    q_profile:
        Target dictionary with one-dimensional ``x`` and ``y`` arrays.  ``x``
        is interpreted according to ``q_coordinate``.
    Ip_seed:
        Initial total plasma current target [A].
    p_axis_target:
        TokaMaker axis pressure target relative to its zero LCFS pressure [Pa].
    q_coordinate:
        ``"s"`` for normalized toroidal flux or ``"psi"`` for normalized
        poloidal flux.
    source_mode:
        TokaMaker source convention.  The requested :math:`F F'` workflow uses
        ``1``; ``0`` interprets the same input slot as :math:`F'`.
    oft_environment:
        Optional existing OFT runtime.  OFT permits only one runtime per Python
        kernel, so the class reuses its registered instance when omitted.
    initial_psi_evaluator:
        Optional callable used to warm-start the coarse solve from an existing
        equilibrium.  It receives the TokaMaker finite-element node locations
        with shape ``(n, 2)`` and must return the *absolute*, TokaMaker-native
        poloidal flux at those nodes.  When omitted, :meth:`solve_initial`
        falls back to TokaMaker's analytical ``init_psi()`` guess.
    initial_ffp_scale, initial_pressure_scale:
        Optional physical source-amplitude seeds installed before the warm
        solve.  These are useful when ``ffp_profile`` and ``pressure_profile``
        contain dimensional derivatives imported from an equilibrium file.
    nonlinear_tolerance, max_nonlinear_tolerance:
        The Picard tolerance every solve starts with, and how far
        :meth:`_solve` may loosen it, by decades, for an iteration that stalls.
    refinement_dx_sequence:
        Optional strictly decreasing mesh-size continuation from
        ``coarse_dx`` to ``fine_dx``.  This is useful for shaped equilibria
        whose nonlinear branch cannot be retained by one large remeshing step.
        The default is the single direct refinement ``(fine_dx,)``.
    """

    def __init__(
        self,
        *,
        lcfs: ArrayLike,
        F0: float,
        pressure_profile: Mapping[str, Any],
        ffp_profile: Mapping[str, Any],
        q_profile: Mapping[str, Any],
        Ip_seed: float,
        p_axis_target: float,
        q_coordinate: str = "s",
        coarse_dx: float = 0.04,
        fine_dx: float = 0.02,
        refinement_dx_sequence: Sequence[float] | None = None,
        fe_order: int = 3,
        source_mode: int = 1,
        nthreads: int = 2,
        oft_environment: OFT_env | None = None,
        initial_psi_evaluator: Callable[[FloatArray], ArrayLike] | None = None,
        initial_ffp_scale: float | None = None,
        initial_pressure_scale: float | None = None,
        nonlinear_tolerance: float = 1.0e-8,
        max_nonlinear_tolerance: float = 1.0e-5,
        nonlinear_max_iterations: int = 100,
        solver_output: bool = False,
        fit_config: QFitConfig | None = None,
    ) -> None:
        self.lcfs = np.asarray(lcfs, dtype=float).copy()
        if (
            self.lcfs.ndim != 2
            or self.lcfs.shape[1] != 2
            or self.lcfs.shape[0] < 3
            or not np.all(np.isfinite(self.lcfs))
        ):
            raise ValueError("lcfs must be a finite array with shape (n, 2), n >= 3")

        self.F0 = float(F0)
        self.Ip_seed = float(Ip_seed)
        self.p_axis_target = float(p_axis_target)
        self.coarse_dx = float(coarse_dx)
        self.fine_dx = float(fine_dx)
        self.refinement_dx_sequence = (
            (self.fine_dx,)
            if refinement_dx_sequence is None
            else tuple(float(value) for value in refinement_dx_sequence)
        )
        self.fe_order = int(fe_order)
        self.source_mode = int(source_mode)
        self.nthreads = int(nthreads)
        self._oft_environment = oft_environment
        self.initial_psi_evaluator = initial_psi_evaluator
        self.initial_ffp_scale = (
            None if initial_ffp_scale is None else float(initial_ffp_scale)
        )
        self.initial_pressure_scale = (
            None if initial_pressure_scale is None else float(initial_pressure_scale)
        )
        self.nonlinear_tolerance = float(nonlinear_tolerance)
        self.max_nonlinear_tolerance = float(max_nonlinear_tolerance)
        self.nonlinear_max_iterations = int(nonlinear_max_iterations)
        self.solver_output = bool(solver_output)
        self.fit_config = fit_config or QFitConfig()

        if not np.isfinite(self.F0):
            raise ValueError("F0 must be finite")
        if not np.isfinite(self.Ip_seed) or self.Ip_seed <= 0.0:
            raise ValueError("Ip_seed must be finite and positive")
        if not np.isfinite(self.p_axis_target) or self.p_axis_target <= 0.0:
            raise ValueError("p_axis_target must be finite and positive")
        if self.coarse_dx <= 0.0 or self.fine_dx <= 0.0:
            raise ValueError("coarse_dx and fine_dx must be positive")
        if self.fine_dx >= self.coarse_dx:
            raise ValueError("fine_dx must be smaller than coarse_dx")
        if not self.refinement_dx_sequence:
            raise ValueError("refinement_dx_sequence cannot be empty")
        if not all(
            np.isfinite(value) and value > 0.0
            for value in self.refinement_dx_sequence
        ):
            raise ValueError("refinement_dx_sequence must contain finite positive values")
        refinement_steps = (self.coarse_dx, *self.refinement_dx_sequence)
        if any(next_dx >= previous_dx for previous_dx, next_dx in zip(
            refinement_steps[:-1], refinement_steps[1:]
        )):
            raise ValueError("refinement_dx_sequence must be strictly decreasing")
        if not np.isclose(
            self.refinement_dx_sequence[-1],
            self.fine_dx,
            rtol=0.0,
            atol=10.0 * np.finfo(float).eps * max(1.0, abs(self.fine_dx)),
        ):
            raise ValueError("refinement_dx_sequence must end at fine_dx")
        if self.fe_order < 1:
            raise ValueError("fe_order must be positive")
        if self.source_mode not in (0, 1):
            raise ValueError("source_mode must be 0 (F') or 1 (F F')")
        if self.initial_psi_evaluator is not None and not callable(
            self.initial_psi_evaluator
        ):
            raise TypeError("initial_psi_evaluator must be callable")
        for value, name in (
            (self.initial_ffp_scale, "initial_ffp_scale"),
            (self.initial_pressure_scale, "initial_pressure_scale"),
        ):
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite when provided")

        self.pressure_profile = self._copy_profile(
            pressure_profile, "pressure_profile"
        )
        self.initial_ffp_profile = self._copy_profile(ffp_profile, "ffp_profile")
        if self.initial_ffp_profile["type"] != "linterp":
            raise ValueError("ffp_profile must use type='linterp' for knot fitting")

        self.ffp_x = self.initial_ffp_profile["x"].copy()
        initial_ffp_y = self.initial_ffp_profile["y"].copy()
        if initial_ffp_y[0] == 0.0:
            raise ValueError("the first ffp_profile value cannot be zero")
        # TokaMaker fixes the source amplitude through global targets.  The
        # first knot therefore removes a scale degeneracy from the inverse fit.
        self.ffp_y = initial_ffp_y / initial_ffp_y[0]

        self.q_coordinate = q_coordinate.lower()
        if self.q_coordinate not in {"s", "psi"}:
            raise ValueError("q_coordinate must be 's' or 'psi'")
        self.q_x, self.q_y = self._copy_xy(q_profile, "q_profile")
        if self.q_x[0] > 0.0 or self.q_x[-1] < 1.0:
            raise ValueError("q_profile must span the full normalized interval [0, 1]")

        self._validate_fit_config()
        cfg = self.fit_config
        self.ffp_bounds = self._covering_ffp_bounds()
        self.fit_locations = np.linspace(*cfg.fit_range, cfg.fit_points)
        self.target_q = np.interp(self.fit_locations, self.q_x, self.q_y)
        self.q_sample_psi = np.linspace(*cfg.q_sample_range, cfg.q_sample_points)

        self.coarse_solver: TokaMaker | None = None
        self.fine_solver: TokaMaker | None = None
        self.initial_equilibrium: Any | None = None
        self.coarse_equilibrium: Any | None = None
        self.fine_equilibrium: Any | None = None
        self.initial_samples: EquilibriumSamples | None = None
        self.coarse_samples: EquilibriumSamples | None = None
        self.fine_samples: EquilibriumSamples | None = None
        self.fit_result: QFitResult | None = None
        self.initialization_used_external_psi: bool | None = None
        self.refinement_used_interpolated_psi: bool | None = None
        self.refinement_used_external_psi: bool | None = None
        self.refinement_used_cold_start: bool | None = None
        self.refinement_warm_start_error: str | None = None
        self.refinement_stage_initializers: tuple[str, ...] = ()
        self.refinement_stage_iterations: tuple[int, ...] = ()

        self._Ip_fit = self.Ip_seed

    @staticmethod
    def _copy_xy(profile: Mapping[str, Any], name: str) -> tuple[FloatArray, FloatArray]:
        if "x" not in profile or "y" not in profile:
            raise KeyError(f"{name} must contain 'x' and 'y'")
        x = np.asarray(profile["x"], dtype=float).copy()
        y = np.asarray(profile["y"], dtype=float).copy()
        if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 2:
            raise ValueError(f"{name} x and y must be equal-length one-dimensional arrays")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError(f"{name} contains non-finite values")
        if np.any(np.diff(x) <= 0.0):
            raise ValueError(f"{name} x values must be strictly increasing")
        return x, y

    @classmethod
    def _copy_profile(cls, profile: Mapping[str, Any], name: str) -> dict[str, Any]:
        x, y = cls._copy_xy(profile, name)
        profile_type = str(profile.get("type", "linterp"))
        if profile_type != "linterp":
            raise ValueError(f"{name} currently supports only type='linterp'")
        if x[0] < 0.0 or x[-1] > 1.0:
            raise ValueError(f"{name} x values must lie in [0, 1]")
        if x[0] != 0.0 or x[-1] != 1.0:
            raise ValueError(f"{name} must include both normalized-flux endpoints")
        return {"type": profile_type, "x": x, "y": y}

    def _validate_fit_config(self) -> None:
        cfg = self.fit_config
        if not (0.0 < cfg.fit_range[0] < cfg.fit_range[1] < 1.0):
            raise ValueError("fit_range must lie strictly inside (0, 1)")
        if not (0.0 < cfg.q_sample_range[0] < cfg.q_sample_range[1] < 1.0):
            raise ValueError("q_sample_range must lie strictly inside (0, 1)")
        if cfg.fit_points < 2 or cfg.q_sample_points < 3:
            raise ValueError("fit_points and q_sample_points are too small")
        if cfg.max_iterations < 1 or cfg.target_rms_error <= 0.0:
            raise ValueError("max_iterations and target_rms_error must be positive")
        if cfg.profile_fd_step_abs <= 0.0 or cfg.log_current_fd_step <= 0.0:
            raise ValueError("finite-difference steps must be positive")
        if not cfg.damping_factors or any(value <= 0.0 for value in cfg.damping_factors):
            raise ValueError("damping_factors must contain positive values")
        lower, upper = cfg.ffp_bounds
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError("ffp_bounds must be ordered")

    def _covering_ffp_bounds(self) -> tuple[float | None, float | None]:
        """``ffp_bounds`` widened, where needed, to contain the source shape.

        The configured bounds suit an F F' that keeps its axis sign and stays
        within a few times its axis value.  A source whose F F' changes sign
        (DIII-D) or peaks well off axis has knots outside them, and such a
        knot could neither be perturbed nor kept by the fit.  A violated bound
        therefore moves past the source's extreme knot by a quarter of the
        shape's spread, and never by less than two finite-difference steps.
        """
        cfg = self.fit_config
        lower, upper = cfg.ffp_bounds
        margin = max(0.25 * float(np.ptp(self.ffp_y)), 2.0 * cfg.profile_fd_step_abs)
        if lower is not None and self.ffp_y.min() < lower:
            lower = float(self.ffp_y.min()) - margin
        if upper is not None and self.ffp_y.max() > upper:
            upper = float(self.ffp_y.max()) + margin
        if (lower, upper) != tuple(cfg.ffp_bounds) and cfg.verbose:
            shown = ", ".join("None" if v is None else f"{v:.3g}" for v in (lower, upper))
            print(f"The source F F' shape spans [{self.ffp_y.min():.3g}, "
                  f"{self.ffp_y.max():.3g}] of its axis value; fitting its knots "
                  f"within [{shown}] instead of {cfg.ffp_bounds}.")
        return lower, upper

    @property
    def solver(self) -> TokaMaker:
        """Return the finest completed solver, or the coarse solver."""
        if self.fine_solver is not None:
            return self.fine_solver
        if self.coarse_solver is not None:
            return self.coarse_solver
        raise RuntimeError("the initial equilibrium has not been solved")

    @property
    def fitted_ffp_profile(self) -> dict[str, Any]:
        """Return the current normalized fitted :math:`F F'` shape."""
        return {
            "type": "linterp",
            "x": self.ffp_x.copy(),
            "y": self.ffp_y.copy(),
        }

    def _build_solver(self, mesh_dx: float) -> TokaMaker:
        mesh = gs_Domain()
        mesh.define_region("plasma", mesh_dx, "plasma")
        mesh.add_polygon(self.lcfs, "plasma")
        mesh_points, mesh_cells, mesh_regions = mesh.build_mesh()

        if self._oft_environment is None:
            # OFT permits only one runtime per Python kernel.  Reusing its
            # registered instance also makes repeated VSCode cells practical.
            self._oft_environment = getattr(OFT_env, "instance", None)
            if self._oft_environment is None:
                self._oft_environment = OFT_env(nthreads=self.nthreads)
        solver = TokaMaker(self._oft_environment)
        solver.setup_mesh(mesh_points, mesh_cells, reg=mesh_regions)
        solver.settings.free_boundary = False
        solver.settings.mode = self.source_mode
        solver.settings.nl_tol = self.nonlinear_tolerance
        solver.settings.maxits = self.nonlinear_max_iterations
        solver.settings.pm = self.solver_output
        solver.setup(order=self.fe_order, F0=self.F0)
        return solver

    def _solve(self, solver: TokaMaker) -> tuple[Any, int]:
        """``solver.solve``, loosening ``nl_tol`` by decades while it stalls.

        A source F F' that changes sign (DIII-D) gives the Ip target only a
        weak hold on its scale, and the Picard iteration can settle into a
        two-state cycle a few 1e-6 above a tight tolerance.  On "Exceeded
        maxits" the solve resumes from its stalled iterate at ten times the
        tolerance, up to ``max_nonlinear_tolerance``.  The loosened tolerance
        holds for every later solve and mesh, so each decade costs one
        ``maxits`` run at most.
        """
        while True:
            try:
                equilibrium, iterations = solver.solve(return_its=True)
                return equilibrium, int(iterations)
            except ValueError as error:
                loosened = 10.0 * self.nonlinear_tolerance
                ceiling = self.max_nonlinear_tolerance * (1.0 + 1.0e-9)
                if "maxits" not in str(error) or loosened > ceiling:
                    raise
                print(
                    "The nonlinear solve stalled above "
                    f"nl_tol={self.nonlinear_tolerance:.0e}; continuing at {loosened:.0e}.",
                    flush=True,
                )
                self.nonlinear_tolerance = loosened
                solver.settings.nl_tol = loosened
                solver.update_settings()

    def _set_inputs(
        self,
        solver: TokaMaker,
        ffp_y: ArrayLike,
        plasma_current: float,
    ) -> None:
        ffp_y_array = np.asarray(ffp_y, dtype=float)
        if ffp_y_array.shape != self.ffp_x.shape:
            raise ValueError("ffp_y has the wrong shape")
        solver.set_profiles(
            pp_prof=self.pressure_profile,
            ffp_prof={
                "type": "linterp",
                "x": self.ffp_x,
                "y": ffp_y_array,
            },
        )
        solver.set_targets(Ip=float(plasma_current), pax=self.p_axis_target)

    def _evaluate_external_psi(self, solver: TokaMaker) -> FloatArray:
        """Evaluate and validate the configured absolute-psi seed on a mesh."""
        if self.initial_psi_evaluator is None:
            raise RuntimeError("no external psi evaluator was configured")
        node_locations = np.asarray(solver.r[:, :2], dtype=float)
        psi_initial = np.asarray(
            self.initial_psi_evaluator(node_locations),
            dtype=float,
        )
        if psi_initial.shape != (solver.np,):
            raise ValueError(
                "initial_psi_evaluator must return one value per TokaMaker "
                f"node; expected {(solver.np,)}, got {psi_initial.shape}"
            )
        if not np.all(np.isfinite(psi_initial)):
            raise ValueError("initial_psi_evaluator returned non-finite values")
        if np.ptp(psi_initial) <= np.finfo(float).eps:
            raise ValueError("initial_psi_evaluator returned a constant flux")
        return np.ascontiguousarray(psi_initial)

    def solve_initial(self) -> EquilibriumSamples:
        """Solve the exact supplied profiles on the coarse fixed-LCFS mesh."""
        if self.initial_samples is not None:
            return self.initial_samples

        solver = self._build_solver(self.coarse_dx)
        self.coarse_solver = solver
        solver.set_profiles(
            pp_prof=self.pressure_profile,
            ffp_prof=self.initial_ffp_profile,
        )
        solver.set_targets(Ip=self.Ip_seed, pax=self.p_axis_target)
        if self.initial_psi_evaluator is None:
            solver.init_psi()
            self.initialization_used_external_psi = False
        else:
            psi_initial = self._evaluate_external_psi(solver)
            if self.initial_ffp_scale is not None:
                solver.ffp_scale = self.initial_ffp_scale
            if self.initial_pressure_scale is not None:
                solver.p_scale = self.initial_pressure_scale
            solver.set_psi(psi_initial, update_bounds=True)
            self.initialization_used_external_psi = True
        equilibrium, iterations = self._solve(solver)

        self.initial_equilibrium = equilibrium
        self.coarse_equilibrium = equilibrium
        self.initial_samples = self.sample_equilibrium(
            solver,
            label="initial coarse",
            mesh_dx=self.coarse_dx,
            nonlinear_iterations=int(iterations),
        )
        return self.initial_samples

    def _q_at_fit_locations(self, solver: TokaMaker) -> FloatArray:
        if self.q_coordinate == "psi":
            _, q_values, *_ = solver.get_q(psi=self.fit_locations)
            return np.asarray(q_values, dtype=float)

        _, q_dense, *_ = solver.get_q(psi=self.q_sample_psi)
        q_dense = np.asarray(q_dense, dtype=float)
        s_dense = normalized_toroidal_flux(self.q_sample_psi, q_dense)
        s_augmented = np.r_[0.0, s_dense, 1.0]
        q_augmented = np.r_[q_dense[0], q_dense, q_dense[-1]]
        return np.interp(self.fit_locations, s_augmented, q_augmented)

    def q_at_fit_locations(
        self,
        solver: TokaMaker | None = None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return fit coordinates, realized q, and target q."""
        active_solver = solver or self.solver
        return (
            self.fit_locations.copy(),
            self._q_at_fit_locations(active_solver),
            self.target_q.copy(),
        )

    def _solve_trial(
        self,
        *,
        ffp_y: FloatArray,
        plasma_current: float,
        starting_equilibrium: Any,
    ) -> tuple[FloatArray, Any, int]:
        if self.coarse_solver is None:
            raise RuntimeError("the coarse solver has not been constructed")
        solver = self.coarse_solver
        solver.replace_eq(source_eq=starting_equilibrium)
        self._set_inputs(solver, ffp_y, plasma_current)
        equilibrium, iterations = self._solve(solver)
        return self._q_at_fit_locations(solver), equilibrium, int(iterations)

    def _profile_fd_delta(self, knot: int) -> float:
        cfg = self.fit_config
        delta = cfg.profile_fd_step_abs
        lower, upper = self.ffp_bounds
        if upper is not None and self.ffp_y[knot] + delta > upper:
            delta = -delta
        if lower is not None and self.ffp_y[knot] + delta < lower:
            raise ValueError(f"cannot perturb FF' knot {knot} within ffp_bounds")
        return delta

    @staticmethod
    def _difference_operator(
        n_parameters: int,
        n_shape_parameters: int,
    ) -> FloatArray:
        """Penalize oscillatory differences between adjacent knot corrections."""
        rows = max(n_shape_parameters - 1, 0)
        operator = np.zeros((rows, n_parameters), dtype=float)
        for row in range(rows):
            operator[row, row] = 1.0
            operator[row, row + 1] = -1.0
        return operator

    @staticmethod
    def _errors(residual: FloatArray) -> tuple[float, float]:
        return float(np.sqrt(np.mean(residual**2))), float(np.max(np.abs(residual)))

    def fit_coarse(self) -> QFitResult:
        """Fit q on the coarse mesh with state-isolated damped Newton trials."""
        if self.fit_result is not None:
            return self.fit_result
        self.solve_initial()
        assert self.coarse_solver is not None
        assert self.initial_equilibrium is not None

        cfg = self.fit_config
        solver = self.coarse_solver
        history: list[QFitIteration] = []
        current = float(self.Ip_seed)

        # Activate the normalized parameterization without changing the supplied
        # FF' shape.  This removes only its irrelevant global scale.
        q_base, accepted_equilibrium, baseline_iterations = self._solve_trial(
            ffp_y=self.ffp_y,
            plasma_current=current,
            starting_equilibrium=self.initial_equilibrium,
        )
        accepted_iterations = baseline_iterations
        initial_residual = q_base - self.target_q
        initial_rms, initial_max = self._errors(initial_residual)
        convergence_limit = cfg.target_rms_error * (1.0 + 1.0e-3)
        converged = initial_rms <= convergence_limit
        reason = "target RMS reached" if converged else "maximum iterations reached"

        if cfg.verbose:
            print(
                "initial q fit: "
                f"RMS(dq)={initial_rms:.5e}, max|dq|={initial_max:.5e}, "
                f"Ip={current:.3f} A"
            )

        for iteration in range(cfg.max_iterations):
            residual = q_base - self.target_q
            rms_before, max_before = self._errors(residual)
            if rms_before <= convergence_limit:
                converged = True
                reason = "target RMS reached"
                break

            fit_knots = np.arange(
                1,
                self.ffp_y.size,
                dtype=int,
            )
            n_shape_parameters = fit_knots.size
            n_parameters = n_shape_parameters + int(cfg.fit_plasma_current)
            jacobian = np.empty((self.fit_locations.size, n_parameters), dtype=float)
            nonlinear_solves = 0
            failed_trials = 0

            for column, knot in enumerate(fit_knots):
                delta = self._profile_fd_delta(knot)
                perturbed = self.ffp_y.copy()
                perturbed[knot] += delta
                try:
                    q_perturbed, _, trial_iterations = self._solve_trial(
                        ffp_y=perturbed,
                        plasma_current=current,
                        starting_equilibrium=accepted_equilibrium,
                    )
                except Exception as error:
                    failed_trials += 1
                    delta = -delta
                    perturbed = self.ffp_y.copy()
                    perturbed[knot] += delta
                    try:
                        q_perturbed, _, trial_iterations = self._solve_trial(
                            ffp_y=perturbed,
                            plasma_current=current,
                            starting_equilibrium=accepted_equilibrium,
                        )
                    except Exception as fallback_error:
                        solver.replace_eq(source_eq=accepted_equilibrium)
                        raise RuntimeError(
                            f"both finite-difference trials failed for FF' knot {knot}"
                        ) from fallback_error
                    if cfg.verbose:
                        print(f"  forward FF' perturbation failed at knot {knot}: {error}")
                nonlinear_solves += 1
                jacobian[:, column] = (q_perturbed - q_base) / delta

            if cfg.fit_plasma_current:
                try:
                    q_current, _, trial_iterations = self._solve_trial(
                        ffp_y=self.ffp_y,
                        plasma_current=current * np.exp(cfg.log_current_fd_step),
                        starting_equilibrium=accepted_equilibrium,
                    )
                except Exception as error:
                    failed_trials += 1
                    try:
                        q_current, _, trial_iterations = self._solve_trial(
                            ffp_y=self.ffp_y,
                            plasma_current=current * np.exp(-cfg.log_current_fd_step),
                            starting_equilibrium=accepted_equilibrium,
                        )
                    except Exception as fallback_error:
                        solver.replace_eq(source_eq=accepted_equilibrium)
                        raise RuntimeError(
                            "both plasma-current finite differences failed"
                        ) from fallback_error
                    current_delta = -cfg.log_current_fd_step
                    if cfg.verbose:
                        print(f"  forward plasma-current perturbation failed: {error}")
                else:
                    current_delta = cfg.log_current_fd_step
                nonlinear_solves += 1
                jacobian[:, -1] = (q_current - q_base) / current_delta

            regularization = max(
                cfg.regularization_floor,
                cfg.regularization_initial * cfg.regularization_decay**iteration,
            )
            difference_operator = self._difference_operator(
                n_parameters,
                n_shape_parameters,
            )
            augmented_matrix = np.vstack(
                (
                    jacobian,
                    np.sqrt(regularization) * difference_operator,
                    np.sqrt(cfg.ridge) * np.eye(n_parameters),
                )
            )
            augmented_rhs = np.r_[
                -residual,
                np.zeros(difference_operator.shape[0]),
                np.zeros(n_parameters),
            ]
            step, *_ = np.linalg.lstsq(augmented_matrix, augmented_rhs, rcond=None)
            shape_step = np.clip(
                step[:n_shape_parameters],
                -cfg.max_profile_step,
                cfg.max_profile_step,
            )
            log_current_step = 0.0
            if cfg.fit_plasma_current:
                log_current_step = float(
                    np.clip(step[-1], -cfg.max_log_current_step, cfg.max_log_current_step)
                )

            accepted_damping: float | None = None
            q_after = q_base
            rms_after = rms_before
            max_after = max_before
            current_after = current
            equilibrium_after = accepted_equilibrium
            iterations_after = accepted_iterations

            lower, upper = self.ffp_bounds
            lower_bound = -np.inf if lower is None else lower
            upper_bound = np.inf if upper is None else upper
            for damping in cfg.damping_factors:
                candidate_shape = self.ffp_y.copy()
                candidate_shape[fit_knots] += damping * shape_step
                candidate_shape[fit_knots] = np.clip(
                    candidate_shape[fit_knots], lower_bound, upper_bound
                )
                candidate_current = current * np.exp(damping * log_current_step)
                try:
                    q_candidate, candidate_equilibrium, trial_iterations = self._solve_trial(
                        ffp_y=candidate_shape,
                        plasma_current=candidate_current,
                        starting_equilibrium=accepted_equilibrium,
                    )
                except Exception as error:
                    failed_trials += 1
                    if cfg.verbose:
                        print(f"  damping={damping:g} trial failed: {error}")
                    continue
                nonlinear_solves += 1
                candidate_rms, candidate_max = self._errors(q_candidate - self.target_q)
                if candidate_rms < rms_before:
                    accepted_damping = float(damping)
                    self.ffp_y = candidate_shape
                    current_after = float(candidate_current)
                    q_after = q_candidate
                    rms_after = candidate_rms
                    max_after = candidate_max
                    equilibrium_after = candidate_equilibrium
                    iterations_after = trial_iterations
                    break

            history.append(
                QFitIteration(
                    iteration=iteration,
                    rms_before=rms_before,
                    max_before=max_before,
                    plasma_current_before=current,
                    regularization=regularization,
                    accepted_damping=accepted_damping,
                    rms_after=rms_after,
                    max_after=max_after,
                    plasma_current_after=current_after,
                    nonlinear_solves=nonlinear_solves,
                    failed_trials=failed_trials,
                )
            )
            if cfg.verbose:
                damping_text = "none" if accepted_damping is None else f"{accepted_damping:g}"
                print(
                    f"iteration {iteration:2d}: RMS(dq) {rms_before:.5e} -> "
                    f"{rms_after:.5e}, max|dq|={max_after:.5e}, "
                    f"damping={damping_text}, Ip={current_after:.3f} A"
                )

            if accepted_damping is None:
                reason = "no improving damped Newton step"
                break

            current = current_after
            q_base = q_after
            accepted_equilibrium = equilibrium_after
            accepted_iterations = iterations_after
            if rms_after <= convergence_limit:
                converged = True
                reason = "target RMS reached"
                break
        else:
            reason = "maximum iterations reached"

        solver.replace_eq(source_eq=accepted_equilibrium)
        final_q = self._q_at_fit_locations(solver)
        final_rms, final_max = self._errors(final_q - self.target_q)
        converged = final_rms <= convergence_limit
        if converged:
            reason = "target RMS reached"

        self._Ip_fit = current
        self.coarse_equilibrium = accepted_equilibrium
        self.coarse_samples = self.sample_equilibrium(
            solver,
            label="fitted coarse",
            mesh_dx=self.coarse_dx,
            nonlinear_iterations=accepted_iterations,
        )
        self.fit_result = QFitResult(
            converged=converged,
            reason=reason,
            target_coordinate=self.q_coordinate,
            target_locations=self.fit_locations.copy(),
            target_q=self.target_q.copy(),
            initial_rms_error=initial_rms,
            initial_max_error=initial_max,
            final_rms_error=final_rms,
            final_max_error=final_max,
            plasma_current=current,
            ffp_x=self.ffp_x.copy(),
            ffp_y=self.ffp_y.copy(),
            history=tuple(history),
        )
        return self.fit_result

    def _interpolate_absolute_psi(self, coarse: TokaMaker, fine: TokaMaker) -> FloatArray:
        interpolator = coarse.get_field_eval("psi")
        fine_points = np.asarray(fine.r[:, :2])
        inside_coarse_mesh = self._points_inside_polygon(
            fine_points,
            np.asarray(coarse.lim_contour),
        )
        psi_fine = np.full(
            fine_points.shape[0],
            float(coarse.psi_bounds[0]),
            dtype=float,
        )
        # The independently triangulated fine LCFS lies slightly outside the
        # coarse boundary chords.  OFT's interpolator has no outside-domain
        # status, so evaluate only points actually contained by the coarse
        # limiter and impose the fixed-boundary flux in the thin outer strip.
        for index in np.flatnonzero(inside_coarse_mesh):
            point = fine_points[index]
            psi_fine[index] = interpolator.eval(point)[0]
        if not np.all(np.isfinite(psi_fine)):
            raise RuntimeError("coarse-to-fine interpolation produced non-finite psi")

        coarse_psi = coarse.get_psi(normalized=False)
        coarse_bounds = np.asarray(coarse.psi_bounds, dtype=float)
        lower = min(float(np.min(coarse_psi)), float(np.min(coarse_bounds)))
        upper = max(float(np.max(coarse_psi)), float(np.max(coarse_bounds)))
        scale = max(upper - lower, abs(lower), abs(upper), np.finfo(float).tiny)
        tolerance = 2.0e-3 * scale
        if (
            psi_fine.min() < lower - tolerance
            or psi_fine.max() > upper + tolerance
        ):
            raise RuntimeError(
                "interpolated psi lies outside the coarse equilibrium range: "
                f"fine=[{psi_fine.min():.8e}, {psi_fine.max():.8e}], "
                f"coarse=[{lower:.8e}, {upper:.8e}]"
            )
        # High-order interpolation can overshoot a nodal extremum by roundoff
        # near the magnetic axis or X-point.  Keep the prolonged field within
        # the physical bounds after the explicit guard above.
        np.clip(psi_fine, lower, upper, out=psi_fine)
        return psi_fine

    @staticmethod
    def _points_inside_polygon(points: FloatArray, polygon: FloatArray) -> NDArray[np.bool_]:
        """Vectorized ray test used to guard non-nested mesh interpolation."""
        if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
            raise ValueError("invalid limiter polygon")
        x = points[:, 0]
        y = points[:, 1]
        inside = np.zeros(points.shape[0], dtype=bool)
        for index in range(polygon.shape[0]):
            previous = (index - 1) % polygon.shape[0]
            xi, yi = polygon[index]
            xj, yj = polygon[previous]
            crosses = (yi > y) != (yj > y)
            if yj == yi:
                continue
            crossing_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            inside ^= crosses & (x < crossing_x)
        return inside

    def _solve_refinement_stage(
        self,
        source: TokaMaker,
        mesh_dx: float,
        warm_rms_limit: float,
    ) -> tuple[TokaMaker, Any, int, str, str | None]:
        """Solve one remeshing stage and report how its psi was initialized."""
        stage = self._build_solver(mesh_dx)
        self._set_inputs(stage, self.ffp_y, self._Ip_fit)
        stage_baseline = stage.copy_eq()

        def check_q(label: str) -> None:
            q_values = self._q_at_fit_locations(stage)
            q_rms, _ = self._errors(q_values - self.target_q)
            if q_rms > warm_rms_limit:
                raise RuntimeError(
                    f"{label} q profile failed the refinement guard: "
                    f"RMS={q_rms:.3e} > {warm_rms_limit:.3e}"
                )

        try:
            psi_stage = self._interpolate_absolute_psi(source, stage)
            stage.ffp_scale = source.ffp_scale
            stage.p_scale = source.p_scale
            stage.set_psi(psi_stage, update_bounds=True)
            equilibrium, iterations = self._solve(stage)
            check_q("mesh-prolongated")
            return stage, equilibrium, int(iterations), "interpolated", None
        except Exception as interpolation_or_solve_error:
            first_error = str(interpolation_or_solve_error)
            if self.fit_config.verbose:
                print(
                    f"mesh continuation to dx={mesh_dx:g} failed: "
                    f"{interpolation_or_solve_error}"
                )

        try:
            stage.replace_eq(source_eq=stage_baseline)
            self._set_inputs(stage, self.ffp_y, self._Ip_fit)
            psi_external = self._evaluate_external_psi(stage)
            stage.ffp_scale = source.ffp_scale
            stage.p_scale = source.p_scale
            stage.set_psi(psi_external, update_bounds=True)
            if self.fit_config.verbose:
                print(
                    f"retrying dx={mesh_dx:g} from the external absolute-psi seed"
                )
            equilibrium, iterations = self._solve(stage)
            check_q("externally warm-started")
            return stage, equilibrium, int(iterations), "external", first_error
        except Exception as external_warm_start_error:
            second_error = str(external_warm_start_error)
            if self.fit_config.verbose:
                print(
                    f"external warm start at dx={mesh_dx:g} failed; retrying "
                    f"from init_psi(): {external_warm_start_error}"
                )

        stage.replace_eq(source_eq=stage_baseline)
        self._set_inputs(stage, self.ffp_y, self._Ip_fit)
        stage.init_psi()
        equilibrium, iterations = self._solve(stage)
        combined_error = (
            f"mesh continuation: {first_error}; "
            f"external seed: {second_error}"
        )
        return stage, equilibrium, int(iterations), "cold", combined_error

    def refine(self) -> EquilibriumSamples:
        """Continue the accepted equilibrium through the configured fine meshes."""
        if self.fine_samples is not None:
            return self.fine_samples
        self.fit_coarse()
        assert self.coarse_solver is not None
        assert self.coarse_equilibrium is not None

        source = self.coarse_solver
        source.replace_eq(source_eq=self.coarse_equilibrium)
        coarse_rms = self.fit_result.final_rms_error if self.fit_result else 0.0
        warm_rms_limit = max(5.0e-3, 20.0 * coarse_rms)
        stage_modes: list[str] = []
        stage_iterations: list[int] = []
        stage_errors: list[str] = []
        equilibrium: Any = self.coarse_equilibrium

        for mesh_dx in self.refinement_dx_sequence:
            source, equilibrium, iterations, mode, error = (
                self._solve_refinement_stage(
                    source,
                    mesh_dx,
                    warm_rms_limit,
                )
            )
            stage_modes.append(mode)
            stage_iterations.append(iterations)
            if error is not None:
                stage_errors.append(f"dx={mesh_dx:g}: {error}")
            if self.fit_config.verbose:
                print(
                    f"refinement dx={mesh_dx:g}: initializer={mode}, "
                    f"nonlinear iterations={iterations}"
                )

        self.fine_solver = source
        self.fine_equilibrium = equilibrium
        self.refinement_stage_initializers = tuple(stage_modes)
        self.refinement_stage_iterations = tuple(stage_iterations)
        self.refinement_used_interpolated_psi = all(
            mode == "interpolated" for mode in stage_modes
        )
        self.refinement_used_external_psi = any(
            mode == "external" for mode in stage_modes
        )
        self.refinement_used_cold_start = any(mode == "cold" for mode in stage_modes)
        self.refinement_warm_start_error = (
            " | ".join(stage_errors) if stage_errors else None
        )
        self.fine_samples = self.sample_equilibrium(
            source,
            label="fitted fine",
            mesh_dx=self.fine_dx,
            nonlinear_iterations=stage_iterations[-1],
        )
        return self.fine_samples

    def run(self) -> QFitResult:
        """Run the initial solve, coarse q fit, and fine-mesh refinement."""
        result = self.fit_coarse()
        self.refine()
        return result

    def sample_equilibrium(
        self,
        solver: TokaMaker,
        *,
        label: str,
        mesh_dx: float,
        nonlinear_iterations: int,
        psi: ArrayLike | None = None,
    ) -> EquilibriumSamples:
        """Sample q, source profiles, and global values for plotting/tests."""
        psi_array = (
            np.linspace(0.01, 0.99, 250)
            if psi is None
            else np.asarray(psi, dtype=float)
        )
        if psi_array.ndim != 1 or np.any(np.diff(psi_array) <= 0.0):
            raise ValueError("diagnostic psi must be a strictly increasing 1D array")

        psi_q, q, *_ = solver.get_q(psi=psi_array)
        psi_profiles, F, Fprime, pressure, Pprime = solver.get_profiles(psi=psi_array)
        psi_q = np.asarray(psi_q, dtype=float)
        psi_profiles = np.asarray(psi_profiles, dtype=float)
        if not np.allclose(psi_q, psi_profiles, rtol=0.0, atol=1.0e-12):
            raise RuntimeError("get_q and get_profiles returned inconsistent psi grids")

        q = np.asarray(q, dtype=float)
        F = np.asarray(F, dtype=float)
        Fprime = np.asarray(Fprime, dtype=float)
        pressure = np.asarray(pressure, dtype=float)
        Pprime = np.asarray(Pprime, dtype=float)
        arrays = (psi_q, q, F, Fprime, pressure, Pprime)
        if not all(np.all(np.isfinite(values)) for values in arrays):
            raise RuntimeError(f"{label} diagnostics contain non-finite values")

        globals_out = solver.get_globals()
        plasma_current = float(globals_out[0])
        return EquilibriumSamples(
            label=label,
            mesh_dx=float(mesh_dx),
            mesh_points=int(solver.np),
            mesh_cells=int(solver.nc),
            nonlinear_iterations=int(nonlinear_iterations),
            psi=psi_q.copy(),
            s=normalized_toroidal_flux(psi_q, q),
            q=q.copy(),
            F=F.copy(),
            Fprime=Fprime.copy(),
            FFprime=(F * Fprime).copy(),
            pressure=pressure.copy(),
            Pprime=Pprime.copy(),
            magnetic_axis=np.asarray(solver.o_point, dtype=float).copy(),
            plasma_current=plasma_current,
            ffp_scale=float(solver.ffp_scale),
            pressure_scale=float(solver.p_scale),
        )

    def save_eqdsk(self, filename: str, **kwargs: Any) -> None:
        """Save the finest completed fitted equilibrium as a gEQDSK file."""
        if self.fit_result is None:
            raise RuntimeError("fit_coarse() must complete before saving")
        self.solver.save_eqdsk(filename, **kwargs)


__all__ = [
    "EquilibriumSamples",
    "FixedBoundaryQFitter",
    "QFitConfig",
    "QFitIteration",
    "QFitResult",
    "cumulative_trapezoid",
    "normalized_toroidal_flux",
]
