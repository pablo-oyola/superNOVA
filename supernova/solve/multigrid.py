r"""Nested compatible-B-spline multigrid classification for solver5.

The orchestrator in this module builds the point-local ideal-MHD operator once
on a finest composite Gauss rule.  Three (or more) knot-insertion-nested radial
spaces then differ only in Galerkin projection, boundary reduction, and their
eigenvalue solves. Exact ``gamma_s == 0`` uses generalized EPS; all other cases
use the quadratic PEP.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import operator
from typing import Mapping, Sequence

import numpy as np
import xarray as xr
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment

from supernova.radial import BSplineFE
from supernova.eigen.petsc import NEPResult
from supernova.operators.blocks import mu_0
from supernova.core.solver3 import _build_local_M_split
from supernova.core.solver4 import MixedRadialData, _build_mixed_radial_data
from supernova.core.solver5 import (
    LinearGeneralizedEPSSolver,
    QuadraticPEPSolver,
    _build_solver5_eigensolver_from_matrices,
    superSolver5,
)
from supernova.solve.evidence import (
    CLASSIFICATION_VERSION,
    AdaptiveDiscoveryOptions,
    EVIDENCE_LABELS,
    EvidenceV2Config,
    MeshPerturbationOptions,
    PhysicalDisplacementMetric,
    ReacquisitionOptions,
    aggregate_catalog_status,
    classify_tracks_evidence_v2,
    cluster_aware_mode_assignment,
    deterministic_shifted_bspline,
    enriched_residual,
    empirical_candidate_stability,
    fixed_coarse_space_retention,
    mixed_space_prolongation,
    physical_displacement_paired_overlaps,
    physical_principal_angle_cosines,
    radial_power_diagnostics,
    select_reacquisition_candidates,
    weighted_principal_angle_cosines,
    window_catalog_status,
)


_LABELS = (
    "physical_gap",
    "continuum_resonance",
    "numerical_artifact",
    "inconclusive",
)


@dataclass(frozen=True)
class MultigridSearchOptions:
    """Fixed-window TOAR/SINVERT frequency search controls."""

    nev: int = 8
    tol: float = 1.0e-8
    maxit: int = 200
    window_count: int = 40
    window_padding: float = 0.10
    dedup_relative_s: float = 1.0e-6
    dedup_overlap: float = 0.99
    region_vscale: float = 0.25
    root_filter_tolerance: float = 1.0e-10
    pep_type: str = "TOAR"
    st_type: str = "SINVERT"
    which: str = "TARGET_MAGNITUDE"
    prefix: str = "nova_multigrid_pep"
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.nev) < 1:
            raise ValueError("nev must be positive.")
        if not np.isfinite(self.tol) or self.tol <= 0.0:
            raise ValueError("tol must be finite and positive.")
        if int(self.maxit) < 1:
            raise ValueError("maxit must be positive.")
        try:
            window_count = int(self.window_count)
        except (TypeError, ValueError, OverflowError):
            window_count = 0
        if (
            isinstance(self.window_count, bool)
            or window_count != self.window_count
            or window_count < 1
        ):
            raise ValueError("window_count must be a positive integer.")
        if not np.isfinite(self.window_padding) or self.window_padding < 0.0:
            raise ValueError("window_padding must be finite and non-negative.")
        for name in ("dedup_relative_s", "root_filter_tolerance"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not 0.0 <= self.dedup_overlap <= 1.0:
            raise ValueError("dedup_overlap must lie in [0, 1].")
        if not np.isfinite(self.region_vscale) or self.region_vscale <= 0.0:
            raise ValueError("region_vscale must be finite and positive.")


@dataclass(frozen=True)
class ModeClassificationThresholds:
    """Conservative numerical, convergence, and continuum evidence gates."""

    backward_error: float = 1.0e-8
    imaginary_frequency_ratio: float = 1.0e-6
    frequency_gate: float = 2.0e-2
    fine_frequency_change: float = 1.0e-3
    medium_fine_overlap: float = 0.98
    assignment_ambiguity: float = 1.0e-3
    continuum_clearance: float = 5.0e-3
    continuum_band: float = 5.0e-3
    continuum_power_fraction: float = 0.10
    width_contraction: float = 0.80
    ipr_growth: float = 1.25

    def __post_init__(self) -> None:
        positive = (
            "backward_error",
            "imaginary_frequency_ratio",
            "frequency_gate",
            "fine_frequency_change",
            "continuum_clearance",
            "continuum_band",
            "ipr_growth",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        unit_interval = (
            "medium_fine_overlap",
            "assignment_ambiguity",
            "continuum_power_fraction",
            "width_contraction",
        )
        for name in unit_interval:
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")


@dataclass(frozen=True)
class MultigridPEPResult:
    """All raw searches, tracks, continuum data, and classification evidence."""

    level_solutions: tuple[xr.Dataset, ...]
    mode_tracks: xr.Dataset
    window_diagnostics: xr.Dataset
    continuum: xr.Dataset
    classification: xr.Dataset

    @property
    def labels(self) -> xr.DataArray:
        return self.classification["label"]


@dataclass(frozen=True)
class _WindowRecord:
    level: int
    window: int
    parent: int
    depth: int
    core_omega_min: float
    core_omega_max: float
    solve_omega_min: float
    solve_omega_max: float
    target_s: complex
    nconv: int
    nreturned: int
    saturated: bool
    leaf: bool
    complete: bool
    reason: str
    iterations: int
    effective_nev: int = -1
    catalog_status: str = "partial"


def _fixed_frequency_windows(
    omega_range: tuple[float, float],
    window_count: int,
    padding: float,
) -> np.ndarray:
    """Return contiguous cores and clipped padded solve bounds.

    ``padding`` is the fraction of one core width added on each side.  The
    returned columns are ``core_min``, ``core_max``, ``solve_min``, and
    ``solve_max``.
    """
    omega_min, omega_max = _validate_omega_range(omega_range)
    try:
        count = int(window_count)
    except (TypeError, ValueError, OverflowError):
        count = 0
    if isinstance(window_count, bool) or count != window_count:
        raise ValueError("window_count must be a positive integer.")
    if count < 1:
        raise ValueError("window_count must be a positive integer.")
    if not np.isfinite(padding) or padding < 0.0:
        raise ValueError("padding must be finite and non-negative.")
    edges = np.linspace(omega_min, omega_max, count + 1)
    core_min = edges[:-1]
    core_max = edges[1:]
    width = core_max - core_min
    solve_min = np.maximum(omega_min, core_min - padding * width)
    solve_max = np.minimum(omega_max, core_max + padding * width)
    return np.column_stack((core_min, core_max, solve_min, solve_max))


def _core_ownership_mask(
    omega: np.ndarray,
    core_min: float,
    core_max: float,
    *,
    final_window: bool,
) -> np.ndarray:
    """Assign shared core edges uniquely to the window on their right."""
    real = np.asarray(omega, dtype=np.complex128).real
    upper = real <= core_max if final_window else real < core_max
    return np.isfinite(real) & (real >= core_min) & upper


def _coerce_options(value, cls):
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        return cls(**dict(value))
    raise TypeError(f"Expected {cls.__name__}, a mapping, or None.")


def _validate_omega_range(omega_range) -> tuple[float, float]:
    values = np.asarray(omega_range)
    if values.shape != (2,) or np.iscomplexobj(values):
        raise ValueError("omega_range must contain two real normalized frequencies.")
    low, high = (float(value) for value in values)
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or high <= low:
        raise ValueError(
            "omega_range must be finite, positive, and strictly increasing."
        )
    return low, high


def _simple_breakpoints(high: BSplineFE) -> np.ndarray:
    knots = np.asarray(high.knots, dtype=np.float64)
    breakpoints, counts = np.unique(knots, return_counts=True)
    if breakpoints.size < 2:
        raise ValueError("The base B-spline mesh must contain at least one span.")
    if not np.all(np.isfinite(breakpoints)) or not np.all(np.diff(breakpoints) > 0.0):
        raise ValueError("The base B-spline breakpoints must be finite and increasing.")
    if np.any(counts[1:-1] != 1):
        raise ValueError(
            "Nested midpoint refinement requires simple interior breakpoints."
        )
    if counts[0] != high.order or counts[-1] != high.order:
        raise ValueError(
            "The base B-spline must have endpoint multiplicity exactly equal "
            "to its order."
        )
    return breakpoints


def _midpoint_refine_breakpoints(breakpoints: np.ndarray) -> np.ndarray:
    breakpoints = np.asarray(breakpoints, dtype=np.float64)
    midpoints = 0.5 * (breakpoints[:-1] + breakpoints[1:])
    return np.sort(np.concatenate((breakpoints, midpoints)))


def _clamped_knots(breakpoints: np.ndarray, order: int) -> np.ndarray:
    return np.concatenate(
        (
            np.repeat(breakpoints[0], order),
            breakpoints[1:-1],
            np.repeat(breakpoints[-1], order),
        )
    )


def _nested_bspline_levels(
    base: BSplineFE,
    levels: int = 3,
) -> tuple[BSplineFE, ...]:
    """Return private knot-nested spaces with only the finest one assembled."""
    if not isinstance(base, BSplineFE):
        raise TypeError("base_radialfe must be a BSplineFE.")
    if isinstance(levels, bool) or int(levels) != levels or int(levels) < 1:
        raise ValueError("levels must be a positive integer.")
    levels = int(levels)
    breakpoints = _simple_breakpoints(base)
    output = []
    for level in range(levels):
        output.append(
            BSplineFE(
                knots=_clamped_knots(breakpoints, base.order),
                order=base.order,
                domain=(breakpoints[0], breakpoints[-1]),
                quad_order=base.quadrature_order,
                build_matrices=level == levels - 1,
            )
        )
        if level < levels - 1:
            breakpoints = _midpoint_refine_breakpoints(breakpoints)
    if not output[-1].build_matrices:
        raise ValueError("The finest B-spline level must provide native quadrature.")
    return tuple(output)


def build_resolved_kinetic_energy_metric(
    solver,
    points: np.ndarray,
    m_pol: np.ndarray,
    *,
    max_delta_m: int | None = None,
) -> PhysicalDisplacementMetric:
    r"""Build the resolved perpendicular kinetic-energy matching metric.

    The Cheng--Chance displacement components are orthogonal, giving the
    geometry factors ``sqrt(g)/g^{psi psi}`` for ``xi^psi`` and
    ``sqrt(g)*g^{psi psi}/B^2`` for ``xi^s``.  In Boozer coordinates these are
    ``h/(B^2*g^{psi psi})`` and the continuum profile ``J`` respectively.
    The solver's normalized mass density supplies the radial kinetic weight.
    """
    points = np.asarray(points, dtype=float)
    m_pol = np.asarray(m_pol)
    if points.ndim != 1 or points.size == 0 or np.any(np.diff(points) <= 0.0):
        raise ValueError("Metric points must be a strictly increasing array.")
    if m_pol.ndim != 1 or m_pol.size == 0 or np.unique(m_pol).size != m_pol.size:
        raise ValueError("Metric m_pol must contain unique mode numbers.")
    maximum_difference = int(np.max(np.abs(m_pol[:, None] - m_pol[None, :])))
    if max_delta_m is None:
        max_delta_m = maximum_difference
    max_delta_m = int(max_delta_m)
    if max_delta_m < 0:
        raise ValueError("max_delta_m must be non-negative.")
    max_delta_m = min(max_delta_m, maximum_difference)

    radial_profile = solver.equ("h/(B0**2*gpsi_psi)", normalize=True)
    radial_data = radial_profile.data.interp(x=points)
    tangential_data = solver.J.data.interp(x=points)

    def harmonics(data) -> np.ndarray:
        values = data.reindex(
            mpol=np.arange(max_delta_m + 1, dtype=np.int32),
            ntor=np.asarray([0], dtype=np.int32),
            fill_value=0.0 + 0.0j,
        ).transpose("x", "mpol", "ntor")
        return np.ascontiguousarray(values.values[:, :, 0], dtype=np.complex128)

    radial = harmonics(radial_data)
    tangential = harmonics(tangential_data)
    orientation_samples = np.concatenate(
        (
            radial[:, 0].real[np.isfinite(radial[:, 0].real)],
            tangential[:, 0].real[np.isfinite(tangential[:, 0].real)],
        )
    )
    orientation = float(np.sign(np.median(orientation_samples)))
    if orientation == 0.0 or not np.isfinite(orientation):
        raise ValueError("Could not determine the Boozer-Jacobian orientation.")
    radial *= orientation
    tangential *= orientation
    tolerance = 1.0e-10 * max(
        1.0,
        float(np.max(np.abs(radial[:, 0]))),
        float(np.max(np.abs(tangential[:, 0]))),
    )
    if np.any(np.abs(radial[:, 0].imag) > tolerance) or np.any(
        np.abs(tangential[:, 0].imag) > tolerance
    ):
        raise ValueError("Physical metric zero harmonics must be real.")
    if np.any(radial[:, 0].real <= 0.0) or np.any(tangential[:, 0].real <= 0.0):
        raise ValueError("Physical metric zero harmonics must be positive.")

    def assert_positive_metric(harmonic_values: np.ndarray, name: str) -> None:
        differences = m_pol[:, None] - m_pol[None, :]
        absolute = np.abs(differences)
        matrices = np.zeros(
            (points.size, m_pol.size, m_pol.size),
            dtype=np.complex128,
        )
        available = absolute <= max_delta_m
        for row, column in zip(*np.nonzero(available)):
            delta = int(differences[row, column])
            coefficient = harmonic_values[:, abs(delta)]
            if delta != 0:
                coefficient = 0.5 * (
                    coefficient if delta > 0 else coefficient.conjugate()
                )
            matrices[:, row, column] = coefficient
        eigenvalues = np.linalg.eigvalsh(matrices)
        scale = np.maximum(np.max(np.abs(eigenvalues), axis=1), 1.0)
        invalid = eigenvalues[:, 0] < -1.0e-10 * scale
        if np.any(invalid):
            raise ValueError(
                f"{name} physical metric is not positive semidefinite at "
                f"{np.count_nonzero(invalid)} radial quadrature points; retain all "
                "harmonic differences rather than applying the operator max_dm cutoff."
            )

    assert_positive_metric(radial, "radial")
    assert_positive_metric(tangential, "tangential")

    density_data = solver.density.data.interp(x=points).reindex(
        mpol=np.asarray([0], dtype=np.int32),
        ntor=np.asarray([0], dtype=np.int32),
        fill_value=np.nan,
    ).transpose("x", "mpol", "ntor")
    density_complex = np.asarray(density_data.values[:, 0, 0], dtype=np.complex128)
    density_tolerance = 1.0e-12 * np.maximum(1.0, np.abs(density_complex.real))
    if np.any(np.abs(density_complex.imag) > density_tolerance):
        raise ValueError("Mass density must be real for the kinetic-energy metric.")
    mass_amu = float(solver.cont.mass.to("amu").v)
    mass_density = density_complex.real * mass_amu
    return PhysicalDisplacementMetric(
        points=points,
        m_pol=m_pol,
        mass_density=mass_density,
        radial_geometry_harmonics=radial,
        tangential_geometry_harmonics=tangential,
    )


def _relative_complex_distance(left, right) -> np.ndarray:
    """Symmetric relative distance appropriate for small complex ``s`` roots."""
    left = np.asarray(left, dtype=np.complex128)
    right = np.asarray(right, dtype=np.complex128)
    scale = np.maximum(
        np.finfo(float).tiny,
        np.maximum(np.abs(left), np.abs(right)),
    )
    return np.abs(left - right) / scale


def _weighted_xi_overlap(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Phase-invariant harmonic-resolved radial ``xi^psi`` overlap."""
    left = np.asarray(left, dtype=np.complex128)
    right = np.asarray(right, dtype=np.complex128)
    weights = np.asarray(weights, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("xi fields must have equal shape (psi, m_pol).")
    if weights.shape != (left.shape[0],):
        raise ValueError("weights must match the xi radial dimension.")
    inner = np.sum(weights[:, None] * left.conjugate() * right)
    left_norm = np.sum(weights[:, None] * np.abs(left) ** 2)
    right_norm = np.sum(weights[:, None] * np.abs(right) ** 2)
    denominator = np.sqrt(max(0.0, float(left_norm.real * right_norm.real)))
    if not np.isfinite(denominator) or denominator <= np.finfo(float).tiny:
        return 0.0
    return float(np.clip(abs(inner) / denominator, 0.0, 1.0))


def _best_error(dataset: xr.Dataset) -> np.ndarray:
    size = dataset.sizes.get("eigenpair", 0)
    best = np.full(size, np.nan, dtype=np.float64)
    for name in ("slepc_backward_error", "backward_error", "residual_norm"):
        if name not in dataset:
            continue
        values = np.asarray(dataset[name].values, dtype=np.float64)
        use = ~np.isfinite(best) & np.isfinite(values)
        best[use] = values[use]
    return best


def _deduplicate_window_dataset(
    dataset: xr.Dataset,
    weights: np.ndarray,
    *,
    relative_s: float,
    minimum_overlap: float,
    physical_metric: PhysicalDisplacementMetric | None = None,
    tangential_fields: np.ndarray | None = None,
) -> xr.Dataset:
    """Deduplicate only cross-window roots meeting both spectral/shape gates."""
    size = dataset.sizes.get("eigenpair", 0)
    if size < 2:
        return dataset.assign_coords(eigenpair=np.arange(size, dtype=np.int64))
    roots = np.asarray(dataset["s"].values, dtype=np.complex128)
    windows = np.asarray(dataset["window_id"].values, dtype=np.int64)
    fields = np.asarray(dataset["xi_psi"].values, dtype=np.complex128)
    if tangential_fields is not None:
        tangential = np.asarray(tangential_fields, dtype=np.complex128)
        if tangential.shape != fields.shape:
            raise ValueError("tangential_fields must match the reconstructed xi_psi array.")
    else:
        tangential = (
            np.asarray(dataset["xi_s"].values, dtype=np.complex128)
            if physical_metric is not None and "xi_s" in dataset
            else None
        )
    if tangential is not None and physical_metric is None:
        raise ValueError("tangential_fields require a physical_metric.")
    use_physical = tangential is not None and physical_metric is not None
    quality = _best_error(dataset)
    core_owned = (
        np.asarray(dataset["core_owned"].values, dtype=bool)
        if "core_owned" in dataset
        else np.ones(size, dtype=bool)
    )
    spectral = _relative_complex_distance(roots[:, None], roots[None, :]) <= relative_s
    visited = np.zeros(size, dtype=bool)
    keep: list[int] = []
    for seed in range(size):
        if visited[seed]:
            continue
        pending = [seed]
        component: list[int] = []
        while pending:
            index = pending.pop()
            if visited[index]:
                continue
            visited[index] = True
            component.append(index)
            pending.extend(
                int(value)
                for value in np.flatnonzero(spectral[index] & ~visited)
            )
        component_array = np.asarray(component, dtype=int)
        groups = {
            int(window): component_array[windows[component_array] == window]
            for window in np.unique(windows[component_array])
        }
        if len(groups) == 1:
            keep.extend(component)
            continue

        def group_priority(indices: np.ndarray):
            return (
                -int(np.count_nonzero(core_owned[indices])),
                float(np.nanmin(np.nan_to_num(quality[indices], nan=np.inf))),
                int(np.min(indices)),
            )

        ordered_groups = sorted(groups.values(), key=group_priority)
        retained_groups: list[np.ndarray] = []
        for indices in ordered_groups:
            duplicate_basis = False
            for retained in retained_groups:
                if indices.size == retained.size:
                    if use_physical:
                        cosines = physical_principal_angle_cosines(
                            fields[retained],
                            tangential[retained],
                            fields[indices],
                            tangential[indices],
                            weights,
                            physical_metric,
                        )
                    else:
                        cosines = weighted_principal_angle_cosines(
                            fields[retained],
                            fields[indices],
                            weights,
                        )
                    if (
                        cosines.size == indices.size
                        and np.min(cosines) >= minimum_overlap
                    ):
                        duplicate_basis = True
                        break
            if not duplicate_basis:
                retained_indices = []
                for index in indices:
                    individual_duplicate = False
                    for retained in retained_groups:
                        for retained_index in retained:
                            if use_physical:
                                overlap = physical_displacement_paired_overlaps(
                                    fields[index : index + 1],
                                    tangential[index : index + 1],
                                    fields[retained_index : retained_index + 1],
                                    tangential[retained_index : retained_index + 1],
                                    weights,
                                    physical_metric,
                                )[0]
                            else:
                                overlap = _weighted_xi_overlap(
                                    fields[index],
                                    fields[retained_index],
                                    weights,
                                )
                            if overlap >= minimum_overlap:
                                individual_duplicate = True
                                break
                        if individual_duplicate:
                            break
                    if not individual_duplicate:
                        retained_indices.append(int(index))
                if retained_indices:
                    retained_array = np.asarray(retained_indices, dtype=int)
                    retained_groups.append(retained_array)
                    keep.extend(retained_indices)
    keep_array = np.asarray(keep, dtype=np.int64)
    if keep_array.size:
        omega = np.asarray(dataset["omega"].values)[keep_array]
        keep_array = keep_array[np.argsort(omega.real)]
    output = dataset.isel(eigenpair=keep_array).assign_coords(
        eigenpair=np.arange(keep_array.size, dtype=np.int64)
    )
    output.attrs.update(
        dedup_overlap_metric=(
            physical_metric.metric_name if use_physical else "xi_psi_proxy"
        )
    )
    return output


def _maximum_cardinality_overlap_assignment(
    reference_frequency: np.ndarray,
    candidate_frequency: np.ndarray,
    overlaps: np.ndarray,
    *,
    frequency_gate: float,
    ambiguity_delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Maximize gated cardinality first, then phase-invariant overlap."""
    reference = np.asarray(reference_frequency, dtype=np.float64)
    candidate = np.asarray(candidate_frequency, dtype=np.float64)
    overlaps = np.asarray(overlaps, dtype=np.float64)
    if overlaps.shape != (reference.size, candidate.size):
        raise ValueError("overlaps has incompatible assignment shape.")
    if reference.size == 0 or candidate.size == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i.copy(), np.empty(0), np.empty(0, dtype=bool)
    scale = np.maximum(
        np.finfo(float).tiny,
        np.maximum(np.abs(reference[:, None]), np.abs(candidate[None, :])),
    )
    frequency_difference = np.abs(reference[:, None] - candidate[None, :]) / scale
    allowed = np.isfinite(frequency_difference) & (frequency_difference <= frequency_gate)
    shape_cost = 1.0 - np.clip(np.nan_to_num(overlaps, nan=0.0), 0.0, 1.0)
    real_cost = shape_cost + 1.0e-9 * frequency_difference / frequency_gate
    nrow, ncol = allowed.shape
    size = nrow + ncol
    unmatched = float(size + 2)
    forbidden = unmatched * float(size + 2)
    augmented = np.zeros((size, size), dtype=np.float64)
    augmented[:nrow, :ncol] = np.where(allowed, real_cost, forbidden)
    augmented[:nrow, ncol:] = unmatched
    augmented[nrow:, :ncol] = unmatched
    rows, columns = linear_sum_assignment(augmented)
    real = (rows < nrow) & (columns < ncol)
    rows = rows[real]
    columns = columns[real]
    valid = allowed[rows, columns]
    rows = rows[valid].astype(np.int64)
    columns = columns[valid].astype(np.int64)

    ambiguous = np.zeros(rows.size, dtype=bool)
    for ii, (row, column) in enumerate(zip(rows, columns)):
        selected = overlaps[row, column]
        row_alternatives = allowed[row].copy()
        row_alternatives[column] = False
        column_alternatives = allowed[:, column].copy()
        column_alternatives[row] = False
        if np.any(overlaps[row, row_alternatives] >= selected - ambiguity_delta):
            ambiguous[ii] = True
        if np.any(overlaps[column_alternatives, column] >= selected - ambiguity_delta):
            ambiguous[ii] = True
    return rows, columns, frequency_difference[rows, columns], ambiguous


def _assignment_ambiguity_masks(
    reference_frequency: np.ndarray,
    candidate_frequency: np.ndarray,
    overlaps: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    frequency_gate: float,
    ambiguity_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mark every selected or losing mode involved in a near-degenerate choice."""
    reference = np.asarray(reference_frequency, dtype=np.float64)
    candidate = np.asarray(candidate_frequency, dtype=np.float64)
    overlaps = np.asarray(overlaps, dtype=np.float64)
    scale = np.maximum(
        np.finfo(float).tiny,
        np.maximum(np.abs(reference[:, None]), np.abs(candidate[None, :])),
    )
    difference = np.abs(reference[:, None] - candidate[None, :]) / scale
    allowed = np.isfinite(difference) & (difference <= frequency_gate)
    reference_ambiguous = np.zeros(reference.size, dtype=bool)
    candidate_ambiguous = np.zeros(candidate.size, dtype=bool)
    for row, column in zip(rows, columns):
        selected = overlaps[row, column]
        row_alternatives = allowed[row].copy()
        row_alternatives[column] = False
        row_alternatives &= overlaps[row] >= selected - ambiguity_delta
        column_alternatives = allowed[:, column].copy()
        column_alternatives[row] = False
        column_alternatives &= overlaps[:, column] >= selected - ambiguity_delta
        if np.any(row_alternatives) or np.any(column_alternatives):
            reference_ambiguous[row] = True
            candidate_ambiguous[column] = True
            candidate_ambiguous[row_alternatives] = True
            reference_ambiguous[column_alternatives] = True
    return reference_ambiguous, candidate_ambiguous


def _shape_metrics(
    xi_psi: np.ndarray,
    psi: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float]:
    power = np.sum(np.abs(np.asarray(xi_psi)) ** 2, axis=1)
    psi = np.asarray(psi, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights * power))
    if not np.isfinite(total) or total <= np.finfo(float).tiny:
        return np.nan, np.nan, np.nan
    peak = float(psi[int(np.argmax(power))])
    mean = float(np.sum(weights * psi * power) / total)
    width = float(np.sqrt(max(0.0, np.sum(weights * (psi - mean) ** 2 * power) / total)))
    ipr = float(np.sum(weights * power**2) / total**2)
    return peak, width, ipr


def _reconstruct_tangential_displacement(
    dataset: xr.Dataset,
    mixed: MixedRadialData | None,
) -> np.ndarray | None:
    """Reconstruct ``xi_s`` on the shared quadrature, or report unavailable."""
    if "xi_s" in dataset:
        return np.asarray(dataset["xi_s"].values, dtype=np.complex128)
    if mixed is None or "xi_s_coefficient" not in dataset:
        return None
    coefficients = np.asarray(
        dataset["xi_s_coefficient"].values,
        dtype=np.complex128,
    )
    return np.einsum(
        "qd,edm->eqm",
        np.asarray(mixed.phi_low_q, dtype=float),
        coefficients,
        optimize=True,
    )


def _track_level_solutions(
    level_solutions: tuple[xr.Dataset, ...],
    weights: np.ndarray,
    thresholds: ModeClassificationThresholds,
    radial_levels: tuple[BSplineFE, ...] | None = None,
    mixed_levels: tuple[MixedRadialData, ...] | None = None,
    physical_metric: PhysicalDisplacementMetric | None = None,
) -> xr.Dataset:
    """Build adjacent-level chains with cluster-aware eigenspace matching."""
    nlevels = len(level_solutions)
    if nlevels == 0:
        raise ValueError("At least one level solution is required.")
    psi = np.asarray(level_solutions[-1].coords["psi"].values, dtype=np.float64)
    m_pol = np.asarray(level_solutions[-1].coords["m_pol"].values)
    for dataset in level_solutions:
        if not np.array_equal(dataset.coords["psi"].values, psi):
            raise ValueError("All levels must use the identical master psi grid.")
        if not np.array_equal(dataset.coords["m_pol"].values, m_pol):
            raise ValueError("All levels must use the identical harmonic contract.")
    if mixed_levels is not None and len(mixed_levels) != nlevels:
        raise ValueError("mixed_levels must match the number of level solutions.")
    if physical_metric is not None:
        if not np.array_equal(physical_metric.points, psi):
            raise ValueError("Physical metric points do not match the master quadrature.")
        if not np.array_equal(physical_metric.m_pol, m_pol):
            raise ValueError("Physical metric harmonics do not match the level solutions.")

    tracks: list[list[int]] = [
        [index] + [-1] * (nlevels - 1)
        for index in range(level_solutions[0].sizes.get("eigenpair", 0))
    ]
    transition_overlap: dict[tuple[int, int], float] = {}
    transition_frequency: dict[tuple[int, int], float] = {}
    transition_ambiguity: dict[tuple[int, int], bool] = {}
    transition_cluster_id: dict[tuple[int, int], int] = {}
    transition_cluster_reference_dimension: dict[tuple[int, int], int] = {}
    transition_cluster_candidate_dimension: dict[tuple[int, int], int] = {}
    transition_cluster_min: dict[tuple[int, int], float] = {}
    transition_cluster_mean: dict[tuple[int, int], float] = {}
    transition_cluster_rms: dict[tuple[int, int], float] = {}
    transition_metric: dict[tuple[int, int], str] = {}

    for level in range(1, nlevels):
        reference_tracks = [
            track_index
            for track_index, track in enumerate(tracks)
            if track[level - 1] >= 0
        ]
        reference_indices = np.asarray(
            [tracks[track][level - 1] for track in reference_tracks],
            dtype=np.int64,
        )
        candidate_count = level_solutions[level].sizes.get("eigenpair", 0)
        candidate_indices = np.arange(candidate_count, dtype=np.int64)

        reference_omega = np.asarray(
            level_solutions[level - 1]["omega"].values,
            dtype=np.complex128,
        )[reference_indices]
        candidate_omega = np.asarray(
            level_solutions[level]["omega"].values,
            dtype=np.complex128,
        )
        reference_xi = np.asarray(
            level_solutions[level - 1]["xi_psi"].values,
            dtype=np.complex128,
        )[reference_indices]
        candidate_xi = np.asarray(
            level_solutions[level]["xi_psi"].values,
            dtype=np.complex128,
        )
        reference_tangential = None
        candidate_tangential = None
        transition_physical_metric = None
        if physical_metric is not None:
            reference_tangential_all = _reconstruct_tangential_displacement(
                level_solutions[level - 1],
                None if mixed_levels is None else mixed_levels[level - 1],
            )
            candidate_tangential = _reconstruct_tangential_displacement(
                level_solutions[level],
                None if mixed_levels is None else mixed_levels[level],
            )
            if reference_tangential_all is not None and candidate_tangential is not None:
                reference_tangential = reference_tangential_all[reference_indices]
                transition_physical_metric = physical_metric
        assignment = cluster_aware_mode_assignment(
            reference_omega.real,
            candidate_omega.real,
            reference_xi,
            candidate_xi,
            weights,
            frequency_gate=thresholds.frequency_gate,
            tie_delta=thresholds.assignment_ambiguity,
            reference_tangential=reference_tangential,
            candidate_tangential=candidate_tangential,
            physical_metric=transition_physical_metric,
        )
        metric_name = (
            physical_metric.metric_name
            if transition_physical_metric is not None
            else "xi_psi_proxy"
        )
        rows = assignment["rows"]
        columns = assignment["columns"]
        reference_ambiguous = assignment["reference_dimension_ambiguous"]
        candidate_ambiguous = assignment["candidate_dimension_ambiguous"]
        for row in np.flatnonzero(reference_ambiguous):
            transition_ambiguity[(reference_tracks[int(row)], level - 1)] = True
        matched_candidates: set[int] = set()
        candidate_tracks: dict[int, int] = {}
        for pair, (row, column) in enumerate(zip(rows, columns)):
            track_index = reference_tracks[int(row)]
            candidate_index = int(candidate_indices[column])
            tracks[track_index][level] = candidate_index
            matched_candidates.add(candidate_index)
            candidate_tracks[candidate_index] = track_index
            transition_overlap[(track_index, level)] = float(
                assignment["individual_overlap"][pair]
            )
            transition_frequency[(track_index, level)] = float(
                assignment["frequency_change"][pair]
            )
            transition_ambiguity[(track_index, level)] = bool(
                assignment["cluster_dimension_ambiguous"][pair]
            )
            transition_cluster_id[(track_index, level)] = int(
                assignment["cluster_id"][pair]
            )
            transition_cluster_reference_dimension[(track_index, level)] = int(
                assignment["cluster_reference_dimension"][pair]
            )
            transition_cluster_candidate_dimension[(track_index, level)] = int(
                assignment["cluster_candidate_dimension"][pair]
            )
            transition_cluster_min[(track_index, level)] = float(
                assignment["cluster_min_cosine"][pair]
            )
            transition_cluster_mean[(track_index, level)] = float(
                assignment["cluster_mean_cosine"][pair]
            )
            transition_cluster_rms[(track_index, level)] = float(
                assignment["cluster_rms_cosine"][pair]
            )
            transition_metric[(track_index, level)] = metric_name

        for candidate_index in candidate_indices:
            candidate_index = int(candidate_index)
            if candidate_index in matched_candidates:
                continue
            track = [-1] * nlevels
            track[level] = candidate_index
            tracks.append(track)
            candidate_tracks[candidate_index] = len(tracks) - 1
        for candidate_index, is_ambiguous in enumerate(candidate_ambiguous):
            if is_ambiguous:
                transition_ambiguity[(candidate_tracks[candidate_index], level)] = True

    ntracks = len(tracks)
    mode_index = np.full((ntracks, nlevels), -1, dtype=np.int64)
    s = np.full((ntracks, nlevels), np.nan + 1.0j * np.nan, dtype=np.complex128)
    omega = s.copy()
    residual = np.full((ntracks, nlevels), np.nan)
    backward = np.full((ntracks, nlevels), np.nan)
    peak = np.full((ntracks, nlevels), np.nan)
    mean_radius = np.full((ntracks, nlevels), np.nan)
    width = np.full((ntracks, nlevels), np.nan)
    ipr = np.full((ntracks, nlevels), np.nan)
    kr = np.full((ntracks, nlevels), np.nan)
    h_peak = np.full((ntracks, nlevels), np.nan)
    width_over_h = np.full((ntracks, nlevels), np.nan)
    h_times_ipr = np.full((ntracks, nlevels), np.nan)
    h_times_kr = np.full((ntracks, nlevels), np.nan)
    coarse_retention = np.full((ntracks, nlevels), np.nan)
    overlap = np.full((ntracks, nlevels), np.nan)
    frequency_change = np.full((ntracks, nlevels), np.nan)
    ambiguous = np.zeros((ntracks, nlevels), dtype=bool)
    cluster_id = np.full((ntracks, nlevels), -1, dtype=np.int64)
    cluster_reference_dimension = np.full((ntracks, nlevels), -1, dtype=np.int64)
    cluster_candidate_dimension = np.full((ntracks, nlevels), -1, dtype=np.int64)
    cluster_min = np.full((ntracks, nlevels), np.nan)
    cluster_mean = np.full((ntracks, nlevels), np.nan)
    cluster_rms = np.full((ntracks, nlevels), np.nan)
    overlap_metric = np.full((ntracks, nlevels), "unknown", dtype="U32")
    cluster_dimension_ambiguous = np.zeros((ntracks, nlevels), dtype=bool)
    reacquisition_status = np.full(
        (ntracks, nlevels), "not_attempted", dtype="U48"
    )
    xi = np.full(
        (ntracks, nlevels, psi.size, m_pol.size),
        np.nan + 1.0j * np.nan,
        dtype=np.complex128,
    )

    for track_index, track in enumerate(tracks):
        for level, index in enumerate(track):
            if index < 0:
                continue
            dataset = level_solutions[level]
            mode_index[track_index, level] = index
            s[track_index, level] = dataset["s"].values[index]
            omega[track_index, level] = dataset["omega"].values[index]
            if "residual_norm" in dataset:
                residual[track_index, level] = dataset["residual_norm"].values[index]
            backward[track_index, level] = _best_error(dataset)[index]
            xi[track_index, level] = dataset["xi_psi"].values[index]
            derivative = None
            breakpoints = None
            if radial_levels is not None and level < len(radial_levels):
                radial = radial_levels[level]
                breakpoints = np.unique(np.asarray(radial.knots, dtype=float))
                if "xi_psi_coefficient" in dataset:
                    coefficient = np.asarray(
                        dataset["xi_psi_coefficient"].values[index],
                        dtype=np.complex128,
                    )
                    derivative = np.einsum(
                        "qd,dm->qm",
                        np.asarray(radial.eval_basis_derivative(psi), dtype=float).T,
                        coefficient,
                        optimize=True,
                    )
            shape = radial_power_diagnostics(
                xi[track_index, level],
                psi,
                weights,
                derivative=derivative,
                breakpoints=breakpoints,
                metric="xi_psi_proxy",
            )
            peak[track_index, level] = float(shape["peak_radius"])
            mean_radius[track_index, level] = float(shape["mean_radius"])
            width[track_index, level] = float(shape["rms_width"])
            ipr[track_index, level] = float(shape["ipr"])
            kr[track_index, level] = float(shape["kr"])
            h_peak[track_index, level] = float(shape["h_peak"])
            width_over_h[track_index, level] = float(shape["width_over_h"])
            h_times_ipr[track_index, level] = float(shape["h_times_ipr"])
            h_times_kr[track_index, level] = float(shape["h_times_kr"])
            if radial_levels is not None:
                coarse_retention[track_index, level] = fixed_coarse_space_retention(
                    xi[track_index, level],
                    psi,
                    weights,
                    radial_levels[0],
                )
            overlap[track_index, level] = transition_overlap.get(
                (track_index, level), np.nan
            )
            frequency_change[track_index, level] = transition_frequency.get(
                (track_index, level), np.nan
            )
            ambiguous[track_index, level] = transition_ambiguity.get(
                (track_index, level), False
            )
            cluster_id[track_index, level] = transition_cluster_id.get(
                (track_index, level), -1
            )
            cluster_reference_dimension[track_index, level] = (
                transition_cluster_reference_dimension.get((track_index, level), -1)
            )
            cluster_candidate_dimension[track_index, level] = (
                transition_cluster_candidate_dimension.get((track_index, level), -1)
            )
            cluster_min[track_index, level] = transition_cluster_min.get(
                (track_index, level), np.nan
            )
            cluster_mean[track_index, level] = transition_cluster_mean.get(
                (track_index, level), np.nan
            )
            cluster_rms[track_index, level] = transition_cluster_rms.get(
                (track_index, level), np.nan
            )
            overlap_metric[track_index, level] = transition_metric.get(
                (track_index, level), "unknown"
            )
            cluster_dimension_ambiguous[track_index, level] = ambiguous[
                track_index, level
            ]
            if "reacquisition_status" in dataset:
                reacquisition_status[track_index, level] = str(
                    dataset["reacquisition_status"].values[index]
                )

    return xr.Dataset(
        data_vars={
            "mode_index": (("track", "level"), mode_index),
            "present": (("track", "level"), mode_index >= 0),
            "s": (("track", "level"), s),
            "omega": (("track", "level"), omega),
            "residual_norm": (("track", "level"), residual),
            "backward_error": (("track", "level"), backward),
            "xi_psi_overlap": (("track", "level"), overlap),
            "frequency_change": (("track", "level"), frequency_change),
            "assignment_ambiguous": (("track", "level"), ambiguous),
            "cluster_id": (("track", "level"), cluster_id),
            "cluster_reference_dimension": (
                ("track", "level"),
                cluster_reference_dimension,
            ),
            "cluster_candidate_dimension": (
                ("track", "level"),
                cluster_candidate_dimension,
            ),
            "cluster_min_principal_cosine": (("track", "level"), cluster_min),
            "cluster_mean_principal_cosine": (("track", "level"), cluster_mean),
            "cluster_rms_principal_cosine": (("track", "level"), cluster_rms),
            "overlap_metric": (("track", "level"), overlap_metric),
            "cluster_dimension_ambiguous": (
                ("track", "level"),
                cluster_dimension_ambiguous,
            ),
            "reacquisition_status": (
                ("track", "level"),
                reacquisition_status,
            ),
            "peak_radius": (("track", "level"), peak),
            "power_weighted_mean_radius": (("track", "level"), mean_radius),
            "rms_width": (("track", "level"), width),
            "ipr": (("track", "level"), ipr),
            "kr": (("track", "level"), kr),
            "h_peak": (("track", "level"), h_peak),
            "width_over_h": (("track", "level"), width_over_h),
            "h_times_ipr": (("track", "level"), h_times_ipr),
            "h_times_kr": (("track", "level"), h_times_kr),
            "coarse_space_retention": (
                ("track", "level"),
                coarse_retention,
            ),
            "xi_psi": (("track", "level", "psi", "m_pol"), xi),
        },
        coords={
            "track": np.arange(ntracks, dtype=np.int64),
            "level": np.arange(nlevels, dtype=np.int64),
            "psi": psi,
            "m_pol": m_pol,
        },
        attrs={
            "frequency_gate": thresholds.frequency_gate,
            "overlap_measure": (
                physical_metric.definition
                if physical_metric is not None
                else "phase-invariant harmonic-resolved composite-Gauss dpsi L2"
            ),
            "overlap_metric": (
                physical_metric.metric_name
                if physical_metric is not None
                else "xi_psi_proxy"
            ),
            "overlap_metric_fallback": "xi_psi_proxy",
            "parallel_displacement_in_overlap": (
                bool(physical_metric.parallel_component_included)
                if physical_metric is not None
                else False
            ),
            "power_density_metric": "xi_psi_proxy",
            "ipr_convention": "integral(P^2)dpsi/(integral(P)dpsi)^2",
            "assignment_priority": (
                "maximum cardinality; isolated xi_psi overlap; nearly tied "
                "components use weighted principal angles"
            ),
        },
    )


def _attach_enriched_residuals(
    tracks: xr.Dataset,
    level_solutions: tuple[xr.Dataset, ...],
    projected_matrices: tuple[tuple, ...],
    mixed_levels: tuple[MixedRadialData, ...],
    solver,
    sb,
    *,
    bc_tol: float | None,
) -> xr.Dataset:
    """Evaluate each coarse eigenpair in the adjacent finer operator."""
    ntracks = tracks.sizes.get("track", 0)
    nlevels = tracks.sizes.get("level", 0)
    values = np.full((ntracks, nlevels), np.nan)
    if nlevels < 2 or "eigenvector" not in level_solutions[0]:
        return tracks.assign(
            enriched_residual=(("track", "level"), values)
        )
    transfers = [
        mixed_space_prolongation(
            mixed_levels[level],
            mixed_levels[level + 1],
            tracks.sizes.get("m_pol", 0),
        )
        for level in range(nlevels - 1)
    ]
    fine_boundary = []
    for level in range(1, nlevels):
        bc = solver._build_nep_boundary_condition(
            sb,
            bc_kind="full",
            mixed=mixed_levels[level],
        )
        bc.get_sparse_prolongation(tol=bc_tol)
        fine_boundary.append(bc)
    mode_index = np.asarray(tracks["mode_index"].values, dtype=int)
    present = np.asarray(tracks["present"].values, dtype=bool)
    for track in range(ntracks):
        for coarse_level in range(nlevels - 1):
            if not present[track, coarse_level]:
                continue
            index = int(mode_index[track, coarse_level])
            full_coarse = np.asarray(
                level_solutions[coarse_level]["eigenvector"].values[index],
                dtype=np.complex128,
            )
            full_fine = transfers[coarse_level] @ full_coarse
            bc = fine_boundary[coarse_level]
            free = np.asarray(bc.sparse_free, dtype=int)
            reduced_fine = np.asarray(full_fine[free], dtype=np.complex128)
            prolongation = bc.get_sparse_prolongation(tol=bc_tol)
            reconstruction = prolongation @ reduced_fine
            relative_constraint_error = np.linalg.norm(reconstruction - full_fine) / max(
                np.linalg.norm(full_fine), np.finfo(float).tiny
            )
            if relative_constraint_error > 1.0e-8:
                # The prolonged coarse mode should satisfy the same endpoint
                # constraints.  If it does not, retain unknown rather than
                # evaluating a different projected vector.
                continue
            s = complex(level_solutions[coarse_level]["s"].values[index])
            values[track, coarse_level + 1] = enriched_residual(
                projected_matrices[coarse_level + 1],
                s,
                reduced_fine,
            )
    output = tracks.copy()
    output["enriched_residual"] = (("track", "level"), values)
    output["enriched_residual"].attrs.update(
        definition=(
            "fine-operator residual of the exactly knot-inserted coarse full "
            "vector, normalized by the sum of polynomial-term norms"
        ),
        distinction="not the SLEPc residual in the same discrete space",
    )
    return output


def _root_identity(
    level_solutions: Sequence[xr.Dataset],
    level: int,
    index: int,
) -> tuple[str, int, int]:
    """Identify a saved root independently of later track-row reordering."""
    dataset = level_solutions[level]
    if "source_eigenpair" in dataset:
        source = dataset["source_eigenpair"].values[index]
        if np.isfinite(source):
            return "source", int(level), int(source)
    return "index", int(level), int(index)


def _track_root_identities(
    tracks: xr.Dataset,
    level_solutions: Sequence[xr.Dataset],
    selected: np.ndarray,
) -> set[tuple[str, int, int]]:
    """Collect persistent root identities for selected input tracks."""
    selected = np.asarray(selected, dtype=bool)
    if selected.shape != (tracks.sizes.get("track", 0),):
        raise ValueError("selected must contain one Boolean value per input track")
    present = np.asarray(tracks["present"].values, dtype=bool)
    indices = np.asarray(tracks["mode_index"].values, dtype=int)
    identities: set[tuple[str, int, int]] = set()
    for track in np.flatnonzero(selected):
        for level in np.flatnonzero(present[track]):
            identities.add(
                _root_identity(level_solutions, int(level), int(indices[track, level]))
            )
    return identities


def _track_mask_from_root_identities(
    tracks: xr.Dataset,
    level_solutions: Sequence[xr.Dataset],
    identities: set[tuple[str, int, int]],
) -> np.ndarray:
    """Remap eligibility after deduplication or cross-level retracking."""
    present = np.asarray(tracks["present"].values, dtype=bool)
    indices = np.asarray(tracks["mode_index"].values, dtype=int)
    selected = np.zeros(present.shape[0], dtype=bool)
    for track in range(present.shape[0]):
        selected[track] = any(
            _root_identity(level_solutions, int(level), int(indices[track, level]))
            in identities
            for level in np.flatnonzero(present[track])
        )
    return selected


def _continuum_vector_overlaps(
    previous: np.ndarray,
    current: np.ndarray,
) -> np.ndarray:
    """Return phase-invariant overlaps between adjacent continuum eigenvectors."""
    previous = np.asarray(previous, dtype=np.complex128)
    current = np.asarray(current, dtype=np.complex128)
    if previous.ndim != 2 or current.ndim != 2:
        raise ValueError("Continuum eigenvectors must be two-dimensional.")
    if previous.shape[0] != current.shape[0]:
        raise ValueError("Adjacent continuum eigenvectors must share a state space.")
    numerator = np.abs(previous.conjugate().T @ current)
    previous_norm = np.linalg.norm(previous, axis=0)
    current_norm = np.linalg.norm(current, axis=0)
    denominator = previous_norm[:, None] * current_norm[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        overlaps = numerator / denominator
    return np.where(
        np.isfinite(overlaps),
        np.clip(overlaps.real, 0.0, 1.0),
        np.nan,
    )


def _continuum_branch_order(
    previous_roots: np.ndarray,
    current_roots: np.ndarray,
    previous_vectors: np.ndarray,
    current_vectors: np.ndarray,
    *,
    ambiguity_delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match adjacent full-continuum branches by eigenvector character."""
    previous_roots = np.asarray(previous_roots, dtype=np.complex128)
    current_roots = np.asarray(current_roots, dtype=np.complex128)
    if previous_roots.shape != current_roots.shape or previous_roots.ndim != 1:
        raise ValueError("Adjacent continuum spectra must have equal one-dimensional shape.")
    overlaps = _continuum_vector_overlaps(previous_vectors, current_vectors)
    if overlaps.shape != (previous_roots.size, current_roots.size):
        raise ValueError("Continuum overlap matrix has an incompatible shape.")

    spectral_distance = _relative_complex_distance(
        previous_roots[:, None],
        current_roots[None, :],
    )
    spectral_tiebreak = np.nan_to_num(
        spectral_distance / (1.0 + spectral_distance),
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )
    finite_overlap = np.isfinite(overlaps)
    # Eigenvector character determines the assignment.  The spectral term only
    # breaks numerical ties and therefore cannot hide a genuine branch crossing.
    cost = np.where(finite_overlap, 1.0 - overlaps, 2.0)
    cost += 1.0e-9 * spectral_tiebreak
    rows, columns = linear_sum_assignment(cost)
    order = np.empty(previous_roots.size, dtype=np.int64)
    order[rows] = columns

    selected_overlap = np.full(previous_roots.size, np.nan)
    ambiguous = np.zeros(previous_roots.size, dtype=bool)
    for row, column in zip(rows, columns):
        selected = overlaps[row, column]
        selected_overlap[row] = selected
        if not np.isfinite(selected):
            ambiguous[row] = True
            continue
        row_alternatives = np.ones(current_roots.size, dtype=bool)
        row_alternatives[column] = False
        column_alternatives = np.ones(previous_roots.size, dtype=bool)
        column_alternatives[row] = False
        ambiguous[row] = bool(
            np.any(overlaps[row, row_alternatives] >= selected - ambiguity_delta)
            or np.any(overlaps[column_alternatives, column] >= selected - ambiguity_delta)
        )
    return order, selected_overlap, ambiguous


def _solve_master_continuum(
    sb,
    psi: np.ndarray,
    *,
    ambiguity_delta: float = 1.0e-3,
) -> xr.Dataset:
    """Solve and eigenvector-track the stored full sound-Alfven pencils."""
    A = sb.A
    B = sb.B
    density = np.asarray(sb.density, dtype=np.complex128)
    psi = np.asarray(psi, dtype=np.float64)
    if A.shape != B.shape or len(A.shape) != 3 or A.shape[1] != A.shape[2]:
        raise ValueError("Stored continuum A/B tensors must be equal square pencils.")
    if A.shape[0] != psi.size or density.shape != (psi.size,):
        raise ValueError("Continuum tensors, density, and master psi grid must align.")
    if not np.isfinite(ambiguity_delta) or not 0.0 <= ambiguity_delta <= 1.0:
        raise ValueError("ambiguity_delta must lie in [0, 1].")
    nbranch = A.shape[1]
    values = np.full((psi.size, nbranch), np.nan + 1.0j * np.nan)
    transition_overlap = np.full((psi.size, nbranch), np.nan)
    transition_ambiguous = np.zeros((psi.size, nbranch), dtype=bool)
    previous_vectors = None
    for index in range(psi.size):
        A_slice = A[index]
        B_slice = B[index]
        if hasattr(A_slice, "toarray"):
            A_slice = A_slice.toarray()
        if hasattr(B_slice, "toarray"):
            B_slice = B_slice.toarray()
        roots, vectors = eig(
            np.asarray(A_slice, dtype=np.complex128),
            np.asarray(B_slice, dtype=np.complex128),
            left=False,
            right=True,
        )
        roots = np.asarray(roots, dtype=np.complex128)
        vectors = np.asarray(vectors, dtype=np.complex128)
        scale = complex(mu_0) * density[index]
        with np.errstate(divide="ignore", invalid="ignore"):
            roots = roots / scale
        if index == 0:
            order = np.lexsort(
                (
                    np.nan_to_num(roots.imag, nan=np.inf),
                    np.nan_to_num(roots.real, nan=np.inf),
                )
            )
        else:
            order, overlap, ambiguous = _continuum_branch_order(
                values[index - 1],
                roots,
                previous_vectors,
                vectors,
                ambiguity_delta=ambiguity_delta,
            )
            transition_overlap[index] = overlap
            transition_ambiguous[index] = ambiguous
        values[index] = roots[order]
        previous_vectors = vectors[:, order]
    return xr.Dataset(
        data_vars={
            "s": (("psi", "branch"), values),
            "omega": (("psi", "branch"), np.sqrt(values)),
            "branch_assignment_overlap": (
                ("psi", "branch"),
                transition_overlap,
            ),
            "branch_assignment_ambiguous": (
                ("psi", "branch"),
                transition_ambiguous,
            ),
        },
        coords={
            "psi": psi,
            "branch": np.arange(values.shape[1], dtype=np.int64),
        },
        attrs={
            "which": "full",
            "normalization": (
                "s=lambda_continuum/(mu_0*density)=omega**2 in omegaA units"
            ),
            "ntor": int(sb.ntor),
            "m_pol": np.array2string(np.asarray(sb.m_range), separator=","),
            "max_dm": int(sb.max_dm),
            "source": "solver_blocks2.A/B on the finest master operator grid",
            "branch_tracking": (
                "adjacent-psi maximum-overlap assignment of generalized right "
                "eigenvectors; relative-s is tie-break only"
            ),
            "branch_assignment_ambiguity_delta": float(ambiguity_delta),
        },
    )


def _continuum_evidence(
    omega: complex,
    xi_psi: np.ndarray,
    shape_psi: np.ndarray,
    weights: np.ndarray,
    continuum: xr.Dataset,
    thresholds: ModeClassificationThresholds,
) -> tuple[bool, float, float, bool]:
    """Return crossing, clearance, proximity, and full-coverage validity."""
    frequency = abs(float(np.real(omega)))
    if not np.isfinite(frequency) or frequency <= np.finfo(float).tiny:
        return False, np.nan, np.nan, False
    cont_psi = np.asarray(continuum.coords["psi"].values, dtype=np.float64)
    cont_omega = np.asarray(continuum["omega"].values, dtype=np.complex128)
    grid_valid = bool(
        cont_psi.ndim == 1
        and cont_psi.size >= 2
        and np.all(np.isfinite(cont_psi))
        and np.all(np.diff(cont_psi) > 0.0)
        and cont_omega.ndim == 2
        and cont_omega.shape[0] == cont_psi.size
        and cont_omega.shape[1] > 0
    )
    if not grid_valid:
        return False, np.nan, np.nan, False
    real = cont_omega.real
    imag_ratio = np.abs(cont_omega.imag) / np.maximum(
        np.finfo(float).tiny,
        np.abs(real),
    )
    valid_continuum = (
        np.isfinite(cont_omega.real)
        & np.isfinite(cont_omega.imag)
        & (imag_ratio <= thresholds.imaginary_frequency_ratio)
    )
    branch_tracking_complete = True
    if "branch_assignment_ambiguous" in continuum:
        branch_ambiguous = np.asarray(
            continuum["branch_assignment_ambiguous"].values,
            dtype=bool,
        )
        branch_tracking_complete = bool(
            branch_ambiguous.shape == cont_omega.shape
            and not np.any(branch_ambiguous)
        )
    coverage_complete = bool(
        np.all(valid_continuum) and branch_tracking_complete
    )
    real = np.where(
        valid_continuum,
        real,
        np.nan,
    )
    delta = real - frequency
    crossing = False
    for branch in range(delta.shape[1]):
        values = delta[:, branch]
        finite = np.isfinite(values)
        valid_values = values[finite]
        if valid_values.size and np.any(valid_values == 0.0):
            crossing = True
            break
        adjacent = finite[:-1] & finite[1:]
        if np.any(values[:-1][adjacent] * values[1:][adjacent] < 0.0):
            crossing = True
            break
    relative = np.abs(delta) / frequency
    clearance = float(np.nanmin(relative)) if np.any(np.isfinite(relative)) else np.nan
    if crossing:
        clearance = 0.0

    shape_psi = np.asarray(shape_psi, dtype=np.float64)
    interpolated = np.full((shape_psi.size, real.shape[1]), np.nan)
    for branch in range(real.shape[1]):
        valid_indices = np.flatnonzero(np.isfinite(real[:, branch]))
        if valid_indices.size == 0:
            continue
        segments = np.split(
            valid_indices,
            np.flatnonzero(np.diff(valid_indices) != 1) + 1,
        )
        for segment in segments:
            if segment.size >= 2:
                inside = (shape_psi >= cont_psi[segment[0]]) & (
                    shape_psi <= cont_psi[segment[-1]]
                )
                interpolated[inside, branch] = np.interp(
                    shape_psi[inside],
                    cont_psi[segment],
                    real[segment, branch],
                )
            else:
                exact = np.isclose(
                    shape_psi,
                    cont_psi[segment[0]],
                    rtol=0.0,
                    atol=16.0 * np.finfo(float).eps,
                )
                interpolated[exact, branch] = real[segment[0], branch]
    relative_interpolated = np.abs(interpolated - frequency) / frequency
    finite_rows = np.any(np.isfinite(relative_interpolated), axis=1)
    proximity = np.full(shape_psi.size, np.nan)
    if np.any(finite_rows):
        proximity[finite_rows] = np.min(
            np.where(
                np.isfinite(relative_interpolated[finite_rows]),
                relative_interpolated[finite_rows],
                np.inf,
            ),
            axis=1,
        )
    power = np.sum(np.abs(np.asarray(xi_psi)) ** 2, axis=1)
    weighted_power = np.asarray(weights) * power
    total = float(np.sum(weighted_power))
    if total <= np.finfo(float).tiny or not np.isfinite(total):
        power_fraction = np.nan
    else:
        power_fraction = float(
            np.sum(weighted_power[proximity <= thresholds.continuum_band]) / total
        )
    return crossing, clearance, power_fraction, coverage_complete


def _classify_tracks_legacy(
    tracks: xr.Dataset,
    continuum: xr.Dataset,
    weights: np.ndarray,
    level_complete: np.ndarray,
    thresholds: ModeClassificationThresholds,
) -> xr.Dataset:
    """Apply the conservative four-label contract and retain every evidence field."""
    ntracks = tracks.sizes.get("track", 0)
    nlevels = tracks.sizes.get("level", 0)
    labels = np.full(ntracks, "inconclusive", dtype="U24")
    reasons = np.full(ntracks, "", dtype="U512")
    numerical_valid = np.zeros(ntracks, dtype=bool)
    chain_numerically_valid = np.zeros(ntracks, dtype=bool)
    level_numerically_valid = np.zeros((ntracks, nlevels), dtype=bool)
    converged_chain = np.zeros(ntracks, dtype=bool)
    strict_gap = np.zeros(ntracks, dtype=bool)
    crossing = np.zeros(ntracks, dtype=bool)
    ambiguous = np.zeros(ntracks, dtype=bool)
    present_all = np.zeros(ntracks, dtype=bool)
    clearance = np.full(ntracks, np.nan)
    continuum_power = np.full(ntracks, np.nan)
    imaginary_ratio = np.full(ntracks, np.nan)
    final_frequency_change = np.full(ntracks, np.nan)
    final_overlap = np.full(ntracks, np.nan)
    width_ratio = np.full(ntracks, np.nan)
    ipr_ratio = np.full(ntracks, np.nan)
    loss_of_shape = np.zeros(ntracks, dtype=bool)
    resonance_signature = np.zeros(ntracks, dtype=bool)
    continuum_complete = np.zeros(ntracks, dtype=bool)
    last_level = np.full(ntracks, -1, dtype=np.int64)

    search_complete = bool(np.all(np.asarray(level_complete, dtype=bool)))
    shape_psi = np.asarray(tracks.coords["psi"].values, dtype=np.float64)
    for track in range(ntracks):
        present = np.asarray(tracks["present"].values[track], dtype=bool)
        available = np.flatnonzero(present)
        if available.size == 0:
            reasons[track] = "no_level_root"
            continue
        final = int(available[-1])
        last_level[track] = final
        present_all[track] = bool(np.all(present))
        ambiguous[track] = bool(
            np.any(tracks["assignment_ambiguous"].values[track, available])
        )
        mode_omega = complex(tracks["omega"].values[track, final])
        imaginary_ratio[track] = abs(mode_omega.imag) / max(
            np.finfo(float).tiny,
            abs(mode_omega.real),
        )
        error = float(tracks["backward_error"].values[track, final])
        for level in available:
            level_omega = complex(tracks["omega"].values[track, level])
            level_imaginary_ratio = abs(level_omega.imag) / max(
                np.finfo(float).tiny,
                abs(level_omega.real),
            )
            level_error = float(tracks["backward_error"].values[track, level])
            level_numerically_valid[track, level] = bool(
                np.isfinite(level_error)
                and level_error <= thresholds.backward_error
                and np.isfinite(level_imaginary_ratio)
                and level_imaginary_ratio <= thresholds.imaginary_frequency_ratio
            )
        numerical_valid[track] = level_numerically_valid[track, final]
        chain_numerically_valid[track] = bool(
            present_all[track]
            and np.all(level_numerically_valid[track, available])
        )
        final_frequency_change[track] = tracks["frequency_change"].values[
            track, final
        ]
        final_overlap[track] = tracks["xi_psi_overlap"].values[track, final]
        if final > 0 and present[final - 1]:
            previous_width = tracks["rms_width"].values[track, final - 1]
            previous_ipr = tracks["ipr"].values[track, final - 1]
            if np.isfinite(previous_width) and previous_width > 0.0:
                width_ratio[track] = (
                    tracks["rms_width"].values[track, final] / previous_width
                )
            if np.isfinite(previous_ipr) and previous_ipr > 0.0:
                ipr_ratio[track] = tracks["ipr"].values[track, final] / previous_ipr
        loss_of_shape[track] = bool(
            np.isfinite(final_overlap[track])
            and final_overlap[track] < thresholds.medium_fine_overlap
        )
        (
            crossing[track],
            clearance[track],
            continuum_power[track],
            continuum_complete[track],
        ) = (
            _continuum_evidence(
                mode_omega,
                tracks["xi_psi"].values[track, final],
                shape_psi,
                weights,
                continuum,
                thresholds,
            )
        )
        strict_gap[track] = bool(
            continuum_complete[track]
            and not crossing[track]
            and np.isfinite(clearance[track])
            and clearance[track] >= thresholds.continuum_clearance
        )
        resonance_signature[track] = bool(
            (
                np.isfinite(continuum_power[track])
                and continuum_power[track] >= thresholds.continuum_power_fraction
            )
            or loss_of_shape[track]
            or (
                np.isfinite(width_ratio[track])
                and width_ratio[track] < thresholds.width_contraction
            )
            or (
                np.isfinite(ipr_ratio[track])
                and ipr_ratio[track] > thresholds.ipr_growth
            )
        )
        converged_chain[track] = bool(
            nlevels >= 3
            and present_all[track]
            and chain_numerically_valid[track]
            and search_complete
            and not ambiguous[track]
            and np.isfinite(final_frequency_change[track])
            and final_frequency_change[track] <= thresholds.fine_frequency_change
            and np.isfinite(final_overlap[track])
            and final_overlap[track] >= thresholds.medium_fine_overlap
        )

        codes: list[str] = []
        if not numerical_valid[track]:
            labels[track] = "numerical_artifact"
            codes.append("invalid_backward_error_or_reality")
        elif not search_complete:
            labels[track] = "inconclusive"
            codes.append("incomplete_search_windows")
        elif ambiguous[track]:
            labels[track] = "inconclusive"
            codes.append("ambiguous_mode_assignment")
        elif nlevels < 3:
            labels[track] = "inconclusive"
            codes.append("insufficient_levels")
        elif not continuum_complete[track]:
            labels[track] = "inconclusive"
            codes.append("incomplete_continuum_coverage")
        elif converged_chain[track] and strict_gap[track]:
            labels[track] = "physical_gap"
            codes.extend(("converged_three_level_chain", "strict_continuum_gap"))
        elif crossing[track] and resonance_signature[track]:
            labels[track] = "continuum_resonance"
            codes.extend(("full_continuum_crossing", "resonance_signature"))
        elif not present_all[track]:
            labels[track] = "numerical_artifact"
            codes.append("complete_search_nonpersistent_root")
        else:
            labels[track] = "inconclusive"
            if not crossing[track] and not strict_gap[track]:
                codes.append("marginal_continuum_clearance")
            if crossing[track] and not resonance_signature[track]:
                codes.append("crossing_without_resonance_signature")
            if not converged_chain[track]:
                codes.append("unconverged_chain")
            if present_all[track] and not chain_numerically_valid[track]:
                codes.append("invalid_intermediate_level")
        reasons[track] = ";".join(codes) or "conflicting_evidence"

    return xr.Dataset(
        data_vars={
            "label": ("track", labels),
            "reason": ("track", reasons),
            "last_level": ("track", last_level),
            "numerically_valid": ("track", numerical_valid),
            "chain_numerically_valid": ("track", chain_numerically_valid),
            "level_numerically_valid": (
                ("track", "level"),
                level_numerically_valid,
            ),
            "converged_chain": ("track", converged_chain),
            "present_all_levels": ("track", present_all),
            "assignment_ambiguous": ("track", ambiguous),
            "strict_continuum_crossing": ("track", crossing),
            "strict_gap": ("track", strict_gap),
            "minimum_relative_continuum_clearance": ("track", clearance),
            "continuum_band_power_fraction": ("track", continuum_power),
            "imaginary_frequency_ratio": ("track", imaginary_ratio),
            "fine_frequency_change": ("track", final_frequency_change),
            "medium_fine_overlap": ("track", final_overlap),
            "width_ratio": ("track", width_ratio),
            "ipr_ratio": ("track", ipr_ratio),
            "loss_of_shape_convergence": ("track", loss_of_shape),
            "resonance_signature": ("track", resonance_signature),
            "continuum_complete": ("track", continuum_complete),
        },
        coords={
            "track": tracks.coords["track"].values,
            "level": tracks.coords["level"].values,
        },
        attrs={
            "allowed_labels": ",".join(_LABELS),
            "search_complete": search_complete,
            "thresholds": repr(thresholds),
            "classification_priority": (
                "invalid; incomplete/ambiguous/insufficient; physical gap; "
                "continuum resonance; nonpersistent artifact; inconclusive"
            ),
        },
    )


def _classify_tracks(
    tracks: xr.Dataset,
    continuum: xr.Dataset,
    weights: np.ndarray,
    level_complete: np.ndarray,
    thresholds: ModeClassificationThresholds,
    *,
    classification_version: str = "legacy",
    evidence_config: EvidenceV2Config | None = None,
    catalog_status: str | np.ndarray | None = None,
) -> xr.Dataset:
    """Dispatch the preserved legacy classifier or the evidence-v2 pipeline."""
    version = str(classification_version).strip().lower()
    if version == "legacy":
        result = _classify_tracks_legacy(
            tracks,
            continuum,
            weights,
            level_complete,
            thresholds,
        )
        result.attrs["classification_version"] = "legacy"
        return result
    if version != CLASSIFICATION_VERSION:
        raise ValueError(
            f"classification_version must be 'legacy' or {CLASSIFICATION_VERSION!r}."
        )
    if catalog_status is None:
        catalog_status = np.where(
            np.asarray(level_complete, dtype=bool),
            "certified_complete",
            "partial",
        )
    return classify_tracks_evidence_v2(
        tracks,
        continuum,
        weights,
        catalog_status,
        evidence_config,
    )


def _window_diagnostics_dataset(
    records: list[_WindowRecord],
    nlevels: int,
) -> xr.Dataset:
    grouped = [[record for record in records if record.level == level] for level in range(nlevels)]
    nwindow = max((len(items) for items in grouped), default=0)
    shape = (nlevels, nwindow)
    valid = np.zeros(shape, dtype=bool)
    integer = {
        name: np.full(shape, -1, dtype=np.int64)
        for name in (
            "window_id",
            "parent",
            "depth",
            "nconv",
            "nreturned",
            "iterations",
            "effective_nev",
        )
    }
    floating = {
        name: np.full(shape, np.nan)
        for name in (
            "core_omega_min",
            "core_omega_max",
            "solve_omega_min",
            "solve_omega_max",
            # Compatibility aliases for the PEP solve bounds.
            "omega_min",
            "omega_max",
        )
    }
    target = np.full(shape, np.nan + 1.0j * np.nan, dtype=np.complex128)
    boolean = {
        name: np.zeros(shape, dtype=bool)
        for name in ("saturated", "leaf", "complete")
    }
    reason = np.full(shape, "", dtype="U64")
    catalog_status = np.full(shape, "partial", dtype="U24")
    for level, items in enumerate(grouped):
        for column, record in enumerate(items):
            valid[level, column] = True
            integer["window_id"][level, column] = record.window
            integer["parent"][level, column] = record.parent
            integer["depth"][level, column] = record.depth
            integer["nconv"][level, column] = record.nconv
            integer["nreturned"][level, column] = record.nreturned
            integer["iterations"][level, column] = record.iterations
            integer["effective_nev"][level, column] = record.effective_nev
            floating["core_omega_min"][level, column] = record.core_omega_min
            floating["core_omega_max"][level, column] = record.core_omega_max
            floating["solve_omega_min"][level, column] = record.solve_omega_min
            floating["solve_omega_max"][level, column] = record.solve_omega_max
            floating["omega_min"][level, column] = record.solve_omega_min
            floating["omega_max"][level, column] = record.solve_omega_max
            target[level, column] = record.target_s
            boolean["saturated"][level, column] = record.saturated
            boolean["leaf"][level, column] = record.leaf
            boolean["complete"][level, column] = record.complete
            reason[level, column] = record.reason
            catalog_status[level, column] = record.catalog_status
    data_vars = {"valid": (("level", "window"), valid)}
    data_vars.update(
        {name: (("level", "window"), value) for name, value in integer.items()}
    )
    data_vars.update(
        {name: (("level", "window"), value) for name, value in floating.items()}
    )
    data_vars.update(
        {name: (("level", "window"), value) for name, value in boolean.items()}
    )
    data_vars["target_s"] = (("level", "window"), target)
    data_vars["reason"] = (("level", "window"), reason)
    data_vars["catalog_status"] = (("level", "window"), catalog_status)
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "level": np.arange(nlevels, dtype=np.int64),
            "window": np.arange(nwindow, dtype=np.int64),
        },
    )


def _is_converged_reason(reason) -> bool:
    """Interpret positive SLEPc reasons and explicit converged fake reasons."""
    try:
        return operator.index(reason) > 0
    except (TypeError, ValueError):
        text = str(reason).strip().lower()
        rejected = ("diverg", "fail", "error", "iterating", "maxit", "max_it")
        return "converg" in text and not any(token in text for token in rejected)


class superSolver5Multigrid:
    """Solve and conservatively classify solver5 roots on nested radial spaces."""

    def __init__(
        self,
        equ,
        cont,
        base_radialfe: BSplineFE,
        levels: int = 3,
    ):
        self.equ = equ
        self.cont = cont
        self.radial_levels = _nested_bspline_levels(base_radialfe, levels=levels)
        self.levels = len(self.radial_levels)
        self.finest_radialfe = self.radial_levels[-1]
        self.master_points, self.master_weights = (
            self.finest_radialfe.get_quadrature_points()
        )
        self.master_points = np.asarray(self.master_points, dtype=np.float64)
        self.master_weights = np.asarray(self.master_weights, dtype=np.float64)
        self.mixed_levels = tuple(
            _build_mixed_radial_data(
                radialfe,
                self.master_points,
                self.master_weights,
            )
            for radialfe in self.radial_levels
        )
        self.solver = superSolver5(equ, cont, self.finest_radialfe)

    def build_shifted_mesh_variant(
        self,
        options: MeshPerturbationOptions | Mapping[str, object] | None = None,
    ) -> "superSolver5Multigrid":
        """Return a one-level same-resolution solver for selective mesh checks.

        The axis, edge, degree, and number of B-splines match the production
        finest space.  This method only prepares the deterministic variant;
        callers choose a small candidate set and narrow frequency windows.
        """
        options = _coerce_options(options, MeshPerturbationOptions)
        shifted = deterministic_shifted_bspline(self.finest_radialfe, options)
        variant = type(self)(self.equ, self.cont, shifted, levels=1)
        variant.mesh_perturbation = options
        return variant

    @staticmethod
    def _window_region(
        omega_min: float,
        omega_max: float,
        search: MultigridSearchOptions,
    ) -> tuple[complex, dict]:
        s_min = omega_min**2
        s_max = omega_max**2
        center = 0.5 * (s_min + s_max)
        radius = 0.5 * (s_max - s_min)
        return complex(center), {
            "type": "ellipse",
            "center": complex(center),
            "radius": float(radius),
            "vscale": float(search.region_vscale),
        }

    def _build_window_eigensolver(
        self,
        *,
        matrices: tuple,
        target_s: complex,
        region: dict,
        search: MultigridSearchOptions,
        prefix: str,
        verbose: bool,
    ):
        """Use the same transparent exact-zero dispatch as ``superSolver5``."""
        T0, T1, Tminus1 = matrices
        return _build_solver5_eigensolver_from_matrices(
            gamma_s=getattr(getattr(self, "solver", None), "gamma_s", 1.0),
            T0=T0,
            T1=T1,
            Tminus1=Tminus1,
            target_s=target_s,
            region=region,
            prefix=prefix,
            pep_type=search.pep_type,
            st_type=search.st_type,
            which=search.which,
            nev=search.nev,
            tol=search.tol,
            maxit=search.maxit,
            options=dict(search.options),
            verbose=verbose,
            compute_residuals=True,
            quadratic_solver_type=QuadraticPEPSolver,
            linear_solver_type=LinearGeneralizedEPSSolver,
        )

    @staticmethod
    def _root_indices_in_window(
        roots: np.ndarray,
        omega_min: float,
        omega_max: float,
        region: Mapping[str, object],
        tolerance: float,
    ) -> np.ndarray:
        roots = np.asarray(roots, dtype=np.complex128)
        omega = np.sqrt(roots)
        finite = (
            np.isfinite(roots.real)
            & np.isfinite(roots.imag)
            & np.isfinite(omega.real)
            & np.isfinite(omega.imag)
        )
        scale = max(1.0, abs(omega_min), abs(omega_max))
        inside_frequency = (
            omega.real >= omega_min - tolerance * scale
        ) & (omega.real <= omega_max + tolerance * scale)
        center = complex(region["center"])
        radius = float(region["radius"])
        vscale = float(region["vscale"])
        horizontal = (roots.real - center.real) / radius
        vertical = (roots.imag - center.imag) / (radius * vscale)
        inside_ellipse = horizontal**2 + vertical**2 <= 1.0 + tolerance
        return np.flatnonzero(finite & inside_frequency & inside_ellipse)

    @staticmethod
    def _metric_values(pep, name: str, indices: np.ndarray) -> np.ndarray:
        values = getattr(pep, name, None)
        if values is None:
            return np.full(indices.size, np.nan)
        values = np.asarray(values, dtype=np.float64)
        return values[indices]

    def _pack_window_result(
        self,
        *,
        sb,
        mixed: MixedRadialData,
        result: NEPResult,
        pep,
        indices: np.ndarray,
        level: int,
        window: int,
        depth: int,
        core_omega_min: float,
        core_omega_max: float,
        solve_omega_min: float,
        solve_omega_max: float,
        final_window: bool,
        bc_tol: float | None,
    ) -> xr.Dataset:
        layout = self.solver._mixed_layout(sb, mixed=mixed)
        solver_vectors = np.asarray(result.eigenvectors, dtype=np.complex128)
        if solver_vectors.ndim != 2:
            raise ValueError("Eigensolver vectors must have shape (eigenpair, dof).")
        solver_vectors = solver_vectors[indices]
        bc = self.solver._build_nep_boundary_condition(
            sb,
            bc_kind="full",
            mixed=mixed,
        )
        reduced_dofs = bc.get_sparse_free_dofs(tol=bc_tol)
        if solver_vectors.shape[1] == layout.n_trial:
            full_vectors = solver_vectors.copy()
            source_space = "full"
        elif solver_vectors.shape[1] == reduced_dofs:
            full_vectors = np.vstack(
                [bc.expand_sparse(vector, tol=bc_tol) for vector in solver_vectors]
            ) if solver_vectors.shape[0] else np.empty((0, layout.n_trial), dtype=np.complex128)
            source_space = "reduced"
        else:
            raise ValueError(
                "Eigensolver vector width does not match the mixed full or reduced "
                f"space: got {solver_vectors.shape[1]}, full={layout.n_trial}, "
                f"reduced={reduced_dofs}."
            )

        count = indices.size

        def coefficients(field_index: int, nradial: int) -> np.ndarray:
            return full_vectors[:, layout.trial_slice(field_index)].reshape(
                count,
                nradial,
                layout.n_modes,
            )

        p1 = coefficients(0, layout.n_high)
        xi_coeff = coefficients(1, layout.n_high)
        xi_s = coefficients(2, layout.n_low)
        div_xi = coefficients(3, layout.n_low)
        xi_psi = np.einsum(
            "qd,edm->eqm",
            mixed.phi_high_q,
            xi_coeff,
            optimize=True,
        )
        roots = np.asarray(result.eigenvalues, dtype=np.complex128)[indices]
        omega = np.sqrt(roots)
        core_owned = _core_ownership_mask(
            omega,
            core_omega_min,
            core_omega_max,
            final_window=final_window,
        )
        return xr.Dataset(
            data_vars={
                "s": ("eigenpair", roots),
                "omega": ("eigenpair", omega),
                "eigenvector": (("eigenpair", "dof"), full_vectors),
                "p1_coefficient": (
                    ("eigenpair", "radial_dof_high", "m_pol"),
                    p1,
                ),
                "xi_psi_coefficient": (
                    ("eigenpair", "radial_dof_high", "m_pol"),
                    xi_coeff,
                ),
                "xi_s_coefficient": (
                    ("eigenpair", "radial_dof_low", "m_pol"),
                    xi_s,
                ),
                "div_xi_coefficient": (
                    ("eigenpair", "radial_dof_low", "m_pol"),
                    div_xi,
                ),
                "xi_psi": (("eigenpair", "psi", "m_pol"), xi_psi),
                "residual_norm": (
                    "eigenpair",
                    self._metric_values(pep, "last_residual_norms", indices),
                ),
                "absolute_residual_norm": (
                    "eigenpair",
                    self._metric_values(
                        pep,
                        "last_absolute_residual_norms",
                        indices,
                    ),
                ),
                "backward_error": (
                    "eigenpair",
                    self._metric_values(pep, "last_backward_errors", indices),
                ),
                "slepc_error_estimate": (
                    "eigenpair",
                    self._metric_values(
                        pep,
                        "last_slepc_error_estimates",
                        indices,
                    ),
                ),
                "slepc_backward_error": (
                    "eigenpair",
                    self._metric_values(
                        pep,
                        "last_slepc_backward_errors",
                        indices,
                    ),
                ),
                "window_id": ("eigenpair", np.full(count, window, dtype=np.int64)),
                "window_depth": ("eigenpair", np.full(count, depth, dtype=np.int64)),
                "core_owned": ("eigenpair", core_owned),
            },
            coords={
                "eigenpair": np.arange(count, dtype=np.int64),
                "dof": np.arange(layout.n_trial, dtype=np.int64),
                "radial_dof_high": np.arange(layout.n_high, dtype=np.int64),
                "radial_dof_low": np.arange(layout.n_low, dtype=np.int64),
                "psi": self.master_points,
                "m_pol": np.asarray(sb.m_range),
            },
            attrs={
                "level": int(level),
                "window": int(window),
                "split_variable": "omega2",
                "omega_normalization": "omegaA",
                "omega_is_species_mass_normalized": True,
                "density_includes_species_mass": True,
                "species_mass_amu": self.solver._species_mass_amu(),
                "core_omega_min": core_omega_min,
                "core_omega_max": core_omega_max,
                "solve_omega_min": solve_omega_min,
                "solve_omega_max": solve_omega_max,
                "omega_min": solve_omega_min,
                "omega_max": solve_omega_max,
                "nconv_raw": int(result.nconv),
                "iterations": int(result.iterations),
                "reason": str(result.reason),
                "slepc_eigenvector_space": source_space,
                "n_radial_high": layout.n_high,
                "n_radial_low": layout.n_low,
                "ntor": int(sb.ntor),
                "mmax": int(getattr(sb, "mmax", np.max(np.abs(sb.m_range)))),
                "max_dm": int(sb.max_dm),
                "eigensolver_backend": getattr(pep, "backend", "PEP"),
                "gamma_zero_reduction": getattr(pep, "backend", "PEP") == "EPS",
            },
        )

    def _search_level(
        self,
        *,
        level: int,
        matrices: tuple,
        sb,
        mixed: MixedRadialData,
        omega_range: tuple[float, float],
        search: MultigridSearchOptions,
        bc_tol: float | None,
        verbose: bool,
        target_offset: float = 0.0,
    ) -> tuple[xr.Dataset, list[_WindowRecord], bool]:
        T0, T1, Tminus1 = matrices
        windows = _fixed_frequency_windows(
            omega_range,
            search.window_count,
            search.window_padding,
        )
        records: list[_WindowRecord] = []
        window_datasets: list[xr.Dataset] = []
        pep = None
        try:
            for window, bounds in enumerate(windows):
                (
                    core_omega_min,
                    core_omega_max,
                    solve_omega_min,
                    solve_omega_max,
                ) = (float(value) for value in bounds)
                target_s, region = self._window_region(
                    solve_omega_min,
                    solve_omega_max,
                    search,
                )
                if target_offset:
                    target_omega = 0.5 * (core_omega_min + core_omega_max)
                    target_omega += float(target_offset) * (
                        core_omega_max - core_omega_min
                    )
                    target_s = complex(target_omega**2)
                if pep is None:
                    pep = self._build_window_eigensolver(
                        matrices=(T0, T1, Tminus1),
                        target_s=target_s,
                        region=region,
                        prefix=f"{search.prefix}_l{level}_",
                        search=search,
                        verbose=verbose,
                    )
                else:
                    pep.retarget(target_s, region)
                result = pep.solve(allow_empty=True)
                indices = self._root_indices_in_window(
                    result.eigenvalues,
                    solve_omega_min,
                    solve_omega_max,
                    region,
                    search.root_filter_tolerance,
                )
                packed = self._pack_window_result(
                    sb=sb,
                    mixed=mixed,
                    result=result,
                    pep=pep,
                    indices=indices,
                    level=level,
                    window=window,
                    depth=0,
                    core_omega_min=core_omega_min,
                    core_omega_max=core_omega_max,
                    solve_omega_min=solve_omega_min,
                    solve_omega_max=solve_omega_max,
                    final_window=window == len(windows) - 1,
                    bc_tol=bc_tol,
                )
                window_datasets.append(packed)
                effective_nev = int(getattr(pep, "effective_nev", search.nev))
                if effective_nev < 1:
                    raise RuntimeError(
                        f"SLEPc reported an invalid effective nev={effective_nev}."
                    )
                reason_converged = _is_converged_reason(result.reason)
                catalog_status = window_catalog_status(
                    nconv=int(result.nconv),
                    effective_nev=effective_nev,
                    solver_converged=reason_converged,
                )
                saturated = catalog_status == "capacity_limited"
                # Targeted shift-invert has no all-roots-in-interval guarantee.
                complete = catalog_status == "certified_complete"
                records.append(
                    _WindowRecord(
                        level=level,
                        window=window,
                        parent=-1,
                        depth=0,
                        core_omega_min=core_omega_min,
                        core_omega_max=core_omega_max,
                        solve_omega_min=solve_omega_min,
                        solve_omega_max=solve_omega_max,
                        target_s=target_s,
                        nconv=int(result.nconv),
                        nreturned=int(indices.size),
                        saturated=saturated,
                        leaf=True,
                        complete=complete,
                        reason=str(result.reason),
                        iterations=int(result.iterations),
                        effective_nev=effective_nev,
                        catalog_status=catalog_status,
                    )
                )
        finally:
            if pep is not None:
                pep.destroy()

        combined = xr.concat(
            window_datasets,
            dim="eigenpair",
            data_vars="all",
            coords="minimal",
            compat="override",
            combine_attrs="drop_conflicts",
        ).assign_coords(
            eigenpair=np.arange(
                sum(
                    dataset.sizes.get("eigenpair", 0)
                    for dataset in window_datasets
                ),
                dtype=np.int64,
            )
        )
        combined = _deduplicate_window_dataset(
            combined,
            self.master_weights,
            relative_s=search.dedup_relative_s,
            minimum_overlap=search.dedup_overlap,
        )
        level_catalog_status = aggregate_catalog_status(
            record.catalog_status for record in records
        )
        complete = level_catalog_status == "certified_complete"
        combined.attrs.update(
            {
                "level": level,
                "search_complete": bool(complete),
                "catalog_status": level_catalog_status,
                "catalog_completion_semantics": (
                    "targeted search; capacity_limited is not solver failure and "
                    "partial is not certified interval completeness"
                ),
                "window_count": len(records),
                "leaf_window_count": len(records),
                "ancestor_fallback_window_count": 0,
                "saturated_leaf_count": sum(record.saturated for record in records),
                "search_strategy": "fixed_padded_windows",
                "window_padding_fraction": float(search.window_padding),
                "quadrature_source_level": self.levels - 1,
                "target_offset_window_fraction": float(target_offset),
            }
        )
        return combined, records, bool(complete)

    def _adaptive_discovery_level(
        self,
        *,
        level: int,
        matrices: tuple,
        sb,
        mixed: MixedRadialData,
        omega_range: tuple[float, float],
        search: MultigridSearchOptions,
        discovery: AdaptiveDiscoveryOptions,
        bc_tol: float | None,
        verbose: bool,
    ) -> tuple[xr.Dataset, list[_WindowRecord], bool]:
        """Run K/offset discovery passes and stop on global-candidate stability."""
        pass_datasets: list[xr.Dataset] = []
        records: list[_WindowRecord] = []
        previous_signature = None
        stable_count = 0
        last_record_start = 0
        for pass_index, nev in enumerate(discovery.nev_passes):
            offset_datasets = []
            last_record_start = len(records)
            for offset_index, offset in enumerate(discovery.target_offsets):
                dataset, offset_records, _ = self._search_level(
                    level=level,
                    matrices=matrices,
                    sb=sb,
                    mixed=mixed,
                    omega_range=omega_range,
                    search=replace(search, nev=int(nev)),
                    bc_tol=bc_tol,
                    verbose=verbose,
                    target_offset=float(offset),
                )
                unique_offset = (
                    pass_index * len(discovery.target_offsets) + offset_index
                ) * search.window_count
                if "window_id" in dataset:
                    dataset = dataset.copy()
                    dataset["window_id"] = dataset["window_id"] + unique_offset
                offset_records = [
                    replace(record, window=record.window + unique_offset)
                    for record in offset_records
                ]
                offset_datasets.append(dataset)
                records.extend(offset_records)
            pass_dataset = xr.concat(
                offset_datasets,
                dim="eigenpair",
                data_vars="all",
                coords="minimal",
                compat="override",
                combine_attrs="drop_conflicts",
            ).assign_coords(
                eigenpair=np.arange(
                    sum(item.sizes.get("eigenpair", 0) for item in offset_datasets)
                )
            )
            pass_dataset = _deduplicate_window_dataset(
                pass_dataset,
                self.master_weights,
                relative_s=search.dedup_relative_s,
                minimum_overlap=search.dedup_overlap,
            )
            breakpoints = np.unique(np.asarray(self.radial_levels[level].knots))
            h = float(np.median(np.diff(breakpoints)))
            global_mask = np.zeros(pass_dataset.sizes.get("eigenpair", 0), dtype=bool)
            for index in range(global_mask.size):
                _, width, _ = _shape_metrics(
                    pass_dataset["xi_psi"].values[index],
                    self.master_points,
                    self.master_weights,
                )
                retention = fixed_coarse_space_retention(
                    pass_dataset["xi_psi"].values[index],
                    self.master_points,
                    self.master_weights,
                    self.radial_levels[0],
                )
                global_mask[index] = bool(
                    (np.isfinite(width) and width / h >= discovery.global_width_over_h_min)
                    or (
                        np.isfinite(retention)
                        and retention >= discovery.global_coarse_retention_min
                    )
                )
            signature = np.asarray(
                pass_dataset["omega"].values.real[global_mask],
                dtype=float,
            )
            if previous_signature is not None and empirical_candidate_stability(
                previous_signature,
                signature,
                relative_gate=search.dedup_relative_s,
            ):
                stable_count += 1
            else:
                stable_count = 0
            previous_signature = signature
            pass_datasets.append(pass_dataset)
            if stable_count >= discovery.stable_passes:
                break

        combined = xr.concat(
            pass_datasets,
            dim="eigenpair",
            data_vars="all",
            coords="minimal",
            compat="override",
            combine_attrs="drop_conflicts",
        ).assign_coords(
            eigenpair=np.arange(
                sum(item.sizes.get("eigenpair", 0) for item in pass_datasets)
            )
        )
        combined = _deduplicate_window_dataset(
            combined,
            self.master_weights,
            relative_s=search.dedup_relative_s,
            minimum_overlap=search.dedup_overlap,
        )
        stable = stable_count >= discovery.stable_passes
        for index, record in enumerate(records):
            leaf = index >= last_record_start
            status = "empirically_stable" if stable and leaf else record.catalog_status
            records[index] = replace(
                record,
                leaf=leaf,
                complete=False,
                catalog_status=status,
            )
        combined.attrs.update(
            search_strategy="adaptive_discovery",
            discovery_nev_passes=repr(discovery.nev_passes),
            discovery_target_offsets=repr(discovery.target_offsets),
            physical_candidate_catalog_stable=bool(stable),
            catalog_status="empirically_stable"
            if stable
            else aggregate_catalog_status(
                record.catalog_status for record in records if record.leaf
            ),
        )
        return combined, records, False

    def _reacquire_missing_roots(
        self,
        *,
        tracks: xr.Dataset,
        level_solutions: list[xr.Dataset],
        projected: tuple[tuple, ...],
        sb,
        search: MultigridSearchOptions,
        thresholds: ModeClassificationThresholds,
        options: ReacquisitionOptions,
        bc_tol: float | None,
        verbose: bool,
        candidate_mask: np.ndarray | None = None,
        physical_metric: PhysicalDisplacementMetric | None = None,
    ) -> tuple[xr.Dataset, list[_WindowRecord]]:
        """Selectively continue global candidates missing on an adjacent mesh."""
        if not options.enabled or self.levels < 2:
            return tracks, []
        if candidate_mask is not None:
            candidate_mask = np.asarray(candidate_mask, dtype=bool)
            if candidate_mask.shape != (tracks.sizes.get("track", 0),):
                raise ValueError(
                    "candidate_mask must contain one Boolean value per input track"
                )

        eligible_roots: set[tuple[str, int, int]] | None = None
        if candidate_mask is not None:
            eligible_roots = _track_root_identities(
                tracks,
                level_solutions,
                candidate_mask,
            )

        def current_candidate_mask(current_tracks: xr.Dataset) -> np.ndarray | None:
            if eligible_roots is None:
                return None
            return _track_mask_from_root_identities(
                current_tracks,
                level_solutions,
                eligible_roots,
            )

        attempt_records: list[_WindowRecord] = []
        negative_status: list[tuple[int, int, str]] = []
        for level in range(1, self.levels):
            present = np.asarray(tracks["present"].values, dtype=bool)
            candidates = select_reacquisition_candidates(
                tracks,
                level,
                options,
                eligible=current_candidate_mask(tracks),
            )
            if candidates.size == 0:
                continue
            T0, T1, Tminus1 = projected[level]
            transfer = mixed_space_prolongation(
                self.mixed_levels[level - 1],
                self.mixed_levels[level],
                tracks.sizes.get("m_pol", 0),
            )
            fine_bc = self.solver._build_nep_boundary_condition(
                sb,
                bc_kind="full",
                mixed=self.mixed_levels[level],
            )
            fine_prolongation = fine_bc.get_sparse_prolongation(tol=bc_tol)
            free = np.asarray(fine_bc.sparse_free, dtype=int)
            local_search = replace(
                search,
                nev=max(int(search.nev) * options.nev_multiplier, int(search.nev) + 4),
            )
            pep = None
            additions: list[xr.Dataset] = []
            try:
                for ordinal, track in enumerate(candidates):
                    previous_index = int(tracks["mode_index"].values[track, level - 1])
                    previous_omega = complex(
                        tracks["omega"].values[track, level - 1]
                    )
                    predicted = previous_omega.real
                    if level >= 2 and present[track, level - 2]:
                        coarser = complex(tracks["omega"].values[track, level - 2])
                        predicted += 0.5 * (previous_omega.real - coarser.real)
                    half_width = max(
                        abs(predicted) * options.relative_half_width,
                        np.finfo(float).eps,
                    )
                    omega_min = max(np.finfo(float).tiny, predicted - half_width)
                    omega_max = predicted + half_width
                    target_s, region = self._window_region(
                        omega_min,
                        omega_max,
                        local_search,
                    )
                    if pep is None:
                        pep = self._build_window_eigensolver(
                            matrices=(T0, T1, Tminus1),
                            target_s=target_s,
                            region=region,
                            prefix=f"{search.prefix}_reacq_l{level}_",
                            search=local_search,
                            verbose=verbose,
                        )
                    else:
                        pep.retarget(target_s, region)
                    full_previous = np.asarray(
                        level_solutions[level - 1]["eigenvector"].values[
                            previous_index
                        ],
                        dtype=np.complex128,
                    )
                    full_initial = transfer @ full_previous
                    reduced_initial = np.asarray(full_initial[free], dtype=np.complex128)
                    constraint_error = np.linalg.norm(
                        fine_prolongation @ reduced_initial - full_initial
                    ) / max(np.linalg.norm(full_initial), np.finfo(float).tiny)
                    if (
                        options.use_initial_space
                        and constraint_error <= 1.0e-8
                        and hasattr(pep, "set_initial_space")
                    ):
                        pep.set_initial_space(reduced_initial)
                    result = pep.solve(allow_empty=True)
                    indices = self._root_indices_in_window(
                        result.eigenvalues,
                        omega_min,
                        omega_max,
                        region,
                        search.root_filter_tolerance,
                    )
                    window = 1_000_000 + level * 100_000 + int(track)
                    packed = self._pack_window_result(
                        sb=sb,
                        mixed=self.mixed_levels[level],
                        result=result,
                        pep=pep,
                        indices=indices,
                        level=level,
                        window=window,
                        depth=-1,
                        core_omega_min=omega_min,
                        core_omega_max=omega_max,
                        solve_omega_min=omega_min,
                        solve_omega_max=omega_max,
                        final_window=True,
                        bc_tol=bc_tol,
                    )
                    effective_nev = int(getattr(pep, "effective_nev", local_search.nev))
                    status = window_catalog_status(
                        nconv=int(result.nconv),
                        effective_nev=effective_nev,
                        solver_converged=_is_converged_reason(result.reason),
                    )
                    best = -1
                    best_overlap = -np.inf
                    previous_shape = np.asarray(
                        tracks["xi_psi"].values[track, level - 1],
                        dtype=np.complex128,
                    )
                    packed_omega = np.asarray(
                        packed["omega"].values,
                        dtype=np.complex128,
                    )
                    packed_error = _best_error(packed)
                    imaginary_ratio = np.abs(packed_omega.imag) / np.maximum(
                        np.abs(packed_omega.real),
                        np.finfo(float).tiny,
                    )
                    algebraically_valid = (
                        np.isfinite(packed_error)
                        & (packed_error <= thresholds.backward_error)
                        & np.isfinite(imaginary_ratio)
                        & (
                            imaginary_ratio
                            <= thresholds.imaginary_frequency_ratio
                        )
                    )
                    for candidate in range(packed.sizes.get("eigenpair", 0)):
                        if not algebraically_valid[candidate]:
                            continue
                        overlap = _weighted_xi_overlap(
                            previous_shape,
                            packed["xi_psi"].values[candidate],
                            self.master_weights,
                        )
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best = candidate
                    if best >= 0 and best_overlap >= options.minimum_overlap:
                        selected = packed.isel(eigenpair=[best]).assign_coords(eigenpair=[0])
                        selected["reacquisition_status"] = (
                            "eigenpair",
                            np.asarray(["reacquired"], dtype="U48"),
                        )
                        selected["reacquisition_overlap"] = (
                            "eigenpair",
                            np.asarray([best_overlap]),
                        )
                        selected["source_eigenpair"] = (
                            "eigenpair",
                            np.asarray([np.nan]),
                        )
                        additions.append(selected)
                        reacquisition_status = "reacquired"
                    else:
                        if status == "capacity_limited":
                            reacquisition_status = "capacity_limited"
                        elif status == "solver_failed":
                            reacquisition_status = "search_failed"
                        else:
                            reacquisition_status = "partial"
                        negative_status.append(
                            (level, previous_index, reacquisition_status)
                        )
                    attempt_records.append(
                        _WindowRecord(
                            level=level,
                            window=window,
                            parent=int(track),
                            depth=-1,
                            core_omega_min=omega_min,
                            core_omega_max=omega_max,
                            solve_omega_min=omega_min,
                            solve_omega_max=omega_max,
                            target_s=target_s,
                            nconv=int(result.nconv),
                            nreturned=int(indices.size),
                            saturated=status == "capacity_limited",
                            leaf=False,
                            complete=False,
                            reason=f"{result.reason};reacquisition={reacquisition_status}",
                            iterations=int(result.iterations),
                            effective_nev=effective_nev,
                            catalog_status=status,
                        )
                    )
            finally:
                if pep is not None:
                    pep.destroy()
            if additions:
                if "reacquisition_status" not in level_solutions[level]:
                    level_solutions[level] = level_solutions[level].assign(
                        reacquisition_status=(
                            "eigenpair",
                            np.full(
                                level_solutions[level].sizes.get("eigenpair", 0),
                                "not_attempted",
                                dtype="U48",
                            ),
                        )
                    )
                level_solutions[level] = xr.concat(
                    [level_solutions[level], *additions],
                    dim="eigenpair",
                    data_vars="all",
                    coords="minimal",
                    compat="override",
                    combine_attrs="drop_conflicts",
                ).assign_coords(
                    eigenpair=np.arange(
                        level_solutions[level].sizes.get("eigenpair", 0)
                        + len(additions)
                    )
                )
                level_solutions[level] = _deduplicate_window_dataset(
                    level_solutions[level],
                    self.master_weights,
                    relative_s=search.dedup_relative_s,
                    minimum_overlap=search.dedup_overlap,
                    physical_metric=physical_metric,
                    tangential_fields=_reconstruct_tangential_displacement(
                        level_solutions[level],
                        self.mixed_levels[level],
                    ) if physical_metric is not None else None,
                )
                tracks = _track_level_solutions(
                    tuple(level_solutions),
                    self.master_weights,
                    thresholds,
                    radial_levels=self.radial_levels,
                    mixed_levels=self.mixed_levels,
                    physical_metric=physical_metric,
                )

        if negative_status:
            status_values = np.asarray(tracks["reacquisition_status"].values).astype("U48")
            present = np.asarray(tracks["present"].values, dtype=bool)
            mode_index = np.asarray(tracks["mode_index"].values, dtype=int)
            for level, previous_index, status in negative_status:
                candidates = np.flatnonzero(
                    present[:, level - 1]
                    & ~present[:, level]
                    & (mode_index[:, level - 1] == previous_index)
                )
                if candidates.size == 0:
                    continue
                status_values[int(candidates[0]), level] = status
            tracks = tracks.copy()
            tracks["reacquisition_status"] = (
                ("track", "level"),
                status_values,
            )
        return tracks, attempt_records

    def solve_pep(
        self,
        ntor: int,
        omega_range,
        *,
        mmax: int | None = None,
        max_dm: int = 10,
        regularizer: float = 0.0,
        apply_bcs: bool = True,
        bc_kind: str | None = "full",
        bc_tol: float | None = None,
        search: MultigridSearchOptions | Mapping[str, object] | None = None,
        thresholds: ModeClassificationThresholds | Mapping[str, object] | None = None,
        classification_version: str = CLASSIFICATION_VERSION,
        evidence_config: EvidenceV2Config | Mapping[str, object] | None = None,
        adaptive_discovery: AdaptiveDiscoveryOptions | Mapping[str, object] | None = None,
        reacquisition: ReacquisitionOptions | Mapping[str, object] | None = None,
        verbose: bool = False,
    ) -> MultigridPEPResult:
        """Search a normalized-omega band and classify all multigrid tracks."""
        self.solver._require_full_bc_nep(apply_bcs, bc_kind)
        omega_range = _validate_omega_range(omega_range)
        search = _coerce_options(search, MultigridSearchOptions)
        thresholds = _coerce_options(thresholds, ModeClassificationThresholds)
        evidence_config = _coerce_options(evidence_config, EvidenceV2Config)
        discovery = (
            None
            if adaptive_discovery is None
            else _coerce_options(adaptive_discovery, AdaptiveDiscoveryOptions)
        )
        reacquisition_options = (
            None
            if reacquisition is None
            else _coerce_options(reacquisition, ReacquisitionOptions)
        )

        # These are deliberately the only local block/tensor construction calls.
        sb = self.solver.build_solver_blocks(
            ntor=ntor,
            mmax=mmax,
            max_dm=max_dm,
        )
        physical_metric = None
        physical_metric_fallback_reason = "none"
        try:
            physical_metric = build_resolved_kinetic_energy_metric(
                self.solver,
                self.master_points,
                np.asarray(sb.m_range),
            )
        except (FloatingPointError, ValueError) as error:
            physical_metric_fallback_reason = f"{type(error).__name__}: {error}"
        local_split = _build_local_M_split(sb, regularizer=regularizer)
        projected = tuple(
            self.solver._assemble_split_matrices_from_local(
                sb,
                mixed,
                local_split,
                apply_bcs=apply_bcs,
                bc_kind=bc_kind,
                bc_tol=bc_tol,
                verbose=verbose,
            )
            for mixed in self.mixed_levels
        )

        level_solutions: list[xr.Dataset] = []
        records: list[_WindowRecord] = []
        complete: list[bool] = []
        for level, (matrices, mixed) in enumerate(zip(projected, self.mixed_levels)):
            use_discovery = discovery is not None and level < min(2, self.levels)
            search_method = self._adaptive_discovery_level if use_discovery else self._search_level
            search_kwargs = dict(
                level=level,
                matrices=matrices,
                sb=sb,
                mixed=mixed,
                omega_range=omega_range,
                search=search,
                bc_tol=bc_tol,
                verbose=verbose,
            )
            if use_discovery:
                search_kwargs["discovery"] = discovery
            dataset, level_records, level_complete = search_method(**search_kwargs)
            dataset.attrs.update(
                {
                    "regularizer": float(regularizer),
                    "apply_bcs": bool(apply_bcs),
                    "bc_kind": str(bc_kind),
                    "bc_tol": None if bc_tol is None else float(bc_tol),
                    "search": repr(search),
                    "search_requested_nev": int(search.nev),
                    "search_tolerance": float(search.tol),
                    "search_max_iterations": int(search.maxit),
                    "search_window_count": int(search.window_count),
                    "search_window_padding": float(search.window_padding),
                    "search_region_vscale": float(search.region_vscale),
                    "search_strategy": (
                        "adaptive_discovery" if use_discovery else "fixed_padded_windows"
                    ),
                    "adaptive_discovery": repr(discovery),
                }
            )
            level_solutions.append(dataset)
            records.extend(level_records)
            complete.append(level_complete)

        continuum = _solve_master_continuum(
            sb,
            np.asarray(self.solver.unique_points, dtype=np.float64),
            ambiguity_delta=thresholds.assignment_ambiguity,
        )
        continuum.attrs.update(
            {
                "omega_normalization": "omegaA",
                "omega_is_species_mass_normalized": True,
                "density_includes_species_mass": True,
                "species_mass_amu": self.solver._species_mass_amu(),
            }
        )
        level_tuple = tuple(level_solutions)
        tracks = _track_level_solutions(
            level_tuple,
            self.master_weights,
            thresholds,
            radial_levels=getattr(self, "radial_levels", None),
            mixed_levels=getattr(self, "mixed_levels", None),
            physical_metric=physical_metric,
        )
        tracks.attrs["overlap_metric_fallback_reason"] = (
            physical_metric_fallback_reason
        )
        if reacquisition_options is not None:
            tracks, reacquisition_records = self._reacquire_missing_roots(
                tracks=tracks,
                level_solutions=level_solutions,
                projected=projected,
                sb=sb,
                search=search,
                thresholds=thresholds,
                options=reacquisition_options,
                bc_tol=bc_tol,
                verbose=verbose,
                physical_metric=physical_metric,
            )
            records.extend(reacquisition_records)
            level_tuple = tuple(level_solutions)
        tracks = _attach_enriched_residuals(
            tracks,
            level_tuple,
            projected,
            self.mixed_levels,
            self.solver,
            sb,
            bc_tol=bc_tol,
        )
        level_catalog_status = np.asarray(
            [
                aggregate_catalog_status(
                    record.catalog_status
                    for record in records
                    if record.level == level and record.leaf
                )
                for level in range(self.levels)
            ],
            dtype="U24",
        )
        classification = _classify_tracks(
            tracks,
            continuum,
            self.master_weights,
            np.asarray(complete, dtype=bool),
            thresholds,
            classification_version=classification_version,
            evidence_config=evidence_config,
            catalog_status=level_catalog_status,
        )
        diagnostics = _window_diagnostics_dataset(records, self.levels)
        diagnostics["level_complete"] = (
            "level",
            np.asarray(complete, dtype=bool),
        )
        diagnostics["level_catalog_status"] = (
            "level",
            level_catalog_status,
        )
        diagnostics.attrs.update(
            {
                "search": repr(search),
                "omega_min": omega_range[0],
                "omega_max": omega_range[1],
                "run_catalog_status": aggregate_catalog_status(level_catalog_status),
                "classification_version": str(classification_version),
                "reacquisition": repr(reacquisition_options),
                "overlap_metric": (
                    physical_metric.metric_name
                    if physical_metric is not None
                    else "xi_psi_proxy"
                ),
                "overlap_metric_fallback_reason": (
                    physical_metric_fallback_reason
                ),
                "spectrum_slicing_applicable": False,
                "spectrum_slicing_reason": (
                    "solver5 currently declares a GENERAL quadratic PEP; "
                    "Hermitian/hyperbolic structure and definiteness have not "
                    "been certified, so all-in-interval slicing is disabled"
                ),
            }
        )
        return MultigridPEPResult(
            level_solutions=level_tuple,
            mode_tracks=tracks,
            window_diagnostics=diagnostics,
            continuum=continuum,
            classification=classification,
        )


__all__ = [
    "EvidenceV2Config",
    "ModeClassificationThresholds",
    "MultigridPEPResult",
    "MultigridSearchOptions",
    "build_resolved_kinetic_energy_metric",
    "superSolver5Multigrid",
]
