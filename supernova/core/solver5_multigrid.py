r"""Nested compatible-B-spline multigrid classification for solver5.

The orchestrator in this module builds the point-local ideal-MHD operator once
on a finest composite Gauss rule.  Three (or more) knot-insertion-nested radial
spaces then differ only in Galerkin projection, boundary reduction, and their
quadratic PEP solves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import operator
from typing import Mapping

import numpy as np
import xarray as xr
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment

from ..radial import BSplineFE
from ._petsc_helpers import NEPResult
from ._solver_kernel2 import mu_0
from .solver3 import _build_local_M_split
from .solver4 import MixedRadialData, _build_mixed_radial_data
from .solver5 import QuadraticPEPSolver, superSolver5


_LABELS = (
    "physical_gap",
    "continuum_resonance",
    "numerical_artifact",
    "inconclusive",
)


@dataclass(frozen=True)
class MultigridSearchOptions:
    """Adaptive TOAR/SINVERT frequency-window search controls."""

    nev: int = 8
    tol: float = 1.0e-8
    maxit: int = 200
    window_overlap: float = 0.10
    max_depth: int = 8
    dedup_relative_s: float = 1.0e-6
    dedup_overlap: float = 0.99
    region_vscale: float = 0.10
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
        if not 0.0 <= self.window_overlap < 1.0:
            raise ValueError("window_overlap must lie in [0, 1).")
        if int(self.max_depth) < 0:
            raise ValueError("max_depth must be non-negative.")
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
    omega_min: float
    omega_max: float
    target_s: complex
    nconv: int
    nreturned: int
    saturated: bool
    leaf: bool
    complete: bool
    reason: str
    iterations: int
    effective_nev: int = -1


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
) -> xr.Dataset:
    """Deduplicate only cross-window roots meeting both spectral/shape gates."""
    size = dataset.sizes.get("eigenpair", 0)
    if size < 2:
        return dataset.assign_coords(eigenpair=np.arange(size, dtype=np.int64))
    roots = np.asarray(dataset["s"].values, dtype=np.complex128)
    windows = np.asarray(dataset["window_id"].values, dtype=np.int64)
    fields = np.asarray(dataset["xi_psi"].values, dtype=np.complex128)
    quality = _best_error(dataset)
    order = np.arange(size, dtype=np.int64)
    order = order[
        np.lexsort(
            (
                np.nan_to_num(roots.imag[order], nan=np.inf),
                np.nan_to_num(roots.real[order], nan=np.inf),
                np.nan_to_num(quality[order], nan=np.inf),
            )
        )
    ]
    keep: list[int] = []
    for index in order:
        duplicate = False
        for retained in keep:
            if windows[index] == windows[retained]:
                continue
            if _relative_complex_distance(roots[index], roots[retained]) > relative_s:
                continue
            overlap = _weighted_xi_overlap(
                fields[index],
                fields[retained],
                weights,
            )
            if overlap >= minimum_overlap:
                duplicate = True
                break
        if not duplicate:
            keep.append(int(index))
    keep_array = np.asarray(keep, dtype=np.int64)
    if keep_array.size:
        omega = np.asarray(dataset["omega"].values)[keep_array]
        keep_array = keep_array[np.argsort(omega.real)]
    return dataset.isel(eigenpair=keep_array).assign_coords(
        eigenpair=np.arange(keep_array.size, dtype=np.int64)
    )


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


def _track_level_solutions(
    level_solutions: tuple[xr.Dataset, ...],
    weights: np.ndarray,
    thresholds: ModeClassificationThresholds,
) -> xr.Dataset:
    """Build adjacent-level maximum-cardinality mode chains."""
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

    tracks: list[list[int]] = [
        [index] + [-1] * (nlevels - 1)
        for index in range(level_solutions[0].sizes.get("eigenpair", 0))
    ]
    transition_overlap: dict[tuple[int, int], float] = {}
    transition_frequency: dict[tuple[int, int], float] = {}
    transition_ambiguity: dict[tuple[int, int], bool] = {}

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
        overlaps = np.zeros(
            (reference_indices.size, candidate_indices.size),
            dtype=np.float64,
        )
        for row in range(reference_indices.size):
            for column in range(candidate_indices.size):
                overlaps[row, column] = _weighted_xi_overlap(
                    reference_xi[row],
                    candidate_xi[column],
                    weights,
                )
        rows, columns, frequency_change, ambiguous = (
            _maximum_cardinality_overlap_assignment(
                reference_omega.real,
                candidate_omega.real,
                overlaps,
                frequency_gate=thresholds.frequency_gate,
                ambiguity_delta=thresholds.assignment_ambiguity,
            )
        )
        reference_ambiguous, candidate_ambiguous = _assignment_ambiguity_masks(
            reference_omega.real,
            candidate_omega.real,
            overlaps,
            rows,
            columns,
            frequency_gate=thresholds.frequency_gate,
            ambiguity_delta=thresholds.assignment_ambiguity,
        )
        for row, is_ambiguous in enumerate(reference_ambiguous):
            if is_ambiguous:
                transition_ambiguity[(reference_tracks[row], level - 1)] = True
        matched_candidates: set[int] = set()
        candidate_tracks: dict[int, int] = {}
        for pair, (row, column) in enumerate(zip(rows, columns)):
            track_index = reference_tracks[int(row)]
            candidate_index = int(candidate_indices[column])
            tracks[track_index][level] = candidate_index
            matched_candidates.add(candidate_index)
            candidate_tracks[candidate_index] = track_index
            transition_overlap[(track_index, level)] = float(overlaps[row, column])
            transition_frequency[(track_index, level)] = float(
                frequency_change[pair]
            )
            transition_ambiguity[(track_index, level)] = bool(ambiguous[pair])

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
    width = np.full((ntracks, nlevels), np.nan)
    ipr = np.full((ntracks, nlevels), np.nan)
    overlap = np.full((ntracks, nlevels), np.nan)
    frequency_change = np.full((ntracks, nlevels), np.nan)
    ambiguous = np.zeros((ntracks, nlevels), dtype=bool)
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
            peak[track_index, level], width[track_index, level], ipr[
                track_index, level
            ] = _shape_metrics(xi[track_index, level], psi, weights)
            overlap[track_index, level] = transition_overlap.get(
                (track_index, level), np.nan
            )
            frequency_change[track_index, level] = transition_frequency.get(
                (track_index, level), np.nan
            )
            ambiguous[track_index, level] = transition_ambiguity.get(
                (track_index, level), False
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
            "peak_radius": (("track", "level"), peak),
            "rms_width": (("track", "level"), width),
            "ipr": (("track", "level"), ipr),
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
                "phase-invariant harmonic-resolved composite-Gauss dpsi L2"
            ),
            "assignment_priority": "maximum cardinality, then xi_psi overlap",
        },
    )


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


def _classify_tracks(
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
        for name in ("omega_min", "omega_max")
    }
    target = np.full(shape, np.nan + 1.0j * np.nan, dtype=np.complex128)
    boolean = {
        name: np.zeros(shape, dtype=bool)
        for name in ("saturated", "leaf", "complete")
    }
    reason = np.full(shape, "", dtype="U64")
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
            floating["omega_min"][level, column] = record.omega_min
            floating["omega_max"][level, column] = record.omega_max
            target[level, column] = record.target_s
            boolean["saturated"][level, column] = record.saturated
            boolean["leaf"][level, column] = record.leaf
            boolean["complete"][level, column] = record.complete
            reason[level, column] = record.reason
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
        omega_min: float,
        omega_max: float,
        bc_tol: float | None,
    ) -> xr.Dataset:
        layout = self.solver._mixed_layout(sb, mixed=mixed)
        solver_vectors = np.asarray(result.eigenvectors, dtype=np.complex128)
        if solver_vectors.ndim != 2:
            raise ValueError("PEP eigenvectors must have shape (eigenpair, dof).")
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
                "PEP eigenvector width does not match the mixed full or reduced "
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
        return xr.Dataset(
            data_vars={
                "s": ("eigenpair", roots),
                "omega": ("eigenpair", np.sqrt(roots)),
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
                "omega_min": omega_min,
                "omega_max": omega_max,
                "nconv_raw": int(result.nconv),
                "iterations": int(result.iterations),
                "reason": str(result.reason),
                "slepc_eigenvector_space": source_space,
                "n_radial_high": layout.n_high,
                "n_radial_low": layout.n_low,
                "ntor": int(sb.ntor),
                "mmax": int(getattr(sb, "mmax", np.max(np.abs(sb.m_range)))),
                "max_dm": int(sb.max_dm),
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
    ) -> tuple[xr.Dataset, list[_WindowRecord], bool]:
        T0, T1, Tminus1 = matrices
        pending = [(omega_range[0], omega_range[1], 0, -1)]
        records: list[_WindowRecord] = []
        window_datasets: dict[int, xr.Dataset] = {}
        pep = None
        next_window = 0
        try:
            while pending:
                omega_min, omega_max, depth, parent = pending.pop()
                window = next_window
                next_window += 1
                target_s, region = self._window_region(omega_min, omega_max, search)
                if pep is None:
                    pep = QuadraticPEPSolver(
                        A0=Tminus1,
                        A1=T0,
                        A2=T1,
                        target_s=target_s,
                        region=region,
                        prefix=f"{search.prefix}_l{level}_",
                        pep_type=search.pep_type,
                        st_type=search.st_type,
                        which=search.which,
                        nev=search.nev,
                        tol=search.tol,
                        maxit=search.maxit,
                        options=dict(search.options),
                        verbose=verbose,
                        compute_residuals=True,
                    )
                else:
                    pep.retarget(target_s, region)
                result = pep.solve(allow_empty=True)
                indices = self._root_indices_in_window(
                    result.eigenvalues,
                    omega_min,
                    omega_max,
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
                    depth=depth,
                    omega_min=omega_min,
                    omega_max=omega_max,
                    bc_tol=bc_tol,
                )
                window_datasets[window] = packed
                effective_nev = int(getattr(pep, "effective_nev", search.nev))
                if effective_nev < 1:
                    raise RuntimeError(
                        f"SLEPc reported an invalid effective nev={effective_nev}."
                    )
                saturated = int(result.nconv) >= effective_nev
                leaf = not saturated or depth >= search.max_depth
                reason_converged = _is_converged_reason(result.reason)
                complete = bool(leaf and not saturated and reason_converged)
                records.append(
                    _WindowRecord(
                        level=level,
                        window=window,
                        parent=parent,
                        depth=depth,
                        omega_min=omega_min,
                        omega_max=omega_max,
                        target_s=target_s,
                        nconv=int(result.nconv),
                        nreturned=int(indices.size),
                        saturated=saturated,
                        leaf=leaf,
                        complete=complete,
                        reason=str(result.reason),
                        iterations=int(result.iterations),
                        effective_nev=effective_nev,
                    )
                )
                if leaf:
                    continue
                midpoint = 0.5 * (omega_min + omega_max)
                half_overlap = 0.5 * search.window_overlap * (omega_max - omega_min)
                left = (omega_min, midpoint + half_overlap, depth + 1, window)
                right = (midpoint - half_overlap, omega_max, depth + 1, window)
                pending.append(right)
                pending.append(left)
        finally:
            if pep is not None:
                pep.destroy()

        leaf_records = [record for record in records if record.leaf]
        if not leaf_records:
            raise RuntimeError("Adaptive search produced no leaf windows.")
        record_by_window = {record.window: record for record in records}
        fallback_windows: set[int] = set()
        for record in leaf_records:
            if record.complete:
                continue
            parent = record.parent
            while parent >= 0:
                fallback_windows.add(parent)
                parent = record_by_window[parent].parent
        retained_windows = {
            record.window for record in leaf_records
        } | fallback_windows
        retained_datasets = [
            window_datasets[record.window]
            for record in records
            if record.window in retained_windows
        ]
        combined = xr.concat(
            retained_datasets,
            dim="eigenpair",
            data_vars="all",
            coords="minimal",
            compat="override",
            combine_attrs="drop_conflicts",
        ).assign_coords(
            eigenpair=np.arange(
                sum(
                    dataset.sizes.get("eigenpair", 0)
                    for dataset in retained_datasets
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
        complete = all(record.complete for record in records if record.leaf)
        combined.attrs.update(
            {
                "level": level,
                "search_complete": bool(complete),
                "window_count": len(records),
                "leaf_window_count": len(leaf_records),
                "ancestor_fallback_window_count": len(fallback_windows),
                "saturated_leaf_count": sum(
                    record.leaf and record.saturated for record in records
                ),
                "quadrature_source_level": self.levels - 1,
            }
        )
        return combined, records, bool(complete)

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
        verbose: bool = False,
    ) -> MultigridPEPResult:
        """Search a normalized-omega band and classify all multigrid tracks."""
        self.solver._require_full_bc_nep(apply_bcs, bc_kind)
        omega_range = _validate_omega_range(omega_range)
        search = _coerce_options(search, MultigridSearchOptions)
        thresholds = _coerce_options(thresholds, ModeClassificationThresholds)

        # These are deliberately the only local block/tensor construction calls.
        sb = self.solver.build_solver_blocks(
            ntor=ntor,
            mmax=mmax,
            max_dm=max_dm,
        )
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
            dataset, level_records, level_complete = self._search_level(
                level=level,
                matrices=matrices,
                sb=sb,
                mixed=mixed,
                omega_range=omega_range,
                search=search,
                bc_tol=bc_tol,
                verbose=verbose,
            )
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
                    "search_window_overlap": float(search.window_overlap),
                    "search_max_depth": int(search.max_depth),
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
        )
        classification = _classify_tracks(
            tracks,
            continuum,
            self.master_weights,
            np.asarray(complete, dtype=bool),
            thresholds,
        )
        diagnostics = _window_diagnostics_dataset(records, self.levels)
        diagnostics["level_complete"] = (
            "level",
            np.asarray(complete, dtype=bool),
        )
        diagnostics.attrs.update(
            {
                "search": repr(search),
                "omega_min": omega_range[0],
                "omega_max": omega_range[1],
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
    "ModeClassificationThresholds",
    "MultigridPEPResult",
    "MultigridSearchOptions",
    "superSolver5Multigrid",
]
