"""Evidence-v2 diagnostics and classification for solver5 multiresolution roots.

This module is deliberately independent of PETSc/SLEPc.  It consumes returned
eigenpairs and their diagnostics; catalog completeness is retained as metadata
and is never used as a veto for positive per-mode evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import xarray as xr
from scipy.interpolate import BSpline
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.sparse import block_diag, csr_matrix, kron


CLASSIFICATION_VERSION = "evidence_v2"

EVIDENCE_LABELS = (
    "solver_rejected",
    "physical_gap",
    "continuum_discretization",
    "continuum_coupled_candidate",
    "discretization_artifact",
    "unresolved",
)

VERIFICATION_LEVELS = (
    "baseline",
    "mesh_robust",
    "regularization_verified",
)

CATALOG_STATUSES = (
    "solver_failed",
    "partial",
    "capacity_limited",
    "empirically_stable",
    "certified_complete",
)

REACQUISITION_STATUSES = (
    "not_attempted",
    "reacquired",
    "search_failed",
    "not_found_in_locally_complete_search",
    "capacity_limited",
)

TRUE = "true"
FALSE = "false"
UNKNOWN = "unknown"


def tri_state(value) -> str:
    """Return the serializable three-valued evidence representation."""
    if value is None:
        return UNKNOWN
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in (TRUE, FALSE, UNKNOWN):
            return normalized
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return UNKNOWN
    return TRUE if bool(value) else FALSE


def tri_and(*values) -> str:
    states = [tri_state(value) for value in values]
    if FALSE in states:
        return FALSE
    if UNKNOWN in states:
        return UNKNOWN
    return TRUE


def tri_or(*values) -> str:
    states = [tri_state(value) for value in values]
    if TRUE in states:
        return TRUE
    if UNKNOWN in states:
        return UNKNOWN
    return FALSE


@dataclass(frozen=True)
class EvidenceV2Config:
    """Configurable gates for evidence calculation and final classification."""

    backward_error_max: float = 1.0e-8
    imaginary_frequency_ratio_max: float = 1.0e-6
    adjacent_frequency_gate: float = 2.0e-2
    fine_frequency_change_max: float = 1.0e-3
    subspace_overlap_min: float = 0.98
    gap_uncertainty_multiplier: float = 3.0

    resonance_peak_distance_h_max: float = 2.0
    resonance_power_radius_h: float = 3.0
    resonance_power_fraction_min: float = 0.50
    continuum_scaling_exponent_min: float = 0.60

    assignment_tie_delta: float = 1.0e-3
    continuum_relevance_relative_band: float = 5.0e-2
    continuum_band_relative: float = 5.0e-3
    minimum_levels: int = 3
    singular_diagnostic_min_count: int = 2
    coarse_retention_min: float = 0.50
    enriched_residual_reduction_max: float = 0.50
    mesh_frequency_change_max: float = 2.0e-3
    mesh_overlap_min: float = 0.95
    mesh_peak_shift_h_max: float = 2.0

    def __post_init__(self) -> None:
        positive = (
            "backward_error_max",
            "imaginary_frequency_ratio_max",
            "adjacent_frequency_gate",
            "fine_frequency_change_max",
            "gap_uncertainty_multiplier",
            "resonance_peak_distance_h_max",
            "resonance_power_radius_h",
            "continuum_scaling_exponent_min",
            "continuum_relevance_relative_band",
            "continuum_band_relative",
            "coarse_retention_min",
            "enriched_residual_reduction_max",
            "mesh_frequency_change_max",
            "mesh_peak_shift_h_max",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in (
            "subspace_overlap_min",
            "assignment_tie_delta",
            "resonance_power_fraction_min",
            "mesh_overlap_min",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        if int(self.minimum_levels) < 2:
            raise ValueError("minimum_levels must be at least two.")
        if int(self.singular_diagnostic_min_count) < 2:
            raise ValueError("singular_diagnostic_min_count must be at least two.")


@dataclass(frozen=True)
class MeshPerturbationOptions:
    """Deterministic same-resolution interior-knot perturbation controls."""

    amplitude_fraction: float = 0.15
    seed: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.amplitude_fraction):
            raise ValueError("amplitude_fraction must be finite.")
        if not 0.0 < self.amplitude_fraction <= 0.25:
            raise ValueError("amplitude_fraction must lie in (0, 0.25].")


@dataclass(frozen=True)
class AdaptiveDiscoveryOptions:
    """Two-pass discovery controls for coarse/medium physical candidates."""

    nev_passes: tuple[int, ...] = (4, 8, 16)
    target_offsets: tuple[float, ...] = (-0.25, 0.0, 0.25)
    stable_passes: int = 2
    global_width_over_h_min: float = 3.0
    global_coarse_retention_min: float = 0.50

    def __post_init__(self) -> None:
        if len(self.nev_passes) < 2 or any(int(value) < 1 for value in self.nev_passes):
            raise ValueError("nev_passes must contain at least two positive values.")
        if any(not -0.5 < float(value) < 0.5 for value in self.target_offsets):
            raise ValueError("target_offsets must lie strictly inside (-0.5, 0.5).")
        if int(self.stable_passes) < 1:
            raise ValueError("stable_passes must be positive.")


@dataclass(frozen=True)
class ReacquisitionOptions:
    """Selective narrow-search controls for missing adjacent-level roots."""

    enabled: bool = True
    max_candidates_per_level: int = 24
    nev_multiplier: int = 4
    relative_half_width: float = 3.0e-3
    minimum_overlap: float = 0.90
    global_width_over_h_min: float = 3.0
    global_coarse_retention_min: float = 0.50
    use_initial_space: bool = True

    def __post_init__(self) -> None:
        if int(self.max_candidates_per_level) < 1:
            raise ValueError("max_candidates_per_level must be positive.")
        if int(self.nev_multiplier) < 2:
            raise ValueError("nev_multiplier must be at least two.")
        if not np.isfinite(self.relative_half_width) or self.relative_half_width <= 0.0:
            raise ValueError("relative_half_width must be finite and positive.")
        if not 0.0 <= self.minimum_overlap <= 1.0:
            raise ValueError("minimum_overlap must lie in [0, 1].")


def empirical_candidate_stability(
    previous_frequency: np.ndarray,
    current_frequency: np.ndarray,
    *,
    relative_gate: float = 1.0e-3,
) -> bool:
    """Return whether a physical/global candidate set gained no new member."""
    previous = np.sort(np.asarray(previous_frequency, dtype=float))
    current = np.sort(np.asarray(current_frequency, dtype=float))
    previous = previous[np.isfinite(previous)]
    current = current[np.isfinite(current)]
    if previous.size != current.size:
        return False
    if previous.size == 0:
        return True
    scale = np.maximum(
        np.maximum(np.abs(previous), np.abs(current)),
        np.finfo(float).tiny,
    )
    return bool(np.all(np.abs(previous - current) / scale <= relative_gate))


def window_catalog_status(
    *,
    nconv: int,
    effective_nev: int,
    solver_converged: bool,
    empirically_stable: bool = False,
    certified_complete: bool = False,
) -> str:
    """Describe a targeted window without claiming false interval completeness."""
    if not solver_converged:
        return "solver_failed"
    if certified_complete:
        return "certified_complete"
    if int(effective_nev) < 1:
        return "solver_failed"
    if int(nconv) >= int(effective_nev):
        return "capacity_limited"
    if empirically_stable:
        return "empirically_stable"
    return "partial"


def aggregate_catalog_status(statuses: Iterable[str]) -> str:
    """Aggregate reporting status while preserving its non-mode-evidence role."""
    values = tuple(str(status) for status in statuses)
    if not values or any(value not in CATALOG_STATUSES for value in values):
        return "partial"
    if any(value == "solver_failed" for value in values):
        return "solver_failed"
    if any(value == "capacity_limited" for value in values):
        return "capacity_limited"
    if all(value == "certified_complete" for value in values):
        return "certified_complete"
    if all(value in ("empirically_stable", "certified_complete") for value in values):
        return "empirically_stable"
    return "partial"


def catalog_status_from_diagnostics(diagnostics: xr.Dataset) -> tuple[np.ndarray, str]:
    """Recover level/run catalog status from new or legacy window diagnostics."""
    nlevels = diagnostics.sizes.get("level", 0)
    if "catalog_status" in diagnostics:
        raw = np.asarray(diagnostics["catalog_status"].values).astype(str)
        valid = (
            np.asarray(diagnostics["valid"].values, dtype=bool)
            if "valid" in diagnostics
            else np.ones(raw.shape, dtype=bool)
        )
        level_status = np.asarray(
            [aggregate_catalog_status(raw[level, valid[level]]) for level in range(nlevels)],
            dtype="U24",
        )
    else:
        valid = (
            np.asarray(diagnostics["valid"].values, dtype=bool)
            if "valid" in diagnostics
            else np.ones((nlevels, diagnostics.sizes.get("window", 0)), dtype=bool)
        )
        nconv = np.asarray(diagnostics.get("nconv", xr.zeros_like(diagnostics["valid"])).values)
        effective = np.asarray(
            diagnostics.get("effective_nev", xr.ones_like(diagnostics["valid"])).values
        )
        if "reason" in diagnostics:
            reason = np.asarray(diagnostics["reason"].values).astype(str)
            converged = np.vectorize(
                lambda text: "converg" in text.lower() or text.strip().lstrip("+").isdigit()
            )(reason)
        else:
            converged = np.ones(valid.shape, dtype=bool)
        level_status = []
        for level in range(nlevels):
            statuses = [
                window_catalog_status(
                    nconv=int(nconv[level, column]),
                    effective_nev=int(effective[level, column]),
                    solver_converged=bool(converged[level, column]),
                )
                for column in np.flatnonzero(valid[level])
            ]
            level_status.append(aggregate_catalog_status(statuses))
        level_status = np.asarray(level_status, dtype="U24")
    return level_status, aggregate_catalog_status(level_status)


def _knot_multiplicities(knots: np.ndarray) -> dict[float, int]:
    values, counts = np.unique(np.asarray(knots, dtype=float), return_counts=True)
    return {float(value): int(count) for value, count in zip(values, counts)}


def bspline_prolongation_matrix(coarse, fine) -> csr_matrix:
    """Return the exact coefficient map for a knot-insertion-nested B-spline pair."""
    if int(coarse.degree) != int(fine.degree):
        raise ValueError("Nested B-spline spaces must have the same degree.")
    coarse_counts = _knot_multiplicities(coarse.knots)
    fine_counts = _knot_multiplicities(fine.knots)
    for knot, count in coarse_counts.items():
        if fine_counts.get(knot, 0) < count:
            raise ValueError("The fine knot vector does not contain the coarse knots.")
    missing = []
    for knot, count in fine_counts.items():
        missing.extend([knot] * (count - coarse_counts.get(knot, 0)))
    columns = np.zeros((int(fine.dofs), int(coarse.dofs)), dtype=float)
    for column in range(int(coarse.dofs)):
        coefficient = np.zeros(int(coarse.dofs), dtype=float)
        coefficient[column] = 1.0
        spline = BSpline(coarse.knots, coefficient, int(coarse.degree), extrapolate=False)
        for knot in missing:
            spline = spline.insert_knot(knot)
        if not np.array_equal(np.asarray(spline.t), np.asarray(fine.knots)):
            raise RuntimeError("Knot insertion did not produce the requested fine space.")
        columns[:, column] = np.asarray(spline.c[: int(fine.dofs)], dtype=float)
    return csr_matrix(columns)


def mixed_space_prolongation(coarse_mixed, fine_mixed, n_modes: int) -> csr_matrix:
    """Prolong all four mixed fields without mixing their physical components."""
    high = bspline_prolongation_matrix(coarse_mixed.pair.high, fine_mixed.pair.high)
    low = bspline_prolongation_matrix(coarse_mixed.pair.low, fine_mixed.pair.low)
    harmonic_identity = csr_matrix(np.eye(int(n_modes)))
    high_fields = kron(high, harmonic_identity, format="csr")
    low_fields = kron(low, harmonic_identity, format="csr")
    return block_diag((high_fields, high_fields, low_fields, low_fields), format="csr")


def _mass_orthonormal_rows(
    vectors: np.ndarray,
    weights: np.ndarray,
    *,
    rank_tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, int]:
    vectors = np.asarray(vectors, dtype=np.complex128)
    weights = np.asarray(weights, dtype=float)
    if vectors.ndim != 3:
        raise ValueError("vectors must have shape (mode, psi, component).")
    if weights.shape != (vectors.shape[1],):
        raise ValueError("weights must match the radial dimension.")
    flattened = vectors.reshape(vectors.shape[0], -1)
    repeated = np.repeat(weights, vectors.shape[2])
    gram = (flattened.conjugate() * repeated[None, :]) @ flattened.T
    values, basis = eigh(0.5 * (gram + gram.conjugate().T))
    maximum = max(float(np.max(values).real), np.finfo(float).tiny)
    keep = values > rank_tolerance * maximum
    if not np.any(keep):
        return np.empty((0, flattened.shape[1]), dtype=np.complex128), 0
    transform = basis[:, keep] / np.sqrt(values[keep])[None, :]
    orthonormal = transform.conjugate().T @ flattened
    return orthonormal, int(np.count_nonzero(keep))


def weighted_principal_angle_cosines(
    coarse_vectors: np.ndarray,
    fine_vectors: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return principal-angle cosines in the weighted ``xi_psi`` proxy norm."""
    coarse, coarse_rank = _mass_orthonormal_rows(coarse_vectors, weights)
    fine, fine_rank = _mass_orthonormal_rows(fine_vectors, weights)
    if coarse_rank == 0 or fine_rank == 0:
        return np.empty(0, dtype=float)
    repeated = np.repeat(np.asarray(weights, dtype=float), coarse_vectors.shape[2])
    cross_gram = (coarse.conjugate() * repeated[None, :]) @ fine.T
    singular = np.linalg.svd(cross_gram, compute_uv=False)
    return np.clip(np.asarray(singular.real, dtype=float), 0.0, 1.0)


def cluster_aware_mode_assignment(
    reference_frequency: np.ndarray,
    candidate_frequency: np.ndarray,
    reference_vectors: np.ndarray,
    candidate_vectors: np.ndarray,
    weights: np.ndarray,
    *,
    frequency_gate: float,
    tie_delta: float,
) -> dict[str, np.ndarray]:
    """Match isolated modes and compare near-degenerate components as subspaces.

    Connected components are built from selected assignment edges plus allowed
    alternatives whose overlap is nearly tied with the best row or column
    choice.  Equal-dimensional components are valid clusters; unequal rank is
    retained explicitly as unresolved dimensional ambiguity.
    """
    reference_frequency = np.asarray(reference_frequency, dtype=float)
    candidate_frequency = np.asarray(candidate_frequency, dtype=float)
    reference_vectors = np.asarray(reference_vectors, dtype=np.complex128)
    candidate_vectors = np.asarray(candidate_vectors, dtype=np.complex128)
    nreference = reference_frequency.size
    ncandidate = candidate_frequency.size
    if reference_vectors.shape[0] != nreference or candidate_vectors.shape[0] != ncandidate:
        raise ValueError("Frequency/vector counts do not match.")
    if nreference == 0 or ncandidate == 0:
        empty_i = np.empty(0, dtype=int)
        empty_f = np.empty(0, dtype=float)
        return {
            "rows": empty_i,
            "columns": empty_i.copy(),
            "frequency_change": empty_f,
            "individual_overlap": empty_f.copy(),
            "cluster_id": empty_i.copy(),
            "cluster_reference_dimension": empty_i.copy(),
            "cluster_candidate_dimension": empty_i.copy(),
            "cluster_min_cosine": empty_f.copy(),
            "cluster_mean_cosine": empty_f.copy(),
            "cluster_rms_cosine": empty_f.copy(),
            "cluster_dimension_ambiguous": np.empty(0, dtype=bool),
            "reference_dimension_ambiguous": np.zeros(nreference, dtype=bool),
            "candidate_dimension_ambiguous": np.zeros(ncandidate, dtype=bool),
        }
    overlaps = np.zeros((nreference, ncandidate), dtype=float)
    for row in range(nreference):
        for column in range(ncandidate):
            cosines = weighted_principal_angle_cosines(
                reference_vectors[row : row + 1],
                candidate_vectors[column : column + 1],
                weights,
            )
            overlaps[row, column] = cosines[0] if cosines.size else 0.0
    scale = np.maximum(
        np.maximum(np.abs(reference_frequency[:, None]), np.abs(candidate_frequency[None, :])),
        np.finfo(float).tiny,
    )
    frequency_change = np.abs(
        reference_frequency[:, None] - candidate_frequency[None, :]
    ) / scale
    allowed = np.isfinite(frequency_change) & (frequency_change <= frequency_gate)
    size = nreference + ncandidate
    unmatched = float(size + 2)
    forbidden = unmatched * float(size + 2)
    cost = np.full((size, size), 0.0, dtype=float)
    cost[:nreference, :ncandidate] = np.where(
        allowed,
        1.0 - overlaps + 1.0e-9 * frequency_change / frequency_gate,
        forbidden,
    )
    cost[:nreference, ncandidate:] = unmatched
    cost[nreference:, :ncandidate] = unmatched
    assigned_rows, assigned_columns = linear_sum_assignment(cost)
    real = (assigned_rows < nreference) & (assigned_columns < ncandidate)
    assigned_rows = assigned_rows[real]
    assigned_columns = assigned_columns[real]
    keep = allowed[assigned_rows, assigned_columns]
    rows = assigned_rows[keep].astype(int)
    columns = assigned_columns[keep].astype(int)

    active = np.zeros_like(allowed)
    active[rows, columns] = True
    row_best = np.max(np.where(allowed, overlaps, -np.inf), axis=1)
    column_best = np.max(np.where(allowed, overlaps, -np.inf), axis=0)
    near_row = allowed & (overlaps >= row_best[:, None] - tie_delta)
    near_column = allowed & (overlaps >= column_best[None, :] - tie_delta)
    active |= near_row & near_column

    row_component = np.full(nreference, -1, dtype=int)
    column_component = np.full(ncandidate, -1, dtype=int)
    components: list[tuple[np.ndarray, np.ndarray]] = []
    for seed in range(nreference):
        if row_component[seed] >= 0 or not np.any(active[seed]):
            continue
        component = len(components)
        pending_rows = [seed]
        pending_columns: list[int] = []
        row_set: set[int] = set()
        column_set: set[int] = set()
        while pending_rows or pending_columns:
            while pending_rows:
                row = pending_rows.pop()
                if row in row_set:
                    continue
                row_set.add(row)
                for column in np.flatnonzero(active[row]):
                    if int(column) not in column_set:
                        pending_columns.append(int(column))
            while pending_columns:
                column = pending_columns.pop()
                if column in column_set:
                    continue
                column_set.add(column)
                for row in np.flatnonzero(active[:, column]):
                    if int(row) not in row_set:
                        pending_rows.append(int(row))
        component_rows = np.asarray(sorted(row_set), dtype=int)
        component_columns = np.asarray(sorted(column_set), dtype=int)
        row_component[component_rows] = component
        column_component[component_columns] = component
        components.append((component_rows, component_columns))

    pair_cluster = np.full(rows.size, -1, dtype=int)
    reference_dimension = np.ones(rows.size, dtype=int)
    candidate_dimension = np.ones(rows.size, dtype=int)
    minimum = overlaps[rows, columns].copy()
    mean = minimum.copy()
    rms = minimum.copy()
    dimension_ambiguous = np.zeros(rows.size, dtype=bool)
    reference_dimension_ambiguous = np.zeros(nreference, dtype=bool)
    candidate_dimension_ambiguous = np.zeros(ncandidate, dtype=bool)
    for component_rows, component_columns in components:
        if component_rows.size != component_columns.size:
            reference_dimension_ambiguous[component_rows] = True
            candidate_dimension_ambiguous[component_columns] = True
    for pair, (row, column) in enumerate(zip(rows, columns)):
        component = row_component[row]
        if component < 0:
            component = len(components)
            component_rows = np.asarray([row])
            component_columns = np.asarray([column])
        else:
            component_rows, component_columns = components[component]
        pair_cluster[pair] = component
        reference_dimension[pair] = component_rows.size
        candidate_dimension[pair] = component_columns.size
        dimension_ambiguous[pair] = component_rows.size != component_columns.size
        if component_rows.size > 1 or component_columns.size > 1:
            cosines = weighted_principal_angle_cosines(
                reference_vectors[component_rows],
                candidate_vectors[component_columns],
                weights,
            )
            summary = principal_cosine_summary(cosines)
            minimum[pair] = summary["minimum"]
            mean[pair] = summary["mean"]
            rms[pair] = summary["rms"]
    return {
        "rows": rows,
        "columns": columns,
        "frequency_change": frequency_change[rows, columns],
        "individual_overlap": overlaps[rows, columns],
        "cluster_id": pair_cluster,
        "cluster_reference_dimension": reference_dimension,
        "cluster_candidate_dimension": candidate_dimension,
        "cluster_min_cosine": minimum,
        "cluster_mean_cosine": mean,
        "cluster_rms_cosine": rms,
        "cluster_dimension_ambiguous": dimension_ambiguous,
        "reference_dimension_ambiguous": reference_dimension_ambiguous,
        "candidate_dimension_ambiguous": candidate_dimension_ambiguous,
    }


def principal_cosine_summary(cosines: np.ndarray) -> dict[str, float]:
    cosines = np.asarray(cosines, dtype=float)
    finite = cosines[np.isfinite(cosines)]
    if finite.size == 0:
        return {"minimum": np.nan, "mean": np.nan, "rms": np.nan}
    return {
        "minimum": float(np.min(finite)),
        "mean": float(np.mean(finite)),
        "rms": float(np.sqrt(np.mean(finite**2))),
    }


def trapezoid_weights(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 1 or points.size < 2 or np.any(np.diff(points) <= 0.0):
        raise ValueError("points must be a strictly increasing one-dimensional grid.")
    weights = np.empty(points.size, dtype=float)
    weights[0] = 0.5 * (points[1] - points[0])
    weights[-1] = 0.5 * (points[-1] - points[-2])
    weights[1:-1] = 0.5 * (points[2:] - points[:-2])
    return weights


def migrate_legacy_mode_dataset(modes: xr.Dataset) -> tuple[xr.Dataset, np.ndarray]:
    """Convert a portable schema-v1 mode table into evidence-v2 inputs.

    Legacy artifacts store the final ``xi_psi`` field and per-level scalar
    histories. Quantities that were not serialized remain unavailable and are
    represented by ``NaN`` so the classifier preserves them as ``unknown``.
    """
    ntracks = modes.sizes.get("track", 0)
    nlevels = modes.sizes.get("level", 0)
    points = np.asarray(modes.coords["x"].values, dtype=float)
    m_pol = np.asarray(modes.coords["m_pol"].values)

    def history(name: str, *, fill=np.nan, dtype=float):
        stored_name = f"{name}_by_level"
        if stored_name in modes:
            return np.asarray(modes[stored_name].values, dtype=dtype)
        return np.full((ntracks, nlevels), fill, dtype=dtype)

    frequency = history("frequency_khz", dtype=float)
    present = history("present", fill=False, dtype=bool)
    if not np.any(present):
        present = np.isfinite(frequency)
    base_spans = int(modes.attrs.get("base_spans", 0) or 0)
    radial_h = (
        1.0 / (base_spans * 2.0 ** np.arange(nlevels))
        if base_spans > 0
        else np.full(nlevels, np.nan)
    )
    h_peak = np.broadcast_to(radial_h[None, :], (ntracks, nlevels)).copy()
    peak = history("peak_radius", dtype=float)
    if not np.any(np.isfinite(peak)) and "center" in modes:
        peak = np.broadcast_to(
            np.asarray(modes["center"].values, dtype=float)[:, None],
            (ntracks, nlevels),
        ).copy()
    if "xi_psi_real" not in modes or "xi_psi_imag" not in modes:
        raise ValueError("Stored modes are missing xi_psi real/imaginary fields.")
    xi_final = np.asarray(modes["xi_psi_real"].values, dtype=float) + 1.0j * np.asarray(
        modes["xi_psi_imag"].values,
        dtype=float,
    )
    tracks = xr.Dataset(
        {
            "present": (("track", "level"), present),
            # Units cancel in relative convergence tests and match the legacy
            # continuum frequency_khz values supplied during reclassification.
            "omega": (("track", "level"), frequency.astype(np.complex128)),
            "backward_error": (
                ("track", "level"),
                history("backward_error", dtype=float),
            ),
            "frequency_change": (
                ("track", "level"),
                history("frequency_change", dtype=float),
            ),
            "xi_psi_overlap": (
                ("track", "level"),
                history("xi_psi_overlap", dtype=float),
            ),
            "assignment_ambiguous": (
                ("track", "level"),
                history("assignment_ambiguous", fill=False, dtype=bool),
            ),
            "peak_radius": (("track", "level"), peak),
            "rms_width": (
                ("track", "level"),
                history("rms_width", dtype=float),
            ),
            "ipr": (("track", "level"), history("ipr", dtype=float)),
            "kr": (("track", "level"), np.full((ntracks, nlevels), np.nan)),
            "h_peak": (("track", "level"), h_peak),
            "reacquisition_status": (
                ("track", "level"),
                np.full((ntracks, nlevels), "not_attempted", dtype="U48"),
            ),
            "xi_psi_final": (("track", "psi", "m_pol"), xi_final),
        },
        coords={
            "track": modes.coords["track"].values,
            "level": modes.coords["level"].values,
            "psi": points,
            "m_pol": m_pol,
        },
        attrs={
            "migration_source_schema": int(modes.attrs.get("schema_version", 1)),
            "frequency_units": "kHz",
            "radial_h_source": (
                "1/(base_spans*2**level)" if base_spans else "unknown"
            ),
        },
    )
    return tracks, trapezoid_weights(points)


def local_element_size(point: float, breakpoints: np.ndarray) -> float:
    breakpoints = np.asarray(breakpoints, dtype=float)
    if breakpoints.ndim != 1 or breakpoints.size < 2:
        return np.nan
    index = int(np.searchsorted(breakpoints, point, side="right") - 1)
    index = min(max(index, 0), breakpoints.size - 2)
    return float(breakpoints[index + 1] - breakpoints[index])


def radial_power_diagnostics(
    displacement: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    derivative: np.ndarray | None = None,
    breakpoints: np.ndarray | None = None,
    metric: str = "xi_psi_proxy",
) -> dict[str, float | str]:
    """Calculate quadrature-weighted radial localization and roughness evidence.

    ``ipr`` uses ``integral(P**2) / integral(P)**2`` on normalized poloidal
    flux.  ``kr`` is the positive square root of the derivative-power ratio.
    """
    values = np.asarray(displacement, dtype=np.complex128)
    points = np.asarray(points, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != points.size:
        raise ValueError("displacement must have shape (psi, component).")
    if weights is None:
        weights = trapezoid_weights(points)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != points.shape:
        raise ValueError("weights must match points.")
    power = np.sum(np.abs(values) ** 2, axis=1)
    total = float(np.sum(weights * power))
    output: dict[str, float | str] = {
        "power_density_metric": str(metric),
        "ipr_convention": "integral(P^2)dpsi/(integral(P)dpsi)^2",
    }
    if not np.isfinite(total) or total <= np.finfo(float).tiny:
        output.update(
            peak_radius=np.nan,
            mean_radius=np.nan,
            rms_width=np.nan,
            ipr=np.nan,
            kr=np.nan,
            h_peak=np.nan,
            width_over_h=np.nan,
            h_times_ipr=np.nan,
            h_times_kr=np.nan,
        )
        return output
    peak = float(points[int(np.nanargmax(power))])
    mean = float(np.sum(weights * points * power) / total)
    variance = float(np.sum(weights * (points - mean) ** 2 * power) / total)
    width = float(np.sqrt(max(0.0, variance)))
    ipr = float(np.sum(weights * power**2) / total**2)
    if derivative is None:
        derivative_values = np.gradient(values, points, axis=0, edge_order=1)
    else:
        derivative_values = np.asarray(derivative, dtype=np.complex128)
        if derivative_values.shape != values.shape:
            raise ValueError("derivative must match displacement.")
    derivative_power = np.sum(np.abs(derivative_values) ** 2, axis=1)
    kr2 = float(np.sum(weights * derivative_power) / total)
    kr = float(np.sqrt(max(0.0, kr2)))
    if breakpoints is None:
        h_peak = float(np.median(np.diff(points)))
    else:
        h_peak = local_element_size(peak, breakpoints)
    output.update(
        peak_radius=peak,
        mean_radius=mean,
        rms_width=width,
        ipr=ipr,
        kr=kr,
        h_peak=h_peak,
        width_over_h=width / h_peak if h_peak > 0.0 else np.nan,
        h_times_ipr=h_peak * ipr if h_peak > 0.0 else np.nan,
        h_times_kr=h_peak * kr if h_peak > 0.0 else np.nan,
    )
    return output


def _log_slope(h: np.ndarray, values: np.ndarray) -> tuple[float, str]:
    h = np.asarray(h, dtype=float)
    values = np.asarray(values, dtype=float)
    use = np.isfinite(h) & (h > 0.0) & np.isfinite(values) & (values > 0.0)
    count = int(np.count_nonzero(use))
    if count < 2:
        return np.nan, "unknown"
    slope = float(np.polyfit(np.log(h[use]), np.log(values[use]), 1)[0])
    return slope, "regression" if count >= 3 else "pairwise_lower_confidence"


def refinement_scaling_evidence(
    h: np.ndarray,
    width: np.ndarray,
    ipr: np.ndarray,
    kr: np.ndarray,
    config: EvidenceV2Config,
) -> dict[str, float | str | int]:
    """Fit localization exponents and require multiple consistent diagnostics."""
    alpha_width, width_fit = _log_slope(h, width)
    ipr_slope, ipr_fit = _log_slope(h, ipr)
    kr_slope, kr_fit = _log_slope(h, kr)
    h_ipr_slope, h_ipr_fit = _log_slope(h, np.asarray(h) * np.asarray(ipr))
    h_kr_slope, h_kr_fit = _log_slope(h, np.asarray(h) * np.asarray(kr))
    beta_ipr = -ipr_slope if np.isfinite(ipr_slope) else np.nan
    beta_kr = -kr_slope if np.isfinite(kr_slope) else np.nan
    checks = np.asarray(
        [
            np.isfinite(alpha_width)
            and alpha_width >= config.continuum_scaling_exponent_min,
            np.isfinite(beta_ipr)
            and beta_ipr >= config.continuum_scaling_exponent_min,
            np.isfinite(beta_kr)
            and beta_kr >= config.continuum_scaling_exponent_min,
        ],
        dtype=bool,
    )
    available = np.asarray(
        [np.isfinite(alpha_width), np.isfinite(beta_ipr), np.isfinite(beta_kr)],
        dtype=bool,
    )
    if int(np.count_nonzero(available)) < config.singular_diagnostic_min_count:
        singular = UNKNOWN
    else:
        singular = tri_state(
            int(np.count_nonzero(checks)) >= config.singular_diagnostic_min_count
        )
    return {
        "alpha_width": alpha_width,
        "beta_ipr": beta_ipr,
        "beta_kr": beta_kr,
        "h_ipr_log_slope": h_ipr_slope,
        "h_kr_log_slope": h_kr_slope,
        "width_fit_confidence": width_fit,
        "ipr_fit_confidence": ipr_fit,
        "kr_fit_confidence": kr_fit,
        "h_ipr_fit_confidence": h_ipr_fit,
        "h_kr_fit_confidence": h_kr_fit,
        "singular_diagnostic_count": int(np.count_nonzero(checks)),
        "singular_refinement_scaling": singular,
    }


def fixed_coarse_space_retention(
    displacement: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray,
    coarse_basis,
) -> float:
    """Return weighted power retained by projection into one fixed coarse space."""
    values = np.asarray(displacement, dtype=np.complex128)
    if values.ndim == 1:
        values = values[:, None]
    design = np.asarray(coarse_basis.eval_basis(points), dtype=float).T
    weights = np.asarray(weights, dtype=float)
    weighted_design = design * np.sqrt(weights)[:, None]
    projected = np.zeros_like(values)
    for component in range(values.shape[1]):
        rhs = values[:, component] * np.sqrt(weights)
        coefficient, *_ = np.linalg.lstsq(weighted_design, rhs, rcond=None)
        projected[:, component] = design @ coefficient
    numerator = float(np.sum(weights[:, None] * np.abs(projected) ** 2))
    denominator = float(np.sum(weights[:, None] * np.abs(values) ** 2))
    if not np.isfinite(denominator) or denominator <= np.finfo(float).tiny:
        return np.nan
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def enriched_residual(
    matrices: Sequence,
    s: complex,
    vector: np.ndarray,
) -> float:
    """Evaluate a prolonged eigenvector in the next-finer polynomial operator."""
    if len(matrices) != 3:
        raise ValueError("matrices must be (T0, T1, Tminus1).")
    T0, T1, Tminus1 = matrices
    vector = np.asarray(vector, dtype=np.complex128)
    terms = (Tminus1 @ vector, s * (T0 @ vector), s**2 * (T1 @ vector))
    numerator = float(np.linalg.norm(terms[0] + terms[1] + terms[2]))
    denominator = float(sum(np.linalg.norm(term) for term in terms))
    if denominator <= np.finfo(float).tiny or not np.isfinite(denominator):
        return np.nan
    return numerator / denominator


def deterministic_shifted_bspline(base, options: MeshPerturbationOptions | None = None):
    """Build a valid same-size B-spline with deterministic shifted interior knots."""
    from ..radial import BSplineFE

    options = MeshPerturbationOptions() if options is None else options
    breakpoints = np.unique(np.asarray(base.knots, dtype=float))
    if breakpoints.size < 3:
        raise ValueError("A shifted variant requires at least one interior knot.")
    rng = np.random.default_rng(int(options.seed))
    shifted = breakpoints.copy()
    for index in range(1, breakpoints.size - 1):
        left_h = breakpoints[index] - breakpoints[index - 1]
        right_h = breakpoints[index + 1] - breakpoints[index]
        local_h = min(left_h, right_h)
        alternating = -1.0 if index % 2 else 1.0
        jitter = 0.85 + 0.30 * rng.random()
        delta = alternating * options.amplitude_fraction * jitter * local_h
        lower = shifted[index - 1] + 0.25 * local_h
        upper = breakpoints[index + 1] - 0.25 * local_h
        shifted[index] = np.clip(breakpoints[index] + delta, lower, upper)
    if np.any(np.diff(shifted) <= 0.0):
        raise RuntimeError("The deterministic knot perturbation lost monotonicity.")
    knots = np.concatenate(
        (
            np.repeat(shifted[0], int(base.order)),
            shifted[1:-1],
            np.repeat(shifted[-1], int(base.order)),
        )
    )
    variant = BSplineFE(
        knots=knots,
        order=int(base.order),
        domain=(float(shifted[0]), float(shifted[-1])),
        quad_order=int(base.quadrature_order),
        build_matrices=bool(base.build_matrices),
    )
    variant.mesh_perturbation_amplitude_fraction = float(options.amplitude_fraction)
    variant.mesh_perturbation_seed = int(options.seed)
    return variant


def mesh_perturbation_evidence(
    *,
    reference_frequency: float,
    shifted_frequency: float | None,
    overlap: float | None,
    reference_peak: float | None,
    shifted_peak: float | None,
    reference_width: float | None,
    shifted_width: float | None,
    local_h: float | None,
    reference_continuum_aligned: bool | None,
    shifted_continuum_aligned: bool | None,
    config: EvidenceV2Config | None = None,
) -> dict[str, float | str]:
    """Summarize one explicitly selected non-nested mesh validation."""
    config = EvidenceV2Config() if config is None else config
    if shifted_frequency is None or not np.isfinite(shifted_frequency):
        return {
            "shifted_mesh_frequency_change": np.nan,
            "shifted_mesh_overlap": np.nan,
            "shifted_mesh_peak_distance_over_h": np.nan,
            "shifted_mesh_width_change": np.nan,
            "mesh_robust": FALSE,
            "perturbation_evidence_supports_nonconvergence": TRUE,
        }
    frequency_change = abs(shifted_frequency - reference_frequency) / max(
        abs(reference_frequency), abs(shifted_frequency), np.finfo(float).tiny
    )
    peak_distance_h = (
        abs(float(shifted_peak) - float(reference_peak)) / float(local_h)
        if shifted_peak is not None
        and reference_peak is not None
        and local_h is not None
        and np.isfinite(shifted_peak)
        and np.isfinite(reference_peak)
        and np.isfinite(local_h)
        and local_h > 0.0
        else np.nan
    )
    width_change = (
        abs(float(shifted_width) - float(reference_width))
        / max(abs(float(reference_width)), np.finfo(float).tiny)
        if shifted_width is not None
        and reference_width is not None
        and np.isfinite(shifted_width)
        and np.isfinite(reference_width)
        else np.nan
    )
    checks = [frequency_change <= config.mesh_frequency_change_max]
    if overlap is not None and np.isfinite(overlap):
        checks.append(float(overlap) >= config.mesh_overlap_min)
    if np.isfinite(peak_distance_h):
        checks.append(peak_distance_h <= config.mesh_peak_shift_h_max)
    if reference_continuum_aligned is not None and shifted_continuum_aligned is not None:
        checks.append(bool(reference_continuum_aligned) == bool(shifted_continuum_aligned))
    robust = tri_state(len(checks) >= 3 and all(checks))
    nonconvergent = tri_state(
        frequency_change > config.mesh_frequency_change_max
        and (
            overlap is None
            or not np.isfinite(overlap)
            or float(overlap) < config.mesh_overlap_min
        )
    )
    return {
        "shifted_mesh_frequency_change": frequency_change,
        "shifted_mesh_overlap": float(overlap) if overlap is not None else np.nan,
        "shifted_mesh_peak_distance_over_h": peak_distance_h,
        "shifted_mesh_width_change": width_change,
        "mesh_robust": robust,
        "perturbation_evidence_supports_nonconvergence": nonconvergent,
    }


class ContinuumQuery:
    """Frequency-local, set-based queries over a sampled continuum spectrum."""

    def __init__(
        self,
        continuum: xr.Dataset,
        *,
        imaginary_ratio_max: float = 1.0e-6,
        variable: str = "omega",
    ) -> None:
        if variable not in continuum:
            raise KeyError(f"Continuum dataset has no {variable!r} variable.")
        self.psi = np.asarray(continuum.coords["psi"].values, dtype=float)
        self.omega = np.asarray(continuum[variable].values, dtype=np.complex128)
        if self.psi.ndim != 1 or self.omega.ndim != 2:
            raise ValueError("Continuum must have psi and (psi, branch) frequencies.")
        if self.omega.shape[0] != self.psi.size or np.any(np.diff(self.psi) <= 0.0):
            raise ValueError("Continuum radial coordinates are inconsistent.")
        denominator = np.maximum(np.abs(self.omega.real), np.finfo(float).tiny)
        imaginary_ratio = np.abs(self.omega.imag) / denominator
        self.valid = (
            np.isfinite(self.omega.real)
            & np.isfinite(self.omega.imag)
            & (imaginary_ratio <= float(imaginary_ratio_max))
        )
        self.real = np.where(self.valid, np.abs(self.omega.real), np.nan)
        self.ambiguity = (
            np.asarray(continuum["branch_assignment_ambiguous"].values, dtype=bool)
            if "branch_assignment_ambiguous" in continuum
            else np.zeros_like(self.valid)
        )
        self._segments_prepared = False
        self._segment_psi_left = np.empty(0, dtype=float)
        self._segment_psi_right = np.empty(0, dtype=float)
        self._segment_frequency_left = np.empty(0, dtype=float)
        self._segment_frequency_right = np.empty(0, dtype=float)

    def nearest_distance(self, omega: float) -> float | None:
        frequency = abs(float(np.real(omega)))
        distance = np.abs(self.real - frequency)
        if not np.any(np.isfinite(distance)):
            return None
        return float(np.nanmin(distance))

    def _adjacent_assignments(self, left: int, right: int):
        left_indices = np.flatnonzero(np.isfinite(self.real[left]))
        right_indices = np.flatnonzero(np.isfinite(self.real[right]))
        if left_indices.size == 0 or right_indices.size == 0:
            return ()
        # Sorting is a label-free optimal assignment on the real line and does
        # not require permanent continuum branch identities through degeneracy.
        left_order = left_indices[np.argsort(self.real[left, left_indices])]
        right_order = right_indices[np.argsort(self.real[right, right_indices])]
        count = min(left_order.size, right_order.size)
        return tuple(
            (int(left_branch), int(right_branch))
            for left_branch, right_branch in zip(left_order[:count], right_order[:count])
        )

    def _prepare_segments(self) -> None:
        if self._segments_prepared:
            return
        psi_left: list[float] = []
        psi_right: list[float] = []
        frequency_left: list[float] = []
        frequency_right: list[float] = []
        for left in range(self.psi.size - 1):
            for left_branch, right_branch in self._adjacent_assignments(left, left + 1):
                psi_left.append(float(self.psi[left]))
                psi_right.append(float(self.psi[left + 1]))
                frequency_left.append(float(self.real[left, left_branch]))
                frequency_right.append(float(self.real[left + 1, right_branch]))
        self._segment_psi_left = np.asarray(psi_left, dtype=float)
        self._segment_psi_right = np.asarray(psi_right, dtype=float)
        self._segment_frequency_left = np.asarray(frequency_left, dtype=float)
        self._segment_frequency_right = np.asarray(frequency_right, dtype=float)
        self._segments_prepared = True

    def find_resonance_surfaces(self, omega: float) -> np.ndarray:
        """Locate crossings using only adjacent local continuum sets."""
        surfaces, _, _ = self.resonance_surface_data(omega)
        return surfaces

    def resonance_surface_data(
        self,
        omega: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return set-based crossing radii, local element sizes, and slopes."""
        frequency = abs(float(np.real(omega)))
        self._prepare_segments()
        first = self._segment_frequency_left - frequency
        second = self._segment_frequency_right - frequency
        crossing = (first == 0.0) | (second == 0.0) | (first * second < 0.0)
        if not np.any(crossing):
            empty = np.empty(0, dtype=float)
            return empty, empty.copy(), empty.copy()
        first = first[crossing]
        second = second[crossing]
        psi_left = self._segment_psi_left[crossing]
        psi_right = self._segment_psi_right[crossing]
        denominator = np.abs(first) + np.abs(second)
        fraction = np.divide(
            np.abs(first),
            denominator,
            out=np.full_like(denominator, 0.5),
            where=denominator > 0.0,
        )
        surfaces = psi_left + fraction * (psi_right - psi_left)
        h = psi_right - psi_left
        slope = np.abs(
            self._segment_frequency_right[crossing]
            - self._segment_frequency_left[crossing]
        ) / h
        rounded = np.round(surfaces, decimals=14)
        _, unique_indices = np.unique(rounded, return_index=True)
        return surfaces[unique_indices], h[unique_indices], slope[unique_indices]

    def local_relative_distance(self, points: np.ndarray, omega: float) -> np.ndarray:
        """Return nearest continuum-set distance at each requested radius."""
        points = np.asarray(points, dtype=float)
        frequency = abs(float(np.real(omega)))
        right = np.searchsorted(self.psi, points, side="left")
        right = np.clip(right, 0, self.psi.size - 1)
        left = np.clip(right - 1, 0, self.psi.size - 1)
        choose_right = np.abs(self.psi[right] - points) < np.abs(self.psi[left] - points)
        indices = np.where(choose_right, right, left)
        distance = np.full(points.size, np.nan)
        for output_index, radial_index in enumerate(indices):
            values = self.real[int(radial_index)]
            if np.any(np.isfinite(values)):
                distance[output_index] = np.nanmin(np.abs(values - frequency)) / max(
                    frequency, np.finfo(float).tiny
                )
        return distance

    def coverage_for_frequency(self, omega: float, uncertainty: float) -> str:
        """Return tri-state coverage only for branches capable of approaching omega."""
        frequency = abs(float(np.real(omega)))
        band = max(
            3.0 * abs(float(uncertainty)),
            0.05 * np.finfo(float).eps,
            0.05 * frequency,
        )
        relevant = np.any(np.isfinite(self.real) & (np.abs(self.real - frequency) <= band), axis=0)
        if not np.any(relevant):
            # No nearby branch: all-radii finite set coverage is enough for a gap query.
            if np.all(np.any(np.isfinite(self.real), axis=1)):
                return TRUE
            return UNKNOWN
        for branch in np.flatnonzero(relevant):
            near = np.isfinite(self.real[:, branch]) & (
                np.abs(self.real[:, branch] - frequency) <= 2.0 * band
            )
            if np.any(near & self.ambiguity[:, branch]):
                # Degenerate labels are harmless because queries are set-based.
                pass
            missing = ~self.valid[:, branch]
            if np.any(missing):
                neighbors = np.convolve(near.astype(int), np.ones(3, dtype=int), mode="same") > 0
                if np.any(missing & neighbors):
                    return UNKNOWN
        return TRUE

    def local_continuum_subspace(self, psi: float, omega: float):
        """Return unavailable until local continuum eigenvectors are supplied."""
        return None

    def continuum_uncertainty(self, omega: float) -> float | None:
        frequency = abs(float(np.real(omega)))
        self._prepare_segments()
        if self._segment_frequency_left.size == 0:
            return None
        distance = np.minimum(
            np.abs(self._segment_frequency_left - frequency),
            np.abs(self._segment_frequency_right - frequency),
        )
        index = int(np.nanargmin(distance))
        uncertainty = 0.5 * abs(
            self._segment_frequency_right[index] - self._segment_frequency_left[index]
        )
        return float(uncertainty) if np.isfinite(uncertainty) else None


def resonance_power_evidence(
    query: ContinuumQuery,
    omega: float,
    displacement: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray,
    *,
    peak_radius: float,
    radius_h: float,
    numerical_frequency_uncertainty: float = 0.0,
    continuum_band_relative: float = 5.0e-3,
) -> dict[str, object]:
    surfaces, continuum_h, slopes = query.resonance_surface_data(omega)
    if surfaces.size == 0:
        return {
            "resonance_radii": "",
            "peak_distance_over_h": np.nan,
            "resonant_power_fraction": 0.0,
            "continuum_band_power_fraction": 0.0,
            "has_continuum_crossing": FALSE,
        }
    distance_over_h = np.abs(surfaces - float(peak_radius)) / continuum_h
    values = np.asarray(displacement, dtype=np.complex128)
    if values.ndim == 1:
        values = values[:, None]
    power = np.sum(np.abs(values) ** 2, axis=1)
    total = float(np.sum(weights * power))
    numerical_radius = np.divide(
        abs(float(numerical_frequency_uncertainty)),
        slopes,
        out=np.zeros_like(slopes),
        where=slopes > np.finfo(float).tiny,
    )
    resonance_radius = radius_h * continuum_h + numerical_radius
    nearest_scaled = np.min(
        np.abs(points[:, None] - surfaces[None, :]) / resonance_radius[None, :],
        axis=1,
    )
    fraction = (
        float(np.sum(weights[nearest_scaled <= 1.0] * power[nearest_scaled <= 1.0]) / total)
        if total > np.finfo(float).tiny
        else np.nan
    )
    local_relative_distance = query.local_relative_distance(points, omega)
    band_fraction = (
        float(
            np.sum(
                weights[local_relative_distance <= continuum_band_relative]
                * power[local_relative_distance <= continuum_band_relative]
            )
            / total
        )
        if total > np.finfo(float).tiny
        else np.nan
    )
    return {
        "resonance_radii": ",".join(f"{value:.12g}" for value in surfaces),
        "peak_distance_over_h": float(np.min(distance_over_h)),
        "resonant_power_fraction": fraction,
        "continuum_band_power_fraction": band_fraction,
        "has_continuum_crossing": TRUE,
    }


def _value(dataset: xr.Dataset, name: str, track: int, default=np.nan):
    if name not in dataset:
        return default
    value = dataset[name].values[track]
    return value.item() if np.asarray(value).ndim == 0 else value


def _provided_state(dataset: xr.Dataset, name: str, track: int) -> str | None:
    if name not in dataset:
        return None
    return tri_state(_value(dataset, name, track, None))


def _last_present_level(present: np.ndarray) -> int:
    available = np.flatnonzero(present)
    return int(available[-1]) if available.size else -1


def _level_history(dataset: xr.Dataset, name: str, track: int, nlevels: int) -> np.ndarray:
    if name not in dataset:
        return np.full(nlevels, np.nan)
    values = np.asarray(dataset[name].values[track])
    if values.shape != (nlevels,):
        return np.full(nlevels, np.nan)
    return values


def _reason_code_sets(flags: Mapping[str, str]) -> tuple[str, str, str]:
    satisfied = sorted(name for name, state in flags.items() if state == TRUE)
    failed = sorted(name for name, state in flags.items() if state == FALSE)
    unknown = sorted(name for name, state in flags.items() if state == UNKNOWN)
    return ";".join(satisfied), ";".join(failed), ";".join(unknown)


def classify_tracks_evidence_v2(
    tracks: xr.Dataset,
    continuum: xr.Dataset,
    weights: np.ndarray,
    catalog_status: str | Sequence[str],
    config: EvidenceV2Config | None = None,
) -> xr.Dataset:
    """Calculate independent evidence first, then assign evidence-v2 labels."""
    config = EvidenceV2Config() if config is None else config
    ntracks = tracks.sizes.get("track", 0)
    nlevels = tracks.sizes.get("level", 0)
    if isinstance(catalog_status, str):
        run_catalog_status = catalog_status
        level_catalog_status = np.full(nlevels, catalog_status, dtype="U24")
    else:
        level_catalog_status = np.asarray(catalog_status).astype("U24")
        run_catalog_status = aggregate_catalog_status(level_catalog_status)
    if run_catalog_status not in CATALOG_STATUSES:
        raise ValueError(f"Unknown catalog status {run_catalog_status!r}.")
    query = ContinuumQuery(
        continuum,
        imaginary_ratio_max=config.imaginary_frequency_ratio_max,
    )
    points = np.asarray(tracks.coords.get("psi", tracks.coords.get("x", [])).values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if points.size and weights.shape != points.shape:
        raise ValueError("weights must match the track radial grid.")

    labels = np.full(ntracks, "unresolved", dtype="U32")
    verification = np.full(ntracks, "baseline", dtype="U32")
    reason_codes = np.full(ntracks, "", dtype="U2048")
    satisfied_codes = np.full(ntracks, "", dtype="U2048")
    failed_codes = np.full(ntracks, "", dtype="U2048")
    unknown_codes = np.full(ntracks, "", dtype="U2048")
    last_level = np.full(ntracks, -1, dtype=int)
    level_valid = np.full((ntracks, nlevels), UNKNOWN, dtype="U7")

    evidence_names = (
        "discrete_valid",
        "chain_converged",
        "subspace_converged",
        "frequency_converged",
        "strong_global_limit",
        "strict_gap",
        "has_continuum_crossing",
        "local_continuum_alignment",
        "continuum_polarization_match",
        "singular_refinement_scaling",
        "mesh_robust",
        "targeted_absence_confirmed",
        "enriched_residual_converged",
        "perturbation_evidence_supports_nonconvergence",
        "roots_valid_on_at_least_two_levels",
        "frequency_changes_decrease",
    )
    evidence = {name: np.full(ntracks, UNKNOWN, dtype="U7") for name in evidence_names}
    float_names = (
        "imaginary_frequency_ratio",
        "fine_frequency_change",
        "medium_fine_overlap",
        "frequency_change_ratio",
        "alpha_width",
        "beta_ipr",
        "beta_kr",
        "coarse_space_retention_final",
        "minimum_relative_continuum_clearance",
        "uncertainty_normalized_clearance",
        "mode_frequency_uncertainty",
        "continuum_frequency_uncertainty",
        "peak_distance_over_h",
        "resonant_power_fraction",
        "continuum_band_power_fraction",
        "alfven_continuum_overlap",
        "sound_continuum_overlap",
        "mixed_continuum_overlap",
    )
    floats = {name: np.full(ntracks, np.nan) for name in float_names}
    resonance_radii = np.full(ntracks, "", dtype="U4096")
    overlap_metric = np.full(ntracks, "unknown", dtype="U32")
    power_metric = np.full(ntracks, "unknown", dtype="U32")
    discrete_summary = np.full(ntracks, "unknown", dtype="U32")
    tracking_summary = np.full(ntracks, "unknown", dtype="U32")
    shape_summary = np.full(ntracks, "unknown", dtype="U32")
    continuum_summary = np.full(ntracks, "unknown", dtype="U32")
    perturbation_summary = np.full(ntracks, "unknown", dtype="U32")

    for track in range(ntracks):
        present = (
            np.asarray(tracks["present"].values[track], dtype=bool)
            if "present" in tracks
            else np.ones(nlevels, dtype=bool)
        )
        final = _last_present_level(present)
        last_level[track] = final
        if final < 0:
            flags = {name: UNKNOWN for name in evidence_names}
            satisfied, failed, unknown = _reason_code_sets(flags)
            unknown_codes[track] = unknown
            reason_codes[track] = "no_returned_root"
            continue

        omega_history = _level_history(tracks, "omega", track, nlevels).astype(complex)
        error_history = _level_history(tracks, "backward_error", track, nlevels).astype(float)
        valid_count = 0
        for level in np.flatnonzero(present):
            omega = complex(omega_history[level])
            imaginary_ratio = abs(omega.imag) / max(abs(omega.real), np.finfo(float).tiny)
            error = float(error_history[level])
            if not np.isfinite(error) or not np.isfinite(imaginary_ratio):
                level_valid[track, level] = UNKNOWN
            else:
                level_valid[track, level] = tri_state(
                    error <= config.backward_error_max
                    and imaginary_ratio <= config.imaginary_frequency_ratio_max
                )
            valid_count += level_valid[track, level] == TRUE
        final_omega = complex(omega_history[final])
        final_imaginary_ratio = abs(final_omega.imag) / max(
            abs(final_omega.real), np.finfo(float).tiny
        )
        floats["imaginary_frequency_ratio"][track] = final_imaginary_ratio
        discrete_valid = level_valid[track, final]
        roots_valid_two = tri_state(valid_count >= 2) if np.count_nonzero(present) >= 2 else UNKNOWN

        frequency_change = _level_history(tracks, "frequency_change", track, nlevels).astype(float)
        individual_overlap = _level_history(tracks, "xi_psi_overlap", track, nlevels).astype(float)
        cluster_overlap = _level_history(
            tracks, "cluster_min_principal_cosine", track, nlevels
        ).astype(float)
        chosen_overlap = cluster_overlap[final] if np.isfinite(cluster_overlap[final]) else individual_overlap[final]
        overlap_metric[track] = (
            "xi_psi_proxy_subspace"
            if np.isfinite(cluster_overlap[final])
            else "xi_psi_proxy"
        )
        floats["fine_frequency_change"][track] = frequency_change[final]
        floats["medium_fine_overlap"][track] = chosen_overlap
        frequency_converged = (
            tri_state(frequency_change[final] <= config.fine_frequency_change_max)
            if np.isfinite(frequency_change[final])
            else UNKNOWN
        )
        subspace_converged = (
            tri_state(chosen_overlap >= config.subspace_overlap_min)
            if np.isfinite(chosen_overlap)
            else UNKNOWN
        )
        if nlevels >= 3 and np.all(np.isfinite(frequency_change[-2:])):
            denominator = abs(frequency_change[-2])
            ratio = frequency_change[-1] / denominator if denominator > 0.0 else np.nan
            floats["frequency_change_ratio"][track] = ratio
            frequency_decreases = tri_state(np.isfinite(ratio) and ratio < 1.0)
        else:
            frequency_decreases = UNKNOWN
        cluster_ambiguity = (
            np.asarray(tracks["cluster_dimension_ambiguous"].values[track], dtype=bool)
            if "cluster_dimension_ambiguous" in tracks
            else np.asarray(
                tracks.get(
                    "assignment_ambiguous",
                    xr.DataArray(np.zeros((ntracks, nlevels), dtype=bool), dims=("track", "level")),
                ).values[track],
                dtype=bool,
            )
        )
        reacquisition = (
            np.asarray(tracks["reacquisition_status"].values[track]).astype(str)
            if "reacquisition_status" in tracks
            else np.full(nlevels, "not_attempted", dtype="U48")
        )
        presence_after_reacquisition = bool(
            np.all(present | (reacquisition == "reacquired"))
        )
        chain_valid = tri_and(
            tri_state(nlevels >= config.minimum_levels),
            tri_state(presence_after_reacquisition),
            tri_state(np.all(level_valid[track, present] == TRUE))
            if np.all(level_valid[track, present] != UNKNOWN)
            else UNKNOWN,
            tri_state(not np.any(cluster_ambiguity[present])),
            frequency_converged,
            subspace_converged,
        )

        width = _level_history(tracks, "rms_width", track, nlevels).astype(float)
        ipr = _level_history(tracks, "ipr", track, nlevels).astype(float)
        kr = _level_history(tracks, "kr", track, nlevels).astype(float)
        h = _level_history(tracks, "h_peak", track, nlevels).astype(float)
        scaling = refinement_scaling_evidence(h, width, ipr, kr, config)
        for name in ("alpha_width", "beta_ipr", "beta_kr"):
            floats[name][track] = float(scaling[name])
        singular_scaling = str(scaling["singular_refinement_scaling"])
        provided_scaling = _provided_state(tracks, "singular_refinement_scaling", track)
        if provided_scaling is not None:
            singular_scaling = provided_scaling
        retention = _level_history(tracks, "coarse_space_retention", track, nlevels).astype(float)
        if np.isfinite(retention[final]):
            floats["coarse_space_retention_final"][track] = retention[final]
        smooth_scaling_checks = [
            abs(value) < config.continuum_scaling_exponent_min
            for value in (
                scaling["alpha_width"],
                scaling["beta_ipr"],
                scaling["beta_kr"],
            )
            if np.isfinite(value)
        ]
        smooth_scaling = (
            tri_state(len(smooth_scaling_checks) >= 2 and all(smooth_scaling_checks))
            if len(smooth_scaling_checks) >= 2
            else UNKNOWN
        )
        strong_global = tri_or(
            tri_and(chain_valid, smooth_scaling),
            tri_state(retention[final] >= config.coarse_retention_min)
            if np.isfinite(retention[final])
            else UNKNOWN,
        )

        mode_uncertainty = np.nan
        if final > 0 and present[final - 1]:
            mode_uncertainty = abs(final_omega.real - omega_history[final - 1].real)
        if np.isfinite(error_history[final]):
            residual_uncertainty = abs(final_omega.real) * error_history[final]
            mode_uncertainty = (
                max(mode_uncertainty, residual_uncertainty)
                if np.isfinite(mode_uncertainty)
                else residual_uncertainty
            )
        floats["mode_frequency_uncertainty"][track] = mode_uncertainty
        continuum_uncertainty = query.continuum_uncertainty(final_omega.real)
        if continuum_uncertainty is not None:
            floats["continuum_frequency_uncertainty"][track] = continuum_uncertainty
        nearest = query.nearest_distance(final_omega.real)
        relative_clearance = (
            nearest / max(abs(final_omega.real), np.finfo(float).tiny)
            if nearest is not None
            else np.nan
        )
        floats["minimum_relative_continuum_clearance"][track] = relative_clearance
        combined_uncertainty = (
            mode_uncertainty
            + (continuum_uncertainty if continuum_uncertainty is not None else np.nan)
        )
        normalized_clearance = (
            nearest / combined_uncertainty
            if nearest is not None
            and np.isfinite(combined_uncertainty)
            and combined_uncertainty > 0.0
            else np.nan
        )
        floats["uncertainty_normalized_clearance"][track] = normalized_clearance
        coverage = query.coverage_for_frequency(
            final_omega.real,
            mode_uncertainty if np.isfinite(mode_uncertainty) else 0.0,
        )

        shape_variable = None
        if "xi_psi" in tracks:
            shape_variable = tracks["xi_psi"]
        elif "xi_psi_final" in tracks:
            shape_variable = tracks["xi_psi_final"]
        if shape_variable is not None and points.size:
            if "level" in shape_variable.dims:
                shape = np.asarray(shape_variable.values[track, final], dtype=np.complex128)
            else:
                shape = np.asarray(shape_variable.values[track], dtype=np.complex128)
            shape_diag = radial_power_diagnostics(
                shape,
                points,
                weights,
                metric="xi_psi_proxy",
            )
            power_metric[track] = "xi_psi_proxy"
            peak = (
                float(_level_history(tracks, "peak_radius", track, nlevels)[final])
                if np.isfinite(_level_history(tracks, "peak_radius", track, nlevels)[final])
                else float(shape_diag["peak_radius"])
            )
            resonance = resonance_power_evidence(
                query,
                final_omega.real,
                shape,
                points,
                weights,
                peak_radius=peak,
                radius_h=config.resonance_power_radius_h,
                numerical_frequency_uncertainty=(
                    combined_uncertainty if np.isfinite(combined_uncertainty) else 0.0
                ),
                continuum_band_relative=config.continuum_band_relative,
            )
            crossing = str(resonance["has_continuum_crossing"])
            resonance_radii[track] = str(resonance["resonance_radii"])
            floats["peak_distance_over_h"][track] = float(resonance["peak_distance_over_h"])
            floats["resonant_power_fraction"][track] = float(resonance["resonant_power_fraction"])
            floats["continuum_band_power_fraction"][track] = float(
                resonance["continuum_band_power_fraction"]
            )
            local_alignment = tri_and(
                crossing,
                tri_or(
                    tri_state(
                        np.isfinite(floats["peak_distance_over_h"][track])
                        and floats["peak_distance_over_h"][track]
                        <= config.resonance_peak_distance_h_max
                    ),
                    tri_state(
                        np.isfinite(floats["resonant_power_fraction"][track])
                        and floats["resonant_power_fraction"][track]
                        >= config.resonance_power_fraction_min
                    ),
                ),
            )
        else:
            crossing = _provided_state(tracks, "has_continuum_crossing", track) or UNKNOWN
            local_alignment = _provided_state(tracks, "local_continuum_alignment", track) or UNKNOWN
        provided_crossing = _provided_state(tracks, "has_continuum_crossing", track)
        provided_alignment = _provided_state(tracks, "local_continuum_alignment", track)
        if provided_crossing is not None:
            crossing = provided_crossing
        if provided_alignment is not None:
            local_alignment = provided_alignment
        strict_gap = tri_and(
            coverage,
            tri_state(crossing == FALSE),
            tri_state(
                normalized_clearance > config.gap_uncertainty_multiplier
            )
            if np.isfinite(normalized_clearance)
            else UNKNOWN,
        )
        provided_gap = _provided_state(tracks, "strict_gap", track)
        if provided_gap is not None:
            strict_gap = provided_gap

        polarization = _provided_state(tracks, "continuum_polarization_match", track) or UNKNOWN
        for name in (
            "alfven_continuum_overlap",
            "sound_continuum_overlap",
            "mixed_continuum_overlap",
        ):
            value = _value(tracks, name, track, np.nan)
            if np.asarray(value).ndim == 0:
                floats[name][track] = float(value)

        targeted_absence = tri_state(
            np.any(reacquisition == "not_found_in_locally_complete_search")
        )
        if np.any(reacquisition == "capacity_limited") and targeted_absence != TRUE:
            targeted_absence = UNKNOWN
        enriched = _level_history(tracks, "enriched_residual", track, nlevels).astype(float)
        finite_enriched = enriched[np.isfinite(enriched)]
        if finite_enriched.size >= 2 and finite_enriched[0] > 0.0:
            enriched_converged = tri_state(
                finite_enriched[-1] / finite_enriched[0]
                <= config.enriched_residual_reduction_max
            )
        elif finite_enriched.size:
            enriched_converged = UNKNOWN
        else:
            enriched_converged = _provided_state(
                tracks, "enriched_residual_converged", track
            ) or UNKNOWN
        perturb_nonconvergence = _provided_state(
            tracks, "perturbation_evidence_supports_nonconvergence", track
        ) or UNKNOWN
        mesh_robust = _provided_state(tracks, "mesh_robust", track) or UNKNOWN
        regularization_verified = _provided_state(
            tracks, "regularization_verified", track
        ) or FALSE
        if regularization_verified == TRUE:
            verification[track] = "regularization_verified"
        elif mesh_robust == TRUE:
            verification[track] = "mesh_robust"

        flags = {
            "discrete_valid": discrete_valid,
            "chain_converged": chain_valid,
            "subspace_converged": subspace_converged,
            "frequency_converged": frequency_converged,
            "strong_global_limit": strong_global,
            "strict_gap": strict_gap,
            "has_continuum_crossing": crossing,
            "local_continuum_alignment": local_alignment,
            "continuum_polarization_match": polarization,
            "singular_refinement_scaling": singular_scaling,
            "mesh_robust": mesh_robust,
            "targeted_absence_confirmed": targeted_absence,
            "enriched_residual_converged": enriched_converged,
            "perturbation_evidence_supports_nonconvergence": perturb_nonconvergence,
            "roots_valid_on_at_least_two_levels": roots_valid_two,
            "frequency_changes_decrease": frequency_decreases,
        }
        for name, state in flags.items():
            evidence[name][track] = state

        if discrete_valid == FALSE:
            label = "solver_rejected"
            winning = ("final_root_failed_algebraic_validity",)
        elif chain_valid == TRUE and strict_gap == TRUE:
            label = "physical_gap"
            winning = ("converged_nested_grid_chain", "uncertainty_resolved_strict_gap")
        elif (
            discrete_valid == TRUE
            and roots_valid_two == TRUE
            and local_alignment == TRUE
            and singular_scaling == TRUE
        ):
            label = "continuum_discretization"
            winning = ("same_sequence_local_continuum_alignment", "singular_mesh_scaling")
        elif (
            discrete_valid == TRUE
            and crossing == TRUE
            and tri_or(chain_valid, subspace_converged, strong_global) == TRUE
        ):
            label = "continuum_coupled_candidate"
            winning = ("global_limit_with_continuum_crossing",)
        elif (
            discrete_valid == TRUE
            and targeted_absence == TRUE
            and enriched_converged == FALSE
            and perturb_nonconvergence == TRUE
        ):
            label = "discretization_artifact"
            winning = (
                "credible_targeted_absence",
                "enriched_residual_nonconvergence",
                "independent_mesh_nonconvergence",
            )
        else:
            label = "unresolved"
            winning = ("insufficient_or_conflicting_evidence",)
        labels[track] = label
        satisfied, failed, unknown = _reason_code_sets(flags)
        satisfied_codes[track] = satisfied
        failed_codes[track] = failed
        unknown_codes[track] = unknown
        reason_codes[track] = ";".join(
            (*winning, f"catalog_status_{run_catalog_status}")
        )
        discrete_summary[track] = (
            "valid" if discrete_valid == TRUE else "rejected" if discrete_valid == FALSE else "unknown"
        )
        tracking_summary[track] = (
            "converged" if chain_valid == TRUE else "not_converged" if chain_valid == FALSE else "unknown"
        )
        shape_summary[track] = (
            "singular_scaling"
            if singular_scaling == TRUE
            else "non_singular_scaling"
            if singular_scaling == FALSE
            else "unknown"
        )
        continuum_summary[track] = (
            "strict_gap"
            if strict_gap == TRUE
            else "locally_aligned_crossing"
            if local_alignment == TRUE
            else "crossing"
            if crossing == TRUE
            else "unknown"
        )
        perturbation_summary[track] = (
            "mesh_robust"
            if mesh_robust == TRUE
            else "supports_nonconvergence"
            if perturb_nonconvergence == TRUE
            else "unknown"
        )

    data_vars: dict[str, tuple] = {
        "label": ("track", labels),
        "verification_level": ("track", verification),
        "reason_codes": ("track", reason_codes),
        "reason": ("track", reason_codes),
        "satisfied_reason_codes": ("track", satisfied_codes),
        "failed_reason_codes": ("track", failed_codes),
        "unknown_reason_codes": ("track", unknown_codes),
        "discrete_validity": ("track", discrete_summary),
        "tracking_evidence": ("track", tracking_summary),
        "shape_evidence": ("track", shape_summary),
        "continuum_evidence": ("track", continuum_summary),
        "perturbation_evidence": ("track", perturbation_summary),
        "catalog_status": ("track", np.full(ntracks, run_catalog_status, dtype="U24")),
        "last_level": ("track", last_level),
        "level_discrete_valid": (("track", "level"), level_valid),
        "overlap_metric": ("track", overlap_metric),
        "power_density_metric": ("track", power_metric),
        "resonance_radii": ("track", resonance_radii),
    }
    data_vars.update({name: ("track", values) for name, values in evidence.items()})
    data_vars.update({name: ("track", values) for name, values in floats.items()})
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "track": tracks.coords["track"].values,
            "level": tracks.coords["level"].values,
        },
        attrs={
            "classification_version": CLASSIFICATION_VERSION,
            "allowed_labels": ",".join(EVIDENCE_LABELS),
            "verification_levels": ",".join(VERIFICATION_LEVELS),
            "run_catalog_status": run_catalog_status,
            "level_catalog_status": ",".join(level_catalog_status.astype(str)),
            "thresholds": repr(config),
            "unknown_representation": UNKNOWN,
            "catalog_semantics": "reporting metadata; never a positive per-mode veto",
            "ipr_convention": "integral(P^2)dpsi/(integral(P)dpsi)^2",
        },
    )


__all__ = [
    "CATALOG_STATUSES",
    "CLASSIFICATION_VERSION",
    "ContinuumQuery",
    "AdaptiveDiscoveryOptions",
    "EVIDENCE_LABELS",
    "EvidenceV2Config",
    "MeshPerturbationOptions",
    "REACQUISITION_STATUSES",
    "ReacquisitionOptions",
    "VERIFICATION_LEVELS",
    "aggregate_catalog_status",
    "bspline_prolongation_matrix",
    "catalog_status_from_diagnostics",
    "classify_tracks_evidence_v2",
    "cluster_aware_mode_assignment",
    "deterministic_shifted_bspline",
    "enriched_residual",
    "empirical_candidate_stability",
    "fixed_coarse_space_retention",
    "mixed_space_prolongation",
    "mesh_perturbation_evidence",
    "migrate_legacy_mode_dataset",
    "principal_cosine_summary",
    "radial_power_diagnostics",
    "refinement_scaling_evidence",
    "resonance_power_evidence",
    "trapezoid_weights",
    "tri_and",
    "tri_or",
    "tri_state",
    "weighted_principal_angle_cosines",
    "window_catalog_status",
]
