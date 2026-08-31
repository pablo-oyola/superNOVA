"""Reconstructing modes from their finite-element coefficients.

The solver returns coefficients; a plot, a diagnostic or a comparison wants
fields on a radial grid. This is the batch version used when rebuilding tracks
from saved per-level catalogs -- for a single result,
:meth:`supernova.api.ModeSet.field` is the friendlier path.

Lifted from ``no_sync/window_solver_multigrid.py``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import unyt
import xarray as xr

from supernova.app.schema import FIELD_TITLES, SCHEMA_VERSION


def _omega_range(equilibrium, frequency_khz: Iterable[float]) -> np.ndarray:
    frequency = unyt.unyt_array(
        np.sort(np.asarray(tuple(frequency_khz), dtype=float)),
        "kHz",
        registry=equilibrium.registry,
    )
    return np.asarray(frequency.to("omegaA").v, dtype=float) * (2.0 * np.pi)


def _omega_to_khz(equilibrium, omega) -> np.ndarray:
    omega = np.asarray(omega, dtype=np.complex128)
    omega_a = unyt.Unit("omegaA", registry=equilibrium.registry)
    return (omega.real * omega_a).to("kHz").v / (2.0 * np.pi)


def _omega_to_khz_from_saved_continuum(continuum: xr.Dataset):
    omega = np.asarray(continuum["omega"].values, dtype=np.complex128).real
    frequency = np.asarray(continuum["frequency_khz"].values, dtype=float)
    use = np.isfinite(omega) & (np.abs(omega) > np.finfo(float).tiny) & np.isfinite(
        frequency
    )
    if not np.any(use):
        raise ValueError("Saved continuum does not define an omega-to-kHz conversion")
    scale = float(np.median(frequency[use] / omega[use]))
    relative_error = np.abs(frequency[use] - scale * omega[use]) / np.maximum(
        np.abs(frequency[use]),
        np.finfo(float).tiny,
    )
    if not np.isfinite(scale) or np.nanmax(relative_error) > 1.0e-10:
        raise ValueError("Saved continuum has an inconsistent omega-to-kHz conversion")
    return lambda values: np.asarray(values, dtype=np.complex128).real * scale


def _trapezoid_weights(points: np.ndarray) -> np.ndarray:
    from supernova.solve.evidence import trapezoid_weights

    return trapezoid_weights(points)


def _evaluate_basis(pair, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    high = np.asarray(pair.high.eval_basis(points), dtype=float).T
    low = np.asarray(pair.low.eval_basis(points), dtype=float).T
    return high, low


def _reconstruct_modes_from_levels(
    level_solutions: tuple[xr.Dataset, ...],
    tracks: xr.Dataset,
    evidence: xr.Dataset,
    mixed_levels: tuple,
    omega_to_khz,
    npoints: int,
) -> xr.Dataset:
    """Reconstruct final fields for either a live solve or offline retracking."""
    ntracks = tracks.sizes.get("track", 0)
    m_pol = np.asarray(tracks.coords["m_pol"].values, dtype=np.int64)
    points = np.linspace(0.0, 1.0, int(npoints))
    shape = (ntracks, points.size, m_pol.size)
    fields = {
        name: np.full(shape, np.nan + 1.0j * np.nan, dtype=np.complex128)
        for name in FIELD_TITLES
    }
    center = np.full(ntracks, np.nan)
    radial_width = np.full(ntracks, np.nan)
    dominant_m = np.full(ntracks, -1, dtype=np.int64)
    normalization = np.full(ntracks, np.nan)
    window_id = np.full(ntracks, -1, dtype=np.int64)
    window_depth = np.full(ntracks, -1, dtype=np.int64)
    basis_values: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    final_level = np.asarray(evidence["last_level"].values, dtype=np.int64)
    mode_index = np.asarray(tracks["mode_index"].values, dtype=np.int64)
    for track in range(ntracks):
        level = int(final_level[track])
        if level < 0:
            continue
        index = int(mode_index[track, level])
        if index < 0:
            continue
        solution = level_solutions[level]
        if level not in basis_values:
            basis_values[level] = _evaluate_basis(
                mixed_levels[level].pair,
                points,
            )
        phi_high, phi_low = basis_values[level]

        coefficients = {
            "xi_psi": np.asarray(solution["xi_psi_coefficient"].values[index]),
            "p1": np.asarray(solution["p1_coefficient"].values[index]),
            "xi_s": np.asarray(solution["xi_s_coefficient"].values[index]),
            "div_xi": np.asarray(solution["div_xi_coefficient"].values[index]),
        }
        reconstructed = {
            "xi_psi": phi_high @ coefficients["xi_psi"],
            "p1": phi_high @ coefficients["p1"],
            "xi_s": phi_low @ coefficients["xi_s"],
            "div_xi": phi_low @ coefficients["div_xi"],
        }
        power = np.sum(np.abs(reconstructed["xi_psi"]) ** 2, axis=1)
        norm2 = float(np.trapezoid(power, points))
        if not np.isfinite(norm2) or norm2 <= np.finfo(float).tiny:
            continue
        scale = np.sqrt(norm2)
        normalization[track] = scale
        for name in reconstructed:
            reconstructed[name] /= scale

        anchor = reconstructed["xi_psi"].reshape(-1)
        anchor_value = anchor[int(np.nanargmax(np.abs(anchor)))]
        phase = np.exp(-1.0j * np.angle(anchor_value))
        for name, values in reconstructed.items():
            fields[name][track] = values * phase

        normalized_power = np.sum(np.abs(fields["xi_psi"][track]) ** 2, axis=1)
        total = float(np.trapezoid(normalized_power, points))
        center[track] = float(np.trapezoid(points * normalized_power, points) / total)
        variance = float(
            np.trapezoid(
                (points - center[track]) ** 2 * normalized_power,
                points,
            )
            / total
        )
        radial_width[track] = np.sqrt(max(0.0, variance))
        harmonic_power = np.trapezoid(
            np.abs(fields["xi_psi"][track]) ** 2,
            points,
            axis=0,
        )
        dominant_m[track] = int(m_pol[int(np.nanargmax(harmonic_power))])
        if "window_id" in solution:
            window_id[track] = int(solution["window_id"].values[index])
        if "window_depth" in solution:
            window_depth[track] = int(solution["window_depth"].values[index])

    omega_by_level = np.asarray(tracks["omega"].values, dtype=np.complex128)
    frequency_by_level = omega_to_khz(omega_by_level)
    final_omega = np.full(ntracks, np.nan + 1.0j * np.nan, dtype=np.complex128)
    for track, level in enumerate(final_level):
        if level >= 0:
            final_omega[track] = omega_by_level[track, level]

    dataset = xr.Dataset(
        data_vars={
            **{
                name: (("track", "x", "m_pol"), values)
                for name, values in fields.items()
            },
            "omega": ("track", final_omega),
            "frequency_khz": ("track", omega_to_khz(final_omega)),
            "center": ("track", center),
            "radial_width": ("track", radial_width),
            "dominant_m": ("track", dominant_m),
            "normalization_scale": ("track", normalization),
            "window_id": ("track", window_id),
            "window_depth": ("track", window_depth),
            "frequency_khz_by_level": (
                ("track", "level"),
                frequency_by_level,
            ),
        },
        coords={
            "track": tracks.coords["track"].values,
            "level": tracks.coords["level"].values,
            "x": points,
            "m_pol": m_pol,
        },
        attrs={
            "schema_version": SCHEMA_VERSION,
            "radial_coordinate": "psi_N",
            "omega_normalization": "omegaA",
            "frequency_definition": "Re(omega*omegaA)/(2*pi) in cyclic kHz",
            "phase_alignment": "largest-magnitude xi_psi sample made real-positive",
            "normalization": "integral sum_m |xi_psi_m|^2 dpsi_N = 1",
        },
    )

    track_fields = (
        "mode_index",
        "source_mode_index",
        "present",
        "residual_norm",
        "backward_error",
        "xi_psi_overlap",
        "overlap_metric",
        "frequency_change",
        "assignment_ambiguous",
        "cluster_id",
        "cluster_reference_dimension",
        "cluster_candidate_dimension",
        "cluster_min_principal_cosine",
        "cluster_mean_principal_cosine",
        "cluster_rms_principal_cosine",
        "cluster_dimension_ambiguous",
        "reacquisition_status",
        "peak_radius",
        "power_weighted_mean_radius",
        "rms_width",
        "ipr",
        "kr",
        "h_peak",
        "width_over_h",
        "h_times_ipr",
        "h_times_kr",
        "coarse_space_retention",
        "enriched_residual",
    )
    for name in track_fields:
        if name in tracks:
            dataset[f"{name}_by_level"] = tracks[name]
    for name, variable in evidence.data_vars.items():
        dataset[name] = variable
    dataset.x.attrs.update(
        units="psi_N",
        long_name="normalized poloidal flux",
    )
    dataset.m_pol.attrs.update(long_name="poloidal mode number")
    return dataset


def _reconstruct_final_modes(result, multigrid, equilibrium, npoints: int) -> xr.Dataset:
    return _reconstruct_modes_from_levels(
        result.level_solutions,
        result.mode_tracks,
        result.classification,
        multigrid.mixed_levels,
        lambda omega: _omega_to_khz(equilibrium, omega),
        npoints,
    )
