#!/usr/bin/env python3
r"""Run the DIII-D solver5 multigrid search and explore labeled modes in Dash.

The production defaults deliberately mirror ``no_sync/window_solver.py``:

* the smoothed DIII-D 202020 TokaMaker equilibrium and ``ne.dat`` density,
* ``n=4``, ``mmax=68``, ``max_dm=5``, and ``gamma_s=1e-3``,
* the full normalized-poloidal-flux domain, and
* the same 50--150 kHz cyclic-frequency band.

Unlike the original flat window scan, ``solve`` uses three nested compatible
B-spline spaces and saves the complete conservative label/evidence contract.
Complex arrays are stored as explicit ``*_real``/``*_imag`` variables so the
NetCDF files remain portable across xarray backends.

Examples
--------
Production scan::

    conda run -n desc-petsc-np2 python no_sync/window_solver_multigrid.py solve

Open the saved web explorer::

    conda run -n desc-petsc-np2 python no_sync/window_solver_multigrid.py serve

Build deterministic artifacts for GUI testing without PETSc::

    conda run -n desc-petsc-np2 python no_sync/window_solver_multigrid.py demo
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable, Mapping
from collections import Counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/supernova-matplotlib")

import numpy as np
import unyt
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_EQUILIBRIUM = (
    REPO_ROOT.parent
    / "oldnova"
    / "runs/d3d_202020_n3_comparison/tokamaker_equilibrium"
    / "80f0555a598e2e83325545eb2d4353f890b1e4b14eed1b8a4a296d0b1307d2ca"
    / "tokamaker_smoothed.geqdsk"
)
DEFAULT_DENSITY = REPO_ROOT / "no_sync/D3D/202020/ne.dat"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "no_sync/D3D/202020/multigrid_n4_50_150"
SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, 2)

LABEL_ORDER = (
    "solver_rejected",
    "physical_gap",
    "continuum_discretization",
    "continuum_coupled_candidate",
    "discretization_artifact",
    "unresolved",
)
LABEL_STYLE = {
    "solver_rejected": {
        "title": "Solver rejected",
        "color": "#758295",
        "symbol": "x",
    },
    "physical_gap": {
        "title": "Physical gap",
        "color": "#159A8A",
        "symbol": "circle",
    },
    "continuum_discretization": {
        "title": "Continuum discretization",
        "color": "#D64B5F",
        "symbol": "diamond",
    },
    "continuum_coupled_candidate": {
        "title": "Continuum-coupled candidate",
        "color": "#7C3AED",
        "symbol": "star",
    },
    "discretization_artifact": {
        "title": "Discretization artifact",
        "color": "#475569",
        "symbol": "cross",
    },
    "unresolved": {
        "title": "Unresolved",
        "color": "#E39A31",
        "symbol": "triangle-up",
    },
}
FIELD_TITLES = {
    "xi_psi": "Radial displacement xi^psi",
    "p1": "Pressure perturbation p1",
    "xi_s": "Tangential displacement xi^s",
    "div_xi": "Compression div xi",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve = subparsers.add_parser("solve", help="run the multigrid DIII-D scan")
    solve.add_argument("--equilibrium", type=Path, default=DEFAULT_EQUILIBRIUM)
    solve.add_argument("--density", type=Path, default=DEFAULT_DENSITY)
    solve.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    solve.add_argument("--frequency-khz", nargs=2, type=float, default=(50.0, 150.0))
    solve.add_argument("--cocos-in", type=int, default=7)
    solve.add_argument("--ntor", type=int, default=4)
    solve.add_argument("--mmax", type=int, default=68)
    solve.add_argument("--max-dm", type=int, default=5)
    solve.add_argument("--gamma-s", type=float, default=1.0e-3)
    solve.add_argument("--species", default="D")
    solve.add_argument("--regularizer", type=float, default=0.0)
    solve.add_argument(
        "--base-spans",
        type=int,
        default=38,
        help="coarsest spans; three levels give 38/76/152 spans",
    )
    solve.add_argument("--levels", type=int, default=3)
    solve.add_argument("--order", type=int, default=3)
    solve.add_argument("--quadrature-order", type=int, default=6)
    solve.add_argument("--profile-points", type=int, default=128)
    solve.add_argument("--reconstruction-points", type=int, default=512)
    solve.add_argument("--display-continuum-points", type=int, default=1024)
    solve.add_argument("--nev", type=int, default=8)
    solve.add_argument("--tol", type=float, default=1.0e-8)
    solve.add_argument("--maxit", type=int, default=200)
    solve.add_argument(
        "--window-width-khz",
        type=float,
        default=2.5,
        help="maximum fixed core-window width; defaults to the legacy 2.5 kHz",
    )
    solve.add_argument(
        "--window-padding",
        type=float,
        default=0.10,
        help="fraction of one core width added to each side of a PEP solve region",
    )
    solve.add_argument(
        "--region-vscale",
        type=float,
        default=0.25,
        help="imaginary semiaxis scale for the elliptical PEP region",
    )
    solve.add_argument(
        "--factorization",
        choices=("mumps", "spqr", "petsc_qr"),
        default="mumps",
    )
    solve.add_argument("--save-raw-levels", action="store_true")
    solve.add_argument(
        "--adaptive-discovery",
        action="store_true",
        help="use K=4,8,16 and center/quarter-window offsets on coarse/medium grids",
    )
    solve.add_argument(
        "--no-reacquisition",
        action="store_true",
        help="disable selective narrow searches for missing global candidates",
    )
    solve.add_argument("--reacquisition-max-candidates", type=int, default=24)
    solve.add_argument("--verbose", action="store_true")

    serve = subparsers.add_parser("serve", help="serve the saved Dash explorer")
    serve.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8050)

    reclassify = subparsers.add_parser(
        "reclassify",
        help="classify saved eigenpairs with evidence-v2 without rerunning SLEPc",
    )
    reclassify.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    reclassify.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="defaults to the artifact directory; original files are never overwritten",
    )
    reclassify.add_argument(
        "--source",
        choices=("auto", "raw", "legacy"),
        default="auto",
        help=(
            "prefer raw level_*.nc catalogs and rebuild tracks; 'legacy' preserves "
            "the old modes.nc assignments"
        ),
    )

    reacquire = subparsers.add_parser(
        "reacquire",
        help="selectively continue saved roots missing on the next radial level",
    )
    reacquire.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    reacquire.add_argument(
        "--execute",
        action="store_true",
        help="run the selected narrow SLEPc windows; otherwise only write a dry-run report",
    )
    reacquire.add_argument("--max-candidates-per-level", type=int, default=24)
    reacquire.add_argument("--nev-multiplier", type=int, default=4)
    reacquire.add_argument("--relative-half-width", type=float, default=3.0e-3)
    reacquire.add_argument("--minimum-overlap", type=float, default=0.90)
    reacquire.add_argument("--verbose", action="store_true")

    demo = subparsers.add_parser("demo", help="write deterministic GUI demo artifacts")
    demo.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "demo",
    )
    return parser


def _factorization_options(name: str) -> dict[str, str]:
    if name == "mumps":
        return {
            "st_ksp_type": "preonly",
            "st_pc_type": "lu",
            "st_pc_factor_mat_solver_type": "mumps",
        }
    options = {"st_ksp_type": "preonly", "st_pc_type": "qr"}
    if name == "spqr":
        options["st_pc_factor_mat_solver_type"] = "spqr"
    return options


def _base_spline(spans: int, order: int, quadrature_order: int):
    import supernova as sn

    if spans < 1:
        raise ValueError("base-spans must be positive.")
    if order < 1:
        raise ValueError("order must be positive.")
    breakpoints = np.linspace(0.0, 1.0, spans + 1)
    knots = np.concatenate(
        (
            np.repeat(breakpoints[0], order),
            breakpoints[1:-1],
            np.repeat(breakpoints[-1], order),
        )
    )
    return sn.BSplineFE(
        knots=knots,
        order=order,
        domain=(0.0, 1.0),
        quad_order=quadrature_order,
        build_matrices=False,
    )


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


def _window_count(frequency_band: tuple[float, float], width_khz: float) -> int:
    """Choose equal fixed windows no wider than the requested cyclic-kHz width."""
    low, high = (float(value) for value in frequency_band)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("frequency-khz must contain a finite increasing band.")
    if not np.isfinite(width_khz) or width_khz <= 0.0:
        raise ValueError("window-width-khz must be finite and positive.")
    quotient = (high - low) / float(width_khz)
    nearest = round(quotient)
    if np.isclose(quotient, nearest, rtol=1.0e-12, atol=1.0e-12):
        return max(1, int(nearest))
    return max(1, int(np.ceil(quotient)))


def _load_case(args: argparse.Namespace):
    import pycocos as pc
    import supernova as sn

    if not args.equilibrium.is_file():
        raise FileNotFoundError(args.equilibrium)
    if not args.density.is_file():
        raise FileNotFoundError(args.density)

    eqdsk = pc.EQDSK(args.equilibrium, cocos_in=args.cocos_in)
    # superEquilibrium consumes the cached Boozer map from the EQDSK object.
    eqdsk.compute_coordinates("boozer")

    profile_psi = np.linspace(0.0, 1.0, args.profile_points)
    profile_rhopol = np.sqrt(profile_psi)
    gs_rhopol = np.asarray(eqdsk.gs_profs["rhop"], dtype=float)
    gs_pressure = np.asarray(eqdsk.gs_profs["pres"], dtype=float)
    pressure = np.interp(profile_rhopol, gs_rhopol, gs_pressure)
    pressure_profile = sn.Profile(
        x=profile_psi,
        F_nm=pressure,
        attrs={"units": "Pa", "name": "pressure"},
    )

    density_psi, density_value = np.loadtxt(
        args.density,
        unpack=True,
        usecols=(0, 1),
        skiprows=1,
    )
    density = np.interp(profile_rhopol, np.sqrt(density_psi), density_value)
    density_profile = sn.Profile(
        x=profile_psi,
        F_nm=density,
        attrs={"units": "m^-3", "name": "density"},
    )

    equilibrium = sn.superEquilibrium(
        eqdsk,
        density=density_profile,
        pressure=pressure_profile,
    )
    continuum = sn.superContinuum(
        equilibrium,
        gamma_s=args.gamma_s,
        species=args.species,
    )
    return equilibrium, continuum


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


def _storage_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Return a standards-friendly dataset with complex variables split."""
    output = xr.Dataset(coords={name: value for name, value in dataset.coords.items()})
    for name, variable in dataset.data_vars.items():
        if np.issubdtype(variable.dtype, np.complexfloating):
            output[f"{name}_real"] = variable.real.astype(np.float64)
            output[f"{name}_imag"] = variable.imag.astype(np.float64)
            output[f"{name}_real"].attrs.update(variable.attrs)
            output[f"{name}_imag"].attrs.update(variable.attrs)
            output[f"{name}_real"].attrs["complex_component"] = "real"
            output[f"{name}_imag"].attrs["complex_component"] = "imaginary"
        else:
            output[name] = variable
            output[name].attrs.update(variable.attrs)
    for variable in output.variables.values():
        variable.attrs = {
            key: _netcdf_attribute(value) for key, value in variable.attrs.items()
        }
    output.attrs.update(
        {key: _netcdf_attribute(value) for key, value in dataset.attrs.items()}
    )
    output.attrs["complex_storage"] = "paired <name>_real and <name>_imag variables"
    return output


def _netcdf_attribute(value):
    """Coerce rich Python/xarray metadata to NetCDF4 attribute scalars."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return "None"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True, default=str)
    if isinstance(value, (tuple, list)):
        if any(isinstance(item, (bool, Mapping, tuple, list)) for item in value):
            return json.dumps(value, default=str)
        return value
    if isinstance(value, (str, bytes, int, float, np.ndarray)):
        return value
    return repr(value)


def _netcdf_encoding(dataset: xr.Dataset) -> dict[str, dict]:
    encoding: dict[str, dict] = {}
    for name, variable in dataset.data_vars.items():
        if variable.size == 0 or variable.dtype.kind in "OUS":
            continue
        chunks = []
        for dim, size in zip(variable.dims, variable.shape):
            if dim == "track":
                chunks.append(1)
            elif dim in ("x", "psi"):
                chunks.append(min(int(size), 256))
            else:
                chunks.append(int(size))
        entry: dict[str, object] = {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
        }
        if chunks:
            entry["chunksizes"] = tuple(chunks)
        encoding[name] = entry
    return encoding


def _write_dataset(dataset: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    stored = _storage_dataset(dataset)
    stored.to_netcdf(
        temporary,
        engine="netcdf4",
        format="NETCDF4",
        encoding=_netcdf_encoding(stored),
    )
    os.replace(temporary, path)


def _load_portable_dataset(path: Path, names: Iterable[str]) -> xr.Dataset:
    """Load selected variables and restore explicitly split complex arrays."""
    with xr.open_dataset(path, engine="netcdf4") as source:
        variables: dict[str, xr.DataArray] = {}
        for name in names:
            if name in source:
                variables[name] = source[name].load()
                continue
            real_name = f"{name}_real"
            imaginary_name = f"{name}_imag"
            if real_name not in source or imaginary_name not in source:
                continue
            restored = source[real_name].load().astype(np.float64) + 1.0j * source[
                imaginary_name
            ].load().astype(np.float64)
            restored.attrs.update(source[real_name].attrs)
            restored.attrs.pop("complex_component", None)
            variables[name] = restored
        dataset = xr.Dataset(variables)
        dataset.attrs.update(source.attrs)
    return dataset


def _raw_level_paths(artifact_dir: Path, expected_levels: int) -> tuple[Path, ...]:
    paths = tuple(artifact_dir / f"level_{level}.nc" for level in range(expected_levels))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Raw-level retracking requires a contiguous level_*.nc catalog; missing "
            + ", ".join(str(path) for path in missing)
        )
    return paths


def _load_raw_level(path: Path, *, include_eigenvector: bool = False) -> xr.Dataset:
    fields = [
        "s",
        "omega",
        "residual_norm",
        "backward_error",
        "slepc_backward_error",
        "absolute_residual_norm",
        "slepc_error_estimate",
        "window_id",
        "window_depth",
        "core_owned",
        "reacquisition_status",
        "xi_psi",
        "xi_psi_coefficient",
        "p1_coefficient",
        "xi_s_coefficient",
        "div_xi_coefficient",
    ]
    if include_eigenvector:
        fields.append("eigenvector")
    dataset = _load_portable_dataset(path, fields)
    required = {"s", "omega", "xi_psi", "xi_psi_coefficient", "window_id"}
    if include_eigenvector:
        required.add("eigenvector")
    missing = sorted(required.difference(dataset.data_vars))
    if missing:
        raise ValueError(f"{path} is missing raw retracking variables: {missing}")
    count = dataset.sizes.get("eigenpair", 0)
    dataset["source_eigenpair"] = (
        "eigenpair",
        np.asarray(dataset.coords["eigenpair"].values, dtype=np.int64),
    )
    dataset = dataset.assign_coords(eigenpair=np.arange(count, dtype=np.int64))
    return dataset


def _attach_source_mode_indices(
    tracks: xr.Dataset,
    level_solutions: tuple[xr.Dataset, ...] | list[xr.Dataset],
) -> xr.Dataset:
    """Record original raw indices, using -2 for newly reacquired roots."""
    source_mode_index = np.full(
        (tracks.sizes.get("track", 0), tracks.sizes.get("level", 0)),
        -1,
        dtype=np.int64,
    )
    mode_index = np.asarray(tracks["mode_index"].values, dtype=np.int64)
    for track in range(source_mode_index.shape[0]):
        for level in range(source_mode_index.shape[1]):
            index = int(mode_index[track, level])
            if index < 0:
                continue
            if "source_eigenpair" not in level_solutions[level]:
                source_mode_index[track, level] = -2
                continue
            value = level_solutions[level]["source_eigenpair"].values[index]
            source_mode_index[track, level] = int(value) if np.isfinite(value) else -2
    output = tracks.copy()
    output["source_mode_index"] = (
        ("track", "level"),
        source_mode_index,
    )
    return output


def _retrack_raw_levels(
    artifact_dir: Path,
    manifest: Mapping[str, object],
    *,
    include_eigenvectors: bool = False,
):
    """Rebuild nested spaces, deduplicate saved roots, and rematch all levels."""
    from supernova.core.solver4 import _build_mixed_radial_data
    from supernova.core.solver5_multigrid import (
        ModeClassificationThresholds,
        MultigridSearchOptions,
        _deduplicate_window_dataset,
        _nested_bspline_levels,
        _track_level_solutions,
    )

    configuration = dict(manifest.get("configuration", {}))
    levels = int(configuration.get("levels", 0))
    if levels < 1:
        raise ValueError("manifest configuration does not define a positive level count")
    radial_levels = _nested_bspline_levels(
        _base_spline(
            int(configuration["base_spans"]),
            int(configuration["order"]),
            int(configuration["quadrature_order"]),
        ),
        levels=levels,
    )
    points, weights = radial_levels[-1].get_quadrature_points()
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mixed_levels = tuple(
        _build_mixed_radial_data(radial, points, weights) for radial in radial_levels
    )
    search = MultigridSearchOptions()
    thresholds = ModeClassificationThresholds()
    raw_levels = tuple(
        _load_raw_level(path, include_eigenvector=include_eigenvectors)
        for path in _raw_level_paths(artifact_dir, expected_levels=levels)
    )
    for level, dataset in enumerate(raw_levels):
        if not np.array_equal(np.asarray(dataset.coords["psi"].values), points):
            raise ValueError(
                f"level_{level}.nc psi grid does not match reconstructed finest quadrature"
            )
    deduplicated = tuple(
        _deduplicate_window_dataset(
            dataset,
            weights,
            relative_s=search.dedup_relative_s,
            minimum_overlap=search.dedup_overlap,
        )
        for dataset in raw_levels
    )
    tracks = _track_level_solutions(
        deduplicated,
        weights,
        thresholds,
        radial_levels=radial_levels,
    )
    tracks = _attach_source_mode_indices(tracks, deduplicated)
    tracks.attrs.update(
        tracking_source="raw level_*.nc catalogs",
        raw_root_counts=tuple(
            dataset.sizes.get("eigenpair", 0) for dataset in raw_levels
        ),
        deduplicated_root_counts=tuple(
            dataset.sizes.get("eigenpair", 0) for dataset in deduplicated
        ),
        slepc_was_run=False,
    )
    return deduplicated, tracks, radial_levels, mixed_levels, points, weights


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


def _legacy_track_association(
    legacy_modes: xr.Dataset,
    tracks: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Associate retracked sequences with legacy rows through saved root indices."""
    legacy_lookup: dict[tuple[int, int], int] = {}
    legacy_indices = np.asarray(legacy_modes["mode_index_by_level"].values, dtype=int)
    legacy_present = np.asarray(legacy_modes["present_by_level"].values, dtype=bool)
    for legacy_track in range(legacy_indices.shape[0]):
        for level in np.flatnonzero(legacy_present[legacy_track]):
            level = int(level)
            legacy_lookup[(level, int(legacy_indices[legacy_track, level]))] = legacy_track

    source_indices = np.asarray(tracks["source_mode_index"].values, dtype=int)
    present = np.asarray(tracks["present"].values, dtype=bool)
    association = np.full(source_indices.shape[0], -1, dtype=np.int64)
    labels = np.full(source_indices.shape[0], "unmatched", dtype="U32")
    legacy_labels = np.asarray(legacy_modes["label"].values).astype(str)
    for track in range(source_indices.shape[0]):
        legacy_track = -1
        for level in np.flatnonzero(present[track])[::-1]:
            source_index = int(source_indices[track, level])
            if source_index < 0:
                continue
            legacy_track = legacy_lookup.get((int(level), source_index), -1)
            if legacy_track >= 0:
                break
        if legacy_track >= 0:
            association[track] = legacy_track
            labels[track] = legacy_labels[legacy_track]
    return association, labels


def _legacy_modes_to_tracks(modes: xr.Dataset) -> tuple[xr.Dataset, np.ndarray]:
    """Migrate the portable schema-v1 mode table into classifier inputs."""
    from supernova.core.solver5_multigrid_evidence import (
        migrate_legacy_mode_dataset,
    )

    return migrate_legacy_mode_dataset(modes)


def _load_legacy_modes(artifact_dir: Path) -> xr.Dataset:
    with xr.open_dataset(artifact_dir / "modes.nc", engine="netcdf4") as source:
        required = [
            name
            for name in source.data_vars
            if name.endswith("_by_level")
            or name
            in {
                "xi_psi_real",
                "xi_psi_imag",
                "center",
                "frequency_khz",
                "label",
                "reason",
                "last_level",
            }
        ]
        modes = source[required].load()
        modes.attrs.update(source.attrs)
    return modes


def _trapezoid_weights(points: np.ndarray) -> np.ndarray:
    from supernova.core.solver5_multigrid_evidence import trapezoid_weights

    return trapezoid_weights(points)


def _reclassification_table(
    modes: xr.Dataset,
    tracks: xr.Dataset,
    evidence: xr.Dataset,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ntracks = modes.sizes.get("track", 0)
    history_fields = (
        "frequency_khz",
        "backward_error",
        "xi_psi_overlap",
        "rms_width",
        "ipr",
    )
    evidence_fields = (
        "label",
        "verification_level",
        "overlap_metric",
        "power_density_metric",
        "alpha_width",
        "beta_ipr",
        "beta_kr",
        "coarse_space_retention_final",
        "minimum_relative_continuum_clearance",
        "uncertainty_normalized_clearance",
        "resonance_radii",
        "peak_distance_over_h",
        "resonant_power_fraction",
        "alfven_continuum_overlap",
        "sound_continuum_overlap",
        "mixed_continuum_overlap",
        "catalog_status",
        "reason_codes",
        "satisfied_reason_codes",
        "failed_reason_codes",
        "unknown_reason_codes",
    )
    h_history = np.asarray(tracks["h_peak"].values, dtype=float)
    width_history = np.asarray(tracks["rms_width"].values, dtype=float)
    ipr_history = np.asarray(tracks["ipr"].values, dtype=float)
    kr_history = np.asarray(tracks["kr"].values, dtype=float)
    for track in range(ntracks):
        legacy_label = (
            str(modes["legacy_label"].values[track])
            if "legacy_label" in modes
            else str(modes["label"].values[track])
        )
        row: dict[str, object] = {
            "track": track,
            "legacy_label": legacy_label,
            "width_over_h": json.dumps((width_history[track] / h_history[track]).tolist()),
            "h_times_ipr": json.dumps((h_history[track] * ipr_history[track]).tolist()),
            "h_times_kr": json.dumps((h_history[track] * kr_history[track]).tolist()),
        }
        for name in history_fields:
            stored = f"{name}_by_level"
            if stored in modes:
                row[f"{name}_per_resolution"] = json.dumps(
                    np.asarray(modes[stored].values[track]).tolist()
                )
        for name in evidence_fields:
            if name not in evidence:
                continue
            value = evidence[name].values[track]
            row[name] = value.item() if hasattr(value, "item") else value
        rows.append(row)
    return rows


def _run_reclassify(args: argparse.Namespace) -> None:
    """Reclassify saved roots without constructing eigenoperators or calling SLEPc."""
    from supernova.core.solver5_multigrid_evidence import (
        EvidenceV2Config,
        catalog_status_from_diagnostics,
        classify_tracks_evidence_v2,
    )

    artifact_dir = Path(args.artifact_dir)
    output_dir = artifact_dir if args.output_dir is None else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(artifact_dir)
    modes = _load_legacy_modes(artifact_dir)
    continuum = _load_portable_dataset(
        artifact_dir / "continuum.nc",
        ("omega", "frequency_khz", "branch_assignment_ambiguous"),
    )
    with xr.open_dataset(
        artifact_dir / "window_diagnostics.nc",
        engine="netcdf4",
    ) as source:
        diagnostics = source.load()
    level_status, run_status = catalog_status_from_diagnostics(diagnostics)
    expected_levels = int(dict(manifest.get("configuration", {})).get("levels", 0))
    raw_available = expected_levels > 0 and all(
        (artifact_dir / f"level_{level}.nc").is_file()
        for level in range(expected_levels)
    )
    source_choice = getattr(args, "source", "auto")
    use_raw = source_choice == "raw" or (
        source_choice == "auto" and raw_available
    )
    if source_choice == "raw" and not raw_available:
        raise FileNotFoundError("--source raw requested, but the complete level_*.nc set is absent")

    modes_for_table = modes
    identical_legacy_histories = None
    if use_raw:
        (
            level_solutions,
            tracks,
            _radial_levels,
            mixed_levels,
            _points,
            weights,
        ) = _retrack_raw_levels(artifact_dir, manifest)
        classification_continuum = continuum[["omega", "branch_assignment_ambiguous"]]
    else:
        tracks, weights = _legacy_modes_to_tracks(modes)
        classification_continuum = continuum[
            ["frequency_khz", "branch_assignment_ambiguous"]
        ].rename({"frequency_khz": "omega"})

    evidence = classify_tracks_evidence_v2(
        tracks,
        classification_continuum,
        weights,
        level_status,
        EvidenceV2Config(),
    )
    evidence.attrs.update(
        source_artifact_directory=str(artifact_dir.resolve()),
        source_modes="level_*.nc" if use_raw else "modes.nc",
        source_continuum="continuum.nc",
        source_window_diagnostics="window_diagnostics.nc",
        run_catalog_status=run_status,
        tracking_source="raw_level_retracking" if use_raw else "legacy_modes_tracks",
        slepc_was_run=False,
    )

    if use_raw:
        converter = _omega_to_khz_from_saved_continuum(continuum)
        reconstruction_points = int(
            dict(manifest.get("configuration", {})).get(
                "reconstruction_points",
                modes.sizes.get("x", 512),
            )
        )
        modes_for_table = _reconstruct_modes_from_levels(
            level_solutions,
            tracks,
            evidence,
            mixed_levels,
            converter,
            reconstruction_points,
        )
        legacy_track, legacy_label = _legacy_track_association(modes, tracks)
        modes_for_table["legacy_track"] = ("track", legacy_track)
        modes_for_table["legacy_label"] = ("track", legacy_label)
        mapped = legacy_track >= 0
        legacy_indices = np.asarray(modes["mode_index_by_level"].values, dtype=int)
        legacy_present = np.asarray(modes["present_by_level"].values, dtype=bool)
        source_indices = np.asarray(tracks["source_mode_index"].values, dtype=int)
        current_present = np.asarray(tracks["present"].values, dtype=bool)
        identical_legacy_histories = np.zeros(legacy_track.size, dtype=bool)
        identical_legacy_histories[mapped] = np.all(
            source_indices[mapped] == legacy_indices[legacy_track[mapped]],
            axis=1,
        ) & np.all(
            current_present[mapped] == legacy_present[legacy_track[mapped]],
            axis=1,
        )
        modes_for_table.attrs.update(
            source_artifact_directory=str(artifact_dir.resolve()),
            source_modes="level_*.nc",
            tracking_source="raw_level_retracking",
            classification_version="evidence_v2",
            slepc_was_run=False,
        )
        compact_tracks = tracks.drop_vars("xi_psi", errors="ignore")
        _write_dataset(compact_tracks, output_dir / "mode_tracks_evidence_v2.nc")
        _write_dataset(modes_for_table, output_dir / "modes_evidence_v2.nc")

    _write_dataset(evidence, output_dir / "classification_evidence_v2.nc")
    rows = _reclassification_table(modes_for_table, tracks, evidence)
    _write_summary_csv(rows, output_dir / "summary_evidence_v2.csv")

    before = Counter(np.asarray(modes["label"].values).astype(str).tolist())
    after = Counter(np.asarray(evidence["label"].values).astype(str).tolist())
    legacy_inconclusive = (
        np.asarray(modes_for_table["legacy_label"].values).astype(str) == "inconclusive"
        if "legacy_label" in modes_for_table
        else np.asarray(modes["label"].values).astype(str) == "inconclusive"
    )
    former_inconclusive = Counter(
        np.asarray(evidence["label"].values).astype(str)[legacy_inconclusive].tolist()
    )
    reason_counter: Counter[str] = Counter()
    for field, prefix in (
        ("reason_codes", "decision"),
        ("satisfied_reason_codes", "satisfied"),
        ("failed_reason_codes", "failed"),
        ("unknown_reason_codes", "unknown"),
    ):
        for value in np.asarray(evidence[field].values).astype(str):
            reason_counter.update(
                f"{prefix}:{code}" for code in value.split(";") if code
            )
    report = {
        "classification_version": "evidence_v2",
        "source_artifact_directory": str(artifact_dir.resolve()),
        "run_catalog_status": run_status,
        "level_catalog_status": level_status.tolist(),
        "before_label_counts": dict(sorted(before.items())),
        "after_label_counts": dict(sorted(after.items())),
        "previously_inconclusive_count": int(before.get("inconclusive", 0)),
        "previously_inconclusive_after": dict(sorted(former_inconclusive.items())),
        "previously_inconclusive_mapped_count": int(np.count_nonzero(legacy_inconclusive)),
        "most_common_reason_codes": reason_counter.most_common(20),
        "solver_rejected_count": int(after.get("solver_rejected", 0)),
        "catalog_status_is_per_mode_veto": False,
        "slepc_was_run": False,
        "tracking_source": "raw_level_retracking" if use_raw else "legacy_modes_tracks",
        "legacy_track_count": int(modes.sizes.get("track", 0)),
        "retracked_track_count": int(evidence.sizes.get("track", 0)),
    }
    if use_raw:
        report["raw_root_counts"] = list(tracks.attrs["raw_root_counts"])
        report["deduplicated_root_counts"] = list(
            tracks.attrs["deduplicated_root_counts"]
        )
        report["legacy_track_histories_identical"] = int(
            np.count_nonzero(identical_legacy_histories)
        )
        report["legacy_track_histories_changed"] = int(
            identical_legacy_histories.size
            - np.count_nonzero(identical_legacy_histories)
        )
        evidence_manifest = dict(manifest)
        evidence_manifest.update(
            schema_version=SCHEMA_VERSION,
            created_utc=datetime.now(timezone.utc).isoformat(),
            git_commit=_git_commit(),
            classification_version="evidence_v2",
            mode_count=int(evidence.sizes.get("track", 0)),
            classification_counts=dict(sorted(after.items())),
            source_manifest="manifest.json",
            slepc_was_run=False,
        )
        artifacts = dict(evidence_manifest.get("artifacts", {}))
        artifacts.update(
            modes="modes_evidence_v2.nc",
            classification="classification_evidence_v2.nc",
            mode_tracks="mode_tracks_evidence_v2.nc",
            summary="summary_evidence_v2.csv",
            reclassification_report="reclassification_report.json",
        )
        evidence_manifest["artifacts"] = artifacts
        (output_dir / "manifest_evidence_v2.json").write_text(
            json.dumps(evidence_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (output_dir / "reclassification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved evidence-v2 outputs to {output_dir}")


def _reacquisition_selection_rows(
    tracks: xr.Dataset,
    options,
    omega_to_khz,
    eligible: np.ndarray | None = None,
) -> list[dict[str, object]]:
    from supernova.core.solver5_multigrid_evidence import (
        select_reacquisition_candidates,
    )

    omega = np.asarray(tracks["omega"].values, dtype=np.complex128)
    rows: list[dict[str, object]] = []
    for level in range(1, tracks.sizes.get("level", 0)):
        candidates = select_reacquisition_candidates(
            tracks,
            level,
            options,
            eligible=eligible,
        )
        for rank, track in enumerate(candidates):
            frequency = float(omega_to_khz(omega[int(track), level - 1]))
            rows.append(
                {
                    "target_level": int(level),
                    "track": int(track),
                    "rank": int(rank),
                    "source_frequency_khz": frequency,
                    "width_over_h": float(
                        tracks["width_over_h"].values[int(track), level - 1]
                    ),
                    "coarse_space_retention": float(
                        tracks["coarse_space_retention"].values[
                            int(track), level - 1
                        ]
                    ),
                }
            )
    return rows


def _run_reacquire(args: argparse.Namespace) -> None:
    """Continue only saved tracks missing on their immediately finer mesh."""
    import supernova as sn
    from supernova.core.solver3 import _build_local_M_split
    from supernova.core.solver5_multigrid import (
        ModeClassificationThresholds,
        MultigridSearchOptions,
        _attach_enriched_residuals,
        _window_diagnostics_dataset,
    )
    from supernova.core.solver5_multigrid_evidence import (
        EvidenceV2Config,
        ReacquisitionOptions,
        catalog_status_from_diagnostics,
        classify_tracks_evidence_v2,
    )

    artifact_dir = Path(args.artifact_dir)
    base_manifest_path = artifact_dir / "manifest.json"
    if not base_manifest_path.is_file():
        raise FileNotFoundError(base_manifest_path)
    manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    configuration = dict(manifest.get("configuration", {}))
    options = ReacquisitionOptions(
        max_candidates_per_level=args.max_candidates_per_level,
        nev_multiplier=args.nev_multiplier,
        relative_half_width=args.relative_half_width,
        minimum_overlap=args.minimum_overlap,
    )
    compact_path = artifact_dir / "mode_tracks_evidence_v2.nc"
    if not compact_path.is_file():
        raise FileNotFoundError(
            f"Missing {compact_path}; run the raw reclassify command first."
        )
    selection_tracks = _load_portable_dataset(
        compact_path,
        ("omega", "present", "width_over_h", "coarse_space_retention"),
    )
    continuum = _load_portable_dataset(
        artifact_dir / "continuum.nc",
        ("omega", "frequency_khz", "branch_assignment_ambiguous"),
    )
    with xr.open_dataset(
        artifact_dir / "classification_evidence_v2.nc",
        engine="netcdf4",
    ) as source:
        labels = np.asarray(source["label"].values).astype(str)
        discrete_valid = np.asarray(source["discrete_valid"].values).astype(str)
    eligible = (labels == "unresolved") & (discrete_valid == "true")
    converter = _omega_to_khz_from_saved_continuum(continuum)
    selected = _reacquisition_selection_rows(
        selection_tracks,
        options,
        converter,
        eligible=eligible,
    )
    dry_report = {
        "source_artifact_directory": str(artifact_dir.resolve()),
        "execute_requested": bool(args.execute),
        "selected_window_count_before_cascades": len(selected),
        "selected_by_target_level": {
            str(level): sum(row["target_level"] == level for row in selected)
            for level in range(1, selection_tracks.sizes.get("level", 0))
        },
        "selection": selected,
        "options": asdict(options),
        "selection_semantics": (
            "forward adjacent-level gaps only; successful level-1 reacquisition may "
            "create additional level-2 candidates"
        ),
        "slepc_was_run": False,
    }
    (artifact_dir / "reacquisition_dry_run.json").write_text(
        json.dumps(dry_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not args.execute or not selected:
        print(json.dumps(dry_report, indent=2, sort_keys=True))
        print(f"Saved dry-run selection to {artifact_dir / 'reacquisition_dry_run.json'}")
        return

    case_args = argparse.Namespace(
        equilibrium=Path(configuration["equilibrium"]),
        density=Path(configuration["density"]),
        cocos_in=int(configuration["cocos_in"]),
        profile_points=int(configuration["profile_points"]),
        gamma_s=float(configuration["gamma_s"]),
        species=str(configuration["species"]),
    )
    equilibrium, continuum_solver = _load_case(case_args)
    multigrid = sn.superSolver5Multigrid(
        equilibrium,
        continuum_solver,
        _base_spline(
            int(configuration["base_spans"]),
            int(configuration["order"]),
            int(configuration["quadrature_order"]),
        ),
        levels=int(configuration["levels"]),
    )
    ntor = int(configuration["ntor"])
    mmax = int(configuration["mmax"])
    max_dm = int(configuration["max_dm"])
    regularizer = float(configuration.get("regularizer", 0.0))
    sb = multigrid.solver.build_solver_blocks(
        ntor=ntor,
        mmax=mmax,
        max_dm=max_dm,
    )
    local_split = _build_local_M_split(sb, regularizer=regularizer)
    projected = tuple(
        multigrid.solver._assemble_split_matrices_from_local(
            sb,
            mixed,
            local_split,
            apply_bcs=True,
            bc_kind="full",
            bc_tol=None,
            verbose=args.verbose,
        )
        for mixed in multigrid.mixed_levels
    )
    level_solutions, tracks, *_contract = _retrack_raw_levels(
        artifact_dir,
        manifest,
        include_eigenvectors=True,
    )
    level_solutions = list(level_solutions)
    search = MultigridSearchOptions(
        nev=int(configuration["nev"]),
        tol=float(configuration["tol"]),
        maxit=int(configuration["maxit"]),
        window_count=int(configuration.get("search_window_count", 1)),
        window_padding=float(configuration["window_padding"]),
        region_vscale=float(configuration["region_vscale"]),
        prefix="nova_multigrid_saved_reacq",
        options=_factorization_options(str(configuration["factorization"])),
    )
    thresholds = ModeClassificationThresholds()
    tracks, records = multigrid._reacquire_missing_roots(
        tracks=tracks,
        level_solutions=level_solutions,
        projected=projected,
        sb=sb,
        search=search,
        thresholds=thresholds,
        options=options,
        bc_tol=None,
        verbose=args.verbose,
        candidate_mask=eligible,
    )
    tracks = _attach_source_mode_indices(tracks, level_solutions)
    tracks = _attach_enriched_residuals(
        tracks,
        tuple(level_solutions),
        projected,
        multigrid.mixed_levels,
        multigrid.solver,
        sb,
        bc_tol=None,
    )
    with xr.open_dataset(
        artifact_dir / "window_diagnostics.nc",
        engine="netcdf4",
    ) as source:
        diagnostics = source.load()
    level_status, run_status = catalog_status_from_diagnostics(diagnostics)
    evidence = classify_tracks_evidence_v2(
        tracks,
        continuum[["omega", "branch_assignment_ambiguous"]],
        multigrid.master_weights,
        level_status,
        EvidenceV2Config(),
    )
    evidence.attrs.update(
        source_artifact_directory=str(artifact_dir.resolve()),
        source_modes="level_*.nc plus targeted reacquisition",
        tracking_source="raw_level_retracking_with_reacquisition",
        run_catalog_status=run_status,
        slepc_was_run=True,
        slepc_solve_count=len(records),
    )
    modes = _reconstruct_modes_from_levels(
        tuple(level_solutions),
        tracks,
        evidence,
        multigrid.mixed_levels,
        converter,
        int(configuration["reconstruction_points"]),
    )
    legacy_modes = _load_legacy_modes(artifact_dir)
    legacy_track, legacy_label = _legacy_track_association(legacy_modes, tracks)
    modes["legacy_track"] = ("track", legacy_track)
    modes["legacy_label"] = ("track", legacy_label)
    modes.attrs.update(
        source_artifact_directory=str(artifact_dir.resolve()),
        source_modes="level_*.nc plus targeted reacquisition",
        tracking_source="raw_level_retracking_with_reacquisition",
        classification_version="evidence_v2",
        slepc_was_run=True,
        slepc_solve_count=len(records),
    )

    names = {
        "modes": "modes_reacquired_evidence_v2.nc",
        "classification": "classification_reacquired_evidence_v2.nc",
        "mode_tracks": "mode_tracks_reacquired_evidence_v2.nc",
        "diagnostics": "reacquisition_diagnostics.nc",
        "report": "reacquisition_report.json",
    }
    for level, dataset in enumerate(level_solutions):
        _write_dataset(dataset, artifact_dir / f"level_{level}_reacquired.nc")
    _write_dataset(
        tracks.drop_vars("xi_psi", errors="ignore"),
        artifact_dir / names["mode_tracks"],
    )
    _write_dataset(evidence, artifact_dir / names["classification"])
    _write_dataset(modes, artifact_dir / names["modes"])
    reacquisition_diagnostics = _window_diagnostics_dataset(
        records,
        tracks.sizes.get("level", 0),
    )
    _write_dataset(reacquisition_diagnostics, artifact_dir / names["diagnostics"])

    labels = Counter(np.asarray(evidence["label"].values).astype(str).tolist())
    statuses = Counter(
        np.asarray(tracks["reacquisition_status"].values).astype(str).ravel().tolist()
    )
    report = {
        **dry_report,
        "execute_requested": True,
        "slepc_was_run": True,
        "slepc_solve_count": len(records),
        "label_counts": dict(sorted(labels.items())),
        "reacquisition_status_counts": dict(sorted(statuses.items())),
        "attempts": [
            {
                "target_level": int(record.level),
                "track": int(record.parent),
                "frequency_min": float(record.core_omega_min),
                "frequency_max": float(record.core_omega_max),
                "nconv": int(record.nconv),
                "nreturned": int(record.nreturned),
                "effective_nev": int(record.effective_nev),
                "catalog_status": str(record.catalog_status),
                "reason": str(record.reason),
            }
            for record in records
        ],
    }
    (artifact_dir / names["report"]).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    latest_manifest = dict(manifest)
    latest_manifest.update(
        schema_version=SCHEMA_VERSION,
        created_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=_git_commit(),
        classification_version="evidence_v2",
        mode_count=int(evidence.sizes.get("track", 0)),
        classification_counts=dict(sorted(labels.items())),
        source_manifest="manifest.json",
        slepc_was_run=True,
        slepc_solve_count=len(records),
    )
    artifacts = dict(latest_manifest.get("artifacts", {}))
    artifacts.update(names)
    artifacts["levels"] = [
        f"level_{level}_reacquired.nc" for level in range(len(level_solutions))
    ]
    latest_manifest["artifacts"] = artifacts
    (artifact_dir / "manifest_reacquired_evidence_v2.json").write_text(
        json.dumps(latest_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved targeted reacquisition artifacts to {artifact_dir}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def _summary_rows(modes: xr.Dataset) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for track in range(modes.sizes.get("track", 0)):
        row: dict[str, object] = {
            "track": track,
            "frequency_khz": float(modes["frequency_khz"].values[track]),
            "label": str(modes["label"].values[track]),
            "reason": str(modes["reason"].values[track]),
            "center_psi_N": float(modes["center"].values[track]),
            "radial_width_psi_N": float(modes["radial_width"].values[track]),
            "dominant_m": int(modes["dominant_m"].values[track]),
            "last_level": int(modes["last_level"].values[track]),
        }
        for name in (
            "classification_version",
            "verification_level",
            "catalog_status",
            "reason_codes",
            "numerically_valid",
            "discrete_valid",
            "chain_converged",
            "subspace_converged",
            "frequency_converged",
            "singular_refinement_scaling",
            "converged_chain",
            "strict_continuum_crossing",
            "has_continuum_crossing",
            "local_continuum_alignment",
            "strict_gap",
            "minimum_relative_continuum_clearance",
            "uncertainty_normalized_clearance",
            "continuum_band_power_fraction",
            "resonance_radii",
            "peak_distance_over_h",
            "resonant_power_fraction",
            "alfven_continuum_overlap",
            "sound_continuum_overlap",
            "mixed_continuum_overlap",
            "fine_frequency_change",
            "medium_fine_overlap",
            "alpha_width",
            "beta_ipr",
            "beta_kr",
            "coarse_space_retention_final",
            "width_ratio",
            "ipr_ratio",
        ):
            if name in modes:
                value = modes[name].values[track]
                row[name] = value.item() if hasattr(value, "item") else value
        rows.append(row)
    return rows


def _write_summary_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["track", "frequency_khz", "label"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _overview_figure(
    modes: xr.Dataset,
    continuum: xr.Dataset,
    frequency_band: tuple[float, float],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(11.5, 6.8))
    psi = np.asarray(continuum.coords["psi"].values, dtype=float)
    continuum_khz = np.asarray(continuum["frequency_khz"].values, dtype=float)
    axis.plot(psi, continuum_khz, color="#708090", lw=0.45, alpha=0.34)
    labeled_modes = False
    for label in LABEL_ORDER:
        selected = np.flatnonzero(np.asarray(modes["label"].values) == label)
        if selected.size == 0:
            continue
        labeled_modes = True
        style = LABEL_STYLE[label]
        axis.errorbar(
            modes["center"].values[selected],
            modes["frequency_khz"].values[selected],
            xerr=modes["radial_width"].values[selected],
            fmt="o",
            color=style["color"],
            ms=5.5,
            elinewidth=1.1,
            capsize=0,
            alpha=0.9,
            label=f"{style['title']} ({selected.size})",
        )
    axis.axhspan(*frequency_band, color="#3B82F6", alpha=0.045)
    axis.set(
        xlim=(0.0, 1.0),
        ylim=(max(0.0, frequency_band[0] - 5.0), frequency_band[1] + 5.0),
        xlabel=r"Normalized poloidal flux $\psi_N$",
        ylabel="Cyclic frequency [kHz]",
        title="solver5 multigrid classifications over the slow-sound continuum",
    )
    axis.grid(True, alpha=0.18)
    if labeled_modes:
        axis.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _continuum_for_storage(result, equilibrium) -> xr.Dataset:
    continuum = result.continuum.copy()
    continuum["frequency_khz"] = (
        ("psi", "branch"),
        _omega_to_khz(equilibrium, continuum["omega"].values),
    )
    continuum.frequency_khz.attrs.update(
        units="kHz",
        definition="Re(omega*omegaA)/(2*pi)",
    )
    return continuum


def _slow_sound_display_continuum(
    continuum_solver,
    *,
    ntor: int,
    mmax: int,
    points: int,
) -> xr.Dataset:
    """Compute the readable slow-sound background used only for plotting."""
    psi = np.linspace(0.0, 1.0, int(points))
    eigenvalues = continuum_solver.solve(
        ntor=ntor,
        which="slow_sound",
        normalize_eigvals=True,
        psi_vals=psi,
        classify=False,
        mmax=mmax,
    )
    frequency_khz = np.sqrt(eigenvalues.real).to("kHz").v / (2.0 * np.pi)
    return xr.Dataset(
        data_vars={
            "frequency_khz": (
                ("psi", "branch"),
                np.asarray(frequency_khz, dtype=float),
            )
        },
        coords={
            "psi": psi,
            "branch": np.arange(np.asarray(frequency_khz).shape[1], dtype=np.int64),
        },
        attrs={
            "continuum_kind": "slow_sound",
            "purpose": "display_only",
            "classification_continuum": "full sound-Alfven continuum in continuum.nc",
            "frequency_definition": "sqrt(Re(lambda))/(2*pi) in cyclic kHz",
            "radial_coordinate": "psi_N",
        },
    )


def _artifact_manifest(
    *,
    args: argparse.Namespace | None,
    modes: xr.Dataset,
    frequency_band: tuple[float, float],
    demo: bool,
) -> dict[str, object]:
    labels, counts = np.unique(modes["label"].values.astype(str), return_counts=True)
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "demo": demo,
        "frequency_khz": list(map(float, frequency_band)),
        "mode_count": int(modes.sizes.get("track", 0)),
        "classification_counts": {
            str(label): int(count) for label, count in zip(labels, counts)
        },
        "radial_coordinate": "psi_N",
        "frequency_definition": "Re(omega*omegaA)/(2*pi) in cyclic kHz",
        "artifacts": {
            "modes": "modes.nc",
            "continuum": "continuum.nc",
            "display_continuum": "display_continuum.nc",
            "window_diagnostics": "window_diagnostics.nc",
            "summary": "summary.csv",
            "overview": "overview.png",
            "static_preview": "explorer_preview.html",
        },
        "labels": {
            key: {"title": value["title"], "color": value["color"]}
            for key, value in LABEL_STYLE.items()
        },
    }
    if args is not None:
        manifest["configuration"] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "command"
        }
        manifest["equilibrium_sha256"] = _sha256(args.equilibrium)
        manifest["density_sha256"] = _sha256(args.density)
    return manifest


def _save_artifacts(
    artifact_dir: Path,
    modes: xr.Dataset,
    continuum: xr.Dataset,
    diagnostics: xr.Dataset,
    frequency_band: tuple[float, float],
    manifest: Mapping[str, object],
    *,
    display_continuum: xr.Dataset | None = None,
    raw_levels: tuple[xr.Dataset, ...] = (),
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_dataset(modes, artifact_dir / "modes.nc")
    _write_dataset(continuum, artifact_dir / "continuum.nc")
    if display_continuum is None:
        display_continuum = continuum
    _write_dataset(display_continuum, artifact_dir / "display_continuum.nc")
    _write_dataset(diagnostics, artifact_dir / "window_diagnostics.nc")
    for level, dataset in enumerate(raw_levels):
        _write_dataset(dataset, artifact_dir / f"level_{level}.nc")
    rows = _summary_rows(modes)
    _write_summary_csv(rows, artifact_dir / "summary.csv")
    _overview_figure(
        modes,
        display_continuum,
        frequency_band,
        artifact_dir / "overview.png",
    )
    _write_static_preview(artifact_dir, modes, display_continuum)
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_static_preview(
    artifact_dir: Path,
    modes: xr.Dataset,
    continuum: xr.Dataset,
) -> None:
    """Write a server-free first-view preview of the Dash dashboard."""
    import plotly.io as pio

    labels = np.asarray(modes["label"].values).astype(str)
    spectrum = _spectrum_figure(modes, continuum, LABEL_ORDER)
    spectrum_html = pio.to_html(
        spectrum,
        full_html=False,
        include_plotlyjs="inline",
        config={"displaylogo": False, "responsive": True},
    )
    if modes.sizes.get("track", 0) == 0:
        html_text = f"""<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width'><title>solver5 multigrid preview</title>
<style>body{{margin:0;background:#eef3f8;color:#17233c;font-family:Inter,system-ui,sans-serif}}
header{{background:linear-gradient(118deg,#101c35,#1f355e 68%,#28527a);color:white;padding:22px 30px}}
h1{{margin:4px 0;font-size:25px}}main{{padding:20px;max-width:1500px;margin:auto}}
.notice,.card{{background:white;border:1px solid #dce3ec;border-radius:13px;padding:15px;margin-bottom:14px}}
.notice{{border-left:5px solid #e39a31}}</style></head><body>
<header><small>superNOVA / solver5</small><h1>Multigrid mode explorer</h1></header>
<main><div class='notice'><b>No classified roots were returned.</b><br>
The continuum and search diagnostics were saved; broaden the band, increase resolution,
or inspect window completion before interpreting the empty result.</div>
<section class='card'>{spectrum_html}</section></main></body></html>"""
        (artifact_dir / "explorer_preview.html").write_text(
            html_text,
            encoding="utf-8",
        )
        return
    physical = np.flatnonzero(labels == "physical_gap")
    track = int(physical[0] if physical.size else 0)
    profile = _profile_figure(artifact_dir, track, "xi_psi", "magnitude", 10)
    convergence = _convergence_figure(modes, track)
    profile_html = pio.to_html(
        profile,
        full_html=False,
        include_plotlyjs=False,
        config={"displaylogo": False, "responsive": True},
    )
    convergence_html = pio.to_html(
        convergence,
        full_html=False,
        include_plotlyjs=False,
        config={"displaylogo": False, "responsive": True},
    )
    label = labels[track]
    style = LABEL_STYLE.get(label, LABEL_STYLE["unresolved"])
    counts = {name: int(np.count_nonzero(labels == name)) for name in LABEL_ORDER}
    metrics = "".join(
        f"<div class='metric'><span>{LABEL_STYLE[name]['title']}</span>"
        f"<strong style='color:{LABEL_STYLE[name]['color']}'>{counts[name]}</strong></div>"
        for name in LABEL_ORDER
    )
    evidence = {
        "Frequency": f"{float(modes['frequency_khz'].values[track]):.5f} kHz",
        "Dominant m": str(int(modes["dominant_m"].values[track])),
        "Continuum clearance": _format_metric(
            modes["minimum_relative_continuum_clearance"].values[track],
            percent=True,
        ),
        "Continuum-band power": _format_metric(
            modes["continuum_band_power_fraction"].values[track],
            percent=True,
        ),
        "Fine-grid drift": _format_metric(
            modes["fine_frequency_change"].values[track],
            scientific=True,
        ),
        "Medium/fine overlap": _format_metric(
            modes["medium_fine_overlap"].values[track]
        ),
    }
    evidence_html = "".join(
        f"<div class='ev'><span>{name}</span><strong>{value}</strong></div>"
        for name, value in evidence.items()
    )
    html_text = f"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>solver5 multigrid mode explorer preview</title>
<style>
:root{{--ink:#17233c;--muted:#64748b;--line:#dce3ec;--bg:#eef3f8}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,system-ui,sans-serif}}
.top{{background:linear-gradient(118deg,#101c35,#1f355e 68%,#28527a);color:white;padding:22px 30px}}
.eyebrow{{color:#8fd7e1;font-weight:700;letter-spacing:.13em;text-transform:uppercase;font-size:11px}}
h1{{font-size:25px;margin:5px 0 4px}} .sub{{color:#c7d4e8;font-size:13px}}
.content{{padding:18px 22px;max-width:1800px;margin:auto}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}}
.metric,.card{{background:white;border:1px solid var(--line);border-radius:13px;box-shadow:0 4px 13px rgba(20,38,65,.05)}}
.metric{{padding:12px 15px}} .metric span{{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;font-weight:700;letter-spacing:.06em}}
.metric strong{{font-size:23px}} .card{{padding:7px 10px;margin-bottom:14px;overflow:hidden}}
.details{{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(300px,.65fr);gap:14px}}
.evidence{{padding:17px 18px}} .title{{display:flex;align-items:center;gap:10px}}
.title h2{{font-size:19px;margin:0}} .pill{{color:white;border-radius:99px;padding:5px 10px;font-size:11px;font-weight:750}}
.reason{{margin-top:9px;background:#f6f8fb;border:1px solid #e4e9f0;border-radius:9px;padding:10px;font:11px/1.45 ui-monospace,monospace;color:#4d5c72}}
.evgrid{{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:12px}}
.ev{{border-left:3px solid #b8c5d6;padding:7px 9px;background:#fbfcfe}} .ev span{{display:block;color:#6b788d;font-size:10px;text-transform:uppercase}} .ev strong{{font-size:14px}}
.notice{{background:#e9f4f7;color:#34556c;border-radius:9px;padding:9px 12px;font-size:11px;margin-bottom:14px}}
@media(max-width:900px){{.metrics{{grid-template-columns:1fr 1fr}}.details{{grid-template-columns:1fr}}}}
</style></head><body>
<header class='top'><div class='eyebrow'>superNOVA / solver5</div><h1>Multigrid mode explorer</h1><div class='sub'>Three-level radial persistence, continuum evidence, and harmonic structure</div></header>
<main class='content'><div class='notice'>Static first-view preview. The background is the slow-sound continuum; labels retain full-continuum evidence. Run the Dash <code>serve</code> command for mode-type toggles, selection, and field controls.</div>
<section class='metrics'>{metrics}</section><section class='card'>{spectrum_html}</section>
<section class='details'><div class='card'>{profile_html}</div><div class='card evidence'>
<div class='title'><h2>Track {track}</h2><span class='pill' style='background:{style['color']}'>{style['title']}</span></div>
<div class='reason'>{str(modes['reason'].values[track])}</div><div class='evgrid'>{evidence_html}</div></div></section>
<section class='card'>{convergence_html}</section></main></body></html>"""
    (artifact_dir / "explorer_preview.html").write_text(html_text, encoding="utf-8")


def _run_solve(args: argparse.Namespace) -> None:
    import supernova as sn

    frequency_band = tuple(np.sort(np.asarray(args.frequency_khz, dtype=float)))
    window_count = _window_count(frequency_band, args.window_width_khz)
    equilibrium, continuum_solver = _load_case(args)
    multigrid = sn.superSolver5Multigrid(
        equilibrium,
        continuum_solver,
        _base_spline(args.base_spans, args.order, args.quadrature_order),
        levels=args.levels,
    )
    search = sn.MultigridSearchOptions(
        nev=args.nev,
        tol=args.tol,
        maxit=args.maxit,
        window_count=window_count,
        window_padding=args.window_padding,
        region_vscale=args.region_vscale,
        options=_factorization_options(args.factorization),
    )
    print(
        f"Searching n={args.ntor}, mmax={args.mmax}, "
        f"{frequency_band[0]:.3f}-{frequency_band[1]:.3f} kHz in "
        f"{window_count} fixed windows"
    )
    result = multigrid.solve_pep(
        ntor=args.ntor,
        omega_range=_omega_range(equilibrium, frequency_band),
        mmax=args.mmax,
        max_dm=args.max_dm,
        regularizer=args.regularizer,
        search=search,
        adaptive_discovery=(
            sn.AdaptiveDiscoveryOptions() if args.adaptive_discovery else None
        ),
        reacquisition=(
            None
            if args.no_reacquisition
            else sn.ReacquisitionOptions(
                max_candidates_per_level=args.reacquisition_max_candidates
            )
        ),
        verbose=args.verbose,
    )
    modes = _reconstruct_final_modes(
        result,
        multigrid,
        equilibrium,
        args.reconstruction_points,
    )
    modes.attrs.update(
        ntor=args.ntor,
        mmax=args.mmax,
        max_dm=args.max_dm,
        gamma_s=args.gamma_s,
        species=args.species,
        regularizer=args.regularizer,
        frequency_min_khz=frequency_band[0],
        frequency_max_khz=frequency_band[1],
        base_spans=args.base_spans,
        levels=args.levels,
        finest_spans=args.base_spans * 2 ** (args.levels - 1),
        search_window_count=window_count,
        search_window_width_khz=(frequency_band[1] - frequency_band[0]) / window_count,
        search_window_padding=args.window_padding,
        search_region_vscale=args.region_vscale,
        search=repr(search),
    )
    continuum = _continuum_for_storage(result, equilibrium)
    display_continuum = _slow_sound_display_continuum(
        continuum_solver,
        ntor=args.ntor,
        mmax=args.mmax,
        points=args.display_continuum_points,
    )
    manifest = _artifact_manifest(
        args=args,
        modes=modes,
        frequency_band=frequency_band,
        demo=False,
    )
    raw_levels = result.level_solutions if args.save_raw_levels else ()
    _save_artifacts(
        args.artifact_dir,
        modes,
        continuum,
        result.window_diagnostics,
        frequency_band,
        manifest,
        display_continuum=display_continuum,
        raw_levels=raw_levels,
    )
    print(f"Saved artifacts to {args.artifact_dir}")
    for label in LABEL_ORDER:
        count = int(np.count_nonzero(modes["label"].values == label))
        print(f"  {LABEL_STYLE[label]['title']:22s}: {count}")


def _demo_datasets() -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, tuple[float, float]]:
    rng = np.random.default_rng(20260818)
    frequency_band = (50.0, 150.0)
    ntracks, nlevels, nx = 16, 3, 192
    m_pol = np.arange(12, 32, dtype=np.int64)
    x = np.linspace(0.0, 1.0, nx)
    labels = np.asarray(
        [
            "physical_gap",
            "continuum_discretization",
            "physical_gap",
            "unresolved",
            "continuum_coupled_candidate",
            "solver_rejected",
            "physical_gap",
            "continuum_discretization",
            "unresolved",
            "discretization_artifact",
            "physical_gap",
            "continuum_coupled_candidate",
            "physical_gap",
            "unresolved",
            "solver_rejected",
            "continuum_discretization",
        ],
        dtype="U32",
    )
    frequencies = np.linspace(56.0, 143.0, ntracks) + rng.normal(0.0, 1.2, ntracks)
    centers = np.linspace(0.12, 0.90, ntracks)
    widths = np.where(labels == "continuum_discretization", 0.030, 0.090)
    widths = np.where(labels == "solver_rejected", 0.018, widths)
    fields = {
        name: np.zeros((ntracks, nx, m_pol.size), dtype=np.complex128)
        for name in FIELD_TITLES
    }
    dominant_m = np.empty(ntracks, dtype=np.int64)
    for track in range(ntracks):
        dominant_index = int((3 * track + 4) % m_pol.size)
        dominant_m[track] = m_pol[dominant_index]
        envelope = np.exp(-0.5 * ((x - centers[track]) / widths[track]) ** 2)
        harmonic = np.exp(
            -0.5 * ((np.arange(m_pol.size) - dominant_index) / 2.2) ** 2
        )
        phase = np.exp(1.0j * (0.55 * x[:, None] * (m_pol[None, :] - dominant_m[track])))
        xi = envelope[:, None] * harmonic[None, :] * phase
        xi /= np.sqrt(np.trapezoid(np.sum(np.abs(xi) ** 2, axis=1), x))
        fields["xi_psi"][track] = xi
        fields["p1"][track] = 0.34j * xi * (1.0 - 0.35 * x[:, None])
        fields["xi_s"][track] = 0.62 * xi * np.cos(np.pi * x[:, None])
        fields["div_xi"][track] = -0.28j * xi * np.sin(2.0 * np.pi * x[:, None])

    level = np.arange(nlevels)
    drift = np.asarray([0.018, 0.0045, 0.0])
    signed = np.where(np.arange(ntracks) % 2, -1.0, 1.0)
    frequency_levels = frequencies[:, None] * (1.0 + signed[:, None] * drift)
    present = np.ones((ntracks, nlevels), dtype=bool)
    present[np.isin(labels, ["solver_rejected", "discretization_artifact"]), 0] = False
    frequency_levels[~present] = np.nan
    overlap = np.tile(np.asarray([np.nan, 0.965, 0.994]), (ntracks, 1))
    overlap[labels == "continuum_discretization", -1] = 0.972
    overlap[labels == "solver_rejected", -1] = 0.61
    error = np.tile(np.asarray([2.0e-7, 4.0e-10, 8.0e-12]), (ntracks, 1))
    error[labels == "solver_rejected", -1] = 2.0e-5
    reasons = np.asarray(
        [
            {
                "physical_gap": "converged_three_level_chain;strict_continuum_gap",
                "continuum_discretization": "same_sequence_local_continuum_alignment;singular_mesh_scaling",
                "continuum_coupled_candidate": "global_limit_with_continuum_crossing",
                "solver_rejected": "final_root_failed_algebraic_validity",
                "discretization_artifact": "credible_targeted_absence;independent_mesh_nonconvergence",
                "unresolved": "insufficient_or_conflicting_evidence",
            }[label]
            for label in labels
        ],
        dtype="U160",
    )
    modes = xr.Dataset(
        data_vars={
            **{
                name: (("track", "x", "m_pol"), values)
                for name, values in fields.items()
            },
            "omega": ("track", frequencies.astype(np.complex128) / 100.0),
            "frequency_khz": ("track", frequencies),
            "center": ("track", centers),
            "radial_width": ("track", widths),
            "dominant_m": ("track", dominant_m),
            "normalization_scale": ("track", np.ones(ntracks)),
            "window_id": ("track", np.arange(ntracks)),
            "window_depth": ("track", np.full(ntracks, 4)),
            "frequency_khz_by_level": (("track", "level"), frequency_levels),
            "present_by_level": (("track", "level"), present),
            "backward_error_by_level": (("track", "level"), error),
            "xi_psi_overlap_by_level": (("track", "level"), overlap),
            "frequency_change_by_level": (
                ("track", "level"),
                np.abs(frequency_levels / frequencies[:, None] - 1.0),
            ),
            "rms_width_by_level": (
                ("track", "level"),
                widths[:, None] * np.asarray([1.16, 1.05, 1.0]),
            ),
            "ipr_by_level": (
                ("track", "level"),
                (1.0 / widths[:, None]) * np.asarray([0.86, 0.95, 1.0]),
            ),
            "label": ("track", labels),
            "reason": ("track", reasons),
            "last_level": ("track", np.full(ntracks, 2, dtype=np.int64)),
            "numerically_valid": ("track", labels != "solver_rejected"),
            "chain_numerically_valid": ("track", labels != "solver_rejected"),
            "converged_chain": (
                "track",
                np.isin(labels, ["physical_gap", "continuum_coupled_candidate"]),
            ),
            "present_all_levels": ("track", np.all(present, axis=1)),
            "assignment_ambiguous": ("track", labels == "unresolved"),
            "strict_continuum_crossing": (
                "track",
                np.isin(labels, ["continuum_discretization", "continuum_coupled_candidate"]),
            ),
            "strict_gap": ("track", labels == "physical_gap"),
            "minimum_relative_continuum_clearance": (
                "track",
                np.where(labels == "physical_gap", 0.032, 0.0015),
            ),
            "continuum_band_power_fraction": (
                "track",
                np.where(labels == "continuum_discretization", 0.58, 0.025),
            ),
            "fine_frequency_change": (
                "track",
                np.abs(frequency_levels[:, -1] / frequency_levels[:, -2] - 1.0),
            ),
            "medium_fine_overlap": ("track", overlap[:, -1]),
            "width_ratio": ("track", np.full(ntracks, 0.95)),
            "ipr_ratio": ("track", np.full(ntracks, 1.05)),
            "continuum_complete": ("track", np.ones(ntracks, dtype=bool)),
        },
        coords={"track": np.arange(ntracks), "level": level, "x": x, "m_pol": m_pol},
        attrs={
            "schema_version": SCHEMA_VERSION,
            "demo": True,
            "ntor": 4,
            "mmax": 68,
            "radial_coordinate": "psi_N",
            "frequency_min_khz": frequency_band[0],
            "frequency_max_khz": frequency_band[1],
            "normalization": "integral sum_m |xi_psi_m|^2 dpsi_N = 1",
            "phase_alignment": "largest-magnitude xi_psi sample made real-positive",
        },
    )

    psi = np.linspace(0.0, 1.0, 256)
    branch = np.arange(24)
    base = 42.0 + 4.4 * branch[None, :]
    slope = (psi[:, None] - 0.5) * (18.0 * np.sin(0.75 * branch)[None, :])
    ripple = 3.5 * np.sin(2.0 * np.pi * psi[:, None] + 0.35 * branch[None, :])
    continuum_khz = base + slope + ripple
    continuum = xr.Dataset(
        data_vars={
            "omega": (("psi", "branch"), continuum_khz.astype(np.complex128) / 100.0),
            "frequency_khz": (("psi", "branch"), continuum_khz),
            "branch_assignment_ambiguous": (
                ("psi", "branch"),
                np.zeros_like(continuum_khz, dtype=bool),
            ),
        },
        coords={"psi": psi, "branch": branch},
        attrs={"demo": True, "omega_normalization": "omegaA"},
    )
    diagnostics = xr.Dataset(
        data_vars={
            "level_complete": ("level", np.ones(nlevels, dtype=bool)),
            "valid": (("level", "window"), np.ones((nlevels, 5), dtype=bool)),
            "nreturned": (
                ("level", "window"),
                rng.integers(2, 8, size=(nlevels, 5)),
            ),
        },
        coords={"level": level, "window": np.arange(5)},
        attrs={"demo": True, "omega_min": 0.5, "omega_max": 1.5},
    )
    return modes, continuum, diagnostics, frequency_band


def _run_demo(args: argparse.Namespace) -> None:
    modes, continuum, diagnostics, frequency_band = _demo_datasets()
    manifest = _artifact_manifest(
        args=None,
        modes=modes,
        frequency_band=frequency_band,
        demo=True,
    )
    _save_artifacts(
        args.artifact_dir,
        modes,
        continuum,
        diagnostics,
        frequency_band,
        manifest,
        display_continuum=continuum.isel(branch=slice(0, None, 2)).assign_attrs(
            continuum_kind="slow_sound",
            purpose="display_only",
        ),
    )
    print(f"Saved deterministic GUI demo to {args.artifact_dir}")


def _load_manifest(artifact_dir: Path) -> dict[str, object]:
    reacquired_path = artifact_dir / "manifest_reacquired_evidence_v2.json"
    evidence_path = artifact_dir / "manifest_evidence_v2.json"
    if reacquired_path.is_file():
        path = reacquired_path
    elif evidence_path.is_file():
        path = evidence_path
    else:
        path = artifact_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing artifact manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported artifact schema {manifest.get('schema_version')!r}; "
            f"expected one of {SUPPORTED_SCHEMA_VERSIONS}."
        )
    return manifest


def _mode_artifact_path(
    artifact_dir: Path,
    manifest: Mapping[str, object] | None = None,
) -> Path:
    if manifest is None:
        has_manifest = (artifact_dir / "manifest.json").is_file() or (
            artifact_dir / "manifest_evidence_v2.json"
        ).is_file() or (
            artifact_dir / "manifest_reacquired_evidence_v2.json"
        ).is_file()
        manifest = _load_manifest(artifact_dir) if has_manifest else {}
    configured = dict(manifest.get("artifacts", {})).get("modes")
    if configured:
        path = artifact_dir / str(configured)
        if path.is_file():
            return path
    evidence_path = artifact_dir / "modes_evidence_v2.nc"
    return evidence_path if evidence_path.is_file() else artifact_dir / "modes.nc"


def _resolve_serve_artifact_dir(artifact_dir: Path) -> Path:
    """Prefer production artifacts, falling back to the bundled demo run."""
    artifact_dir = Path(artifact_dir)
    if (artifact_dir / "manifest.json").is_file():
        return artifact_dir
    demo_dir = artifact_dir / "demo"
    if (demo_dir / "manifest.json").is_file():
        return demo_dir
    raise FileNotFoundError(
        "No explorer artifacts were found. Checked "
        f"{artifact_dir / 'manifest.json'} and {demo_dir / 'manifest.json'}. "
        "Run the demo or solve command first, or pass --artifact-dir explicitly."
    )


def _load_explorer_metadata(artifact_dir: Path):
    manifest = _load_manifest(artifact_dir)
    mode_path = _mode_artifact_path(artifact_dir, manifest)
    with xr.open_dataset(mode_path, engine="netcdf4") as dataset:
        names = [
            "frequency_khz",
            "center",
            "radial_width",
            "dominant_m",
            "label",
            "reason",
            "last_level",
            "minimum_relative_continuum_clearance",
            "continuum_band_power_fraction",
            "fine_frequency_change",
            "medium_fine_overlap",
            "backward_error_by_level",
            "frequency_khz_by_level",
            "xi_psi_overlap_by_level",
            "rms_width_by_level",
            "ipr_by_level",
            "peak_radius_by_level",
            "present_by_level",
            "width_over_h_by_level",
            "h_times_ipr_by_level",
            "h_times_kr_by_level",
            "coarse_space_retention_by_level",
            "kr_by_level",
        ]
        available = [name for name in names if name in dataset]
        modes = dataset[available].load()
        modes.attrs.update(dataset.attrs)
    evidence_name = dict(manifest.get("artifacts", {})).get(
        "classification",
        "classification_evidence_v2.nc",
    )
    evidence_path = artifact_dir / str(evidence_name)
    if evidence_path.is_file():
        with xr.open_dataset(evidence_path, engine="netcdf4") as dataset:
            migrated = dataset.load()
            for name, variable in migrated.data_vars.items():
                modes[name] = variable
            modes.attrs.update(migrated.attrs)
    display_path = artifact_dir / "display_continuum.nc"
    if not display_path.is_file():
        display_path = artifact_dir / "continuum.nc"
    with xr.open_dataset(display_path, engine="netcdf4") as dataset:
        continuum = dataset[["frequency_khz"]].load()
        continuum.attrs.update(dataset.attrs)
    with xr.open_dataset(artifact_dir / "continuum.nc", engine="netcdf4") as dataset:
        full_continuum = dataset[["frequency_khz"]].load()
        full_continuum.attrs.update(dataset.attrs)
        full_continuum.attrs["continuum_kind"] = "full_sound_alfven"
    return manifest, modes, continuum, full_continuum


def _load_mode_field(artifact_dir: Path, track: int, field: str):
    if field not in FIELD_TITLES:
        raise ValueError(f"Unknown field {field!r}.")
    with xr.open_dataset(_mode_artifact_path(artifact_dir), engine="netcdf4") as dataset:
        real = np.asarray(dataset[f"{field}_real"].isel(track=track).values)
        imag = np.asarray(dataset[f"{field}_imag"].isel(track=track).values)
        x = np.asarray(dataset.coords["x"].values)
        m_pol = np.asarray(dataset.coords["m_pol"].values)
    return x, m_pol, real + 1.0j * imag


def _spectrum_figure(modes: xr.Dataset, continuum: xr.Dataset, labels: Iterable[str]):
    import plotly.graph_objects as go

    selected_labels = set(labels)
    continuum_kind = str(continuum.attrs.get("continuum_kind", "slow_sound"))
    continuum_title = (
        "Full sound-Alfven"
        if continuum_kind == "full_sound_alfven"
        else "Slow-sound"
    )
    psi = np.asarray(continuum.coords["psi"].values, dtype=float)
    frequency = np.asarray(continuum["frequency_khz"].values, dtype=float)
    continuum_x: list[float | None] = []
    continuum_y: list[float | None] = []
    for branch in range(frequency.shape[1]):
        continuum_x.extend(psi.tolist())
        continuum_x.append(None)
        continuum_y.extend(frequency[:, branch].tolist())
        continuum_y.append(None)
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=continuum_x,
            y=continuum_y,
            mode="lines",
            line={"color": "rgba(73, 89, 108, 0.24)", "width": 0.7},
            hoverinfo="skip",
            name=f"{continuum_title} continuum",
        )
    )
    mode_labels = np.asarray(modes["label"].values).astype(str)
    for label in LABEL_ORDER:
        if label not in selected_labels:
            continue
        indices = np.flatnonzero(mode_labels == label)
        if indices.size == 0:
            continue
        style = LABEL_STYLE[label]
        hover = [
            (
                f"<b>Track {index}</b><br>{style['title']}<br>"
                f"f={modes['frequency_khz'].values[index]:.4f} kHz<br>"
                f"psi center={modes['center'].values[index]:.4f}<br>"
                f"m dominant={int(modes['dominant_m'].values[index])}"
            )
            for index in indices
        ]
        figure.add_trace(
            go.Scatter(
                x=modes["center"].values[indices],
                y=modes["frequency_khz"].values[indices],
                mode="markers",
                marker={
                    "color": style["color"],
                    "symbol": style["symbol"],
                    "size": 10,
                    "line": {"color": "white", "width": 1.2},
                },
                error_x={
                    "type": "data",
                    "array": modes["radial_width"].values[indices],
                    "color": style["color"],
                    "thickness": 1.2,
                    "width": 0,
                },
                ids=[f"track-{int(index)}" for index in indices],
                customdata=indices[:, None],
                hovertext=hover,
                hovertemplate="%{hovertext}<extra></extra>",
                name=f"{style['title']} ({indices.size})",
            )
        )
    band = np.asarray(
        [modes.attrs.get("frequency_min_khz", 50.0), modes.attrs.get("frequency_max_khz", 150.0)],
        dtype=float,
    )
    figure.update_layout(
        template="plotly_white",
        margin={"l": 64, "r": 22, "t": 46, "b": 56},
        title={
            "text": f"{continuum_title} continuum and classified mode centers",
            "x": 0.02,
        },
        xaxis={"title": "Normalized poloidal flux psi_N", "range": [0, 1]},
        yaxis={"title": "Cyclic frequency [kHz]", "range": [max(0, band[0] - 5), band[1] + 5]},
        legend={"orientation": "h", "y": 1.02, "x": 1.0, "xanchor": "right", "yanchor": "bottom"},
        hovermode="closest",
        clickmode="event+select",
        uirevision="spectrum",
    )
    return figure


def _resolution_diagnostic_figure(
    modes: xr.Dataset,
    continuum: xr.Dataset,
    labels: Iterable[str],
    color_by: str,
):
    """Plot every track/level root over the selected continuum background."""
    import plotly.graph_objects as go

    selected_labels = set(labels)
    psi = np.asarray(continuum.coords["psi"].values, dtype=float)
    continuum_frequency = np.asarray(continuum["frequency_khz"].values, dtype=float)
    figure = go.Figure()
    for branch in range(continuum_frequency.shape[1]):
        figure.add_trace(
            go.Scattergl(
                x=psi,
                y=continuum_frequency[:, branch],
                mode="lines",
                line={"color": "rgba(73, 89, 108, 0.18)", "width": 0.55},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    frequency = np.asarray(modes["frequency_khz_by_level"].values, dtype=float)
    present = np.asarray(modes["present_by_level"].values, dtype=bool)
    if "peak_radius_by_level" in modes:
        peak = np.asarray(modes["peak_radius_by_level"].values, dtype=float)
    else:
        peak = np.broadcast_to(
            np.asarray(modes["center"].values, dtype=float)[:, None],
            frequency.shape,
        )
    labels_by_track = np.asarray(modes["label"].values).astype(str)
    track_index, level_index = np.indices(frequency.shape)
    finite = present & np.isfinite(frequency) & np.isfinite(peak)
    finite &= np.isin(labels_by_track[:, None], tuple(selected_labels))

    nlevel = frequency.shape[1]
    base_spans = int(modes.attrs.get("base_spans", 0) or 0)
    h = (
        1.0 / (base_spans * 2.0 ** np.arange(nlevel))
        if base_spans
        else np.full(nlevel, np.nan)
    )
    numeric_title = {
        "width_over_h": "width / h",
        "h_times_ipr": "h * IPR",
        "h_times_kr": "h * k_r",
        "coarse_space_retention": "fixed-coarse retention",
    }
    numeric = color_by in numeric_title
    if numeric:
        stored_name = f"{color_by}_by_level"
        if stored_name in modes:
            color = np.asarray(modes[stored_name].values, dtype=float)
        elif color_by == "width_over_h" and "rms_width_by_level" in modes:
            color = np.asarray(modes["rms_width_by_level"].values, dtype=float) / h[None, :]
        elif color_by == "h_times_ipr" and "ipr_by_level" in modes:
            color = np.asarray(modes["ipr_by_level"].values, dtype=float) * h[None, :]
        elif color_by == "h_times_kr" and "kr_by_level" in modes:
            color = np.asarray(modes["kr_by_level"].values, dtype=float) * h[None, :]
        elif color_by == "coarse_space_retention" and "coarse_space_retention_by_level" in modes:
            color = np.asarray(modes["coarse_space_retention_by_level"].values, dtype=float)
        else:
            color = np.full(frequency.shape, np.nan)
        selected = np.flatnonzero(finite.ravel())
        figure.add_trace(
            go.Scattergl(
                x=peak.ravel()[selected],
                y=frequency.ravel()[selected],
                mode="markers",
                marker={
                    "color": color.ravel()[selected],
                    "colorscale": "Viridis",
                    "size": 8,
                    "opacity": 0.88,
                    "colorbar": {"title": numeric_title[color_by]},
                    "line": {"color": "rgba(255,255,255,.7)", "width": 0.5},
                },
                customdata=np.column_stack(
                    (track_index.ravel()[selected], level_index.ravel()[selected])
                ),
                ids=[f"track-{int(value)}" for value in track_index.ravel()[selected]],
                hovertemplate=(
                    "Track %{customdata[0]} level %{customdata[1]}<br>"
                    "peak psi_N=%{x:.5f}<br>f=%{y:.5f} kHz<br>"
                    + numeric_title[color_by]
                    + "=%{marker.color:.5g}<extra></extra>"
                ),
                name="all resolutions",
            )
        )
    else:
        for label in LABEL_ORDER:
            if label not in selected_labels:
                continue
            mask = finite & (labels_by_track[:, None] == label)
            selected = np.flatnonzero(mask.ravel())
            if not selected.size:
                continue
            style = LABEL_STYLE[label]
            figure.add_trace(
                go.Scattergl(
                    x=peak.ravel()[selected],
                    y=frequency.ravel()[selected],
                    mode="markers",
                    marker={
                        "color": style["color"],
                        "symbol": style["symbol"],
                        "size": 8,
                        "opacity": 0.84,
                    },
                    customdata=np.column_stack(
                        (track_index.ravel()[selected], level_index.ravel()[selected])
                    ),
                    ids=[f"track-{int(value)}" for value in track_index.ravel()[selected]],
                    hovertemplate=(
                        "Track %{customdata[0]} level %{customdata[1]}<br>"
                        "peak psi_N=%{x:.5f}<br>f=%{y:.5f} kHz<extra></extra>"
                    ),
                    name=style["title"],
                )
            )
    kind = str(continuum.attrs.get("continuum_kind", "continuum")).replace("_", " ")
    figure.update_layout(
        template="plotly_white",
        margin={"l": 64, "r": 22, "t": 46, "b": 56},
        title={"text": f"All resolutions over {kind}", "x": 0.02},
        xaxis={"title": "Mode power-peak psi_N", "range": [0.0, 1.0]},
        yaxis={"title": "Cyclic frequency [kHz]"},
        legend={"orientation": "h", "y": 1.02, "x": 1.0, "xanchor": "right", "yanchor": "bottom"},
        hovermode="closest",
        uirevision=f"resolution-{color_by}-{kind}",
    )
    return figure


def _coerce_clicked_track(value, track_count: int) -> int | None:
    """Extract one valid track index from a Plotly event value."""
    while isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != 1:
            return None
        value = value[0]
    if isinstance(value, str):
        match = re.fullmatch(r"track-(\d+)", value.strip())
        if match:
            value = match.group(1)
    try:
        track = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not 0 <= track < int(track_count):
        return None
    return track


def _track_from_spectrum_click(
    click_data,
    current,
    track_count: int,
) -> int:
    """Resolve Plotly click payload variants without selecting continuum lines."""
    try:
        point = click_data["points"][0]
    except (KeyError, IndexError, TypeError):
        return int(current)
    for key in ("id", "customdata"):
        track = _coerce_clicked_track(point.get(key), track_count)
        if track is not None:
            return track
    hover = str(point.get("hovertext", ""))
    match = re.search(r"\bTrack\s+(\d+)\b", hover)
    if match:
        track = _coerce_clicked_track(match.group(1), track_count)
        if track is not None:
            return track
    return int(current)


def _profile_figure(
    artifact_dir: Path,
    track: int,
    field: str,
    component: str,
    top_harmonics: int,
):
    import plotly.graph_objects as go

    x, m_pol, values = _load_mode_field(artifact_dir, track, field)
    harmonic_power = np.trapezoid(np.abs(values) ** 2, x, axis=0)
    count = min(max(1, int(top_harmonics)), m_pol.size)
    selected = np.argsort(harmonic_power)[-count:]
    selected = selected[np.argsort(m_pol[selected])]
    figure = go.Figure()
    if component == "magnitude":
        transform = np.abs
        ytitle = "Magnitude"
    elif component == "real":
        transform = np.real
        ytitle = "Real part (phase aligned)"
    else:
        transform = np.imag
        ytitle = "Imaginary part (phase aligned)"
    colors = [
        f"hsl({int(220 - 175 * index / max(1, selected.size - 1))}, 66%, 44%)"
        for index in range(selected.size)
    ]
    for color, index in zip(colors, selected):
        figure.add_trace(
            go.Scattergl(
                x=x,
                y=transform(values[:, index]),
                mode="lines",
                line={"width": 1.6, "color": color},
                name=f"m={int(m_pol[index])}",
                hovertemplate=(
                    f"m={int(m_pol[index])}<br>psi_N=%{{x:.4f}}<br>value=%{{y:.5g}}<extra></extra>"
                ),
            )
        )
    if component == "magnitude":
        envelope = np.sqrt(np.sum(np.abs(values) ** 2, axis=1))
        figure.add_trace(
            go.Scatter(
                x=x,
                y=envelope,
                mode="lines",
                line={"width": 2.4, "color": "#17233C", "dash": "dash"},
                name="total envelope",
                hovertemplate="psi_N=%{x:.4f}<br>envelope=%{y:.5g}<extra></extra>",
            )
        )
    figure.update_layout(
        template="plotly_white",
        margin={"l": 64, "r": 18, "t": 46, "b": 52},
        title={"text": f"{FIELD_TITLES[field]}: strongest {count} harmonics", "x": 0.02},
        xaxis={"title": "Normalized poloidal flux psi_N", "range": [0, 1]},
        yaxis={"title": ytitle, "zeroline": True, "zerolinecolor": "#BAC4D2"},
        legend={"orientation": "h", "y": 1.02, "x": 1.0, "xanchor": "right", "yanchor": "bottom"},
        hovermode="x unified",
        uirevision=f"profile-{track}-{field}-{component}",
    )
    return figure


def _convergence_figure(modes: xr.Dataset, track: int):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    levels = np.asarray(modes.coords["level"].values, dtype=int)
    frequency = np.asarray(modes["frequency_khz_by_level"].values[track], dtype=float)
    error = np.asarray(modes["backward_error_by_level"].values[track], dtype=float)
    overlap = np.asarray(modes["xi_psi_overlap_by_level"].values[track], dtype=float)
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=levels,
            y=frequency,
            mode="lines+markers",
            line={"color": "#3066BE", "width": 2.4},
            marker={"size": 8},
            name="frequency",
            hovertemplate="level=%{x}<br>f=%{y:.6f} kHz<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=levels,
            y=overlap,
            mode="lines+markers",
            line={"color": "#159A8A", "width": 2.0, "dash": "dot"},
            marker={"size": 7},
            name="xi overlap",
            hovertemplate="level=%{x}<br>overlap=%{y:.6f}<extra></extra>",
        ),
        secondary_y=True,
    )
    finite_error = np.isfinite(error) & (error > 0)
    if np.any(finite_error):
        figure.add_trace(
            go.Scatter(
                x=levels[finite_error],
                y=np.clip(-np.log10(error[finite_error]), 0.0, None) / 12.0,
                mode="markers",
                marker={"color": "#D64B5F", "size": 9, "symbol": "diamond"},
                name="scaled -log10(error)",
                hovertext=[f"backward error={value:.3e}" for value in error[finite_error]],
                hovertemplate="level=%{x}<br>%{hovertext}<extra></extra>",
            ),
            secondary_y=True,
        )
    figure.update_layout(
        template="plotly_white",
        margin={"l": 64, "r": 58, "t": 46, "b": 52},
        title={"text": "Nested-grid convergence evidence", "x": 0.02},
        legend={"orientation": "h", "y": 1.02, "x": 1.0, "xanchor": "right", "yanchor": "bottom"},
    )
    figure.update_xaxes(title_text="Radial level", tickmode="array", tickvals=levels)
    figure.update_yaxes(title_text="Frequency [kHz]", secondary_y=False)
    figure.update_yaxes(title_text="Overlap / scaled error", range=[0.0, 1.05], secondary_y=True)
    return figure


def _format_metric(value, *, percent: bool = False, scientific: bool = False) -> str:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(scalar):
        return "n/a"
    if percent:
        return f"{100.0 * scalar:.2f}%"
    if scientific:
        return f"{scalar:.3e}"
    return f"{scalar:.5f}"


def _build_dash_app(artifact_dir: Path):
    from dash import Dash, Input, Output, State, ctx, dcc, html

    manifest, modes, continuum, full_continuum = _load_explorer_metadata(artifact_dir)
    frequencies = np.asarray(modes["frequency_khz"].values, dtype=float)
    labels = np.asarray(modes["label"].values).astype(str)
    order = np.argsort(frequencies)
    if order.size == 0:
        app = Dash(__name__, title="solver5 multigrid mode explorer")
        app.layout = html.Div(
            style={
                "background": "#eef3f8",
                "minHeight": "100vh",
                "padding": "24px",
                "fontFamily": "Inter, system-ui, sans-serif",
            },
            children=[
                html.H1("Multigrid mode explorer", style={"color": "#17233c"}),
                html.Div(
                    "No classified roots were returned. The continuum and search "
                    "diagnostics remain available in the artifact directory.",
                    style={
                        "background": "white",
                        "borderLeft": "5px solid #E39A31",
                        "borderRadius": "10px",
                        "padding": "14px",
                        "marginBottom": "14px",
                    },
                ),
                dcc.Graph(
                    figure=_spectrum_figure(modes, continuum, LABEL_ORDER),
                    style={"height": "650px"},
                ),
            ],
        )
        return app
    options = [
        {
            "label": (
                f"Track {track:03d} | {frequencies[track]:8.4f} kHz | "
                f"{LABEL_STYLE.get(labels[track], LABEL_STYLE['unresolved'])['title']}"
            ),
            "value": int(track),
        }
        for track in order
    ]
    physical = np.flatnonzero(labels == "physical_gap")
    initial_track = int(physical[0] if physical.size else order[0])
    counts = {label: int(np.count_nonzero(labels == label)) for label in LABEL_ORDER}
    continuum_backgrounds = {
        "slow_sound": continuum,
        "full_sound_alfven": full_continuum,
    }

    app = Dash(__name__, title="solver5 multigrid mode explorer")
    app.index_string = """<!DOCTYPE html>
<html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
:root { --ink:#17233c; --muted:#64748b; --panel:#ffffff; --line:#dce3ec; --bg:#eef3f8; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
.shell { min-height:100vh; }
.topbar { background:linear-gradient(118deg,#101c35 0%,#1f355e 68%,#28527a 100%); color:white; padding:22px 30px 20px; box-shadow:0 8px 22px rgba(15,28,53,.18); }
.topline { display:flex; align-items:flex-start; justify-content:space-between; gap:24px; }
.eyebrow { color:#8fd7e1; font-weight:700; letter-spacing:.13em; text-transform:uppercase; font-size:11px; }
h1 { font-size:25px; line-height:1.15; margin:5px 0 4px; font-weight:720; }
.subtitle { color:#c7d4e8; font-size:13px; }
.runbadge { border:1px solid rgba(255,255,255,.23); background:rgba(255,255,255,.08); border-radius:12px; padding:10px 13px; font-size:12px; color:#dbe7f7; white-space:nowrap; }
.content { padding:20px 24px 32px; max-width:1800px; margin:auto; }
.metric-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:16px; }
.metric { background:var(--panel); border:1px solid var(--line); border-radius:13px; padding:14px 16px; box-shadow:0 4px 13px rgba(20,38,65,.045); }
.metric-label { font-size:11px; letter-spacing:.075em; text-transform:uppercase; color:var(--muted); font-weight:700; }
.metric-value { font-size:24px; font-weight:730; margin-top:3px; }
.layout { display:grid; grid-template-columns:minmax(280px,340px) minmax(0,1fr); gap:16px; align-items:start; }
.sidebar,.card { background:var(--panel); border:1px solid var(--line); border-radius:14px; box-shadow:0 5px 15px rgba(20,38,65,.05); }
.sidebar { padding:17px; position:sticky; top:14px; }
.sidebar h3 { margin:0 0 13px; font-size:15px; }
.control-label { display:block; color:#526177; font-size:12px; font-weight:700; margin:15px 0 7px; }
.hint { margin:13px 0 0; padding:11px 12px; border-radius:10px; background:#eef7fa; color:#34556c; font-size:12px; line-height:1.45; }
.toggle-list label { display:flex !important; align-items:center; gap:9px; padding:6px 2px !important; cursor:pointer; font-size:12px; color:#435168; }
.toggle-list input[type="checkbox"] { appearance:none; width:34px; height:18px; border-radius:999px; background:#cbd5e1; position:relative; margin:0; transition:.18s ease; flex:0 0 auto; }
.toggle-list input[type="checkbox"]:after { content:""; position:absolute; width:14px; height:14px; border-radius:50%; background:white; top:2px; left:2px; box-shadow:0 1px 3px rgba(0,0,0,.28); transition:.18s ease; }
.toggle-list input[type="checkbox"]:checked { background:#3066be; }
.toggle-list input[type="checkbox"]:checked:after { transform:translateX(16px); }
.label-key { display:grid; grid-template-columns:11px 1fr auto; gap:8px; align-items:center; font-size:12px; padding:6px 0; color:#435168; }
.dot { width:10px; height:10px; border-radius:50%; }
.main { display:grid; gap:16px; }
.card { padding:8px 10px 2px; overflow:hidden; }
.details { display:grid; grid-template-columns:minmax(0,1.35fr) minmax(300px,.65fr); gap:16px; }
.evidence { padding:17px 18px; }
.mode-title { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:8px; }
.mode-title h2 { margin:0; font-size:19px; }
.pill { display:inline-flex; border-radius:999px; padding:5px 10px; color:white; font-size:11px; font-weight:750; letter-spacing:.035em; }
.reason { background:#f6f8fb; border:1px solid #e4e9f0; border-radius:9px; padding:10px 11px; color:#4d5c72; font:11px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; overflow-wrap:anywhere; }
.evidence-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:12px; }
.ev { border-left:3px solid #b8c5d6; padding:7px 9px; background:#fbfcfe; }
.ev span { display:block; color:#6b788d; font-size:10px; text-transform:uppercase; letter-spacing:.05em; }
.ev strong { font-size:14px; }
.footer-note { color:#708097; font-size:11px; margin:12px 2px 2px; }
@media (max-width:1050px) { .layout{grid-template-columns:1fr}.sidebar{position:static}.details{grid-template-columns:1fr}.metric-grid{grid-template-columns:1fr 1fr} }
@media (max-width:600px) { .content{padding:12px}.topbar{padding:18px}.topline{display:block}.runbadge{margin-top:12px}.metric-grid{grid-template-columns:1fr} }
</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"""

    label_keys = [
        html.Div(
            className="label-key",
            children=[
                html.Span(className="dot", style={"background": LABEL_STYLE[label]["color"]}),
                html.Span(LABEL_STYLE[label]["title"]),
                html.Strong(str(counts[label])),
            ],
        )
        for label in LABEL_ORDER
    ]
    app.layout = html.Div(
        className="shell",
        children=[
            html.Header(
                className="topbar",
                children=[
                    html.Div(
                        className="topline",
                        children=[
                            html.Div(
                                children=[
                                    html.Div("superNOVA / solver5", className="eyebrow"),
                                    html.H1("Multigrid mode explorer"),
                                    html.Div(
                                        "Three-level radial persistence, continuum evidence, and harmonic structure",
                                        className="subtitle",
                                    ),
                                ]
                            ),
                            html.Div(
                                f"n={modes.attrs.get('ntor', 4)}  |  "
                                f"{manifest['frequency_khz'][0]:.0f}-{manifest['frequency_khz'][1]:.0f} kHz  |  "
                                f"{manifest['mode_count']} tracks",
                                className="runbadge",
                            ),
                        ],
                    )
                ],
            ),
            html.Main(
                className="content",
                children=[
                    html.Div(
                        className="metric-grid",
                        children=[
                            html.Div([html.Div(LABEL_STYLE[label]["title"], className="metric-label"), html.Div(counts[label], className="metric-value", style={"color": LABEL_STYLE[label]["color"]})], className="metric")
                            for label in LABEL_ORDER
                        ],
                    ),
                    html.Div(
                        className="layout",
                        children=[
                            html.Aside(
                                className="sidebar",
                                children=[
                                    html.H3("Explore a mode"),
                                    html.Label("Selected track", className="control-label"),
                                    dcc.Dropdown(id="mode-select", options=options, value=initial_track, clearable=False),
                                    html.Label("Visible classifications", className="control-label"),
                                    dcc.Checklist(
                                        id="label-filter",
                                        options=[{"label": LABEL_STYLE[label]["title"], "value": label} for label in LABEL_ORDER],
                                        value=list(LABEL_ORDER),
                                        className="toggle-list",
                                    ),
                                    html.Label("Continuum background", className="control-label"),
                                    dcc.RadioItems(
                                        id="continuum-background",
                                        options=[
                                            {"label": "Slow-sound (readable)", "value": "slow_sound"},
                                            {"label": "Full sound-Alfven", "value": "full_sound_alfven"},
                                        ],
                                        value="slow_sound",
                                        className="toggle-list",
                                    ),
                                    html.Label("Resolution-point color", className="control-label"),
                                    dcc.Dropdown(
                                        id="diagnostic-color",
                                        options=[
                                            {"label": "Classification", "value": "classification"},
                                            {"label": "width / h", "value": "width_over_h"},
                                            {"label": "h * IPR", "value": "h_times_ipr"},
                                            {"label": "h * k_r", "value": "h_times_kr"},
                                            {"label": "Fixed-coarse retention", "value": "coarse_space_retention"},
                                        ],
                                        value="classification",
                                        clearable=False,
                                    ),
                                    html.Label("Field", className="control-label"),
                                    dcc.Dropdown(id="field-select", options=[{"label": title, "value": name} for name, title in FIELD_TITLES.items()], value="xi_psi", clearable=False),
                                    html.Label("Component", className="control-label"),
                                    dcc.RadioItems(
                                        id="component-select",
                                        options=[{"label": "Magnitude", "value": "magnitude"}, {"label": "Real", "value": "real"}, {"label": "Imaginary", "value": "imag"}],
                                        value="magnitude",
                                        labelStyle={"display": "inline-block", "marginRight": "12px", "fontSize": "12px"},
                                    ),
                                    html.Label("Strongest harmonics", className="control-label"),
                                    dcc.Slider(id="harmonic-count", min=3, max=min(20, int(modes.sizes.get('m_pol', 20))), step=1, value=min(10, int(modes.sizes.get('m_pol', 10))), marks={3: "3", 5: "5", 10: "10", min(20, int(modes.sizes.get('m_pol', 20))): str(min(20, int(modes.sizes.get('m_pol', 20))))}),
                                    html.Div(label_keys, style={"marginTop": "17px", "paddingTop": "10px", "borderTop": "1px solid #e3e8ef"}),
                                    html.Div("Click any mode marker to select it. The background is the readable slow-sound continuum; labels retain full-continuum classification evidence. Red diamonds are continuum resonances.", className="hint"),
                                ],
                            ),
                            html.Section(
                                className="main",
                                children=[
                                    html.Div(dcc.Loading(dcc.Graph(id="spectrum", figure=_spectrum_figure(modes, continuum, LABEL_ORDER), style={"height": "520px"})), className="card"),
                                    html.Div(
                                        dcc.Loading(
                                            dcc.Graph(
                                                id="resolution-diagnostic",
                                                figure=_resolution_diagnostic_figure(
                                                    modes,
                                                    continuum,
                                                    LABEL_ORDER,
                                                    "classification",
                                                ),
                                                style={"height": "560px"},
                                            )
                                        ),
                                        className="card",
                                    ),
                                    html.Div(
                                        className="details",
                                        children=[
                                            html.Div(dcc.Loading(dcc.Graph(id="profile", style={"height": "480px"})), className="card"),
                                            html.Div(
                                                className="card evidence",
                                                children=[
                                                    html.Div([html.H2(id="mode-heading"), html.Span(id="mode-label", className="pill")], className="mode-title"),
                                                    html.Div(id="mode-reason", className="reason"),
                                                    html.Div(
                                                        className="evidence-grid",
                                                        children=[
                                                            html.Div([html.Span("Frequency"), html.Strong(id="metric-frequency")], className="ev"),
                                                            html.Div([html.Span("Dominant m"), html.Strong(id="metric-m")], className="ev"),
                                                            html.Div([html.Span("Continuum clearance"), html.Strong(id="metric-clearance")], className="ev"),
                                                            html.Div([html.Span("Continuum-band power"), html.Strong(id="metric-power")], className="ev"),
                                                            html.Div([html.Span("Fine-grid drift"), html.Strong(id="metric-drift")], className="ev"),
                                                            html.Div([html.Span("Medium/fine overlap"), html.Strong(id="metric-overlap")], className="ev"),
                                                        ],
                                                    ),
                                                    html.Div("Catalog incompleteness is reported separately; missing per-mode evidence remains explicitly unresolved.", className="footer-note"),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(dcc.Loading(dcc.Graph(id="convergence", style={"height": "390px"})), className="card"),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        Output("spectrum", "figure"),
        Output("resolution-diagnostic", "figure"),
        Input("label-filter", "value"),
        Input("continuum-background", "value"),
        Input("diagnostic-color", "value"),
    )
    def update_spectrum(selected_labels, background, color_by):
        selected_continuum = continuum_backgrounds.get(background, continuum)
        return (
            _spectrum_figure(modes, selected_continuum, selected_labels or []),
            _resolution_diagnostic_figure(
                modes,
                selected_continuum,
                selected_labels or [],
                color_by,
            ),
        )

    @app.callback(
        Output("mode-select", "value"),
        Input("spectrum", "clickData"),
        Input("resolution-diagnostic", "clickData"),
        State("mode-select", "value"),
        prevent_initial_call=True,
    )
    def select_from_spectrum(spectrum_click, diagnostic_click, current):
        click_data = (
            diagnostic_click
            if ctx.triggered_id == "resolution-diagnostic"
            else spectrum_click
        )
        return _track_from_spectrum_click(
            click_data,
            current,
            modes.sizes.get("track", 0),
        )

    detail_outputs = [
        Output("profile", "figure"),
        Output("convergence", "figure"),
        Output("mode-heading", "children"),
        Output("mode-label", "children"),
        Output("mode-label", "style"),
        Output("mode-reason", "children"),
        Output("metric-frequency", "children"),
        Output("metric-m", "children"),
        Output("metric-clearance", "children"),
        Output("metric-power", "children"),
        Output("metric-drift", "children"),
        Output("metric-overlap", "children"),
    ]

    @app.callback(
        *detail_outputs,
        Input("mode-select", "value"),
        Input("field-select", "value"),
        Input("component-select", "value"),
        Input("harmonic-count", "value"),
    )
    def update_mode(track, field, component, harmonic_count):
        track = int(track)
        label = str(modes["label"].values[track])
        style = LABEL_STYLE.get(label, LABEL_STYLE["unresolved"])
        profile = _profile_figure(artifact_dir, track, field, component, harmonic_count)
        convergence = _convergence_figure(modes, track)
        return (
            profile,
            convergence,
            f"Track {track}",
            style["title"],
            {"background": style["color"]},
            str(modes["reason"].values[track]),
            f"{float(modes['frequency_khz'].values[track]):.5f} kHz",
            str(int(modes["dominant_m"].values[track])),
            _format_metric(modes["minimum_relative_continuum_clearance"].values[track], percent=True),
            _format_metric(modes["continuum_band_power_fraction"].values[track], percent=True),
            _format_metric(modes["fine_frequency_change"].values[track], scientific=True),
            _format_metric(modes["medium_fine_overlap"].values[track]),
        )

    return app


def _run_serve(args: argparse.Namespace) -> None:
    artifact_dir = _resolve_serve_artifact_dir(args.artifact_dir)
    if artifact_dir != args.artifact_dir:
        print(
            f"No production manifest at {args.artifact_dir}; "
            f"using demo artifacts from {artifact_dir}."
        )
    app = _build_dash_app(artifact_dir)
    print(f"Serving http://{args.host}:{args.port}")
    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        use_reloader=False,
        threaded=False,
    )


def main() -> None:
    args = _parser().parse_args()
    if args.command == "solve":
        _run_solve(args)
    elif args.command == "reclassify":
        _run_reclassify(args)
    elif args.command == "reacquire":
        _run_reacquire(args)
    elif args.command == "demo":
        _run_demo(args)
    else:
        _run_serve(args)


if __name__ == "__main__":
    main()
