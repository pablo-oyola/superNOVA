"""Reading and writing a scan's artifact directory.

A run leaves a manifest, the mode and continuum datasets, the per-level raw
catalogs, a summary CSV and an overview figure. Complex arrays are stored as
explicit ``*_real``/``*_imag`` variables so the NetCDF files stay portable
across xarray backends, and the manifest records the git commit and a hash of
its inputs so a directory can be traced back to what produced it.

Lifted from ``no_sync/window_solver_multigrid.py``, which was not under version
control.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable, Mapping

import numpy as np
import xarray as xr

from supernova.app.reconstruct import _omega_to_khz
from supernova.app.report import _overview_figure
from supernova.app.schema import (
    FIELD_TITLES,
    LABEL_ORDER,
    LABEL_STYLE,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
)

# Provenance is recorded relative to the installed package, so an artifact
# directory names the commit of the code that produced it wherever it ran.
REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _raw_level_paths(
    artifact_dir: Path,
    expected_levels: int,
    *,
    variant: str = "base",
) -> tuple[Path, ...]:
    if variant not in ("base", "reacquired"):
        raise ValueError("raw level variant must be 'base' or 'reacquired'.")
    suffix = "_reacquired" if variant == "reacquired" else ""
    paths = tuple(
        artifact_dir / f"level_{level}{suffix}.nc"
        for level in range(expected_levels)
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Raw-level retracking requires a contiguous level_*.nc catalog; missing "
            + ", ".join(str(path) for path in missing)
        )
    return paths


def _physical_metric_path(artifact_dir: Path) -> Path:
    return Path(artifact_dir) / "physical_overlap_metric_full.nc"


def _load_physical_overlap_metric(path: Path):
    from supernova.solve.evidence import PhysicalDisplacementMetric

    dataset = _load_portable_dataset(
        path,
        (
            "mass_density",
            "radial_geometry_harmonic",
            "tangential_geometry_harmonic",
        ),
    )
    with xr.open_dataset(path, engine="netcdf4") as source:
        dataset = dataset.assign_coords(
            m_pol=np.asarray(source.coords["m_pol"].values),
        )
        dataset.attrs.update(source.attrs)
    return PhysicalDisplacementMetric.from_dataset(dataset)


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
        "source_eigenpair",
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
    if "source_eigenpair" not in dataset:
        dataset["source_eigenpair"] = (
            "eigenpair",
            np.asarray(dataset.coords["eigenpair"].values, dtype=np.int64),
        )
    dataset = dataset.assign_coords(eigenpair=np.arange(count, dtype=np.int64))
    return dataset


def _mode_artifact_path(
    artifact_dir: Path,
    manifest: Mapping[str, object] | None = None,
) -> Path:
    if manifest is None:
        has_manifest = (artifact_dir / "manifest.json").is_file() or (
            artifact_dir / "manifest_evidence_v2.json"
        ).is_file() or (
            artifact_dir / "manifest_reacquired_evidence_v2.json"
        ).is_file() or (
            artifact_dir / "manifest_physical_evidence_v2.json"
        ).is_file() or (
            artifact_dir / "manifest_physical_full_evidence_v2.json"
        ).is_file()
        manifest = _load_manifest(artifact_dir) if has_manifest else {}
    configured = dict(manifest.get("artifacts", {})).get("modes")
    if configured:
        path = artifact_dir / str(configured)
        if path.is_file():
            return path
    evidence_path = artifact_dir / "modes_evidence_v2.nc"
    return evidence_path if evidence_path.is_file() else artifact_dir / "modes.nc"


def _load_manifest(artifact_dir: Path) -> dict[str, object]:
    physical_full_path = artifact_dir / "manifest_physical_full_evidence_v2.json"
    physical_path = artifact_dir / "manifest_physical_evidence_v2.json"
    reacquired_path = artifact_dir / "manifest_reacquired_evidence_v2.json"
    evidence_path = artifact_dir / "manifest_evidence_v2.json"
    if physical_full_path.is_file():
        path = physical_full_path
    elif physical_path.is_file():
        path = physical_path
    elif reacquired_path.is_file():
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


def _artifact_manifest(
    *,
    args: argparse.Namespace | None,
    modes: xr.Dataset,
    frequency_band: tuple[float, float],
) -> dict[str, object]:
    labels, counts = np.unique(modes["label"].values.astype(str), return_counts=True)
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
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
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
