"""Focused artifact tests for no_sync/window_solver_multigrid.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import xarray as xr


SCRIPT = Path(__file__).resolve().parents[1] / "no_sync/window_solver_multigrid.py"
SPEC = importlib.util.spec_from_file_location("window_solver_multigrid", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _synthetic_datasets() -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, tuple[float, float]]:
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
        for name in MODULE.FIELD_TITLES
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
            "schema_version": MODULE.SCHEMA_VERSION,
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


class WindowSolverMultigridScriptTests(unittest.TestCase):
    def test_reacquisition_selection_is_eligible_global_and_ranked(self):
        from supernova.solve.evidence import (
            ReacquisitionOptions,
            select_reacquisition_candidates,
        )

        tracks = xr.Dataset(
            {
                "present": (("track", "level"), [[True, False]] * 3 + [[True, True]]),
                "width_over_h": (
                    ("track", "level"),
                    [[4.0, np.nan], [1.0, np.nan], [5.0, np.nan], [4.0, 4.0]],
                ),
                "coarse_space_retention": (
                    ("track", "level"),
                    [[0.2, np.nan], [0.8, np.nan], [0.9, np.nan], [0.7, 0.7]],
                ),
            },
            coords={"track": np.arange(4), "level": [0, 1]},
        )
        selected = select_reacquisition_candidates(
            tracks,
            1,
            ReacquisitionOptions(max_candidates_per_level=1),
            eligible=np.asarray([True, True, False, True]),
        )

        np.testing.assert_array_equal(selected, [1])
        with self.assertRaisesRegex(ValueError, "one Boolean value per track"):
            select_reacquisition_candidates(
                tracks,
                1,
                ReacquisitionOptions(),
                eligible=np.asarray([True]),
            )

    def test_reacquisition_eligibility_survives_track_reordering(self):
        from supernova.solve.multigrid import (
            _track_mask_from_root_identities,
            _track_root_identities,
        )

        levels = [
            xr.Dataset(
                {"source_eigenpair": ("eigenpair", [10.0, 11.0])},
                coords={"eigenpair": [0, 1]},
            ),
            xr.Dataset(
                {"source_eigenpair": ("eigenpair", [20.0, 21.0, np.nan])},
                coords={"eigenpair": [0, 1, 2]},
            ),
        ]
        original = xr.Dataset(
            {
                "present": (("track", "level"), [[True, False], [True, True]]),
                "mode_index": (("track", "level"), [[0, -1], [1, 0]]),
            },
            coords={"track": [0, 1], "level": [0, 1]},
        )
        identities = _track_root_identities(
            original,
            levels,
            np.asarray([False, True]),
        )
        retracked = xr.Dataset(
            {
                "present": (
                    ("track", "level"),
                    [[True, True], [False, True], [True, True]],
                ),
                "mode_index": (
                    ("track", "level"),
                    [[0, 2], [-1, 1], [1, 0]],
                ),
            },
            coords={"track": [0, 1, 2], "level": [0, 1]},
        )

        np.testing.assert_array_equal(
            _track_mask_from_root_identities(retracked, levels, identities),
            [False, False, True],
        )

    def test_reacquire_dry_run_selects_without_slepc(self):
        tracks = xr.Dataset(
            {
                "omega": (
                    ("track", "level"),
                    np.asarray(
                        [
                            [1.2 + 0.0j, np.nan + 0.0j, np.nan + 0.0j],
                            [1.4 + 0.0j, 1.41 + 0.0j, np.nan + 0.0j],
                            [1.6 + 0.0j, np.nan + 0.0j, np.nan + 0.0j],
                        ]
                    ),
                ),
                "present": (
                    ("track", "level"),
                    [[True, False, False], [True, True, False], [True, False, False]],
                ),
                "width_over_h": (
                    ("track", "level"),
                    [[4.0, np.nan, np.nan], [4.0, 5.0, np.nan], [4.0, np.nan, np.nan]],
                ),
                "coarse_space_retention": (
                    ("track", "level"),
                    [[0.6, np.nan, np.nan], [0.7, 0.7, np.nan], [0.8, np.nan, np.nan]],
                ),
            },
            coords={"track": [0, 1, 2], "level": [0, 1, 2]},
        )
        classification = xr.Dataset(
            {
                "label": ("track", ["unresolved", "unresolved", "physical_gap"]),
                "discrete_valid": ("track", ["true", "true", "true"]),
            },
            coords={"track": [0, 1, 2]},
        )
        continuum = xr.Dataset(
            {
                "omega": (("psi", "branch"), [[1.0 + 0.0j], [2.0 + 0.0j]]),
                "frequency_khz": (("psi", "branch"), [[50.0], [100.0]]),
                "branch_assignment_ambiguous": (
                    ("psi", "branch"),
                    np.zeros((2, 1), dtype=bool),
                ),
            },
            coords={"psi": [0.0, 1.0], "branch": [0]},
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            MODULE._write_dataset(tracks, output / "mode_tracks_evidence_v2.nc")
            MODULE._write_dataset(
                classification,
                output / "classification_evidence_v2.nc",
            )
            MODULE._write_dataset(continuum, output / "continuum.nc")
            (output / "manifest.json").write_text(
                json.dumps({"configuration": {}}),
                encoding="utf-8",
            )

            MODULE._run_reacquire(
                SimpleNamespace(
                    artifact_dir=output,
                    execute=False,
                    max_candidates_per_level=24,
                    nev_multiplier=4,
                    relative_half_width=3.0e-3,
                    minimum_overlap=0.90,
                    verbose=False,
                )
            )

            report = json.loads((output / "reacquisition_dry_run.json").read_text())
            self.assertIs(report["slepc_was_run"], False)
            self.assertEqual(report["selected_window_count_before_cascades"], 2)
            self.assertEqual(report["selected_by_target_level"], {"1": 1, "2": 1})
            self.assertFalse(
                (output / "manifest_reacquired_evidence_v2.json").exists()
            )

    def test_reclassify_rebuilds_tracks_from_raw_levels_without_slepc(self):
        from supernova.solve.multigrid import _nested_bspline_levels
        from supernova.solve.evidence import (
            bspline_prolongation_matrix,
        )

        base_spans = 2
        order = 3
        quadrature_order = 4
        radial_levels = _nested_bspline_levels(
            MODULE._base_spline(base_spans, order, quadrature_order),
            levels=3,
        )
        points, _ = radial_levels[-1].get_quadrature_points()
        coefficients = [np.linspace(0.2, 1.0, radial_levels[0].dofs)[:, None]]
        for coarse, fine in zip(radial_levels[:-1], radial_levels[1:]):
            coefficients.append(
                np.asarray(bspline_prolongation_matrix(coarse, fine) @ coefficients[-1])
            )

        frequencies = (2.0004, 2.0001, 2.0)
        raw_levels = []
        for level, (radial, coefficient, omega) in enumerate(
            zip(radial_levels, coefficients, frequencies)
        ):
            field = radial.eval_basis(points).T @ coefficient
            low_dofs = radial.dofs - 1
            raw_levels.append(
                xr.Dataset(
                    {
                        "s": ("eigenpair", np.asarray([omega**2], dtype=complex)),
                        "omega": ("eigenpair", np.asarray([omega], dtype=complex)),
                        "slepc_backward_error": ("eigenpair", [1.0e-12]),
                        "residual_norm": ("eigenpair", [1.0e-12]),
                        "window_id": ("eigenpair", [level]),
                        "window_depth": ("eigenpair", [0]),
                        "core_owned": ("eigenpair", [True]),
                        "xi_psi": (
                            ("eigenpair", "psi", "m_pol"),
                            field[None, :, :],
                        ),
                        "xi_psi_coefficient": (
                            ("eigenpair", "radial_dof_high", "m_pol"),
                            coefficient[None, :, :],
                        ),
                        "p1_coefficient": (
                            ("eigenpair", "radial_dof_high", "m_pol"),
                            coefficient[None, :, :],
                        ),
                        "xi_s_coefficient": (
                            ("eigenpair", "radial_dof_low", "m_pol"),
                            np.zeros((1, low_dofs, 1), dtype=complex),
                        ),
                        "div_xi_coefficient": (
                            ("eigenpair", "radial_dof_low", "m_pol"),
                            np.zeros((1, low_dofs, 1), dtype=complex),
                        ),
                    },
                    coords={
                        "eigenpair": [0],
                        "psi": points,
                        "m_pol": [10],
                        "radial_dof_high": np.arange(radial.dofs),
                        "radial_dof_low": np.arange(low_dofs),
                    },
                )
            )

        legacy_modes = xr.Dataset(
            {
                "frequency_khz": ("track", [100.0]),
                "center": ("track", [0.5]),
                "label": ("track", ["inconclusive"]),
                "reason": ("track", ["incomplete_search_windows"]),
                "last_level": ("track", [2]),
                "mode_index_by_level": (("track", "level"), [[0, 0, 0]]),
                "present_by_level": (("track", "level"), [[True, True, True]]),
            },
            coords={"track": [0], "level": [0, 1, 2]},
            attrs={"schema_version": 1},
        )
        continuum = xr.Dataset(
            {
                "omega": (
                    ("psi", "branch"),
                    np.asarray([[0.5], [0.5]], dtype=complex),
                ),
                "frequency_khz": (("psi", "branch"), [[25.0], [25.0]]),
                "branch_assignment_ambiguous": (
                    ("psi", "branch"),
                    np.zeros((2, 1), dtype=bool),
                ),
            },
            coords={"psi": [0.0, 1.0], "branch": [0]},
        )
        diagnostics = xr.Dataset(
            {
                "valid": (("level", "window"), np.ones((3, 1), dtype=bool)),
                "catalog_status": (
                    ("level", "window"),
                    np.full((3, 1), "capacity_limited", dtype="U24"),
                ),
            },
            coords={"level": [0, 1, 2], "window": [0]},
        )
        manifest = {
            "schema_version": 1,
            "frequency_khz": [50.0, 150.0],
            "artifacts": {"modes": "modes.nc"},
            "configuration": {
                "base_spans": base_spans,
                "levels": 3,
                "order": order,
                "quadrature_order": quadrature_order,
                "reconstruction_points": 32,
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            MODULE._write_dataset(legacy_modes, output / "modes.nc")
            MODULE._write_dataset(continuum, output / "continuum.nc")
            MODULE._write_dataset(diagnostics, output / "window_diagnostics.nc")
            for level, dataset in enumerate(raw_levels):
                MODULE._write_dataset(dataset, output / f"level_{level}.nc")
            (output / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            MODULE._run_reclassify(
                SimpleNamespace(
                    artifact_dir=output,
                    output_dir=None,
                    source="auto",
                )
            )

            report = json.loads((output / "reclassification_report.json").read_text())
            self.assertEqual(report["tracking_source"], "raw_level_retracking")
            self.assertIs(report["slepc_was_run"], False)
            self.assertEqual(report["after_label_counts"], {"physical_gap": 1})
            self.assertEqual(report["previously_inconclusive_after"], {"physical_gap": 1})
            self.assertEqual(report["legacy_track_histories_identical"], 1)
            self.assertEqual(report["legacy_track_histories_changed"], 0)
            with xr.open_dataset(
                output / "modes_evidence_v2.nc",
                engine="netcdf4",
            ) as saved:
                self.assertEqual(saved["label"].item(), "physical_gap")
                self.assertEqual(saved["legacy_track"].item(), 0)
                self.assertEqual(saved.attrs["tracking_source"], "raw_level_retracking")
            self.assertEqual(
                MODULE._mode_artifact_path(output),
                output / "modes_evidence_v2.nc",
            )

    def test_synthetic_modes_are_normalized_and_cover_every_label(self):
        modes, continuum, diagnostics, band = _synthetic_datasets()

        np.testing.assert_allclose(band, (50.0, 150.0))
        self.assertEqual(set(modes["label"].values.tolist()), set(MODULE.LABEL_ORDER))
        self.assertEqual(continuum["frequency_khz"].dims, ("psi", "branch"))
        self.assertTrue(diagnostics["level_complete"].values.all())

        xi = modes["xi_psi"].values
        power = np.sum(np.abs(xi) ** 2, axis=2)
        norms = np.trapezoid(power, modes.x.values, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=2.0e-13, atol=2.0e-13)

    def test_netcdf_storage_splits_complex_and_sanitizes_attributes(self):
        dataset = xr.Dataset(
            {"mode": ("track", np.array([1.0 + 2.0j]))},
            coords={"track": [0]},
            attrs={"complete": True, "optional": None},
        )
        stored = MODULE._storage_dataset(dataset)

        self.assertNotIn("mode", stored)
        np.testing.assert_array_equal(stored["mode_real"], [1.0])
        np.testing.assert_array_equal(stored["mode_imag"], [2.0])
        self.assertEqual(stored.attrs["complete"], 1)
        self.assertEqual(stored.attrs["optional"], "None")

    def test_empty_search_still_writes_continuum_artifacts(self):
        modes, continuum, diagnostics, band = _synthetic_datasets()
        empty = modes.isel(track=slice(0, 0))
        manifest = MODULE._artifact_manifest(
            args=None,
            modes=empty,
            frequency_band=band,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            MODULE._save_artifacts(
                output,
                empty,
                continuum,
                diagnostics,
                band,
                manifest,
            )

            for name in (
                "continuum.nc",
                "display_continuum.nc",
                "window_diagnostics.nc",
                "summary.csv",
                "overview.png",
                "manifest.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            written = json.loads((output / "manifest.json").read_text())
            self.assertEqual(written["mode_count"], 0)
            self.assertEqual(written["classification_counts"], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
