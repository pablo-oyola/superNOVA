"""Focused artifact and GUI tests for no_sync/window_solver_multigrid.py."""

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


class WindowSolverMultigridScriptTests(unittest.TestCase):
    def test_reclassify_rebuilds_tracks_from_raw_levels_without_slepc(self):
        from supernova.core.solver5_multigrid import _nested_bspline_levels
        from supernova.core.solver5_multigrid_evidence import (
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

    def test_demo_modes_are_normalized_and_cover_every_label(self):
        modes, continuum, diagnostics, band = MODULE._demo_datasets()

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

    def test_demo_artifacts_round_trip_and_build_dash_app(self):
        modes, continuum, diagnostics, band = MODULE._demo_datasets()
        display_continuum = continuum.isel(branch=slice(0, None, 2)).assign_attrs(
            continuum_kind="slow_sound",
            purpose="display_only",
        )
        manifest = MODULE._artifact_manifest(
            args=None,
            modes=modes,
            frequency_band=band,
            demo=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            MODULE._save_artifacts(
                output,
                modes,
                continuum,
                diagnostics,
                band,
                manifest,
                display_continuum=display_continuum,
            )

            self.assertIs(
                json.loads((output / "manifest.json").read_text())["demo"],
                True,
            )
            with xr.open_dataset(output / "modes.nc", engine="netcdf4") as saved:
                self.assertEqual(
                    saved["xi_psi_real"].dims,
                    ("track", "x", "m_pol"),
                )
                self.assertEqual(
                    saved["xi_psi_imag"].dims,
                    ("track", "x", "m_pol"),
                )
                self.assertNotIn("xi_psi", saved)
                self.assertTrue(saved.attrs["complex_storage"].startswith("paired"))

            _, _, loaded_continuum, loaded_full_continuum = (
                MODULE._load_explorer_metadata(output)
            )
            self.assertEqual(
                loaded_continuum.sizes["branch"],
                display_continuum.sizes["branch"],
            )
            self.assertEqual(loaded_continuum.attrs["continuum_kind"], "slow_sound")
            self.assertEqual(
                loaded_full_continuum.sizes["branch"],
                continuum.sizes["branch"],
            )
            self.assertEqual(
                loaded_full_continuum.attrs["continuum_kind"],
                "full_sound_alfven",
            )

            app = MODULE._build_dash_app(output)
            self.assertEqual(app.title, "solver5 multigrid mode explorer")
            self.assertEqual(len(app.callback_map), 3)
            spectrum_callback = next(
                key for key in app.callback_map if key.startswith("..spectrum.figure")
            )
            self.assertEqual(
                app.callback_map[spectrum_callback]["inputs"],
                [
                    {"id": "label-filter", "property": "value"},
                    {"id": "continuum-background", "property": "value"},
                    {"id": "diagnostic-color", "property": "value"},
                ],
            )
            self.assertIsNotNone(app.layout)
            client = app.server.test_client()
            self.assertEqual(client.get("/").status_code, 200)
            layout_response = client.get("/_dash-layout")
            self.assertEqual(layout_response.status_code, 200)
            layout_json = json.dumps(layout_response.get_json())
            self.assertIn("label-filter", layout_json)
            self.assertIn("Visible classifications", layout_json)
            self.assertIn("continuum-background", layout_json)
            self.assertIn("resolution-diagnostic", layout_json)

            spectrum_click = client.post(
                "/_dash-update-component",
                json={
                    "output": "mode-select.value",
                    "outputs": {"id": "mode-select", "property": "value"},
                    "changedPropIds": ["spectrum.clickData"],
                    "inputs": [
                        {
                            "id": "spectrum",
                            "property": "clickData",
                            "value": {"points": [{"customdata": [7]}]},
                        },
                        {
                            "id": "resolution-diagnostic",
                            "property": "clickData",
                            "value": None,
                        },
                    ],
                    "state": [
                        {"id": "mode-select", "property": "value", "value": 0}
                    ],
                },
            )
            self.assertEqual(spectrum_click.status_code, 200)
            self.assertEqual(
                spectrum_click.get_json()["response"]["mode-select"]["value"],
                7,
            )
            diagnostic_click = client.post(
                "/_dash-update-component",
                json={
                    "output": "mode-select.value",
                    "outputs": {"id": "mode-select", "property": "value"},
                    "changedPropIds": ["resolution-diagnostic.clickData"],
                    "inputs": [
                        {
                            "id": "spectrum",
                            "property": "clickData",
                            "value": None,
                        },
                        {
                            "id": "resolution-diagnostic",
                            "property": "clickData",
                            "value": {
                                "points": [
                                    {"id": "track-9", "customdata": [9, 2]}
                                ]
                            },
                        },
                    ],
                    "state": [
                        {"id": "mode-select", "property": "value", "value": 7}
                    ],
                },
            )
            self.assertEqual(diagnostic_click.status_code, 200)
            self.assertEqual(
                diagnostic_click.get_json()["response"]["mode-select"]["value"],
                9,
            )
            detail_callback = next(
                key for key in app.callback_map if key.startswith("..profile.figure")
            )
            detail_outputs = [
                {
                    "id": output.component_id,
                    "property": output.component_property,
                }
                for output in app.callback_map[detail_callback]["output"]
            ]
            detail_response = client.post(
                "/_dash-update-component",
                json={
                    "output": detail_callback,
                    "outputs": detail_outputs,
                    "changedPropIds": ["mode-select.value"],
                    "inputs": [
                        {"id": "mode-select", "property": "value", "value": 9},
                        {
                            "id": "field-select",
                            "property": "value",
                            "value": "xi_psi",
                        },
                        {
                            "id": "component-select",
                            "property": "value",
                            "value": "magnitude",
                        },
                        {
                            "id": "harmonic-count",
                            "property": "value",
                            "value": 10,
                        },
                    ],
                    "state": [],
                },
            )
            self.assertEqual(detail_response.status_code, 200)
            detail_payload = detail_response.get_json()["response"]
            self.assertEqual(detail_payload["mode-heading"]["children"], "Track 9")
            self.assertGreater(
                len(detail_payload["profile"]["figure"]["data"]),
                0,
            )

    def test_serve_prefers_production_artifacts_then_falls_back_to_demo(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            demo = output / "demo"
            demo.mkdir()
            (demo / "manifest.json").write_text("{}", encoding="utf-8")

            self.assertEqual(MODULE._resolve_serve_artifact_dir(output), demo)

            (output / "manifest.json").write_text("{}", encoding="utf-8")
            self.assertEqual(MODULE._resolve_serve_artifact_dir(output), output)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                FileNotFoundError,
                "Run the demo or solve command first",
            ):
                MODULE._resolve_serve_artifact_dir(Path(directory))

    def test_explorer_figures_expose_track_selection_and_harmonics(self):
        modes, continuum, diagnostics, band = MODULE._demo_datasets()
        manifest = MODULE._artifact_manifest(
            args=None,
            modes=modes,
            frequency_band=band,
            demo=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            MODULE._save_artifacts(
                output,
                modes,
                continuum,
                diagnostics,
                band,
                manifest,
            )
            _, compact_modes, compact_continuum, full_continuum = (
                MODULE._load_explorer_metadata(output)
            )

            spectrum = MODULE._spectrum_figure(
                compact_modes,
                compact_continuum,
                MODULE.LABEL_ORDER,
            )
            marker_traces = [
                trace for trace in spectrum.data if trace.customdata is not None
            ]
            self.assertEqual(spectrum.data[0].name, "Slow-sound continuum")
            self.assertEqual(len(marker_traces), len(MODULE.LABEL_ORDER))
            self.assertEqual(
                sum(len(trace.x) for trace in marker_traces),
                modes.sizes["track"],
            )

            physical_only = MODULE._spectrum_figure(
                compact_modes,
                compact_continuum,
                ["physical_gap"],
            )
            self.assertEqual(physical_only.data[0].name, "Slow-sound continuum")
            self.assertEqual(
                [trace.name for trace in physical_only.data[1:]],
                ["Physical gap (5)"],
            )
            continuum_only = MODULE._spectrum_figure(
                compact_modes,
                compact_continuum,
                ["continuum_discretization"],
            )
            self.assertEqual(
                [trace.name for trace in continuum_only.data[1:]],
                ["Continuum discretization (3)"],
            )
            no_modes = MODULE._spectrum_figure(
                compact_modes,
                compact_continuum,
                [],
            )
            self.assertEqual(len(no_modes.data), 1)
            self.assertEqual(no_modes.data[0].name, "Slow-sound continuum")

            full_spectrum = MODULE._spectrum_figure(
                compact_modes,
                full_continuum,
                [],
            )
            self.assertEqual(full_spectrum.data[0].name, "Full sound-Alfven continuum")
            self.assertIn(
                "Full sound-Alfven",
                full_spectrum.layout.title.text,
            )

            diagnostic = MODULE._resolution_diagnostic_figure(
                compact_modes,
                full_continuum,
                MODULE.LABEL_ORDER,
                "h_times_kr",
            )
            self.assertIn("full sound alfven", diagnostic.layout.title.text)
            colorbars = [
                trace.marker.colorbar.title.text
                for trace in diagnostic.data
                if getattr(trace, "marker", None) is not None
                and getattr(trace.marker, "colorbar", None) is not None
                and getattr(trace.marker.colorbar, "title", None) is not None
                and trace.marker.colorbar.title.text
            ]
            self.assertIn("h * k_r", colorbars)

            profile = MODULE._profile_figure(
                output,
                0,
                "xi_psi",
                "magnitude",
                7,
            )
            self.assertEqual(len(profile.data), 8)
            convergence = MODULE._convergence_figure(compact_modes, 0)
            self.assertGreaterEqual(len(convergence.data), 2)

    def test_empty_search_still_writes_continuum_artifacts(self):
        modes, continuum, diagnostics, band = MODULE._demo_datasets()
        empty = modes.isel(track=slice(0, 0))
        manifest = MODULE._artifact_manifest(
            args=None,
            modes=empty,
            frequency_band=band,
            demo=True,
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

            preview = (output / "explorer_preview.html").read_text()
            self.assertIn("No classified roots were returned", preview)
            app = MODULE._build_dash_app(output)
            self.assertEqual(len(app.callback_map), 0)
            self.assertIsNotNone(app.layout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
