"""PETSc-free regression tests for the evidence-v2 mode pipeline."""

import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest
import xarray as xr

from supernova.core.solver5_multigrid_evidence import (
    EvidenceV2Config,
    MeshPerturbationOptions,
    PhysicalDisplacementMetric,
    aggregate_catalog_status,
    bspline_prolongation_matrix,
    classify_tracks_evidence_v2,
    deterministic_shifted_bspline,
    migrate_legacy_mode_dataset,
    physical_displacement_cross_gram,
    physical_displacement_paired_overlaps,
    physical_principal_angle_cosines,
    refinement_scaling_evidence,
    weighted_principal_angle_cosines,
    window_catalog_status,
)
from supernova.core.solver5_multigrid import (
    ModeClassificationThresholds,
    _nested_bspline_levels,
    _track_level_solutions,
)
from supernova.radial import BSplineFE
from supernova.core.solver4 import _build_mixed_radial_data


def _continuum(values, *, ambiguous=None):
    values = np.asarray(values, dtype=np.complex128)
    if values.ndim == 1:
        values = values[:, None]
    data = {"omega": (("psi", "branch"), values)}
    if ambiguous is not None:
        data["branch_assignment_ambiguous"] = (
            ("psi", "branch"),
            np.asarray(ambiguous, dtype=bool),
        )
    return xr.Dataset(
        data,
        coords={"psi": np.linspace(0.0, 1.0, values.shape[0]), "branch": np.arange(values.shape[1])},
    )


def _track(
    omega=(2.001, 2.0004, 2.0),
    *,
    error=1.0e-12,
    present=(True, True, True),
    overlap=(np.nan, 0.995, 0.995),
    frequency_change=(np.nan, 3.0e-4, 2.0e-4),
    width=(0.20, 0.20, 0.20),
    ipr=(2.0, 2.0, 2.0),
    kr=(4.0, 4.0, 4.0),
    h=(0.25, 0.125, 0.0625),
    center=0.5,
):
    omega = np.asarray(omega, dtype=np.complex128)
    nlevel = omega.size
    psi = np.linspace(0.0, 1.0, 65)
    shape = np.exp(-0.5 * ((psi - center) / 0.07) ** 2)[:, None]
    xi = np.repeat(shape[None, None, :, :], nlevel, axis=1)
    return xr.Dataset(
        {
            "present": (("track", "level"), np.asarray(present, dtype=bool)[None, :]),
            "omega": (("track", "level"), omega[None, :]),
            "backward_error": (
                ("track", "level"),
                np.broadcast_to(np.asarray(error, dtype=float), (1, nlevel)).copy(),
            ),
            "frequency_change": (("track", "level"), np.asarray(frequency_change)[None, :]),
            "xi_psi_overlap": (("track", "level"), np.asarray(overlap)[None, :]),
            "assignment_ambiguous": (("track", "level"), np.zeros((1, nlevel), dtype=bool)),
            "rms_width": (("track", "level"), np.asarray(width)[None, :]),
            "ipr": (("track", "level"), np.asarray(ipr)[None, :]),
            "kr": (("track", "level"), np.asarray(kr)[None, :]),
            "h_peak": (("track", "level"), np.asarray(h)[None, :]),
            "peak_radius": (("track", "level"), np.full((1, nlevel), center)),
            "xi_psi": (("track", "level", "psi", "m_pol"), xi),
        },
        coords={"track": [0], "level": np.arange(nlevel), "psi": psi, "m_pol": [10]},
    )


def _classify(tracks, continuum, catalog="capacity_limited"):
    return classify_tracks_evidence_v2(
        tracks,
        continuum,
        np.ones(tracks.sizes["psi"]) / tracks.sizes["psi"],
        catalog,
        EvidenceV2Config(),
    )


def test_capacity_limited_catalog_does_not_veto_physical_gap():
    result = _classify(_track(), _continuum([[1.0], [1.0]]))
    assert result["label"].item() == "physical_gap"
    assert result["catalog_status"].item() == "capacity_limited"
    assert result["chain_converged"].item() == "true"
    assert result["strict_gap"].item() == "true"


@pytest.mark.parametrize("catalog", ["partial", "capacity_limited", "empirically_stable", "certified_complete"])
def test_catalog_status_does_not_change_positive_mode_label(catalog):
    result = _classify(_track(), _continuum([[1.0], [1.0]]), catalog)
    assert result["label"].item() == "physical_gap"


def test_large_backward_error_is_solver_rejected():
    tracks = _track(error=np.asarray([[1.0e-12, 1.0e-12, 1.0e-4]]))
    result = _classify(tracks, _continuum([[1.0], [1.0]]))
    assert result["label"].item() == "solver_rejected"
    assert result["label"].item() != "discretization_artifact"


def test_rotated_degenerate_eigenspace_has_unit_principal_cosines():
    psi = np.linspace(0.0, 1.0, 9)
    first = np.sin(np.pi * psi)
    second = np.sin(2.0 * np.pi * psi)
    coarse = np.stack((first, second))[:, :, None]
    rotation = np.asarray([[1.0, 1.0], [-1.0, 1.0]]) / np.sqrt(2.0)
    fine = np.einsum("ij,jpk->ipk", rotation, coarse)
    cosines = weighted_principal_angle_cosines(coarse, fine, np.ones(psi.size))
    np.testing.assert_allclose(cosines, np.ones(2), atol=1.0e-12)


def _physical_metric(points, m_pol, *, radial=2.0, tangential=3.0):
    points = np.asarray(points, dtype=float)
    return PhysicalDisplacementMetric(
        points=points,
        m_pol=np.asarray(m_pol),
        mass_density=np.linspace(1.0, 2.0, points.size),
        radial_geometry_harmonics=np.full((points.size, 1), radial, dtype=complex),
        tangential_geometry_harmonics=np.full(
            (points.size, 1), tangential, dtype=complex
        ),
    )


def test_physical_overlap_is_normalized_and_phase_invariant():
    points = np.asarray([0.2, 0.7])
    weights = np.asarray([0.4, 0.6])
    metric = _physical_metric(points, [3, 4])
    radial = np.asarray([[[1.0, 2.0j], [0.5, -1.0]]], dtype=complex)
    tangential = np.asarray([[[0.2j, 1.0], [2.0, -0.4j]]], dtype=complex)
    phase = 1.7 * np.exp(0.63j)
    overlap = physical_displacement_paired_overlaps(
        radial,
        tangential,
        phase * radial,
        phase * tangential,
        weights,
        metric,
    )
    np.testing.assert_allclose(overlap, [1.0], atol=2.0e-14)


def test_physical_cross_gram_includes_nonzero_geometry_harmonics():
    points = np.asarray([0.2, 0.7])
    modes = np.asarray([3, 4])
    density = np.asarray([1.1, 1.7])
    weights = np.asarray([0.4, 0.6])
    radial_harmonics = np.asarray(
        [[2.0, 0.4 + 0.2j], [2.3, -0.3 + 0.1j]],
        dtype=complex,
    )
    tangential_harmonics = np.asarray(
        [[3.0, -0.2 + 0.3j], [2.7, 0.25 - 0.15j]],
        dtype=complex,
    )
    metric = PhysicalDisplacementMetric(
        points=points,
        m_pol=modes,
        mass_density=density,
        radial_geometry_harmonics=radial_harmonics,
        tangential_geometry_harmonics=tangential_harmonics,
    )
    left_radial = np.asarray(
        [[[1.0, 0.2j], [0.4, -0.5]], [[-0.3j, 0.7], [0.6j, 0.1]]]
    )
    left_tangential = 0.7 * left_radial + (0.1 - 0.2j)
    right_radial = np.asarray(
        [
            [[0.2, 0.8j], [-0.4j, 0.3]],
            [[1.1j, -0.2], [0.5, 0.4j]],
            [[-0.6, 0.1j], [0.2j, -0.9]],
        ]
    )
    right_tangential = -0.4j * right_radial + 0.3

    actual = physical_displacement_cross_gram(
        left_radial,
        left_tangential,
        right_radial,
        right_tangential,
        weights,
        metric,
    )

    expected = np.zeros((left_radial.shape[0], right_radial.shape[0]), complex)
    for point in range(points.size):
        radial_matrix = np.asarray(
            [
                [radial_harmonics[point, 0], 0.5 * radial_harmonics[point, 1].conjugate()],
                [0.5 * radial_harmonics[point, 1], radial_harmonics[point, 0]],
            ]
        )
        tangential_matrix = np.asarray(
            [
                [
                    tangential_harmonics[point, 0],
                    0.5 * tangential_harmonics[point, 1].conjugate(),
                ],
                [
                    0.5 * tangential_harmonics[point, 1],
                    tangential_harmonics[point, 0],
                ],
            ]
        )
        expected += weights[point] * density[point] * (
            left_radial[:, point].conjugate()
            @ radial_matrix
            @ right_radial[:, point].T
            + left_tangential[:, point].conjugate()
            @ tangential_matrix
            @ right_tangential[:, point].T
        )
    np.testing.assert_allclose(actual, expected, atol=2.0e-14)


def test_physical_principal_angles_accept_rotated_degenerate_subspace():
    points = np.linspace(0.05, 0.95, 11)
    metric = _physical_metric(points, [7])
    radial = np.stack((np.sin(np.pi * points), np.sin(2.0 * np.pi * points)))[
        :, :, None
    ]
    tangential = np.stack((np.cos(np.pi * points), np.cos(2.0 * np.pi * points)))[
        :, :, None
    ]
    rotation = np.asarray([[1.0, 1.0], [-1.0, 1.0]]) / np.sqrt(2.0)
    fine_radial = np.einsum("ij,jpk->ipk", rotation, radial)
    fine_tangential = np.einsum("ij,jpk->ipk", rotation, tangential)
    cosines = physical_principal_angle_cosines(
        radial,
        tangential,
        fine_radial,
        fine_tangential,
        np.ones(points.size),
        metric,
    )
    np.testing.assert_allclose(cosines, np.ones(2), atol=2.0e-12)


def test_rotated_degenerate_cluster_tracks_by_subspace_not_individual_vectors():
    psi = np.linspace(0.0, 1.0, 17)
    basis = np.stack((np.sin(np.pi * psi), np.sin(2.0 * np.pi * psi)))[:, :, None]
    rotations = (
        np.eye(2),
        np.asarray([[1.0, 1.0], [-1.0, 1.0]]) / np.sqrt(2.0),
        np.asarray([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2.0),
    )

    def level(rotation, drift):
        fields = np.einsum("ij,jpk->ipk", rotation, basis)
        omega = np.asarray([1.0 + drift, 1.0002 + drift])
        return xr.Dataset(
            {
                "s": ("eigenpair", omega.astype(complex) ** 2),
                "omega": ("eigenpair", omega.astype(complex)),
                "xi_psi": (("eigenpair", "psi", "m_pol"), fields),
                "slepc_backward_error": ("eigenpair", [1.0e-12, 1.0e-12]),
            },
            coords={"eigenpair": [0, 1], "psi": psi, "m_pol": [10]},
        )

    tracks = _track_level_solutions(
        (level(rotations[0], 4.0e-4), level(rotations[1], 1.0e-4), level(rotations[2], 0.0)),
        np.ones(psi.size),
        ModeClassificationThresholds(),
    )
    assert not np.any(tracks["cluster_dimension_ambiguous"].values)
    np.testing.assert_allclose(
        tracks["cluster_min_principal_cosine"].values[:, 1:],
        1.0,
        atol=1.0e-12,
    )


def test_exact_knot_insertion_prolongation_preserves_coarse_field():
    order = 3
    breakpoints = np.asarray([0.0, 0.4, 1.0])
    base = BSplineFE(
        knots=np.concatenate(
            (
                np.repeat(0.0, order),
                breakpoints[1:-1],
                np.repeat(1.0, order),
            )
        ),
        order=order,
        domain=(0.0, 1.0),
        quad_order=4,
        build_matrices=False,
    )
    coarse, fine = _nested_bspline_levels(base, levels=2)
    transfer = bspline_prolongation_matrix(coarse, fine)
    coefficient = np.asarray([0.2, -0.7, 1.3, 0.4])
    points = np.linspace(0.0, 1.0, 101)
    coarse_field = coarse.eval_basis(points).T @ coefficient
    fine_field = fine.eval_basis(points).T @ (transfer @ coefficient)
    np.testing.assert_allclose(fine_field, coarse_field, atol=2.0e-13)


def test_nested_tracking_uses_physical_metric_and_has_explicit_proxy_fallback():
    order = 3
    base = BSplineFE(
        knots=np.concatenate(
            (np.repeat(0.0, order), [0.45], np.repeat(1.0, order))
        ),
        order=order,
        domain=(0.0, 1.0),
        quad_order=4,
        build_matrices=False,
    )
    coarse, fine = _nested_bspline_levels(base, levels=2)
    points, weights = fine.get_quadrature_points()
    mixed = tuple(
        _build_mixed_radial_data(level, points, weights)
        for level in (coarse, fine)
    )
    high_coarse = np.linspace(0.2, 1.0, coarse.dofs)
    low_coarse = np.linspace(-0.4, 0.7, mixed[0].pair.low.dofs)
    high_fine = bspline_prolongation_matrix(coarse, fine) @ high_coarse
    low_fine = (
        bspline_prolongation_matrix(mixed[0].pair.low, mixed[1].pair.low)
        @ low_coarse
    )

    def level(index, high, low, radial, phase=1.0):
        xi_psi = phase * (radial.eval_basis(points).T @ high)[:, None]
        return xr.Dataset(
            {
                "s": ("eigenpair", np.asarray([(1.0 + 1.0e-4 * index) ** 2], complex)),
                "omega": ("eigenpair", np.asarray([1.0 + 1.0e-4 * index], complex)),
                "xi_psi": (("eigenpair", "psi", "m_pol"), xi_psi[None, :, :]),
                "xi_s_coefficient": (
                    ("eigenpair", "radial_dof_low", "m_pol"),
                    (phase * low)[None, :, None],
                ),
                "slepc_backward_error": ("eigenpair", [1.0e-12]),
            },
            coords={
                "eigenpair": [0],
                "psi": points,
                "m_pol": [8],
                "radial_dof_low": np.arange(low.size),
            },
        )

    levels = (
        level(1, high_coarse, low_coarse, coarse),
        level(0, high_fine, low_fine, fine, phase=np.exp(0.51j)),
    )
    metric = _physical_metric(points, [8])
    tracks = _track_level_solutions(
        levels,
        weights,
        ModeClassificationThresholds(),
        radial_levels=(coarse, fine),
        mixed_levels=mixed,
        physical_metric=metric,
    )
    assert tracks["overlap_metric"].values[0, 1] == "kinetic_energy"
    np.testing.assert_allclose(tracks["xi_psi_overlap"].values[0, 1], 1.0, atol=2.0e-12)

    proxy_levels = tuple(dataset.drop_vars("xi_s_coefficient") for dataset in levels)
    fallback = _track_level_solutions(
        proxy_levels,
        weights,
        ModeClassificationThresholds(),
        radial_levels=(coarse, fine),
        mixed_levels=mixed,
        physical_metric=metric,
    )
    assert fallback["overlap_metric"].values[0, 1] == "xi_psi_proxy"


def test_deterministic_shifted_mesh_preserves_topology_and_records_seed():
    order = 3
    breakpoints = np.linspace(0.0, 1.0, 9)
    base = BSplineFE(
        knots=np.concatenate(
            (np.repeat(0.0, order), breakpoints[1:-1], np.repeat(1.0, order))
        ),
        order=order,
        domain=(0.0, 1.0),
        quad_order=4,
        build_matrices=False,
    )
    shifted = deterministic_shifted_bspline(
        base,
        MeshPerturbationOptions(amplitude_fraction=0.15, seed=17),
    )
    assert shifted.dofs == base.dofs
    assert shifted.degree == base.degree
    assert shifted.knots[0] == base.knots[0]
    assert shifted.knots[-1] == base.knots[-1]
    assert np.all(np.diff(np.unique(shifted.knots)) > 0.0)
    assert not np.array_equal(np.unique(shifted.knots)[1:-1], breakpoints[1:-1])
    assert shifted.mesh_perturbation_seed == 17


def test_smooth_sequence_is_not_continuum_like():
    config = EvidenceV2Config()
    evidence = refinement_scaling_evidence(
        np.asarray([0.25, 0.125, 0.0625]),
        np.asarray([0.20, 0.20, 0.20]),
        np.asarray([2.0, 2.0, 2.0]),
        np.asarray([4.0, 4.0, 4.0]),
        config,
    )
    assert evidence["singular_refinement_scaling"] == "false"


def test_grid_localized_aligned_sequence_is_continuum_discretization():
    h = np.asarray([0.25, 0.125, 0.0625])
    tracks = _track(
        omega=(1.0, 1.0, 1.0),
        frequency_change=(np.nan, 0.0, 0.0),
        width=h,
        ipr=1.0 / h,
        kr=1.0 / h,
    )
    result = _classify(tracks, _continuum([[0.8], [1.2]]))
    assert result["local_continuum_alignment"].item() == "true"
    assert result["singular_refinement_scaling"].item() == "true"
    assert result["label"].item() == "continuum_discretization"


def test_global_converged_crossing_is_continuum_coupled_candidate():
    result = _classify(
        _track(omega=(1.0004, 1.0001, 1.0)),
        _continuum([[0.8], [1.2]]),
    )
    assert result["chain_converged"].item() == "true"
    assert result["has_continuum_crossing"].item() == "true"
    assert result["label"].item() == "continuum_coupled_candidate"


def test_coarse_retention_alone_does_not_assert_a_strong_global_limit():
    tracks = _track(
        present=(False, False, True),
        overlap=(np.nan, np.nan, np.nan),
        frequency_change=(np.nan, np.nan, np.nan),
    )
    tracks["coarse_space_retention"] = (
        ("track", "level"),
        np.asarray([[np.nan, np.nan, 1.0]]),
    )
    result = _classify(tracks, _continuum([[0.8], [1.2]]))
    assert result["strong_global_limit"].item() != "true"
    assert result["label"].item() == "unresolved"


def test_capacity_limited_reacquisition_does_not_make_artifact():
    tracks = _track(present=(True, True, False))
    tracks["reacquisition_status"] = (
        ("track", "level"),
        np.asarray([["not_attempted", "not_attempted", "capacity_limited"]], dtype="U48"),
    )
    result = _classify(tracks, _continuum([[1.0], [1.0]]))
    assert result["targeted_absence_confirmed"].item() == "unknown"
    assert result["label"].item() == "unresolved"


def test_partial_reacquisition_keeps_targeted_absence_unknown():
    tracks = _track(present=(True, True, False))
    tracks["reacquisition_status"] = (
        ("track", "level"),
        np.asarray([["not_attempted", "not_attempted", "partial"]], dtype="U48"),
    )
    result = _classify(tracks, _continuum([[1.0], [1.0]]))
    assert result["targeted_absence_confirmed"].item() == "unknown"
    assert result["label"].item() == "unresolved"


def test_nonpersistence_needs_targeted_absence_and_independent_evidence():
    tracks = _track(present=(True, True, False))
    tracks["reacquisition_status"] = (
        ("track", "level"),
        np.asarray(
            [["not_attempted", "not_attempted", "not_found_in_locally_complete_search"]],
            dtype="U48",
        ),
    )
    tracks["enriched_residual_converged"] = ("track", np.asarray([False]))
    tracks["perturbation_evidence_supports_nonconvergence"] = (
        "track",
        np.asarray([True]),
    )
    result = _classify(tracks, _continuum([[1.0], [1.0]]))
    assert result["label"].item() == "discretization_artifact"


def test_unrelated_branch_ambiguity_does_not_invalidate_gap():
    continuum = _continuum(
        [[1.0, 10.0], [1.0, 10.0]],
        ambiguous=[[False, True], [False, True]],
    )
    result = _classify(_track(), continuum)
    assert result["strict_gap"].item() == "true"
    assert result["label"].item() == "physical_gap"


def test_missing_optional_evidence_remains_unknown():
    tracks = _track()
    tracks = tracks.drop_vars(["xi_psi_overlap", "kr"])
    result = _classify(tracks, _continuum([[1.0], [1.0]]))
    assert result["subspace_converged"].item() == "unknown"
    assert "subspace_converged" in result["unknown_reason_codes"].item()


def test_unknown_algebraic_validity_cannot_receive_positive_label():
    h = np.asarray([0.25, 0.125, 0.0625])
    tracks = _track(
        error=np.asarray([[1.0e-12, 1.0e-12, np.nan]]),
        omega=(1.0, 1.0, 1.0),
        frequency_change=(np.nan, 0.0, 0.0),
        width=h,
        ipr=1.0 / h,
        kr=1.0 / h,
    )
    result = _classify(tracks, _continuum([[0.8], [1.2]]))
    assert result["discrete_valid"].item() == "unknown"
    assert result["label"].item() == "unresolved"


def test_window_catalog_semantics_and_legacy_saturation_regression():
    statuses = [
        window_catalog_status(
            nconv=4,
            effective_nev=4,
            solver_converged=True,
        )
        for _level in range(3)
        for _window in range(40)
    ]
    assert set(statuses) == {"capacity_limited"}
    assert aggregate_catalog_status(statuses) == "capacity_limited"
    result = _classify(
        _track(),
        _continuum([[1.0], [1.0]]),
        aggregate_catalog_status(statuses),
    )
    assert result["label"].item() == "physical_gap"


def test_legacy_serialized_mode_schema_can_be_migrated_and_reclassified():
    x = np.linspace(0.0, 1.0, 17)
    shape = np.exp(-((x - 0.5) / 0.12) ** 2)[None, :, None]
    modes = xr.Dataset(
        {
            "xi_psi_real": (("track", "x", "m_pol"), shape),
            "xi_psi_imag": (("track", "x", "m_pol"), np.zeros_like(shape)),
            "frequency_khz_by_level": (("track", "level"), [[100.05, 100.01, 100.0]]),
            "present_by_level": (("track", "level"), [[True, True, True]]),
            "backward_error_by_level": (("track", "level"), [[1.0e-12] * 3]),
            "frequency_change_by_level": (("track", "level"), [[np.nan, 4.0e-4, 1.0e-4]]),
            "xi_psi_overlap_by_level": (("track", "level"), [[np.nan, 0.995, 0.995]]),
            "assignment_ambiguous_by_level": (("track", "level"), [[False] * 3]),
            "peak_radius_by_level": (("track", "level"), [[0.5] * 3]),
            "rms_width_by_level": (("track", "level"), [[0.15] * 3]),
            "ipr_by_level": (("track", "level"), [[2.0] * 3]),
            "center": ("track", [0.5]),
            "label": ("track", ["inconclusive"]),
            "reason": ("track", ["incomplete_search_windows"]),
            "last_level": ("track", [2]),
        },
        coords={"track": [0], "level": [0, 1, 2], "x": x, "m_pol": [10]},
        attrs={"schema_version": 1, "base_spans": 40},
    )
    tracks, weights = migrate_legacy_mode_dataset(modes)
    result = classify_tracks_evidence_v2(
        tracks,
        _continuum([[50.0], [50.0]]),
        weights,
        "capacity_limited",
    )
    assert result.attrs["classification_version"] == "evidence_v2"
    assert result["label"].item() == "physical_gap"


def test_explorer_prefers_validated_full_physical_artifacts():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "no_sync"
        / "window_solver_multigrid.py"
    )
    spec = importlib.util.spec_from_file_location(
        "window_solver_multigrid_test_module",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as directory:
        artifact_dir = Path(directory)
        manifests = {
            "manifest.json": "modes.nc",
            "manifest_physical_evidence_v2.json": "modes_physical_evidence_v2.nc",
            "manifest_physical_full_evidence_v2.json": (
                "modes_physical_full_evidence_v2.nc"
            ),
        }
        for manifest_name, modes_name in manifests.items():
            (artifact_dir / manifest_name).write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "artifacts": {"modes": modes_name},
                    }
                ),
                encoding="utf-8",
            )
            (artifact_dir / modes_name).touch()

        manifest = module._load_manifest(artifact_dir)
        assert manifest["artifacts"]["modes"] == (
            "modes_physical_full_evidence_v2.nc"
        )
        assert module._mode_artifact_path(artifact_dir).name == (
            "modes_physical_full_evidence_v2.nc"
        )
