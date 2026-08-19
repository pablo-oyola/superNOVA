"""PETSc-free regression tests for the evidence-v2 mode pipeline."""

import numpy as np
import pytest
import xarray as xr

from supernova.core.solver5_multigrid_evidence import (
    EvidenceV2Config,
    MeshPerturbationOptions,
    aggregate_catalog_status,
    bspline_prolongation_matrix,
    classify_tracks_evidence_v2,
    deterministic_shifted_bspline,
    migrate_legacy_mode_dataset,
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


def test_capacity_limited_reacquisition_does_not_make_artifact():
    tracks = _track(present=(True, True, False))
    tracks["reacquisition_status"] = (
        ("track", "level"),
        np.asarray([["not_attempted", "not_attempted", "capacity_limited"]], dtype="U48"),
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
