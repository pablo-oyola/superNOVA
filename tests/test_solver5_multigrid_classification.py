"""Deterministic classification tests for the solver5 multigrid workflow."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from scipy.sparse import coo_array

import supernova
import supernova.core.solver5_multigrid as multigrid
from supernova.core.solver5_multigrid import (
    ModeClassificationThresholds,
    MultigridPEPResult,
    MultigridSearchOptions,
    _WindowRecord,
    _classify_tracks,
    _continuum_evidence,
    _maximum_cardinality_overlap_assignment,
    _solve_master_continuum,
    _track_level_solutions,
    _window_diagnostics_dataset,
)


def _crossing_continuum() -> xr.Dataset:
    """One branch crossing omega=1 between, rather than at, stored points."""
    return xr.Dataset(
        data_vars={
            "s": (("psi", "branch"), np.square([[0.8], [1.2]])),
            "omega": (("psi", "branch"), [[0.8 + 0.0j], [1.2 + 0.0j]]),
        },
        coords={"psi": [0.0, 1.0], "branch": [0]},
    )


def _classification_tracks(
    frequencies=(2.0, 1.0, 2.0, 1.205),
) -> xr.Dataset:
    """Return converged three-level tracks whose final evidence is controlled."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    ntrack = frequencies.size
    nlevel = 3
    psi = np.array([0.0, 0.5, 1.0])
    omega = np.repeat(frequencies[:, None], nlevel, axis=1).astype(np.complex128)
    xi_psi = np.zeros((ntrack, nlevel, psi.size, 1), dtype=np.complex128)
    xi_psi[:, :, 1, 0] = 1.0
    backward_error = np.full((ntrack, nlevel), 1.0e-12)
    if ntrack >= 3:
        backward_error[2, -1] = 1.0e-4

    return xr.Dataset(
        data_vars={
            "present": (("track", "level"), np.ones((ntrack, nlevel), dtype=bool)),
            "omega": (("track", "level"), omega),
            "backward_error": (("track", "level"), backward_error),
            "frequency_change": (
                ("track", "level"),
                np.tile([np.nan, 5.0e-4, 5.0e-4], (ntrack, 1)),
            ),
            "xi_psi_overlap": (
                ("track", "level"),
                np.tile([np.nan, 0.995, 0.995], (ntrack, 1)),
            ),
            "assignment_ambiguous": (
                ("track", "level"),
                np.zeros((ntrack, nlevel), dtype=bool),
            ),
            "rms_width": (
                ("track", "level"),
                np.full((ntrack, nlevel), 0.2),
            ),
            "ipr": (("track", "level"), np.ones((ntrack, nlevel))),
            "xi_psi": (("track", "level", "psi", "m_pol"), xi_psi),
        },
        coords={
            "track": np.arange(ntrack),
            "level": np.arange(nlevel),
            "psi": psi,
            "m_pol": [10],
        },
    )


def _level_solution(omega: float, phase: float) -> xr.Dataset:
    psi = np.array([0.0, 0.5, 1.0])
    shape = np.array([1.0, 2.0, 1.0]) * np.exp(1.0j * phase)
    return xr.Dataset(
        data_vars={
            "s": ("eigenpair", [complex(omega**2)]),
            "omega": ("eigenpair", [complex(omega)]),
            "xi_psi": (("eigenpair", "psi", "m_pol"), shape[None, :, None]),
            "residual_norm": ("eigenpair", [1.0e-13]),
            "slepc_backward_error": ("eigenpair", [1.0e-13]),
        },
        coords={"eigenpair": [0], "psi": psi, "m_pol": [10]},
    )


def test_assignment_maximizes_cardinality_before_overlap_greedy_trap():
    # r0 can use either candidate, while r1 can only use c0.  A greedy
    # highest-overlap choice r0-c0 would therefore lose the second match.
    reference = np.array([1.00, 1.02])
    candidate = np.array([1.01, 0.99])
    overlaps = np.array([[0.99, 0.80], [0.90, 0.10]])

    rows, columns, differences, _ = _maximum_cardinality_overlap_assignment(
        reference,
        candidate,
        overlaps,
        frequency_gate=0.02,
        ambiguity_delta=1.0e-3,
    )

    assert set(zip(rows.tolist(), columns.tolist())) == {(0, 1), (1, 0)}
    assert differences.size == 2
    assert np.all(differences <= 0.02)


def test_assignment_marks_near_degenerate_alternatives_ambiguous():
    rows, columns, _, ambiguous = _maximum_cardinality_overlap_assignment(
        np.array([1.0000, 1.0010]),
        np.array([1.0002, 1.0008]),
        np.array([[0.9900, 0.9895], [0.9897, 0.9900]]),
        frequency_gate=0.02,
        ambiguity_delta=1.0e-3,
    )

    assert set(zip(rows.tolist(), columns.tolist())) == {(0, 0), (1, 1)}
    assert ambiguous.tolist() == [True, True]


def test_continuum_crossing_and_mode_proximity_are_interpolated():
    crossing, clearance, power_fraction, complete = _continuum_evidence(
        1.0 + 0.0j,
        np.array([[0.0], [1.0], [0.0]], dtype=np.complex128),
        np.array([0.25, 0.50, 0.75]),
        np.ones(3),
        _crossing_continuum(),
        ModeClassificationThresholds(),
    )

    assert crossing is True
    assert clearance == pytest.approx(0.0)
    assert power_fraction == pytest.approx(1.0)
    assert complete is True


def test_continuum_branches_follow_eigenvectors_through_rank_exchange():
    identity = np.eye(2, dtype=np.complex128)
    A = np.array(
        [
            np.diag([0.8**2, 1.2**2]),
            np.diag([1.2**2, 0.8**2]),
        ],
        dtype=np.complex128,
    )
    sb = SimpleNamespace(
        A=coo_array(A),
        B=coo_array(np.repeat(identity[None, :, :], 2, axis=0)),
        density=np.ones(2),
        ntor=3,
        m_range=np.array([1]),
        max_dm=0,
    )

    continuum = _solve_master_continuum(sb, np.array([0.0, 1.0]))

    np.testing.assert_allclose(
        continuum["omega"].values.real,
        [[0.8, 1.2], [1.2, 0.8]],
    )
    assert not np.any(continuum["branch_assignment_ambiguous"].values)
    crossing, clearance, _, complete = _continuum_evidence(
        1.0 + 0.0j,
        np.ones((3, 1), dtype=np.complex128),
        np.array([0.0, 0.5, 1.0]),
        np.ones(3),
        continuum,
        ModeClassificationThresholds(),
    )
    assert crossing is True
    assert clearance == pytest.approx(0.0)
    assert complete is True


def test_invalid_continuum_segment_cannot_certify_or_bridge_a_gap():
    continuum = xr.Dataset(
        {
            "omega": (
                ("psi", "branch"),
                np.array([[0.8], [np.nan], [1.2]], dtype=np.complex128),
            ),
            "s": (
                ("psi", "branch"),
                np.array([[0.64], [np.nan], [1.44]], dtype=np.complex128),
            ),
        },
        coords={"psi": [0.0, 0.5, 1.0], "branch": [0]},
    )
    crossing, _, power_fraction, complete = _continuum_evidence(
        1.0 + 0.0j,
        np.array([[0.0], [1.0], [0.0]], dtype=np.complex128),
        np.array([0.25, 0.50, 0.75]),
        np.ones(3),
        continuum,
        ModeClassificationThresholds(),
    )

    assert crossing is False
    assert complete is False
    assert power_fraction == pytest.approx(0.0)

    classified = _classify_tracks(
        _classification_tracks(frequencies=(2.0,)),
        continuum,
        np.ones(3),
        np.ones(3, dtype=bool),
        ModeClassificationThresholds(),
    )
    assert classified["label"].item() == "inconclusive"
    assert not bool(classified["strict_gap"].item())
    assert "incomplete_continuum_coverage" in classified["reason"].item()


def test_ambiguous_continuum_branch_tracking_cannot_certify_a_gap():
    continuum = _crossing_continuum()
    continuum["branch_assignment_ambiguous"] = (
        ("psi", "branch"),
        np.array([[False], [True]]),
    )
    tracks = _classification_tracks(frequencies=(2.0,))

    classified = _classify_tracks(
        tracks,
        continuum,
        np.ones(3),
        np.ones(3, dtype=bool),
        ModeClassificationThresholds(),
    )

    assert classified["label"].item() == "inconclusive"
    assert not bool(classified["continuum_complete"].item())
    assert not bool(classified["strict_gap"].item())
    assert "incomplete_continuum_coverage" in classified["reason"].item()


def test_classifier_produces_all_four_labels_with_independent_evidence():
    classified = _classify_tracks(
        _classification_tracks(),
        _crossing_continuum(),
        np.ones(3),
        np.ones(3, dtype=bool),
        ModeClassificationThresholds(),
    )

    assert classified["label"].values.tolist() == [
        "physical_gap",
        "continuum_resonance",
        "numerical_artifact",
        "inconclusive",
    ]
    assert bool(classified["strict_gap"].values[0])
    assert bool(classified["strict_continuum_crossing"].values[1])
    assert bool(classified["resonance_signature"].values[1])
    assert not bool(classified["numerically_valid"].values[2])
    assert "marginal_continuum_clearance" in classified["reason"].values[3]


def test_invalid_intermediate_level_cannot_certify_a_physical_gap():
    tracks = _classification_tracks(frequencies=(2.0,))
    tracks["backward_error"].values[0, 0] = 1.0e-4

    classified = _classify_tracks(
        tracks,
        _crossing_continuum(),
        np.ones(3),
        np.ones(3, dtype=bool),
        ModeClassificationThresholds(),
    )

    assert bool(classified["numerically_valid"].item())
    assert not bool(classified["chain_numerically_valid"].item())
    assert not bool(classified["converged_chain"].item())
    assert classified["label"].item() == "inconclusive"
    assert "invalid_intermediate_level" in classified["reason"].item()


@pytest.mark.parametrize(
    ("level_complete", "ambiguous", "reason"),
    [
        ([True, False, True], False, "incomplete_search_windows"),
        ([True, True, True], True, "ambiguous_mode_assignment"),
    ],
)
def test_incomplete_or_ambiguous_track_never_receives_physical_gap(
    level_complete,
    ambiguous,
    reason,
):
    tracks = _classification_tracks(frequencies=(2.0,))
    tracks["assignment_ambiguous"].values[0, -1] = ambiguous

    classified = _classify_tracks(
        tracks,
        _crossing_continuum(),
        np.ones(3),
        np.asarray(level_complete, dtype=bool),
        ModeClassificationThresholds(),
    )

    assert bool(classified["strict_gap"].values[0])
    assert classified["label"].item() == "inconclusive"
    assert reason in classified["reason"].item()


def test_losing_near_degenerate_track_is_explicitly_ambiguous():
    psi = np.array([0.0, 0.5, 1.0])

    def level(frequencies):
        frequencies = np.asarray(frequencies, dtype=float)
        fields = np.ones((frequencies.size, psi.size, 1), dtype=np.complex128)
        return xr.Dataset(
            data_vars={
                "s": ("eigenpair", frequencies.astype(complex) ** 2),
                "omega": ("eigenpair", frequencies.astype(complex)),
                "xi_psi": (("eigenpair", "psi", "m_pol"), fields),
                "residual_norm": ("eigenpair", np.full(frequencies.size, 1.0e-12)),
                "slepc_backward_error": (
                    "eigenpair",
                    np.full(frequencies.size, 1.0e-12),
                ),
            },
            coords={
                "eigenpair": np.arange(frequencies.size),
                "psi": psi,
                "m_pol": [10],
            },
        )

    thresholds = ModeClassificationThresholds()
    tracks = _track_level_solutions(
        (level([2.0, 2.001]), level([2.0005]), level([2.0005])),
        np.ones(psi.size),
        thresholds,
    )
    continuum = xr.Dataset(
        {
            "omega": (("psi", "branch"), np.full((2, 1), 3.0 + 0.0j)),
            "s": (("psi", "branch"), np.full((2, 1), 9.0 + 0.0j)),
        },
        coords={"psi": [0.0, 1.0], "branch": [0]},
    )
    classified = _classify_tracks(
        tracks,
        continuum,
        np.ones(psi.size),
        np.ones(3, dtype=bool),
        thresholds,
    )

    assert tracks.sizes["track"] == 2
    assert np.all(classified["assignment_ambiguous"].values)
    assert classified["label"].values.tolist() == ["inconclusive", "inconclusive"]
    assert all(
        "ambiguous_mode_assignment" in reason
        for reason in classified["reason"].values.tolist()
    )


def test_public_defaults_and_result_xarray_topology():
    search = MultigridSearchOptions()
    thresholds = ModeClassificationThresholds()
    assert (search.nev, search.tol, search.maxit) == (8, 1.0e-8, 200)
    assert (search.window_count, search.window_padding) == (40, 0.10)
    assert search.region_vscale == pytest.approx(0.25)
    assert search.dedup_relative_s == pytest.approx(1.0e-6)
    assert search.dedup_overlap == pytest.approx(0.99)
    assert thresholds.frequency_gate == pytest.approx(0.02)
    assert thresholds.fine_frequency_change == pytest.approx(1.0e-3)
    assert thresholds.medium_fine_overlap == pytest.approx(0.98)
    assert thresholds.continuum_clearance == pytest.approx(5.0e-3)

    levels = (
        _level_solution(2.0010, 0.0),
        _level_solution(2.0004, 0.7),
        _level_solution(2.0000, -1.1),
    )
    weights = np.array([0.25, 0.50, 0.25])
    tracks = _track_level_solutions(levels, weights, thresholds)
    continuum = _crossing_continuum()
    classification = _classify_tracks(
        tracks,
        continuum,
        weights,
        np.ones(3, dtype=bool),
        thresholds,
    )
    records = [
        _WindowRecord(
            level=level,
            window=0,
            parent=-1,
            depth=0,
            core_omega_min=1.8,
            core_omega_max=2.2,
            solve_omega_min=1.76,
            solve_omega_max=2.24,
            target_s=4.0 + 0.0j,
            nconv=1,
            nreturned=1,
            saturated=False,
            leaf=True,
            complete=True,
            reason="converged",
            iterations=3,
        )
        for level in range(3)
    ]
    diagnostics = _window_diagnostics_dataset(records, nlevels=3)
    diagnostics["level_complete"] = ("level", np.ones(3, dtype=bool))
    result = MultigridPEPResult(
        level_solutions=levels,
        mode_tracks=tracks,
        window_diagnostics=diagnostics,
        continuum=continuum,
        classification=classification,
    )

    assert dict(result.mode_tracks.sizes) == {
        "track": 1,
        "level": 3,
        "psi": 3,
        "m_pol": 1,
    }
    assert dict(result.window_diagnostics.sizes) == {"level": 3, "window": 1}
    assert dict(result.continuum.sizes) == {"psi": 2, "branch": 1}
    assert dict(result.classification.sizes) == {"track": 1, "level": 3}
    assert result.labels.identical(result.classification["label"])
    assert result.labels.item() == "physical_gap"

    assert supernova.MultigridSearchOptions is MultigridSearchOptions
    assert supernova.ModeClassificationThresholds is ModeClassificationThresholds
    assert supernova.MultigridPEPResult is MultigridPEPResult
    assert supernova.superSolver5Multigrid is multigrid.superSolver5Multigrid
    assert set(multigrid.__all__) == {
        "EvidenceV2Config",
        "ModeClassificationThresholds",
        "MultigridPEPResult",
        "MultigridSearchOptions",
        "superSolver5Multigrid",
    }
