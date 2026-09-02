"""Deterministic unit tests for the generic symmetry-verification layer."""

import json

import numpy as np
import pytest
from scipy import sparse

from supernova.solve.stability import StabilitySolver
from supernova.radial.profile import Profile
from supernova.verification import (
    Parity,
    Tolerance,
    build_reflection_operator,
    check_eigenvector_parity,
    check_eigenvector_transformed_realness,
    check_eigenspace_transformed_realness,
    check_grid_parity,
    check_hermitian,
    check_matrix_adjointness,
    check_operator_intertwining,
    check_parity_sector_mixing,
    check_profile_fourier_parity,
    check_reflection_covariance,
    check_reflection_involution,
    check_skew_hermitian,
    check_transformed_realness,
    check_weighted_adjointness,
    cluster_eigenvalues,
    compare_positive_subblock,
    format_summary,
    infer_field_phase_transform,
    parity_adapt_eigenspace,
    phase_transform_matrix,
    results_to_json,
)


STRICT = Tolerance(atol=2.0e-14, rtol=2.0e-13, floor=1.0e-15)


def test_verification_result_serialization_and_unclassified_status_are_json_safe():
    theta = 2.0 * np.pi * np.arange(8) / 8
    unknown = check_grid_parity(
        np.cos(theta),
        Parity.UNKNOWN,
        theta=theta,
        tolerance=STRICT,
        name="unclassified coefficient",
    )
    assert unknown.passed is None
    assert unknown.status == "UNCLASSIFIED"
    assert json.loads(unknown.to_json())["status"] == "UNCLASSIFIED"
    assert json.loads(results_to_json([unknown]))[0]["passed"] is None
    summary = format_summary([unknown])
    assert "1 unclassified" in summary
    assert "UNCLASSIFIED: unclassified coefficient" in summary


def test_periodic_weighted_grid_parity_and_worst_radial_location():
    theta = 2.0 * np.pi * np.arange(16) / 16
    radial = np.array([0.2, 0.5, 0.9])
    weights = 1.0 + 0.2 * np.cos(theta)
    even = (1.0 + radial[:, None]) * (
        2.0 + 0.3 * np.cos(theta)[None, :] - 0.1 * np.cos(3.0 * theta)[None, :]
    )
    odd = (1.0 + radial[:, None]) * (
        0.4 * np.sin(theta)[None, :] + 0.2 * np.sin(2.0 * theta)[None, :]
    )
    even_result = check_grid_parity(
        even,
        Parity.EVEN,
        theta=theta,
        weights=weights,
        radial_coordinates=radial,
        tolerance=STRICT,
    )
    odd_result = check_grid_parity(
        odd,
        Parity.ODD,
        theta=theta,
        weights=weights,
        radial_coordinates=radial,
        tolerance=STRICT,
    )
    assert even_result.passed
    assert odd_result.passed

    broken = even.copy()
    broken[2, 3] += 0.1
    failure = check_grid_parity(
        broken,
        Parity.EVEN,
        theta=theta,
        weights=weights,
        radial_coordinates=radial,
        tolerance=STRICT,
    )
    assert not failure.passed
    assert failure.worst_location["radial_index"] == 2
    assert failure.worst_location["radial_location"] == pytest.approx(0.9)


def test_grid_parity_rejects_a_periodic_grid_without_reflected_nodes():
    theta = np.array([0.0, 0.7, 2.0, 4.0])
    with pytest.raises(ValueError, match="not closed under periodic reflection"):
        check_grid_parity(np.cos(theta), Parity.EVEN, theta=theta)


def test_one_sided_profile_fourier_parity_and_forbidden_components():
    radius = np.array([0.1, 0.6, 0.95])
    even_amplitudes = np.zeros((radius.size, 3, 1), dtype=np.complex128)
    even_amplitudes[:, 0, 0] = np.array([2.0, 2.1, 2.2])
    even_amplitudes[:, 1, 0] = np.array([0.3, 0.2, 0.1])
    even_amplitudes[:, 2, 0] = np.array([-0.1, 0.05, 0.2])
    even_profile = Profile(
        radius,
        even_amplitudes,
        mpol=[0, 1, 2],
        ntor=[0],
    )
    even_report = check_profile_fourier_parity(
        even_profile,
        Parity.EVEN,
        tolerance=STRICT,
        name="even synthetic",
    )
    assert even_report.passed
    assert even_report.parity.relative_error < 1.0e-14
    assert even_report.forbidden_component.relative_error < 1.0e-14

    odd_amplitudes = np.zeros_like(even_amplitudes)
    odd_amplitudes[:, 1, 0] = 1j * np.array([0.3, 0.2, 0.1])
    odd_amplitudes[:, 2, 0] = 1j * np.array([-0.1, 0.05, 0.2])
    odd_profile = Profile(
        radius,
        odd_amplitudes,
        mpol=[0, 1, 2],
        ntor=[0],
    )
    odd_report = check_profile_fourier_parity(
        odd_profile,
        Parity.ODD,
        tolerance=STRICT,
        name="odd synthetic",
    )
    assert odd_report.passed

    contaminated = odd_profile.copy()
    contaminated.data.loc[dict(x=radius[1], mpol=2, ntor=0)] += 2.0e-3
    failure = check_profile_fourier_parity(
        contaminated,
        Parity.ODD,
        tolerance=STRICT,
    )
    assert not failure.passed
    assert failure.forbidden_component.worst_location["harmonic"] == 2
    assert failure.forbidden_component.worst_location["radial_location"] == pytest.approx(
        radius[1]
    )


def test_one_sided_profile_rejects_nonaxisymmetric_theta_parity():
    profile = Profile(
        np.array([0.5]),
        np.zeros((1, 2, 2), dtype=np.complex128),
        mpol=[0, 1],
        ntor=[0, 1],
    )
    with pytest.raises(ValueError, match="only representable for ntor=0"):
        check_profile_fourier_parity(profile, Parity.EVEN, ntor=1)


def test_dense_and_sparse_frobenius_adjoint_gates():
    rng = np.random.default_rng(1405)
    raw = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    hermitian = raw + raw.conj().T
    skew = sparse.csr_matrix(raw - raw.conj().T)
    assert check_hermitian(hermitian, tolerance=STRICT).passed
    assert check_skew_hermitian(skew, tolerance=STRICT).passed

    broken = hermitian.copy()
    broken[1, 4] += 1.0e-4j
    failure = check_matrix_adjointness(
        broken,
        kind="hermitian",
        tolerance=STRICT,
    )
    assert not failure.passed
    assert {failure.worst_location["row"], failure.worst_location["column"]} == {
        1,
        4,
    }
    serialized = json.loads(failure.to_json())
    assert set(serialized["worst_location"]["actual"]) == {"real", "imag"}


def test_weighted_adjointness_dense_and_sparse():
    weight_dense = np.diag([1.0, 2.0, 4.0])
    hermitian_form = np.array(
        [[2.0, 0.3 + 0.2j, -0.1], [0.3 - 0.2j, 1.0, 0.4j], [-0.1, -0.4j, 3.0]]
    )
    operator_dense = np.linalg.solve(weight_dense, hermitian_form)
    dense_result = check_weighted_adjointness(
        operator_dense,
        weight_dense,
        tolerance=STRICT,
    )
    sparse_result = check_weighted_adjointness(
        sparse.csr_matrix(operator_dense),
        sparse.csr_matrix(weight_dense),
        tolerance=STRICT,
    )
    assert dense_result.passed
    assert sparse_result.passed


def test_rectangular_sparse_two_sided_phase_transform_and_realness():
    row_labels = np.array(["p", "s"], dtype=object)
    trial_labels = np.array(["s", "p", "s"], dtype=object)
    phases = {"p": 0.0, "s": 0.5 * np.pi}
    real_matrix = np.array([[1.0, -0.3, 0.4], [0.2, 0.5, -0.7]])
    row_phase = np.array([phases[label] for label in row_labels])
    trial_phase = np.array([phases[label] for label in trial_labels])
    original = (
        np.exp(1j * row_phase)[:, None]
        * real_matrix
        * np.exp(-1j * trial_phase)[None, :]
    )
    original_sparse = sparse.csr_matrix(original)

    result = check_transformed_realness(
        original_sparse,
        row_labels=row_labels,
        trial_labels=trial_labels,
        field_phases=phases,
        tolerance=STRICT,
    )
    assert result.passed
    transformed = phase_transform_matrix(
        original_sparse,
        row_labels=row_labels,
        trial_labels=trial_labels,
        field_phases=phases,
    )
    np.testing.assert_allclose(transformed.toarray(), real_matrix, atol=2.0e-15)
    with pytest.raises(ValueError, match="require explicit trial"):
        check_transformed_realness(original_sparse, row_phases=row_phase)


def test_exhaustive_quarter_turn_phase_inference_across_matrices():
    labels = np.array(["p", "s"], dtype=object)
    true_phases = np.array([0.0, 0.5 * np.pi])
    diagonal = np.exp(1j * true_phases)

    real_zero = np.array([[1.0, 0.3], [-0.6, 2.0]])
    real_one = np.array([[0.2, -1.1], [0.8, 0.4]])
    matrices = {
        "T0": diagonal[:, None] * real_zero * np.conjugate(diagonal)[None, :],
        "T1": sparse.csr_matrix(
            diagonal[:, None] * real_one * np.conjugate(diagonal)[None, :]
        ),
    }
    inference = infer_field_phase_transform(
        matrices,
        row_labels=labels,
        trial_labels=labels,
        reference_field="p",
        tolerance=STRICT,
    )
    assert inference.passed
    assert inference.tested_assignments == 4
    assert inference.quarter_turns["p"] == 0
    assert inference.quarter_turns["s"] in (1, 3)
    assert all(result.passed for result in inference.matrix_results.values())
    assert json.dumps(inference.to_dict(), allow_nan=False)


def test_reflection_operator_involution_covariance_and_sector_mixing():
    modes = np.array([-2, -1, 1, 2])
    field_mix = np.array([[0.0, 1.0], [1.0, 0.0]])
    reflection = build_reflection_operator(
        modes,
        field_names=["a", "b"],
        radial_dofs=[2, 2],
        field_reflection=field_mix,
    )
    assert check_reflection_involution(reflection, tolerance=STRICT).passed

    operator = sparse.eye(reflection.shape[0], format="csr") + 0.3 * reflection
    assert check_reflection_covariance(
        operator,
        reflection,
        tolerance=STRICT,
    ).passed
    mixing = check_parity_sector_mixing(
        operator,
        reflection,
        tolerance=STRICT,
    )
    assert mixing.passed

    broken = operator.tolil()
    broken[0, 1] += 0.2
    assert not check_reflection_covariance(
        broken.tocsr(),
        reflection,
        tolerance=STRICT,
    ).passed


def test_rectangular_operator_intertwining_uses_separate_row_and_trial_actions():
    rng = np.random.default_rng(31)
    modes = np.array([-1, 1])
    row_action = build_reflection_operator(
        modes,
        field_names=["a", "b"],
        radial_dofs=[1, 1],
        field_signs={"a": 1.0, "b": -1.0},
    )
    trial_action = build_reflection_operator(
        modes,
        field_names=["a", "b"],
        radial_dofs=[2, 2],
        field_signs={"a": 1.0, "b": -1.0},
    )
    seed = sparse.csr_matrix(
        rng.normal(size=(row_action.shape[0], trial_action.shape[0]))
    )
    operator = 0.5 * (seed + row_action @ seed @ trial_action)
    result = check_operator_intertwining(
        operator,
        row_action,
        trial_action,
        tolerance=STRICT,
    )
    assert result.passed
    sector_result = check_parity_sector_mixing(
        operator,
        trial_action,
        row_reflection=row_action,
        tolerance=STRICT,
    )
    assert sector_result.passed


def _layout_indices(source_modes, selected_modes, radial_counts):
    slots = {int(mode): index for index, mode in enumerate(source_modes)}
    offsets = np.cumsum(
        [0, *(count * len(source_modes) for count in radial_counts)]
    )
    output = []
    for field, radial_count in enumerate(radial_counts):
        for radial in range(radial_count):
            output.extend(
                offsets[field]
                + radial * len(source_modes)
                + np.array([slots[int(mode)] for mode in selected_modes])
            )
    return np.asarray(output, dtype=np.int64)


def test_positive_subblock_comparison_supports_mixed_row_and_trial_radial_counts():
    rng = np.random.default_rng(2026)
    positive_modes = np.array([1, 2])
    doubled_modes = np.array([-2, -1, 1, 2])
    row_counts = (1, 1, 1, 1)
    trial_counts = (2, 2, 1, 1)
    fields = ("p1", "xi_psi", "xi_s", "div_xi")
    doubled = sparse.csr_matrix(
        rng.normal(
            size=(
                sum(row_counts) * doubled_modes.size,
                sum(trial_counts) * doubled_modes.size,
            )
        )
    )
    row_indices = _layout_indices(doubled_modes, positive_modes, row_counts)
    trial_indices = _layout_indices(doubled_modes, positive_modes, trial_counts)
    production = doubled[row_indices][:, trial_indices].copy()
    result = compare_positive_subblock(
        production,
        doubled,
        production_m_values=positive_modes,
        doubled_m_values=doubled_modes,
        row_radial_dofs=row_counts,
        trial_radial_dofs=trial_counts,
        field_names=fields,
        tolerance=STRICT,
    )
    assert result.passed
    np.testing.assert_array_equal(result.metadata["row_indices"], row_indices)
    np.testing.assert_array_equal(result.metadata["trial_indices"], trial_indices)

    changed = production.tolil()
    changed[3, 5] += 1.0e-3
    failure = compare_positive_subblock(
        changed.tocsr(),
        doubled,
        production_m_values=positive_modes,
        doubled_m_values=doubled_modes,
        row_radial_dofs=row_counts,
        trial_radial_dofs=trial_counts,
        field_names=fields,
        tolerance=STRICT,
    )
    assert not failure.passed
    assert (failure.worst_location["row"], failure.worst_location["column"]) == (
        3,
        5,
    )


def test_eigenvector_transformed_realness_and_reflection_parity():
    labels = np.array(["p", "s", "p", "s"], dtype=object)
    field_phases = {"p": 0.0, "s": 0.5 * np.pi}
    real_vector = np.array([1.0, -0.4, 0.2, 0.8])
    dof_phases = np.array([field_phases[label] for label in labels])
    vector = np.exp(0.37j) * np.exp(1j * dof_phases) * real_vector
    weight = sparse.diags([1.0, 2.0, 3.0, 4.0], format="csr")
    realness = check_eigenvector_transformed_realness(
        vector,
        weight,
        labels=labels,
        field_phases=field_phases,
        tolerance=STRICT,
    )
    assert realness.passed
    assert realness.metadata["phase_method"] == "bilinear"

    reflection = build_reflection_operator(np.array([-1, 1]))
    even = check_eigenvector_parity(
        np.array([1.0, 1.0]),
        reflection,
        tolerance=STRICT,
    )
    odd = check_eigenvector_parity(
        np.array([1.0, -1.0]),
        reflection,
        tolerance=STRICT,
    )
    assert even.classification == "EVEN"
    assert even.expectation == pytest.approx(1.0)
    assert odd.classification == "ODD"
    assert odd.expectation == pytest.approx(-1.0)


def test_w_orthonormal_parity_adaptation_handles_a_complex_mixed_cluster():
    reflection = build_reflection_operator(np.array([-1, 1]))
    parity_basis = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    unitary_mix = np.array(
        [[1.0, 1.0j], [1.0j, 1.0]],
        dtype=np.complex128,
    ) / np.sqrt(2.0)
    clustered_vectors = parity_basis @ unitary_mix
    weight = 2.0 * np.eye(2)
    adapted = parity_adapt_eigenspace(
        clustered_vectors,
        reflection,
        weight,
        tolerance=STRICT,
    )
    assert adapted.passed
    np.testing.assert_allclose(
        np.sort(adapted.parity_eigenvalues),
        np.array([-1.0, 1.0]),
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        adapted.basis.conj().T @ weight @ adapted.basis,
        np.eye(2),
        atol=2.0e-13,
    )
    assert json.dumps(adapted.to_dict(), allow_nan=False)


def test_eigenvalue_clustering_is_deterministic_and_transitive():
    values = np.array(
        [1.0, 1.0 + 0.9e-5, 2.0, 1.0 + 1.8e-5, 2.0 + 0.5e-5]
    )
    clusters = cluster_eigenvalues(values, atol=1.0e-5, rtol=0.0)
    assert len(clusters) == 2
    np.testing.assert_array_equal(clusters[0], np.array([0, 1, 3]))
    np.testing.assert_array_equal(clusters[1], np.array([2, 4]))


def test_transformed_real_eigenspace_accepts_complex_rotations_of_a_real_cluster():
    real_basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, -0.2],
            [0.3, 0.7],
        ]
    )
    unitary_mix = np.array(
        [[1.0, 1.0j], [1.0j, 1.0]],
        dtype=np.complex128,
    ) / np.sqrt(2.0)
    labels = np.array(["p", "s", "p", "s"], dtype=object)
    field_phases = {"p": 0.0, "s": 0.5 * np.pi}
    dof_phases = np.array([field_phases[label] for label in labels])
    vectors = np.exp(1j * dof_phases)[:, None] * real_basis @ unitary_mix
    weight = sparse.diags([1.0, 2.0, 1.5, 0.8], format="csr")
    result = check_eigenspace_transformed_realness(
        vectors,
        weight,
        labels=labels,
        field_phases=field_phases,
        tolerance=STRICT,
    )
    assert result.passed
    assert result.real_rank == vectors.shape[1]
    np.testing.assert_allclose(np.imag(result.basis), 0.0, atol=2.0e-13)
    np.testing.assert_allclose(
        result.basis.conj().T @ weight @ result.basis,
        np.eye(2),
        atol=2.0e-13,
    )


def test_transformed_real_eigenspace_rejects_non_conjugation_invariant_line():
    vector = np.array([[1.0], [1.0j]]) / np.sqrt(2.0)
    result = check_eigenspace_transformed_realness(
        vector,
        tolerance=STRICT,
    )
    assert not result.passed
    assert not result.conjugation_invariance.passed
    assert not result.real_basis.passed
    assert result.real_rank == 2


@pytest.mark.parametrize(
    "values",
    [
        np.array([-4, -2, 0, 3, 4], dtype=np.int32),
        np.array([-4.0, -1.0, 2.0, 4.0]),
    ],
)
def test_solver2_explicit_m_validator_accepts_sorted_unique_signed_integers(values):
    validated = StabilitySolver._validate_explicit_m_values(values, available_mmax=4)
    np.testing.assert_array_equal(validated, values.astype(np.int64))
    assert validated.dtype == np.int64


def test_solver2_explicit_signed_assembly_forwards_modes_without_touching_cache(
    monkeypatch,
):
    solver = object.__new__(StabilitySolver)
    solver.M = Profile(
        np.array([0.5]),
        np.zeros((1, 11, 1), dtype=np.complex128),
        mpol=np.arange(-5, 6),
        ntor=[0],
    )
    production_cache = object()
    solver.solver_blocks = production_cache
    assembled_blocks = object()
    calls = []

    def fake_assemble(self, *, ntor, m_range, max_dm):
        calls.append(
            {
                "ntor": ntor,
                "m_range": m_range.copy(),
                "max_dm": max_dm,
            }
        )
        return assembled_blocks

    monkeypatch.setattr(
        StabilitySolver,
        "_assemble_solver_blocks_for_m_range",
        fake_assemble,
    )
    requested = np.array([-5.0, -2.0, 0.0, 3.0, 5.0])

    result = solver.build_solver_blocks_for_m_values(
        ntor=6,
        m_values=requested,
        max_dm=2,
    )

    assert result is assembled_blocks
    assert solver.solver_blocks is production_cache
    assert len(calls) == 1
    assert calls[0]["ntor"] == 6
    assert calls[0]["max_dm"] == 2
    np.testing.assert_array_equal(
        calls[0]["m_range"],
        np.array([-5, -2, 0, 3, 5], dtype=np.int64),
    )
    assert calls[0]["m_range"].dtype == np.int64


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.array([-2, -2, 1]), "duplicate"),
        (np.array([-2, 1.5, 2]), "finite integer"),
        (np.array([-2, 2, 1]), "strictly increasing"),
        (np.array([-5, -1, 2]), "available equilibrium harmonic range"),
    ],
)
def test_solver2_explicit_m_validator_rejects_invalid_lists(values, message):
    with pytest.raises(ValueError, match=message):
        StabilitySolver._validate_explicit_m_values(values, available_mmax=4)


@pytest.mark.parametrize(
    "values",
    [
        np.array([], dtype=int),
        np.array([[-1, 1]], dtype=int),
        np.array([False, True]),
        np.array([-1.0, np.nan]),
        np.array([-1.0, np.inf]),
        np.array([-1.0 + 0.0j, 1.0 + 0.0j]),
        np.array([-1, object()], dtype=object),
        np.array(["-1", "1"]),
        np.array([float(2**63)]),
        np.array([float(np.iinfo(np.int64).max) * 2.0]),
        np.array([np.iinfo(np.int64).min], dtype=np.int64),
        np.array([np.iinfo(np.uint64).max], dtype=np.uint64),
    ],
)
def test_solver2_explicit_m_validator_rejects_nonnumeric_and_overflow_inputs(values):
    with pytest.raises(ValueError):
        StabilitySolver._validate_explicit_m_values(values, available_mmax=4)
