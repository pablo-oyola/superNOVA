"""Focused tests for the solver4 compatible mixed projection."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import coo_array, csr_matrix, isspmatrix_csr

import supernova
import supernova.core.solver4 as solver4_module
from supernova.core.solver5 import superSolver5
from supernova.core.solver4 import (
    MixedFieldLayout,
    _build_mixed_radial_data,
    _project_local_tensor_mixed,
    superSolver4,
)
from supernova.radial import BSplineFE


def _high_space():
    return BSplineFE(
        knots=np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]),
        order=3,
        domain=(0.0, 1.0),
        quad_order=5,
    )


def test_mixed_projection_applies_radial_chain_only_to_differential_rows():
    beta = 0.25
    local = np.arange(1.0, 33.0, dtype=np.complex128).reshape(2, 4, 4)
    tensor = coo_array(local)
    phi_high_q = np.eye(2, dtype=np.float64)
    phi_low_q = np.ones((2, 1), dtype=np.float64)
    weights_q = np.array([0.4, 0.6], dtype=np.float64)
    derivative = csr_matrix((1, 2), dtype=np.float64)

    baseline = _project_local_tensor_mixed(
        SimpleNamespace(m_range=np.array([2]), dpsi_hat_dx=1.0),
        tensor,
        phi_high_q,
        phi_low_q,
        weights_q,
        derivative,
        include_derivative=False,
    ).toarray()
    corrected = _project_local_tensor_mixed(
        SimpleNamespace(m_range=np.array([2]), dpsi_hat_dx=beta),
        tensor,
        phi_high_q,
        phi_low_q,
        weights_q,
        derivative,
        include_derivative=False,
    ).toarray()

    np.testing.assert_allclose(corrected[:2], beta * baseline[:2])
    np.testing.assert_allclose(corrected[2:], baseline[2:])


def _dense_oracle(M, mixed, n_modes, include_derivative):
    n_high = mixed.pair.high.get_ndofs()
    n_low = mixed.pair.low.get_ndofs()
    layout = MixedFieldLayout(n_high, n_low, n_modes)
    out = np.zeros((layout.n_rows, layout.n_trial), dtype=np.complex128)

    for q in range(M.shape[0]):
        for local_row in range(4 * n_modes):
            row_field, row_mode = divmod(local_row, n_modes)
            for local_col in range(4 * n_modes):
                value = M[q, local_row, local_col]
                if value == 0.0:
                    continue
                col_field, col_mode = divmod(local_col, n_modes)
                trial_basis = (
                    mixed.phi_high_q if col_field < 2 else mixed.phi_low_q
                )
                for radial_row in range(n_low):
                    for radial_col in range(trial_basis.shape[1]):
                        out[
                            layout.row_index(row_field, radial_row, row_mode),
                            layout.trial_index(col_field, radial_col, col_mode),
                        ] -= (
                            mixed.weights_q[q]
                            * mixed.phi_low_q[q, radial_row]
                            * value
                            * trial_basis[q, radial_col]
                        )

    if include_derivative:
        G = mixed.G_low_high.toarray()
        for field in (0, 1):
            for mode in range(n_modes):
                for radial_row in range(n_low):
                    for radial_col in range(n_high):
                        out[
                            layout.row_index(field, radial_row, mode),
                            layout.trial_index(field, radial_col, mode),
                        ] += G[radial_row, radial_col]
    return out


def test_mixed_layout_is_field_major_and_harmonic_fast():
    layout = MixedFieldLayout(n_high=6, n_low=5, n_modes=3)

    assert layout.n_rows == 60
    assert layout.n_trial == 66
    assert layout.trial_offsets == (0, 18, 36, 51)
    assert layout.row_index(2, 4, 1) == 2 * 15 + 4 * 3 + 1
    assert layout.trial_index(0, 5, 2) == 17
    assert layout.trial_index(1, 0, 1) == 19
    assert layout.trial_index(2, 4, 2) == 50
    assert layout.trial_index(3, 4, 2) == 65


def test_mixed_layout_rejects_incompatible_dimensions():
    with pytest.raises(ValueError, match="n_low=n_high-1"):
        MixedFieldLayout(n_high=6, n_low=4, n_modes=2)


def test_mixed_radial_data_builds_exact_strong_derivative():
    mixed = _build_mixed_radial_data(_high_space())
    direct = mixed.phi_low_q.T @ (
        mixed.weights_q[:, None]
        * mixed.pair.high.eval_basis_derivative(mixed.points_q).T
    )

    assert mixed.pair.high.get_ndofs() == 4
    assert mixed.pair.low.get_ndofs() == 3
    assert isspmatrix_csr(mixed.mass_low)
    assert isspmatrix_csr(mixed.G_low_high)
    np.testing.assert_allclose(mixed.G_low_high.toarray(), direct, atol=1e-13)


def test_mixed_radial_data_accepts_one_external_master_quadrature():
    fine = BSplineFE(
        knots=np.array(
            [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
        ),
        order=3,
        domain=(0.0, 1.0),
        quad_order=5,
    )
    points, weights = fine.get_quadrature_points()

    mixed = _build_mixed_radial_data(_high_space(), points, weights)

    np.testing.assert_array_equal(mixed.points_q, points)
    np.testing.assert_array_equal(mixed.weights_q, weights)
    assert mixed.pair.low.build_matrices is False
    assert mixed.pair.low.mass_matrix is None
    direct = mixed.phi_low_q.T @ (
        weights[:, None] * mixed.pair.high.eval_basis_derivative(points).T
    )
    np.testing.assert_allclose(mixed.G_low_high.toarray(), direct, atol=1e-13)


@pytest.mark.parametrize("include_derivative", [False, True])
def test_mixed_projection_matches_direct_dense_quadrature(include_derivative):
    mixed = _build_mixed_radial_data(_high_space())
    n_modes = 2
    n_state = 4 * n_modes
    rng = np.random.default_rng(314159)
    dense_M = rng.normal(size=(mixed.points_q.size, n_state, n_state))
    dense_M = dense_M + 1.0j * rng.normal(size=dense_M.shape)
    dense_M[np.abs(dense_M) < 0.7] = 0.0
    sb = SimpleNamespace(m_range=np.array([-1, 2]))

    projected = _project_local_tensor_mixed(
        sb,
        coo_array(dense_M),
        mixed.phi_high_q,
        mixed.phi_low_q,
        mixed.weights_q,
        mixed.G_low_high,
        include_derivative=include_derivative,
    )
    expected = _dense_oracle(dense_M, mixed, n_modes, include_derivative)

    assert isinstance(projected, csr_matrix)
    assert projected.shape == (24, 28)
    np.testing.assert_allclose(projected.toarray(), expected, rtol=1e-13, atol=1e-13)


def test_derivative_blocks_have_positive_sign_and_only_high_trial_fields():
    mixed = _build_mixed_radial_data(_high_space())
    n_modes = 2
    n_state = 4 * n_modes
    zero_M = coo_array((mixed.points_q.size, n_state, n_state))
    sb = SimpleNamespace(m_range=np.arange(n_modes))
    layout = MixedFieldLayout(4, 3, n_modes)

    projected = _project_local_tensor_mixed(
        sb,
        zero_M,
        mixed.phi_high_q,
        mixed.phi_low_q,
        mixed.weights_q,
        mixed.G_low_high,
        include_derivative=True,
    ).toarray()
    expected = _dense_oracle(
        np.zeros(zero_M.shape, dtype=np.complex128),
        mixed,
        n_modes,
        True,
    )

    np.testing.assert_allclose(projected, expected, atol=1e-14)
    assert np.count_nonzero(projected[2 * layout.n_low * n_modes :, :]) == 0
    assert np.count_nonzero(projected[:, layout.trial_offsets[2] :]) == 0


def test_single_local_entry_uses_negative_galerkin_sign_and_mixed_offsets():
    mixed = _build_mixed_radial_data(_high_space())
    n_modes = 2
    n_state = 4 * n_modes
    dense_M = np.zeros((mixed.points_q.size, n_state, n_state), dtype=np.complex128)
    dense_M[1, 2 * n_modes + 1, 0] = 2.0 - 3.0j
    sb = SimpleNamespace(m_range=np.arange(n_modes))

    projected = _project_local_tensor_mixed(
        sb,
        coo_array(dense_M),
        mixed.phi_high_q,
        mixed.phi_low_q,
        mixed.weights_q,
        mixed.G_low_high,
    ).toarray()
    expected = _dense_oracle(dense_M, mixed, n_modes, False)

    np.testing.assert_allclose(projected, expected, atol=1e-14)
    nonzero_rows, nonzero_cols = np.nonzero(projected)
    assert np.all(nonzero_rows >= 2 * mixed.pair.low.get_ndofs() * n_modes)
    assert np.all(nonzero_cols < mixed.pair.high.get_ndofs() * n_modes)


def test_projector_validates_local_state_and_derivative_shapes():
    mixed = _build_mixed_radial_data(_high_space())
    sb = SimpleNamespace(m_range=np.array([0, 1]))

    with pytest.raises(ValueError, match="state dimensions"):
        _project_local_tensor_mixed(
            sb,
            coo_array((mixed.points_q.size, 7, 7)),
            mixed.phi_high_q,
            mixed.phi_low_q,
            mixed.weights_q,
            mixed.G_low_high,
        )

    with pytest.raises(ValueError, match="G_low_high"):
        _project_local_tensor_mixed(
            sb,
            coo_array((mixed.points_q.size, 8, 8)),
            mixed.phi_high_q,
            mixed.phi_low_q,
            mixed.weights_q,
            np.zeros((2, 2)),
        )


def _synthetic_solver(monkeypatch, seed=2718):
    solver = object.__new__(superSolver4)
    solver.radialfe = _high_space()
    mixed = solver._get_mixed_radial_data()
    n_modes = 2
    n_state = 4 * n_modes
    sb = SimpleNamespace(
        m_range=np.array([-1, 2]),
        points_q=mixed.points_q.copy(),
        weights_q=mixed.weights_q.copy(),
    )
    solver.build_solver_blocks = lambda **kwargs: sb

    rng = np.random.default_rng(seed)
    local = []
    for _ in range(3):
        dense = rng.normal(size=(mixed.points_q.size, n_state, n_state))
        dense = dense + 1.0j * rng.normal(size=dense.shape)
        dense[np.abs(dense) < 1.0] = 0.0
        local.append(coo_array(dense))
    monkeypatch.setattr(
        solver4_module,
        "_build_local_M_split",
        lambda sb, regularizer=0.0: tuple(local),
    )
    return solver, sb, mixed, tuple(local)


def test_solver4_split_assembly_and_full_boundary_reduction(monkeypatch):
    solver, sb, mixed, local = _synthetic_solver(monkeypatch)
    layout = solver._mixed_layout(sb)

    raw = solver.assemble_split_matrices(
        ntor=4,
        apply_bcs=False,
    )
    axis = solver.assemble_split_matrices(
        ntor=4,
        apply_bcs=True,
        bc_kind="axis",
    )
    full = solver.assemble_split_matrices(
        ntor=4,
        apply_bcs=True,
        bc_kind="full",
    )

    assert all(isinstance(matrix, csr_matrix) for matrix in raw + axis + full)
    assert all(matrix.shape == (layout.n_rows, layout.n_trial) for matrix in raw)
    assert all(
        matrix.shape == (layout.n_rows, layout.n_trial - layout.n_modes)
        for matrix in axis
    )
    assert all(matrix.shape == (layout.n_rows, layout.n_rows) for matrix in full)

    bc = solver._build_nep_boundary_condition(sb, bc_kind="full")
    P = bc.get_sparse_prolongation()
    assert np.linalg.matrix_rank(bc.C) == 2 * layout.n_modes
    np.testing.assert_allclose(bc.C @ P.toarray(), 0.0, atol=1e-13)
    for raw_matrix, full_matrix in zip(raw, full):
        np.testing.assert_allclose(
            full_matrix.toarray(),
            (raw_matrix @ P).toarray(),
            rtol=1e-13,
            atol=1e-13,
        )

    s = 0.8 + 0.2j
    combined_local = local[0] + s * local[1] + (1.0 / s) * local[2]
    direct = _project_local_tensor_mixed(
        sb,
        combined_local,
        mixed.phi_high_q,
        mixed.phi_low_q,
        mixed.weights_q,
        mixed.G_low_high,
        include_derivative=True,
    )
    split = raw[0] + s * raw[1] + (1.0 / s) * raw[2]
    np.testing.assert_allclose(split.toarray(), direct.toarray(), atol=1e-12)


def test_solver4_result_expands_reduced_vector_and_names_all_fields(monkeypatch):
    solver, sb, _, _ = _synthetic_solver(monkeypatch, seed=99)
    solver._species_mass_amu = lambda: 2.0
    layout = solver._mixed_layout(sb)
    bc = solver._build_nep_boundary_condition(sb, bc_kind="full")
    rng = np.random.default_rng(7)
    reduced = rng.normal(size=(2, layout.n_rows))
    reduced = reduced + 1.0j * rng.normal(size=reduced.shape)
    result = SimpleNamespace(
        eigenvalues=np.array([0.8 + 0.1j, 1.2 - 0.05j]),
        eigenvectors=reduced,
        nconv=2,
        iterations=6,
        reason="CONVERGED_TOL",
    )
    nep = SimpleNamespace(
        last_residual_norms=np.array([1.0e-9, 2.0e-9]),
        last_absolute_residual_norms=np.array([2.0e-9, 4.0e-9]),
        spectral_shift=0.9,
    )
    region = {
        "type": "ellipse",
        "center": 1.0,
        "radius": 0.2,
        "vscale": 0.25,
    }

    dataset = solver._split_result_dataset(
        result=result,
        nep=nep,
        ntor=4,
        target_omega=1.0,
        target_s=1.0,
        region=region,
        mmax=5,
        max_dm=3,
        regularizer=0.0,
        apply_bcs=True,
        bc_kind="full",
        bc_tol=None,
        verbose=False,
        compute_residuals=True,
        which="TARGET_MAGNITUDE",
        nep_type="NLEIGS",
        description="synthetic solver4 result",
    )

    expected_full = np.vstack([bc.expand_sparse(vector) for vector in reduced])
    np.testing.assert_allclose(dataset["eigenvector"], expected_full)
    np.testing.assert_allclose(bc.C @ expected_full.T, 0.0, atol=1e-13)
    assert dataset["p1_coefficient"].shape == (2, layout.n_high, layout.n_modes)
    assert dataset["xi_psi_coefficient"].shape == (
        2,
        layout.n_high,
        layout.n_modes,
    )
    assert dataset["xi_s_coefficient"].shape == (2, layout.n_low, layout.n_modes)
    assert dataset["div_xi_coefficient"].shape == (
        2,
        layout.n_low,
        layout.n_modes,
    )
    np.testing.assert_allclose(
        dataset["xi_psi_coefficient"],
        expected_full[:, layout.trial_slice(1)].reshape(
            2,
            layout.n_high,
            layout.n_modes,
        ),
    )
    assert dataset.attrs["bc_expanded_eigenvectors"] is True
    assert dataset.attrs["slepc_eigenvector_space"] == "reduced"
    assert dataset.attrs["radial_discretization"] == "compatible_mixed_bspline"
    assert dataset.attrs["full_dofs"] == layout.n_trial


@pytest.mark.parametrize(
    "bc_options",
    [
        {"apply_bcs": False, "bc_kind": "full"},
        {"apply_bcs": True, "bc_kind": "axis"},
        {"apply_bcs": True, "bc_kind": "edge"},
        {"apply_bcs": True, "bc_kind": "none"},
    ],
)
def test_solver4_rejects_rectangular_inputs_before_petsc(bc_options):
    """_require_full_bc_nep now guards the surviving PEP entry point."""
    solver = object.__new__(superSolver5)

    with pytest.raises(ValueError, match="require apply_bcs=True"):
        solver.build_pep_solver(
            ntor=2,
            target_omega=1.0,
            region={"center": 1.0, "radius": 0.1},
            **bc_options,
        )


def test_supernova_exports_solver4_without_replacing_solver3():
    assert supernova.superSolver4 is superSolver4
    assert supernova.superSolver3 is not supernova.superSolver4


