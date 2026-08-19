"""Deterministic PETSc-free tests for solver5 multigrid search mechanics."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import supernova.core.solver5_multigrid as multigrid_module
from supernova.core._petsc_helpers import NEPResult
from supernova.core.solver4 import _build_mixed_radial_data
from supernova.core.solver5_multigrid import (
    MultigridSearchOptions,
    ModeClassificationThresholds,
    _WindowRecord,
    _core_ownership_mask,
    _deduplicate_window_dataset,
    _fixed_frequency_windows,
    _is_converged_reason,
    _nested_bspline_levels,
    _window_diagnostics_dataset,
    superSolver5Multigrid,
)
from supernova.core.solver5_multigrid_evidence import (
    AdaptiveDiscoveryOptions,
    ReacquisitionOptions,
)
from supernova.radial import BSplineFE


def _base_spline(*, build_matrices=True) -> BSplineFE:
    order = 3
    breakpoints = np.array([0.0, 0.3, 0.7, 1.0])
    knots = np.concatenate(
        (
            np.repeat(breakpoints[0], order),
            breakpoints[1:-1],
            np.repeat(breakpoints[-1], order),
        )
    )
    return BSplineFE(
        knots=knots,
        order=order,
        domain=(0.0, 1.0),
        quad_order=4,
        build_matrices=build_matrices,
    )


def test_nested_knot_counts_and_adjacent_space_embedding():
    base = _base_spline()
    levels = _nested_bspline_levels(base, levels=3)

    interior_counts = [np.unique(level.knots)[1:-1].size for level in levels]
    assert interior_counts == [2, 5, 11]
    assert [level.build_matrices for level in levels] == [False, False, True]
    assert all(level is not base for level in levels)
    assert base.mass_matrix is not None
    assert levels[0].mass_matrix is None

    sample_points = np.linspace(0.0, 1.0, 101)
    for coarse, fine in zip(levels[:-1], levels[1:]):
        assert set(np.unique(coarse.knots)).issubset(np.unique(fine.knots))
        coarse_values = coarse.eval_basis(sample_points).T
        fine_values = fine.eval_basis(sample_points).T
        embedding, *_ = np.linalg.lstsq(
            fine_values,
            coarse_values,
            rcond=None,
        )
        np.testing.assert_allclose(
            fine_values @ embedding,
            coarse_values,
            rtol=2.0e-12,
            atol=2.0e-12,
        )


def test_all_levels_evaluate_on_the_single_finest_quadrature():
    levels = _nested_bspline_levels(_base_spline(), levels=3)

    for coarse in levels[:-1]:
        with pytest.raises(RuntimeError, match="evaluation-only"):
            coarse.get_quadrature_points()

    master_points, master_weights = levels[-1].get_quadrature_points()
    mixed_levels = tuple(
        _build_mixed_radial_data(level, master_points, master_weights)
        for level in levels
    )

    for level, mixed in zip(levels, mixed_levels):
        np.testing.assert_array_equal(mixed.points_q, master_points)
        np.testing.assert_array_equal(mixed.weights_q, master_weights)
        assert mixed.pair.high is level
        assert mixed.pair.low.build_matrices is False
        assert mixed.pair.low.mass_matrix is None


def test_nested_levels_reject_noncanonical_endpoint_multiplicity():
    base = _base_spline(build_matrices=False)
    base.knots = np.insert(base.knots, 0, base.knots[0])

    with pytest.raises(ValueError, match="endpoint multiplicity exactly"):
        _nested_bspline_levels(base, levels=3)


def test_fixed_windows_cover_band_with_contiguous_unique_cores():
    windows = _fixed_frequency_windows((1.0, 3.0), 4, 0.10)

    np.testing.assert_allclose(windows[:, :2], [
        [1.0, 1.5],
        [1.5, 2.0],
        [2.0, 2.5],
        [2.5, 3.0],
    ])
    np.testing.assert_allclose(windows[:, 1][:-1], windows[:, 0][1:])
    assert windows[0, 0] == 1.0
    assert windows[-1, 1] == 3.0


def test_fixed_window_padding_is_clipped_to_outer_band():
    windows = _fixed_frequency_windows((1.0, 3.0), 4, 0.20)

    np.testing.assert_allclose(
        windows[:, 2:],
        [[1.0, 1.6], [1.4, 2.1], [1.9, 2.6], [2.4, 3.0]],
    )


def test_half_open_core_ownership_is_unique_and_includes_final_edge():
    windows = _fixed_frequency_windows((1.0, 3.0), 4, 0.10)
    omega = np.array([1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.complex128)
    ownership = np.stack(
        [
            _core_ownership_mask(
                omega,
                core_min,
                core_max,
                final_window=index == len(windows) - 1,
            )
            for index, (core_min, core_max, _, _) in enumerate(windows)
        ]
    )

    np.testing.assert_array_equal(ownership.sum(axis=0), np.ones(omega.size))
    np.testing.assert_array_equal(np.argmax(ownership, axis=0), [0, 1, 2, 3, 3])


def test_cross_window_deduplication_requires_spectral_and_xi_gates():
    weights = np.array([0.2, 0.5, 0.3])
    shape = np.array(
        [[1.0, 0.0], [2.0, 0.0], [1.0, 0.0]],
        dtype=np.complex128,
    )
    orthogonal_shape = np.array(
        [[0.0, 1.0], [0.0, -1.0], [0.0, 2.0]],
        dtype=np.complex128,
    )
    roots = 4.0 * np.array(
        [1.0 + 5.0e-7, 1.0, 1.0 + 3.0e-7, 1.0 + 2.0e-7, 1.0 + 2.0e-5],
        dtype=np.complex128,
    )
    tags = np.array(
        ["spectral_shape_worse", "best", "shape_gate", "same_window", "spectral_gate"]
    )
    fields = np.stack(
        (
            shape,
            np.exp(0.73j) * shape,
            orthogonal_shape,
            -1.0j * shape,
            shape,
        )
    )
    dataset = xr.Dataset(
        data_vars={
            "s": ("eigenpair", roots),
            "omega": ("eigenpair", np.sqrt(roots)),
            "xi_psi": (("eigenpair", "psi", "m_pol"), fields),
            "window_id": ("eigenpair", np.array([0, 1, 2, 1, 3])),
            "slepc_backward_error": (
                "eigenpair",
                np.array([1.0e-6, 1.0e-12, 1.0e-10, 1.0e-11, 1.0e-9]),
            ),
            "tag": ("eigenpair", tags),
        },
        coords={
            "eigenpair": np.arange(roots.size),
            "psi": np.array([0.2, 0.5, 0.8]),
            "m_pol": np.array([3, 4]),
        },
    )

    deduplicated = _deduplicate_window_dataset(
        dataset,
        weights,
        relative_s=1.0e-6,
        minimum_overlap=0.99,
    )

    assert set(deduplicated["tag"].values.tolist()) == {
        "best",
        "shape_gate",
        "same_window",
        "spectral_gate",
    }
    np.testing.assert_array_equal(
        deduplicated.coords["eigenpair"],
        np.arange(4),
    )


def test_cross_window_deduplication_prefers_core_owned_copy_before_error():
    roots = np.array([4.0 + 0.0j, 4.0 * (1.0 + 2.0e-7) + 0.0j])
    fields = np.ones((2, 2, 1), dtype=np.complex128)
    dataset = xr.Dataset(
        data_vars={
            "s": ("eigenpair", roots),
            "omega": ("eigenpair", np.sqrt(roots)),
            "xi_psi": (("eigenpair", "psi", "m_pol"), fields),
            "window_id": ("eigenpair", [0, 1]),
            "core_owned": ("eigenpair", [False, True]),
            "slepc_backward_error": ("eigenpair", [1.0e-14, 1.0e-8]),
            "tag": ("eigenpair", ["padded_better_error", "core_owned"]),
        },
        coords={"eigenpair": [0, 1], "psi": [0.25, 0.75], "m_pol": [4]},
    )

    deduplicated = _deduplicate_window_dataset(
        dataset,
        np.ones(2),
        relative_s=1.0e-6,
        minimum_overlap=0.99,
    )

    assert deduplicated["tag"].item() == "core_owned"


def test_cross_window_deduplication_uses_degenerate_subspace_overlap():
    psi = np.linspace(0.0, 1.0, 9)
    basis = np.stack((np.sin(np.pi * psi), np.sin(2.0 * np.pi * psi)))[:, :, None]
    rotation = np.asarray([[1.0, 1.0], [-1.0, 1.0]]) / np.sqrt(2.0)
    rotated = np.einsum("ij,jpk->ipk", rotation, basis)
    fields = np.concatenate((basis, rotated), axis=0)
    dataset = xr.Dataset(
        {
            "s": ("eigenpair", np.asarray([4.0, 4.0 + 1.0e-8, 4.0, 4.0 + 1.0e-8])),
            "omega": ("eigenpair", np.sqrt(np.asarray([4.0, 4.0 + 1.0e-8, 4.0, 4.0 + 1.0e-8]))),
            "xi_psi": (("eigenpair", "psi", "m_pol"), fields),
            "window_id": ("eigenpair", [0, 0, 1, 1]),
            "core_owned": ("eigenpair", [True, True, False, False]),
            "slepc_backward_error": ("eigenpair", [1.0e-12] * 4),
        },
        coords={"eigenpair": np.arange(4), "psi": psi, "m_pol": [4]},
    )
    deduplicated = _deduplicate_window_dataset(
        dataset,
        np.ones(psi.size),
        relative_s=1.0e-6,
        minimum_overlap=0.99,
    )
    assert deduplicated.sizes["eigenpair"] == 2
    np.testing.assert_array_equal(deduplicated["window_id"].values, [0, 0])


def test_fixed_search_retargets_once_per_window_and_reports_saturation(
    monkeypatch,
):
    events = []

    class FakePEP:
        def __init__(self, **kwargs):
            self.region = dict(kwargs["region"])
            self.solve_count = 0
            self.last_residual_norms = None
            self.last_absolute_residual_norms = None
            self.last_backward_errors = None
            self.last_slepc_error_estimates = None
            self.last_slepc_backward_errors = None
            # Simulate a prefixed PETSc option overriding the requested nev.
            self.effective_nev = 2
            events.append(("init", kwargs["target_s"], dict(self.region)))

        def retarget(self, target_s, region):
            self.region = dict(region)
            events.append(("retarget", target_s, dict(region)))

        def solve(self, *, allow_empty):
            assert allow_empty is True
            nconv = (2, 1, 2)[self.solve_count]
            self.solve_count += 1
            center = complex(self.region["center"])
            offsets = np.linspace(-0.01, 0.01, nconv)
            roots = center * (1.0 + offsets)
            self.last_residual_norms = np.full(nconv, 1.0e-10)
            self.last_absolute_residual_norms = np.full(nconv, 2.0e-10)
            self.last_backward_errors = np.full(nconv, 3.0e-10)
            self.last_slepc_error_estimates = np.full(nconv, 4.0e-10)
            self.last_slepc_backward_errors = np.full(nconv, 5.0e-10)
            events.append(("solve", nconv))
            return NEPResult(
                eigenvalues=roots,
                eigenvectors=np.ones((nconv, 1), dtype=np.complex128),
                nconv=nconv,
                reason="fake-converged",
                iterations=7,
            )

        def destroy(self):
            events.append(("destroy",))

    monkeypatch.setattr(multigrid_module, "QuadraticPEPSolver", FakePEP)
    orchestrator = object.__new__(superSolver5Multigrid)
    orchestrator.levels = 1
    orchestrator.master_weights = np.array([0.4, 0.6])

    def fake_pack_window_result(**kwargs):
        indices = np.asarray(kwargs["indices"], dtype=np.int64)
        roots = np.asarray(kwargs["result"].eigenvalues)[indices]
        count = indices.size
        fields = np.ones((count, 2, 1), dtype=np.complex128)
        return xr.Dataset(
            data_vars={
                "s": ("eigenpair", roots),
                "omega": ("eigenpair", np.sqrt(roots)),
                "xi_psi": (("eigenpair", "psi", "m_pol"), fields),
                "window_id": (
                    "eigenpair",
                    np.full(count, kwargs["window"], dtype=np.int64),
                ),
                "slepc_backward_error": ("eigenpair", np.full(count, 5.0e-10)),
            },
            coords={
                "eigenpair": np.arange(count),
                "psi": np.array([0.25, 0.75]),
                "m_pol": np.array([4]),
            },
        )

    orchestrator._pack_window_result = fake_pack_window_result
    search = MultigridSearchOptions(
        nev=8,
        window_count=3,
        window_padding=0.10,
    )

    dataset, records, complete = orchestrator._search_level(
        level=0,
        matrices=(object(), object(), object()),
        sb=object(),
        mixed=object(),
        omega_range=(1.0, 3.0),
        search=search,
        bc_tol=None,
        verbose=False,
    )

    assert [event[0] for event in events] == [
        "init",
        "solve",
        "retarget",
        "solve",
        "retarget",
        "solve",
        "destroy",
    ]
    assert [record.depth for record in records] == [0, 0, 0]
    assert [record.parent for record in records] == [-1, -1, -1]
    assert [record.leaf for record in records] == [True, True, True]
    assert [record.saturated for record in records] == [True, False, True]
    assert [record.complete for record in records] == [False, False, False]
    assert [record.catalog_status for record in records] == [
        "capacity_limited",
        "partial",
        "capacity_limited",
    ]
    assert [record.effective_nev for record in records] == [2, 2, 2]
    assert complete is False
    assert dataset.attrs["search_complete"] is False
    assert dataset.attrs["saturated_leaf_count"] == 2
    assert dataset.attrs["ancestor_fallback_window_count"] == 0
    assert dataset.attrs["search_strategy"] == "fixed_padded_windows"
    assert dataset.sizes["eigenpair"] == 5

    diagnostics = _window_diagnostics_dataset(records, nlevels=1)
    leaf = diagnostics["valid"] & diagnostics["leaf"]
    assert int(diagnostics["saturated"].where(leaf, False).sum()) == 2
    assert int(diagnostics["complete"].where(leaf, False).sum()) == 0
    np.testing.assert_allclose(
        diagnostics["core_omega_min"].values[0],
        [1.0, 5.0 / 3.0, 7.0 / 3.0],
    )
    np.testing.assert_allclose(
        diagnostics["core_omega_max"].values[0],
        [5.0 / 3.0, 7.0 / 3.0, 3.0],
    )
    assert diagnostics["solve_omega_min"].values[0, 0] == 1.0
    assert diagnostics["solve_omega_max"].values[0, -1] == 3.0
    np.testing.assert_array_equal(
        diagnostics["effective_nev"].values[0, diagnostics["valid"].values[0]],
        np.full(3, 2),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window_count": 0}, "window_count"),
        ({"window_count": 2.5}, "window_count"),
        ({"window_count": True}, "window_count"),
        ({"window_count": np.inf}, "window_count"),
        ({"window_padding": -0.1}, "window_padding"),
        ({"window_padding": np.nan}, "window_padding"),
    ],
)
def test_fixed_window_options_reject_invalid_counts_and_padding(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MultigridSearchOptions(**kwargs)


def test_only_positive_or_explicit_converged_reasons_complete_a_window():
    assert _is_converged_reason(1)
    assert _is_converged_reason("CONVERGED_TOL")
    assert _is_converged_reason("fake-converged")
    assert not _is_converged_reason(0)
    assert not _is_converged_reason(-1)
    assert not _is_converged_reason("DIVERGED_ITS")
    assert not _is_converged_reason("maxit reached")
    assert not _is_converged_reason(None)


def test_solve_pep_constructs_local_blocks_once_for_all_levels(monkeypatch):
    local_split = (object(), object(), object())
    local_calls = []
    projection_calls = []
    search_calls = []
    solver_blocks = SimpleNamespace()

    class FakeSolver:
        unique_points = np.array([0.25, 0.75])

        @staticmethod
        def _species_mass_amu():
            return 2.0

        def _require_full_bc_nep(self, apply_bcs, bc_kind):
            assert apply_bcs is True
            assert bc_kind == "full"

        def build_solver_blocks(self, **kwargs):
            assert kwargs == {"ntor": 3, "mmax": 5, "max_dm": 2}
            return solver_blocks

        def _assemble_split_matrices_from_local(
            self,
            sb,
            mixed,
            supplied_local_split,
            **kwargs,
        ):
            assert sb is solver_blocks
            assert supplied_local_split is local_split
            projection_calls.append((mixed, kwargs))
            return (f"T0-{mixed}", f"T1-{mixed}", f"Tm1-{mixed}")

    orchestrator = object.__new__(superSolver5Multigrid)
    orchestrator.solver = FakeSolver()
    orchestrator.levels = 3
    orchestrator.mixed_levels = ("coarse", "medium", "fine")
    orchestrator.master_weights = np.array([0.4, 0.6])

    def fake_build_local(sb, *, regularizer):
        local_calls.append((sb, regularizer))
        return local_split

    def fake_search_level(**kwargs):
        search_calls.append(kwargs)
        level = kwargs["level"]
        return xr.Dataset(attrs={"level": level}), [], True

    monkeypatch.setattr(multigrid_module, "_build_local_M_split", fake_build_local)
    monkeypatch.setattr(
        multigrid_module,
        "_solve_master_continuum",
        lambda sb, psi, **kwargs: xr.Dataset(coords={"psi": psi}),
    )
    monkeypatch.setattr(
        multigrid_module,
        "_track_level_solutions",
        lambda solutions, weights, thresholds, **kwargs: xr.Dataset(
            coords={"track": np.array([], dtype=np.int64)}
        ),
    )
    monkeypatch.setattr(
        multigrid_module,
        "_classify_tracks",
        lambda tracks, continuum, weights, complete, thresholds, **kwargs: xr.Dataset(
            data_vars={"label": ("track", np.array([], dtype="U24"))},
            coords={"track": np.array([], dtype=np.int64)},
        ),
    )
    orchestrator._search_level = fake_search_level

    result = orchestrator.solve_pep(
        ntor=3,
        omega_range=(1.0, 2.0),
        mmax=5,
        max_dm=2,
        regularizer=7.5e-7,
    )

    assert local_calls == [(solver_blocks, 7.5e-7)]
    assert [call[0] for call in projection_calls] == [
        "coarse",
        "medium",
        "fine",
    ]
    assert [call["matrices"] for call in search_calls] == [
        ("T0-coarse", "T1-coarse", "Tm1-coarse"),
        ("T0-medium", "T1-medium", "Tm1-medium"),
        ("T0-fine", "T1-fine", "Tm1-fine"),
    ]
    np.testing.assert_array_equal(
        result.window_diagnostics["level_complete"],
        np.ones(3, dtype=bool),
    )


def test_adaptive_discovery_stops_after_stable_global_candidate_pass(monkeypatch):
    orchestrator = object.__new__(superSolver5Multigrid)
    radial = _base_spline(build_matrices=False)
    orchestrator.radial_levels = (radial,)
    orchestrator.master_points = np.asarray([0.25, 0.75])
    orchestrator.master_weights = np.asarray([0.5, 0.5])
    requested_nev = []

    def fake_search_level(**kwargs):
        requested_nev.append(kwargs["search"].nev)
        dataset = xr.Dataset(
            {
                "s": ("eigenpair", np.asarray([4.0 + 0.0j])),
                "omega": ("eigenpair", np.asarray([2.0 + 0.0j])),
                "xi_psi": (
                    ("eigenpair", "psi", "m_pol"),
                    np.ones((1, 2, 1), dtype=np.complex128),
                ),
                "window_id": ("eigenpair", np.asarray([0])),
                "slepc_backward_error": ("eigenpair", np.asarray([1.0e-12])),
            },
            coords={"eigenpair": [0], "psi": [0.25, 0.75], "m_pol": [4]},
        )
        record = _WindowRecord(
            level=kwargs["level"],
            window=0,
            parent=-1,
            depth=0,
            core_omega_min=1.0,
            core_omega_max=3.0,
            solve_omega_min=1.0,
            solve_omega_max=3.0,
            target_s=4.0 + 0.0j,
            nconv=kwargs["search"].nev,
            nreturned=1,
            saturated=True,
            leaf=True,
            complete=False,
            reason="fake-converged",
            iterations=2,
            effective_nev=kwargs["search"].nev,
            catalog_status="capacity_limited",
        )
        return dataset, [record], False

    orchestrator._search_level = fake_search_level
    dataset, records, complete = orchestrator._adaptive_discovery_level(
        level=0,
        matrices=(object(), object(), object()),
        sb=object(),
        mixed=object(),
        omega_range=(1.0, 3.0),
        search=MultigridSearchOptions(window_count=1),
        discovery=AdaptiveDiscoveryOptions(
            nev_passes=(4, 8, 16),
            target_offsets=(0.0,),
            stable_passes=1,
        ),
        bc_tol=None,
        verbose=False,
    )

    assert requested_nev == [4, 8]
    assert complete is False
    assert dataset.attrs["physical_candidate_catalog_stable"] is True
    assert dataset.attrs["catalog_status"] == "empirically_stable"
    assert [record.leaf for record in records] == [False, True]
    assert records[-1].catalog_status == "empirically_stable"


def test_missing_root_reacquisition_uses_prolonged_initial_space(monkeypatch):
    captured = {"initial": None, "destroyed": False}

    class FakePEP:
        effective_nev = 8

        def __init__(self, **kwargs):
            self.region = kwargs["region"]

        def set_initial_space(self, vectors):
            captured["initial"] = np.asarray(vectors).copy()

        def solve(self, *, allow_empty):
            assert allow_empty is True
            return NEPResult(
                eigenvalues=np.asarray([1.0 + 0.0j]),
                eigenvectors=np.asarray([[1.0, 2.0]], dtype=np.complex128),
                nconv=1,
                reason="fake-converged",
                iterations=3,
            )

        def destroy(self):
            captured["destroyed"] = True

    class FakeBoundaryCondition:
        sparse_free = np.asarray([0, 1])

        @staticmethod
        def get_sparse_prolongation(*, tol):
            assert tol is None
            return np.eye(2)

    class FakeSolver:
        @staticmethod
        def _build_nep_boundary_condition(sb, *, bc_kind, mixed):
            assert bc_kind == "full"
            return FakeBoundaryCondition()

    monkeypatch.setattr(multigrid_module, "QuadraticPEPSolver", FakePEP)
    monkeypatch.setattr(
        multigrid_module,
        "mixed_space_prolongation",
        lambda coarse, fine, n_modes: np.eye(2),
    )

    orchestrator = object.__new__(superSolver5Multigrid)
    orchestrator.levels = 2
    orchestrator.solver = FakeSolver()
    orchestrator.mixed_levels = ("coarse", "fine")
    orchestrator.radial_levels = ("coarse", "fine")
    orchestrator.master_weights = np.asarray([0.5, 0.5])

    tracks = xr.Dataset(
        {
            "present": (("track", "level"), [[True, False]]),
            "width_over_h": (("track", "level"), [[5.0, np.nan]]),
            "coarse_space_retention": (("track", "level"), [[0.8, np.nan]]),
            "mode_index": (("track", "level"), [[0, -1]]),
            "omega": (("track", "level"), [[1.0 + 0.0j, np.nan + 0.0j]]),
            "xi_psi": (
                ("track", "level", "psi", "m_pol"),
                np.ones((1, 2, 2, 1), dtype=np.complex128),
            ),
            "reacquisition_status": (
                ("track", "level"),
                np.asarray([["not_attempted", "not_attempted"]], dtype="U48"),
            ),
        },
        coords={"track": [0], "level": [0, 1], "psi": [0.25, 0.75], "m_pol": [4]},
    )
    coarse = xr.Dataset(
        {"eigenvector": (("eigenpair", "dof"), [[1.0 + 0.0j, 2.0 + 0.0j]])},
        coords={"eigenpair": [0], "dof": [0, 1]},
    )
    fine = xr.Dataset(
        {
            "s": ("eigenpair", np.empty(0, dtype=np.complex128)),
            "omega": ("eigenpair", np.empty(0, dtype=np.complex128)),
            "eigenvector": (
                ("eigenpair", "dof"),
                np.empty((0, 2), dtype=np.complex128),
            ),
            "xi_psi": (
                ("eigenpair", "psi", "m_pol"),
                np.empty((0, 2, 1), dtype=np.complex128),
            ),
            "window_id": ("eigenpair", np.empty(0, dtype=np.int64)),
            "slepc_backward_error": ("eigenpair", np.empty(0)),
        },
        coords={"eigenpair": [], "dof": [0, 1], "psi": [0.25, 0.75], "m_pol": [4]},
    )

    def fake_pack_window_result(**kwargs):
        return xr.Dataset(
            {
                "s": ("eigenpair", np.asarray([1.0 + 0.0j])),
                "omega": ("eigenpair", np.asarray([1.0 + 0.0j])),
                "eigenvector": (
                    ("eigenpair", "dof"),
                    np.asarray([[1.0 + 0.0j, 2.0 + 0.0j]]),
                ),
                "xi_psi": (
                    ("eigenpair", "psi", "m_pol"),
                    np.ones((1, 2, 1), dtype=np.complex128),
                ),
                "window_id": ("eigenpair", np.asarray([kwargs["window"]])),
                "slepc_backward_error": ("eigenpair", np.asarray([1.0e-12])),
            },
            coords={"eigenpair": [0], "dof": [0, 1], "psi": [0.25, 0.75], "m_pol": [4]},
        )

    def fake_retrack(solutions, weights, thresholds, **kwargs):
        assert solutions[1].sizes["eigenpair"] == 1
        updated = tracks.copy(deep=True)
        updated["present"].values[0, 1] = True
        updated["mode_index"].values[0, 1] = 0
        updated["omega"].values[0, 1] = 1.0 + 0.0j
        updated["reacquisition_status"].values[0, 1] = "reacquired"
        return updated

    orchestrator._pack_window_result = fake_pack_window_result
    monkeypatch.setattr(multigrid_module, "_track_level_solutions", fake_retrack)
    updated, records = orchestrator._reacquire_missing_roots(
        tracks=tracks,
        level_solutions=[coarse, fine],
        projected=((object(), object(), object()), (object(), object(), object())),
        sb=object(),
        search=MultigridSearchOptions(nev=4, window_count=1),
        thresholds=ModeClassificationThresholds(),
        options=ReacquisitionOptions(nev_multiplier=2),
        bc_tol=None,
        verbose=False,
    )

    np.testing.assert_allclose(captured["initial"], [1.0, 2.0])
    assert captured["destroyed"] is True
    assert updated["reacquisition_status"].item(1) == "reacquired"
    assert len(records) == 1
    assert "reacquisition=reacquired" in records[0].reason
