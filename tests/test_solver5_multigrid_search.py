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
    _deduplicate_window_dataset,
    _is_converged_reason,
    _nested_bspline_levels,
    _window_diagnostics_dataset,
    superSolver5Multigrid,
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


def test_adaptive_search_retargets_once_per_child_and_reports_saturation(
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
        max_depth=1,
        window_overlap=0.10,
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
    assert [record.depth for record in records] == [0, 1, 1]
    assert [record.leaf for record in records] == [False, True, True]
    assert [record.saturated for record in records] == [True, False, True]
    assert [record.complete for record in records] == [False, True, False]
    assert [record.effective_nev for record in records] == [2, 2, 2]
    assert complete is False
    assert dataset.attrs["search_complete"] is False
    assert dataset.attrs["saturated_leaf_count"] == 1
    assert dataset.attrs["ancestor_fallback_window_count"] == 1
    assert dataset.sizes["eigenpair"] == 5

    diagnostics = _window_diagnostics_dataset(records, nlevels=1)
    leaf = diagnostics["valid"] & diagnostics["leaf"]
    assert int(diagnostics["saturated"].where(leaf, False).sum()) == 1
    assert int(diagnostics["complete"].where(leaf, False).sum()) == 1
    np.testing.assert_array_equal(
        diagnostics["effective_nev"].values[0, diagnostics["valid"].values[0]],
        np.full(3, 2),
    )


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
        lambda solutions, weights, thresholds: xr.Dataset(
            coords={"track": np.array([], dtype=np.int64)}
        ),
    )
    monkeypatch.setattr(
        multigrid_module,
        "_classify_tracks",
        lambda tracks, continuum, weights, complete, thresholds: xr.Dataset(
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
