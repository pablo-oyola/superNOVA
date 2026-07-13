r"""
Standalone structural diagnostic for solver4 compatible boundary conditions.

Run from the repository root with

    conda run -n nova env NUMBA_DISABLE_JIT=1 python no_sync/test_boundary4.py

The module is guarded so importing it does not execute the diagnostic.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "supernova-matplotlib"))

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from supernova.core._boundary4 import idealSeparatrixBC, regularityAxisBC
from supernova.radial import BSplineFE, CompatibleBSplinePair


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_diagnostic() -> dict[str, object]:
    """Run compatible-space and mixed-boundary structural checks."""
    high = BSplineFE(
        knots=np.linspace(0.0, 1.0, 8),
        order=4,
        domain=(0.0, 1.0),
        quad_order=7,
    )
    pair = CompatibleBSplinePair.from_high(high)
    low = pair.low

    points_q, weights_q = high.get_quadrature_points()
    low_points_q, low_weights_q = low.get_quadrature_points()
    np.testing.assert_allclose(low_points_q, points_q, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(low_weights_q, weights_q, rtol=0.0, atol=0.0)

    phi_high_q = high.eval_basis(points_q).T
    phi_low_q = low.eval_basis(points_q).T
    dphi_high_q = high.eval_basis_derivative(points_q).T
    derivative = pair.derivative.toarray()

    compatible_dphi = phi_low_q @ derivative
    np.testing.assert_allclose(
        compatible_dphi,
        dphi_high_q,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    mass_low = phi_low_q.T @ (weights_q[:, None] * phi_low_q)
    G_low_high = mass_low @ derivative
    G_direct = phi_low_q.T @ (weights_q[:, None] * dphi_high_q)
    np.testing.assert_allclose(
        G_low_high,
        G_direct,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    m_numbers = np.arange(-3, 4)
    n_modes = m_numbers.size
    n_high = high.get_ndofs()
    n_low = low.get_ndofs()
    n_trial = 2 * n_modes * (n_high + n_low)
    n_rows = 4 * n_modes * n_low

    _require(n_low == n_high - 1, "Compatible pair does not satisfy n_low=n_high-1.")
    _require(
        n_trial - n_rows == 2 * n_modes,
        "Mixed trial/test dimension gap is not two constraints per harmonic.",
    )

    bc_axis = regularityAxisBC(high, low, m_numbers)
    bc_edge = idealSeparatrixBC(high, low, n_modes)
    bc_full = bc_axis * bc_edge

    axis_rank = np.linalg.matrix_rank(bc_axis.C)
    edge_rank = np.linalg.matrix_rank(bc_edge.C)
    full_rank = np.linalg.matrix_rank(bc_full.C)
    _require(axis_rank == n_modes, f"Axis BC rank is {axis_rank}, expected {n_modes}.")
    _require(edge_rank == n_modes, f"Edge BC rank is {edge_rank}, expected {n_modes}.")
    _require(
        full_rank == 2 * n_modes,
        f"Combined BC rank is {full_rank}, expected {2 * n_modes}.",
    )

    P = bc_full.get_sparse_prolongation()
    _require(P.shape == (n_trial, n_rows), f"Unexpected prolongation shape {P.shape}.")
    null_residual = np.asarray(bc_full.C @ P.toarray())
    np.testing.assert_allclose(null_residual, 0.0, rtol=0.0, atol=1.0e-13)

    # A deterministic sparse rectangular matrix exercises the same one-sided
    # shape transition as the solver4 projected operators.
    row = np.arange(n_rows, dtype=np.int64)
    col = (3 * row + 1) % n_trial
    data = 1.0 + 0.01 * row
    T_pre = (
        eye(n_rows, n_trial, dtype=np.complex128, format="csr")
        + coo_matrix((data, (row, col)), shape=(n_rows, n_trial))
    ).tocsr()
    T_axis = bc_axis.reduce_sparse_trial_matrix(T_pre)
    T_edge = bc_edge.reduce_sparse_trial_matrix(T_pre)
    T_post = bc_full.reduce_sparse_trial_matrix(T_pre)

    _require(T_pre.shape == (n_rows, n_trial), f"Unexpected pre-BC shape {T_pre.shape}.")
    _require(
        T_axis.shape == (n_rows, n_rows + n_modes),
        f"Unexpected axis-only shape {T_axis.shape}.",
    )
    _require(
        T_edge.shape == (n_rows, n_rows + n_modes),
        f"Unexpected edge-only shape {T_edge.shape}.",
    )
    _require(T_post.shape == (n_rows, n_rows), f"Unexpected full-BC shape {T_post.shape}.")
    _require(isinstance(T_post, csr_matrix), "One-sided reduction did not return CSR.")

    # Put distinct complex sentinels in the high-space xi^psi endpoint
    # coefficients, then compare Cx against direct spline endpoint traces.
    xipsi_offset = n_high * n_modes
    x_bad = np.zeros(n_trial, dtype=np.complex128)
    xipsi_bad = x_bad[
        xipsi_offset : xipsi_offset + n_high * n_modes
    ].reshape(n_high, n_modes)
    xipsi_bad[0] = 10.0 + np.arange(n_modes) + 1.0j * np.arange(n_modes)
    xipsi_bad[-1] = 20.0 + np.arange(n_modes) - 1.0j * np.arange(n_modes)

    phi_axis = high.eval_basis(np.array([high.get_mesh()[0]])).T
    phi_edge = high.eval_basis(np.array([high.get_mesh()[-1]])).T
    direct_axis = (phi_axis @ xipsi_bad).reshape(-1)
    direct_edge = (phi_edge @ xipsi_bad).reshape(-1)
    np.testing.assert_allclose(bc_axis.C @ x_bad, direct_axis, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(bc_edge.C @ x_bad, direct_edge, rtol=0.0, atol=1.0e-13)

    x_constrained = bc_full.expand_sparse(bc_full.project_sparse(x_bad))
    xipsi_constrained = x_constrained[
        xipsi_offset : xipsi_offset + n_high * n_modes
    ].reshape(n_high, n_modes)
    constrained_axis = (phi_axis @ xipsi_constrained).reshape(-1)
    constrained_edge = (phi_edge @ xipsi_constrained).reshape(-1)
    np.testing.assert_allclose(bc_full.C @ x_constrained, 0.0, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(constrained_axis, 0.0, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(constrained_edge, 0.0, rtol=0.0, atol=1.0e-13)

    summary = {
        "n_high": n_high,
        "n_low": n_low,
        "n_modes": n_modes,
        "pre_bc_shape": T_pre.shape,
        "axis_bc_shape": T_axis.shape,
        "edge_bc_shape": T_edge.shape,
        "full_bc_shape": T_post.shape,
        "bc_rank": full_rank,
        "max_derivative_identity_error": float(
            np.max(np.abs(compatible_dphi - dphi_high_q))
        ),
        "max_G_identity_error": float(np.max(np.abs(G_low_high - G_direct))),
        "max_nullspace_residual": float(np.max(np.abs(null_residual))),
        "max_endpoint_residual": float(
            max(np.max(np.abs(constrained_axis)), np.max(np.abs(constrained_edge)))
        ),
    }
    return summary


def main() -> None:
    summary = run_diagnostic()
    print("solver4 compatible-boundary diagnostic passed")
    for name, value in summary.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
