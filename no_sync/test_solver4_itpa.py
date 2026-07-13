r"""Small ITPA smoke diagnostic for the compatible mixed-spline solver4.

The default run includes a full-boundary NLEIGS solve and is intended for the
``nova-slepc`` environment.  Use ``--assembly-only`` for the structural checks
without PETSc/SLEPc.

Examples
--------
conda run -n nova-slepc env NUMBA_DISABLE_JIT=1 \
    python no_sync/test_solver4_itpa.py

conda run -n nova env NUMBA_DISABLE_JIT=1 \
    python no_sync/test_solver4_itpa.py --assembly-only
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "supernova-matplotlib"))

import numpy as np
import unyt
from scipy.sparse.csgraph import structural_rank

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import supernova as sn
from no_sync.solver4_analysis import reconstruct_solver4_field
from supernova.core.itpa_benchmark import ITPAEq


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def build_small_case(
    *,
    ntor: int = 6,
    mmax: int = 12,
    nfe: int = 8,
    ns: int = 64,
    ntheta: int = 64,
):
    """Build a modest analytical ITPA equilibrium and solver4 instance."""
    equ = ITPAEq(
        aminor=1.0 * unyt.m,
        Rmajor=10.0 * unyt.m,
        q0=1.71,
        qedge=1.87,
        B0=3.0 * unyt.T,
        ns=ns,
        ntheta=ntheta,
        m_max=mmax,
    )
    cont = sn.superContinuum(
        equ,
        gamma_s=5.0 / 3.0,
        species="H",
        dense=True,
    )
    high = sn.BSplineFE(
        knots=np.linspace(0.0, 1.0, nfe + 1),
        order=3,
        domain=(0.0, 1.0),
        quad_order=5,
    )
    solver = sn.superSolver4(equ, cont, radialfe=high)
    sb = solver.build_solver_blocks(ntor=ntor, mmax=mmax, max_dm=3)
    return solver, sb


def check_assembly(solver, sb, *, ntor: int, mmax: int) -> dict[str, object]:
    """Check compatible identities and raw/full-BC matrix dimensions."""
    mixed = solver._get_mixed_radial_data()
    pair = mixed.pair
    layout = solver._mixed_layout(sb)

    dphi_direct = pair.high.eval_basis_derivative(mixed.points_q).T
    dphi_compatible = mixed.phi_low_q @ pair.derivative.toarray()
    G_direct = mixed.phi_low_q.T @ (
        mixed.weights_q[:, None] * dphi_direct
    )
    np.testing.assert_allclose(
        dphi_compatible,
        dphi_direct,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        mixed.G_low_high.toarray(),
        G_direct,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    raw = solver.assemble_split_matrices(
        ntor=ntor,
        mmax=mmax,
        max_dm=3,
        regularizer=1.0e-5,
        apply_bcs=False,
    )
    reduced = solver.assemble_split_matrices(
        ntor=ntor,
        mmax=mmax,
        max_dm=3,
        regularizer=1.0e-5,
        apply_bcs=True,
        bc_kind="full",
    )
    _require(
        all(matrix.shape == (layout.n_rows, layout.n_trial) for matrix in raw),
        f"Unexpected raw split shapes {[matrix.shape for matrix in raw]}.",
    )
    _require(
        all(matrix.shape == (layout.n_rows, layout.n_rows) for matrix in reduced),
        f"Unexpected reduced split shapes {[matrix.shape for matrix in reduced]}.",
    )

    bc = solver._build_nep_boundary_condition(sb, bc_kind="full")
    P = bc.get_sparse_prolongation()
    rank = int(np.linalg.matrix_rank(bc.C))
    _require(rank == 2 * layout.n_modes, f"Unexpected full BC rank {rank}.")
    _require(P.shape == (layout.n_trial, layout.n_rows), f"Unexpected P shape {P.shape}.")
    np.testing.assert_allclose(bc.C @ P.toarray(), 0.0, atol=1.0e-13)
    for raw_matrix, reduced_matrix in zip(raw, reduced):
        np.testing.assert_allclose(
            reduced_matrix.toarray(),
            (raw_matrix @ P).toarray(),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    combined_pattern = (abs(reduced[0]) + abs(reduced[1]) + abs(reduced[2])).tocsr()
    pattern_rank = int(structural_rank(combined_pattern))

    return {
        "n_high": layout.n_high,
        "n_low": layout.n_low,
        "n_modes": layout.n_modes,
        "raw_shape": raw[0].shape,
        "full_bc_shape": reduced[0].shape,
        "bc_rank": rank,
        "combined_structural_rank": pattern_rank,
        "max_derivative_error": float(
            np.max(np.abs(dphi_direct - dphi_compatible))
        ),
        "max_G_error": float(
            np.max(np.abs(G_direct - mixed.G_low_high.toarray()))
        ),
    }


def _frequency_region(solver, frequency_window_khz=(60.0, 76.0)):
    frequency = unyt.unyt_array(
        frequency_window_khz,
        "kHz",
        registry=solver.equ.registry,
    )
    omega_min, omega_max = frequency.to("omegaA").v * (2.0 * np.pi)
    s_min = omega_min**2
    s_max = omega_max**2
    return 0.5 * (omega_min + omega_max), {
        "type": "ellipse",
        "center": 0.5 * (s_min + s_max),
        "radius": 0.5 * abs(s_max - s_min),
        "vscale": 0.25,
    }


def run_nleigs_smoke(solver, sb, *, ntor: int, mmax: int):
    """Run a small full-BC NLEIGS solve and check the packed mixed fields."""
    target_omega, region = _frequency_region(solver)
    result = solver.solve_nep_split(
        ntor=ntor,
        target_omega=target_omega,
        region=region,
        mmax=mmax,
        max_dm=3,
        regularizer=1.0e-5,
        apply_bcs=True,
        bc_kind="full",
        nev=2,
        max_eigenpairs=2,
        tol=1.0e-7,
        maxit=40,
        compute_residuals=True,
        verbose=True,
    )
    _require(result.sizes.get("eigenpair", 0) > 0, "NLEIGS returned no eigenpairs.")
    required = {
        "eigenvector",
        "p1_coefficient",
        "xi_psi_coefficient",
        "xi_s_coefficient",
        "div_xi_coefficient",
    }
    _require(required.issubset(result.data_vars), "Result is missing mixed field arrays.")

    mixed = solver._get_mixed_radial_data()
    xi_psi = np.asarray(result["xi_psi_coefficient"].values)
    phi_axis = mixed.pair.high.eval_basis(np.array([0.0])).T
    phi_edge = mixed.pair.high.eval_basis(np.array([1.0])).T
    axis_values = np.einsum("ij,ejm->eim", phi_axis, xi_psi)
    edge_values = np.einsum("ij,ejm->eim", phi_edge, xi_psi)
    coefficient_scale = max(1.0, float(np.max(np.abs(xi_psi))))
    endpoint_residual = max(
        float(np.max(np.abs(axis_values))),
        float(np.max(np.abs(edge_values))),
    )
    _require(
        endpoint_residual <= 1.0e-10 * coefficient_scale,
        f"xi^psi endpoint residual is {endpoint_residual:.3e}.",
    )

    psi, m_pol, reconstructed = reconstruct_solver4_field(
        result,
        solver,
        field="xi_psi",
        eigenpair_index=0,
        n_plot=128,
    )
    _require(np.iscomplexobj(reconstructed), "Field reconstruction lost complex dtype.")
    _require(
        reconstructed.shape == (psi.size, m_pol.size),
        f"Unexpected reconstructed field shape {reconstructed.shape}.",
    )
    _require(np.all(np.isfinite(reconstructed)), "Reconstructed field is non-finite.")

    print("solver4 full-BC ITPA NLEIGS smoke solve passed")
    print(f"  nconv: {result.attrs['nconv']}")
    print(f"  omega: {np.asarray(result['omega'].values)}")
    print(f"  residuals: {np.asarray(result.get('residual_norm', []))}")
    print(f"  max xi^psi endpoint residual: {endpoint_residual:.3e}")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assembly-only",
        action="store_true",
        help="skip the PETSc/SLEPc NLEIGS solve",
    )
    parser.add_argument("--ntor", type=int, default=6)
    parser.add_argument("--mmax", type=int, default=12)
    parser.add_argument("--nfe", type=int, default=8)
    args = parser.parse_args(argv)

    solver, sb = build_small_case(
        ntor=args.ntor,
        mmax=args.mmax,
        nfe=args.nfe,
    )
    summary = check_assembly(
        solver,
        sb,
        ntor=args.ntor,
        mmax=args.mmax,
    )
    print("solver4 ITPA mixed assembly diagnostic passed")
    for name, value in summary.items():
        print(f"  {name}: {value}")

    if not args.assembly_only:
        run_nleigs_smoke(
            solver,
            sb,
            ntor=args.ntor,
            mmax=args.mmax,
        )


if __name__ == "__main__":
    main()
