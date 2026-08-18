"""
Minimal split-rational NEP experiment for superSolver3 with ITPAEq.
"""

from pathlib import Path
import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse.linalg import norm as sparse_norm
from scipy.spatial.distance import squareform
import unyt

_working_directory = Path.cwd().resolve()
REPO_ROOT = next(
    (
        candidate
        for candidate in (_working_directory, *_working_directory.parents)
        if (candidate / "pyproject.toml").is_file()
        and (candidate / "supernova" / "__init__.py").is_file()
    ),
    None,
)
if REPO_ROOT is None:
    raise RuntimeError(
        "Run this script from the superNOVA_2 repository or one of its child folders."
    )
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import supernova as sn
from supernova.core.continuum import continuum_crossings, get_knots_clustered_at
from supernova.core.desc_helper import DESCSolver
from supernova.core.itpa_benchmark import ITPAEq
from supernova.core._solver_kernel2 import build_M_at, get_T_sparse
from supernova.core.solver3 import (
    _build_local_M_split,
    _project_local_tensor,
    _slice_to_quadrature,
)
from no_sync.solver3_plots import (
    plot_slow_sound_with_eigenmode_extensions,
    plot_xipsi_eigenpair,
)
from no_sync.solver3_analysis import xipsi_effective_kpsi

plt.ion()

WRITE_SOLVER_BLOCKS = os.environ.get("SUPERNOVA_ITPA_WRITE_BLOCKS", "0") == "1"
RUN_CISS = os.environ.get("SUPERNOVA_ITPA_RUN_CISS", "0") == "1"
BC_KIND = os.environ.get("SUPERNOVA_ITPA_BC_KIND", "full")
MAX_DM = int(os.environ.get("SUPERNOVA_ITPA_MAX_DM", "10"))
NEV = int(os.environ.get("SUPERNOVA_ITPA_NEV", "10"))
MAX_EIGENPAIRS = int(os.environ.get("SUPERNOVA_ITPA_MAX_EIGENPAIRS", str(NEV)))
NFE = int(os.environ.get("SUPERNOVA_ITPA_NFE", "200"))
REGULARIZER = float(os.environ.get("SUPERNOVA_ITPA_REGULARIZER", "1e-5"))
WD_ORIENTATION = os.environ.get("SUPERNOVA_ITPA_WD_ORIENTATION", "stored").lower()

itpa_eigenfunction_comparisons = [
    {"eigenpair_index": 1, "panel": "A", "codes": None},
]


def _psi_to_rho(equ, psi):
    return np.interp(np.asarray(psi, dtype=float), equ._psi, equ._rho)


def _psi_to_rhotor(equ, psi):
    q_profile = equ("q")
    q_data = q_profile.data.sel(mpol=0, ntor=0).real
    psi_q = np.asarray(q_data.x, dtype=float)
    q = np.asarray(q_data.values, dtype=float)
    if not (np.all(q > 0.0) or np.all(q < 0.0)):
        raise ValueError("q must be finite, non-zero, and keep one sign.")

    s_tor = np.zeros_like(psi_q)
    s_tor[1:] = np.cumsum(
        0.5 * (q[1:] + q[:-1]) * np.diff(psi_q)
    )
    s_tor /= s_tor[-1]
    return np.sqrt(np.interp(np.asarray(psi, dtype=float), psi_q, s_tor))


def _dense_matrix(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def apply_wd_orientation_override(solver, sb, orientation):
    """
    Optionally override the weak-derivative matrix for orientation experiments.

    The BSplineFE object stores ``weak_derivative_matrix[i, j] = int dphi_i phi_j``.
    Solver blocks normally use its transpose, ``int phi_i dphi_j``.
    """
    orientation = str(orientation).strip().lower()
    if orientation in {"", "stored", "solver", "current"}:
        print("WD orientation: stored solver block orientation (int phi_i dphi_j)")
        return "stored"

    raw_wd = solver.radialfe.weak_derivative_matrix
    if orientation in {"raw", "basis", "untransposed"}:
        sb.WD = raw_wd.copy()
        print("WD orientation override: raw radialfe orientation (int dphi_i phi_j)")
        return "raw"

    if orientation in {"transposed", "transpose"}:
        sb.WD = raw_wd.T.tocsr() if hasattr(raw_wd.T, "tocsr") else raw_wd.T.copy()
        print("WD orientation override: explicit transpose (int phi_i dphi_j)")
        return "transposed"

    if orientation in {"negative", "neg", "minus"}:
        sb.WD = -sb.WD
        print("WD orientation override: negative of stored solver block orientation")
        return "negative"

    raise ValueError(
        "SUPERNOVA_ITPA_WD_ORIENTATION must be one of "
        "'stored', 'raw', 'transposed', or 'negative'."
    )


def _solver3_xipsi_coeffs(result, solver, sb, eigenpair_index):
    """
    Return full-space xi^psi coefficients for one solver3 eigenpair.
    """
    n_radial = solver.radialfe.get_ndofs()
    n_modes = sb.m_range.size
    n_state = 4 * n_modes
    eigenvector = result["eigenvector"].isel(eigenpair=eigenpair_index).values
    if eigenvector.size != n_radial * n_state:
        raise ValueError(
            "Expected full-space solver3 eigenvector. "
            "Check result.attrs['eigenvector_space'] and result.attrs['full_dofs']."
        )
    state = eigenvector.reshape((n_radial, n_state))
    return eigenvector, state[:, n_modes : 2 * n_modes]


def xipsi_kinetic_energy(result, solver, sb, eigenpair_index=0):
    """
    Estimate the kinetic energy carried by xi^psi in the simplified limit.
    """
    _, xipsi_coeffs = _solver3_xipsi_coeffs(
        result, solver, sb, eigenpair_index
    )
    psi = np.asarray(sb.points_q)
    xipsi = np.asarray(sb.phi_q) @ xipsi_coeffs

    density = solver.equ.density.data.sel(mpol=0, ntor=0).real.interp(x=psi).values
    density *= solver.cont.mass.to("kg").v
    h = solver.equ("h").data.sel(mpol=0, ntor=0).real.interp(x=psi).values
    q = solver.equ.qprof.data.sel(mpol=0, ntor=0).real.interp(x=psi).values
    radius = _psi_to_rho(solver.equ, psi) * solver.equ.aminor

    integrand = (
        density
        * h
        * q**2
        / solver.equ.Baxis**2
        / radius**2
        * np.sum(np.abs(xipsi) ** 2, axis=1)
    )
    return float(np.sum(np.asarray(sb.weights_q) * integrand))


def xipsi_normalized_overlap(result, solver, sb, eigenpair_a, eigenpair_b):
    """
    Compute the normalized xi^psi overlap between two eigenpairs.
    """
    _, coeffs_a = _solver3_xipsi_coeffs(result, solver, sb, eigenpair_a)
    _, coeffs_b = _solver3_xipsi_coeffs(result, solver, sb, eigenpair_b)
    psi = np.asarray(sb.points_q)
    xipsi_a = np.asarray(sb.phi_q) @ coeffs_a
    xipsi_b = np.asarray(sb.phi_q) @ coeffs_b

    density = solver.equ.density.data.sel(mpol=0, ntor=0).real.interp(x=psi).values
    density *= solver.cont.mass.to("kg").v
    h = solver.equ("h").data.sel(mpol=0, ntor=0).real.interp(x=psi).values
    q = solver.equ.qprof.data.sel(mpol=0, ntor=0).real.interp(x=psi).values
    radius = _psi_to_rho(solver.equ, psi) * solver.equ.aminor
    weight = density * h * q**2 / solver.equ.Baxis**2 / radius**2
    radial_weight = np.asarray(sb.weights_q) * weight

    overlap_m = np.sum(
        radial_weight[:, None] * np.conj(xipsi_a) * xipsi_b,
        axis=0,
    )
    integral_a = np.sum(radial_weight[:, None] * np.abs(xipsi_a) ** 2)
    integral_b = np.sum(radial_weight[:, None] * np.abs(xipsi_b) ** 2)
    return float(np.abs(np.sum(overlap_m)) / np.sqrt(integral_a * integral_b))


def reduce_xipsi_eigenmodes(result, solver, sb, correlation_threshold=0.7):
    """
    Keep one representative xi^psi eigenmode from each correlated cluster.
    """
    if not 0.0 <= correlation_threshold <= 1.0:
        raise ValueError("correlation_threshold must be between 0 and 1")

    n_eigenpairs = result.sizes["eigenpair"]
    if n_eigenpairs < 2:
        return result

    correlation = np.eye(n_eigenpairs)
    for eigenpair_a in range(n_eigenpairs):
        for eigenpair_b in range(eigenpair_a):
            correlation[eigenpair_a, eigenpair_b] = xipsi_normalized_overlap(
                result, solver, sb, eigenpair_a, eigenpair_b
            )
            correlation[eigenpair_b, eigenpair_a] = correlation[eigenpair_a, eigenpair_b]

    distance = 1.0 - np.clip(correlation, 0.0, 1.0)
    clusters = fcluster(
        linkage(squareform(distance, checks=False), method="complete"),
        t=1.0 - correlation_threshold,
        criterion="distance",
    )

    representatives = []
    for cluster in np.unique(clusters):
        members = np.flatnonzero(clusters == cluster)
        cluster_correlation = correlation[np.ix_(members, members)]
        representatives.append(members[np.argmax(cluster_correlation.sum(axis=1))])

    return result.isel(eigenpair=np.sort(representatives))


def summarize_solver3_xipsi_eigenpairs(
    result,
    solver,
    sb,
    bc_kind=None,
    n_plot=512,
    edge_derivative_rho=0.9,
):
    """
    Compute residual, boundary, roughness, and localization diagnostics.
    """
    if result.sizes.get("eigenpair", 0) == 0:
        return []

    if bc_kind is None:
        bc_kind = result.attrs.get("bc_kind", "full")

    mesh = solver.radialfe.get_mesh()
    psi_plot = np.linspace(mesh[0], mesh[-1], n_plot)
    rho_plot = _psi_to_rho(solver.equ, psi_plot)
    phi_plot = solver.radialfe.eval_basis(psi_plot)
    phi_q = np.asarray(sb.phi_q)
    dphi_q = solver.radialfe.eval_basis_derivative(sb.points_q).T
    weights_q = np.asarray(sb.weights_q, dtype=float)
    rho_q = _psi_to_rho(solver.equ, sb.points_q)
    edge_derivative_mask = rho_q >= edge_derivative_rho
    eps = np.finfo(float).eps

    bc = None
    if result.attrs.get("apply_bcs", False) and str(bc_kind).lower() != "none":
        bc = solver._build_nep_boundary_condition(sb, bc_kind=bc_kind)

    rows = []
    for eigenpair_index in range(result.sizes["eigenpair"]):
        eigenvector, xipsi_coeffs = _solver3_xipsi_coeffs(
            result, solver, sb, eigenpair_index
        )
        xipsi_all = phi_plot.T @ xipsi_coeffs
        envelope = np.linalg.norm(xipsi_all, axis=1)
        mode_norms = np.linalg.norm(xipsi_all, axis=0)
        total_mode_norm = float(np.sum(mode_norms))
        dominant_index = int(np.argmax(mode_norms))
        dominant_m = int(sb.m_range[dominant_index])

        diff1 = np.diff(xipsi_all, axis=0)
        diff2 = np.diff(xipsi_all, n=2, axis=0)
        xipsi_norm = float(np.linalg.norm(xipsi_all))
        roughness = float(np.linalg.norm(diff2) / max(xipsi_norm, eps))
        total_variation = float(np.sum(np.abs(diff1)) / max(np.sum(np.abs(xipsi_all)), eps))

        peak = float(np.max(envelope))
        peak_index = int(np.argmax(envelope))
        edge_ratio = float(envelope[-1] / max(peak, eps))
        axis_ratio = float(envelope[0] / max(peak, eps))
        dominance = float(mode_norms[dominant_index] / max(total_mode_norm, eps))

        xipsi_q = phi_q @ xipsi_coeffs
        dxipsi_q = dphi_q @ xipsi_coeffs
        xipsi_q_norm2 = float(np.sum(weights_q[:, None] * np.abs(xipsi_q) ** 2))
        dxipsi_q_norm2 = float(np.sum(weights_q[:, None] * np.abs(dxipsi_q) ** 2))
        radial_derivative_ratio = float(
            np.sqrt(max(dxipsi_q_norm2, 0.0))
            / max(np.sqrt(max(xipsi_q_norm2, 0.0)), eps)
        )
        derivative_envelope = np.linalg.norm(dxipsi_q, axis=1)
        derivative_peak_rho = float(rho_q[int(np.argmax(derivative_envelope))])
        if dxipsi_q_norm2 > 0.0 and np.any(edge_derivative_mask):
            edge_derivative_norm2 = float(
                np.sum(
                    weights_q[edge_derivative_mask, None]
                    * np.abs(dxipsi_q[edge_derivative_mask]) ** 2
                )
            )
            edge_derivative_fraction = edge_derivative_norm2 / dxipsi_q_norm2
        else:
            edge_derivative_fraction = np.nan

        bc_residual = np.nan
        if bc is not None:
            bc_residual = float(
                np.linalg.norm(bc.C @ eigenvector) / max(np.linalg.norm(eigenvector), eps)
            )

        residual = np.nan
        if "residual_norm" in result:
            residual = float(result["residual_norm"].isel(eigenpair=eigenpair_index).values)

        rows.append(
            {
                "eigenpair": eigenpair_index,
                "s": result["s"].isel(eigenpair=eigenpair_index).values.item(),
                "omega": result["omega"].isel(eigenpair=eigenpair_index).values.item(),
                "residual": residual,
                "bc_residual": bc_residual,
                "dominant_m": dominant_m,
                "dominance": dominance,
                "roughness": roughness,
                "total_variation": total_variation,
                "peak_rho": float(rho_plot[peak_index]),
                "edge_ratio": edge_ratio,
                "axis_ratio": axis_ratio,
                "radial_derivative_ratio": radial_derivative_ratio,
                "derivative_peak_rho": derivative_peak_rho,
                "edge_derivative_fraction": edge_derivative_fraction,
            }
        )

    return rows


def print_solver3_xipsi_summary(rows):
    """
    Print a compact eigenpair quality table.
    """
    if not rows:
        print("No eigenpair diagnostics available.")
        return

    header = (
        "eig  dom_m  residual    bc_res      rough      TV         "
        "peak_rho  edge/peak  dxi/xi    dxi_peak  edge_dxi  dominance"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['eigenpair']:3d}  "
            f"{row['dominant_m']:5d}  "
            f"{row['residual']:.3e}  "
            f"{row['bc_residual']:.3e}  "
            f"{row['roughness']:.3e}  "
            f"{row['total_variation']:.3e}  "
            f"{row['peak_rho']:.3f}     "
            f"{row['edge_ratio']:.3e}   "
            f"{row['radial_derivative_ratio']:.3e}  "
            f"{row['derivative_peak_rho']:.3f}     "
            f"{row['edge_derivative_fraction']:.3e}  "
            f"{row['dominance']:.3f}"
        )


def plot_solver3_xipsi_summary(rows):
    """
    Plot residual/roughness/localization diagnostics for all eigenpairs.
    """
    if not rows:
        raise ValueError("No eigenpair diagnostics available.")

    eigenpair = np.array([row["eigenpair"] for row in rows])
    residual = np.array([row["residual"] for row in rows])
    roughness = np.array([row["roughness"] for row in rows])
    peak_rho = np.array([row["peak_rho"] for row in rows])
    dominant_m = np.array([row["dominant_m"] for row in rows])
    radial_derivative_ratio = np.array([row["radial_derivative_ratio"] for row in rows])
    edge_derivative_fraction = np.array([row["edge_derivative_fraction"] for row in rows])

    fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True)
    ax = ax.ravel()
    ax[0].semilogy(eigenpair, residual, "o-")
    ax[0].set_ylabel("reduced residual")
    ax[1].semilogy(eigenpair, roughness, "o-")
    ax[1].set_ylabel(r"$\xi^\psi$ roughness")
    ax[2].plot(eigenpair, peak_rho, "o-")
    ax[2].set_ylabel("peak r/a")
    ax[3].plot(eigenpair, dominant_m, "o-")
    ax[3].set_ylabel("dominant m")
    ax[4].semilogy(eigenpair, radial_derivative_ratio, "o-")
    ax[4].set_ylabel(r"$||\partial_\psi\xi^\psi||/||\xi^\psi||$")
    ax[5].plot(eigenpair, edge_derivative_fraction, "o-")
    ax[5].set_ylabel("edge derivative fraction")
    for axis in ax:
        axis.grid(True, alpha=0.3)
        axis.set_xlabel("eigenpair")
    fig.tight_layout()
    return fig, ax


def check_radial_derivative_matrix(solver, sb):
    """
    Verify radial quadrature and weak-derivative matrix orientations.
    """
    phi_q = np.asarray(sb.phi_q)
    dphi_q = solver.radialfe.eval_basis_derivative(sb.points_q).T
    weights_q = np.asarray(sb.weights_q, dtype=float)
    direct_mass = np.einsum("q,qi,qj->ij", weights_q, phi_q, phi_q, optimize=True)
    direct_wd = np.einsum("q,qi,qj->ij", weights_q, phi_q, dphi_q, optimize=True)
    direct_stiffness = np.einsum("q,qi,qj->ij", weights_q, dphi_q, dphi_q, optimize=True)

    stored_mass = _dense_matrix(solver.radialfe.mass_matrix)
    stored_wd = _dense_matrix(sb.WD)
    raw_wd = _dense_matrix(solver.radialfe.weak_derivative_matrix)
    stored_stiffness = _dense_matrix(solver.radialfe.stiffness_matrix)

    phi_left = solver.radialfe.eval_basis([solver.radialfe.get_mesh()[0]]).reshape(-1)
    phi_right = solver.radialfe.eval_basis([solver.radialfe.get_mesh()[-1]]).reshape(-1)
    boundary_trace = np.outer(phi_right, phi_right) - np.outer(phi_left, phi_left)

    out = {
        "points_shape": np.asarray(sb.points_q).shape,
        "weights_shape": weights_q.shape,
        "weights_sum": float(np.sum(weights_q)),
        "weights_min": float(np.min(weights_q)),
        "weights_max": float(np.max(weights_q)),
        "phi_q_shape": phi_q.shape,
        "dphi_q_shape": dphi_q.shape,
        "shape": stored_wd.shape,
        "mass_max_abs_error": float(np.max(np.abs(stored_mass - direct_mass))),
        "stiffness_max_abs_error": float(np.max(np.abs(stored_stiffness - direct_stiffness))),
        "wd_vs_int_phi_dphi_max_abs_error": float(np.max(np.abs(stored_wd - direct_wd))),
        "wd_vs_raw_int_dphi_phi_max_abs_error": float(np.max(np.abs(stored_wd - raw_wd))),
        "integration_by_parts_error": float(
            np.max(np.abs(stored_wd + stored_wd.T - boundary_trace))
        ),
        "raw_plus_stored_boundary_error": float(
            np.max(np.abs(raw_wd + stored_wd - boundary_trace))
        ),
    }
    print("radial quadrature and weak-derivative matrix check:")
    for key, value in out.items():
        print(f"  {key}: {value}")
    return out


def check_solver3_split_operator_consistency(
    solver,
    sb,
    omega,
    regularizer=0.0,
    bc_kind="full",
    bc_tol=None,
):
    """
    Compare direct T(omega) assembly with split T0+s*T1+Tminus1/s.
    """
    nq = np.asarray(sb.weights_q).shape[0]
    s_value = omega**2

    M_direct = build_M_at(sb=sb, omega=omega, regularizer=regularizer)
    M_direct_q = _slice_to_quadrature(M_direct, nq, "M_direct")
    T_direct = get_T_sparse(sb=sb, M=M_direct_q, omega=omega, regularizer=regularizer)

    M0, M1, Mminus1 = _build_local_M_split(sb, regularizer=regularizer)
    T0 = _project_local_tensor(sb, _slice_to_quadrature(M0, nq, "M0"), include_derivative=True)
    T1 = _project_local_tensor(sb, _slice_to_quadrature(M1, nq, "M1"), include_derivative=False)
    Tminus1 = _project_local_tensor(
        sb, _slice_to_quadrature(Mminus1, nq, "Mminus1"), include_derivative=False
    )
    T_split = (T0 + s_value * T1 + (1.0 / s_value) * Tminus1).tocsr()

    diff = (T_split - T_direct).tocsr()
    direct_norm = sparse_norm(T_direct)
    diff_norm = sparse_norm(diff)
    diff_max = 0.0 if diff.nnz == 0 else float(np.max(np.abs(diff.data)))

    out = {
        "direct_shape": T_direct.shape,
        "split_shape": T_split.shape,
        "direct_nnz": int(T_direct.nnz),
        "split_nnz": int(T_split.nnz),
        "diff_nnz": int(diff.nnz),
        "diff_norm": float(diff_norm),
        "relative_diff_norm": float(diff_norm / max(direct_norm, np.finfo(float).eps)),
        "diff_max_abs": diff_max,
    }

    kind = solver._normalize_bc_kind(bc_kind)
    if kind != "none":
        bc = solver._build_nep_boundary_condition(sb, bc_kind=kind)
        T_direct_reduced = bc.reduce_sparse_matrix(T_direct, tol=bc_tol)
        T_split_reduced = bc.reduce_sparse_matrix(T_split, tol=bc_tol)
        reduced_diff = (T_split_reduced - T_direct_reduced).tocsr()
        reduced_direct_norm = sparse_norm(T_direct_reduced)
        reduced_diff_norm = sparse_norm(reduced_diff)
        reduced_diff_max = (
            0.0 if reduced_diff.nnz == 0 else float(np.max(np.abs(reduced_diff.data)))
        )
        out.update(
            {
                "reduced_shape": T_direct_reduced.shape,
                "reduced_diff_nnz": int(reduced_diff.nnz),
                "reduced_diff_norm": float(reduced_diff_norm),
                "reduced_relative_diff_norm": float(
                    reduced_diff_norm / max(reduced_direct_norm, np.finfo(float).eps)
                ),
                "reduced_diff_max_abs": reduced_diff_max,
            }
        )

    print("split/direct operator consistency:")
    for key, value in out.items():
        print(f"  {key}: {value}")
    return out


def plot_itpa_code_eigenfunctions(
    result,
    solver,
    sb,
    csv_path,
    eigenpair_index=0,
    panel="A",
    codes=None,
    mpol=(9, 10, 11, 12),
    n_plot=512,
):
    """Compare one superNOVA eigenmode with the ITPA code eigenfunctions."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["panel"] == panel and int(row["m"]) in mpol:
                if codes is None or row["code"] in codes:
                    rows.append(row)

    if not rows:
        raise ValueError(f"No CSV eigenfunctions found for panel {panel!r}.")

    _, xipsi_coeffs = _solver3_xipsi_coeffs(
        result, solver, sb, eigenpair_index
    )
    mode_indices = []
    for m_value in mpol:
        matches = np.flatnonzero(sb.m_range == m_value)
        if matches.size == 0:
            raise ValueError(f"m={m_value} is outside the solver harmonic range.")
        mode_indices.append(int(matches[0]))

    mesh = solver.radialfe.get_mesh()
    psi_plot = np.linspace(mesh[0], mesh[-1], n_plot)
    rho_tor = _psi_to_rhotor(solver.equ, psi_plot)
    xipsi = solver.radialfe.eval_basis(psi_plot).T @ xipsi_coeffs
    psi_geometry = solver.equ.magspace.psi.values
    R_geometry = solver.equ.magspace["R"].values
    z_geometry = solver.equ.magspace["z"].values
    z_axis = np.mean(z_geometry[0])
    r_geometry = np.mean(
        np.sqrt(
            (R_geometry - solver.equ.Rmajor) ** 2
            + (z_geometry - z_axis) ** 2
        ),
        axis=1,
    )
    r_minor = np.interp(psi_plot, psi_geometry, r_geometry)
    xipsi_over_r = np.divide(
        xipsi,
        r_minor[:, None],
        out=np.zeros_like(xipsi),
        where=r_minor[:, None] != 0.0,
    )
    xipsi_over_r[0] = 0.0
    xipsi_over_r = np.abs(xipsi_over_r[:, mode_indices])
    xipsi_over_r /= np.max(xipsi_over_r)

    code_names = list(dict.fromkeys(row["code"] for row in rows))
    code_colors = dict(zip(code_names, plt.get_cmap("tab10").colors))
    line_styles = {"solid": "-", "dashed": "--", "dotted": ":"}
    fig, axes = plt.subplots(len(mpol), 1, sharex=True, figsize=(7, 8))
    axes = np.atleast_1d(axes)

    for mode_column, (ax, m_value) in enumerate(zip(axes, mpol)):
        for code in code_names:
            code_rows = [
                row for row in rows
                if row["code"] == code and int(row["m"]) == m_value
            ]
            if not code_rows:
                continue
            ax.plot(
                [float(row["sqrt_s"]) for row in code_rows],
                [float(row["normalized_amplitude"]) for row in code_rows],
                line_styles.get(code_rows[0]["line_style"], "-"),
                color=code_colors[code],
                label=code,
            )
        ax.plot(
            rho_tor,
            xipsi_over_r[:, mode_column],
            color="black",
            lw=1.5,
            ls=":",
            label=f"superNOVA eigenpair {eigenpair_index}",
        )
        ax.set_ylabel(rf"$m={m_value}$")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.25)

    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    axes[-1].set_xlabel(r"$\rho_{\rm tor}$")
    fig.suptitle(
        rf"ITPA eigenfunctions, panel {panel}: normalized $|\xi^\psi_m/r|$"
    )
    fig.tight_layout()
    return fig, axes


# %% Configuration.
species = "H"
gamma_s = 5 / 3
ntor = 6
mmax = 50
nfe = 50
order = 3
antenna_freq = 68.0
freq_window = [antenna_freq - 10., antenna_freq + 10]  # kHz

aminor = 1.0 * unyt.m
Rmajor = 10.0 * unyt.m
q0 = 1.71
qedge = 1.87
B0 = 3.0 * unyt.T
ns = 1024
ntheta = 512

# %% Building the equilibrium and continuum.
equ = ITPAEq(
    aminor=aminor,
    Rmajor=Rmajor,
    q0=q0,
    qedge=qedge,
    B0=B0,
    ns=ns,
    ntheta=ntheta,
    m_max=mmax,
)

theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
equ_desc = DESCSolver(
    LCFS={
        "R": Rmajor + aminor * np.cos(theta),
        "Z": aminor * np.sin(theta),
        "theta": theta,
    },
    Baxis=B0,
    q=equ.qprof,
    density=equ.density,
    pressure=equ.pressure,
    m_max=mmax,
    boundary_m_max=1,
    desc_resolution=(10, 10),
    mag_resolution=(ns, ntheta),
).solve()

fig_equ, ax_equ = equ.plot_summary()

R_desc = equ_desc.magspace["R"].values
z_desc = equ_desc.magspace["z"].values
rho_desc = np.mean(
    np.sqrt((R_desc - Rmajor.to("m").value) ** 2 + z_desc**2), axis=1
) / aminor.to("m").value

ax_equ[0, 0].plot(rho_desc, equ_desc.magspace["q"], "--", label="q DESC")
ax_equ[0, 0].legend()
ax_equ[0, 1].plot(
    rho_desc,
    equ_desc.magspace["pressure"] / 1.0e3,
    "--",
    color="tab:red",
    label="pressure DESC",
)
ax_equ[0, 1].legend()
for ii in np.linspace(0, R_desc.shape[0] - 1, 8, dtype=int):
    ax_equ[1, 0].plot(
        np.r_[R_desc[ii], R_desc[ii, 0]],
        np.r_[z_desc[ii], z_desc[ii, 0]],
        "--",
        color="tab:blue",
        lw=0.9,
    )
ax_equ[1, 0].set_title("magnetic-coordinate grid (DESC dashed)")

cont = sn.superContinuum(equ_desc, gamma_s=gamma_s, species=species, dense=True)

# %% Continuum guide for radial knots.
x = np.linspace(0.0, 1.0, 1000)
rho = _psi_to_rho(equ, x)
w = cont.solve(
    ntor=ntor,
    which="full",
    normalize_eigvals=True,
    return_eigvecs=True,
    psi_vals=x,
    classify=True,
    mmax=mmax,
)
wsaw = cont.solve(
    ntor=ntor,
    which="slow_sound",
    normalize_eigvals=True,
    psi_vals=x,
    classify=False,
    mmax=mmax,
)

fig, ax = w[-1].plot(x=x, scale=1e-3/(2 * np.pi))
# fig, ax = plt.subplots()
ax.plot(x, np.sqrt(wsaw.real).to("kHz").v / (2 * np.pi), "-")
ax.set_xlabel("$r/a$")
ax.set_ylabel("Frequency (kHz)")
ax.set_ylim([0, 100])
ax.axhline(antenna_freq, color="k", ls="--", lw=1.0, label="Drive frequency")

crossings, branches = continuum_crossings(x, wsaw, antenna_freq)
ax.plot(
    _psi_to_rho(equ, crossings),
    np.ones_like(crossings) * antenna_freq,
    label="Continuum crossings",
)

# Plotting the TAE frequency line.
B0 = equ("B0").data.sel(mpol=0).squeeze().real.values[0] * unyt.T
xB0 = equ("B0").data.sel(mpol=0).squeeze().x.values
dens = equ("density").data.sel(mpol=0).squeeze().real.values * unyt.m**-3
mass = 2.014 * unyt.Unit("amu", registry=equ.registry)
vA = B0 / np.sqrt(unyt.mu_0 * dens * mass)
qprof = equ("q").data.sel(mpol=0).squeeze().real.values

R0 = equ.equ.fluxdata.Raxis * unyt.m
omega_TAE = vA / (2 * qprof * R0) / (2 * np.pi)  # in Hz
ax.plot(_psi_to_rho(equ, xB0), omega_TAE.to("kHz").v, ":", color="0.5", lw=1.0, label="TAE frequency")

# %% Split-rational NEP.
# knots = get_knots_clustered_at(crossings, nfe, threshold=0.1)
knots = np.linspace(0.0, 1.0, nfe + 1)  # Uniform knots for testing.
radialfe = sn.BSplineFE(knots=knots, order=order, domain=[0, 1], quad_order=6)
solver = sn.superSolver3(equ_desc, cont, radialfe=radialfe)

freq_window = unyt.unyt_array(freq_window, "kHz", registry=solver.equ.registry)
omega_window = freq_window.to("omegaA").v * 2 * np.pi

omega_min, omega_max = omega_window
target_omega = 0.5 * (omega_min + omega_max)

omega_center = 0.5 * (omega_min + omega_max)
s_min = omega_min**2
s_max = omega_max**2
s_center = 0.5 * (s_min + s_max)
s_radius = 0.5 * abs(s_max - s_min)


region = {
    "type": "ellipse",
    "center": s_center,
    "radius": s_radius,
    "vscale": 0.25,
}

bc_kind = BC_KIND  # Try "edge", "axis", "full", or "none".

sb = solver.build_solver_blocks(ntor=ntor, mmax=mmax)
wd_orientation = apply_wd_orientation_override(solver, sb, WD_ORIENTATION)
radial_derivative_check = check_radial_derivative_matrix(solver, sb)
if WRITE_SOLVER_BLOCKS:
    try:
        sbdat = sb.to_xarray()
        sbdat.to_netcdf(
            REPO_ROOT / "no_sync" / "ITPA" / "solver_blocks_itpa.h5",
            engine="h5netcdf",
        )
    except ModuleNotFoundError as exc:
        print(f"Skipping solver block dump: {exc}")

split_check = check_solver3_split_operator_consistency(
    solver,
    sb,
    target_omega,
    regularizer=REGULARIZER,
    bc_kind=bc_kind,
)

# T0, T1, Tminus1 = solver.assemble_split_matrices(
#     ntor=ntor,
#     max_dm=5,
#     regularizer=1e-3,
#     apply_bcs=False,
#     verbose=True,
# )
# print("T0", T0.shape, T0.nnz)
# print("T1", T1.shape, T1.nnz)
# print("Tminus1", Tminus1.shape, Tminus1.nnz)

tmp = solver.solve_nep_split(
    ntor=ntor,
    target_omega=target_omega,
    max_dm=10,
    regularizer=0.0,
    apply_bcs=True,
    bc_kind=bc_kind,
    region=region,
    verbose=True,
    nev=6,
    max_eigenpairs=6,
    compute_residuals=True,
)

# quality_rows = summarize_solver3_xipsi_eigenpairs(
#     tmp,
#     solver,
#     sb,
#     bc_kind=bc_kind,
# )
# print_solver3_xipsi_summary(quality_rows)
# fig_quality, ax_quality = plot_solver3_xipsi_summary(quality_rows)

# smoothest_eigenpair = min(quality_rows, key=lambda row: row["roughness"])["eigenpair"]
# best_residual_eigenpair = min(quality_rows, key=lambda row: row["residual"])["eigenpair"]
# lowest_radial_derivative_eigenpair = min(
#     quality_rows,
#     key=lambda row: row["radial_derivative_ratio"],
# )["eigenpair"]
# print("smoothest eigenpair:", smoothest_eigenpair)
# print("best-residual eigenpair:", best_residual_eigenpair)
# print("lowest-radial-derivative eigenpair:", lowest_radial_derivative_eigenpair)

fig_cont_nep, ax_cont_nep = plot_slow_sound_with_eigenmode_extensions(
    tmp,
    solver,
    sb,
    x,
    wsaw,
    region=region,
    extension_threshold=0.67,
)

# %%
fig_xipsi, ax_xipsi = plot_xipsi_eigenpair(
    tmp,
    solver,
    sb,
    eigenpair_index=5,
    mpol=None,
    field="xi^psi",
)

# xipsi_energy = xipsi_kinetic_energy(tmp, solver, sb, eigenpair_index=2)
# print(f"xi^psi kinetic-energy diagnostic: {xipsi_energy:.6e}")

# kpsieff = np.zeros(tmp.sizes["eigenpair"], dtype=float)
# for eigenpair_index in range(tmp.sizes["eigenpair"]):
#     kpsieff[eigenpair_index] = xipsi_effective_kpsi(tmp, solver, sb, eigenpair_index=eigenpair_index)

# fig, ax = plt.subplots()
# ax.semilogy(np.arange(tmp.sizes["eigenpair"]), kpsieff, "o-")
# ax.axhline(2*np.pi*nfe*order, color='black', ls='--', lw=1.0, label=r"$2\pi N_\mathrm{fe}$")
# ax.legend()

# xipsi_overlap = xipsi_normalized_overlap(
#     tmp, solver, sb, eigenpair_a=3, eigenpair_b=5
# )
# print(f"xi^psi normalized overlap (3, 5): {xipsi_overlap*100:.2f}%")

# n_eigenpairs = tmp.sizes["eigenpair"]
# overlap_matrix = np.full((n_eigenpairs, n_eigenpairs), np.nan)
# for eigenpair_a in range(n_eigenpairs):
#     for eigenpair_b in range(eigenpair_a + 1):
#         overlap_matrix[eigenpair_a, eigenpair_b] = xipsi_normalized_overlap(
#             tmp, solver, sb, eigenpair_a, eigenpair_b
#         )

# %% Discover self-similar modes.
# Applying a mask to discover self-similar modes.
# correlation_threshold = 1.
# tmp_reduced = reduce_xipsi_eigenmodes(
#     tmp, solver, sb, correlation_threshold=correlation_threshold
# )
# print("representative eigenpairs:", tmp_reduced["eigenpair"].values)

# mask = np.abs(overlap_matrix) < correlation_threshold
# overlap_matrix[mask] = np.nan
# overlap_matrix[np.diag_indices(n_eigenpairs)] = np.nan  # Self-overlap is always 1.0.

# fig_overlap, ax_overlap = plt.subplots()
# overlap_image = ax_overlap.imshow(overlap_matrix, vmin=0.0, vmax=1.0, cmap="viridis")
# fig_overlap.colorbar(overlap_image, ax=ax_overlap, label=r"normalized overlap")
# ax_overlap.set_xlabel("eigenpair b")
# ax_overlap.set_ylabel("eigenpair a")
# ax_overlap.set_xticks(np.arange(n_eigenpairs))
# ax_overlap.set_yticks(np.arange(n_eigenpairs))
# fig_overlap.tight_layout()

# %% Comparison to codes.
plot_itpa_code_eigenfunctions(
    tmp,
    solver,
    sb,
    REPO_ROOT / "no_sync" / "ITPA" / "konies2018_figure5_eigenfunctions.csv",
    eigenpair_index=3,
    panel="A",
    codes=None,
)
