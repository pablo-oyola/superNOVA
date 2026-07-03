"""
Minimal split-rational NEP experiment for superSolver3 with ITPAEq.
"""

from pathlib import Path
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.sparse.linalg import norm as sparse_norm
import unyt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import supernova as sn
from supernova.core.continuum import continuum_crossings, get_knots_clustered_at
from supernova.core.itpa_benchmark import ITPAEq
from supernova.core._solver_kernel2 import build_M_at, get_T_sparse
from supernova.core.solver3 import (
    _build_local_M_split,
    _project_local_tensor,
    _slice_to_quadrature,
)

plt.ion()

WRITE_SOLVER_BLOCKS = os.environ.get("SUPERNOVA_ITPA_WRITE_BLOCKS", "0") == "1"
RUN_CISS = os.environ.get("SUPERNOVA_ITPA_RUN_CISS", "0") == "1"
BC_KIND = os.environ.get("SUPERNOVA_ITPA_BC_KIND", "full")
MAX_DM = int(os.environ.get("SUPERNOVA_ITPA_MAX_DM", "10"))
NEV = int(os.environ.get("SUPERNOVA_ITPA_NEV", "10"))
MAX_EIGENPAIRS = int(os.environ.get("SUPERNOVA_ITPA_MAX_EIGENPAIRS", str(NEV)))
NFE = int(os.environ.get("SUPERNOVA_ITPA_NFE", "200"))
REGULARIZER = float(os.environ.get("SUPERNOVA_ITPA_REGULARIZER", "1e-5"))


def _psi_to_rho(equ, psi):
    return np.interp(np.asarray(psi, dtype=float), equ._psi, equ._rho)


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


def summarize_solver3_xipsi_eigenpairs(
    result,
    solver,
    sb,
    bc_kind=None,
    n_plot=512,
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
        "peak_rho  edge/peak  axis/peak  dominance"
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
            f"{row['axis_ratio']:.3e}   "
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

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True)
    ax = ax.ravel()
    ax[0].semilogy(eigenpair, residual, "o-")
    ax[0].set_ylabel("reduced residual")
    ax[1].semilogy(eigenpair, roughness, "o-")
    ax[1].set_ylabel(r"$\xi^\psi$ roughness")
    ax[2].plot(eigenpair, peak_rho, "o-")
    ax[2].set_ylabel("peak r/a")
    ax[3].plot(eigenpair, dominant_m, "o-")
    ax[3].set_ylabel("dominant m")
    for axis in ax:
        axis.grid(True, alpha=0.3)
        axis.set_xlabel("eigenpair")
    fig.tight_layout()
    return fig, ax


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


def plot_xipsi_eigenpair(
    result,
    solver,
    sb,
    eigenpair_index=0,
    mpol=None,
    n_plot=512,
):
    """
    Plot xi^psi for one full-space solver3 eigenvector versus r/a.

    The solver3 full state is ordered as [p1, xi^psi, xi^s, div_xi] for each
    radial basis coefficient. ``solve_nep_split`` expands reduced BC solves
    before returning the dataset.

    If ``mpol`` is None, plot ``|xi^psi_m|`` for every available poloidal
    harmonic. If ``mpol`` is an integer, plot real/imag/absolute value only for
    that harmonic. If ``mpol`` is an array, plot ``|xi^psi_m|`` for that subset.
    """
    if result.sizes.get("eigenpair", 0) == 0:
        raise ValueError("No converged eigenpairs are available to plot.")

    n_radial = solver.radialfe.get_ndofs()
    n_modes = sb.m_range.size
    n_state = 4 * n_modes
    eigenvector = result["eigenvector"].isel(eigenpair=eigenpair_index).values
    if eigenvector.size != n_radial * n_state:
        raise ValueError(
            "plot_xipsi_eigenpair expects a full-space eigenvector. "
            "Check tmp.attrs['eigenvector_space'] and tmp.attrs['full_dofs']."
        )

    state = eigenvector.reshape((n_radial, n_state))
    xipsi_coeffs = state[:, n_modes : 2 * n_modes]

    mode_indices = None
    if mpol is not None:
        requested_mpol = np.atleast_1d(mpol).astype(int)
        mode_indices = []
        missing_mpol = []
        for m_value in requested_mpol:
            matches = np.flatnonzero(sb.m_range == m_value)
            if matches.size == 0:
                missing_mpol.append(int(m_value))
            else:
                mode_indices.append(int(matches[0]))
        if missing_mpol:
            raise ValueError(
                f"Requested mpol={missing_mpol}, but available range is "
                f"{sb.m_range[0]}..{sb.m_range[-1]}."
            )
        mode_indices = np.asarray(mode_indices, dtype=int)

    mesh = solver.radialfe.get_mesh()
    psi_plot = np.linspace(mesh[0], mesh[-1], n_plot)
    rho_plot = _psi_to_rho(solver.equ, psi_plot)
    phi_plot = solver.radialfe.eval_basis(psi_plot)
    xipsi_all = phi_plot.T @ xipsi_coeffs

    s_value = result["s"].isel(eigenpair=eigenpair_index).values.item()
    omega_value = result["omega"].isel(eigenpair=eigenpair_index).values.item()
    residual_value = None
    if "residual_norm" in result:
        residual_value = result["residual_norm"].isel(eigenpair=eigenpair_index).values.item()
    residual_label = "" if residual_value is None else rf", res={residual_value:.1e}"

    fig, ax = plt.subplots()

    if mpol is None:
        mode_norms = np.linalg.norm(xipsi_all, axis=0)
        dominant_index = int(np.argmax(mode_norms))
        norm = mpl.colors.Normalize(vmin=float(sb.m_range.min()), vmax=float(sb.m_range.max()))
        cmap = plt.get_cmap("viridis")
        for ii, m_value in enumerate(sb.m_range):
            linewidth = 2.0 if ii == dominant_index else 0.9
            alpha = 1.0 if ii == dominant_index else 0.75
            ax.plot(
                rho_plot,
                np.abs(xipsi_all[:, ii]),
                color=cmap(norm(float(m_value))),
                lw=linewidth,
                alpha=alpha,
            )

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(r"$m$")
        ax.set_ylabel(r"$|\xi^\psi_m(r/a)|$")
        ax.set_title(
            rf"eigenpair {eigenpair_index}, all $m$; dominant $m={int(sb.m_range[dominant_index])}$, "
            rf"$s={s_value:.4g}$, $\omega={omega_value:.4g}$"
            rf"{residual_label}"
        )
    elif mode_indices.size == 1:
        mode_index = int(mode_indices[0])
        xipsi = xipsi_all[:, mode_index]
        ax.plot(rho_plot, xipsi.real, label=r"$\Re\,\xi^\psi$")
        ax.plot(rho_plot, xipsi.imag, "--", label=r"$\Im\,\xi^\psi$")
        ax.plot(rho_plot, np.abs(xipsi), ":", label=r"$|\xi^\psi|$")
        ax.set_ylabel(r"$\xi^\psi_m(r/a)$")
        ax.set_title(
            rf"eigenpair {eigenpair_index}, $m={int(sb.m_range[mode_index])}$, "
            rf"$s={s_value:.4g}$, $\omega={omega_value:.4g}$"
            rf"{residual_label}"
        )
        ax.legend()
    else:
        mode_norms = np.linalg.norm(xipsi_all[:, mode_indices], axis=0)
        dominant_local = int(np.argmax(mode_norms))
        dominant_index = int(mode_indices[dominant_local])
        selected_m = sb.m_range[mode_indices]
        norm = mpl.colors.Normalize(vmin=float(selected_m.min()), vmax=float(selected_m.max()))
        cmap = plt.get_cmap("viridis")
        for mode_index in mode_indices:
            m_value = sb.m_range[mode_index]
            linewidth = 2.0 if mode_index == dominant_index else 0.9
            alpha = 1.0 if mode_index == dominant_index else 0.75
            ax.plot(
                rho_plot,
                np.abs(xipsi_all[:, mode_index]),
                color=cmap(norm(float(m_value))),
                lw=linewidth,
                alpha=alpha,
            )
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(r"$m$")
        ax.set_ylabel(r"$|\xi^\psi_m(r/a)|$")
        ax.set_title(
            rf"eigenpair {eigenpair_index}, selected $m$; dominant $m={int(sb.m_range[dominant_index])}$, "
            rf"$s={s_value:.4g}$, $\omega={omega_value:.4g}$"
            rf"{residual_label}"
        )

    ax.set_xlabel(r"$r/a$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def _omega_norm_to_frequency_khz(omega_norm, solver):
    """
    Convert normalized angular frequency omega/omega_A to cyclic kHz.
    """
    omega_norm = np.asarray(omega_norm, dtype=np.complex128)
    omega_unit = unyt.Unit("omegaA", registry=solver.equ.registry)
    return (omega_norm.real * omega_unit).to("kHz").v / (2 * np.pi)


def _search_frequency_band_khz(region, solver):
    """
    Project the real-axis part of an s=omega^2 ellipse to positive frequency.
    """
    center = complex(region["center"])
    radius = float(region["radius"])
    s_left = max(0.0, center.real - radius)
    s_right = max(0.0, center.real + radius)
    omega_band = np.sqrt([s_left, s_right])
    return _omega_norm_to_frequency_khz(omega_band, solver)


def _xipsi_envelope_for_eigenpair(result, solver, sb, eigenpair_index, psi_plot):
    """
    Return the radial xi^psi envelope summed over all poloidal harmonics.
    """
    n_radial = solver.radialfe.get_ndofs()
    n_modes = sb.m_range.size
    n_state = 4 * n_modes
    eigenvector = result["eigenvector"].isel(eigenpair=eigenpair_index).values
    if eigenvector.size != n_radial * n_state:
        raise ValueError(
            "The continuum/eigenmode overlay expects full-space eigenvectors. "
            "Check tmp.attrs['eigenvector_space'] and tmp.attrs['full_dofs']."
        )

    state = eigenvector.reshape((n_radial, n_state))
    xipsi_coeffs = state[:, n_modes : 2 * n_modes]
    phi_plot = solver.radialfe.eval_basis(psi_plot)
    xipsi_all = phi_plot.T @ xipsi_coeffs
    return np.linalg.norm(xipsi_all, axis=1)


def plot_slow_sound_with_eigenmode_extensions(
    result,
    solver,
    sb,
    psi_continuum,
    slow_sound,
    region=None,
    extension_threshold=0.2,
    n_plot=512,
    max_eigenpairs=None,
):
    """
    Plot the slow-sound continuum with NEP eigenmode frequencies and radial extent.

    The mode extent is the interval where the all-m xi^psi envelope exceeds
    ``extension_threshold`` times its peak.
    """
    if result.sizes.get("eigenpair", 0) == 0:
        raise ValueError("No converged eigenpairs are available to plot.")

    psi_continuum = np.asarray(psi_continuum, dtype=float)
    rho_continuum = _psi_to_rho(solver.equ, psi_continuum)
    slow_sound_khz = np.sqrt(slow_sound.real).to("kHz").v / (2 * np.pi)

    fig, ax = plt.subplots()
    continuum_label = "slow sound continuum"
    for branch in range(slow_sound_khz.shape[1]):
        ax.plot(
            rho_continuum,
            slow_sound_khz[:, branch],
            color="0.65",
            lw=0.8,
            alpha=0.75,
            label=continuum_label if branch == 0 else None,
        )

    if region is not None:
        fmin, fmax = _search_frequency_band_khz(region, solver)
        ax.axhspan(fmin, fmax, color="tab:orange", alpha=0.18, label="NEP search band")

    mesh = solver.radialfe.get_mesh()
    psi_plot = np.linspace(mesh[0], mesh[-1], n_plot)
    rho_plot = _psi_to_rho(solver.equ, psi_plot)
    n_eigenpairs = result.sizes["eigenpair"]
    if max_eigenpairs is not None:
        n_eigenpairs = min(n_eigenpairs, int(max_eigenpairs))

    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(1, n_eigenpairs)))
    for eigenpair_index in range(n_eigenpairs):
        envelope = _xipsi_envelope_for_eigenpair(result, solver, sb, eigenpair_index, psi_plot)
        peak = float(np.max(envelope))
        if peak == 0.0:
            continue

        active = envelope >= extension_threshold * peak
        if not np.any(active):
            continue

        active_rho = rho_plot[active]
        peak_rho = rho_plot[int(np.argmax(envelope))]
        omega_value = result["omega"].isel(eigenpair=eigenpair_index).values.item()
        frequency_khz = float(_omega_norm_to_frequency_khz(omega_value, solver))
        residual = result["residual_norm"].values[eigenpair_index] if "residual_norm" in result else np.nan
        label = rf"NEP $k={eigenpair_index}$, res={residual:.1e}"
        color = colors[eigenpair_index]

        ax.hlines(
            frequency_khz,
            active_rho[0],
            active_rho[-1],
            color=color,
            lw=2.5,
            label=label,
        )
        ax.plot(peak_rho, frequency_khz, "o", color=color, ms=4)

    ax.set_xlabel(r"$r/a$")
    ax.set_ylabel("Frequency (kHz)")
    extension_percent = 100 * extension_threshold
    ax.set_title(
        rf"Slow-sound continuum and NEP mode extensions "
        rf"($|\xi^\psi|$ > {extension_percent:.0f}% peak)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return fig, ax


# %% Configuration.
species = "D"
gamma_s = 5 / 3
ntor = 6
mmax = 50
nfe = NFE
order = 3
antenna_freq = 47.5
freq_window = [antenna_freq - 10., antenna_freq + 10-0]  # kHz

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

fig_equ, ax_equ = equ.plot_summary()
cont = sn.superContinuum(equ, gamma_s=gamma_s, species=species, dense=True)

# %% Continuum guide for radial knots.
x = np.linspace(0.0, 1.0, 1000)
rho = _psi_to_rho(equ, x)
# w = cont.solve(
#     ntor=ntor,
#     which="full",
#     normalize_eigvals=True,
#     return_eigvecs=True,
#     psi_vals=x,
#     classify=True,
#     mmax=mmax,
# )
wsaw = cont.solve(
    ntor=ntor,
    which="slow_sound",
    normalize_eigvals=True,
    psi_vals=x,
    classify=False,
    mmax=mmax,
)

# fig, ax = w[-1].plot(x=x)
fig, ax = plt.subplots()
ax.plot(rho, np.sqrt(wsaw.real).to("kHz").v / (2 * np.pi), "-")
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
solver = sn.superSolver3(equ, cont, radialfe=radialfe)

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
if WRITE_SOLVER_BLOCKS:
    try:
        sbdat = sb.to_xarray()
        sbdat.to_netcdf("no_sync/solver_blocks_itpa.h5", engine="h5netcdf")
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
    max_dm=MAX_DM,
    regularizer=REGULARIZER,
    apply_bcs=True,
    bc_kind=bc_kind,
    region=region,
    verbose=True,
    nev=NEV,
    max_eigenpairs=MAX_EIGENPAIRS,
    compute_residuals=True,
)

quality_rows = summarize_solver3_xipsi_eigenpairs(
    tmp,
    solver,
    sb,
    bc_kind=bc_kind,
)
print_solver3_xipsi_summary(quality_rows)
fig_quality, ax_quality = plot_solver3_xipsi_summary(quality_rows)

smoothest_eigenpair = min(quality_rows, key=lambda row: row["roughness"])["eigenpair"]
best_residual_eigenpair = min(quality_rows, key=lambda row: row["residual"])["eigenpair"]
print("smoothest eigenpair:", smoothest_eigenpair)
print("best-residual eigenpair:", best_residual_eigenpair)

fig_cont_nep, ax_cont_nep = plot_slow_sound_with_eigenmode_extensions(
    tmp,
    solver,
    sb,
    x,
    wsaw,
    region=region,
    extension_threshold=0.67,
)

fig_xipsi, ax_xipsi = plot_xipsi_eigenpair(tmp, solver, sb, eigenpair_index=9, mpol=None)
fig_xipsi_smooth, ax_xipsi_smooth = plot_xipsi_eigenpair(
    tmp,
    solver,
    sb,
    eigenpair_index=smoothest_eigenpair,
    mpol=None,
)
fig_xipsi_residual, ax_xipsi_residual = plot_xipsi_eigenpair(
    tmp,
    solver,
    sb,
    eigenpair_index=best_residual_eigenpair,
    mpol=None,
)

# %% Solving using the CISS pathway
if RUN_CISS:
    tmp = solver.solve_nep_split_ciss(
        ntor=ntor,
        target_omega=target_omega,
        max_dm=5,
        regularizer=1e-4,
        apply_bcs=True,
        bc_kind=bc_kind,
        region=region,
        verbose=True,
        nev=20,
        max_eigenpairs=20,
        compute_residuals=True,
        ciss_sizes={"ip": 32, "bs": 16, "ms": 4, "bsmax": 64},
        ciss_extraction="RITZ",
    )
