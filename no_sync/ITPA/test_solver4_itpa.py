r"""Interactive ITPA workflow for the compatible mixed-spline solver4.

This file follows the repository's ``# %%`` diagnostic style.  Run individual
cells from VS Code/Spyder to keep the equilibrium, continuum, mixed matrices,
DESC reconstruction, eigenmodes, and figures available in the interactive
namespace.

For a non-interactive whole-file check:

    conda run -n desc-petsc-np2 env MPLBACKEND=Agg NUMBA_DISABLE_JIT=1 \
        python no_sync/ITPA/test_solver4_itpa.py
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "supernova-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import unyt
import xarray as xr
from scipy.sparse.csgraph import structural_rank

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
from no_sync.solver4_analysis import (
    reconstruct_solver4_field,
    xipsi_effective_kpsi,
    xipsi_radial_location,
)
from no_sync.solver4_plots import (
    add_q_profile_axis,
    plot_slow_sound_with_eigenmode_extensions,
    plot_xipsi_eigenpair,
)
from supernova.core.desc_helper import DESCSolver
from supernova.core.itpa_benchmark import ITPAEq


plt.ion()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _psi_to_rho(equilibrium, psi):
    """Map normalized poloidal flux to the benchmark minor-radius coordinate."""
    psi = np.asarray(psi, dtype=float)
    magspace = getattr(equilibrium, "magspace", None)
    if magspace is not None and {"R", "z"}.issubset(magspace.data_vars):
        psi_geometry = np.asarray(magspace.coords["psi"], dtype=float)
        R_geometry = np.asarray(magspace["R"], dtype=float)
        z_geometry = np.asarray(magspace["z"], dtype=float)
        R_axis = float(getattr(equilibrium, "Rmajor", np.mean(R_geometry[0])))
        z_axis = float(np.mean(z_geometry[0]))
        rho_geometry = np.mean(
            np.sqrt((R_geometry - R_axis) ** 2 + (z_geometry - z_axis) ** 2),
            axis=1,
        ) / float(equilibrium.aminor)
        if psi_geometry[0] > 0.0:
            psi_geometry = np.r_[0.0, psi_geometry]
            rho_geometry = np.r_[0.0, rho_geometry]
        if psi_geometry[-1] < 1.0:
            psi_geometry = np.r_[psi_geometry, 1.0]
            rho_geometry = np.r_[rho_geometry, 1.0]
        return np.interp(psi, psi_geometry, rho_geometry)
    if hasattr(equilibrium, "_psi") and hasattr(equilibrium, "_rho"):
        return np.interp(psi, equilibrium._psi, equilibrium._rho)
    return np.sqrt(np.clip(psi, 0.0, None))


def _psi_to_rhotor(equilibrium, psi):
    """Map normalized poloidal flux to ``sqrt(s)`` for the ITPA comparison."""
    psi = np.asarray(psi, dtype=float)
    if hasattr(equilibrium, "_psi") and hasattr(equilibrium, "_s"):
        s_tor = np.interp(psi, equilibrium._psi, equilibrium._s)
        return np.sqrt(np.clip(s_tor, 0.0, 1.0))

    q_profile = equilibrium("q")
    q_data = q_profile.data.sel(mpol=0, ntor=0).real
    psi_q = np.asarray(q_data.x, dtype=float)
    q = np.asarray(q_data.values, dtype=float)
    if not np.all(np.isfinite(q)) or not (
        np.all(q > 0.0) or np.all(q < 0.0)
    ):
        raise ValueError("q must be finite, non-zero, and keep one sign.")

    s_tor = np.zeros_like(psi_q)
    s_tor[1:] = np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(psi_q))
    if not np.isfinite(s_tor[-1]) or s_tor[-1] == 0.0:
        raise ValueError("The integrated toroidal flux must have a finite edge value.")
    s_tor /= s_tor[-1]
    return np.sqrt(
        np.clip(np.interp(psi, psi_q, s_tor), 0.0, 1.0)
    )


_ITPA_QUANTITY_ALIASES = {
    "xi_psi": "xi_psi",
    "xipsi": "xi_psi",
    "xi^psi": "xi_psi",
    "xi_r": "xi_r",
    "xir": "xi_r",
    "xi^r": "xi_r",
    "es": "es",
    "phi": "es",
    "phi_es": "es",
    "electrostatic_potential": "es",
}
_ITPA_QUANTITY_LABELS = {
    "xi_psi": r"\xi^\psi",
    "xi_r": r"\xi^r",
    "es": r"\phi^{\rm ES}",
}


def _canonical_itpa_quantity(quantity):
    """Return the canonical name for an ITPA comparison quantity."""
    key = str(quantity).strip().lower().replace(" ", "_")
    try:
        return _ITPA_QUANTITY_ALIASES[key]
    except KeyError as error:
        raise ValueError(
            "quantity must be 'xi_psi', 'xi_r', or 'ES'; "
            f"got {quantity!r}."
        ) from error


def _surface_profile(equilibrium, name, psi):
    """Interpolate a real equilibrium flux function onto ``psi``."""
    psi = np.asarray(psi, dtype=float)
    magspace = getattr(equilibrium, "magspace", None)
    if magspace is not None and name in magspace:
        data = magspace[name]
        surface_dims = [dim for dim in data.dims if dim != "psi"]
        if surface_dims:
            data = data.mean(surface_dims)
        psi_profile = np.asarray(data.coords["psi"], dtype=float)
        values = np.asarray(data, dtype=np.complex128)
    else:
        profile = equilibrium(name, normalize=False)
        data = profile.data.sel(mpol=0, ntor=0)
        psi_profile = np.asarray(data.x, dtype=float)
        values = np.asarray(data, dtype=np.complex128)

    if np.max(np.abs(values.imag), initial=0.0) > 1.0e-12:
        raise ValueError(f"The equilibrium profile {name!r} must be real.")
    values = values.real
    if not np.all(np.isfinite(values)):
        raise ValueError(f"The equilibrium profile {name!r} is non-finite.")
    return np.interp(psi, psi_profile, values)


def _normalized_grad_psi(equilibrium, psi):
    """Return ``d psi_norm / dr`` for the circular ITPA surfaces."""
    magspace = getattr(equilibrium, "magspace", None)
    if magspace is None or "gpsi_psi" not in magspace:
        raise ValueError("xi_r plotting requires equilibrium gpsi_psi data.")

    gpsi_psi = magspace["gpsi_psi"]
    surface_dims = [dim for dim in gpsi_psi.dims if dim != "psi"]
    grad_psi = np.sqrt(np.clip(np.asarray(gpsi_psi, dtype=float), 0.0, None))
    if surface_dims:
        axes = tuple(gpsi_psi.get_axis_num(dim) for dim in surface_dims)
        grad_psi = np.mean(grad_psi, axis=axes)
    psi_geometry = np.asarray(gpsi_psi.coords["psi"], dtype=float)

    psi_edge = getattr(equilibrium, "chi_edge", None)
    if psi_edge is None:
        psi_edge = getattr(equilibrium, "_psi_edge", None)
    if psi_edge is None and "psi_pol" in magspace:
        psi_pol = np.asarray(magspace["psi_pol"], dtype=float)
        psi_edge = psi_pol[-1] - psi_pol[0]
    if psi_edge is None:
        raise ValueError(
            "xi_r plotting requires the physical edge poloidal flux."
        )
    psi_edge = abs(float(psi_edge))
    if not np.isfinite(psi_edge) or psi_edge == 0.0:
        raise ValueError("The physical edge poloidal flux must be finite and non-zero.")

    grad_psi_norm = np.asarray(grad_psi, dtype=float) / psi_edge
    if not np.all(np.isfinite(grad_psi_norm)):
        raise ValueError("The normalized poloidal-flux gradient is non-finite.")
    return np.interp(np.asarray(psi, dtype=float), psi_geometry, grad_psi_norm)


def _itpa_plot_quantity(
    result,
    solver,
    *,
    eigenpair_index,
    mpol,
    quantity,
    n_plot,
):
    """Reconstruct the selected solver4 quantity for an ITPA comparison."""
    quantity = _canonical_itpa_quantity(quantity)
    psi_plot, result_mpol, xi_psi = reconstruct_solver4_field(
        result,
        solver,
        field="xi_psi",
        eigenpair_index=eigenpair_index,
        n_plot=n_plot,
    )
    mode_indices = []
    for m_value in mpol:
        matches = np.flatnonzero(result_mpol == m_value)
        if matches.size == 0:
            raise ValueError(f"m={m_value} is outside the solver harmonic range.")
        mode_indices.append(int(matches[0]))
    selected_mpol = result_mpol[np.asarray(mode_indices)]
    values = xi_psi[:, mode_indices]

    if quantity == "xi_r":
        grad_psi_norm = _normalized_grad_psi(solver.equ, psi_plot)
        values = np.divide(
            values,
            grad_psi_norm[:, None],
            out=np.zeros_like(values),
            where=grad_psi_norm[:, None] != 0.0,
        )
        values[0] = 0.0
    elif quantity == "es":
        # The Boozer-coordinate relation supplied for this diagnostic is
        # xi^psi_nm = i phi_nm (I n + g m) / h.  Recover the canonical
        # unscaled Boozer I as h-gq so this diagnostic also checks the identity.
        h = _surface_profile(solver.equ, "h", psi_plot)
        g = _surface_profile(solver.equ, "g", psi_plot)
        q = _surface_profile(solver.equ, "q", psi_plot)
        I = h - g * q
        # The displayed relation uses exp(i*(m*theta+n*zeta)), so its
        # n=-ntor for the same labelled harmonic. superNOVA stores the
        # conjugate exp(i*(ntor*zeta-m*theta)) coefficient; conjugating the
        # displayed relation changes its leading +i to -i.
        n_boozer = -int(result.attrs["ntor"])
        denominator = (
            I[:, None] * n_boozer
            + g[:, None] * np.asarray(selected_mpol, dtype=float)[None, :]
        )
        denominator_scale = max(float(np.max(np.abs(denominator))), 1.0)
        if np.any(np.abs(denominator) <= 1.0e-12 * denominator_scale):
            raise ValueError(
                "Cannot reconstruct ES where the Boozer factor I*n + g*m "
                "vanishes."
            )
        values = h[:, None] * values / (-1j * denominator)

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"The reconstructed {_ITPA_QUANTITY_LABELS[quantity]} field is non-finite."
        )
    return psi_plot, selected_mpol, values, quantity


def plot_itpa_code_eigenfunctions(
    result,
    solver,
    csv_path,
    *,
    eigenpair_index=0,
    panel="A",
    codes=None,
    mpol=(9, 10, 11, 12),
    quantity="xi_psi",
    n_plot=512,
):
    """Compare one selectable solver4 field with ITPA code eigenfunctions."""
    csv_path = Path(csv_path)
    required_columns = {
        "panel",
        "code",
        "line_style",
        "m",
        "sqrt_s",
        "normalized_amplitude",
    }
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing_columns = required_columns.difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                f"{csv_path} is missing CSV columns {sorted(missing_columns)}."
            )
        selected_codes = None if codes is None else set(codes)
        mpol = tuple(int(m_value) for m_value in mpol)
        rows = [
            row
            for row in reader
            if row["panel"] == str(panel)
            and int(row["m"]) in mpol
            and (selected_codes is None or row["code"] in selected_codes)
        ]

    if not rows:
        raise ValueError(f"No CSV eigenfunctions found for panel {panel!r}.")

    psi_plot, selected_mpol, solver_values, quantity = _itpa_plot_quantity(
        result,
        solver,
        eigenpair_index=eigenpair_index,
        mpol=mpol,
        quantity=quantity,
        n_plot=n_plot,
    )
    sqrt_s = _psi_to_rhotor(solver.equ, psi_plot)
    solver_amplitude = np.abs(solver_values)
    solver_amplitude /= selected_mpol[None, :]**2
    normalization = float(np.max(solver_amplitude))
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("Cannot normalize a zero or non-finite solver4 eigenfunction.")
    solver_amplitude /= normalization

    code_names = list(dict.fromkeys(row["code"] for row in rows))
    code_colors = dict(zip(code_names, plt.get_cmap("tab10").colors))
    line_styles = {"solid": "-", "dashed": "--", "dotted": ":"}
    fig, axes = plt.subplots(
        len(mpol),
        1,
        sharex=True,
        figsize=(7, 2.0 * len(mpol)),
    )
    axes = np.atleast_1d(axes)

    for mode_column, (axis, m_value) in enumerate(zip(axes, selected_mpol)):
        for code in code_names:
            code_rows = sorted(
                (
                    row
                    for row in rows
                    if row["code"] == code and int(row["m"]) == m_value
                ),
                key=lambda row: float(row["sqrt_s"]),
            )
            if not code_rows:
                continue
            axis.plot(
                [float(row["sqrt_s"]) for row in code_rows],
                [float(row["normalized_amplitude"]) for row in code_rows],
                line_styles.get(code_rows[0]["line_style"], "-"),
                color=code_colors[code],
                label=code,
            )
        axis.plot(
            sqrt_s,
            solver_amplitude[:, mode_column],
            color="black",
            lw=1.5,
            ls=":",
            label=f"superNOVA solver4 eigenpair {eigenpair_index}",
        )
        axis.set_ylabel(rf"$m={m_value}$")
        axis.set_ylim(0.0, 1.05)
        axis.grid(True, alpha=0.25)
        add_q_profile_axis(
            axis,
            solver.equ,
            psi_plot,
            sqrt_s,
            show_ylabel=mode_column == len(mpol) // 2,
        )

    axes[0].legend(loc="upper left", bbox_to_anchor=(1.16, 1.0))
    axes[-1].set_xlabel(r"$\sqrt{s}=\rho_{\rm tor}$")
    fig.suptitle(
        rf"ITPA eigenfunctions, panel {panel}: "
        rf"normalized $|{_ITPA_QUANTITY_LABELS[quantity]}_m|$"
    )
    fig.tight_layout()
    return fig, axes


def check_mixed_assembly(
    solver,
    sb,
    *,
    ntor,
    mmax,
    max_dm,
    regularizer,
):
    """Validate compatible identities and return raw/reduced split operators."""
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
        max_dm=max_dm,
        regularizer=regularizer,
        apply_bcs=False,
    )
    reduced = solver.assemble_split_matrices(
        ntor=ntor,
        mmax=mmax,
        max_dm=max_dm,
        regularizer=regularizer,
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
    prolongation = bc.get_sparse_prolongation()
    bc_rank = int(np.linalg.matrix_rank(bc.C))
    _require(
        bc_rank == 2 * layout.n_modes,
        f"Unexpected full boundary rank {bc_rank}.",
    )
    _require(
        prolongation.shape == (layout.n_trial, layout.n_rows),
        f"Unexpected prolongation shape {prolongation.shape}.",
    )
    np.testing.assert_allclose(
        bc.C @ prolongation.toarray(),
        0.0,
        atol=1.0e-13,
    )
    for raw_matrix, reduced_matrix in zip(raw, reduced):
        np.testing.assert_allclose(
            reduced_matrix.toarray(),
            (raw_matrix @ prolongation).toarray(),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    combined_pattern = (
        abs(reduced[0]) + abs(reduced[1]) + abs(reduced[2])
    ).tocsr()
    pattern_rank = int(structural_rank(combined_pattern))
    _require(
        pattern_rank == layout.n_rows,
        f"Reduced mixed pattern has structural rank {pattern_rank}/{layout.n_rows}.",
    )

    summary = {
        "n_high": layout.n_high,
        "n_low": layout.n_low,
        "n_modes": layout.n_modes,
        "raw_shape": raw[0].shape,
        "full_bc_shape": reduced[0].shape,
        "bc_rank": bc_rank,
        "combined_structural_rank": pattern_rank,
        "max_derivative_error": float(
            np.max(np.abs(dphi_direct - dphi_compatible))
        ),
        "max_G_error": float(
            np.max(np.abs(G_direct - mixed.G_low_high.toarray()))
        ),
        "max_CP_residual": float(
            np.max(np.abs(bc.C @ prolongation.toarray()))
        ),
    }
    return summary, raw, reduced, bc, prolongation


def frequency_region(solver, frequency_window_khz):
    """Convert a physical-frequency window into the solver's omega/s region."""
    frequency = unyt.unyt_array(
        frequency_window_khz,
        "kHz",
        registry=solver.equ.registry,
    )
    omega_min, omega_max = frequency.to("omegaA").v * (2.0 * np.pi)
    s_min = omega_min**2
    s_max = omega_max**2
    target_omega = 0.5 * (omega_min + omega_max)
    region = {
        "type": "ellipse",
        "center": 0.5 * (s_min + s_max),
        "radius": 0.5 * abs(s_max - s_min),
        "vscale": 0.25,
    }
    return target_omega, region


def check_structured_result(result, solver, sb):
    """Check packed fields, complex reconstruction, and endpoint constraints."""
    required = {
        "eigenvector",
        "p1_coefficient",
        "xi_psi_coefficient",
        "xi_s_coefficient",
        "div_xi_coefficient",
    }
    _require(
        required.issubset(result.data_vars),
        f"Result is missing {sorted(required.difference(result.data_vars))}.",
    )
    _require(result.sizes.get("eigenpair", 0) > 0, "NLEIGS returned no eigenpairs.")

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

    bc = solver._build_nep_boundary_condition(sb, bc_kind="full")
    boundary_residual = np.asarray(bc.C @ result["eigenvector"].values.T)
    np.testing.assert_allclose(boundary_residual, 0.0, atol=1.0e-10)

    reconstruction_shapes = {}
    for field in ("p1", "xi_psi", "xi_s", "div_xi"):
        psi_field, m_pol, values = reconstruct_solver4_field(
            result,
            solver,
            field=field,
            eigenpair_index=0,
            n_plot=128,
        )
        _require(np.iscomplexobj(values), f"{field} reconstruction lost complex dtype.")
        _require(np.all(np.isfinite(values)), f"{field} reconstruction is non-finite.")
        reconstruction_shapes[field] = (psi_field.size, m_pol.size)

    return {
        "max_endpoint_residual": endpoint_residual,
        "max_boundary_residual": float(np.max(np.abs(boundary_residual))),
        "reconstruction_shapes": reconstruction_shapes,
    }


# %% Configuration.
species = "H"
gamma_s = 5.0 / 3.0
ntor = 6
mmax = 20
nfe = 100
order = 3
quad_order = 6
ns = 256
ntheta = 256

aminor = 1.0 * unyt.m
Rmajor = 10.0 * unyt.m
q0 = 1.71
qedge = 1.87
B0 = 3.0 * unyt.T

antenna_freq = 68.0  # kHz
freq_window = np.array([60.0, 76.0])  # kHz
eigenpair_index = 0
xipsi_plot_eigenpair_index = 3
itpa_comparison_eigenpair_index = 2

max_dm = 3
regularizer = 1.0e-5
bc_kind = "full"
nev = 4
max_eigenpairs = 4
tol = 1.0e-7
maxit = 80

run_nleigs = True
use_desc_reconstruction = False
desc_resolution = (10, 10)
desc_fast = False
plot_matrix_sparsity = False
plot_all_fields = True
plot_itpa_code_comparison = True
max_modes_to_plot = 10
extension_threshold = 0.67
verbose = False

itpa_code_csv = (
    REPO_ROOT
    / "no_sync"
    / "ITPA"
    / "konies2018_figure5_eigenfunctions.csv"
)
itpa_comparison_panel = "A"
itpa_comparison_codes = None
itpa_comparison_mpol = (9, 10, 11, 12)
itpa_comparison_quantity = "ES"  # "xi_psi", "xi_r", or "ES"

write_results = False
results_path = REPO_ROOT / "no_sync" / "ITPA" / "solver4_itpa_eigenmodes.h5"


# %% Build the analytical ITPA reference equilibrium.
equ_itpa = ITPAEq(
    aminor=aminor,
    Rmajor=Rmajor,
    q0=q0,
    qedge=qedge,
    B0=B0,
    ns=ns,
    ntheta=ntheta,
    m_max=mmax,
)

fig_equ, ax_equ = equ_itpa.plot_summary()
fig_equ.suptitle("Analytical ITPA benchmark equilibrium")


# %% Reconstruct the ITPA equilibrium with DESC and compare its profiles/grid.
equ_desc = None
desc_summary = None
if use_desc_reconstruction:
    theta_desc_lcfs = np.linspace(
        0.0,
        2.0 * np.pi,
        ntheta,
        endpoint=False,
    )
    equ_desc = DESCSolver(
        LCFS={
            "R": Rmajor + aminor * np.cos(theta_desc_lcfs),
            "Z": aminor * np.sin(theta_desc_lcfs),
            "theta": theta_desc_lcfs,
        },
        Baxis=B0,
        q=equ_itpa.qprof,
        density=equ_itpa.density,
        pressure=equ_itpa.pressure,
        m_max=mmax,
        boundary_m_max=1,
        desc_resolution=desc_resolution,
        mag_resolution=(ns, ntheta),
    ).solve(fast=desc_fast)
    desc_summary = equ_desc.print_summary()

    R_desc = np.asarray(equ_desc.magspace["R"])
    z_desc = np.asarray(equ_desc.magspace["z"])
    psi_desc = np.asarray(equ_desc.magspace.coords["psi"])
    rho_desc = _psi_to_rho(equ_desc, psi_desc)

    ax_equ[0, 0].plot(
        rho_desc,
        np.asarray(equ_desc.magspace["q"]),
        "--",
        label="q DESC",
    )
    ax_equ[0, 0].legend()
    ax_equ[0, 1].plot(
        rho_desc,
        np.asarray(equ_desc.magspace["pressure"]) / 1.0e3,
        "--",
        color="tab:red",
        label="pressure DESC",
    )
    ax_equ[0, 1].legend()
    for surface_index in np.linspace(0, R_desc.shape[0] - 1, 8, dtype=int):
        ax_equ[1, 0].plot(
            np.r_[R_desc[surface_index], R_desc[surface_index, 0]],
            np.r_[z_desc[surface_index], z_desc[surface_index, 0]],
            "--",
            color="tab:blue",
            lw=0.9,
        )
    ax_equ[1, 0].set_title("Magnetic-coordinate grid (DESC dashed)")
    fig_equ.suptitle("Analytical ITPA benchmark with DESC reconstruction")

equ = equ_desc if equ_desc is not None else equ_itpa
solver_radial_label = (
    r"$\sqrt{s}=\rho_{\rm tor}$" if equ_desc is not None else r"$r/a$"
)

cont = sn.superContinuum(
    equ,
    gamma_s=gamma_s,
    species=species,
    dense=True,
)


# %% Solve and visualize the slow-sound continuum.
x = np.linspace(0.0, 1.0, 600)
rho = _psi_to_rho(equ, x)
wsaw = cont.solve(
    ntor=ntor,
    which="slow_sound",
    normalize_eigvals=True,
    psi_vals=x,
    classify=False,
    mmax=mmax,
)
wsaw_khz = np.sqrt(wsaw.real).to("kHz").v / (2.0 * np.pi)

fig_cont, ax_cont = plt.subplots()
for branch in range(wsaw_khz.shape[1]):
    ax_cont.plot(
        rho,
        wsaw_khz[:, branch],
        color="tab:blue",
        lw=0.9,
        alpha=0.75,
        label="slow-sound continuum" if branch == 0 else None,
    )
ax_cont.axhspan(
    freq_window[0],
    freq_window[1],
    color="tab:green",
    alpha=0.18,
    label="NLEIGS search window",
)
ax_cont.axhline(
    antenna_freq,
    color="black",
    ls="--",
    lw=1.0,
    label="reference frequency",
)
ax_cont.set_xlabel(r"$r/a$")
ax_cont.set_ylabel("Frequency [kHz]")
ax_cont.set_title(rf"ITPA slow-sound continuum, $n={ntor}$")
ax_cont.grid(True, alpha=0.3)
ax_cont.legend()
fig_cont.tight_layout()


# %% Build solver4 and inspect the compatible high/low spline pair.
knots = np.linspace(0.0, 1.0, nfe + 1)
radialfe = sn.BSplineFE(
    knots=knots,
    order=order,
    domain=(0.0, 1.0),
    quad_order=quad_order,
)
solver = sn.superSolver4(equ, cont, radialfe=radialfe)
sb = solver.build_solver_blocks(ntor=ntor, mmax=mmax, max_dm=max_dm)
mixed = solver._get_mixed_radial_data()
pair = mixed.pair
layout = solver._mixed_layout(sb)

x_basis = np.linspace(0.0, 1.0, 600)
phi_high_plot = pair.high.eval_basis(x_basis)
phi_low_plot = pair.low.eval_basis(x_basis)

fig_basis, ax_basis = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
for basis_value in phi_high_plot:
    ax_basis[0].plot(x_basis, basis_value, lw=1.0)
for basis_value in phi_low_plot:
    ax_basis[1].plot(x_basis, basis_value, lw=1.0)
ax_basis[0].set_ylabel(r"$B_i^H$")
ax_basis[0].set_title(
    rf"High trial space: order={pair.high.order}, $n_H={pair.high.get_ndofs()}$"
)
ax_basis[1].set_ylabel(r"$B_i^L$")
ax_basis[1].set_xlabel(r"Normalized poloidal flux $\psi$")
ax_basis[1].set_title(
    rf"Compatible low space: order={pair.low.order}, $n_L={pair.low.get_ndofs()}$"
)
for axis in ax_basis:
    axis.grid(True, alpha=0.25)
fig_basis.tight_layout()


# %% Validate mixed projection and full one-sided boundary reduction.
(
    assembly_summary,
    split_raw,
    split_reduced,
    bc,
    prolongation,
) = check_mixed_assembly(
    solver,
    sb,
    ntor=ntor,
    mmax=mmax,
    max_dm=max_dm,
    regularizer=regularizer,
)

print("solver4 ITPA mixed assembly diagnostic passed")
for name, value in assembly_summary.items():
    print(f"  {name}: {value}")


# %% Visualize rectangular and reduced sparse operators.
if plot_matrix_sparsity:
    combined_reduced = (
        abs(split_reduced[0])
        + abs(split_reduced[1])
        + abs(split_reduced[2])
    ).tocsr()
    fig_sparsity, ax_sparsity = plt.subplots(1, 3, figsize=(14, 4.5))
    ax_sparsity[0].spy(split_raw[0], markersize=0.35)
    ax_sparsity[0].set_title(r"Raw rectangular $T_0$")
    ax_sparsity[1].spy(prolongation, markersize=0.35)
    ax_sparsity[1].set_title(r"Trial prolongation $P$")
    ax_sparsity[2].spy(combined_reduced, markersize=0.35)
    ax_sparsity[2].set_title(r"Reduced pattern $|T_0|+|T_1|+|T_{-1}|$")
    for axis in ax_sparsity:
        axis.set_xlabel("column")
        axis.set_ylabel("row")
    fig_sparsity.tight_layout()


# %% Define the physical-frequency search region.
target_omega, region = frequency_region(solver, freq_window)
print("NLEIGS search region in s=omega^2:")
for name, value in region.items():
    print(f"  {name}: {value}")


# %% Solve the full-boundary mixed-spline NEP.
tmp = None
if run_nleigs:
    tmp = solver.solve_nep_split(
        ntor=ntor,
        target_omega=target_omega,
        region=region,
        mmax=mmax,
        max_dm=max_dm,
        regularizer=regularizer,
        apply_bcs=True,
        bc_kind=bc_kind,
        nev=4,
        max_eigenpairs=4,
        tol=tol,
        maxit=maxit,
        compute_residuals=True,
        verbose=verbose,
    )
    _require(tmp.sizes.get("eigenpair", 0) > 0, "No solver4 eigenmodes converged.")
    eigenpair_index = min(eigenpair_index, tmp.sizes["eigenpair"] - 1)
    result_summary = check_structured_result(tmp, solver, sb)
    print("solver4 structured-result checks passed")
    for name, value in result_summary.items():
        print(f"  {name}: {value}")


# %% Summarize converged eigenpairs and radial quality metrics.
if tmp is not None:
    omega_values = np.asarray(tmp["omega"].values)
    frequency_khz = (
        omega_values.real * unyt.Unit("omegaA", registry=solver.equ.registry)
    ).to("kHz").v / (2.0 * np.pi)
    residual_norm = (
        np.asarray(tmp["residual_norm"].values)
        if "residual_norm" in tmp
        else np.full(tmp.sizes["eigenpair"], np.nan)
    )
    kpsi_effective = xipsi_effective_kpsi(tmp, solver, sb)
    peak_psi, radial_width = xipsi_radial_location(tmp, solver)

    mode_summary = xr.Dataset(
        data_vars={
            "frequency_khz": ("eigenpair", frequency_khz),
            "residual_norm": ("eigenpair", residual_norm),
            "kpsi_effective": ("eigenpair", kpsi_effective),
            "peak_psi": ("eigenpair", peak_psi),
            "radial_width": ("eigenpair", radial_width),
        },
        coords={"eigenpair": tmp.eigenpair.values},
    )
    print(mode_summary)


# %% Overlay solver4 eigenmode extensions on the continuum.
if tmp is not None:
    fig_cont_nep, ax_cont_nep = plot_slow_sound_with_eigenmode_extensions(
        tmp,
        solver,
        sb,
        x,
        wsaw,
        region=region,
        extension_threshold=extension_threshold,
        max_eigenpairs=len(tmp.eigenpair),
        radial_label=solver_radial_label,
    )


# %% Plot the selected xi^psi harmonics with their complex phase.
if tmp is not None:
    fig_xipsi, ax_xipsi = plot_xipsi_eigenpair(
        tmp,
        solver,
        sb,
        # eigenpair_index=min(
        #     xipsi_plot_eigenpair_index,
        #     tmp.sizes["eigenpair"] - 1,
        # ),
        eigenpair_index=2,
        field="xi^psi",
        max_modes=4,
        radial_label=solver_radial_label,
    )


# %% Compare the selected solver4 mode with the ITPA code benchmark.
if tmp is not None and plot_itpa_code_comparison:
    fig_itpa_codes, ax_itpa_codes = plot_itpa_code_eigenfunctions(
        tmp,
        solver,
        itpa_code_csv,
        # eigenpair_index=min(
        #     itpa_comparison_eigenpair_index,
        #     tmp.sizes["eigenpair"] - 1,
        # ),
        eigenpair_index=2,
        panel="E",
        codes=itpa_comparison_codes,
        mpol=itpa_comparison_mpol,
        quantity="xi_psi",
    )


# %% Visualize all four mixed fields for the selected eigenpair.
if tmp is not None and plot_all_fields:
    field_specs = [
        ("p1", r"$|p_{1,m}|$"),
        ("xi_psi", r"$|\xi^\psi_m|$"),
        ("xi_s", r"$|\xi^s_m|$"),
        ("div_xi", r"$|\nabla\!\cdot\!\xi_m|$"),
    ]
    fig_fields, ax_fields = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        sharex=True,
        constrained_layout=True,
    )
    reconstructed_fields = {}
    for axis, (field, label) in zip(ax_fields.ravel(), field_specs):
        psi_field, m_pol, values = reconstruct_solver4_field(
            tmp,
            solver,
            field=field,
            eigenpair_index=eigenpair_index,
            n_plot=400,
        )
        rho_field = _psi_to_rho(equ, psi_field)
        reconstructed_fields[field] = values
        image = axis.pcolormesh(
            rho_field,
            m_pol,
            np.abs(values).T,
            shading="auto",
            cmap="magma",
        )
        axis.set_title(label)
        axis.set_ylabel(r"Poloidal harmonic $m$")
        fig_fields.colorbar(image, ax=axis, pad=0.02)
    for axis in ax_fields[-1]:
        axis.set_xlabel(r"$r/a$")


# %% Compare effective radial scale and localization across eigenpairs.
if tmp is not None:
    fig_quality, ax_quality = plt.subplots(1, 2, figsize=(10, 4.2))
    ax_quality[0].semilogy(
        mode_summary.eigenpair,
        mode_summary.kpsi_effective,
        "o-",
    )
    ax_quality[0].set_xlabel("Eigenpair")
    ax_quality[0].set_ylabel(r"Effective $k_\psi$")
    ax_quality[0].grid(True, alpha=0.3)

    ax_quality[1].errorbar(
        mode_summary.eigenpair,
        mode_summary.peak_psi,
        yerr=mode_summary.radial_width,
        fmt="o",
        capsize=3,
    )
    ax_quality[1].set_xlabel("Eigenpair")
    ax_quality[1].set_ylabel(r"Peak $\psi$ and RMS width")
    ax_quality[1].set_ylim(0.0, 1.0)
    ax_quality[1].grid(True, alpha=0.3)
    fig_quality.tight_layout()


# %% Optional result persistence.
if tmp is not None and write_results:
    tmp.to_netcdf(results_path, engine="h5netcdf")
    print(f"Wrote solver4 ITPA eigenmodes to {results_path}")
