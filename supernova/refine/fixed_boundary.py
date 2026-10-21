"""Fixed-boundary TokaMaker refinement of a tabulated equilibrium.

A recorded g-file is rarely a consistent Grad-Shafranov equilibrium, and the
continuum is unforgiving about that: on NSTX 128741 the highest sound-Alfven
branch drops from 118,098 kHz to 43,741 kHz once the equilibrium is re-solved,
and most of what disappears was never physical.

This module re-solves the case as a fixed-boundary equilibrium, holding
pressure fixed while fitting the shape of ``F F'`` -- and optionally the total
plasma current -- to the pyCOCOS geometry-derived Boozer q profile.  It writes
three files next to each other:

``<stem>.eqdsk``
    the refined equilibrium, in the COCOS recorded in the sidecar
``<stem>.json``
    the diagnostics sidecar, which carries ``output_cocos`` so no downstream
    script has to hard-code it
``<stem>.png``
    flux surfaces, q profiles and the q residual, source against refined

It replaces the per-device driver scripts, which were 403 lines each and
differed only in their paths.

Refinement runs in whatever environment has OpenFUSIONToolkit; the stability
solve reads the files it leaves behind and needs no such install.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pycocos as pc
from freeqdsk import geqdsk
from matplotlib.path import Path as MplPath
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from supernova.refine.qfit import FixedBoundaryQFitter, QFitConfig


DEFAULT_FIT = QFitConfig(
    fit_range=(0.04, 0.94),
    fit_points=45,
    q_sample_range=(0.01, 0.99),
    q_sample_points=260,
    max_iterations=20,
    # The 13-knot fixed-boundary inverse problem reaches an approximately
    # 1.7e-2 absolute RMS discretization floor on the coarse mesh.
    target_rms_error=2.0e-2,
    profile_fd_step_abs=0.008,
    log_current_fd_step=0.005,
    regularization_initial=2.0e-3,
    regularization_floor=2.0e-4,
    ridge=2.0e-4,
    max_profile_step=0.20,
    max_log_current_step=0.06,
    damping_factors=(1.0, 0.5, 0.25, 0.125, 0.0625),
    ffp_bounds=(0.02, 3.0),
    # Preserve the measured total current; only the F F' shape is fitted.
    fit_plasma_current=False,
    verbose=True,
)


@dataclass(frozen=True)
class RefinementConfig:
    """Everything the MAST and NSTX drivers used to hard-code as constants."""

    source_eqdsk: Path
    output_dir: Path
    stem: str
    source_cocos: int = 5
    output_cocos: int = 7
    run_info: str = ""

    # mesh schedule, in metres
    coarse_dx: float = 0.030
    fine_dx: float = 0.015
    refinement_dx_sequence: tuple[float, ...] = (0.025, 0.020, 0.015)
    fe_order: int = 3

    # Boozer sampling used to build the q target
    boozer_lpsi: int = 81
    boozer_ltheta: int = 128
    boozer_rhopol_range: tuple[float, float] = (0.01, 0.99)
    boozer_spectral_max_mode: int = 16

    ffp_knot_count: int = 13
    export_grid: tuple[int, int] = (257, 257)
    nthreads: int = 2

    # acceptance thresholds
    max_source_q_disagreement: float = 0.03
    max_fine_q_relative_rms: float = 0.05
    max_current_relative_error: float = 1.0e-3

    fit: QFitConfig = field(default_factory=lambda: DEFAULT_FIT)

    @property
    def eqdsk_path(self) -> Path:
        return self.output_dir / f"{self.stem}.eqdsk"

    @property
    def diagnostics_path(self) -> Path:
        return self.output_dir / f"{self.stem}.json"

    @property
    def figure_path(self) -> Path:
        return self.output_dir / f"{self.stem}.png"


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def relative_rms(values: np.ndarray, reference: np.ndarray) -> float:
    scale = max(rms(reference), np.finfo(float).tiny)
    return rms(np.asarray(values) - np.asarray(reference)) / scale


def drop_closing_duplicate(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=float)
    if curve.shape[0] > 1 and np.allclose(
        curve[0], curve[-1], rtol=0.0, atol=1.0e-12
    ):
        return curve[:-1].copy()
    return curve.copy()


def unconverged_fit_message(fit_result, coarse, config: RefinementConfig) -> str:
    """Why the coarse q fit stopped, and where its q residual is worst.

    A residual that refuses to shrink in one region -- the reversed-shear core
    of DIII-D 202020 with 13 knots -- usually means the knots are too sparse
    there for the current profile that q needs.
    """
    x = np.asarray(fit_result.target_locations, dtype=float)
    dq = np.interp(x, coarse.psi, coarse.q) - np.asarray(fit_result.target_q, dtype=float)
    worst = int(np.argmax(np.abs(dq)))
    return (
        f"TokaMaker q fit did not converge: {fit_result.reason}. The q RMS "
        f"{fit_result.final_rms_error:.3g} is above the "
        f"{config.fit.target_rms_error:.3g} target, and the residual is worst, "
        f"dq={dq[worst]:+.3f}, at psi_N={x[worst]:.2f}. "
        f"{config.ffp_knot_count} F F' knots may be too few to shape the "
        "current there; try more."
    )


def header_cocos_candidates(path: Path) -> tuple[int, ...]:
    """The COCOS indices the g-file's header signs allow, per pyCOCOS."""
    from pycocos.io.cocos import identify_cocos

    with Path(path).open("r", encoding="ascii") as stream:
        raw = geqdsk.read(stream)
    q = np.asarray(raw["qpsi"], dtype=float)
    resolution = identify_cocos(float(q[q.size // 2]), float(raw["cpasma"]),
                                float(raw["bcentr"]), float(raw["simagx"]),
                                float(raw["sibdry"]))
    return tuple(int(index) for index in resolution.candidates)


def repair_fixed_boundary_export(
    path: Path,
    *,
    workflow: FixedBoundaryQFitter,
    boundary: np.ndarray,
) -> dict[str, float]:
    """Replace only TokaMaker's undefined exterior by a monotone continuation."""
    with path.open("r", encoding="ascii") as stream:
        eqdsk = geqdsk.read(stream)

    solver = workflow.solver
    bounds = np.asarray(solver.psi_bounds, dtype=float)
    if bounds.shape != (2,):
        raise RuntimeError(f"Unexpected TokaMaker psi_bounds shape: {bounds.shape}")
    evaluator = solver.get_field_eval("psi")
    axis_flux = float(evaluator.eval(np.asarray(solver.o_point)[:2])[0])
    axis_index = int(np.argmin(np.abs(bounds - axis_flux)))
    boundary_flux = float(bounds[1 - axis_index])
    signed_span = boundary_flux - axis_flux
    if not np.isfinite(signed_span) or signed_span == 0.0:
        raise RuntimeError("TokaMaker returned an invalid poloidal-flux span.")

    eqdsk["simagx"] = axis_flux
    eqdsk["sibdry"] = boundary_flux
    r_grid = np.linspace(
        float(eqdsk["rleft"]),
        float(eqdsk["rleft"] + eqdsk["rdim"]),
        int(eqdsk["nx"]),
    )
    z_grid = np.linspace(
        float(eqdsk["zmid"] - 0.5 * eqdsk["zdim"]),
        float(eqdsk["zmid"] + 0.5 * eqdsk["zdim"]),
        int(eqdsk["ny"]),
    )
    rr, zz = np.meshgrid(r_grid, z_grid, indexing="ij")
    points = np.column_stack((rr.ravel(), zz.ravel()))
    inside = MplPath(boundary, closed=True).contains_points(
        points, radius=1.0e-10
    ).reshape(rr.shape)

    # The 500 recorded LCFS vertices already resolve the shaped NSTX boundary
    # more finely than the export grid.  Their nearest-neighbour distance gives
    # a positive outward coordinate without assuming circular geometry.
    distance = cKDTree(boundary).query(points)[0].reshape(rr.shape)
    area_twice = np.sum(
        boundary[:, 0] * np.roll(boundary[:, 1], -1)
        - np.roll(boundary[:, 0], -1) * boundary[:, 1]
    )
    effective_minor_radius = np.sqrt(abs(0.5 * area_twice) / np.pi)
    exterior = ~inside
    eqdsk["psi"][exterior] = (
        boundary_flux
        + np.sign(signed_span)
        * abs(signed_span)
        * distance[exterior]
        / effective_minor_radius
    )
    eqdsk["rbdry"] = np.asarray(boundary[:, 0], dtype=float)
    eqdsk["zbdry"] = np.asarray(boundary[:, 1], dtype=float)
    eqdsk["nbdry"] = int(boundary.shape[0])

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="ascii") as stream:
        geqdsk.write(eqdsk, stream, label="NSTX TokaMaker refined")
    temporary.replace(path)
    return {
        "axis_flux_wb_per_rad": axis_flux,
        "boundary_flux_wb_per_rad": boundary_flux,
        "signed_flux_span_wb_per_rad": signed_span,
        "effective_minor_radius_m": float(effective_minor_radius),
        "exterior_grid_fraction": float(np.mean(exterior)),
    }


def refine_equilibrium(config: RefinementConfig, *, plot: bool = True) -> dict:
    """Refine one equilibrium and write the eqdsk, sidecar and figure.

    Returns the diagnostics dictionary that is also written to disk.  Raises
    if any acceptance threshold in ``config`` is missed, so a refinement that
    silently degraded the equilibrium cannot reach the solver.
    """
    if not config.source_eqdsk.is_file():
        raise FileNotFoundError(config.source_eqdsk)

    source = pc.EQDSK(str(config.source_eqdsk), cocos_in=config.source_cocos)
    boozer = source.compute_coordinates(
        "boozer",
        lpsi=config.boozer_lpsi,
        ltheta=config.boozer_ltheta,
        rhopol_min=config.boozer_rhopol_range[0],
        rhopol_max=config.boozer_rhopol_range[1],
        spectral_max_mode=config.boozer_spectral_max_mode,
    )

    header_candidates = header_cocos_candidates(config.source_eqdsk)
    if config.source_cocos not in header_candidates:
        print(f"Warning: the g-file header signs allow COCOS {header_candidates}, "
              f"not the {config.source_cocos} it was read as.", flush=True)

    psi_axis = float(source.geometry.attrs["psi_ax"])
    psi_lcfs = float(source.geometry.attrs["psi_bdy"])
    psi_span = psi_lcfs - psi_axis
    if psi_span == 0.0:
        raise ValueError("The imported equilibrium has no poloidal flux span.")
    # In COCOS 1 the flux grows outward for one current direction and falls for
    # the other (or for a mis-declared COCOS).  Grad-Shafranov is invariant
    # under psi -> -psi with p' and F F' reversed, so the fit always works on
    # the outward-growing mirror: the refined file carries |Ip| and |F|.
    orientation = 1.0 if psi_span > 0.0 else -1.0
    if orientation < 0.0:
        print("The source flux falls outward in COCOS 1; refining its mirror "
              "psi -> -psi, so the refined file has the opposite current sign.",
              flush=True)

    boozer_psi = (np.asarray(boozer.deriv.psi0, dtype=float) - psi_axis) / psi_span
    boozer_q = np.abs(np.asarray(boozer.deriv.q, dtype=float))
    # pyCOCOS orders the Boozer surfaces by ascending psi, which runs from the
    # edge inward when the flux falls outward.
    order = np.argsort(boozer_psi)
    boozer_psi, boozer_q = boozer_psi[order], boozer_q[order]
    gs = source.gs_profs
    gs_psi = np.asarray(gs.rhop, dtype=float) ** 2
    gs_q = np.abs(np.asarray(gs.q, dtype=float))
    gs_F = np.asarray(gs.fpol, dtype=float)
    gs_pressure = np.asarray(gs.pres, dtype=float)
    gs_pprime = np.asarray(gs.pprime, dtype=float)
    gs_ffprime = np.asarray(gs.ffprime, dtype=float)

    q_tabulated_on_boozer = np.interp(boozer_psi, gs_psi, gs_q)
    source_q_disagreement = relative_rms(boozer_q, q_tabulated_on_boozer)
    if source_q_disagreement > config.max_source_q_disagreement:
        raise RuntimeError(
            "Geometry-derived and tabulated source q disagree by "
            f"{source_q_disagreement:.3%}, above the "
            f"{config.max_source_q_disagreement:.3%} limit. Refining an "
            "equilibrium this inconsistent would fit the wrong target."
        )

    q_profile = {
        "x": np.r_[0.0, boozer_psi, 1.0],
        "y": np.r_[gs_q[0], boozer_q, gs_q[-1]],
    }
    lcfs = drop_closing_duplicate(
        np.column_stack((source._gdata["r_bdy"], source._gdata["z_bdy"]))
    )

    # TokaMaker's native absolute flux decreases outward for this positive-F
    # branch: psi_TM = orientation * (psi_LCFS - psi_COCOS1).  The variable
    # change reverses both Grad-Shafranov derivatives.
    pressure_prime_tm = -orientation * gs_pprime
    ffprime_tm = -orientation * gs_ffprime
    pressure_scale_seed = float(pressure_prime_tm[0])
    ffp_x = np.linspace(0.0, 1.0, config.ffp_knot_count)
    ffp_knots = np.interp(ffp_x, gs_psi, ffprime_tm)
    ffp_scale_seed = float(ffp_knots[0])
    if pressure_scale_seed == 0.0 or ffp_scale_seed == 0.0:
        raise ValueError("Imported source derivatives cannot have a zero axis scale.")

    pressure_profile = {
        "type": "linterp",
        "x": gs_psi,
        "y": pressure_prime_tm / pressure_scale_seed,
    }
    ffp_profile = {
        "type": "linterp",
        "x": ffp_x,
        "y": ffp_knots / ffp_scale_seed,
    }

    psi_interpolator = RegularGridInterpolator(
        (np.asarray(source.Rgrid, dtype=float), np.asarray(source.zgrid, dtype=float)),
        np.asarray(source.flux.psi, dtype=float),
        method="linear",
        bounds_error=True,
    )

    def initial_tokamaker_psi(node_locations: np.ndarray) -> np.ndarray:
        psi_cocos1 = np.asarray(psi_interpolator(node_locations), dtype=float)
        psi_cocos1 = np.clip(psi_cocos1, min(psi_axis, psi_lcfs), max(psi_axis, psi_lcfs))
        return orientation * (psi_lcfs - psi_cocos1)

    F0 = abs(float(gs_F[-1]))
    Ip_source = abs(float(source._gdata["Ip"]))
    pressure_edge = float(gs_pressure[-1])
    p_axis_target = float(gs_pressure[0] - pressure_edge)

    workflow = FixedBoundaryQFitter(
        lcfs=lcfs,
        F0=F0,
        pressure_profile=pressure_profile,
        ffp_profile=ffp_profile,
        q_profile=q_profile,
        Ip_seed=Ip_source,
        p_axis_target=p_axis_target,
        q_coordinate="psi",
        coarse_dx=config.coarse_dx,
        fine_dx=config.fine_dx,
        refinement_dx_sequence=config.refinement_dx_sequence,
        fe_order=config.fe_order,
        source_mode=1,
        nthreads=config.nthreads,
        initial_psi_evaluator=initial_tokamaker_psi,
        initial_ffp_scale=ffp_scale_seed,
        initial_pressure_scale=pressure_scale_seed,
        nonlinear_tolerance=1.0e-7,
        nonlinear_max_iterations=250,
        solver_output=False,
        fit_config=config.fit,
    )

    initial = workflow.solve_initial()
    fit_result = workflow.fit_coarse()
    coarse = workflow.coarse_samples
    if coarse is None:
        raise RuntimeError("The coarse fit produced no samples.")
    if not fit_result.converged:
        raise RuntimeError(unconverged_fit_message(fit_result, coarse, config))
    fine = workflow.refine()

    target_initial = np.interp(initial.psi, q_profile["x"], q_profile["y"])
    target_coarse = np.interp(coarse.psi, q_profile["x"], q_profile["y"])
    target_fine = np.interp(fine.psi, q_profile["x"], q_profile["y"])
    fine_q_relative_rms = relative_rms(fine.q, target_fine)
    Ip_relative_error = abs(fine.plasma_current - Ip_source) / Ip_source

    if fine_q_relative_rms > config.max_fine_q_relative_rms:
        raise RuntimeError(
            f"Fine-grid q relative RMS is too large: {fine_q_relative_rms}"
        )
    if Ip_relative_error > config.max_current_relative_error:
        raise RuntimeError(
            f"Fine-grid total-current relative error is too large: {Ip_relative_error}"
        )
    if workflow.refinement_used_cold_start:
        raise RuntimeError(
            "At least one mesh-refinement stage lost the warm-started branch."
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    boundary_span = np.ptp(lcfs, axis=0)
    export_margin = 0.05 * boundary_span
    nr, nz = config.export_grid
    workflow.save_eqdsk(
        str(config.eqdsk_path),
        nr=nr,
        nz=nz,
        rbounds=np.array(
            [lcfs[:, 0].min() - export_margin[0], lcfs[:, 0].max() + export_margin[0]]
        ),
        zbounds=np.array(
            [lcfs[:, 1].min() - export_margin[1], lcfs[:, 1].max() + export_margin[1]]
        ),
        run_info=config.run_info or config.stem,
        lcfs_pad=0.001,
        lcfs_pressure=pressure_edge,
        cocos=config.output_cocos,
    )
    continuation = repair_fixed_boundary_export(
        config.eqdsk_path,
        workflow=workflow,
        boundary=lcfs,
    )

    diagnostics = {
        "source_eqdsk": str(config.source_eqdsk),
        "output_eqdsk": str(config.eqdsk_path),
        "source_cocos": config.source_cocos,
        "source_cocos_header_candidates": list(header_candidates),
        "source_flux_orientation": int(orientation),
        "output_cocos": config.output_cocos,
        "source_grid": [int(source.Rgrid.size), int(source.zgrid.size)],
        "output_grid": [nr, nz],
        "source_boozer_q_relative_rms_vs_tabulated": source_q_disagreement,
        "fit_converged": bool(fit_result.converged),
        "fit_reason": fit_result.reason,
        "fit_rms_error": float(fit_result.final_rms_error),
        "fit_max_error": float(fit_result.final_max_error),
        "fine_q_relative_rms": fine_q_relative_rms,
        "source_plasma_current_A": Ip_source,
        "fine_plasma_current_A": float(fine.plasma_current),
        "plasma_current_relative_error": Ip_relative_error,
        "coarse_mesh_dx_m": config.coarse_dx,
        "fine_mesh_dx_m": config.fine_dx,
        "coarse_mesh_points": int(coarse.mesh_points),
        "fine_mesh_points": int(fine.mesh_points),
        "coarse_mesh_cells": int(coarse.mesh_cells),
        "fine_mesh_cells": int(fine.mesh_cells),
        "refinement_dx_sequence_m": list(config.refinement_dx_sequence),
        "refinement_stage_initializers": list(workflow.refinement_stage_initializers),
        "refinement_stage_iterations": list(workflow.refinement_stage_iterations),
        "nonlinear_tolerance": float(workflow.nonlinear_tolerance),
        "fixed_boundary_export_repair": continuation,
    }
    config.diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")

    if plot:
        _comparison_figure(
            config, source, lcfs, initial, coarse, fine,
            target_initial, target_coarse, target_fine,
        )
    return diagnostics


def load_diagnostics(path: Path | str) -> dict:
    """Read a refinement sidecar.

    ``Case.from_refined`` uses this to pick up ``output_cocos`` instead of
    making every case script repeat a ``7 if refined else 5`` ternary.
    """
    path = Path(path)
    if path.is_dir():
        candidates = sorted(path.glob("*.json"))
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one diagnostics sidecar in {path}, "
                f"found {len(candidates)}."
            )
        path = candidates[0]
    diagnostics = json.loads(path.read_text())
    for key in ("output_eqdsk", "output_cocos"):
        if key not in diagnostics:
            raise ValueError(f"{path} is not a refinement sidecar: no {key!r}.")
    return diagnostics


def _comparison_figure(config, source, lcfs, initial, coarse, fine,
                       target_initial, target_coarse, target_fine):
    """Source against refined: boundary and axis, q, and the q residual."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    axes[0].plot(lcfs[:, 0], lcfs[:, 1], "k", lw=2.0, label="recorded LCFS")
    source_axis = np.array(
        [float(source.geometry.R_axis), float(source.geometry.z_axis)]
    )
    axes[0].plot(*source_axis, "kx", label="source axis")
    axes[0].plot(*fine.magnetic_axis, "+", color="tab:red", label="refined axis")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("R [m]")
    axes[0].set_ylabel("Z [m]")
    axes[0].set_title("Fixed boundary and axis")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(initial.psi, target_initial, "k", lw=2.0, label="source Boozer")
    axes[1].plot(initial.psi, initial.q, ":", label="initial TokaMaker")
    axes[1].plot(coarse.psi, coarse.q, "-.", label="fitted coarse")
    axes[1].plot(fine.psi, fine.q, "--", label="fitted fine")
    axes[1].set_xlabel(r"$\psi_N$")
    axes[1].set_ylabel("q")
    axes[1].set_title("Safety factor")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(initial.psi, initial.q - target_initial, ":", label="initial")
    axes[2].plot(coarse.psi, coarse.q - target_coarse, "-.", label="coarse")
    axes[2].plot(fine.psi, fine.q - target_fine, "--", label="fine")
    axes[2].axhline(0.0, color="k", lw=0.8)
    axes[2].set_xlabel(r"$\psi_N$")
    axes[2].set_ylabel(r"$q-q_{\mathrm{target}}$")
    axes[2].set_title("q residual")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.suptitle(config.run_info or config.stem)
    fig.tight_layout()
    fig.savefig(config.figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
