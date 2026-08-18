#!/usr/bin/env python3
r"""Run and plot the three-level solver5 multigrid classifier on ITPA.

Example
-------
conda run -n desc-petsc-np2 env NUMBA_DISABLE_JIT=1 \
    python no_sync/solver5_multigrid_example.py --no-show \
    --save no_sync/solver5_multigrid_example.png

The requested cyclic-frequency band in kHz is converted to normalized angular
frequency through ``omega / omegaA``.  No additional species-mass factor is
applied: solver5 already includes the selected species mass in its density.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/supernova-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import unyt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import supernova as sn
from supernova.core.itpa_benchmark import ITPAEq


LABEL_COLORS = {
    "physical_gap": "tab:green",
    "continuum_resonance": "tab:red",
    "numerical_artifact": "tab:gray",
    "inconclusive": "tab:orange",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency-khz", nargs=2, type=float, default=(60.0, 76.0))
    parser.add_argument("--ntor", type=int, default=6)
    parser.add_argument("--mmax", type=int, default=20)
    parser.add_argument("--max-dm", type=int, default=3)
    parser.add_argument("--base-interior", type=int, default=7)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--quadrature-order", type=int, default=6)
    parser.add_argument("--nev", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--tol", type=float, default=1.0e-8)
    parser.add_argument("--maxit", type=int, default=200)
    parser.add_argument("--regularizer", type=float, default=0.0)
    parser.add_argument("--ns", type=int, default=128)
    parser.add_argument("--ntheta", type=int, default=128)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _base_spline(interior: int, order: int, quadrature_order: int) -> sn.BSplineFE:
    if interior < 0:
        raise ValueError("base-interior must be non-negative.")
    breakpoints = np.linspace(0.0, 1.0, interior + 2)
    knots = np.concatenate(
        (
            np.repeat(breakpoints[0], order),
            breakpoints[1:-1],
            np.repeat(breakpoints[-1], order),
        )
    )
    return sn.BSplineFE(
        knots=knots,
        order=order,
        domain=(0.0, 1.0),
        quad_order=quadrature_order,
        build_matrices=False,
    )


def _omega_range(equilibrium, frequency_khz) -> np.ndarray:
    frequency = unyt.unyt_array(
        frequency_khz,
        "kHz",
        registry=equilibrium.registry,
    )
    return np.asarray(frequency.to("omegaA").v, dtype=float) * (2.0 * np.pi)


def _omega_to_khz(equilibrium, omega) -> np.ndarray:
    omega = np.asarray(omega, dtype=np.complex128)
    unit = unyt.Unit("omegaA", registry=equilibrium.registry)
    return (omega.real * unit).to("kHz").v / (2.0 * np.pi)


def _rho_tor(equilibrium, psi) -> np.ndarray:
    psi = np.asarray(psi, dtype=float)
    if hasattr(equilibrium, "_psi") and hasattr(equilibrium, "_s"):
        s_tor = np.interp(psi, equilibrium._psi, equilibrium._s)
        return np.sqrt(np.clip(s_tor, 0.0, 1.0))
    q_profile = equilibrium("q").data.sel(mpol=0, ntor=0).real
    psi_q = np.asarray(q_profile.x, dtype=float)
    q = np.asarray(q_profile.values, dtype=float)
    if not np.all(np.isfinite(q)) or not (np.all(q > 0.0) or np.all(q < 0.0)):
        raise ValueError("q must be finite, nonzero, and retain one sign.")
    s_tor = np.zeros_like(psi_q)
    s_tor[1:] = np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(psi_q))
    if not np.isfinite(s_tor[-1]) or s_tor[-1] == 0.0:
        raise ValueError("Integrated toroidal flux has an invalid edge value.")
    s_tor /= s_tor[-1]
    return np.sqrt(np.clip(np.interp(psi, psi_q, s_tor), 0.0, 1.0))


def _print_summary(result, equilibrium) -> None:
    labels, counts = np.unique(result.labels.values, return_counts=True)
    print("\nClassification counts")
    for label in LABEL_COLORS:
        count = counts[labels == label]
        print(f"  {label:22s}: {int(count[0]) if count.size else 0}")
    print(
        "Search complete by level:",
        result.window_diagnostics["level_complete"].values.tolist(),
    )

    tracks = result.mode_tracks
    evidence = result.classification
    print("\ntrack  frequency[kHz]  label                  error       drift       overlap  reason")
    for track in range(tracks.sizes.get("track", 0)):
        level = int(evidence["last_level"].values[track])
        if level < 0:
            continue
        frequency = float(
            _omega_to_khz(equilibrium, tracks["omega"].values[track, level])
        )
        print(
            f"{track:5d}  {frequency:14.6f}  "
            f"{str(evidence['label'].values[track]):22s}  "
            f"{tracks['backward_error'].values[track, level]:.2e}  "
            f"{evidence['fine_frequency_change'].values[track]:.2e}  "
            f"{evidence['medium_fine_overlap'].values[track]:.4f}  "
            f"{evidence['reason'].values[track]}"
        )


def _plot(result, equilibrium, frequency_band) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    continuum = result.continuum
    rho = _rho_tor(equilibrium, continuum.coords["psi"].values)
    continuum_khz = _omega_to_khz(equilibrium, continuum["omega"].values)
    for branch in range(continuum.sizes["branch"]):
        axes[0].plot(rho, continuum_khz[:, branch], color="0.45", lw=0.7, alpha=0.7)
    axes[0].axhspan(*frequency_band, color="tab:blue", alpha=0.08)

    tracks = result.mode_tracks
    evidence = result.classification
    used_labels = set()
    for track in range(tracks.sizes.get("track", 0)):
        label = str(evidence["label"].values[track])
        color = LABEL_COLORS[label]
        present = np.asarray(tracks["present"].values[track], dtype=bool)
        levels = np.flatnonzero(present)
        if levels.size == 0:
            continue
        frequencies = _omega_to_khz(equilibrium, tracks["omega"].values[track, levels])
        final = int(levels[-1])
        peak = float(tracks["peak_radius"].values[track, final])
        peak_rho = float(_rho_tor(equilibrium, [peak])[0])
        axes[0].scatter(
            peak_rho,
            frequencies[-1],
            color=color,
            s=28,
            zorder=3,
            label=label if label not in used_labels else None,
        )
        axes[1].plot(levels, frequencies, "o-", color=color, lw=1.0, ms=3.5)
        used_labels.add(label)

    axes[0].set_xlabel(r"$\rho_{\rm tor}=\sqrt{s}$")
    axes[0].set_ylabel("Frequency [kHz]")
    axes[0].set_title("Full sound-Alfven continuum and final mode peaks")
    axes[0].grid(True, alpha=0.25)
    if used_labels:
        axes[0].legend()
    axes[1].set_xlabel("Nested radial level")
    axes[1].set_ylabel("Frequency [kHz]")
    axes[1].set_xticks(np.arange(tracks.sizes.get("level", 0)))
    axes[1].set_title("Classified mode tracks")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, axes


def main() -> None:
    args = _arguments()
    frequency_band = np.sort(np.asarray(args.frequency_khz, dtype=float))
    equilibrium = ITPAEq(
        aminor=1.0 * unyt.m,
        Rmajor=10.0 * unyt.m,
        q0=1.71,
        qedge=1.87,
        B0=3.0 * unyt.T,
        ns=args.ns,
        ntheta=args.ntheta,
        m_max=args.mmax,
    )
    continuum = sn.superContinuum(
        equilibrium,
        gamma_s=5.0 / 3.0,
        species="H",
        dense=True,
    )
    multigrid = sn.superSolver5Multigrid(
        equilibrium,
        continuum,
        _base_spline(args.base_interior, args.order, args.quadrature_order),
        levels=3,
    )
    result = multigrid.solve_pep(
        ntor=args.ntor,
        omega_range=_omega_range(equilibrium, frequency_band),
        mmax=args.mmax,
        max_dm=args.max_dm,
        regularizer=args.regularizer,
        search=sn.MultigridSearchOptions(
            nev=args.nev,
            tol=args.tol,
            maxit=args.maxit,
            max_depth=args.max_depth,
        ),
        verbose=args.verbose,
    )
    _print_summary(result, equilibrium)
    fig, _ = _plot(result, equilibrium, frequency_band)
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=180)
        print(f"Saved {args.save}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
