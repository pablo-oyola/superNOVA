"""``python -m supernova`` -- solve, reclassify, reacquire.

A thin argparse shell over :mod:`supernova.app.workflow`.  The subcommands are
the ones that survived the explorer's removal; ``serve`` and ``demo`` went with
it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from supernova.app.workflow import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DENSITY,
    DEFAULT_EQUILIBRIUM,
    _run_reacquire,
    _run_reclassify,
    _run_solve,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve = subparsers.add_parser("solve", help="run the multigrid DIII-D scan")
    solve.add_argument("--equilibrium", type=Path, default=DEFAULT_EQUILIBRIUM)
    solve.add_argument("--density", type=Path, default=DEFAULT_DENSITY)
    solve.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    solve.add_argument("--frequency-khz", nargs=2, type=float, default=(50.0, 150.0))
    solve.add_argument("--cocos-in", type=int, default=7)
    solve.add_argument("--ntor", type=int, default=4)
    solve.add_argument("--mmax", type=int, default=68)
    solve.add_argument("--max-dm", type=int, default=5)
    solve.add_argument("--gamma-s", type=float, default=1.0e-3)
    solve.add_argument("--species", default="D")
    solve.add_argument("--regularizer", type=float, default=0.0)
    solve.add_argument(
        "--base-spans",
        type=int,
        default=38,
        help="coarsest spans; three levels give 38/76/152 spans",
    )
    solve.add_argument("--levels", type=int, default=3)
    solve.add_argument("--order", type=int, default=3)
    solve.add_argument("--quadrature-order", type=int, default=6)
    solve.add_argument("--profile-points", type=int, default=128)
    solve.add_argument("--reconstruction-points", type=int, default=512)
    solve.add_argument("--display-continuum-points", type=int, default=1024)
    solve.add_argument("--nev", type=int, default=8)
    solve.add_argument("--tol", type=float, default=1.0e-8)
    solve.add_argument("--maxit", type=int, default=200)
    solve.add_argument(
        "--window-width-khz",
        type=float,
        default=2.5,
        help="maximum fixed core-window width; defaults to the legacy 2.5 kHz",
    )
    solve.add_argument(
        "--window-padding",
        type=float,
        default=0.10,
        help="fraction of one core width added to each side of a PEP solve region",
    )
    solve.add_argument(
        "--region-vscale",
        type=float,
        default=0.25,
        help="imaginary semiaxis scale for the elliptical PEP region",
    )
    solve.add_argument(
        "--factorization",
        choices=("mumps", "spqr", "petsc_qr"),
        default="mumps",
    )
    solve.add_argument("--save-raw-levels", action="store_true")
    solve.add_argument(
        "--adaptive-discovery",
        action="store_true",
        help="use K=4,8,16 and center/quarter-window offsets on coarse/medium grids",
    )
    solve.add_argument(
        "--no-reacquisition",
        action="store_true",
        help="disable selective narrow searches for missing global candidates",
    )
    solve.add_argument("--reacquisition-max-candidates", type=int, default=24)
    solve.add_argument("--verbose", action="store_true")


    reclassify = subparsers.add_parser(
        "reclassify",
        help="classify saved eigenpairs with evidence-v2 without rerunning SLEPc",
    )
    reclassify.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    reclassify.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="defaults to the artifact directory; original files are never overwritten",
    )
    reclassify.add_argument(
        "--source",
        choices=("auto", "raw", "legacy"),
        default="auto",
        help=(
            "prefer raw level_*.nc catalogs and rebuild tracks; 'legacy' preserves "
            "the old modes.nc assignments"
        ),
    )

    reacquire = subparsers.add_parser(
        "reacquire",
        help="selectively continue saved roots missing on the next radial level",
    )
    reacquire.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    reacquire.add_argument(
        "--execute",
        action="store_true",
        help="run the selected narrow SLEPc windows; otherwise only write a dry-run report",
    )
    reacquire.add_argument("--max-candidates-per-level", type=int, default=24)
    reacquire.add_argument("--nev-multiplier", type=int, default=4)
    reacquire.add_argument("--relative-half-width", type=float, default=3.0e-3)
    reacquire.add_argument("--minimum-overlap", type=float, default=0.90)
    reacquire.add_argument("--verbose", action="store_true")

    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "solve":
        _run_solve(args)
    elif args.command == "reclassify":
        _run_reclassify(args)
    elif args.command == "reacquire":
        _run_reacquire(args)
    else:
        raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
