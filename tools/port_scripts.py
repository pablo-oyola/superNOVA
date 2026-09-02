#!/usr/bin/env python3
"""Port scripts off the removed supernova.core shims and version-numbered names.

    python port_scripts.py no_sync/            # rewrite in place
    python port_scripts.py no_sync/ --dry-run  # show what would change
"""
import argparse, pathlib, re, sys

MODULES = {
    "supernova.core._solver_kernel2": "supernova.operators.blocks",
    "supernova.core._continuum_kernel": "supernova.operators.continuum_kernel",
    "supernova.core._boundary3": "supernova.operators.boundary",
    "supernova.core._boundary4": "supernova.operators.boundary",
    "supernova.core._petsc_helpers": "supernova.eigen.petsc",
    "supernova.core._classify_mode": "supernova.classify",
    "supernova.core.equilibrium": "supernova.equilibrium",
    "supernova.core._mhd_equivalence": "supernova._mhd_equivalence",
    "supernova.core.continuum": "supernova.continuum",
    "supernova.core.isotopes": "supernova.isotopes",
    "supernova.core.itpa_benchmark": "supernova.cases.itpa",
    "supernova.core.desc_helper": "supernova.interop.desc",
    "supernova.core.desc_loader": "supernova.interop._desc_loader",
    "supernova.core.solver5_multigrid_evidence": "supernova.solve.evidence",
    "supernova.core.solver5_multigrid": "supernova.solve.multigrid",
    "supernova.core.solver2": "supernova.solve.stability",
    "supernova.core.solver3": "supernova.solve.stability",
    "supernova.core.solver4": "supernova.solve.stability",
    "supernova.core.solver5": "supernova.solve.stability",
}
NAMES = {
    "superSolver2": "StabilitySolver", "superSolver3": "StabilitySolver",
    "superSolver4": "StabilitySolver", "superSolver5": "StabilitySolver",
    "superSolver5Multigrid": "MultigridSolver",
    "solver_blocks2": "SolverBlocks", "build_solver_blocks2": "build_solver_blocks",
    "EvidenceV2Config": "EvidenceConfig",
    "classify_tracks_evidence_v2": "classify_tracks",
    "_build_solver5_eigensolver_from_matrices": "build_eigensolver",
}
# names that moved out of the old solver modules
HOMES = {
    "_build_local_M_split": "supernova.operators.local",
    "_matrix_stats": "supernova.operators.local",
    "_slice_to_quadrature": "supernova.operators.local",
    "_ellipse_region_dict": "supernova.eigen.region",
    "_as_scalar_complex": "supernova.eigen.region",
    "MixedFieldLayout": "supernova.operators.mixed",
    "MixedRadialData": "supernova.operators.mixed",
    "_build_mixed_radial_data": "supernova.operators.mixed",
    "_project_local_tensor_mixed": "supernova.operators.project",
    "QuadraticPEPSolver": "supernova.eigen.pep",
    "LinearGeneralizedEPSSolver": "supernova.eigen.gevp",
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=pathlib.Path)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

changed = 0
for path in sorted(args.root.rglob("*.py")):
    if "source_snapshot" in str(path):
        continue
    text = original = path.read_text()
    for old, new in sorted(MODULES.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"(?<![\w.]){re.escape(old)}(?![\w])", new, text)
    for old, new in NAMES.items():
        text = re.sub(rf"(?<![\w.]){re.escape(old)}\b", new, text)
    if text != original:
        changed += 1
        print(f"{'would rewrite' if args.dry_run else 'rewrote'} {path}")
        for name, home in HOMES.items():
            if re.search(rf"\b{name}\b", text):
                print(f"    note: {name} now lives in {home}")
        if not args.dry_run:
            path.write_text(text)
print(f"\n{changed} files{' would be' if args.dry_run else ''} changed")
