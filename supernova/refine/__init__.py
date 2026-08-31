"""Fixed-boundary TokaMaker refinement of tabulated equilibria.

Importing this subpackage requires OpenFUSIONToolkit; importing ``supernova``
does not.  Refinement is a separate pre-processing step that hands the solver
a refined eqdsk plus a diagnostics sidecar, so the two never need to share an
environment.

    from supernova.refine import RefinementConfig, refine_equilibrium

    refine_equilibrium(RefinementConfig(
        source_eqdsk=Path("no_sync/MAST/MAST53678.03200.eqdsk"),
        output_dir=Path("no_sync/MAST/tokamaker_refined"),
        stem="MAST53678.03200_tokamaker",
        source_cocos=5,
        run_info="MAST 53678 at 3200 ms: fixed-boundary TokaMaker refinement",
    ))
"""

from .fixed_boundary import (
    DEFAULT_FIT,
    RefinementConfig,
    load_diagnostics,
    refine_equilibrium,
    repair_fixed_boundary_export,
)
from .qfit import (
    EquilibriumSamples,
    FixedBoundaryQFitter,
    QFitConfig,
    QFitIteration,
    QFitResult,
)

__all__ = [
    "DEFAULT_FIT",
    "EquilibriumSamples",
    "FixedBoundaryQFitter",
    "QFitConfig",
    "QFitIteration",
    "QFitResult",
    "RefinementConfig",
    "load_diagnostics",
    "refine_equilibrium",
    "repair_fixed_boundary_export",
]
