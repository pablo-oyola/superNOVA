"""superNOVA: ideal-MHD stability of tokamak equilibria."""

__version__ = "0.1.0"

import pycocos

from supernova.equilibrium import superEquilibrium
from supernova.continuum import superContinuum
from supernova.classify import ContinuumClassifier
from supernova.solve.stability import (
    StabilitySolver,
    superSolver2,
    superSolver3,
    superSolver4,
    superSolver5,
)
from supernova.solve.multigrid import (
    EvidenceV2Config,
    ModeClassificationThresholds,
    MultigridPEPResult,
    MultigridSearchOptions,
    superSolver5Multigrid,
)
from supernova.solve.evidence import (
    AdaptiveDiscoveryOptions,
    ContinuumQuery,
    MeshPerturbationOptions,
    ReacquisitionOptions,
    classify_tracks_evidence_v2,
    deterministic_shifted_bspline,
    migrate_legacy_mode_dataset,
    mesh_perturbation_evidence,
)
from supernova.api import (
    Band,
    Case,
    ModeSet,
    PetscOptions,
    RadialGrid,
    Scan,
    profile_from_columns,
    profile_from_eqdsk,
)
from supernova.radial.profile import Profile
from supernova.radial.base import RadialFE
from supernova.radial.bspline import BSplineFE
