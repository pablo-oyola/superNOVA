"""superNOVA: ideal-MHD stability of tokamak equilibria."""

__version__ = "0.1.0"

import pycocos

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
from supernova.classify import ContinuumClassifier
from supernova.continuum import superContinuum
from supernova.equilibrium import superEquilibrium
from supernova.radial.base import RadialFE
from supernova.radial.bspline import BSplineFE
from supernova.radial.profile import Profile
from supernova.solve.evidence import (
    AdaptiveDiscoveryOptions,
    ContinuumQuery,
    MeshPerturbationOptions,
    ReacquisitionOptions,
    classify_tracks,
    deterministic_shifted_bspline,
    mesh_perturbation_evidence,
    migrate_legacy_mode_dataset,
)
from supernova.solve.modelocate import ModeLocateThresholds, locate_modes
from supernova.solve.multigrid import (
    EvidenceConfig,
    ModeClassificationThresholds,
    MultigridPEPResult,
    MultigridSearchOptions,
    MultigridSolver,
)
from supernova.solve.stability import StabilitySolver
