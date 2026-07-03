__version__ = "0.1.0"

import pycocos
from .core.equilibrium import superEquilibrium
from .core.continuum import superContinuum
from .core._classify_mode import ContinuumClassifier
from .core.solver import superSolver
from .core.solver2 import superSolver2
from .core.solver3 import superSolver3
from .radial.profile import Profile
from .radial.base import RadialFE
from .radial.chebyshev import ChebyshevFE
from .radial.bspline import BSplineFE
