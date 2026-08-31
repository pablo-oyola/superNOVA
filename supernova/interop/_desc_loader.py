"""
DESC auxiliary functions to load the module.
"""

try:
    import desc
except ImportError:
    raise ImportError(
        "DESC is not installed. Please install DESC to use this module."
    )

from desc.geometry import FourierRZToroidalSurface
from desc.profiles import SplineProfile
from desc.equilibrium import Equilibrium as DESC_Equilibrium
from desc.continuation import solve_continuation_automatic
from desc.grid import Grid, LinearGrid

__all__ = ["FourierRZToroidalSurface", "SplineProfile", 
           "DESC_Equilibrium", "solve_continuation_automatic",
           "Grid", "LinearGrid"]