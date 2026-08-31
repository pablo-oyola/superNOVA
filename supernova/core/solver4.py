"""Deprecated alias for the modules that replaced ``supernova.core.solver4``.

The four superSolverN generations are one class now, :class:`~supernova.solve.stability.StabilitySolver`; the compatible mixed spaces and their Galerkin projection lives in
the modules imported below.  Unlike the other aliases in this package this
one cannot bind a single module object, because its contents were split.
Monkeypatching it will not reach the real modules -- import those directly.
"""

import warnings as _warnings

from supernova.solve.stability import StabilitySolver
from supernova.operators.mixed import (
    MixedFieldLayout,
    MixedRadialData,
    _build_mixed_radial_data,
)
from supernova.operators.project import _project_local_tensor_mixed

_warnings.warn(
    "supernova.core.solver4 has been split; superSolverN is now "
    "supernova.solve.stability.StabilitySolver",
    DeprecationWarning,
    stacklevel=2,
)

superSolver4 = StabilitySolver
