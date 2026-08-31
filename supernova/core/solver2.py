"""Deprecated alias for the modules that replaced ``supernova.core.solver2``.

The four superSolverN generations are one class now, :class:`~supernova.solve.stability.StabilitySolver`; the base construction lives in
the modules imported below.  Unlike the other aliases in this package this
one cannot bind a single module object, because its contents were split.
Monkeypatching it will not reach the real modules -- import those directly.
"""

import warnings as _warnings

from supernova.solve.stability import StabilitySolver

_warnings.warn(
    "supernova.core.solver2 has been split; superSolverN is now "
    "supernova.solve.stability.StabilitySolver",
    DeprecationWarning,
    stacklevel=2,
)

superSolver2 = StabilitySolver
