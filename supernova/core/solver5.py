"""Deprecated alias for the modules that replaced ``supernova.core.solver5``.

The four superSolverN generations are one class now, :class:`~supernova.solve.stability.StabilitySolver`; the PEP and GEVP backends lives in
the modules imported below.  Unlike the other aliases in this package this
one cannot bind a single module object, because its contents were split.
Monkeypatching it will not reach the real modules -- import those directly.
"""

import warnings as _warnings

from supernova.solve.stability import StabilitySolver
from supernova.eigen.pep import (
    QuadraticPEPSolver,
    _polynomial_residual_norm,
    _polynomial_backward_error,
    _validate_quadratic_coefficient,
)
from supernova.eigen.gevp import (
    LinearGeneralizedEPSSolver,
    _linear_residuals,
    _validate_gamma_zero_pencil,
    _canonical_linear_coefficient,
)
from supernova.eigen.dispatch import _build_solver5_eigensolver_from_matrices

_warnings.warn(
    "supernova.core.solver5 has been split; superSolverN is now "
    "supernova.solve.stability.StabilitySolver",
    DeprecationWarning,
    stacklevel=2,
)

superSolver5 = StabilitySolver
