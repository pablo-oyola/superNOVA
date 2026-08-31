"""Deprecated alias for the modules that replaced ``supernova.core.solver3``.

The four superSolverN generations are one class now, :class:`~supernova.solve.stability.StabilitySolver`; the local tensor split and the search-region helpers lives in
the modules imported below.  Unlike the other aliases in this package this
one cannot bind a single module object, because its contents were split.
Monkeypatching it will not reach the real modules -- import those directly.
"""

import warnings as _warnings

from supernova.solve.stability import StabilitySolver
from supernova.operators.local import (
    _build_local_M_split,
    _matrix_stats,
    _slice_to_quadrature,
    _select_coo_rows,
    _validate_split_block_shapes,
)
from supernova.eigen.region import (
    _as_scalar_complex,
    _format_complex,
    _ellipse_region_dict,
    _split_residual_norm,
)

_warnings.warn(
    "supernova.core.solver3 has been split; superSolverN is now "
    "supernova.solve.stability.StabilitySolver",
    DeprecationWarning,
    stacklevel=2,
)

superSolver3 = StabilitySolver
