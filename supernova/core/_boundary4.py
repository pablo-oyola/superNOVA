"""Deprecated alias for :mod:`supernova.operators.boundary`.

Unlike the other aliases this one is a copy rather than the same module object.
``_boundary4.BoundaryCondition`` meant the *mixed* class, which the merged
module calls :class:`~supernova.operators.boundary.MixedBoundaryCondition`
while keeping the base under the original name; binding the module itself would
make that rename collide with ``_boundary3``.
"""

import warnings as _warnings

import supernova.operators.boundary as _moved

_warnings.warn(
    "supernova.core._boundary4 has moved to supernova.operators.boundary; "
    "its BoundaryCondition is now MixedBoundaryCondition",
    DeprecationWarning,
    stacklevel=2,
)

globals().update(
    {name: value for name, value in vars(_moved).items() if not name.startswith("__")}
)
BoundaryCondition = _moved.MixedBoundaryCondition
