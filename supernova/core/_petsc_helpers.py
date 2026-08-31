"""Deprecated alias for :mod:`supernova.eigen.petsc`.

Kept for one release so scripts under ``no_sync`` keep importing while they are
ported.  This binds the *same module object* rather than copying its names, so
``monkeypatch.setattr`` and ``isinstance`` behave identically through either
path.  Import from the new location instead.
"""

import sys as _sys
import warnings as _warnings

import supernova.eigen.petsc as _moved

_warnings.warn(
    "supernova.core._petsc_helpers has moved to supernova.eigen.petsc",
    DeprecationWarning,
    stacklevel=2,
)

_sys.modules[__name__] = _moved
