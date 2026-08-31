"""Deprecated alias for :mod:`supernova.operators.blocks`.

Kept for one release so scripts under ``no_sync`` keep importing while they are
ported.  This binds the *same module object* rather than copying its names, so
``monkeypatch.setattr`` and ``isinstance`` behave identically through either
path.  Import from the new location instead.
"""

import sys as _sys
import warnings as _warnings

import supernova.operators.blocks as _moved

_warnings.warn(
    "supernova.core._solver_kernel2 has moved to supernova.operators.blocks",
    DeprecationWarning,
    stacklevel=2,
)

_sys.modules[__name__] = _moved
