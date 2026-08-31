"""Deprecated alias for :mod:`supernova.classify`.

Kept for one release so scripts under ``no_sync`` keep importing while they are
ported.  This binds the *same module object* rather than copying its names, so
``monkeypatch.setattr`` and ``isinstance`` behave identically through either
path.  Import from the new location instead.
"""

import sys as _sys
import warnings as _warnings

import supernova.classify as _moved

_warnings.warn(
    "supernova.core._classify_mode has moved to supernova.classify",
    DeprecationWarning,
    stacklevel=2,
)

_sys.modules[__name__] = _moved
