"""Deprecated alias for :mod:`supernova._mhd_equivalence`.

Kept for one release so scripts under ``no_sync`` keep importing while they are
ported.  This binds the *same module object* rather than copying its names, so
``monkeypatch.setattr`` and ``isinstance`` behave identically through either
path.  Import from the new location instead.
"""

import sys as _sys
import warnings as _warnings

import supernova._mhd_equivalence as _moved

_warnings.warn(
    "supernova.core._mhd_equivalence has moved to supernova._mhd_equivalence",
    DeprecationWarning,
    stacklevel=2,
)

_sys.modules[__name__] = _moved
