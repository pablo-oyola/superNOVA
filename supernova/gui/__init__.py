"""Desktop front ends.

Nothing else in ``supernova`` imports PySide6, and neither does this module
until you ask for a name from it: the imports are deferred so that
``python -m supernova.gui.continuum`` does not import the module twice (which
warns), and so that merely touching ``supernova.gui`` costs nothing.

Launch with ``python -m supernova gui``.
"""

from __future__ import annotations

__all__ = ["ContinuumWindow", "main"]


def __getattr__(name: str):
    if name in __all__:
        from supernova.gui import continuum

        return getattr(continuum, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
