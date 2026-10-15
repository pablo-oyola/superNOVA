"""Desktop front ends.

Nothing else in ``supernova`` imports PySide6, and neither does this module
until you ask for a name from it: the imports are deferred so that
``python -m supernova.gui.continuum`` does not import the module twice (which
warns), and so that merely touching ``supernova.gui`` costs nothing.

Launch with ``python -m supernova gui``: one window with an equilibrium, a
continuum and a solver tab (:mod:`supernova.gui.app`). Each tab also opens on
its own with ``python -m supernova.gui.<equilibrium|continuum|solver>``.
"""

from __future__ import annotations

_HOMES = {
    "SupernovaWindow": "app",
    "main": "app",
    "EquilibriumWindow": "equilibrium",
    "ContinuumWindow": "continuum",
    "SolverWindow": "solver",
}

__all__ = sorted(_HOMES)


def __getattr__(name: str):
    if name in _HOMES:
        from importlib import import_module

        return getattr(import_module(f"supernova.gui.{_HOMES[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
