"""Round-tripping the tabs' states through JSON.

Each tab keeps everything the user can set in one dataclass, so a session is
just those dataclasses written side by side. A session saved from the old
single-window continuum tool is a bare :class:`~supernova.gui._state.
ContinuumState`; it still loads, into the continuum tab.
"""

from __future__ import annotations

from dataclasses import asdict, fields
import json
from pathlib import Path


class JsonState:
    """Mixin for a state dataclass: dict and JSON round trips, strict on keys."""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown session keys: {', '.join(unknown)}")
        return cls(**data)

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def copy(self):
        return type(self).from_dict(json.loads(json.dumps(self.to_dict())))


# the tab each section of an application session belongs to
SECTIONS = ("equilibrium", "continuum", "solver")


def write_session(path, **states) -> None:
    """Write one JSON file holding every tab's state."""
    unknown = sorted(set(states) - set(SECTIONS))
    if unknown:
        raise ValueError(f"Unknown session sections: {', '.join(unknown)}")
    data = {name: asdict(state) for name, state in states.items()}
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def read_session(path) -> dict:
    """Return ``{section: state}`` for whatever the file holds.

    A file whose keys are not section names is taken to be a continuum-only
    session from the single-window tool.
    """
    from supernova.gui._equilibrium import EquilibriumState
    from supernova.gui._solver import SolverState
    from supernova.gui._state import ContinuumState

    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a session.")
    if not set(data) & set(SECTIONS):
        return {"continuum": ContinuumState(**_known(ContinuumState, data))}

    unknown = sorted(set(data) - set(SECTIONS))
    if unknown:
        raise ValueError(f"Unknown session sections: {', '.join(unknown)}")
    loaded = {}
    if "equilibrium" in data:
        loaded["equilibrium"] = EquilibriumState.from_dict(data["equilibrium"])
    if "continuum" in data:
        loaded["continuum"] = ContinuumState(
            **_known(ContinuumState, data["continuum"]))
    if "solver" in data:
        loaded["solver"] = SolverState.from_dict(data["solver"])
    return loaded


def _known(cls, data: dict) -> dict:
    """ContinuumState is strict on keys in its own loader; match it here."""
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"Unknown session keys: {', '.join(unknown)}")
    return data
