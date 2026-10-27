"""The continuum tab's state, as plain data.

Everything the user can set lives here and nowhere else. Two reasons: a session
round-trips through JSON, so a GUI run reproduces without the GUI; and the
solve is a pure function of this object and the equilibrium, so it can be
tested without Qt.

The equilibrium itself is not here: the continuum is solved on the one the
Equilibrium tab loaded. Sessions written when this window loaded its own still
carry its inputs; :func:`split_legacy` sets them aside so they can be handed to
the Equilibrium tab instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

from supernova import defaults

# What the solver calls each branch set, and what a physicist calls it.
BRANCHES = ("slow_sound", "alfven", "full")
BRANCH_LABELS = {
    "slow_sound": "slow sound only",
    "alfven": "no sound (Alfven only)",
    "full": "full sound-Alfven",
}
COORDINATES = ("psi_n", "rho_pol")

# what the window used to load its own equilibrium from
LEGACY_EQUILIBRIUM_KEYS = ("eqdsk", "cocos_in", "boozer_min", "boozer_max",
                           "density_file", "density_units", "density_x",
                           "density_interpolate_in")


def split_legacy(data: dict) -> tuple[dict, dict]:
    """``(continuum settings, old equilibrium inputs)`` from a session dict."""
    data = dict(data)
    legacy = {key: data.pop(key) for key in LEGACY_EQUILIBRIUM_KEYS if key in data}
    return data, legacy


@dataclass
class ContinuumState:
    """Every control in the tab, and nothing else."""

    # continuum
    ntor: int = 2
    gamma_s: float = defaults.GAMMA_S
    species: str = defaults.SPECIES
    mmax: int = 0                     # 0 is auto: ceil(n q) at rho_pol = 0.95
    npsi: int = 256
    which: str = defaults.CONTINUUM_BRANCHES
    classify: bool = False            # label each root Alfvenic / sound / mixed

    # view
    fmax_khz: float = 0.0             # 0 means autoscale
    band_low_khz: float = 0.0         # 0/0 means no band drawn
    band_high_khz: float = 0.0
    cursor_khz: float = 0.0           # the draggable mode line; 0 hides it

    def validate(self) -> list[str]:
        """Return the reasons this state cannot be solved, in reading order."""
        problems = []
        if self.mmax < 0:
            problems.append("mmax must be auto (0) or at least 1.")
        if self.npsi < 8:
            problems.append("Radial points must be at least 8.")
        if self.which not in BRANCHES:
            problems.append(f"Branch must be one of {', '.join(BRANCHES)}.")
        if self.band_high_khz and self.band_low_khz >= self.band_high_khz:
            problems.append("Band must increase.")
        if self.classify and self.which != "full":
            problems.append(
                "Family labels need the full sound-Alfven spectrum; "
                "the pure branches have only one family by construction."
            )
        if self.cursor_khz < 0.0:
            problems.append("The mode frequency cannot be negative.")
        return problems

    @property
    def mmax_limit(self) -> int | None:
        """mmax as the solvers take it: None, the q-based default, when it reads auto (0)."""
        return int(self.mmax) or None

    # what the continuum object depends on, given an equilibrium
    def continuum_key(self) -> tuple:
        return (self.gamma_s, self.species)

    def solve_key(self) -> tuple:
        """Everything the solved arrays depend on, given the equilibrium.

        The band, the frequency cursor and the axis limit are not here: they
        are drawn from the result that already exists, so moving the cursor
        never re-solves.
        """
        return (self.continuum_key(), self.ntor, self.mmax, self.npsi,
                self.which, self.classify)

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def from_dict(cls, data: dict) -> "ContinuumState":
        """The settings in ``data``; an old session's equilibrium inputs are
        ignored here (see :func:`split_legacy`), anything else unknown is an error."""
        data, _ = split_legacy(data)
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown session keys: {', '.join(unknown)}")
        return cls(**data)

    @classmethod
    def from_json(cls, path) -> "ContinuumState":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        return asdict(self)
