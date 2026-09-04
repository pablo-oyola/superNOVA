"""The window's state, as plain data.

Everything the user can set lives here and nowhere else. Two reasons: a session
round-trips through JSON, so a GUI run reproduces without the GUI; and the
solve is a pure function of this object, so it can be tested without Qt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

# What the solver calls each branch set, and what a physicist calls it.
BRANCHES = ("slow_sound", "alfven", "full")
BRANCH_LABELS = {
    "slow_sound": "slow sound only",
    "alfven": "no sound (Alfven only)",
    "full": "full sound-Alfven",
}
COORDINATES = ("psi_n", "rho_pol")


@dataclass
class ContinuumState:
    """Every control in the window, and nothing else."""

    # equilibrium
    eqdsk: str = ""
    cocos_in: int = 5
    boozer_min: float = 0.005
    boozer_max: float = 0.995

    # profiles
    density_file: str = ""
    density_units: str = "m^-3"
    density_x: str = "psi_n"          # what the file's first column is
    density_interpolate_in: str = "rho_pol"

    # continuum
    ntor: int = 2
    gamma_s: float = 5.0 / 3.0
    species: str = "D"
    mmax: int = 40
    npsi: int = 256
    which: str = "slow_sound"
    classify: bool = False            # label each root Alfvenic / sound / mixed

    # view
    fmax_khz: float = 0.0             # 0 means autoscale
    band_low_khz: float = 0.0         # 0/0 means no band drawn
    band_high_khz: float = 0.0
    cursor_khz: float = 0.0           # the draggable mode line; 0 hides it

    def validate(self) -> list[str]:
        """Return the reasons this state cannot be solved, in reading order."""
        problems = []
        if not self.eqdsk:
            problems.append("Choose an equilibrium file.")
        elif not Path(self.eqdsk).is_file():
            problems.append(f"Equilibrium not found: {self.eqdsk}")
        if not self.density_file:
            problems.append("Choose a density profile file.")
        elif not Path(self.density_file).is_file():
            problems.append(f"Density file not found: {self.density_file}")
        if not 0.0 <= self.boozer_min < self.boozer_max <= 1.0:
            problems.append("Boozer domain must satisfy 0 <= min < max <= 1.")
        if self.mmax < 1:
            problems.append("mmax must be at least 1.")
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

    # what the equilibrium depends on; a change here means a slow reload
    def equilibrium_key(self) -> tuple:
        return (self.eqdsk, self.cocos_in, self.boozer_min, self.boozer_max,
                self.density_file, self.density_units, self.density_x,
                self.density_interpolate_in)

    # what the continuum object depends on, given an equilibrium
    def continuum_key(self) -> tuple:
        return (self.gamma_s, self.species)

    def solve_key(self) -> tuple:
        """Everything the solved arrays depend on.

        The band, the frequency cursor and the axis limit are not here: they
        are drawn from the result that already exists, so moving the cursor
        never re-solves.
        """
        return (self.equilibrium_key(), self.continuum_key(), self.ntor,
                self.mmax, self.npsi, self.which, self.classify)

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def from_json(cls, path) -> "ContinuumState":
        data = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown session keys: {', '.join(unknown)}")
        return cls(**data)
