"""Solving the continuum for a given state, with the expensive parts cached.

No Qt here. The compute process owns one of these and calls it with the
equilibrium the Equilibrium tab loaded; everything below is a plain object so
the behaviour can be tested without a display.

The equilibrium is never loaded here: it comes in, with a key that names it
(the compute process's handle). Constructing a continuum takes about 0.1 s and
solving one 0.05-0.2 s, so the continuum is rebuilt only when the equilibrium,
gamma_s or the species changes, and the solve runs on every edit.

Radii in a result are the physical psi_N, as in the solver tab: the solver's
own coordinate spans only the Boozer domain, and the two differ whenever that
domain is not (0, 1).
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from supernova.api import Case
from supernova.defaults import default_mmax
from supernova.gui._state import ContinuumState


# family_code as the classifier writes it
FAMILY_NAMES = {-1: "invalid", 0: "Alfvenic", 1: "sound", 2: "mixed"}
FAMILY_COLOURS = {-1: "#78879D", 0: "#2B5EA6", 1: "#9C5D14", 2: "#0E7466"}


@dataclass(frozen=True)
class ContinuumResult:
    """One solved continuum, in the units the plot wants."""

    psi_n: np.ndarray
    frequency_khz: np.ndarray        # (npsi, n_branches), cyclic kHz
    which: str
    ntor: int
    gamma_s: float
    q_min: float
    q_max: float
    solve_seconds: float
    reloaded_equilibrium: bool
    rebuilt_continuum: bool
    family_code: np.ndarray | None = None    # (npsi, n_branches), or None
    classify_note: str = ""                  # why classification is absent

    @property
    def n_branches(self) -> int:
        return int(self.frequency_khz.shape[1])

    @property
    def f_max_khz(self) -> float:
        finite = self.frequency_khz[np.isfinite(self.frequency_khz)]
        return float(finite.max()) if finite.size else 0.0

    def cuts(self, frequency_khz: float) -> np.ndarray:
        """Where a mode at this frequency meets the continuum.

        Returns the radial positions, sorted. A mode sitting on a continuum
        branch is resonantly damped there, so these are the places a global
        mode at this frequency has to avoid -- which is the question the
        frequency cursor exists to answer.

        Linear interpolation between radial samples; with a few hundred points
        that is far below the physical width of the resonance.
        """
        if not np.isfinite(frequency_khz) or frequency_khz <= 0.0:
            return np.empty(0)
        psi, values = self.psi_n, self.frequency_khz
        crossings = []
        for branch in range(values.shape[1]):
            column = values[:, branch]
            good = np.isfinite(column)
            if good.sum() < 2:
                continue
            x, y = psi[good], column[good]
            offset = y - frequency_khz
            sign_change = np.signbit(offset[:-1]) != np.signbit(offset[1:])
            for index in np.flatnonzero(sign_change):
                y0, y1 = offset[index], offset[index + 1]
                if y1 == y0:
                    continue
                t = y0 / (y0 - y1)
                crossings.append(x[index] + t * (x[index + 1] - x[index]))
            crossings.extend(x[offset == 0.0].tolist())
        return np.unique(np.round(np.asarray(crossings, dtype=float), 9))


def physical_psi_n(equilibrium, x) -> np.ndarray:
    """The physical psi_N at the solver's radial coordinate ``x``."""
    magspace = equilibrium.magspace
    grid = np.asarray(magspace.coords["psi"].values, dtype=float)
    if "psi_physical_norm" not in magspace.coords:
        return np.asarray(x, dtype=float)
    return np.interp(np.asarray(x, dtype=float), grid,
                     np.asarray(magspace.coords["psi_physical_norm"].values, dtype=float))


def _harmonic_band(q_min: float, q_max: float, ntor: int, mmax: int):
    """The poloidal band solve() will use, exactly as it computes it."""
    lowest = int(np.floor(ntor * q_min)) - 3
    highest = min(int(mmax), int(np.ceil(ntor * q_max)) + 3)
    return lowest, highest


def classification_feasibility(case, ntor: int, mmax: int | None) -> tuple[bool, str]:
    """Can the family classifier run for this harmonic band, and if not, why?

    The operators are convolutions: an entry between state harmonics m and m'
    reads the equilibrium harmonic at m - m'. So what matters is the span of
    the band, never how far it sits from m = 0, and the only way to fall short
    is an equilibrium whose profiles carry fewer harmonics than the span.
    """
    q = np.abs(np.asarray(case.equilibrium.qprof.data).squeeze().real)
    q_min, q_max = float(np.nanmin(q)), float(np.nanmax(q))
    available = int(np.abs(case.continuum.H.m_pol).max())
    if mmax is None:
        # the solvers' own default; for n = 0, the equilibrium's harmonic count
        mmax = default_mmax(case.equilibrium, ntor) or available

    lowest, highest = _harmonic_band(q_min, q_max, abs(int(ntor)), int(mmax))
    if highest < lowest:
        return False, "The harmonic band is empty; raise mmax."

    span = highest - lowest
    if available >= span:
        return True, ""

    return False, (
        f"Family labels are unavailable for n = {ntor}. The band spans "
        f"m = {lowest} to {highest}, so the operators need equilibrium "
        f"harmonics up to a mode difference of {span}, and this equilibrium "
        f"carries only {available}.\n\nRebuild it with m_max at least "
        f"{span}, or narrow the band with a smaller mmax. The branches are "
        f"still correct -- only the Alfvenic / sound / mixed labels are "
        f"missing."
    )


class ContinuumModel:
    """Holds the continuum operators for one equilibrium and solves states."""

    def __init__(self):
        self.case: Case | None = None
        self._equilibrium_key = None
        self._continuum_key = None

    def solve(self, state: ContinuumState, equilibrium, key, *,
              progress=None) -> ContinuumResult:
        """Solve ``state`` on ``equilibrium``, reusing whatever is still valid.

        ``key`` names the equilibrium; a different key means a different one.
        ``progress(message)`` is called before each slow step.
        """
        from supernova.continuum import superContinuum

        problems = state.validate()
        if problems:
            raise ValueError(problems[0])

        reloaded = key != self._equilibrium_key
        rebuilt = reloaded or state.continuum_key() != self._continuum_key
        if rebuilt:
            if progress:
                progress("Building the continuum operators...")
            continuum = superContinuum(equilibrium, gamma_s=float(state.gamma_s),
                                       species=state.species, dense=True)
            self.case = Case(equilibrium, continuum,
                             metadata={"gamma_s": float(state.gamma_s),
                                       "species": state.species})
            self._equilibrium_key = key
            self._continuum_key = state.continuum_key()

        x = np.linspace(0.0, 1.0, int(state.npsi))
        family_code = None
        classify_note = ""
        started = time.perf_counter()

        if state.classify and state.which == "full":
            feasible, why = classification_feasibility(
                self.case, int(state.ntor), state.mmax_limit)
            if not feasible:
                # Predicted from the harmonic band, so the expensive classified
                # solve is skipped rather than run and thrown away.
                classify_note = why
                if progress:
                    progress("Family labels unavailable here; solving without them...")
                eigenvalues = self.case.continuum.solve(
                    ntor=int(state.ntor), which="full", normalize_eigvals=True,
                    psi_vals=x, classify=False, mmax=state.mmax_limit,
                )
                elapsed = time.perf_counter() - started
                return self._pack(state, x, eigenvalues, elapsed, reloaded,
                                  rebuilt, None, classify_note)
            if progress:
                progress("Solving and classifying families (slower)...")
            try:
                eigenvalues, _vectors, classifier = self.case.continuum.solve(
                    ntor=int(state.ntor), which="full", normalize_eigvals=True,
                    psi_vals=x, mmax=state.mmax_limit,
                    return_eigvecs=True, classify=True,
                )
                family_code = np.asarray(
                    classifier.to_xarray()["family_code"], dtype=np.int8
                )
            except ValueError as error:
                # A backstop: the check above should have caught this, so a
                # failure here means the prediction is wrong about something.
                classify_note = (
                    "Family labels failed unexpectedly -- the harmonic band "
                    f"looked wide enough. Please report this. ({error})"
                )
                if progress:
                    progress("Classification unavailable; solving without it...")
                eigenvalues = self.case.continuum.solve(
                    ntor=int(state.ntor), which="full", normalize_eigvals=True,
                    psi_vals=x, classify=False, mmax=state.mmax_limit,
                )
        else:
            if progress:
                progress("Solving...")
            eigenvalues = self.case.continuum.solve(
                ntor=int(state.ntor),
                which=state.which,
                normalize_eigvals=True,
                psi_vals=x,
                classify=False,
                mmax=state.mmax_limit,
            )
        elapsed = time.perf_counter() - started

        return self._pack(state, x, eigenvalues, elapsed, reloaded,
                          rebuilt, family_code, classify_note)

    def _pack(self, state, x, eigenvalues, elapsed, reloaded, rebuilt,
              family_code, classify_note) -> ContinuumResult:
        """Turn solver output into the result the plot wants."""
        # eigenvalues are omega**2 normalised to the on-axis Alfven frequency
        with np.errstate(invalid="ignore"):
            khz = np.sqrt(eigenvalues.real).to("kHz").v / (2.0 * np.pi)
        q = np.asarray(self.case.equilibrium.qprof.data).squeeze().real

        return ContinuumResult(
            psi_n=physical_psi_n(self.case.equilibrium, x),
            frequency_khz=np.asarray(khz, dtype=float),
            which=state.which,
            ntor=int(state.ntor),
            gamma_s=float(state.gamma_s),
            q_min=float(np.nanmin(np.abs(q))),
            q_max=float(np.nanmax(np.abs(q))),
            solve_seconds=elapsed,
            reloaded_equilibrium=reloaded,
            rebuilt_continuum=rebuilt,
            family_code=family_code,
            classify_note=classify_note,
        )

    def to_dataset(self, result: ContinuumResult):
        """The result as an xarray Dataset, ready to write."""
        import xarray as xr

        data = {"frequency_khz": (("psi_n", "branch"), result.frequency_khz)}
        if result.family_code is not None:
            # int codes rather than strings: object-dtype arrays do not survive
            # a round trip through HDF5, and the mapping is in the attributes.
            data["family_code"] = (("psi_n", "branch"), result.family_code)

        dataset = xr.Dataset(
            data,
            coords={"psi_n": result.psi_n,
                    "branch": np.arange(result.n_branches)},
            attrs={
                "which": result.which,
                "ntor": result.ntor,
                "gamma_s": result.gamma_s,
                "q_min": result.q_min,
                "q_max": result.q_max,
                "frequency_definition": "cyclic kHz, sqrt(Re omega**2)/(2 pi)",
                "source": "supernova.gui.continuum",
            },
        )
        if result.family_code is not None:
            dataset["family_code"].attrs["labels"] = ", ".join(
                f"{code} = {name}" for code, name in sorted(FAMILY_NAMES.items())
            )
        return dataset

    def save(self, result: ContinuumResult, path) -> str:
        """Write the result, choosing the format from the file extension.

        ``.h5`` / ``.hdf5`` go through h5netcdf, which writes HDF5 that h5py
        and any netCDF4 reader can both open; ``.nc`` uses the netCDF4 engine.
        Returns the engine used, so the caller can say so.
        """
        from pathlib import Path

        path = Path(path)
        engine = "h5netcdf" if path.suffix.lower() in (".h5", ".hdf5") else "netcdf4"
        self.to_dataset(result).to_netcdf(path, engine=engine)
        return engine
