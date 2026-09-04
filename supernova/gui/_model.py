"""Solving the continuum for a given state, with the expensive parts cached.

No Qt here. The window owns one of these and calls it; everything below is a
plain object so the behaviour can be tested without a display.

The cost structure is what shapes it: loading an equilibrium takes seconds for
a real eqdsk, constructing a continuum takes about 0.1 s, and solving one takes
0.05-0.2 s. So the equilibrium is rebuilt only when its inputs change, the
continuum only when gamma_s or the species changes, and the solve runs on every
edit.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from supernova.api import Case, profile_from_columns
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


class ContinuumModel:
    """Holds the loaded Case and turns a state into a result."""

    def __init__(self):
        self.case: Case | None = None
        self._equilibrium_key = None
        self._continuum_key = None

    def solve(self, state: ContinuumState, *, progress=None) -> ContinuumResult:
        """Solve for ``state``, reusing whatever is still valid.

        ``progress(message)`` is called before each slow step so a caller can
        say what it is waiting for.
        """
        problems = state.validate()
        if problems:
            raise ValueError(problems[0])

        reloaded = state.equilibrium_key() != self._equilibrium_key
        if reloaded:
            if progress:
                progress("Loading equilibrium and computing Boozer coordinates...")
            density = profile_from_columns(
                state.density_file,
                units=state.density_units,
                x=state.density_x,
                interpolate_in=state.density_interpolate_in,
            )
            self.case = Case.from_eqdsk(
                state.eqdsk,
                cocos_in=state.cocos_in,
                density=density,
                pressure="eqdsk",
                boozer_domain=(state.boozer_min, state.boozer_max),
                gamma_s=state.gamma_s,
                species=state.species,
            )
            self._equilibrium_key = state.equilibrium_key()
            self._continuum_key = state.continuum_key()

        rebuilt = state.continuum_key() != self._continuum_key
        if rebuilt:
            if progress:
                progress("Rebuilding the continuum operators...")
            self.case = self.case.with_continuum(
                gamma_s=state.gamma_s, species=state.species
            )
            self._continuum_key = state.continuum_key()

        psi_n = np.linspace(0.0, 1.0, int(state.npsi))
        family_code = None
        classify_note = ""
        started = time.perf_counter()

        if state.classify and state.which == "full":
            if progress:
                progress("Solving and classifying families (slower)...")
            try:
                eigenvalues, _vectors, classifier = self.case.continuum.solve(
                    ntor=int(state.ntor), which="full", normalize_eigvals=True,
                    psi_vals=psi_n, mmax=int(state.mmax),
                    return_eigvecs=True, classify=True,
                )
                family_code = np.asarray(
                    classifier.to_xarray()["family_code"], dtype=np.int8
                )
            except ValueError as error:
                # The classifier convolves the equilibrium's poloidal harmonics
                # into a metric matrix, so it needs at least max|m| + 1 of them.
                # A wide harmonic band -- which is what a large n asks for --
                # outruns what the profiles carry, and the shape mismatch that
                # results says nothing useful on its own.
                classify_note = (
                    "Family labels unavailable at this n: the equilibrium "
                    "profiles do not carry enough poloidal harmonics for the "
                    f"selected band. Lower n or mmax, or rebuild the "
                    f"equilibrium with a larger m_max. ({error})"
                )
                if progress:
                    progress("Classification unavailable; solving without it...")
                eigenvalues = self.case.continuum.solve(
                    ntor=int(state.ntor), which="full", normalize_eigvals=True,
                    psi_vals=psi_n, classify=False, mmax=int(state.mmax),
                )
        else:
            if progress:
                progress("Solving...")
            eigenvalues = self.case.continuum.solve(
                ntor=int(state.ntor),
                which=state.which,
                normalize_eigvals=True,
                psi_vals=psi_n,
                classify=False,
                mmax=int(state.mmax),
            )
        elapsed = time.perf_counter() - started

        # eigenvalues are omega**2 normalised to the on-axis Alfven frequency
        khz = np.sqrt(eigenvalues.real).to("kHz").v / (2.0 * np.pi)
        q = np.asarray(self.case.equilibrium.qprof.data).squeeze().real

        return ContinuumResult(
            psi_n=psi_n,
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
