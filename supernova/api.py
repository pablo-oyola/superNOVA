r"""The user-facing surface.

Everything a case script needs, without reaching into private methods.  The
four things every script under ``no_sync`` used to re-implement -- converting a
kHz band into a complex search region, building profiles from an eqdsk and a
two-column file, reconstructing eigenmodes from finite-element coefficients,
and computing mode centroids and widths -- live here once.

    import supernova as sn

    case = sn.Case.from_eqdsk(
        "MAST53678.03200.eqdsk", cocos_in=5,
        density=sn.profile_from_columns("ne.dat", units="m^-3"),
        pressure="eqdsk", boozer_domain=(0.005, 0.995),
        gamma_s=5 / 3, species="D",
    )
    scan = sn.Scan(case, sn.RadialGrid.uniform(spans=150), ntor=2, mmax=80, max_dm=5)
    modes = scan.run(sn.Band(50, 150, units="kHz"), window=1.0, nev=4)

    modes.frequency_khz
    modes.field("xi_psi", n=512)
    modes.centroid, modes.width
    modes.gap_modes()          # GJK's modelocate filter; .locate() for all

The low-level objects stay reachable: :class:`~supernova.solve.stability.
StabilitySolver` and :class:`~supernova.solve.multigrid.MultigridSolver`
take the same arguments they always did, and every result here wraps an
``xarray.Dataset`` that is available as ``.dataset``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import unyt
import xarray as xr

from supernova.continuum import superContinuum
from supernova.equilibrium import superEquilibrium
from supernova.radial.bspline import BSplineFE
from supernova.radial.profile import Profile
from supernova.solve.modelocate import (
    GAP,
    ModeLocateThresholds,
    branch_crossings,
    locate_modes,
)
from supernova.solve.multigrid import _solve_master_continuum
from supernova.solve.stability import StabilitySolver


# --------------------------------------------------------------------------
# PETSc option presets
# --------------------------------------------------------------------------

_FACTORIZATIONS = {
    "mumps": {
        "st_ksp_type": "preonly",
        "st_pc_type": "lu",
        "st_pc_factor_mat_solver_type": "mumps",
    },
    "spqr": {
        "st_ksp_type": "preonly",
        "st_pc_type": "qr",
        "st_pc_factor_mat_solver_type": "spqr",
    },
    "petsc_qr": {
        "st_ksp_type": "preonly",
        "st_pc_type": "qr",
    },
}


@dataclass(frozen=True)
class PetscOptions:
    """The shift-and-invert factorization, by name instead of by dict.

    ``mumps`` is the production choice: an LU that copes with the indefinite
    shifted operator.  ``spqr`` and ``petsc_qr`` are the QR alternatives worth
    trying when MUMPS runs out of memory or reports a zero pivot.
    """

    factorization: str = "mumps"
    extra: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.factorization not in _FACTORIZATIONS:
            raise ValueError(
                f"Unknown factorization {self.factorization!r}. "
                f"Choose one of {sorted(_FACTORIZATIONS)}."
            )

    def as_dict(self) -> dict[str, str]:
        options = dict(_FACTORIZATIONS[self.factorization])
        options.update(self.extra)
        return options


# --------------------------------------------------------------------------
# Frequency bands
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Band:
    """A frequency band, and the one place unit conversion happens.

    The eigensolvers work in ``s = omega**2`` with ``omega`` normalized to the
    on-axis Alfven frequency, while a physicist thinks in kHz.  Every case
    script used to carry its own ``freq_to_elipse`` for this; the conversion
    needs the equilibrium's unit registry, so it is deferred until a
    :class:`Case` is attached with :meth:`for_case`.
    """

    low: float
    high: float
    units: str = "kHz"
    vscale: float = 0.25
    _case: "Case | None" = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not np.isfinite([self.low, self.high]).all():
            raise ValueError("Band edges must be finite.")
        if self.low >= self.high:
            raise ValueError(f"Band must increase: got ({self.low}, {self.high}).")
        if not self.vscale > 0.0:
            raise ValueError("vscale must be positive.")

    def for_case(self, case: "Case") -> "Band":
        """Bind the equilibrium whose Alfven frequency sets the normalization."""
        return Band(self.low, self.high, self.units, self.vscale, case)

    def _require_case(self) -> "Case":
        if self._case is None:
            raise ValueError(
                "This Band is not attached to a Case, so it cannot be converted "
                "to normalized omega. Call band.for_case(case) first; Scan.run "
                "does that for you."
            )
        return self._case

    def to_omega(self, value=None) -> np.ndarray:
        """Convert a frequency in ``self.units`` to normalized angular omega."""
        case = self._require_case()
        values = np.atleast_1d(self.edges if value is None else value)
        frequency = unyt.unyt_array(
            np.asarray(values, dtype=float), self.units, registry=case.registry
        )
        return 2.0 * np.pi * np.asarray(frequency.to("omegaA").v, dtype=float)

    def to_frequency(self, omega) -> np.ndarray:
        """Convert normalized angular omega back to ``self.units``."""
        case = self._require_case()
        angular = unyt.unyt_array(
            np.asarray(omega, dtype=float) / (2.0 * np.pi),
            "omegaA",
            registry=case.registry,
        )
        return np.asarray(angular.to(self.units).v, dtype=float)

    @property
    def edges(self) -> tuple[float, float]:
        return (float(self.low), float(self.high))

    @property
    def omega_range(self) -> tuple[float, float]:
        low, high = self.to_omega()
        return (float(low), float(high))

    def region(self, low=None, high=None) -> tuple[complex, dict]:
        """Return ``(target_s, region)`` for one window, in ``s = omega**2``."""
        omega_low, omega_high = self.to_omega(
            (self.low if low is None else low, self.high if high is None else high)
        )
        s_low, s_high = omega_low**2, omega_high**2
        center = 0.5 * (s_low + s_high)
        radius = 0.5 * (s_high - s_low)
        return complex(np.sqrt(center)), {
            "type": "ellipse",
            "center": center,
            "radius": radius,
            "vscale": self.vscale,
        }

    def windows(self, width: float, overlap: float = 0.0) -> list[tuple[float, float]]:
        """Split the band into windows of ``width`` in ``self.units``."""
        if not width > 0.0:
            raise ValueError("Window width must be positive.")
        if not 0.0 <= overlap < 1.0:
            raise ValueError("Overlap must satisfy 0 <= overlap < 1.")
        step = width * (1.0 - overlap)
        count = max(1, int(np.ceil((self.high - self.low) / step)))
        return [
            (self.low + index * step, min(self.low + index * step + width, self.high))
            for index in range(count)
            if self.low + index * step < self.high
        ]


# --------------------------------------------------------------------------
# Radial discretization
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RadialGrid:
    """The radial finite-element space, described by what a user chooses."""

    knots: np.ndarray
    order: int = 3
    quad_order: int = 6
    domain: tuple[float, float] = (0.0, 1.0)

    @classmethod
    def uniform(cls, spans: int, order: int = 3, quad_order: int = 6,
                domain: tuple[float, float] = (0.0, 1.0)) -> "RadialGrid":
        """``spans`` equal elements over ``domain``."""
        if spans < 1:
            raise ValueError("spans must be at least 1.")
        return cls(np.linspace(domain[0], domain[1], spans + 1),
                   order, quad_order, domain)

    @classmethod
    def graded(cls, spans: int, centers, local, *, growth: float = 0.15,
               order: int = 3, quad_order: int = 6,
               domain: tuple[float, float] = (0.0, 1.0)) -> "RadialGrid":
        """``spans`` equal elements, refined to ``local`` around ``centers``.

        See :func:`supernova.radial.grading.graded_breakpoints`.
        """
        from supernova.radial.grading import graded_breakpoints

        base = (domain[1] - domain[0]) / int(spans)
        return cls(graded_breakpoints(domain, base, centers, local, growth=growth),
                   order, quad_order, domain)

    def build(self) -> BSplineFE:
        return BSplineFE(knots=np.asarray(self.knots, dtype=float),
                         order=self.order, domain=self.domain,
                         quad_order=self.quad_order)

    @property
    def spans(self) -> int:
        return int(np.asarray(self.knots).size - 1)


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

def profile_from_columns(path, *, units: str, x: str = "psi_n",
                         interpolate_in: str = "rho_pol", skiprows: int = 1,
                         usecols=(0, 1), points: int = 128,
                         name: str | None = None) -> Profile:
    r"""Read a two-column profile file onto the solver's ``psi_N`` grid.

    Two independent choices, both of which the case scripts made implicitly:

    ``x``
        what the file's first column *is* -- ``"psi_n"`` (normalized poloidal
        flux, the default) or ``"rho_pol"`` (its square root).
    ``interpolate_in``
        the variable the piecewise-linear interpolation runs in.  These give
        genuinely different profiles for the same data, and the existing case
        scripts interpolate in ``rho_pol``:

            np.interp(sqrt(psi_target), sqrt(psi_file), values)

        so that is the default, and this function reproduces them exactly.
    """
    if x not in ("psi_n", "rho_pol"):
        raise ValueError(f"x must be 'psi_n' or 'rho_pol', got {x!r}.")
    if interpolate_in not in ("psi_n", "rho_pol"):
        raise ValueError(
            f"interpolate_in must be 'psi_n' or 'rho_pol', got {interpolate_in!r}."
        )

    raw_x, raw_y = np.loadtxt(str(path), unpack=True, usecols=usecols,
                              skiprows=skiprows)
    file_psi_n = np.asarray(raw_x, dtype=float)
    if x == "rho_pol":
        file_psi_n = file_psi_n**2

    psi_n = np.linspace(0.0, 1.0, points)
    if interpolate_in == "rho_pol":
        sample, reference = np.sqrt(psi_n), np.sqrt(file_psi_n)
    else:
        sample, reference = psi_n, file_psi_n

    values = np.interp(sample, reference, np.asarray(raw_y, dtype=float))
    return Profile(x=psi_n, F_nm=values,
                   attrs={"units": units, "name": name or Path(path).stem})


def profile_from_eqdsk(eqdsk, quantity: str, *, units: str,
                       points: int = 128) -> Profile:
    """Take a Grad-Shafranov profile from the eqdsk onto the psi_N grid."""
    psi_n = np.linspace(0.0, 1.0, points)
    rhopol = np.asarray(eqdsk.gs_profs["rhop"], dtype=float)
    values = np.interp(np.sqrt(psi_n), rhopol,
                       np.asarray(eqdsk.gs_profs[quantity], dtype=float))
    return Profile(x=psi_n, F_nm=values, attrs={"units": units, "name": quantity})


@dataclass
class Case:
    """An equilibrium with its profiles and its continuum.

    Built once and reused by every scan: the Boozer transform and the continuum
    matrices are the expensive part, and neither depends on the frequency band.
    """

    equilibrium: superEquilibrium
    continuum: superContinuum
    source: Path | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def registry(self):
        return self.equilibrium.registry

    @property
    def gamma_s(self) -> float:
        return float(self.continuum.gamma_s)

    @classmethod
    def from_eqdsk(cls, path, *, cocos_in: int, density: Profile,
                   pressure="eqdsk", boozer_domain=None, gamma_s: float = 5.0 / 3.0,
                   species: str = "D", profile_points: int = 128,
                   dense_continuum: bool = True, **equilibrium_kwargs) -> "Case":
        """Load an eqdsk and build the equilibrium and continuum around it.

        ``pressure="eqdsk"`` takes ``p(psi)`` from the file's Grad-Shafranov
        profiles; pass a :class:`~supernova.radial.profile.Profile` to override.
        """
        import pycocos as pc

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        eqdsk = pc.EQDSK(str(path), cocos_in=cocos_in)

        if isinstance(pressure, str):
            if pressure != "eqdsk":
                raise ValueError(
                    f"pressure must be a Profile or 'eqdsk', got {pressure!r}."
                )
            pressure = profile_from_eqdsk(eqdsk, "pres", units="Pa",
                                          points=profile_points)

        equilibrium = superEquilibrium(
            eqdsk, density=density, pressure=pressure,
            rhodomain_boozer=boozer_domain, **equilibrium_kwargs,
        )
        continuum = superContinuum(equilibrium, gamma_s=gamma_s, species=species,
                                   dense=dense_continuum)
        return cls(equilibrium, continuum, source=path,
                   metadata={"cocos_in": cocos_in, "species": species,
                             "gamma_s": gamma_s})

    def with_continuum(self, *, gamma_s: float | None = None,
                       species: str | None = None,
                       dense: bool = True) -> "Case":
        """Return the same equilibrium with a differently parameterised continuum.

        gamma_s and the ion species change the continuum but not the geometry,
        and the Boozer transform is the expensive step -- seconds for a real
        eqdsk against about 0.1 s for the continuum. Rebuilding the whole Case
        to sweep gamma_s would pay for the transform again for nothing.
        """
        gamma_s = self.gamma_s if gamma_s is None else float(gamma_s)
        species = self.metadata.get("species", "D") if species is None else species
        continuum = superContinuum(self.equilibrium, gamma_s=gamma_s,
                                   species=species, dense=dense)
        metadata = dict(self.metadata)
        metadata.update({"gamma_s": gamma_s, "species": species})
        return Case(self.equilibrium, continuum, source=self.source,
                    metadata=metadata)

    @classmethod
    def from_refined(cls, directory, *, density: Profile, **kwargs) -> "Case":
        """Load a TokaMaker-refined equilibrium, COCOS read from its sidecar.

        The refinement writes ``output_cocos`` into its diagnostics JSON, so the
        ``7 if refined else 5`` ternary that every case script carried is not
        needed here.
        """
        from supernova.refine import load_diagnostics

        diagnostics = load_diagnostics(directory)
        case = cls.from_eqdsk(diagnostics["output_eqdsk"],
                              cocos_in=int(diagnostics["output_cocos"]),
                              density=density, **kwargs)
        case.metadata["refinement"] = diagnostics
        return case


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

_FIELD_SPACE = {"p1": "high", "xi_psi": "high", "xi_s": "low", "div_xi": "low"}


class ModeSet:
    """Eigenpairs from a scan, with the diagnostics scripts kept recomputing.

    Wraps the solver's ``xarray.Dataset`` rather than replacing it -- the raw
    dataset is ``.dataset`` and its schema is unchanged.  What this adds is
    reconstruction of the fields from their finite-element coefficients, which
    previously required reaching into ``solver._get_mixed_radial_data()``.
    """

    def __init__(self, dataset: xr.Dataset, scan: "Scan"):
        self.dataset = dataset
        self._scan = scan
        self._cache: dict[tuple[str, int], xr.DataArray] = {}

    def __len__(self) -> int:
        return int(self.dataset.sizes.get("eigenpair", 0))

    def __repr__(self) -> str:
        return f"<ModeSet: {len(self)} eigenpairs>"

    @property
    def omega(self) -> np.ndarray:
        return np.asarray(self.dataset["omega"].values)

    @property
    def frequency_khz(self) -> np.ndarray:
        """Cyclic frequency in kHz, ``Re(omega)`` through the case's registry."""
        return self._scan.band_for(self.omega.real, units="kHz")

    @property
    def growth_rate_khz(self) -> np.ndarray:
        return self._scan.band_for(self.omega.imag, units="kHz")

    @property
    def m_pol(self) -> np.ndarray:
        return np.asarray(self.dataset["m_pol"].values)

    def field(self, name: str, n: int = 512) -> xr.DataArray:
        """Reconstruct one field on ``n`` radial points.

        ``p1`` and ``xi_psi`` live in the high-order space, ``xi_s`` and
        ``div_xi`` in the compatible low-order one; the right basis is chosen
        here so callers do not have to know which is which.
        """
        if name not in _FIELD_SPACE:
            raise ValueError(
                f"Unknown field {name!r}. Choose one of {sorted(_FIELD_SPACE)}."
            )
        key = (name, int(n))
        if key in self._cache:
            return self._cache[key]

        mesh = self._mesh(name)
        points = np.linspace(mesh[0], mesh[-1], int(n))
        values = self._evaluate(name, points)
        array = xr.DataArray(
            values,
            coords={"x": points, "eigenpair": self.dataset["eigenpair"],
                    "m_pol": self.dataset["m_pol"]},
            dims=["eigenpair", "x", "m_pol"],
            attrs={"long_name": name},
        )
        array.x.attrs = {"units": "psi_N", "long_name": "normalized poloidal flux"}
        self._cache[key] = array
        return array

    def _basis(self, name: str):
        pair = self._scan.compatible_pair
        return pair.high if _FIELD_SPACE[name] == "high" else pair.low

    def _mesh(self, name: str) -> np.ndarray:
        """The finite-element breakpoints of the space ``name`` lives in."""
        return np.asarray(self._basis(name).get_mesh(), dtype=float)

    def _evaluate(self, name: str, points: np.ndarray) -> np.ndarray:
        """``name`` at ``points``, shaped ``(eigenpair, x, m_pol)``."""
        phi = np.asarray(self._basis(name).eval_basis(points), dtype=float).T
        coefficients = np.asarray(self.dataset[f"{name}_coefficient"])
        return np.einsum("ij,kjm->kim", phi, coefficients)

    def locate(self, points=None,
               thresholds: ModeLocateThresholds | None = None,
               continuum: tuple | None = None) -> xr.Dataset:
        """Gerrit Kramer's ``modelocate`` classification of every eigenpair.

        Flags each mode as a gap mode or as peaking on a single grid point, in
        the core, at the edge, or on a continuum crossing; see
        :mod:`supernova.solve.modelocate`.  The envelope is sampled on the
        finite-element breakpoints unless ``points`` is given: they play the
        part of NOVA's radial grid, and the spike and continuum tests compare
        neighbouring grid points.  ``continuum`` is ``(psi, omega)`` -- solver
        radii and ``(npsi, nbranch)`` normalized frequencies -- to test the
        crossings against instead of the scan's full sound-Alfven continuum,
        e.g. the Alfven branches alone when the slow sound is dropped.
        """
        points = (np.unique(self._mesh("xi_psi")) if points is None
                  else np.asarray(points, dtype=float))
        if continuum is None:
            continuum = self._scan.continuum
            psi, branches = continuum.coords["psi"].values, continuum["omega"].values
        else:
            psi, branches = (np.asarray(value) for value in continuum)
        located = locate_modes(
            points,
            self._evaluate("xi_psi", points),
            self.omega,
            [branch_crossings(psi, branches, omega) for omega in self.omega],
            rho=self._scan.rho_pol(points),
            thresholds=thresholds,
        )
        return located.assign_coords(eigenpair=self.dataset["eigenpair"].values)

    def gap_modes(self, **kwargs) -> "ModeSet":
        """The eigenpairs :meth:`locate` classifies as gap modes."""
        keep = np.flatnonzero(self.locate(**kwargs)["mode_class"].values == GAP)
        return ModeSet(self.dataset.isel(eigenpair=keep), self._scan)

    def _envelope(self, n: int = 512):
        power = np.sum(np.abs(np.asarray(self.field("xi_psi", n=n)))**2, axis=2)
        points = np.asarray(self.field("xi_psi", n=n).coords["x"].values)
        norm = np.trapezoid(power, x=points, axis=1)
        return points, power, norm

    @property
    def centroid(self) -> np.ndarray:
        """Power-weighted radial centre of each mode's ``xi^psi`` envelope."""
        points, power, norm = self._envelope()
        out = np.full(len(self), np.nan)
        valid = np.isfinite(norm) & (norm > 0.0)
        if valid.any():
            out[valid] = np.trapezoid(points[None, :] * power[valid], points,
                                      axis=1) / norm[valid]
        return out

    @property
    def width(self) -> np.ndarray:
        """RMS radial width of the same envelope."""
        points, power, norm = self._envelope()
        centre = self.centroid
        out = np.full(len(self), np.nan)
        valid = np.isfinite(norm) & (norm > 0.0)
        if valid.any():
            variance = np.trapezoid(
                (points[None, :] - centre[valid, None])**2 * power[valid],
                points, axis=1) / norm[valid]
            out[valid] = np.sqrt(np.maximum(variance, 0.0))
        return out

    def sort_by_frequency(self) -> "ModeSet":
        order = np.argsort(self.frequency_khz)
        return ModeSet(self.dataset.isel(eigenpair=order), self._scan)

    def to_netcdf(self, path) -> None:
        """Write the raw dataset, splitting complex arrays for portability."""
        dataset = self.dataset
        encoded = dataset.drop_vars(
            [name for name, var in dataset.variables.items()
             if np.iscomplexobj(var.values)])
        for name, var in dataset.variables.items():
            if np.iscomplexobj(var.values):
                encoded[f"{name}_real"] = (var.dims, np.real(var.values))
                encoded[f"{name}_imag"] = (var.dims, np.imag(var.values))
        encoded.attrs = {key: ("" if value is None else value)
                         for key, value in dataset.attrs.items()}
        encoded.to_netcdf(str(path))


# --------------------------------------------------------------------------
# Scans
# --------------------------------------------------------------------------

class Scan:
    """A windowed frequency scan over one radial grid.

    Owns a :class:`~supernova.solve.stability.StabilitySolver` and reuses it:
    the solver blocks and the mixed bases are built once, and each window is a
    retargeted solve.  For the multigrid classification pipeline use
    :class:`~supernova.solve.multigrid.MultigridSolver` directly.
    """

    def __init__(self, case: Case, grid: RadialGrid, *, ntor: int,
                 mmax: int | None = None, max_dm: int = 10,
                 regularizer: float = 0.0):
        self.case = case
        self.grid = grid
        self.ntor = int(ntor)
        self.mmax = mmax
        self.max_dm = int(max_dm)
        self.regularizer = float(regularizer)
        self.solver = StabilitySolver(case.equilibrium, case.continuum, grid.build())
        self._continuum: xr.Dataset | None = None

    def __repr__(self) -> str:
        return (f"<Scan n={self.ntor} spans={self.grid.spans} "
                f"order={self.grid.order} gamma_s={self.case.gamma_s:g}>")

    @property
    def compatible_pair(self):
        """The high/low B-spline pair the solver assembled against."""
        return self.solver._get_mixed_radial_data().pair

    @property
    def continuum(self) -> xr.Dataset:
        """The solver's own sound-Alfven continuum, on its quadrature points.

        Built from the same blocks, harmonics and coupling cut-off as the
        eigensolve, so its crossings are the ones the modes actually see.
        """
        if self._continuum is None:
            blocks = self.solver.build_solver_blocks(
                ntor=self.ntor, mmax=self.mmax, max_dm=self.max_dm)
            self._continuum = _solve_master_continuum(
                blocks, np.asarray(self.solver.unique_points, dtype=float))
        return self._continuum

    def rho_pol(self, x) -> np.ndarray:
        """``sqrt(psi_N)`` at solver radii ``x``.

        The solver's ``x`` spans the active Boozer interval, which is the
        physical ``psi_N`` only when that interval is ``(0, 1)``.
        """
        x = np.asarray(x, dtype=float)
        magspace = self.case.equilibrium.magspace
        if "psi_physical_norm" in magspace.coords:
            x = np.interp(x, magspace.coords["psi"].values,
                          magspace.coords["psi_physical_norm"].values)
        return np.sqrt(np.clip(x, 0.0, None))

    @property
    def backend(self) -> str:
        """Which eigensolver a solve will use, given the adiabatic index."""
        return "gevp" if self.case.gamma_s == 0.0 else "pep"

    def band_for(self, omega, units: str = "kHz") -> np.ndarray:
        angular = unyt.unyt_array(np.asarray(omega, dtype=float) / (2.0 * np.pi),
                                  "omegaA", registry=self.case.registry)
        return np.asarray(angular.to(units).v, dtype=float)

    def solve_window(self, band: Band, *, low=None, high=None, nev: int = 4,
                     options: PetscOptions | Mapping[str, str] | None = None,
                     tol: float = 1.0e-8, maxit: int = 100,
                     verbose: bool = False, **kwargs) -> ModeSet:
        """Solve one window and return its eigenpairs."""
        band = band.for_case(self.case)
        target_omega, region = band.region(low, high)
        if options is None:
            options = PetscOptions()
        petsc_options = (options.as_dict() if isinstance(options, PetscOptions)
                         else dict(options))
        dataset = self.solver.solve_pep(
            ntor=self.ntor, target_omega=target_omega, region=region,
            mmax=self.mmax, max_dm=self.max_dm, regularizer=self.regularizer,
            nev=nev, tol=tol, maxit=maxit, options=petsc_options,
            verbose=verbose, **kwargs,
        )
        return ModeSet(dataset, self)

    def run(self, band: Band, *, window: float, overlap: float = 0.0,
            nev: int = 4, options: PetscOptions | Mapping[str, str] | None = None,
            tol: float = 1.0e-8, maxit: int = 100, verbose: bool = False,
            on_window=None, **kwargs) -> ModeSet:
        """Sweep the band in windows and concatenate what each one returns.

        ``on_window(index, total, low, high, modes)`` is called after every
        window, which is enough to drive a progress bar without this method
        having to know about one.
        """
        band = band.for_case(self.case)
        windows = band.windows(window, overlap)
        collected = []
        for index, (low, high) in enumerate(windows):
            found = self.solve_window(band, low=low, high=high, nev=nev,
                                      options=options, tol=tol, maxit=maxit,
                                      verbose=verbose, **kwargs)
            if on_window is not None:
                on_window(index, len(windows), low, high, found)
            if len(found):
                collected.append(found.dataset.assign_coords(
                    window=("eigenpair", np.full(len(found), index))))
        if not collected:
            raise RuntimeError(
                f"No eigenpairs were found anywhere in {band.edges} {band.units}. "
                "Widen the band, raise nev, or check the continuum for this n."
            )
        merged = xr.concat(collected, dim="eigenpair", data_vars="minimal",
                           coords="minimal", compat="override",
                           combine_attrs="drop_conflicts")
        merged = merged.assign_coords(
            eigenpair=np.arange(merged.sizes["eigenpair"]))
        return ModeSet(merged, self)
