"""The equilibrium tab without Qt: its state, the loaders, and what gets drawn.

Loading is the slow part -- tens of seconds for the Boozer transform of a real
eqdsk at production resolution -- so it happens once per change of the inputs,
on a worker thread, and produces an :class:`EquilibriumView`: a snapshot of
every array the tab draws. The window only ever reads the snapshot. The
underlying :class:`~supernova.equilibrium.superEquilibrium` is carried along
for the solver tab, which uses it on the same worker thread, so the two never
touch it at once.

Sources are a registry (``_LOADERS``), so another input -- TRANSP, say -- is a
loader plus a page of controls, not a change to the drawing code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np

from supernova.gui._session import JsonState
from supernova.gui._state import COORDINATES

SOURCES = ("eqdsk", "itpa")
SOURCE_LABELS = {"eqdsk": "g-EQDSK file", "itpa": "ITPA TAE benchmark (analytic)"}
PROFILE_AXES = ("psi_n", "rho_pol")

# A geometry-only load needs some density: superEquilibrium normalises by its
# on-axis value. This one is flagged everywhere it could mislead -- above all
# in the solver tab, whose kHz scale it sets.
PLACEHOLDER_DENSITY = 1.0e19
PROFILE_POINTS = 128


# ----------------------------------------------------------------------------
# What can be plotted on one flux surface
# ----------------------------------------------------------------------------

# (key, plot label, list label), in the order the list shows them.
# Contravariant components are the gradient dot products the solver uses;
# covariant ones are their inverse, computed per surface. Mixed components
# with exactly one Theta index change sign under the positive-q reflection,
# and these are shown in the solver-facing orientation.
SURFACE_GROUPS = (
    ("Contravariant metric", (
        ("gpsi_psi", r"$g^{\psi\psi}=|\nabla\psi|^2$", "g^ψψ = |∇ψ|²"),
        ("gpsi_theta", r"$g^{\psi\Theta}$", "g^ψΘ"),
        ("gpsi_zeta", r"$g^{\psi\zeta}$", "g^ψζ"),
        ("gtheta_theta", r"$g^{\Theta\Theta}$", "g^ΘΘ"),
        ("gtheta_zeta", r"$g^{\Theta\zeta}$", "g^Θζ"),
        ("gzeta_zeta", r"$g^{\zeta\zeta}$", "g^ζζ"),
    )),
    ("Covariant metric", (
        ("cov_psi_psi", r"$g_{\psi\psi}$", "g_ψψ"),
        ("cov_psi_theta", r"$g_{\psi\Theta}$", "g_ψΘ"),
        ("cov_theta_theta", r"$g_{\Theta\Theta}$", "g_ΘΘ"),
        ("cov_theta_zeta", r"$g_{\Theta\zeta}$", "g_Θζ"),
        ("cov_zeta_zeta", r"$g_{\zeta\zeta}=R^2$", "g_ζζ = R²"),
    )),
    ("Jacobian and field", (
        ("jac", r"$J$ (coordinate map)", "J, from the coordinate map"),
        ("jac_boozer", r"$h/B^2$ (Boozer $J$)", "h/B², the Boozer J"),
        ("B", r"$|B|$", "|B|"),
        ("R", r"$R$", "R"),
        ("Z", r"$Z$", "Z"),
    )),
    ("Curvature, shear and current", (
        ("kappa_Psi", r"$\kappa\cdot\nabla\psi$", "κ·∇ψ (normal curvature)"),
        ("kappa_s", r"$\kappa\cdot\mathbf{s}$", "κ·s (geodesic curvature)"),
        ("shear", r"local shear $\mathcal{S}$", "local shear S"),
        ("J_B", r"$\mathbf{J}\cdot\mathbf{B}$", "J·B"),
        ("J_diag", r"$\mathbf{J}\cdot\mathbf{s}$", "J·s"),
        ("divPsiVec", r"$\nabla\cdot(\nabla\psi/|\nabla\psi|^2)$", "∇·(∇ψ/|∇ψ|²)"),
    )),
)
SURFACE_LABELS = {key: label for _, items in SURFACE_GROUPS for key, label, _ in items}
SURFACE_NAMES = {key: name for _, items in SURFACE_GROUPS for key, _, name in items}
SURFACE_QUANTITIES = tuple(SURFACE_LABELS)

# magspace name -> snapshot name, for the fields copied as they are
_MAGSPACE_FIELDS = {
    "gpsi_psi": "gpsi_psi", "gpsi_theta": "gpsi_theta",
    "gpsi_zeta": "gpsi_zeta", "gtheta_theta": "gtheta_theta",
    "gtheta_zeta": "gtheta_zeta", "gzeta_zeta": "gzeta_zeta",
    "B0": "B", "kappa_Psi": "kappa_Psi", "kappa_s": "kappa_s",
    "shear": "shear", "J_B": "J_B", "J_diag": "J_diag",
    "divPsiVec": "divPsiVec",
}
# the coordinate Jacobian is 'jac' for an eqdsk and 'jac_magnetic' for ITPA
_JACOBIAN_FIELDS = ("jac", "jac_magnetic")

_COVARIANT = {
    "cov_psi_psi": (0, 0), "cov_psi_theta": (0, 1),
    "cov_theta_theta": (1, 1), "cov_theta_zeta": (1, 2),
    "cov_zeta_zeta": (2, 2),
}
_COVARIANT_UNITS = {
    "cov_psi_psi": "m**2/Wb**2", "cov_psi_theta": "m**2/Wb",
    "cov_theta_theta": "m**2", "cov_theta_zeta": "m**2",
    "cov_zeta_zeta": "m**2",
}


# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------

@dataclass
class EquilibriumState(JsonState):
    """Every control in the equilibrium tab, and nothing else."""

    source: str = "eqdsk"

    # g-EQDSK
    eqdsk: str = ""
    cocos_in: int = 5
    density_file: str = ""            # optional: geometry does not need it
    density_uniform_m3: float = 0.0   # used when there is no file; 0 is none
    density_units: str = "m^-3"
    density_x: str = "psi_n"
    density_interpolate_in: str = "rho_pol"
    boozer_min: float = 0.005         # in rho_pol, as pyCOCOS takes it
    boozer_max: float = 0.995
    boozer_npsi: int = 511
    boozer_ntheta: int = 512
    m_max: int = 128

    # ITPA benchmark; the defaults are the published configuration
    itpa_aminor: float = 1.0
    itpa_rmajor: float = 10.0
    itpa_b0: float = 3.0
    itpa_q0: float = 1.71
    itpa_qedge: float = 1.87
    itpa_bt_sign: int = 1
    itpa_ns: int = 256
    itpa_ntheta: int = 256
    itpa_m_max: int = 20

    # view
    surface_psi_n: float = 0.5
    surface_quantities: list = field(default_factory=lambda: [
        "gpsi_psi", "gtheta_theta", "jac", "B"])
    normalize_surface: bool = False
    grid_surfaces: int = 12
    grid_spokes: int = 16
    profile_x: str = "psi_n"
    show_flux_contours: bool = True

    # TokaMaker fixed-boundary refinement
    refine_output_dir: str = ""       # empty: <eqdsk dir>/tokamaker_refined
    refine_stem: str = ""             # empty: <eqdsk stem>_tokamaker
    refine_output_cocos: int = 7
    refine_coarse_dx: float = 0.030
    refine_fine_dx: float = 0.015
    refine_dx_sequence: str = "0.025, 0.020, 0.015"
    refine_fe_order: int = 3
    refine_ffp_knots: int = 13
    refine_nthreads: int = 2
    refine_max_iterations: int = 20
    refine_target_rms: float = 2.0e-2
    refine_fit_current: bool = False
    refine_export_nr: int = 257
    refine_export_nz: int = 257
    # whether a re-solved equilibrium also replaces the one the continuum and
    # solver tabs use; read when a re-solve finishes, not when it is ticked
    refine_share: bool = True

    def validate(self) -> list[str]:
        """Return the reasons this state cannot be loaded, in reading order."""
        problems = []
        if self.source not in SOURCES:
            problems.append(f"Source must be one of {', '.join(SOURCES)}.")
        elif self.source == "eqdsk":
            if not self.eqdsk:
                problems.append("Choose an equilibrium file.")
            elif not Path(self.eqdsk).is_file():
                problems.append(f"Equilibrium not found: {self.eqdsk}")
            if self.density_file and not Path(self.density_file).is_file():
                problems.append(f"Density file not found: {self.density_file}")
            if not np.isfinite(self.density_uniform_m3) or self.density_uniform_m3 < 0.0:
                problems.append("A uniform density must be finite and non-negative.")
            if self.density_x not in COORDINATES:
                problems.append(f"Density column must be one of {', '.join(COORDINATES)}.")
            if self.density_interpolate_in not in COORDINATES:
                problems.append(
                    f"Density interpolation must run in one of {', '.join(COORDINATES)}.")
            if not 0.0 <= self.boozer_min < self.boozer_max <= 1.0:
                problems.append("Boozer domain must satisfy 0 <= min < max <= 1.")
            if self.boozer_npsi < 16 or self.boozer_ntheta < 16:
                problems.append("Boozer resolution must be at least 16 x 16.")
            if self.m_max < 1:
                problems.append("The equilibrium needs at least one poloidal harmonic.")
        else:
            if self.itpa_aminor <= 0.0:
                problems.append("The minor radius must be positive.")
            if self.itpa_rmajor <= self.itpa_aminor:
                problems.append("The major radius must exceed the minor radius.")
            if self.itpa_b0 <= 0.0:
                problems.append("B0 must be positive.")
            if self.itpa_q0 <= 0.0 or self.itpa_qedge <= 0.0:
                problems.append("q0 and q_edge must be positive.")
            if self.itpa_bt_sign not in (-1, 1):
                problems.append("The toroidal-field sign must be +1 or -1.")
            if self.itpa_ns < 4:
                problems.append("ITPA needs at least 4 radial surfaces.")
            if self.itpa_ntheta < 8:
                problems.append("ITPA needs at least 8 poloidal points.")
            if self.itpa_m_max < 1:
                problems.append("The equilibrium needs at least one poloidal harmonic.")
        problems.extend(self.view_problems())
        return problems

    def view_problems(self) -> list[str]:
        problems = []
        if not 0.0 <= self.surface_psi_n <= 1.0:
            problems.append("The flux surface must lie in 0 <= psi_N <= 1.")
        unknown = [name for name in self.surface_quantities
                   if name not in SURFACE_QUANTITIES]
        if unknown:
            problems.append(f"Unknown surface quantities: {', '.join(unknown)}")
        if self.grid_surfaces < 1 or self.grid_spokes < 1:
            problems.append("Draw at least one surface and one spoke.")
        if self.profile_x not in PROFILE_AXES:
            problems.append(f"Profile axis must be one of {', '.join(PROFILE_AXES)}.")
        return problems

    def shared_inputs(self) -> dict:
        """The choices the continuum tab can take over, under its names.

        Empty for ITPA, which the continuum tab cannot load; the density only
        when there is a file, so a placeholder never overwrites a real one.
        """
        if self.source != "eqdsk":
            return {}
        inputs = {"eqdsk": self.eqdsk, "cocos_in": self.cocos_in,
                  "boozer_min": self.boozer_min, "boozer_max": self.boozer_max}
        if self.density_file:
            inputs.update(density_file=self.density_file,
                          density_units=self.density_units,
                          density_x=self.density_x,
                          density_interpolate_in=self.density_interpolate_in)
        return inputs

    def equilibrium_key(self) -> tuple:
        """Everything a load depends on; the view settings are not here."""
        if self.source == "itpa":
            return ("itpa", self.itpa_aminor, self.itpa_rmajor, self.itpa_b0,
                    self.itpa_q0, self.itpa_qedge, self.itpa_bt_sign,
                    self.itpa_ns, self.itpa_ntheta, self.itpa_m_max)
        return ("eqdsk", self.eqdsk, self.cocos_in, self.density_file,
                self.density_uniform_m3, self.density_units, self.density_x,
                self.density_interpolate_in, self.boozer_min, self.boozer_max,
                self.boozer_npsi, self.boozer_ntheta, self.m_max)

    # ------------------------------------------------------------ refinement

    def refinement_paths(self) -> tuple[Path, str]:
        """The output directory and file stem, with the defaults filled in."""
        source = Path(self.eqdsk)
        directory = (Path(self.refine_output_dir) if self.refine_output_dir
                     else source.parent / "tokamaker_refined")
        stem = self.refine_stem or f"{source.stem}_tokamaker"
        return directory, stem

    def refinement_sequence(self) -> tuple[float, ...]:
        text = self.refine_dx_sequence.replace(";", ",").strip()
        if not text:
            return (float(self.refine_fine_dx),)
        return tuple(float(value) for value in text.split(",") if value.strip())

    def refinement_problems(self) -> list[str]:
        """Why a TokaMaker re-solve of this state cannot start."""
        problems = []
        if self.source != "eqdsk":
            problems.append("TokaMaker re-solves an eqdsk; the ITPA case is analytic.")
            return problems
        if not self.eqdsk or not Path(self.eqdsk).is_file():
            problems.append("Choose an existing equilibrium file to refine.")
        if self.refine_coarse_dx <= 0.0 or self.refine_fine_dx <= 0.0:
            problems.append("Mesh sizes must be positive.")
        elif self.refine_fine_dx >= self.refine_coarse_dx:
            problems.append("The fine mesh must be finer than the coarse one.")
        try:
            sequence = self.refinement_sequence()
        except ValueError:
            problems.append("The refinement sequence must be numbers separated by commas.")
        else:
            steps = (self.refine_coarse_dx, *sequence)
            if any(b >= a for a, b in zip(steps[:-1], steps[1:])):
                problems.append(
                    "The refinement sequence must decrease strictly from the coarse size.")
            elif not np.isclose(sequence[-1], self.refine_fine_dx, rtol=0.0, atol=1e-12):
                problems.append("The refinement sequence must end at the fine size.")
        if self.refine_fe_order < 1:
            problems.append("The finite-element order must be positive.")
        if self.refine_ffp_knots < 3:
            problems.append("FF' needs at least 3 knots.")
        if self.refine_nthreads < 1:
            problems.append("TokaMaker needs at least one thread.")
        if self.refine_max_iterations < 1:
            problems.append("The q fit needs at least one iteration.")
        if self.refine_target_rms <= 0.0:
            problems.append("The target q RMS must be positive.")
        if self.refine_export_nr < 33 or self.refine_export_nz < 33:
            problems.append("The exported grid must be at least 33 x 33.")
        return problems

    def refinement_spec(self) -> dict:
        """The plain-data job description the refinement subprocess reads."""
        directory, stem = self.refinement_paths()
        return {
            "source_eqdsk": str(Path(self.eqdsk).resolve()),
            "source_cocos": int(self.cocos_in),
            "output_dir": str(directory.resolve()),
            "stem": stem,
            "output_cocos": int(self.refine_output_cocos),
            # the g-file header field; TokaMaker refuses more than 40 characters
            "run_info": f"TokaMaker {Path(self.eqdsk).stem}"[:40],
            "coarse_dx": float(self.refine_coarse_dx),
            "fine_dx": float(self.refine_fine_dx),
            "refinement_dx_sequence": list(self.refinement_sequence()),
            "fe_order": int(self.refine_fe_order),
            "ffp_knot_count": int(self.refine_ffp_knots),
            "export_grid": [int(self.refine_export_nr), int(self.refine_export_nz)],
            "nthreads": int(self.refine_nthreads),
            "fit": {
                "max_iterations": int(self.refine_max_iterations),
                "target_rms_error": float(self.refine_target_rms),
                "fit_plasma_current": bool(self.refine_fit_current),
            },
        }


# ----------------------------------------------------------------------------
# The snapshot
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class EquilibriumView:
    """Everything the tab draws, copied out of a loaded equilibrium."""

    source: str
    title: str
    equilibrium: object               # superEquilibrium; None outside the compute process
    density_is_placeholder: bool
    psi_n: np.ndarray                 # (npsi,) physical normalised flux
    theta: np.ndarray                 # (ntheta,) solver-facing Boozer angle
    R: np.ndarray                     # (npsi, ntheta)
    Z: np.ndarray
    fields: dict                      # name -> (npsi, ntheta)
    units: dict                       # name -> units string
    profiles: dict                    # name -> (npsi,)
    file_q: tuple | None              # (psi_n, |q|) tabulated in the eqdsk
    boundary: np.ndarray              # (n, 2) last closed flux surface
    axis: tuple
    limiter: np.ndarray | None
    flux_map: tuple | None            # (R, Z, psi_n(R, Z), inside-LCFS mask)
    summary: dict
    load_seconds: float
    inputs: dict = field(default_factory=dict)   # what the other tabs may take over
    handle: int | None = None         # the compute process's name for ``equilibrium``

    @property
    def theta_closes(self) -> bool:
        """Whether the angle grid repeats 2*pi as its last point."""
        return bool(np.isclose(self.theta[-1] - self.theta[0], 2.0 * np.pi))

    def periodic(self, values: np.ndarray) -> np.ndarray:
        """Drop the repeated endpoint so averages do not count it twice."""
        return values[..., :-1] if self.theta_closes else values

    def surface_index(self, psi_n: float) -> int:
        return int(np.argmin(np.abs(self.psi_n - float(psi_n))))

    def profile_x(self, kind: str) -> np.ndarray:
        return np.sqrt(np.clip(self.psi_n, 0.0, None)) if kind == "rho_pol" else self.psi_n

    def surface(self, psi_n: float, names) -> dict:
        """The requested quantities along the surface nearest ``psi_n``."""
        index = self.surface_index(psi_n)
        values = {}
        covariant = None
        for name in names:
            if name in _COVARIANT:
                if covariant is None:
                    covariant = self._covariant(index)
                row, column = _COVARIANT[name]
                values[name] = covariant[:, row, column]
            elif name == "R":
                values[name] = self.R[index]
            elif name == "Z":
                values[name] = self.Z[index]
            elif name in self.fields:
                values[name] = self.fields[name][index]
        return {"index": index, "psi_n": float(self.psi_n[index]),
                "theta": self.theta, "values": values}

    def units_of(self, name: str) -> str:
        if name in _COVARIANT_UNITS:
            return _COVARIANT_UNITS[name]
        if name in ("R", "Z"):
            return "m"
        return self.units.get(name, "")

    def available(self, name: str) -> bool:
        if name in _COVARIANT:
            return all(key in self.fields for key in _CONTRAVARIANT_NAMES)
        return name in ("R", "Z") or name in self.fields

    def _covariant(self, index: int) -> np.ndarray:
        """Invert the contravariant metric pointwise along one surface."""
        g = np.empty((self.theta.size, 3, 3))
        for (row, column), name in _CONTRAVARIANT_LAYOUT.items():
            g[:, row, column] = self.fields[name][index]
            g[:, column, row] = self.fields[name][index]
        out = np.full_like(g, np.nan)
        finite = np.flatnonzero(np.all(np.isfinite(g), axis=(1, 2)))
        invertible = finite[np.abs(np.linalg.det(g[finite])) > 0.0]
        if invertible.size:
            out[invertible] = np.linalg.inv(g[invertible])
        return out

    def to_dataset(self):
        """The profiles and the surface fields as an xarray Dataset."""
        import xarray as xr

        data = {name: ("psi_n", np.asarray(values))
                for name, values in self.profiles.items()}
        data.update({name: (("psi_n", "theta"), np.asarray(values))
                     for name, values in self.fields.items()})
        data["R"] = (("psi_n", "theta"), self.R)
        data["Z"] = (("psi_n", "theta"), self.Z)
        dataset = xr.Dataset(data, coords={"psi_n": self.psi_n, "theta": self.theta})
        for name, units in self.units.items():
            if name in dataset:
                dataset[name].attrs["units"] = units
        dataset["R"].attrs["units"] = dataset["Z"].attrs["units"] = "m"
        dataset.attrs.update({
            "source": self.source,
            "title": self.title,
            "theta_convention": "solver-facing Boozer angle (q > 0)",
            "density_is_placeholder": int(self.density_is_placeholder),
            "written_by": "supernova.gui.equilibrium",
        })
        return dataset


def same_equilibrium(a, b) -> bool:
    """Whether two snapshots -- views or solver results -- share an equilibrium.

    Those sent from the compute process are new objects every time, so they
    compare by the handle it gives them; ones built in this process (tests,
    scripts) by the equilibrium object they carry.
    """
    if a is None or b is None:
        return False
    if a.handle is not None or b.handle is not None:
        return a.handle == b.handle
    return a.equilibrium is not None and a.equilibrium is b.equilibrium


_CONTRAVARIANT_LAYOUT = {
    (0, 0): "gpsi_psi", (0, 1): "gpsi_theta", (0, 2): "gpsi_zeta",
    (1, 1): "gtheta_theta", (1, 2): "gtheta_zeta", (2, 2): "gzeta_zeta",
}
_CONTRAVARIANT_NAMES = tuple(_CONTRAVARIANT_LAYOUT.values())


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

class EquilibriumModel:
    """Turns a state into a view, reloading only when the inputs changed."""

    def __init__(self):
        self.view: EquilibriumView | None = None
        self._key = None

    def load(self, state: EquilibriumState, *, progress=None) -> EquilibriumView:
        problems = state.validate()
        if problems:
            raise ValueError(problems[0])
        key = state.equilibrium_key()
        if self.view is not None and key == self._key:
            return self.view

        started = time.perf_counter()
        equilibrium, extra = _LOADERS[state.source](state, progress)
        if progress:
            progress("Sampling the magnetic coordinates for display...")
        view = build_view(equilibrium, source=state.source,
                          load_seconds=time.perf_counter() - started,
                          inputs=state.shared_inputs(), **extra)
        self.view, self._key = view, key
        return view


def _load_eqdsk(state: EquilibriumState, progress):
    import pycocos as pc

    from supernova.api import profile_from_columns
    from supernova.equilibrium import superEquilibrium
    from supernova.radial.profile import Profile

    if progress:
        progress("Reading the eqdsk and converting it to COCOS 1...")
    eqdsk = pc.EQDSK(str(state.eqdsk), cocos_in=int(state.cocos_in))

    # superEquilibrium reads a profile's x as its active flux coordinate,
    # which spans the Boozer domain rather than (0, 1). So the profiles are
    # sampled at the psi_N each x stands for; sampling at x = psi_N instead
    # stretches them across the domain.
    x = np.linspace(0.0, 1.0, PROFILE_POINTS)
    inner, outer = float(state.boozer_min) ** 2, float(state.boozer_max) ** 2
    psi_n = inner + x * (outer - inner)

    def profile(values, units, name):
        return Profile(x=x, F_nm=np.asarray(values, dtype=float),
                       attrs={"units": units, "name": name})

    if state.density_file:
        # read on a fine psi_N grid, honouring the column and interpolation
        # choices, then sampled where the active grid needs it
        fine = profile_from_columns(
            state.density_file, units=state.density_units, x=state.density_x,
            interpolate_in=state.density_interpolate_in, points=4097)
        values = np.real(np.asarray(fine.data.sel(mpol=0, ntor=0).values))
        density = profile(np.interp(psi_n, np.asarray(fine.x, dtype=float), values),
                          state.density_units, Path(state.density_file).stem)
        placeholder = False
    elif state.density_uniform_m3 > 0.0:
        # a benchmark choice, as in the ITPA case: not a placeholder
        density = profile(np.full(x.size, float(state.density_uniform_m3)),
                          "m^-3", "uniform density")
        placeholder = False
    else:
        density = profile(np.full(x.size, PLACEHOLDER_DENSITY), "m^-3",
                          "placeholder density")
        placeholder = True
    gs = eqdsk.gs_profs
    pressure = profile(np.interp(np.sqrt(psi_n), np.asarray(gs["rhop"], dtype=float),
                                 np.asarray(gs["pres"], dtype=float)), "Pa", "pres")

    if progress:
        progress(f"Computing Boozer coordinates on {state.boozer_npsi} x "
                 f"{state.boozer_ntheta} and the metric (the slow step)...")
    equilibrium = superEquilibrium(
        eqdsk, density=density, pressure=pressure, m_max=int(state.m_max),
        rhodomain_boozer=(float(state.boozer_min), float(state.boozer_max)),
        res_boozer=(int(state.boozer_npsi), int(state.boozer_ntheta)),
    )
    return equilibrium, {"title": Path(state.eqdsk).name,
                         "density_is_placeholder": placeholder}


def _load_itpa(state: EquilibriumState, progress):
    import unyt

    from supernova.cases.itpa import ITPAEq

    if progress:
        progress("Building the analytic ITPA equilibrium...")
    equilibrium = ITPAEq(
        aminor=float(state.itpa_aminor) * unyt.m,
        Rmajor=float(state.itpa_rmajor) * unyt.m,
        q0=float(state.itpa_q0), qedge=float(state.itpa_qedge),
        B0=float(state.itpa_b0) * unyt.T,
        ns=int(state.itpa_ns), ntheta=int(state.itpa_ntheta),
        m_max=int(state.itpa_m_max), Bt_sign=int(state.itpa_bt_sign),
    )
    return equilibrium, {"title": "ITPA TAE benchmark",
                         "density_is_placeholder": False}


_LOADERS = {"eqdsk": _load_eqdsk, "itpa": _load_itpa}


def build_view(equilibrium, *, source: str, title: str,
               density_is_placeholder: bool, load_seconds: float,
               inputs: dict | None = None) -> EquilibriumView:
    """Copy what the tab draws out of a loaded equilibrium."""
    magspace = equilibrium.magspace
    psi_n = np.asarray(
        magspace.coords["psi_physical_norm"].values
        if "psi_physical_norm" in magspace.coords
        else magspace.coords["psi"].values, dtype=float)
    theta = np.asarray(magspace.coords["theta"].values, dtype=float)
    orientation = int(getattr(equilibrium, "_poloidal_orientation", 1))

    fields, units = {}, {}
    for name, key in _MAGSPACE_FIELDS.items():
        if name in magspace:
            fields[key] = _surface_array(magspace[name])
            units[key] = str(magspace[name].attrs.get("units", ""))
    for name in _JACOBIAN_FIELDS:
        if name in magspace:
            fields["jac"] = _surface_array(magspace[name])
            units["jac"] = str(magspace[name].attrs.get("units", "m**3/Wb"))
            break

    profiles = {name: _profile_array(magspace[name])
                for name in ("q", "F", "I", "h", "pressure", "density")
                if name in magspace}
    if "B" in fields and "h" in profiles:
        fields["jac_boozer"] = profiles["h"][:, None] / fields["B"] ** 2
        units["jac_boozer"] = "m**3/Wb"

    R, Z = _magnetic_rz(equilibrium, psi_n, theta, orientation)

    view_kwargs = dict(
        source=source, title=title, equilibrium=equilibrium,
        density_is_placeholder=density_is_placeholder, psi_n=psi_n,
        theta=theta, R=R, Z=Z, fields=fields, units=units,
    )
    flux_span = _physical_flux_span(equilibrium)
    profiles.update(_derived_profiles(profiles, fields, theta, psi_n, flux_span))

    file_q = boundary = limiter = flux_map = None
    axis = (float(equilibrium.Rmajor), 0.0)
    ip = None
    if source == "eqdsk":
        equ = equilibrium.equ
        gs = equ.gs_profs
        file_q = (np.asarray(gs["rhop"].values, float) ** 2,
                  np.abs(np.asarray(gs["q"].values, float)))
        boundary = np.column_stack((np.asarray(equ.boundary["R"].values, float),
                                    np.asarray(equ.boundary["z"].values, float)))
        axis = (float(equ.geometry["R_axis"]), float(equ.geometry["z_axis"]))
        gdata = getattr(equ, "_gdata", {}) or {}
        if int(gdata.get("nlim", 0) or 0) > 2:
            limiter = np.column_stack((np.asarray(gdata["rlim"], float),
                                       np.asarray(gdata["zlim"], float)))
        if "Ip" in gdata:
            ip = float(gdata["Ip"])
        flux_map = _flux_map(equilibrium, boundary)
    else:
        boundary = np.column_stack((R[-1], Z[-1]))

    summary = _summary(equilibrium, profiles, psi_n, file_q, ip, orientation,
                       density_is_placeholder, theta)
    return EquilibriumView(profiles=profiles, file_q=file_q, boundary=boundary,
                           axis=axis, limiter=limiter, flux_map=flux_map,
                           summary=summary, load_seconds=float(load_seconds),
                           inputs=dict(inputs or {}), **view_kwargs)


def _surface_array(data) -> np.ndarray:
    if "zeta" in data.dims:
        data = data.isel(zeta=0)
    return np.asarray(np.real(data.transpose("psi", "theta").values), dtype=float)


def _profile_array(data) -> np.ndarray:
    values = np.asarray(np.real(data.values), dtype=float)
    return values.reshape(-1)


def _physical_flux_span(equilibrium) -> float:
    """psi_boundary - psi_axis in Wb/rad, outward-positive in COCOS 1."""
    attrs = equilibrium.boozer.coords.psi0.attrs
    return float(attrs["psi_boundary"]) - float(attrs["psi_axis"])


def _magnetic_rz(equilibrium, psi_n, theta, orientation):
    """R and Z on the solver's (psi, Theta) grid, in its orientation.

    ITPA stores them; for an eqdsk they come from the fitted pyCOCOS map,
    evaluated exactly on the surfaces the metric was evaluated on. The
    positive-q reflection is Theta' = -Theta, so the map is read at -Theta.
    """
    magspace = equilibrium.magspace
    if "R" in magspace and "z" in magspace:
        return _surface_array(magspace["R"]), _surface_array(magspace["z"])

    boozer = equilibrium.boozer
    attrs = boozer.coords.psi0.attrs
    psi_axis, psi_boundary = float(attrs["psi_axis"]), float(attrs["psi_boundary"])
    psi = psi_axis + psi_n * (psi_boundary - psi_axis)
    angle = theta if orientation == 1 else np.mod(-theta, 2.0 * np.pi)
    psi_mesh, theta_mesh = np.meshgrid(psi, angle, indexing="ij")

    coordinate_map = (getattr(boozer, "coordinate_map", None)
                      or getattr(boozer, "_coordinate_map", None))
    if coordinate_map is not None:
        values = coordinate_map.values(psi_mesh, theta_mesh)
        return (np.asarray(values["R"], dtype=float),
                np.asarray(values["z"], dtype=float))

    # Older pyCOCOS: interpolate the tabulated inverse map instead.
    from scipy.interpolate import RegularGridInterpolator

    psi0 = np.asarray(boozer.coords.psi0.values, dtype=float)
    theta_star = np.asarray(boozer.coords.theta_star.values, dtype=float)
    order = np.argsort(psi0)
    out = []
    for name in ("R_inv", "z_inv"):
        table = np.asarray(boozer.coords[name].values, dtype=float)[order]
        interpolator = RegularGridInterpolator(
            (psi0[order], theta_star), table, bounds_error=False, fill_value=None)
        out.append(interpolator(np.stack((psi_mesh, theta_mesh), axis=-1)))
    return out[0], out[1]


def _flux_map(equilibrium, boundary: np.ndarray):
    """psi_N on the eqdsk's R-Z grid, and which points lie inside the LCFS.

    The mask is a polygon test against the recorded boundary. pyCOCOS's own
    ``inside_lcfs`` is wider than the plasma, and a re-solved file pads the
    flux outside it with values in (0, 1) that would contour as surfaces.
    """
    from matplotlib.path import Path as Polygon

    equ = equilibrium.equ
    flux = equ.fluxdata
    psi_axis = float(equ.geometry.attrs["psi_ax"])
    psi_boundary = float(equ.geometry.attrs["psi_bdy"])
    psipol = np.asarray(flux["psipol"].transpose("R", "z").values, dtype=float)
    psi_n = (psipol - psi_axis) / (psi_boundary - psi_axis)
    R = np.asarray(flux["R"].values, float)
    Z = np.asarray(flux["z"].values, float)
    points = np.column_stack([axis.ravel() for axis in np.meshgrid(R, Z, indexing="ij")])
    inside = Polygon(boundary).contains_points(points).reshape(psi_n.shape)
    return R, Z, psi_n, inside


def _derived_profiles(profiles, fields, theta, psi_n, flux_span):
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        if "q" in profiles:
            q = profiles["q"]
            # rho_pol q'/q = 2 psi_N (dq/dpsi_N) / q
            out["shear"] = 2.0 * psi_n * np.gradient(q, psi_n) / q
        if "pressure" in profiles:
            p = profiles["pressure"]
            out["dp_dpsi"] = _flush(np.gradient(p, psi_n) / flux_span,
                                    np.max(np.abs(p)) / abs(flux_span))
        if "F" in profiles:
            F = profiles["F"]
            out["ffprime"] = _flush(F * np.gradient(F, psi_n) / flux_span,
                                    np.max(F**2) / abs(flux_span))
        if "B" in fields:
            closes = np.isclose(theta[-1] - theta[0], 2.0 * np.pi)
            B = fields["B"][:, :-1] if closes else fields["B"]
            # Boozer: J = h/B**2, so the flux-surface weight is 1/B**2
            weight = 1.0 / B**2
            out["B_min"] = np.nanmin(B, axis=1)
            out["B_max"] = np.nanmax(B, axis=1)
            out["B_avg"] = np.nansum(B * weight, axis=1) / np.nansum(weight, axis=1)
    return out


def _flush(values: np.ndarray, scale: float) -> np.ndarray:
    """Zero what is round-off relative to ``scale``: the derivative of a
    constant F should plot as zero, not as 1e-11 noise."""
    return np.where(np.abs(values) < 1.0e-9 * scale, 0.0, values)


def _summary(equilibrium, profiles, psi_n, file_q, ip, orientation,
             placeholder, theta) -> dict:
    q = profiles.get("q")
    summary = {
        "R_axis_m": float(equilibrium.Rmajor),
        "a_minor_m": float(equilibrium.aminor),
        "B_axis_T": float(np.asarray(equilibrium.Baxis).squeeze()),
        "surfaces": int(psi_n.size),
        "poloidal_points": int(theta.size),
        "psi_n_range": (float(psi_n.min()), float(psi_n.max())),
        "theta_orientation": orientation,
        "density_is_placeholder": bool(placeholder),
    }
    if ip is not None:
        summary["Ip_MA"] = abs(ip) / 1.0e6
    if q is not None:
        summary["q_inner"] = float(q[0])
        summary["q_min"] = float(np.nanmin(q))
        summary["q_edge"] = float(q[-1])
        if psi_n.min() <= 0.95 <= psi_n.max():
            summary["q95"] = float(np.interp(0.95, psi_n, q))
        if file_q is not None:
            tabulated = np.interp(psi_n, file_q[0], file_q[1])
            scale = max(float(np.sqrt(np.mean(tabulated**2))), np.finfo(float).tiny)
            summary["q_boozer_vs_file_rms"] = float(
                np.sqrt(np.mean((q - tabulated) ** 2)) / scale)
    if "density" in profiles:
        summary["n_axis_m3"] = float(profiles["density"][0])
    return summary
