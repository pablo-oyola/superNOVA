<p align="center">
  <img src="docs/img/supernova_logo.png" alt="superNOVA" width="520">
</p>

<p align="center">
  <a href="https://pablo-oyola.github.io/superNOVA/"><b>Documentation</b></a> ·
  <a href="https://github.com/pablo-oyola/superNOVA/actions/workflows/docs.yml"><img src="https://github.com/pablo-oyola/superNOVA/actions/workflows/docs.yml/badge.svg" alt="Docs"></a>
</p>

<!-- docs-intro-start -->
Linear ideal-MHD stability of tokamak equilibria: Boozer coordinates,
sound-Alfvén continuum and global eigenmodes on a radial B-spline finite-element
× poloidal Fourier basis, solved with SLEPc.  The distribution is called
`pynova`; the import name is `supernova`.

- **Equilibria** from a g-EQDSK file (through
  [pyCOCOS](https://github.com/pablo-oyola/pyCOCOS), any COCOS), from DESC, or
  analytic (the ITPA TAE benchmark), with an optional TokaMaker fixed-boundary
  re-solve.
- **Continuum** for any toroidal mode number, with or without sound coupling.
- **Eigenmodes** in a frequency band, swept window by window with
  shift-and-invert, and sorted into gap modes and continuum-resonant ones
  (Kramer's `modelocate`).
- **Two front ends**: a Python API for scripts and a desktop window with
  Equilibrium, Continuum and Solver tabs.
<!-- docs-intro-end -->

<!-- docs-guide-start -->
## Installation

Everything installs with pip into any Python 3.10-3.13 environment, for
example a fresh conda one.  The machine needs C, C++ and Fortran compilers
(`gcc`, `g++`, `gfortran`), `make`, `git` and internet access.  MPI, CMake and
BLAS do not have to be installed first.

```bash
conda create -n supernova python=3.12 -y
conda activate supernova

pip install "pynova[gui] @ git+https://github.com/pablo-oyola/superNOVA.git"
supernova install-solver
```

The first command installs superNOVA and its pure-Python dependencies in a
minute.  The second compiles the eigenvalue backend, PETSc and SLEPc with
their Python bindings, which took 20 minutes on a 20-core machine.  It runs pip
four times in a row, and ends with `supernova check`, which confirms the
build and solves one small shift-and-invert problem through MUMPS.  Run
`supernova check` again at any time.

The backend needs a step of its own because PyPI has PETSc and SLEPc only as
source releases.  The solver needs them built with complex scalars and the
MUMPS LU, and a plain `pip install petsc slepc` cannot do that.
`install-solver` builds a serial PETSc that downloads its own CMake, OpenBLAS,
METIS, MUMPS and SuiteSparse.  Its OpenBLAS picks CPU kernels at run time, so
a build made on a login node also runs on compute nodes with another CPU.
`supernova install-solver --dry-run` prints the pip commands, and
`--extra-options=--with-make-np=8` appends a PETSc configure option, here
for a login node that limits processes.

For a development checkout:

```bash
git clone git@github.com:pablo-oyola/superNOVA.git supernova
cd supernova
pip install -e ".[gui,dev]"
supernova install-solver
```

| Extra | Adds |
|---|---|
| `gui` | the desktop window (PySide6) and HDF5 sessions |
| `desc` | the DESC equilibrium reader (pulls JAX) |
| `tokamaker` | the TokaMaker refinement layer (OpenFUSIONToolkit itself is not on pip) |
| `dev`, `docs` | pytest; Sphinx |

[docs/installation.md](docs/installation.md) explains every build option and
lists the known pitfalls.

### Without the compiled stack

`pip install .` alone installs the pure-Python layers (equilibrium, Boozer
coordinates, continuum, verification).  Anything that reaches the eigenvalue
backend raises an `ImportError` naming `petsc4py`/`slepc4py`.

## Quick start: the ITPA TAE benchmark

The analytic ITPA equilibrium needs no input file and solves in a couple of
seconds, so it is the place to start.  It is a circular plasma with
R₀ = 10 m, a = 1 m, B₀ = 3 T, q rising from 1.71 to 1.87, hydrogen ions, and
the sound coupling switched off (γ = 0).  The benchmark mode is the n = 6
TAE in the gap between the m = 10 and m = 11 continua, near ψ_N ≈ 0.25.

### From a script

```python
import matplotlib.pyplot as plt
import unyt

import supernova as sn
from supernova.cases.itpa import ITPAEq

# 1. Equilibrium and continuum: built once, reused by every scan.
equilibrium = ITPAEq(aminor=1.0 * unyt.m, Rmajor=10.0 * unyt.m, B0=3.0 * unyt.T,
                     q0=1.71, qedge=1.87, ns=256, ntheta=256, m_max=20)
continuum = sn.superContinuum(equilibrium, gamma_s=0.0, species="H")
case = sn.Case(equilibrium, continuum)

# 2. n = 6 on 100 cubic B-spline elements, with the benchmark's harmonics.
#    Left out, mmax is ceil(n q) at rho_pol = 0.95 and max_dm is 20.
scan = sn.Scan(case, sn.RadialGrid.uniform(spans=100), ntor=6, mmax=20,
               max_dm=5, regularizer=1e-5)

# 3. Search 60-76 kHz in one window for 4 roots.
modes = scan.run(sn.Band(60.0, 76.0, units="kHz"), window=16.0, nev=4,
                 tol=1e-7, maxit=80)

print(modes.frequency_khz)            # every root found in the band
tae = modes.gap_modes()               # those modelocate calls gap modes
print(tae.frequency_khz, tae.centroid, tae.width)

# 4. The mode on its continuum, and its poloidal harmonics.
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
psi = scan.continuum.coords["psi"].values
left.plot(psi, scan.band_for(scan.continuum["omega"].values.real), "k.", ms=1)
left.axhline(tae.frequency_khz[0], color="C3")
left.set(xlabel=r"$\psi_N$", ylabel="f [kHz]", ylim=(0, 150))

xi = tae.field("xi_psi").isel(eigenpair=0)      # dims: x, m_pol
for m in xi.m_pol.values:
    right.plot(xi.x, xi.sel(m_pol=m).real, label=f"m = {m}")
right.set(xlabel=r"$\psi_N$", ylabel=r"Re $\xi^\psi$")
right.legend()
plt.show()
```

The scan returns five roots between 69.95 and 71.02 kHz.  Only the first,
at **69.954 kHz**, is a gap mode; the other four sit on the continuum.  The
gap mode is centred at ψ_N ≈ 0.25 and is dominated by the m = 10 and 11
harmonics.

Every result wraps the solver's `xarray.Dataset` (`modes.dataset`).  The main
objects are listed below:

| Object | What it is |
|---|---|
| `Case` | equilibrium + continuum; `Case.from_eqdsk(...)` and `Case.from_refined(...)` build one from a file |
| `RadialGrid` | the radial finite elements: `.uniform(spans)` or `.graded(spans, centers, local)` |
| `Band` | a frequency band in physical units, converted to the solver's normalised ω² |
| `Scan` | one toroidal mode number on one grid; `.run(band, window=...)` sweeps the band |
| `ModeSet` | the roots: `.frequency_khz`, `.growth_rate_khz`, `.field(name)`, `.locate()`, `.gap_modes()`, `.centroid`, `.width`, `.to_netcdf(path)` |
| `PetscOptions` | the shift-and-invert factorisation (`mumps`, `spqr`, `petsc_qr`) plus raw PETSc options |

### From the GUI

```bash
supernova gui                 # or: python -m supernova gui
```

The window has three tabs.  The Equilibrium tab loads an equilibrium and
passes it to the Continuum and Solver tabs.  It is the same object in all
three, held by one compute process.

1. **Equilibrium tab.**  In the *Equilibrium* box, choose
   *ITPA TAE benchmark (analytic)*.  The fields already hold the published
   configuration (a = 1 m, R0 = 10 m, B0 = 3 T, q0 = 1.71, q edge = 1.87,
   m_max = 20).  Set *ion species* to **H** and press **Load equilibrium**.
   The *Shape*, *Profiles* and *Surface* pages show the Boozer grid, the 1-D
   profiles and any metric element on one flux surface.
2. **Continuum tab.**  Set *n* = 6, *gamma_s* = 0 and *branches* to *no
   sound (Alfven only)*; *mmax* stays *auto*.  The continuum redraws on every
   edit.  Under *View*, *f max* = 150 and *shade band* = 60-76 kHz show the
   TAE gap and the search range.
3. **Solver tab.**  Every field opens on the script API's default, so only
   the benchmark's own choices need typing:
   - *Problem*: *n* = 6, *regularizer* = 1e-5, *gamma_s* = 0.  *mmax* stays
     *auto* (m = 7-12 here) and *max dm* at 20.  The species is greyed out
     because it is taken from the Equilibrium tab.
   - *Frequency search*: *band* = 60-76 kHz, *window* = 16 kHz,
     *tol / maxit* = 1e-7 / 80.
   - *Radial grid*: *spans* = 100.

   Then press **Solve**.  The roots appear on the continuum in the *Spectrum*
   page.  *Mode structure* draws the selected root's harmonics.  *show gap
   modes only* keeps the modes `modelocate` accepts.  *Refine selected modes*
   re-solves the chosen modes on graded meshes to check their convergence.
4. **Save.**  *File → Save session…* (Ctrl+Shift+S) writes every tab's
   settings and the solver's results to one HDF5 file.  Reopen it with
   `supernova gui session.h5`; the equilibrium is loaded again and the roots
   are put back.

For a g-EQDSK, choose *g-EQDSK file* in step 1 instead.  Then give the file,
its COCOS and a density, either a two-column file or a uniform value.  Without
a density the geometry still loads, but the kHz scale in the other tabs is
marked as a placeholder.

## Your own equilibrium

The same scan, on an eqdsk with a measured density.  The
[user guide](https://pablo-oyola.github.io/superNOVA/user_guide/equilibria.html)
covers the other sources (DESC, TokaMaker refinement) and every option:

```python
import supernova as sn

case = sn.Case.from_eqdsk(
    "shot.geqdsk", cocos_in=5,
    density=sn.profile_from_columns("ne.dat", units="m^-3"),   # psi_N, n_e
    boozer_domain=(0.005, 0.995), gamma_s=5 / 3, species="D",
)
scan = sn.Scan(case, sn.RadialGrid.uniform(spans=150), ntor=4)   # mmax, max_dm: defaults
modes = scan.run(sn.Band(50, 150, units="kHz"), window=2.5, nev=8,
                 sound_model="nova")
```

`sound_model="nova"` drops the parallel sound coupling, as NOVA-K's
`ksound = 1` does.  The full sound-Alfvén problem at γ = 5/3 (the default
`"ideal"` model) is not ready for production use yet; in the GUI the same
choice is the *sound* row of the Solver tab's *Problem* box.

`ne.dat` has one header line, then ψ_N and n_e columns; `x="rho_pol"` reads
a ρ_pol column instead.  Pressure comes from the eqdsk unless `pressure=` is
given as a `Profile`.  `boozer_domain` is in ρ_pol and keeps the Boozer
transform away from the axis and the separatrix.  `case.with_continuum(gamma_s=0)`
swaps the continuum without repeating the Boozer transform, which is the slow
step.  For a band crossed by many continuum branches, use the multigrid
classification pipeline (`supernova.MultigridSolver`, or `supernova solve`
below).

## Command line

```bash
supernova solve --equilibrium shot.geqdsk --density ne.dat --cocos-in 5 \
                --ntor 4 --frequency-khz 50 150 --artifact-dir runs/n4
supernova reclassify --artifact-dir runs/n4    # re-sort saved roots, no SLEPc
supernova reacquire  --artifact-dir runs/n4    # follow roots lost between levels
supernova gui [session.h5]
```

`solve` runs the multigrid scan: three radial levels, each root tracked
across them and classified from that evidence.  `supernova <command> --help`
lists every option.

`OMP_NUM_THREADS` controls the OpenMP threads used by MUMPS and OpenBLAS.
Under `mpiexec`, keep `OMP_NUM_THREADS × ranks` at or below the core count.
The GUI's Solver tab sets its own split under *Parallel windows*.

## Package layout

| Module | Contents |
|---|---|
| `supernova.api` | `Case`, `Scan`, `Band`, `RadialGrid`, `ModeSet`: the script-facing surface |
| `supernova.equilibrium` | `superEquilibrium`: Boozer coordinates and metric from an eqdsk |
| `supernova.cases.itpa` | `ITPAEq`, the analytic ITPA TAE equilibrium |
| `supernova.continuum` | `superContinuum`: the sound-Alfvén continuum |
| `supernova.radial` | B-spline finite elements, graded meshes, radial profiles |
| `supernova.operators` | block assembly, boundary conditions, projections |
| `supernova.eigen` | SLEPc/PETSc backends (GEVP for γ = 0, PEP otherwise) |
| `supernova.solve` | `StabilitySolver`, `MultigridSolver`, `modelocate`, gap estimates |
| `supernova.refine` | TokaMaker fixed-boundary re-solve of an eqdsk |
| `supernova.interop` | DESC equilibria |
| `supernova.gui` | the desktop window; `python -m supernova.gui.solver` (or `.equilibrium`, `.continuum`) opens one tab |
| `supernova.app` | the workflow behind `supernova solve/reclassify/reacquire` |

## Tests

```bash
pip install -e ".[dev]"
pytest tests
```

`tests/test_itpa_acceptance.py` runs the quick-start case end to end against
reference frequencies.
<!-- docs-guide-end -->

## Conventions

See [CONVENTIONS.md](CONVENTIONS.md) for the coordinate, sign and Fourier
conventions the physics layers share.

## License

MIT. See [LICENSE](LICENSE).
