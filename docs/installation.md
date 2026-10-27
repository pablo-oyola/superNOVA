# Installation reference

This page explains what `supernova install-solver` builds, why each configure
option is there, and the pitfalls known so far.  The short version is the
Installation section of the
[README](https://github.com/pablo-oyola/superNOVA#installation).

## What gets built

`pip install pynova` installs the pure-Python package.  `supernova
install-solver` then runs pip four times in the same environment:

| Step | Package | Version | How |
|---|---|---|---|
| 1 | PETSc, PyPI sdist `petsc` | newest 3.25.x | compiled with the configure options below |
| 2 | petsc4py | same as PETSc | Cython extension, `PETSC_DIR` set to step 1 |
| 3 | SLEPc, PyPI sdist `slepc` | newest 3.25.x | compiled against step 1 |
| 4 | slepc4py | same as SLEPc | Cython extension, `PETSC_DIR` and `SLEPC_DIR` set |

pyCOCOS comes from GitHub at a pinned commit (`8e9aacb`), as a direct git
reference in `pyproject.toml`.

Each step sets `PETSC_DIR` (and `SLEPC_DIR`) to what the previous one
installed, so every library is compiled exactly once, and every step bypasses
the pip cache.  pip keys a built wheel by its source URL, not by its configure
options, so a cached wheel could be an older real-scalar PETSc.

### Why not one `pip install`

The `petsc` sdist builds its wheel in two moves.  It links every downloaded
package into `libpetsc.so`, then deletes the static archives
(`lib/*.a`) to keep the wheel small.  `lib/petsc/conf/petscvariables` still
lists those archives, though: MUMPS is always built static, so the link line
keeps `-lsmumps -ldmumps -lcmumps -lzmumps -lmumps_common -lpord -lmpiseq`.
SLEPc links through those variables, so `pip install petsc slepc` with MUMPS
stops with `cannot find -lsmumps`.  PETSc 3.25 and 3.26 both do this.

Between steps 1 and 2, `install-solver` removes every `-l` flag whose library
is in none of the `-L` directories and that the compilers cannot find
either.  System libraries such as `-lm` and `-lgfortran` stay.  The MUMPS code
is already inside `libpetsc.so`, so nothing is lost.  It is
`drop_missing_libraries` in `supernova/eigen/stack.py`.

Only the minor version is fixed (`3.25`), because the solver is validated on
it.  The bindings are pinned to the exact version of the library they wrap.

## The configure options

| Option | Why |
|---|---|
| `--with-scalar-type=complex` | required: the operators are complex, `PETSc.ScalarType` becomes `complex128` |
| `--download-mumps` | required: MUMPS is the LU behind every shift-and-invert window |
| `--download-metis` | the fill-reducing ordering MUMPS uses |
| `--download-suitesparse` | the `spqr` QR factorization, the fallback when MUMPS runs out of memory |
| `--with-mpi=0` | superNOVA runs in one process; MUMPS then uses its sequential MPI stub, which removes MPI, ScaLAPACK and ParMETIS from the prerequisites |
| `--with-openmp=1` | threads in MUMPS and OpenBLAS, set with `OMP_NUM_THREADS` |
| `--download-openblas` | a BLAS/LAPACK the build controls |
| `--download-openblas-make-options=DYNAMIC_ARCH=1` | OpenBLAS picks its kernels at run time, so the build runs on nodes with another CPU |
| `--download-cmake` | METIS and SuiteSparse need CMake 3.22 or newer; drop the option when one is on `PATH` to save a few minutes |
| `--with-fortran-bindings=0 --with-x=0` | not needed by petsc4py; shortens the build and drops X11 |
| `--COPTFLAGS=-O3 --CXXOPTFLAGS=-O3 --FOPTFLAGS=-O3` | optimised build with no `-march`, so it stays portable; `-march=native` gives a machine-local build |

The options live in `PETSC_CONFIGURE_OPTIONS` in `supernova/eigen/stack.py`.
`--extra-options` appends to them, and a later option overrides an earlier
one, so `--extra-options='--COPTFLAGS="-O3 -march=native"'` gives a
machine-local build.

The `petsc` sdist adds `--with-debugging=0` and `--with-shared-libraries=1`
itself.  It refuses `--with-cc`, `--with-cxx`, `--with-fc` and
`--with-mpi-dir`.  With `--with-mpi=0` PETSc's configure takes the first
`gcc`, `g++` and `gfortran` on `PATH`, so on a cluster load the compiler
module before running `install-solver`.  `make` runs on every core; add
`--extra-options=--with-make-np=8` on a login node that limits processes.

### An MPI build instead

The solver never uses more than one rank, so MPI buys nothing today.  If it
is ever needed, pass `--extra-options="--with-mpi=1 --download-scalapack
--download-parmetis --download-hwloc"` with an MPI with Fortran wrappers on
`PATH` (`mpicc`, `mpicxx`, `mpifort`).  The `mpich` and `openmpi` wheels on
PyPI have no Fortran wrappers, so they cannot build MUMPS.

## Checking the result

`supernova check` (or `python -m supernova.eigen.stack`) asks PETSc and SLEPc
what they were built with, then solves one small complex eigenproblem with
shift-and-invert through MUMPS.  Its exit status is 0 when the eigenvalue
solver can run:

```text
superNOVA solver stack
  PETSc               ok    3.25.6 (petsc4py 3.25.6)
  SLEPc               ok    3.25.2 (slepc4py 3.25.2)
  scalars             ok    complex128
  MUMPS               ok    yes
  shift-invert test   ok    eigenvalue error 4.6e-15, backward error 1.6e-16
  SuiteSparse (spqr)  ok    yes, residual 3.5e-16
  OpenMP              ok    yes, OMP_NUM_THREADS=unset, all cores
  MPI                 ok    no (serial build)
The eigenvalue solver can run.
```

Missing SuiteSparse or OpenMP is reported but does not fail the check.

## Rerunning and reinstalling

A rerun of `install-solver` reuses an installed PETSc that already has
complex scalars and MUMPS, so recovering from a failed SLEPc step takes
minutes, not the full build.  petsc4py, SLEPc and slepc4py are rebuilt on
every run.  `--force` rebuilds PETSc too.

Reinstalling pynova leaves the compiled stack alone: `pip install -e .` again
in a checkout, or add `--force-reinstall --no-deps` to the git install.

## Pitfalls and how they are handled

| Pitfall | Handling |
|---|---|
| **`pip install petsc slepc` fails with MUMPS.**  The petsc wheel deletes the static MUMPS archives but keeps their `-l` flags (see above). | `install-solver` removes the dangling flags between the PETSc and SLEPc builds. |
| **The wheel cache ignores configure options.**  pip caches the wheels it compiles by source URL, so a later install with other options silently reuses the old PETSc. | `install-solver` passes `--no-cache-dir` to every step. |
| **`PETSC_DIR` / `SLEPC_DIR` in the environment.**  Conda environments with conda-forge PETSc export them, and petsc4py then builds against that PETSc. | `install-solver` unsets both and sets its own for each step. |
| **A second PETSc in the same environment.**  A conda-forge `petsc4py` next to the pip one leaves the import order deciding which library loads. | Use a fresh environment.  `install-solver` reinstalls the bindings on every run, so the pip ones win. |
| **Python version.**  The compiled wheels are tied to the interpreter's minor version. | A new environment with another Python compiles the stack again. |
| **`pycocos` name collision.**  PyPI has an unrelated `pycocos` (0.2.x). | Direct git reference pinned to commit `8e9aacb`.  That blocks uploading `pynova` to PyPI; bump the SHA when pyCOCOS moves. |
| **`pynova` name collision.**  PyPI also has an unrelated `pynova` (0.0.4). | No change: git and local installs ignore PyPI.  Rename before any PyPI release. |
| **DESC pins.**  `desc-opt` 0.17.3 caps `scipy<=1.18.0`, `numpy<=2.5.1` and `matplotlib<=3.11.1`, and pulls JAX and about twenty more packages. | The `desc` extra is optional, and `import supernova` never imports `supernova.interop.desc`. |
| **numba and numpy.**  numba 0.65 requires `numpy<2.5`. | Core `numpy>=1.26,<2.5`, consistent with DESC's cap. |
| **matplotlib is imported at module level** by `continuum.py`, `isotopes.py`, `radial/profile.py` and `radial/bspline.py`. | A core dependency.  Making those imports lazy would allow a headless install. |
| **MUMPS workspace.**  Under SLEPc 3.25 the default `ICNTL(14)` runs out (INFOG(1) = -9). | A runtime option, `st_mat_mumps_icntl_14=200`, not an install matter. |
| **Thread oversubscription.**  OpenBLAS, MUMPS and numba all start threads. | Set `OMP_NUM_THREADS`; numba's pool is separate (`NUMBA_NUM_THREADS`). |
| **OpenFUSIONToolkit** (TokaMaker refinement) is not on PyPI. | The `tokamaker` extra only declares `freeqdsk`; `supernova.refine` raises a pointed `ImportError` without OFT. |
| **The GUI on a remote machine.**  PySide6 needs an X server and the xcb libraries; without them Qt stops with "Could not load the Qt platform plugin xcb". | Connect with `ssh -X` (or `-Y`), and ask for `libxcb-cursor0` and `libxkbcommon-x11-0` if they are missing.  Scripts and the CLI need neither. |
