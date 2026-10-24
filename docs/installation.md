# Installation reference

This page explains what `tools/install_stack.py` does, why each choice was
made, and the conflicts it works around.  The short version is the
Installation section of the
[README](https://github.com/pablo-oyola/superNOVA#installation).

## What gets built

| Layer | Source | Version | How |
|---|---|---|---|
| PETSc | PyPI sdist `petsc` | 3.25.3 | compiled by pip with `PETSC_CONFIGURE_OPTIONS` |
| SLEPc | PyPI sdist `slepc` | 3.25.1 | compiled by pip against that PETSc |
| petsc4py / slepc4py | PyPI sdists | 3.25.3 / 3.25.1 | Cython extensions built with `PETSC_DIR`/`SLEPC_DIR` set |
| mpi4py (optional) | PyPI sdist | 4.x | built with the same `mpicc` |
| pyCOCOS | GitHub, pinned commit | `8e9aacb` | direct git reference in `pyproject.toml` |
| pynova | this repository | | plain `pip install .` |

PETSc configure options (fixed in `PETSC_BASE_OPTIONS`):

| Option | Why |
|---|---|
| `--with-scalar-type=complex --with-precision=double` | the NEP/PEP operators are complex; `PETSc.ScalarType` becomes `complex128` |
| `--download-mumps --download-scalapack --download-metis --download-parmetis` | MUMPS LU is the production shift-and-invert factorisation; ScaLAPACK is a hard MUMPS dependency, METIS/ParMETIS give it its orderings |
| `--with-openmp=1` | enables OpenMP in PETSc, MUMPS (`download-mumps-openmp` defaults on) and the downloaded OpenBLAS (`USE_OPENMP=1`) |
| `--download-hwloc` | with pthreads and an MPI-3 MPI this defines `PETSC_HAVE_OPENMP_SUPPORT`, which unlocks `-mat_mumps_use_omp_threads` |
| `--download-openblas` | a BLAS/LAPACK the build fully controls; `--blas system` or `--blas dir=/path` overrides it |
| `--with-debugging=0 --COPTFLAGS=-O3 ...` | optimised build; `--optflags` overrides, e.g. `-O3 -march=native` for a machine-local build |
| `--with-fortran-bindings=0 --with-x=0` | not needed by petsc4py, shortens the build |

The petsc sdist forbids `--with-cc`, `--with-cxx`, `--with-fc` and
`--with-mpi-dir`; compilers come from `MPICC`, `MPICXX`, `MPIFORT`, which the
script sets from `mpicc`/`mpicxx`/`mpifort` on `PATH` or, with
`--download-mpich`, from `gcc`/`g++`/`gfortran` so that PETSc builds MPICH
itself.  In that case `mpiexec` lives under the PETSc prefix:

```bash
PETSC_DIR=$(python -m petsc --prefix)
$PETSC_DIR/bin/mpiexec -n 2 python -m supernova solve ...
```

## Why a script and not one `pip install`

1. `pip install pynova[solver]` cannot pass configure options or order the
   builds.  pip builds every wheel before installing any, so petsc4py's build
   would not see the freshly built `petsc` package.  When `PETSC_DIR` is unset
   petsc4py adds `petsc` to its *setup_requires* and setuptools would then
   download and rebuild PETSc a second time inside the isolated build
   environment, with default (real-scalar, no MUMPS) options.  The script
   installs `petsc`, reads its prefix with `python -m petsc --prefix`, and
   exports `PETSC_DIR` (and later `SLEPC_DIR`) before each dependent step.
2. pip caches built wheels by source URL, not by configure options.  A second
   run with different options would silently reuse the first PETSc.  Every
   stack step therefore runs with `--no-cache-dir`.
3. A `PETSC_DIR`/`SLEPC_DIR` exported by an activated conda environment would
   redirect the source builds to the conda libraries.  The script unsets both
   for the build steps.

## Reproducibility layers

* Compiled stack: exact pins in `STACK_PINS`, exact configure options, all
  third-party libraries downloaded by PETSc's configure at pinned versions.
* Python layer: `pyproject.toml` gives ranges; `requirements/constraints.txt`
  (generated from a validated environment with `pip freeze`) pins them and is
  applied automatically when present.  Pass `--no-constraints` to resolve
  fresh, then regenerate the file.
* Build knobs are command-line flags, so a rerun with the same flags is the
  same build.  `--dry-run` prints the exact commands and configure options.

What is *not* reproduced: the C/C++/Fortran compilers and the system MPI.
Pin those with your OS package manager or use `--download-mpich`.

## Conflicts found and how they are handled

| Conflict | Handling |
|---|---|
| **`pycocos` name collision.** PyPI already has an unrelated `pycocos` (0.2.x); the old `dependencies = ["pycocos"]` would install it. | Direct git reference pinned to commit `8e9aacb` (current `origin/master`). Direct references are fine for local/git installs but block uploading `pynova` to PyPI. Bump the SHA when pyCOCOS moves; the working copy of pyCOCOS had three uncommitted files at the time of pinning, which are not in the pin. |
| **`pynova` name collision.** PyPI also has an unrelated `pynova` (0.0.4). | No change; `pip install .`/git installs ignore PyPI. Rename before any PyPI release. |
| **DESC pins.** `desc-opt` 0.17.2 caps `scipy<=1.17.1`; the reference conda env runs scipy 1.18 with 0.17.2, which pip would refuse to resolve. 0.17.3 raises the caps to `scipy<=1.18.0`, `numpy<=2.5.1`, `matplotlib<=3.11.1`. DESC also pulls jax, jax-finufft, diffrax, equinox, optax, plotly, scikit-image and ~20 more packages. | `desc` extra pinned to `desc-opt>=0.17.3,<0.18` and kept optional; `import supernova` never imports `supernova.interop.desc`. |
| **numba/numpy.** numba 0.65 requires `numpy<2.5`. | Core `numpy>=1.26,<2.5`, consistent with DESC's cap. |
| **matplotlib was listed as optional (`plot` extra) but is imported at module level** by `continuum.py`, `isotopes.py`, `radial/profile.py`, `radial/bspline.py`. | Promoted to a core dependency; the `plot` extra is gone. Making those imports lazy would allow a headless install and is a follow-up. |
| **`tqdm` and `findiff` were imported but not declared.** | Added to core dependencies. |
| **`colorama`** is optional at runtime (`_logger.py` guards it); DESC pins it `<=0.4.6`. | Left undeclared; nothing forces a newer version. |
| **petsc4py/slepc4py had no declared relation to the package.** | New `solver` extra with `>=3.25,<3.26` so `pip check` sees the contract. It never triggers a build on its own (the script installs the pinned versions first). |
| **`build/lib/` was committed** (a stale copy of the package) and `README.md` was referenced by `pyproject.toml` but missing, which only warned. | `build/` untracked and ignored together with `dist/`, `.venv/`, `*.whl`; README added. |
| **Second PETSc in the same interpreter.** Installing on top of a conda env that ships conda-forge PETSc (real or complex) leaves two `petsc4py` builds and `LD_LIBRARY_PATH` deciding which library loads. | The script refuses `--no-venv` when `petsc4py` is already importable unless `--force`, and defaults to a fresh venv. |
| **MPI without Fortran wrappers** (some distro `mpich` builds) breaks MUMPS/ScaLAPACK. | Prerequisite check requires `mpifort`/`mpif90`, or `--download-mpich`. |
| **OpenFUSIONToolkit** (TokaMaker refinement) is not on PyPI. | `tokamaker` extra only declares `freeqdsk`; `supernova.refine` raises a pointed `ImportError` without OFT. |
| **`-mat_mumps_use_omp_threads`** needs `PETSC_HAVE_OPENMP_SUPPORT` (hwloc + pthreads + MPI-3 shared memory). | hwloc is downloaded; the verification prints `openmp_support` and warns when it is off. Plain OpenMP inside MUMPS works regardless. |
| **MUMPS workspace** under SLEPc 3.25 needs `st_mat_mumps_icntl_14=200` on larger cases (INFOG(1)=-9 otherwise). | Runtime option, unchanged by the install; pass it through `PetscOptions.extra`. |
| **Thread oversubscription.** OpenBLAS, MUMPS and numba all spawn threads. | Set `OMP_NUM_THREADS`; numba's pool is separate (`NUMBA_NUM_THREADS`). Under `mpiexec` keep threads × ranks ≤ cores. |

## Verification performed by the script

After the stack is installed it imports petsc4py/slepc4py, checks
`petscconf.h` for `PETSC_USE_COMPLEX`, `PETSC_HAVE_MUMPS`, `PETSC_HAVE_OPENMP`,
`PETSC_HAVE_SCALAPACK` and real MPI, and solves a small complex system through
a MUMPS LU, requiring a residual below 1e-10.  It then imports `supernova`
and `pycocos`, and `desc` when `--desc` was given.

## Rebuilding only pynova

```bash
python tools/install_stack.py --venv .venv --skip-stack -e --desc
```

reinstalls pynova (editable) and its Python dependencies without touching
the compiled stack.
