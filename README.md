# superNOVA

Linear ideal-MHD stability of tokamak equilibria: Boozer coordinates,
continuum spectrum and global eigenmodes on a radial finite-element ×
poloidal Fourier basis.  PyPI-style name `pynova`, import name `supernova`.

## Installation

The solver needs PETSc and SLEPc built with **complex scalars**, the **MUMPS**
LU factorisation and **OpenMP**.  No binary wheel provides that combination,
so the stack is compiled from the official `petsc`/`slepc` source
distributions by one script that pins every version:

```bash
git clone git@github.com:pablo-oyola/superNOVA-2.git supernova
cd supernova
python3.12 tools/install_stack.py --venv .venv          # ~15-30 min, mostly PETSc
source .venv/bin/activate
supernova --help
```

Add `--desc` for the optional DESC equilibrium reader, `--tokamaker` for the
TokaMaker refinement layer, `--dev` for the test dependencies, `-e` for an
editable install, and `--download-mpich` on a machine without an MPI.
`--dry-run` prints the exact commands.  Full reference, prerequisites and the
list of known conflicts: [docs/installation.md](docs/installation.md).

Prerequisites: Python 3.10-3.13 with `venv`, gcc/g++/gfortran, make, cmake,
git, and an MPI with Fortran wrappers (`mpicc`, `mpifort`) unless you pass
`--download-mpich`.  On Debian/Ubuntu:

```bash
sudo apt install build-essential gfortran cmake git python3-venv libopenmpi-dev
```

### Without the compiled stack

`pip install .` alone installs the pure-Python layers (equilibrium, Boozer
coordinates, continuum, verification).  Anything that reaches the eigenvalue
backend raises an `ImportError` naming `petsc4py`/`slepc4py`.

## Running

```bash
supernova solve --help          # also: reclassify, reacquire
python -m supernova --help
```

`OMP_NUM_THREADS` controls the OpenMP threads used by MUMPS and OpenBLAS.
Under `mpiexec`, keep `OMP_NUM_THREADS × ranks` at or below the core count.

## Conventions

See [CONVENTIONS.md](CONVENTIONS.md) for the coordinate, sign and Fourier
conventions the physics layers share.
