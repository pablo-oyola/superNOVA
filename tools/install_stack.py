#!/usr/bin/env python3
"""Reproducible pip installation of superNOVA and its compiled solver stack.

The eigenvalue backend needs PETSc and SLEPc built with **complex scalars**,
the **MUMPS** direct solver and **OpenMP**.  No PyPI wheel ships that
combination, so this script drives the official ``petsc``/``slepc`` source
distributions through pip with the right configure options, then installs
``petsc4py``/``slepc4py`` against the result and finally ``pynova`` itself.

Run it with the interpreter that should seed the virtual environment::

    python3.12 tools/install_stack.py --venv .venv            # solver only
    python3.12 tools/install_stack.py --venv .venv --desc     # + DESC
    python3.12 tools/install_stack.py --venv .venv --download-mpich  # no MPI

Every version of the compiled stack is pinned in ``STACK_PINS`` below; the
pure-Python dependencies are constrained by ``requirements/constraints.txt``
when that file exists.  Knobs that change the build are command-line flags,
not environment variables, so a rerun with the same flags gives the same
build.  See ``docs/installation.md`` for the reasoning behind each option.
"""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Compiled stack.  petsc4py/slepc4py must share the minor version of the
# library they wrap; slepc4py additionally requires petsc4py<3.26.
STACK_PINS = {
    "petsc": "3.25.3",
    "petsc4py": "3.25.3",
    "slepc": "3.25.1",
    "slepc4py": "3.25.1",
    "mpi4py": ">=4.0,<5",
}

# PETSc configure options that define the solver build.  Compilers are *not*
# listed here on purpose: the petsc sdist forbids --with-cc/--with-mpi-dir and
# takes them from MPICC/MPICXX/MPIFORT instead.
PETSC_BASE_OPTIONS = [
    "--with-scalar-type=complex",
    "--with-precision=double",
    "--with-debugging=0",
    "--with-fortran-bindings=0",
    "--with-x=0",
    "--with-openmp=1",
    "--download-hwloc",          # with pthread+MPI-3 shm -> PETSC_HAVE_OPENMP_SUPPORT
    "--download-mumps",          # picks up OpenMP automatically (download-mumps-openmp=1)
    "--download-scalapack",
    "--download-metis",
    "--download-parmetis",
]

VERIFY_SNIPPET = r"""
import json, sys
import petsc4py, slepc4py
petsc4py.init(sys.argv[1:])
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
cfg = petsc4py.get_config()
petsc_dir = cfg["PETSC_DIR"]
conf = open(f"{petsc_dir}/include/petscconf.h").read()
have = lambda m: f"#define {m} 1" in conf
info = {
    "petsc": ".".join(map(str, PETSc.Sys.getVersion())),
    "slepc": ".".join(map(str, SLEPc.Sys.getVersion())),
    "scalar": PETSc.ScalarType.__name__,
    "complex": have("PETSC_USE_COMPLEX"),
    "mumps": have("PETSC_HAVE_MUMPS"),
    "openmp": have("PETSC_HAVE_OPENMP"),
    "openmp_support": have("PETSC_HAVE_OPENMP_SUPPORT"),
    "mpi": not have("PETSC_HAVE_MPIUNI"),
    "petsc_dir": petsc_dir,
}
# A real complex LU through MUMPS on a small matrix.
n = 6
A = PETSc.Mat().createAIJ([n, n], nnz=3)
for i in range(n):
    A[i, i] = 4.0 + 1.0j
    if i > 0: A[i, i - 1] = -1.0 + 0.5j
    if i < n - 1: A[i, i + 1] = -1.0 - 0.5j
A.assemble()
ksp = PETSc.KSP().create(); ksp.setOperators(A)
ksp.setType("preonly"); pc = ksp.getPC(); pc.setType("lu")
pc.setFactorSolverType("mumps"); ksp.setFromOptions()
b = A.createVecRight(); b.set(1.0 + 1.0j); x = b.duplicate()
ksp.solve(b, x)
r = b.duplicate(); A.mult(x, r); r.axpy(-1.0, b)
info["mumps_residual"] = float(r.norm())
info["ok"] = all(info[k] for k in ("complex", "mumps", "openmp", "mpi")) and info["mumps_residual"] < 1e-10
print("STACK_INFO " + json.dumps(info))
"""


class Installer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.log_path = Path(args.log).resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(self.log_path, "a", buffering=1)
        self.python = self._resolve_python()
        self.jobs = args.jobs or max(1, (os.cpu_count() or 2) - 1)

    # ------------------------------------------------------------------ utils
    def say(self, msg: str) -> None:
        print(f"[install_stack] {msg}", flush=True)
        self.log.write(f"[install_stack] {msg}\n")

    def run(self, cmd: list[str], env: dict | None = None, capture: bool = False) -> str:
        # A None value unsets the variable: a PETSC_DIR/SLEPC_DIR exported by
        # an activated conda env must not leak into the source builds.
        full_env = os.environ.copy()
        for k, v in (env or {}).items():
            if v is None:
                full_env.pop(k, None)
            else:
                full_env[k] = v
        printable = " ".join(shlex.quote(c) for c in cmd)
        if env:
            printable = " ".join(f"{k}={shlex.quote(v) if v is not None else '<unset>'}"
                                 for k, v in env.items()) + " " + printable
        self.say(f"$ {printable}")
        if self.args.dry_run:
            return ""
        if capture:
            out = subprocess.run(cmd, env=full_env, check=True, text=True, capture_output=True).stdout
            self.log.write(out)
            return out
        proc = subprocess.Popen(cmd, env=full_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            self.log.write(line)
            if self.args.verbose:
                sys.stdout.write(line)
        proc.wait()
        if proc.returncode != 0:
            raise SystemExit(f"command failed (exit {proc.returncode}); see {self.log_path}")
        return ""

    def pip(self, *pkgs: str, env: dict | None = None, extra: list[str] | None = None) -> None:
        cmd = [str(self.python), "-m", "pip", "install", "--no-cache-dir", *(extra or []), *pkgs]
        self.run(cmd, env=env)

    def _resolve_python(self) -> Path:
        if self.args.no_venv:
            return Path(sys.executable)
        venv = Path(self.args.venv).resolve()
        exe = venv / ("Scripts" if os.name == "nt" else "bin") / "python"
        if not exe.exists():
            import importlib.util
            if importlib.util.find_spec("ensurepip") is None:
                v = f"{sys.version_info[0]}.{sys.version_info[1]}"
                raise SystemExit(
                    f"{sys.executable} cannot create a venv: the ensurepip module is missing "
                    f"(Debian/Ubuntu: `sudo apt install python{v}-venv`). Alternatively seed the "
                    "environment from another interpreter, e.g. a conda or pyenv python.")
            self.say(f"creating virtual environment at {venv} with {sys.executable}")
            if not self.args.dry_run:
                subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        return exe

    # --------------------------------------------------------------- checks
    def check_prerequisites(self) -> dict:
        if sys.version_info < (3, 10):
            raise SystemExit("Python >= 3.10 is required to seed the environment.")
        if platform.system() not in ("Linux", "Darwin"):
            raise SystemExit("The compiled stack is only supported on Linux and macOS.")
        missing = [t for t in ("make", "cmake", "git") if not shutil.which(t)]
        fc = os.environ.get("FC") or shutil.which("gfortran") or shutil.which("flang")
        if not fc:
            missing.append("gfortran (MUMPS and ScaLAPACK are Fortran)")
        if not (shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")):
            missing.append("a C compiler")
        if not (shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")):
            missing.append("a C++ compiler")
        mpi = {}
        if self.args.download_mpich:
            # Plain compilers; PETSc builds MPICH and its wrappers itself.
            mpi = {
                "MPICC": shutil.which("gcc") or shutil.which("cc"),
                "MPICXX": shutil.which("g++") or shutil.which("c++"),
                "MPIFORT": fc,
            }
        else:
            mpicc = os.environ.get("MPICC") or shutil.which("mpicc")
            if not mpicc:
                missing.append("mpicc (install an MPI, or pass --download-mpich)")
            else:
                mpi = {
                    "MPICC": mpicc,
                    "MPICXX": os.environ.get("MPICXX") or shutil.which("mpicxx") or shutil.which("mpic++"),
                    "MPIFORT": os.environ.get("MPIFORT") or shutil.which("mpifort") or shutil.which("mpif90"),
                }
                if not mpi["MPIFORT"]:
                    missing.append("mpifort/mpif90 (the MPI must have Fortran wrappers for MUMPS)")
        if missing:
            raise SystemExit("missing prerequisites:\n  - " + "\n  - ".join(str(m) for m in missing))
        if self.args.no_venv and not self.args.force:
            probe = subprocess.run([str(self.python), "-c", "import petsc4py, sys; print(petsc4py.__file__)"],
                                   capture_output=True, text=True)
            if probe.returncode == 0:
                raise SystemExit(
                    "petsc4py is already importable in this interpreter:\n  " + probe.stdout.strip()
                    + "\nA second PETSc (e.g. conda-forge's) would shadow the pip build. "
                    "Use a fresh --venv, or pass --force to install on top of it.")
        return {k: v for k, v in mpi.items() if v}

    # ---------------------------------------------------------------- stack
    def petsc_options(self) -> list[str]:
        opts = list(PETSC_BASE_OPTIONS)
        opt = self.args.optflags
        opts += [f"--COPTFLAGS={opt}", f"--CXXOPTFLAGS={opt}", f"--FOPTFLAGS={opt}"]
        opts.append(f"--with-make-np={self.jobs}")
        if self.args.download_mpich:
            opts.append("--download-mpich")
        if self.args.blas == "download":
            opts.append("--download-openblas")          # USE_OPENMP=1 because --with-openmp
        elif self.args.blas.startswith("dir="):
            opts.append("--with-blaslapack-dir=" + self.args.blas[4:])
        # "system": let configure autodetect BLAS/LAPACK.
        opts += shlex.split(self.args.petsc_extra or "")
        return opts

    def install_stack(self, mpi_env: dict) -> None:
        a = self.args
        self.pip("pip>=24", "setuptools>=70", "wheel", extra=["--upgrade"])

        env = dict(mpi_env)
        env["PETSC_CONFIGURE_OPTIONS"] = " ".join(shlex.quote(o) for o in self.petsc_options())
        env["PETSC_DIR"] = None
        env["PETSC_ARCH"] = ""
        self.say(f"PETSc configure options:\n" + textwrap.indent("\n".join(self.petsc_options()), "    "))
        self.pip(f"petsc=={STACK_PINS['petsc']}", env=env)

        petsc_dir = self.prefix_of("petsc")
        self.assert_config(petsc_dir)
        env = {"PETSC_DIR": petsc_dir, "PETSC_ARCH": ""}
        self.pip(f"petsc4py=={STACK_PINS['petsc4py']}", env=env)

        env["SLEPC_DIR"] = None
        env["SLEPC_CONFIGURE_OPTIONS"] = a.slepc_extra or ""
        self.pip(f"slepc=={STACK_PINS['slepc']}", env=env)
        slepc_dir = self.prefix_of("slepc")
        env = {"PETSC_DIR": petsc_dir, "PETSC_ARCH": "", "SLEPC_DIR": slepc_dir}
        self.pip(f"slepc4py=={STACK_PINS['slepc4py']}", env=env)

        if a.with_mpi4py:
            mpicc = f"{petsc_dir}/bin/mpicc" if a.download_mpich else mpi_env["MPICC"]
            self.pip(f"mpi4py{STACK_PINS['mpi4py']}", env={"MPICC": mpicc})

    def prefix_of(self, pkg: str) -> str:
        if self.args.dry_run:
            return f"<{pkg} prefix>"
        return self.run([str(self.python), "-m", pkg, "--prefix"], capture=True).strip().splitlines()[-1]

    def assert_config(self, petsc_dir: str) -> None:
        if self.args.dry_run:
            return
        conf = Path(petsc_dir, "include", "petscconf.h").read_text()
        wanted = ["PETSC_USE_COMPLEX", "PETSC_HAVE_MUMPS", "PETSC_HAVE_OPENMP", "PETSC_HAVE_SCALAPACK"]
        bad = [m for m in wanted if f"#define {m} 1" not in conf]
        if bad:
            raise SystemExit(f"PETSc was built without: {', '.join(bad)}; see {self.log_path}")
        if "#define PETSC_HAVE_OPENMP_SUPPORT 1" not in conf:
            self.say("note: PETSC_HAVE_OPENMP_SUPPORT is off (MPI without MPI-3 shared memory?); "
                     "-mat_mumps_use_omp_threads will be unavailable, plain OpenMP in MUMPS still works.")

    # --------------------------------------------------------------- pynova
    def install_pynova(self) -> None:
        a = self.args
        extras = [e for e, on in (("desc", a.desc), ("tokamaker", a.tokamaker), ("dev", a.dev)) if on]
        target = str(REPO_ROOT) + (f"[{','.join(extras)}]" if extras else "")
        extra: list[str] = []
        constraints = Path(a.constraints) if a.constraints else REPO_ROOT / "requirements" / "constraints.txt"
        if constraints.exists() and not a.no_constraints:
            extra += ["-c", str(constraints)]
        if a.pycocos:
            self.say(f"installing pycocos from {a.pycocos} ahead of pynova")
            self.pip(a.pycocos, extra=extra)
        if a.editable:
            extra.append("-e")
        self.pip(target, extra=extra)

    # --------------------------------------------------------------- verify
    def verify(self) -> None:
        if self.args.dry_run:
            return
        out = self.run([str(self.python), "-c", VERIFY_SNIPPET], capture=True)
        line = [l for l in out.splitlines() if l.startswith("STACK_INFO ")]
        if not line:
            raise SystemExit("verification produced no report; see " + str(self.log_path))
        import json
        info = json.loads(line[-1][len("STACK_INFO "):])
        self.say("compiled stack: " + ", ".join(f"{k}={v}" for k, v in info.items() if k != "petsc_dir"))
        if not info["ok"]:
            raise SystemExit("verification failed; see " + str(self.log_path))
        self.run([str(self.python), "-c", "import supernova, pycocos; print('pynova', supernova.__version__)"], capture=True)
        if self.args.desc:
            self.run([str(self.python), "-c", "import desc, supernova.interop.desc; print('desc', desc.__version__)"], capture=True)

    # ------------------------------------------------------------------ main
    def main(self) -> None:
        mpi_env = self.check_prerequisites()
        self.say(f"interpreter: {self.python}")
        self.say(f"log: {self.log_path}")
        if not self.args.skip_stack:
            self.install_stack(mpi_env)
        self.install_pynova()
        self.verify()
        self.say("done.")
        if not self.args.no_venv:
            self.say(f"activate with: source {Path(self.args.venv).resolve()}/bin/activate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = p.add_mutually_exclusive_group()
    where.add_argument("--venv", default=".venv", help="virtual environment to create/use (default: .venv)")
    where.add_argument("--no-venv", action="store_true", help="install into the interpreter running this script")
    p.add_argument("--force", action="store_true", help="with --no-venv: proceed even if petsc4py is already present")
    p.add_argument("--editable", "-e", action="store_true", help="install pynova in editable (develop) mode")
    p.add_argument("--desc", action="store_true", help="also install the optional DESC equilibrium reader")
    p.add_argument("--tokamaker", action="store_true", help="also install the TokaMaker refinement extra")
    p.add_argument("--dev", action="store_true", help="also install the test dependencies")
    p.add_argument("--with-mpi4py", action="store_true", help="also build mpi4py against the same MPI")
    p.add_argument("--download-mpich", action="store_true",
                   help="let PETSc build its own MPICH instead of using mpicc from PATH")
    p.add_argument("--blas", default="download",
                   help="'download' (OpenBLAS with OpenMP, default), 'system' (autodetect), or 'dir=/path'")
    p.add_argument("--optflags", default="-O3", help="optimisation flags for C/C++/Fortran (default: -O3)")
    p.add_argument("--jobs", type=int, help="parallel build jobs (default: cores-1)")
    p.add_argument("--petsc-extra", help="extra PETSc configure options, appended verbatim")
    p.add_argument("--slepc-extra", help="extra SLEPc configure options, appended verbatim")
    p.add_argument("--pycocos", help="path or pip URL to install pycocos from before pynova")
    p.add_argument("--constraints", help="pip constraints file (default: requirements/constraints.txt)")
    p.add_argument("--no-constraints", action="store_true", help="ignore the constraints file")
    p.add_argument("--skip-stack", action="store_true", help="only (re)install pynova; assume the stack is present")
    p.add_argument("--log", default="install-stack.log", help="build log (default: ./install-stack.log)")
    p.add_argument("--verbose", "-v", action="store_true", help="stream build output to the terminal")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    return p.parse_args(argv)


if __name__ == "__main__":
    Installer(parse_args()).main()
