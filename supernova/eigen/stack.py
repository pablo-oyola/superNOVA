"""Install and check the compiled PETSc/SLEPc stack the eigenvalue solver needs.

The solver needs PETSc and SLEPc built with complex scalars and the MUMPS
direct solver.  PyPI has both only as source releases, so pip compiles them.
:func:`install_solver` drives those pip builds in the right order and
:func:`check_stack` asks the result what it was built with, then runs one
small shift-and-invert eigenproblem through MUMPS, the path every frequency
window takes.

Why not one ``pip install``: the ``petsc`` sdist deletes every static archive
from its wheel but keeps their ``-l`` flags in ``petscvariables``.  MUMPS is
always built static, so SLEPc's build, which links with those flags, fails
with ``cannot find -lzmumps`` (PETSc 3.25 and 3.26).  The MUMPS code itself is
inside ``libpetsc.so``, so :func:`install_solver` removes the dangling flags
between the PETSc and the SLEPc builds.

Run them as ``supernova install-solver`` and ``supernova check``, or
``python -m supernova.eigen.stack [install]``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The PETSc/SLEPc minor release the solver is validated on.  petsc4py and
# slepc4py must match the library they wrap to the patch release.
STACK_SERIES = (3, 25)

# What the solver needs, and what makes the build self-contained and portable:
# serial (MUMPS takes its sequential MPI stub), its own CMake, BLAS and METIS,
# OpenBLAS kernels chosen at run time, no -march.  docs/installation.md gives
# the reason for each option.
PETSC_CONFIGURE_OPTIONS = (
    "--with-scalar-type=complex",
    "--with-mpi=0",
    "--with-openmp=1",
    "--with-fortran-bindings=0",
    "--with-x=0",
    "--download-cmake",
    "--download-openblas",
    "--download-openblas-make-options=DYNAMIC_ARCH=1",
    "--download-metis",
    "--download-mumps",
    "--download-suitesparse",
    "--COPTFLAGS=-O3",
    "--CXXOPTFLAGS=-O3",
    "--FOPTFLAGS=-O3",
)

_INSTALL_HINT = (
    "Run `supernova install-solver` (README, section 'Installation'); "
    "`supernova install-solver --force` rebuilds a PETSc that lacks complex "
    "scalars or MUMPS."
)


@dataclass(frozen=True)
class StackItem:
    """One line of the stack report.

    Attributes
    ----------
    name : str
        What was checked.
    value : str
        What was found.
    ok : bool
        Whether it is what the solver needs.
    required : bool
        A failed required item makes the stack unusable; a failed optional one
        only disables a feature.
    """

    name: str
    value: str
    ok: bool
    required: bool = True


def _petsc_has(PETSc, package: str) -> bool:
    try:
        return bool(PETSc.Sys.hasExternalPackage(package))
    except Exception:  # pragma: no cover - very old petsc4py
        return False


def _petscconf(petsc4py) -> str | None:
    """The text of the ``petscconf.h`` petsc4py was built against, if found."""
    try:
        cfg = petsc4py.get_config()
    except Exception:
        return None
    root = Path(cfg.get("PETSC_DIR", ""))
    for path in (root / cfg.get("PETSC_ARCH", "") / "include" / "petscconf.h",
                 root / "include" / "petscconf.h"):
        if path.is_file():
            return path.read_text()
    return None


def _defined(conf: str, macro: str) -> bool:
    return re.search(rf"^#define {macro} 1$", conf, flags=re.MULTILINE) is not None


def _tridiagonal(PETSc, n: int):
    """A small complex non-Hermitian tridiagonal matrix and its dense copy."""
    k = np.arange(n)
    diag = (k + 1.0) + 0.1j * k
    upper = np.full(n - 1, 0.3 - 0.2j)
    lower = np.full(n - 1, -0.25 + 0.15j)
    dense = np.diag(diag) + np.diag(upper, 1) + np.diag(lower, -1)
    mat = PETSc.Mat().createAIJ([n, n], nnz=3, comm=PETSc.COMM_SELF)
    for i in range(n):
        mat[i, i] = diag[i]
        if i + 1 < n:
            mat[i, i + 1] = upper[i]
            mat[i + 1, i] = lower[i]
    mat.assemble()
    return mat, dense


def _mumps_shift_invert(PETSc, SLEPc) -> tuple[float, float]:
    """Solve for the eigenvalue nearest a target with shift-and-invert + MUMPS.

    Returns the relative error against numpy's dense eigenvalues and SLEPc's
    backward error.
    """
    mat, dense = _tridiagonal(PETSc, 40)
    target = 7.3 + 0.6j
    eps = SLEPc.EPS().create(PETSc.COMM_SELF)
    eps.setOperators(mat)
    eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    eps.setDimensions(nev=1)
    eps.setTarget(target)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setTolerances(1.0e-12, 200)
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    ksp = st.getKSP()
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    eps.solve()
    if eps.getConverged() < 1:
        return float("inf"), float("inf")
    value = eps.getEigenvalue(0)
    backward = eps.computeError(0, SLEPc.EPS.ErrorType.BACKWARD)
    reference = np.linalg.eigvals(dense)
    nearest = reference[np.argmin(np.abs(reference - target))]
    eps.destroy()
    mat.destroy()
    return float(abs(value - nearest) / abs(nearest)), float(backward)


def _qr_residual(PETSc, solver: str) -> float:
    """Relative residual of one square solve through a QR factorization."""
    mat, _ = _tridiagonal(PETSc, 40)
    rhs = mat.createVecLeft()
    rhs.set(1.0 + 1.0j)
    sol = mat.createVecRight()
    ksp = PETSc.KSP().create(PETSc.COMM_SELF)
    ksp.setOperators(mat)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.QR)
    pc.setFactorSolverType(solver)
    ksp.solve(rhs, sol)
    res = rhs.duplicate()
    mat.mult(sol, res)
    res.axpy(-1.0, rhs)
    out = res.norm() / rhs.norm()
    ksp.destroy()
    mat.destroy()
    return float(out)


def check_stack() -> list[StackItem]:
    """Inspect the PETSc/SLEPc build and run a small MUMPS shift-and-invert.

    Returns
    -------
    list of StackItem
        One entry per check, required ones first.  The stack is usable when
        every required entry is ``ok``; see :func:`stack_ok`.
    """
    items: list[StackItem] = []
    try:
        import petsc4py
        from petsc4py import PETSc
    except ImportError as exc:
        return [StackItem("petsc4py", f"not importable ({exc})", False)]
    items.append(StackItem(
        "PETSc", f"{'.'.join(map(str, PETSc.Sys.getVersion()))} "
        f"(petsc4py {petsc4py.__version__})", True))
    try:
        import slepc4py
        from slepc4py import SLEPc
    except ImportError as exc:
        items.append(StackItem("slepc4py", f"not importable ({exc})", False))
        SLEPc = None
    else:
        items.append(StackItem(
            "SLEPc", f"{'.'.join(map(str, SLEPc.Sys.getVersion()))} "
            f"(slepc4py {slepc4py.__version__})", True))

    scalar = np.dtype(PETSc.ScalarType)
    items.append(StackItem("scalars", scalar.name, scalar.kind == "c"))
    has_mumps = _petsc_has(PETSc, "mumps")
    items.append(StackItem("MUMPS", "yes" if has_mumps else "no", has_mumps))

    if SLEPc is not None and has_mumps and scalar.kind == "c":
        try:
            rel, backward = _mumps_shift_invert(PETSc, SLEPc)
        except Exception as exc:  # report, do not crash the check
            items.append(StackItem("shift-invert test", f"failed ({exc})", False))
        else:
            ok = rel < 1.0e-10 and backward < 1.0e-10
            items.append(StackItem(
                "shift-invert test",
                f"eigenvalue error {rel:.1e}, backward error {backward:.1e}", ok))

    has_ss = _petsc_has(PETSc, "spqr")
    if has_ss and scalar.kind == "c":
        try:
            res = _qr_residual(PETSc, "spqr")
            items.append(StackItem(
                "SuiteSparse (spqr)", f"yes, residual {res:.1e}", res < 1.0e-10, False))
        except Exception as exc:
            items.append(StackItem("SuiteSparse (spqr)", f"failed ({exc})", False, False))
    else:
        items.append(StackItem(
            "SuiteSparse (spqr)", "no: factorization='spqr' unavailable", False, False))

    has_omp = _petsc_has(PETSc, "openmp")
    threads = os.environ.get("OMP_NUM_THREADS", "unset, all cores")
    items.append(StackItem(
        "OpenMP", f"yes, OMP_NUM_THREADS={threads}" if has_omp else "no: MUMPS runs on one thread",
        has_omp, False))
    conf = _petscconf(petsc4py)
    if conf is None:
        mpi = "unknown"
    else:
        mpi = "no (serial build)" if _defined(conf, "PETSC_HAVE_MPIUNI") else "yes"
    items.append(StackItem("MPI", mpi, True, False))
    return items


def stack_ok(items: list[StackItem]) -> bool:
    """Whether every required check in ``items`` passed."""
    return all(item.ok for item in items if item.required)


def format_report(items: list[StackItem]) -> str:
    """The report as aligned text, one check per line."""
    width = max(len(item.name) for item in items)
    lines = ["superNOVA solver stack"]
    for item in items:
        if item.ok:
            mark = "ok"
        else:
            mark = "FAIL" if item.required else "--"
        lines.append(f"  {item.name:<{width}}  {mark:<4}  {item.value}")
    if stack_ok(items):
        lines.append("The eigenvalue solver can run.")
    else:
        lines.append("The eigenvalue solver cannot run.  " + _INSTALL_HINT)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Installation
# --------------------------------------------------------------------------


def _series_spec(package: str) -> str:
    major, minor = STACK_SERIES
    return f"{package}>={major}.{minor},<{major}.{minor + 1}"


def _missing_tools() -> list[str]:
    wanted = {
        "make": ("make",),
        "a C compiler": ("gcc", "cc", "clang"),
        "a C++ compiler": ("g++", "c++", "clang++"),
        "a Fortran compiler (MUMPS is Fortran)": ("gfortran", "flang", "ifx"),
    }
    return [what for what, names in wanted.items()
            if not any(shutil.which(n) for n in names)]


class _Runner:
    """Runs pip in this interpreter and echoes every command."""

    def __init__(self, dry_run: bool, verbose: bool) -> None:
        self.dry_run = dry_run
        self.verbose = verbose

    def pip(self, *requirements: str, env: dict[str, str | None]) -> None:
        cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
        if self.verbose:
            cmd.append("-v")
        cmd += list(requirements)
        full_env = os.environ.copy()
        for key, value in env.items():
            if value is None:
                full_env.pop(key, None)
            else:
                full_env[key] = value
        shown = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items() if v is not None)
        print(f"$ {shown} {shlex.join(cmd)}".strip(), flush=True)
        if not self.dry_run:
            subprocess.run(cmd, env=full_env, check=True)

    def prefix(self, module: str) -> Path:
        if self.dry_run:
            return Path(f"<{module} prefix>")
        out = subprocess.run([sys.executable, "-m", module, "--prefix"],
                             capture_output=True, text=True, check=True).stdout
        return Path(out.strip().splitlines()[-1])


def _installed_version(package: str) -> str | None:
    importlib.invalidate_caches()
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _usable_petsc(prefix: Path) -> bool:
    """Whether an installed pip ``petsc`` already has complex scalars and MUMPS."""
    conf = prefix / "include" / "petscconf.h"
    if not conf.is_file():
        return False
    text = conf.read_text()
    return _defined(text, "PETSC_USE_COMPLEX") and _defined(text, "PETSC_HAVE_MUMPS")


def drop_missing_libraries(petsc_dir: str | os.PathLike) -> list[str]:
    """Remove ``-l`` flags whose library is gone from a pip-built PETSc.

    The ``petsc`` wheel deletes the static archives of the packages it
    downloaded (MUMPS among them) after linking them into ``libpetsc.so``,
    but ``petscvariables`` still lists them, which breaks every build that
    links through those variables, SLEPc's included.  A flag is dropped only
    when its library is in none of the ``-L`` directories and the compilers
    cannot find it either, so system libraries (``-lm``, ``-lgfortran``) stay.

    Returns
    -------
    list of str
        The library names that were removed; empty when nothing was missing.
    """
    petsc_dir = Path(petsc_dir)
    conf = petsc_dir / "lib" / "petsc" / "conf" / "petscvariables"
    text = conf.read_text()
    expanded = text.replace("${PETSC_DIR}", str(petsc_dir))
    dirs = {Path(d) for d in re.findall(r"(?<!\S)-L(\S+)", expanded)}
    dirs.add(petsc_dir / "lib")
    compilers = [c for c in (shutil.which(n) for n in ("gcc", "cc", "g++", "gfortran")) if c]

    def found(name: str) -> bool:
        for d in dirs:
            if any(d.glob(f"lib{name}.so*")) or (d / f"lib{name}.a").exists():
                return True
        for compiler in compilers:
            for ext in ("so", "a"):
                out = subprocess.run([compiler, f"-print-file-name=lib{name}.{ext}"],
                                     capture_output=True, text=True).stdout.strip()
                if os.path.isabs(out) and os.path.exists(out):
                    return True
        return False

    names = sorted(set(re.findall(r"(?<!\S)-l([\w+.-]+)", text)))
    missing = [n for n in names if not found(n)]
    if missing:
        flags = re.compile(r"[ \t]+-l(?:" + "|".join(map(re.escape, missing)) + r")(?=\s)")
        conf.write_text(flags.sub("", text))
    return missing


def install_solver(extra_options: str = "", force: bool = False,
                   dry_run: bool = False, verbose: bool = False) -> int:
    """Compile and install PETSc, petsc4py, SLEPc and slepc4py with pip.

    Everything goes into the running interpreter's environment, in four pip
    steps, each with ``PETSC_DIR``/``SLEPC_DIR`` pointing at the library the
    previous step installed, so every library is compiled exactly once.  The
    pip cache is bypassed: it keys built wheels by source URL, not by
    configure options, and could hand back an old real-scalar PETSc.

    Parameters
    ----------
    extra_options : str
        PETSc configure options appended to :data:`PETSC_CONFIGURE_OPTIONS`,
        e.g. ``"--with-make-np=8"`` on a login node that limits processes.
    force : bool
        Rebuild PETSc even if an installed one already has complex scalars
        and MUMPS.  Without it a rerun after a failed SLEPc step reuses it;
        petsc4py, SLEPc and slepc4py are rebuilt on every run.
    dry_run : bool
        Print the pip commands without running them.
    verbose : bool
        Stream the full compiler output (``pip -v``).

    Returns
    -------
    int
        The exit status of the final :func:`check_stack` run, 0 on success.
    """
    missing = _missing_tools()
    if missing:
        raise SystemExit("missing build tools: " + ", ".join(missing)
                         + ".  On a cluster, load the compiler module first.")
    run = _Runner(dry_run, verbose)
    # A PETSC_DIR/SLEPC_DIR exported by the environment (conda-forge PETSc
    # does) would redirect the builds; every step sets them explicitly.
    clean = {"PETSC_DIR": None, "PETSC_ARCH": None, "SLEPC_DIR": None}

    petsc_version = _installed_version("petsc")
    reuse = False
    if petsc_version and not force and not dry_run:
        reuse = _usable_petsc(run.prefix("petsc"))
    if reuse:
        print(f"PETSc {petsc_version} is already installed with complex scalars and "
              "MUMPS; reusing it (--force rebuilds it).", flush=True)
    else:
        options = list(PETSC_CONFIGURE_OPTIONS) + shlex.split(extra_options)
        print("Compiling PETSc with\n    " + "\n    ".join(options)
              + "\nThis takes 15-30 minutes.", flush=True)
        # The petsc sdist splits this variable with shlex and then hands the
        # words to os.system unquoted, so a value with a space needs two
        # layers of quoting to reach configure as one option.
        run.pip(_series_spec("petsc"), *(["--force-reinstall", "--no-deps"] if force else []),
                env={**clean, "PETSC_CONFIGURE_OPTIONS": " ".join(
                    shlex.quote(shlex.quote(o)) for o in options)})
        petsc_version = _installed_version("petsc") or "<petsc version>"
    petsc_dir = run.prefix("petsc")
    if not dry_run:
        dropped = drop_missing_libraries(petsc_dir)
        if dropped:
            print("Removed link flags of libraries already inside libpetsc.so: "
                  + " ".join(f"-l{n}" for n in dropped), flush=True)
    with_petsc = {**clean, "PETSC_DIR": str(petsc_dir), "PETSC_ARCH": ""}
    # Always rebuilt (a few minutes each): a binding left over from another
    # PETSc, conda-forge's for instance, must not survive under the new one.
    force_flags = ["--force-reinstall", "--no-deps"]
    run.pip(f"petsc4py=={petsc_version}", *force_flags, env=with_petsc)
    run.pip(_series_spec("slepc"), *force_flags,
            env={**with_petsc, "SLEPC_CONFIGURE_OPTIONS": ""})
    slepc_version = _installed_version("slepc") or "<slepc version>"
    slepc_dir = run.prefix("slepc")
    run.pip(f"slepc4py=={slepc_version}", *force_flags,
            env={**with_petsc, "SLEPC_DIR": str(slepc_dir)})
    if dry_run:
        return 0
    # A fresh interpreter: this one may hold stale imports.
    return subprocess.run([sys.executable, "-m", "supernova.eigen.stack"]).returncode


def main(argv: list[str] | None = None) -> int:
    """``python -m supernova.eigen.stack [install ...]``: check or install."""
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["install"]:
        import argparse

        parser = argparse.ArgumentParser(prog="python -m supernova.eigen.stack install")
        add_install_arguments(parser)
        args = parser.parse_args(argv[1:])
        return install_solver(args.extra_options, args.force, args.dry_run, args.verbose)
    items = check_stack()
    print(format_report(items))
    return 0 if stack_ok(items) else 1


def add_install_arguments(parser) -> None:
    """The ``install-solver`` options, shared by the CLI and ``-m``."""
    parser.add_argument(
        "--extra-options", default="",
        help="PETSc configure options appended to the defaults, "
             "e.g. '--with-make-np=8' or '--COPTFLAGS=\"-O3 -march=native\"'")
    parser.add_argument("--force", action="store_true",
                        help="rebuild PETSc even if a usable one is installed")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the pip commands without running them")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="stream the full compiler output")


if __name__ == "__main__":
    raise SystemExit(main())
