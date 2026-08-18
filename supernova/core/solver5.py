r"""Compatible mixed-spline quadratic polynomial eigenvalue solver.

``superSolver5`` uses the same local tensors, mixed radial spaces, and boundary
reduction as :class:`supernova.core.solver4.superSolver4`, but multiplies the
split-rational equation by ``s = omega**2`` and solves

    (Tminus1 + s*T0 + s**2*T1) x = 0

with SLEPc PEP.  Away from ``s=0`` this has the same eigenpairs as
``T0 + s*T1 + Tminus1/s``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.sparse import csr_matrix

from ._petsc_helpers import (
    NEPResult,
    _require_petsc,
    _require_slepc,
    _vec_read_array,
    apply_petsc_options,
    scipy_csr_to_petsc,
)
from .solver3 import (
    _as_scalar_complex,
    _ellipse_region_dict,
    _format_complex,
    _matrix_stats,
)
from .solver4 import superSolver4


def _polynomial_residual_norm(
    A0: csr_matrix,
    A1: csr_matrix,
    A2: csr_matrix,
    eigenvalue: complex,
    eigenvector: np.ndarray,
) -> tuple[float, float]:
    """Return absolute and vector-normalized quadratic residual norms."""
    residual = A0 @ eigenvector
    residual = residual + eigenvalue * (A1 @ eigenvector)
    residual = residual + eigenvalue**2 * (A2 @ eigenvector)
    abs_norm = float(np.linalg.norm(residual))
    rel_norm = abs_norm / max(1.0, float(np.linalg.norm(eigenvector)))
    return abs_norm, rel_norm


def _validate_quadratic_coefficient(
    name: str,
    matrix: csr_matrix,
) -> None:
    """Reject non-finite stored values before constructing a PETSc matrix."""
    finite = np.isfinite(matrix.data)
    if np.all(finite):
        return

    bad_indices = np.flatnonzero(~finite)
    first_index = int(bad_indices[0])
    first_row = int(np.searchsorted(matrix.indptr, first_index, side="right") - 1)
    first_col = int(matrix.indices[first_index])
    first_value = matrix.data[first_index]
    raise ValueError(
        f"{name} contains {bad_indices.size} non-finite stored value(s); "
        f"the first is at ({first_row}, {first_col}) with value "
        f"{first_value!r}. Quadratic PEP coefficients must be finite before "
        "PETSc construction."
    )


@dataclass
class QuadraticPEPSolver:
    """Thin SLEPc PEP wrapper for ``A0 + s*A1 + s**2*A2``."""

    A0: csr_matrix
    A1: csr_matrix
    A2: csr_matrix
    target_s: complex
    region: dict
    prefix: str = "nova_quadratic_pep"
    pep_type: str = "TOAR"
    st_type: str = "SINVERT"
    which: str = "TARGET_MAGNITUDE"
    nev: int = 6
    tol: float = 1e-8
    maxit: int = 100
    options: dict | None = None
    comm: object | None = None
    verbose: bool = False
    compute_residuals: bool = True

    def __post_init__(self):
        self.A0 = self.A0.tocsr().astype(np.complex128, copy=False)
        self.A1 = self.A1.tocsr().astype(np.complex128, copy=False)
        self.A2 = self.A2.tocsr().astype(np.complex128, copy=False)
        if self.A0.shape[0] != self.A0.shape[1]:
            raise ValueError(f"A0 must be square, got {self.A0.shape}.")
        for name, matrix in (("A1", self.A1), ("A2", self.A2)):
            if matrix.shape != self.A0.shape:
                raise ValueError(
                    f"{name} shape {matrix.shape} does not match A0 {self.A0.shape}."
                )
        for name, matrix in (
            ("A0=Tminus1", self.A0),
            ("A1=T0", self.A1),
            ("A2=T1", self.A2),
        ):
            _validate_quadratic_coefficient(name, matrix)

        petsc = _require_petsc()
        slepc = _require_slepc()

        self.target_s = _as_scalar_complex(self.target_s, "target_s")
        self.region = _ellipse_region_dict(self.region, self.target_s)
        self.size = self.A0.shape[0]
        self.comm = petsc.COMM_SELF if self.comm is None else self.comm
        self.scalar_type = np.dtype(petsc.ScalarType)

        self._vprint("[QuadraticPEPSolver] initializing")
        self._vprint(f"  target_s={_format_complex(self.target_s)}")
        self._vprint(f"  {_matrix_stats('A0=Tminus1', self.A0)}")
        self._vprint(f"  {_matrix_stats('A1=T0', self.A1)}")
        self._vprint(f"  {_matrix_stats('A2=T1', self.A2)}")

        self.mats = [
            scipy_csr_to_petsc(matrix, comm=self.comm)
            for matrix in (self.A0, self.A1, self.A2)
        ]
        self.rg = slepc.RG().create(comm=self.comm)
        self.rg.setType(slepc.RG.Type.ELLIPSE)
        self.rg.setEllipseParameters(
            self.region["center"], self.region["radius"], self.region["vscale"]
        )
        self.rg.setFromOptions()

        self.pep = slepc.PEP().create(comm=self.comm)
        self.pep.setOptionsPrefix(self.prefix)
        self.pep.setType(self._resolve_enum(slepc.PEP.Type, self.pep_type))
        self.pep.setProblemType(slepc.PEP.ProblemType.GENERAL)
        self.pep.setOperators(self.mats)
        self.pep.setRG(self.rg)
        self.pep.setDimensions(int(self.nev))
        self.pep.setTarget(self.target_s)
        self.pep.setWhichEigenpairs(self._resolve_enum(slepc.PEP.Which, self.which))
        self.pep.setTolerances(float(self.tol), int(self.maxit))
        self.pep.getST().setType(self._resolve_enum(slepc.ST.Type, self.st_type))
        if self.options:
            apply_petsc_options(self.prefix, **self.options)
        self.pep.setFromOptions()
        self._configure_mixed_shift_solver()

    def _configure_mixed_shift_solver(self) -> None:
        """Use MUMPS LU for the PEP spectral-transformation shift solve.

        The one-sided compatible-space boundary reduction can leave missing
        entries on the natural-order diagonal even though the reduced matrix
        has full structural rank.  PETSc's default SeqAIJ LU symbolic
        factorization rejects that pattern, whereas MUMPS accepts the general
        sparse structure.  The final ``setFromOptions`` preserves an explicit
        user override for the internal ST KSP/PC and factor package.
        """
        ksp = self.pep.getST().getKSP()
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        ksp.setFromOptions()
        self._vprint(
            "[QuadraticPEPSolver] configured mixed PEP shift solve: "
            f"prefix={ksp.getOptionsPrefix()!r}, ksp={ksp.getType()}, "
            f"pc={pc.getType()}, factor={pc.getFactorSolverType()}"
        )

    def _vprint(self, message: str) -> None:
        if self.verbose:
            print(message)

    @staticmethod
    def _resolve_enum(enum_cls, value):
        if value is None or not isinstance(value, str):
            return value
        key = value.upper()
        if not hasattr(enum_cls, key):
            choices = ", ".join(name for name in dir(enum_cls) if name.isupper())
            raise ValueError(f"Unknown enum value {value!r}. Available values: {choices}")
        return getattr(enum_cls, key)

    def retarget(self, target_s: complex, region: dict | None = None) -> None:
        """Prepare this PEP for a new target without rebuilding operators.

        The previous solution and target-dependent SLEPc setup are discarded.
        The existing SciPy and PETSc polynomial coefficient matrices are retained;
        call :meth:`solve` afterwards to compute eigenpairs for the new window.
        """
        if (
            getattr(self, "pep", None) is None
            or getattr(self, "rg", None) is None
            or len(getattr(self, "mats", ())) != 3
        ):
            raise RuntimeError("Cannot retarget a destroyed QuadraticPEPSolver.")

        new_target_s = _as_scalar_complex(target_s, "target_s")
        new_region = _ellipse_region_dict(region, new_target_s)
        slepc = _require_slepc()

        self.pep.reset()
        self.pep.setOperators(self.mats)
        self.rg.setEllipseParameters(
            new_region["center"],
            new_region["radius"],
            new_region["vscale"],
        )
        self.pep.setRG(self.rg)
        self.pep.setTarget(new_target_s)
        self.pep.getST().setType(self._resolve_enum(slepc.ST.Type, self.st_type))
        self._configure_mixed_shift_solver()

        self.target_s = new_target_s
        self.region = new_region
        self.last_residual_norms = None
        self.last_absolute_residual_norms = None
        self._vprint(
            "[QuadraticPEPSolver] retargeted: "
            f"target_s={_format_complex(self.target_s)}, region={self.region}"
        )

    def solve(self, max_eigenpairs: int | None = None) -> NEPResult:
        """Solve the quadratic PEP and return converged eigenpairs in ``s``."""
        petsc = _require_petsc()
        self._vprint("[QuadraticPEPSolver] calling PEPSolve()")
        self.pep.solve()
        nconv = int(self.pep.getConverged())
        nout = nconv if max_eigenpairs is None else min(nconv, int(max_eigenpairs))
        reason = self.pep.getConvergedReason()
        iterations = int(self.pep.getIterationNumber())
        if nconv == 0:
            raise RuntimeError(
                "SLEPc PEP converged no eigenpairs: "
                f"reason={reason}, iterations={iterations}, "
                f"target_s={_format_complex(self.target_s)}, "
                f"st_type={self.pep.getST().getType()!r}."
            )

        eigenvalues = np.zeros(nout, dtype=np.complex128)
        eigenvectors = np.zeros((nout, self.size), dtype=np.complex128)
        self.last_residual_norms = np.full(nout, np.nan)
        self.last_absolute_residual_norms = np.full(nout, np.nan)
        vr = self.mats[0].createVecRight()
        real_scalars = not np.issubdtype(self.scalar_type, np.complexfloating)
        vi = self.mats[0].createVecRight() if real_scalars else None
        try:
            for ii in range(nout):
                eigenvalues[ii] = self.pep.getEigenpair(ii, vr, vi)
                vector = _vec_read_array(vr).astype(np.complex128, copy=True)
                if vi is not None:
                    vector += 1.0j * _vec_read_array(vi)
                eigenvectors[ii] = vector
                if self.compute_residuals:
                    absolute, relative = _polynomial_residual_norm(
                        self.A0, self.A1, self.A2, eigenvalues[ii], vector
                    )
                    self.last_absolute_residual_norms[ii] = absolute
                    self.last_residual_norms[ii] = relative
                self._vprint(
                    f"  eigenpair[{ii}]: s={_format_complex(eigenvalues[ii])}, "
                    f"omega={_format_complex(np.sqrt(eigenvalues[ii]))}, "
                    f"residual={self.last_residual_norms[ii]:.3e}"
                )
        finally:
            vr.destroy()
            if vi is not None:
                vi.destroy()
        return NEPResult(eigenvalues, eigenvectors, nconv, reason, iterations)

    def destroy(self) -> None:
        if getattr(self, "pep", None) is not None:
            self.pep.destroy()
            self.pep = None
        if getattr(self, "rg", None) is not None:
            self.rg.destroy()
            self.rg = None
        for matrix in getattr(self, "mats", []):
            matrix.destroy()
        self.mats = []


class superSolver5(superSolver4):
    """Solver4 compatible assembly solved as a quadratic SLEPc PEP."""

    def build_pep_solver(
        self,
        ntor: int,
        target_omega: complex,
        region: dict | None,
        mmax: int | None = None,
        max_dm: int = 10,
        regularizer: float = 0.0,
        apply_bcs: bool = True,
        bc_kind: str | None = "full",
        bc_tol: float | None = None,
        prefix: str = "nova_quadratic_pep",
        pep_type: str = "TOAR",
        st_type: str = "SINVERT",
        which: str = "TARGET_MAGNITUDE",
        nev: int = 6,
        tol: float = 1e-8,
        maxit: int = 100,
        options: dict | None = None,
        verbose: bool = False,
        compute_residuals: bool = True,
        m_values: np.ndarray | None = None,
    ) -> QuadraticPEPSolver:
        """Assemble and build PEP with ``[Tminus1, T0, T1]``.

        ``m_values`` is an opt-in diagnostic harmonic list. The production
        q-selected positive-m band remains the default.
        """
        self._require_full_bc_nep(apply_bcs, bc_kind)
        if m_values is not None and mmax is not None:
            raise ValueError("Specify either mmax or m_values, not both.")
        target_omega = _as_scalar_complex(target_omega, "target_omega")
        target_s = target_omega**2
        region = _ellipse_region_dict(region, target_s)
        T0, T1, Tminus1 = self.assemble_split_matrices(
            ntor=ntor,
            mmax=mmax,
            max_dm=max_dm,
            regularizer=regularizer,
            apply_bcs=apply_bcs,
            bc_kind=bc_kind,
            bc_tol=bc_tol,
            verbose=verbose,
            m_values=m_values,
        )
        return QuadraticPEPSolver(
            A0=Tminus1,
            A1=T0,
            A2=T1,
            target_s=target_s,
            region=region,
            prefix=prefix,
            pep_type=pep_type,
            st_type=st_type,
            which=which,
            nev=nev,
            tol=tol,
            maxit=maxit,
            options=options,
            verbose=verbose,
            compute_residuals=compute_residuals,
        )

    def solve_pep(
        self,
        ntor: int,
        target_omega: complex,
        region: dict | None,
        mmax: int | None = None,
        max_dm: int = 10,
        regularizer: float = 0.0,
        apply_bcs: bool = True,
        bc_kind: str | None = "full",
        bc_tol: float | None = None,
        prefix: str = "nova_quadratic_pep",
        pep_type: str = "TOAR",
        st_type: str = "SINVERT",
        which: str = "TARGET_MAGNITUDE",
        nev: int = 6,
        tol: float = 1e-8,
        maxit: int = 100,
        options: dict | None = None,
        max_eigenpairs: int | None = None,
        verbose: bool = False,
        compute_residuals: bool = True,
        m_values: np.ndarray | None = None,
    ) -> xr.Dataset:
        """Solve ``(Tminus1 + s*T0 + s**2*T1)x=0`` with SLEPc PEP.

        Pass an explicit signed ``m_values`` array only for diagnostics that
        require a harmonic space closed under ``m -> -m``.
        """
        if m_values is not None and mmax is not None:
            raise ValueError("Specify either mmax or m_values, not both.")
        target_omega = _as_scalar_complex(target_omega, "target_omega")
        target_s = target_omega**2
        region = _ellipse_region_dict(region, target_s)
        pep = self.build_pep_solver(
            ntor=ntor,
            target_omega=target_omega,
            region=region,
            mmax=mmax,
            max_dm=max_dm,
            regularizer=regularizer,
            apply_bcs=apply_bcs,
            bc_kind=bc_kind,
            bc_tol=bc_tol,
            prefix=prefix,
            pep_type=pep_type,
            st_type=st_type,
            which=which,
            nev=nev,
            tol=tol,
            maxit=maxit,
            options=options,
            verbose=verbose,
            compute_residuals=compute_residuals,
            m_values=m_values,
        )
        try:
            result = pep.solve(max_eigenpairs=max_eigenpairs)
            output = self._split_result_dataset(
                result=result,
                nep=pep,
                ntor=ntor,
                target_omega=target_omega,
                target_s=target_s,
                region=region,
                mmax=mmax,
                max_dm=max_dm,
                regularizer=regularizer,
                apply_bcs=apply_bcs,
                bc_kind=bc_kind,
                bc_tol=bc_tol,
                verbose=verbose,
                compute_residuals=compute_residuals,
                which=which,
                nep_type="PEP",
                description=(
                    "SLEPc quadratic PEP solution of sparse solver5 "
                    "(Tminus1 + s*T0 + s**2*T1) x = 0."
                ),
                m_values=m_values,
                extra_attrs={
                    "problem_type": "POLYNOMIAL",
                    "pep_type": pep_type,
                    "st_type": st_type,
                    "polynomial_degree": 2,
                    "polynomial_coefficients": "[Tminus1, T0, T1]",
                    "zero_root_caveat": (
                        "Multiplication by s can introduce roots at s=0; "
                        "the configured search region excludes zero."
                    ),
                },
            )
        except BaseException as primary_error:
            try:
                pep.destroy()
            except BaseException as cleanup_error:
                # Keep the solve/packing failure as the exception presented to
                # the caller.  The cleanup failure remains available as its
                # explicit cause instead of masking the actionable traceback.
                raise primary_error from cleanup_error
            raise
        else:
            pep.destroy()
            return output
