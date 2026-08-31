r"""Compatible mixed-spline eigenvalue solver.

``superSolver5`` uses the same local tensors, mixed radial spaces, and boundary
reduction as :class:`supernova.core.solver4.superSolver4`, but multiplies the
split-rational equation by ``s = omega**2`` and solves

    (Tminus1 + s*T0 + s**2*T1) x = 0

with SLEPc PEP.  Away from ``s=0`` this has the same eigenpairs as
``T0 + s*T1 + Tminus1/s``.  At exact ``gamma_s == 0`` the inverse
coefficient vanishes and the same public solver methods transparently use the
SLEPc EPS generalized pencil ``(-T0)x = s*T1*x``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.sparse import csr_matrix

from supernova.eigen.petsc import (
    NEPResult,
    _require_petsc,
    _require_slepc,
    _vec_read_array,
    _vec_write_array,
    apply_petsc_options,
    scipy_csr_to_petsc,
)
from supernova.core.solver3 import (
    _as_scalar_complex,
    _ellipse_region_dict,
    _format_complex,
    _matrix_stats,
)
from supernova.core.solver4 import superSolver4


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


def _polynomial_backward_error(
    A0: csr_matrix,
    A1: csr_matrix,
    A2: csr_matrix,
    eigenvalue: complex,
    eigenvector: np.ndarray,
) -> float:
    """Return a normwise operator-scaled quadratic backward error."""
    vector = np.asarray(eigenvector, dtype=np.complex128)
    absolute, _ = _polynomial_residual_norm(
        A0,
        A1,
        A2,
        eigenvalue,
        vector,
    )
    matrix_scale = (
        float(np.linalg.norm(A0.data))
        + abs(eigenvalue) * float(np.linalg.norm(A1.data))
        + abs(eigenvalue) ** 2 * float(np.linalg.norm(A2.data))
    )
    denominator = matrix_scale * float(np.linalg.norm(vector))
    return absolute / max(np.finfo(float).tiny, denominator)


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
        f"{first_value!r}. Eigenproblem coefficients must be finite before "
        "PETSc construction."
    )


def _canonical_linear_coefficient(name: str, matrix: csr_matrix) -> csr_matrix:
    """Return a finite, square, canonical complex CSR coefficient."""
    coefficient = csr_matrix(matrix, dtype=np.complex128).copy()
    coefficient.sum_duplicates()
    coefficient.eliminate_zeros()
    coefficient.sort_indices()
    if coefficient.shape[0] != coefficient.shape[1]:
        raise ValueError(f"{name} must be square, got {coefficient.shape}.")
    _validate_quadratic_coefficient(name, coefficient)
    return coefficient


def _validate_gamma_zero_pencil(
    Tminus1: csr_matrix,
    T0: csr_matrix,
    T1: csr_matrix,
    *,
    gamma_s: float,
) -> tuple[csr_matrix, csr_matrix]:
    """Validate the exact ``gamma_s=0`` reduction and return ``T0, T1``.

    No numerical threshold is used: a small nonzero ``gamma_s`` remains the
    original rational/PEP problem.  Explicitly stored zeros are removed before
    checking that the inverse coefficient vanishes.
    """
    gamma = float(gamma_s)
    if not np.isfinite(gamma):
        raise ValueError("gamma_s must be finite.")
    if gamma != 0.0:
        raise ValueError(
            "The generalized-linear path requires exact gamma_s == 0.0; "
            f"got {gamma!r}."
        )

    inverse = _canonical_linear_coefficient("Tminus1", Tminus1)
    constant = _canonical_linear_coefficient("T0", T0)
    linear = _canonical_linear_coefficient("T1", T1)
    if not (inverse.shape == constant.shape == linear.shape):
        raise ValueError(
            "Tminus1, T0, and T1 must have identical square shapes; got "
            f"{inverse.shape}, {constant.shape}, and {linear.shape}."
        )
    if inverse.nnz != 0:
        raise ValueError(
            "gamma_s=0 linearization requires Tminus1 to be exactly zero; "
            f"found nnz={inverse.nnz}, "
            f"max_abs={float(np.max(np.abs(inverse.data))):.6e}."
        )
    if linear.nnz == 0:
        raise ValueError(
            "T1 is identically zero; no generalized eigenvalue pencil exists."
        )
    return constant, linear


def _linear_residuals(
    T0: csr_matrix,
    T1: csr_matrix,
    eigenvalue: complex,
    eigenvector: np.ndarray,
) -> tuple[float, float, float]:
    """Return absolute, vector-relative, and backward errors for the pencil."""
    vector = np.asarray(eigenvector, dtype=np.complex128)
    value = complex(eigenvalue)
    residual = T0 @ vector + value * (T1 @ vector)
    absolute = float(np.linalg.norm(residual))
    vector_norm = float(np.linalg.norm(vector))
    relative = absolute / max(1.0, vector_norm)
    scale = (
        float(np.linalg.norm(T0.data))
        + abs(value) * float(np.linalg.norm(T1.data))
    ) * vector_norm
    backward = absolute / max(np.finfo(float).tiny, scale)
    return absolute, relative, backward


@dataclass
class LinearGeneralizedEPSSolver:
    """SLEPc EPS wrapper for the exact-zero pencil ``(-T0)x = s*T1*x``."""

    T0: csr_matrix
    T1: csr_matrix
    target_s: complex
    region: dict
    prefix: str = "nova_gamma_zero_eps"
    eps_type: str = "KRYLOVSCHUR"
    st_type: str = "SINVERT"
    which: str = "TARGET_MAGNITUDE"
    nev: int = 6
    tol: float = 1e-8
    maxit: int = 100
    options: dict | None = None
    comm: object | None = None
    verbose: bool = False
    compute_residuals: bool = True

    backend = "EPS"

    def __post_init__(self) -> None:
        self.T0 = _canonical_linear_coefficient("T0", self.T0)
        self.T1 = _canonical_linear_coefficient("T1", self.T1)
        if self.T0.shape != self.T1.shape:
            raise ValueError(
                f"T0 shape {self.T0.shape} does not match T1 {self.T1.shape}."
            )
        if self.T1.nnz == 0:
            raise ValueError("T1 is identically zero.")

        petsc = _require_petsc()
        slepc = _require_slepc()
        self.target_s = _as_scalar_complex(self.target_s, "target_s")
        self.region = _ellipse_region_dict(self.region, self.target_s)
        self.size = self.T0.shape[0]
        self.comm = petsc.COMM_SELF if self.comm is None else self.comm
        self.scalar_type = np.dtype(petsc.ScalarType)

        # EPS solves A*x = s*B*x.  Keep T1 as a descriptor matrix instead of
        # forming inv(T1)*T0 because T1 may be singular.
        self.A = _canonical_linear_coefficient("A=-T0", -self.T0)
        self.B = self.T1
        self.mats = [
            scipy_csr_to_petsc(matrix, comm=self.comm)
            for matrix in (self.A, self.B)
        ]
        self.rg = slepc.RG().create(comm=self.comm)
        self.rg.setType(slepc.RG.Type.ELLIPSE)
        self.eps = slepc.EPS().create(comm=self.comm)
        self.eps.setOptionsPrefix(self.prefix)
        self._configure_eps()
        self._clear_residuals()

    @staticmethod
    def _resolve_enum(enum_cls, value):
        if value is None or not isinstance(value, str):
            return value
        key = value.upper()
        if not hasattr(enum_cls, key):
            choices = ", ".join(name for name in dir(enum_cls) if name.isupper())
            raise ValueError(f"Unknown enum value {value!r}. Available values: {choices}")
        return getattr(enum_cls, key)

    def _vprint(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _configure_eps(self) -> None:
        slepc = _require_slepc()
        self.rg.setEllipseParameters(
            self.region["center"], self.region["radius"], self.region["vscale"]
        )
        self.rg.setFromOptions()
        self.eps.setType(self._resolve_enum(slepc.EPS.Type, self.eps_type))
        self.eps.setProblemType(slepc.EPS.ProblemType.GNHEP)
        self.eps.setOperators(self.mats[0], self.mats[1])
        self.eps.setRG(self.rg)
        self.eps.setDimensions(int(self.nev))
        self.eps.setTarget(self.target_s)
        self.eps.setWhichEigenpairs(
            self._resolve_enum(slepc.EPS.Which, self.which)
        )
        self.eps.setTolerances(float(self.tol), int(self.maxit))
        self.eps.getST().setType(self._resolve_enum(slepc.ST.Type, self.st_type))
        if self.options:
            apply_petsc_options(self.prefix, **self.options)
        self.eps.setFromOptions()
        dimensions = self.eps.getDimensions()
        self.effective_nev = int(dimensions[0])
        effective_tol, effective_maxit = self.eps.getTolerances()
        self.effective_tol = float(effective_tol)
        self.effective_maxit = int(effective_maxit)
        self._configure_mixed_shift_solver()

    def _configure_mixed_shift_solver(self) -> None:
        """Use the same robust MUMPS shift solve as the quadratic path."""
        ksp = self.eps.getST().getKSP()
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        ksp.setFromOptions()
        self._vprint(
            "[LinearGeneralizedEPSSolver] configured shift solve: "
            f"prefix={ksp.getOptionsPrefix()!r}, ksp={ksp.getType()}, "
            f"pc={pc.getType()}, factor={pc.getFactorSolverType()}"
        )

    def _clear_residuals(self) -> None:
        self.last_residual_norms = None
        self.last_absolute_residual_norms = None
        self.last_backward_errors = None
        self.last_slepc_error_estimates = None
        self.last_slepc_backward_errors = None

    def retarget(self, target_s: complex, region: dict | None = None) -> None:
        """Reset target-dependent EPS state while retaining both matrices."""
        if (
            getattr(self, "eps", None) is None
            or getattr(self, "rg", None) is None
            or len(getattr(self, "mats", ())) != 2
        ):
            raise RuntimeError("Cannot retarget a destroyed LinearGeneralizedEPSSolver.")

        new_target = _as_scalar_complex(target_s, "target_s")
        new_region = _ellipse_region_dict(region, new_target)
        self.eps.reset()
        self.target_s = new_target
        self.region = new_region
        self._configure_eps()
        self._clear_residuals()

    def solve(
        self,
        max_eigenpairs: int | None = None,
        *,
        allow_empty: bool = False,
    ) -> NEPResult:
        """Solve the generalized pencil and return eigenpairs in ``s``."""
        slepc = _require_slepc()
        self._vprint("[LinearGeneralizedEPSSolver] calling EPSSolve()")
        self.eps.solve()
        nconv = int(self.eps.getConverged())
        nout = nconv if max_eigenpairs is None else min(nconv, int(max_eigenpairs))
        reason = self.eps.getConvergedReason()
        iterations = int(self.eps.getIterationNumber())
        if nconv == 0 and not allow_empty:
            raise RuntimeError(
                "SLEPc EPS converged no eigenpairs: "
                f"reason={reason}, iterations={iterations}, "
                f"target_s={_format_complex(self.target_s)}, "
                f"st_type={self.eps.getST().getType()!r}."
            )

        eigenvalues = np.zeros(nout, dtype=np.complex128)
        eigenvectors = np.zeros((nout, self.size), dtype=np.complex128)
        self.last_residual_norms = np.full(nout, np.nan)
        self.last_absolute_residual_norms = np.full(nout, np.nan)
        self.last_backward_errors = np.full(nout, np.nan)
        self.last_slepc_error_estimates = np.full(nout, np.nan)
        self.last_slepc_backward_errors = np.full(nout, np.nan)
        vr = self.mats[0].createVecRight()
        real_scalars = not np.issubdtype(self.scalar_type, np.complexfloating)
        vi = self.mats[0].createVecRight() if real_scalars else None
        try:
            for index in range(nout):
                eigenvalues[index] = self.eps.getEigenpair(index, vr, vi)
                vector = _vec_read_array(vr).astype(np.complex128, copy=True)
                if vi is not None:
                    vector += 1.0j * _vec_read_array(vi)
                eigenvectors[index] = vector
                if self.compute_residuals:
                    absolute, relative, backward = _linear_residuals(
                        self.T0, self.T1, eigenvalues[index], vector
                    )
                    self.last_absolute_residual_norms[index] = absolute
                    self.last_residual_norms[index] = relative
                    self.last_backward_errors[index] = backward
                    self.last_slepc_error_estimates[index] = float(
                        self.eps.getErrorEstimate(index)
                    )
                    self.last_slepc_backward_errors[index] = float(
                        self.eps.computeError(index, slepc.EPS.ErrorType.BACKWARD)
                    )
                self._vprint(
                    f"  eigenpair[{index}]: s={_format_complex(eigenvalues[index])}, "
                    f"omega={_format_complex(np.sqrt(eigenvalues[index]))}, "
                    f"residual={self.last_residual_norms[index]:.3e}, "
                    f"backward_error={self.last_backward_errors[index]:.3e}"
                )
        finally:
            vr.destroy()
            if vi is not None:
                vi.destroy()
        return NEPResult(eigenvalues, eigenvectors, nconv, reason, iterations)

    def set_initial_space(self, vectors: np.ndarray) -> None:
        """Supply one or more reduced-space starting vectors to SLEPc EPS."""
        values = np.asarray(vectors, dtype=np.complex128)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.size:
            raise ValueError(
                f"Initial vectors must have shape (n, {self.size}), got {values.shape}."
            )
        petsc_vectors = []
        try:
            for value in values:
                vector = self.mats[0].createVecRight()
                writable = _vec_write_array(vector)
                writable[...] = np.asarray(value, dtype=self.scalar_type)
                vector.assemblyBegin()
                vector.assemblyEnd()
                petsc_vectors.append(vector)
            self.eps.setInitialSpace(petsc_vectors)
        finally:
            for vector in petsc_vectors:
                vector.destroy()

    def destroy(self) -> None:
        if getattr(self, "eps", None) is not None:
            self.eps.destroy()
            self.eps = None
        if getattr(self, "rg", None) is not None:
            self.rg.destroy()
            self.rg = None
        for matrix in getattr(self, "mats", []):
            matrix.destroy()
        self.mats = []


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

    backend = "PEP"

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
        dimensions = self.pep.getDimensions()
        self.effective_nev = int(dimensions[0])
        effective_tol, effective_maxit = self.pep.getTolerances()
        self.effective_tol = float(effective_tol)
        self.effective_maxit = int(effective_maxit)
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
        self.last_backward_errors = None
        self.last_slepc_error_estimates = None
        self.last_slepc_backward_errors = None
        self._vprint(
            "[QuadraticPEPSolver] retargeted: "
            f"target_s={_format_complex(self.target_s)}, region={self.region}"
        )

    def solve(
        self,
        max_eigenpairs: int | None = None,
        *,
        allow_empty: bool = False,
    ) -> NEPResult:
        """Solve the quadratic PEP and return converged eigenpairs in ``s``."""
        petsc = _require_petsc()
        slepc = _require_slepc()
        self._vprint("[QuadraticPEPSolver] calling PEPSolve()")
        self.pep.solve()
        nconv = int(self.pep.getConverged())
        nout = nconv if max_eigenpairs is None else min(nconv, int(max_eigenpairs))
        reason = self.pep.getConvergedReason()
        iterations = int(self.pep.getIterationNumber())
        if nconv == 0 and not allow_empty:
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
        self.last_backward_errors = np.full(nout, np.nan)
        self.last_slepc_error_estimates = np.full(nout, np.nan)
        self.last_slepc_backward_errors = np.full(nout, np.nan)
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
                    self.last_backward_errors[ii] = _polynomial_backward_error(
                        self.A0,
                        self.A1,
                        self.A2,
                        eigenvalues[ii],
                        vector,
                    )
                    self.last_slepc_error_estimates[ii] = float(
                        self.pep.getErrorEstimate(ii)
                    )
                    self.last_slepc_backward_errors[ii] = float(
                        self.pep.computeError(
                            ii,
                            slepc.PEP.ErrorType.BACKWARD,
                        )
                    )
                self._vprint(
                    f"  eigenpair[{ii}]: s={_format_complex(eigenvalues[ii])}, "
                    f"omega={_format_complex(np.sqrt(eigenvalues[ii]))}, "
                    f"residual={self.last_residual_norms[ii]:.3e}, "
                    f"backward_error={self.last_backward_errors[ii]:.3e}"
                )
        finally:
            vr.destroy()
            if vi is not None:
                vi.destroy()
        return NEPResult(eigenvalues, eigenvectors, nconv, reason, iterations)

    def set_initial_space(self, vectors: np.ndarray) -> None:
        """Supply one or more reduced-space starting vectors to SLEPc PEP."""
        values = np.asarray(vectors, dtype=np.complex128)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.size:
            raise ValueError(
                f"Initial vectors must have shape (n, {self.size}), got {values.shape}."
            )
        petsc_vectors = []
        try:
            for value in values:
                vector = self.mats[0].createVecRight()
                writable = _vec_write_array(vector)
                writable[...] = np.asarray(value, dtype=self.scalar_type)
                vector.assemblyBegin()
                vector.assemblyEnd()
                petsc_vectors.append(vector)
            self.pep.setInitialSpace(petsc_vectors)
        finally:
            for vector in petsc_vectors:
                vector.destroy()

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


def _build_solver5_eigensolver_from_matrices(
    *,
    gamma_s: float,
    T0: csr_matrix,
    T1: csr_matrix,
    Tminus1: csr_matrix,
    target_s: complex,
    region: dict,
    prefix: str,
    pep_type: str,
    st_type: str,
    which: str,
    nev: int,
    tol: float,
    maxit: int,
    options: dict | None,
    verbose: bool,
    compute_residuals: bool,
    quadratic_solver_type=None,
    linear_solver_type=None,
) -> QuadraticPEPSolver | LinearGeneralizedEPSSolver:
    """Construct the exact-zero EPS or the general quadratic PEP backend."""
    gamma = float(gamma_s)
    if not np.isfinite(gamma):
        raise ValueError("gamma_s must be finite.")
    quadratic_solver_type = (
        QuadraticPEPSolver
        if quadratic_solver_type is None
        else quadratic_solver_type
    )
    linear_solver_type = (
        LinearGeneralizedEPSSolver
        if linear_solver_type is None
        else linear_solver_type
    )
    if gamma == 0.0:
        T0, T1 = _validate_gamma_zero_pencil(
            Tminus1,
            T0,
            T1,
            gamma_s=gamma,
        )
        return linear_solver_type(
            T0=T0,
            T1=T1,
            target_s=target_s,
            region=region,
            prefix=prefix,
            st_type=st_type,
            which=which,
            nev=nev,
            tol=tol,
            maxit=maxit,
            options=options,
            verbose=verbose,
            compute_residuals=compute_residuals,
        )
    return quadratic_solver_type(
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


class superSolver5(superSolver4):
    """Solver4 assembly solved with PEP, or EPS when ``gamma_s == 0``."""

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
    ) -> QuadraticPEPSolver | LinearGeneralizedEPSSolver:
        """Assemble and build the appropriate solver5 eigensolver.

        The historical method name remains unchanged.  Exact ``gamma_s == 0``
        returns a generalized EPS solver for ``(-T0)x = s*T1*x``; otherwise it
        returns the quadratic PEP with ``[Tminus1, T0, T1]``.

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
        return _build_solver5_eigensolver_from_matrices(
            gamma_s=self.gamma_s,
            T0=T0,
            T1=T1,
            Tminus1=Tminus1,
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
        """Solve solver5 with transparent exact-zero EPS dispatch.

        Pass an explicit signed ``m_values`` array only for diagnostics that
        require a harmonic space closed under ``m -> -m``.
        """
        if m_values is not None and mmax is not None:
            raise ValueError("Specify either mmax or m_values, not both.")
        target_omega = _as_scalar_complex(target_omega, "target_omega")
        target_s = target_omega**2
        region = _ellipse_region_dict(region, target_s)
        eigensolver = self.build_pep_solver(
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
        backend = getattr(eigensolver, "backend", "PEP")
        if backend == "EPS":
            description = (
                "SLEPc generalized EPS solution of the exact gamma_s=0 "
                "solver5 pencil (-T0) x = s*T1 x."
            )
            extra_attrs = {
                "problem_type": "GENERALIZED_LINEAR",
                "eps_type": getattr(eigensolver, "eps_type", "KRYLOVSCHUR"),
                "st_type": st_type,
                "linear_coefficients": "[-T0, T1]",
                "gamma_zero_reduction": True,
                "requested_pep_type": pep_type,
            }
        else:
            description = (
                "SLEPc quadratic PEP solution of sparse solver5 "
                "(Tminus1 + s*T0 + s**2*T1) x = 0."
            )
            extra_attrs = {
                "problem_type": "POLYNOMIAL",
                "pep_type": pep_type,
                "st_type": st_type,
                "polynomial_degree": 2,
                "polynomial_coefficients": "[Tminus1, T0, T1]",
                "gamma_zero_reduction": False,
                "zero_root_caveat": (
                    "Multiplication by s can introduce roots at s=0; "
                    "the configured search region excludes zero."
                ),
            }
        try:
            result = eigensolver.solve(max_eigenpairs=max_eigenpairs)
            output = self._split_result_dataset(
                result=result,
                nep=eigensolver,
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
                nep_type=backend,
                description=description,
                m_values=m_values,
                extra_attrs=extra_attrs,
            )
        except BaseException as primary_error:
            try:
                eigensolver.destroy()
            except BaseException as cleanup_error:
                # Keep the solve/packing failure as the exception presented to
                # the caller.  The cleanup failure remains available as its
                # explicit cause instead of masking the actionable traceback.
                raise primary_error from cleanup_error
            raise
        else:
            eigensolver.destroy()
            return output
