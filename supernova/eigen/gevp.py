r"""Generalized linear eigenvalue problem for exact ``gamma_s == 0``.

With no adiabatic term the ``1/s`` coefficient vanishes identically and the
quadratic problem degenerates.  The pencil is then genuinely linear,

    (-T0) x = s * T1 x,

and SLEPc EPS solves it directly -- no spurious root at the origin, and one
fewer linearization than the PEP path would introduce.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
from supernova.eigen.region import (
    _as_scalar_complex,
    _ellipse_region_dict,
    _format_complex,
)
# The gamma_s == 0 pencil is validated with the same coefficient checks
# as the quadratic problem it degenerates from.
from supernova.eigen.pep import _validate_quadratic_coefficient


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
