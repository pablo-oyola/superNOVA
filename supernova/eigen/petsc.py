r"""
Contains the helpers to generate the PETSc matrices and vectors for the solvers.

The whole superNOVA implementatoin is based on the used of the matrix-free
evaluation to obtain the eigenmodes of the system. There will be two implementations
of the solver:
- Response solver: we will treat the problem as a driven problem, where the Krylov
solver will be used to solve the linear system T(\omega^2) x = b, where b is the drive
term coming from an antenna representation at the edge. This should quite fast in evaluating
the ideal MHD response to a wide range of frequencies, that we can use later to identify
the eigenmodes of the system, via their eigenvectors at the found resonances.
- Nonlinear Eigenvalue solver (NEP) from the SLEPc library. This will be used to 
directly solve the nonlinear eigenvalue problem T(\omega^2) x = 0, where we will be looking
for the eigenvalues \omega^2 and the corresponding eigenvectors x. This will be more
computationally expensive, but it will give us direct access to the eigenmodes of the system,
without the need to scan a wide range of frequencies as in the response solver. 
"""

from __future__ import annotations

try:
    import petsc4py
    from petsc4py import PETSc
except ImportError as exc:
    PETSc = None
    _PETSC_IMPORT_ERROR = exc
else:
    _PETSC_IMPORT_ERROR = None

try:
    import slepc4py
    from slepc4py import SLEPc
except ImportError as exc:
    SLEPc = None
    _SLEPC_IMPORT_ERROR = exc
else:
    _SLEPC_IMPORT_ERROR = None

from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix, issparse


def _require_petsc():
    """
    Return the imported PETSc module or raise a clear optional-dependency error.
    """
    if PETSc is None:
        raise ImportError("petsc4py is required for PETSc-backed solvers.") from _PETSC_IMPORT_ERROR
    return PETSc


def _require_slepc():
    """
    Return the imported SLEPc module or raise a clear optional-dependency error.
    """
    if SLEPc is None:
        raise ImportError("slepc4py is required for SLEPc NEP solvers.") from _SLEPC_IMPORT_ERROR
    return SLEPc

# -------------------------------------------------------------------
# Auxiliary functionst to read and write PETSc vectors as numpy arrays. 
def _vec_read_array(vec):
    try:  # pragma: no cover - requires petsc4py runtime
        return vec.getArray(readonly=True)
    except TypeError:  # pragma: no cover - compatibility fallback
        if hasattr(vec, "getArrayRead"):
            return vec.getArrayRead()
        return vec.getArray()


def _vec_write_array(vec):
    try:  # pragma: no cover - requires petsc4py runtime
        return vec.getArray(readonly=False)
    except TypeError:  # pragma: no cover - compatibility fallback
        return vec.getArray()
    
def apply_petsc_options(prefix: str, **options):
    """
    Insert PETSc options into the PETSc options database.

    Keys should be written without the leading '-'.
    Example: 'ksp_type', not '-ksp_type'.

    For flag options like -ksp_monitor_true_residual, use None or "".
    """
    petsc = _require_petsc()
    OptDB = petsc.Options(prefix)

    for key, value in options.items():
        if value is None:
            OptDB[key] = ""
        else:
            OptDB[key] = value


def _as_csr_matrix(matrix) -> csr_matrix:
    """
    Convert a dense or sparse matrix to canonical CSR form for PETSc insertion.
    """
    mat = matrix.tocsr() if issparse(matrix) else csr_matrix(matrix)
    mat = mat.astype(np.complex128, copy=False)
    mat.sum_duplicates()
    mat.eliminate_zeros()
    mat.sort_indices()
    return mat


def _csr_arrays_for_petsc(matrix):
    """
    Return CSR arrays with PETSc-compatible integer and scalar dtypes.
    """
    petsc = _require_petsc()
    mat = _as_csr_matrix(matrix)
    int_type = getattr(petsc, "IntType", np.int32)
    scalar_type = getattr(petsc, "ScalarType", np.complex128)
    indptr = np.asarray(mat.indptr, dtype=int_type)
    indices = np.asarray(mat.indices, dtype=int_type)
    data = np.asarray(mat.data, dtype=scalar_type)
    return mat, indptr, indices, data


def scipy_csr_to_petsc(matrix, comm=None):
    """
    Create a PETSc AIJ matrix from a SciPy CSR matrix.

    The returned matrix allows new nonzero allocation so the NEP callback can
    reuse it even if the nonlinear operator changes sparsity slightly with the
    trial eigenvalue.
    """
    petsc = _require_petsc()
    mat, indptr, indices, data = _csr_arrays_for_petsc(matrix)
    if comm is None:
        comm = petsc.COMM_SELF

    out = petsc.Mat().createAIJ(
        size=mat.shape,
        csr=(indptr, indices, data),
        comm=comm,
    )
    out.setOption(petsc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    out.assemble()
    return out


def update_petsc_aij_from_csr(petsc_matrix, matrix):
    """
    Overwrite a PETSc AIJ matrix with values from a SciPy CSR matrix.
    """
    petsc = _require_petsc()
    mat, indptr, indices, data = _csr_arrays_for_petsc(matrix)
    if petsc_matrix.getSize() != mat.shape:
        raise ValueError(f"Expected PETSc matrix shape {petsc_matrix.getSize()}, got {mat.shape}.")

    petsc_matrix.setOption(petsc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    petsc_matrix.zeroEntries()
    petsc_matrix.setValuesCSR(indptr, indices, data)
    petsc_matrix.assemble()

class PetscMatrixContext:
    """
    Python matrix context for PETSc shell matrices.
    """

    def __init__(self, operator: callable, **context):
        self.operator = operator
        self.context = context

    def mult(self, mat, x, y):  # pragma: no cover - requires petsc4py runtime
        y.set(0.0) # Ensure the output vector is zeroed before accumulation
        x_arr = _vec_read_array(x)
        y_arr = _vec_write_array(y)
        self.operator(xin=x_arr, xout=y_arr, **self.context)

class KrylovSolver:
    r"""
    Implements the Krylov solver interface using PETSc
    for the evaluation of the matrix-vector products required 
    for the eigenvalue solvers. This will be only used in the driven
    case, where we will solve the linear system:
    
    T(\omega^2) x = b

    where b is the drive term coming from an antenna representation
    at the edge.
    """
    scalar_type: np.dtype
    mat_context: PetscMatrixContext
    options_prefix: str
    size: int
    mat: PETSc.Mat
    ksp: PETSc.KSP
    xvec: PETSc.Vec
    
    def __init__(self, size: int, operator: callable, 
                 context: dict={}, prefix: str='nova', rtol: float=1e-6,
                 atol: float=1e-11, maxit: int=100):
        """
        Sets up the context for the Krylov solver.
        """
        petsc = _require_petsc()
        self.scalar_type = np.dtype(petsc.ScalarType)
        self.mat_context = PetscMatrixContext(operator, **context)
        self.options_prefix = prefix

        # The real problem size is reduced by the number of boundary
        # conditions applied to the system.
        self.size = size

        # Preparing the matrix shell for the operator.
        self.mat = petsc.Mat().create(comm=petsc.COMM_SELF)
        self.mat.setSizes([self.size, self.size])
        self.mat.setType(petsc.Mat.Type.PYTHON)
        self.mat.setPythonContext(self.mat_context)
        self.mat.setUp()

        # Preparing the vectors for the Krylov solver.
        self.ksp = petsc.KSP().create(comm=petsc.COMM_SELF)
        self.ksp.setType(petsc.KSP.Type.GMRES)
        self.ksp.getPC().setType(petsc.PC.Type.NONE) # No preconditioner
        self.ksp.setOptionsPrefix(self.options_prefix)
        self.ksp.setOperators(self.mat)

        # Reasonable starting tolerances; tune for your problem
        self.ksp.setTolerances(rtol=rtol, atol=atol, max_it=maxit)
        self.ksp.setFromOptions()

        # Let's reinforce the no preconditioner use:
        self.ksp.getPC().setType(petsc.PC.Type.NONE)

        # initializing the solution vector.
        self.xvec = self.mat.createVecRight()
        self.xvec.set(0.0)

        # Setting extra options.
        apply_petsc_options(prefix=self.options_prefix,
                            ksp_gmres_restart = 2000,
                            ksp_monitor_true_residual=None, 
                            ksp_view=None, 
                            pc_type='none',
                            ksp_type='gmres',
                            ksp_convergence_test=None,
                            ksp_converged_reason=None,
                            ksp_view_final_residual=None,
                            log_view=None,
                            ksp_rtol=rtol,
                            ksp_atol=atol,
                            ksp_max_it=maxit,
                            ksp_gmres_modifiedgramschmidt=None)
        self.ksp.setFromOptions()

    def swap_context(self, operator: callable, context: dict={},
                     reuse_solution: bool=True):
        """
        Swaps the operator and context used for the matrix-vector products.
        This is useful when we want to solve the linear system for different
        frequencies, where the operator will change due to the change in the
        continuum response.
        """
        self.mat_context.operator = operator
        self.mat_context.context = context

        if reuse_solution:
            # We can reuse the solution vector as an initial guess for the next solve.
            self.ksp.setInitialGuessNonzero(True)

    def solve(self, b: np.ndarray) -> np.ndarray:
        """
        Solves the linear system T x = b using the Krylov solver.

        Parameters
        ----------
        b : np.ndarray
            The right-hand side vector of the linear system.

        Returns
        -------
        x : np.ndarray
            The solution vector of the linear system.
        """
        # Create PETSc vectors for the right-hand side and solution.
        bvec = self.mat.createVecLeft()

        # Setting the antenna into the left-hand side vector.

        bvec[:] = b[:]

        # Solve the linear system.
        self.ksp.solve(bvec, self.xvec)

        # Check for convergence.
        reason = self.ksp.getConvergedReason()
        if int(reason) < 0:
            its = self.ksp.getIterationNumber()
            rnorm = self.ksp.getResidualNorm()
            raise RuntimeError(
                f"Krylov solver did not converge: reason={reason}, "
                f"iterations={its}, residual_norm={rnorm:.3e}"
            )

        # Return the solution as a numpy array.
        return _vec_read_array(self.xvec).copy()


@dataclass
class NEPResult:
    """
    Minimal serializable result from a SLEPc NEP solve.
    """
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    nconv: int
    reason: object
    iterations: int


class NEPSolver:
    """
    Thin SLEPc NEP wrapper for sparse SciPy/PETSc operators.

    ``operator`` must return a square SciPy-compatible sparse matrix for a
    scalar trial eigenvalue. The matrix is copied into a PETSc AIJ matrix inside
    the SLEPc callback, preserving sparse storage for the NEP test path.
    """

    scalar_type: np.dtype
    options_prefix: str
    size: int
    T: PETSc.Mat
    Tprime: PETSc.Mat
    nep: SLEPc.NEP

    def __init__(
        self,
        operator: callable,
        target: complex=0.0,
        operator_derivative: callable=None,
        initial_matrix=None,
        prefix: str="nova_nep",
        nep_type: str="SLP",
        which: str="TARGET_MAGNITUDE",
        nev: int=6,
        tol: float=1e-8,
        maxit: int=100,
        finite_difference_jacobian: bool=True,
        fd_step: float=None,
        options: dict=None,
        comm=None,
    ):
        """
        Sets up a callback-based SLEPc NEP solver.

        Parameters
        ----------
        operator
            Callable returning ``T(lambda)`` as a SciPy sparse matrix.
        target
            Initial target eigenvalue used for SLEPc targeting and matrix
            preallocation.
        operator_derivative
            Optional callable returning ``T'(lambda)`` as a SciPy sparse matrix.
        finite_difference_jacobian
            If True and ``operator_derivative`` is not provided, use a centered
            finite-difference Jacobian callback.
        """
        petsc = _require_petsc()
        slepc = _require_slepc()

        self.operator = operator
        self.operator_derivative = operator_derivative
        self.target = target
        self.options_prefix = prefix
        self.nev = int(nev)
        self.fd_step = fd_step
        self.scalar_type = np.dtype(petsc.ScalarType)
        self.comm = petsc.COMM_SELF if comm is None else comm

        if initial_matrix is None:
            initial_matrix = operator(target)
        initial_matrix = _as_csr_matrix(initial_matrix)
        if initial_matrix.shape[0] != initial_matrix.shape[1]:
            raise ValueError(f"NEP operator must be square, got {initial_matrix.shape}.")
        self.size = initial_matrix.shape[0]

        self.T = scipy_csr_to_petsc(initial_matrix, comm=self.comm)

        nep_type_value = self._resolve_enum(slepc.NEP.Type, nep_type)
        self.Tprime = None
        self._has_derivative = operator_derivative is not None or finite_difference_jacobian
        needs_jacobian = nep_type_value in (slepc.NEP.Type.SLP, slepc.NEP.Type.RII)
        if needs_jacobian and not self._has_derivative:
            raise ValueError(
                f"SLEPc NEP type {nep_type!r} requires a Jacobian for callback operators. "
                "Set finite_difference_jacobian=True or pass operator_derivative."
            )
        if self._has_derivative:
            derivative_matrix = self._evaluate_derivative(target)
            self.Tprime = scipy_csr_to_petsc(derivative_matrix, comm=self.comm)

        self.nep = slepc.NEP().create(comm=self.comm)
        self.nep.setOptionsPrefix(self.options_prefix)
        self.nep.setType(nep_type_value)
        self.nep.setProblemType(slepc.NEP.ProblemType.GENERAL)
        self.nep.setFunction(self._form_function, self.T)
        if self._has_derivative:
            self.nep.setJacobian(self._form_jacobian, self.Tprime)

        self.nep.setDimensions(self.nev)
        self.nep.setTarget(target)
        self.nep.setWhichEigenpairs(self._resolve_enum(slepc.NEP.Which, which))
        self.nep.setTolerances(tol, maxit)

        if options:
            apply_petsc_options(self.options_prefix, **options)
        self.nep.setFromOptions()

    @staticmethod
    def _resolve_enum(enum_cls, value):
        """
        Resolve a SLEPc enum from a string or pass enum values through.
        """
        if value is None or not isinstance(value, str):
            return value
        key = value.upper()
        if not hasattr(enum_cls, key):
            choices = ", ".join(name for name in dir(enum_cls) if name.isupper())
            raise ValueError(f"Unknown enum value {value!r}. Available values: {choices}")
        return getattr(enum_cls, key)

    def _finite_difference_step(self, eigenvalue: complex) -> float:
        """
        Pick a finite-difference step in the eigenvalue variable.
        """
        if self.fd_step is not None:
            return float(self.fd_step)
        return np.sqrt(np.finfo(float).eps) * max(1.0, abs(eigenvalue))

    def _evaluate_derivative(self, eigenvalue: complex) -> csr_matrix:
        """
        Evaluate the user-supplied or finite-difference Jacobian matrix.
        """
        if self.operator_derivative is not None:
            return _as_csr_matrix(self.operator_derivative(eigenvalue))

        step = self._finite_difference_step(eigenvalue)
        plus = _as_csr_matrix(self.operator(eigenvalue + step))
        minus = _as_csr_matrix(self.operator(eigenvalue - step))
        return ((plus - minus) * (0.5 / step)).tocsr()

    def _form_function(self, nep, eigenvalue, F, P):  # pragma: no cover - requires SLEPc runtime
        """
        SLEPc callback filling ``F`` with ``T(eigenvalue)``.
        """
        update_petsc_aij_from_csr(F, self.operator(eigenvalue))

    def _form_jacobian(self, nep, eigenvalue, J):  # pragma: no cover - requires SLEPc runtime
        """
        SLEPc callback filling ``J`` with ``T'(eigenvalue)``.
        """
        update_petsc_aij_from_csr(J, self._evaluate_derivative(eigenvalue))

    def solve(self, max_eigenpairs: int=None) -> NEPResult:
        """
        Solve the NEP and return converged eigenpairs.
        """
        petsc = _require_petsc()
        self.nep.solve()

        nconv = int(self.nep.getConverged())
        if max_eigenpairs is None:
            nout = nconv
        else:
            nout = min(nconv, int(max_eigenpairs))

        eigenvalues = np.zeros(nout, dtype=np.complex128)
        eigenvectors = np.zeros((nout, self.size), dtype=np.complex128)

        vr = self.T.createVecRight()
        real_scalars = not np.issubdtype(np.dtype(petsc.ScalarType), np.complexfloating)
        vi = self.T.createVecRight() if real_scalars else None

        for ii in range(nout):
            eigenvalues[ii] = self.nep.getEigenpair(ii, vr, vi)
            eigenvector = _vec_read_array(vr).astype(np.complex128, copy=True)
            if vi is not None:
                eigenvector += 1j * _vec_read_array(vi)
            eigenvectors[ii] = eigenvector

        reason = self.nep.getConvergedReason() if hasattr(self.nep, "getConvergedReason") else None
        iterations = self.nep.getIterationNumber() if hasattr(self.nep, "getIterationNumber") else -1
        return NEPResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            nconv=nconv,
            reason=reason,
            iterations=int(iterations),
        )
