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


