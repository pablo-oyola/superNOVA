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

try:
    import petsc4py
    from petsc4py import PETSc
except ImportError:
    print("PETSc4Py is not installed. Please install it to use the solvers.")
    PETSc = None

try:
    import slepc4py
    from slepc4py import SLEPc
except ImportError:
    print("SLEPc4Py is not installed. Please install it to use the solvers.")
    SLEPc = None

import numpy as np

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
    OptDB = PETSc.Options(prefix)

    for key, value in options.items():
        if value is None:
            OptDB[key] = ""
        else:
            OptDB[key] = value

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
        self.scalar_type = np.dtype(PETSc.ScalarType)
        self.mat_context = PetscMatrixContext(operator, **context)
        self.options_prefix = prefix

        # The real problem size is reduced by the number of boundary
        # conditions applied to the system.
        self.size = size

        # Preparing the matrix shell for the operator.
        self.mat = PETSc.Mat().create(comm=PETSc.COMM_SELF)
        self.mat.setSizes([self.size, self.size])
        self.mat.setType(PETSc.Mat.Type.PYTHON)
        self.mat.setPythonContext(self.mat_context)
        self.mat.setUp()

        # Preparing the vectors for the Krylov solver.
        self.ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        self.ksp.setType(PETSc.KSP.Type.GMRES)
        self.ksp.getPC().setType(PETSc.PC.Type.NONE) # No preconditioner
        self.ksp.setOptionsPrefix(self.options_prefix)
        self.ksp.setOperators(self.mat)

        # Reasonable starting tolerances; tune for your problem
        self.ksp.setTolerances(rtol=rtol, atol=atol, max_it=maxit)
        self.ksp.setFromOptions()

        # Let's reinforce the no preconditioner use:
        self.ksp.getPC().setType(PETSc.PC.Type.NONE)

        # initializing the solution vector.
        self.xvec = self.mat.createVecRight()
        self.xvec.set(0.0)

        # Setting extra options.
        apply_petsc_options(prefix=self.options_prefix,
                            ksp_gmres_restart = 200,
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

