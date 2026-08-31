"""Choose the eigensolver backend from the adiabatic index.

Exact ``gamma_s == 0`` is not a limit to be approached but a different problem:
the inverse coefficient is identically zero, so the pencil is linear.  Anything
else is the quadratic polynomial problem.  Callers ask for a solve; this is
where that distinction is made, once.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from supernova.eigen.gevp import (
    LinearGeneralizedEPSSolver,
    _validate_gamma_zero_pencil,
)
from supernova.eigen.pep import QuadraticPEPSolver


def build_eigensolver(
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


# The historical name, kept because solve/multigrid.py and a few no_sync
# scripts still call it.
_build_solver5_eigensolver_from_matrices = build_eigensolver
