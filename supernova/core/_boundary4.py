r"""
Boundary conditions for the compatible mixed-spline solver4 trial space.

The solver4 trial vector is field-major,

    [p1_H, xi^psi_H, xi^s_L, div_xi_L],

with radial-major, harmonic-fast ordering inside each field.  All solver4
equations are tested in the lower-order space, so the assembled operator is
rectangular before boundary conditions.  The constraints in this module act
only on the high-space ``xi^psi`` coefficients and reduce such an operator on
the trial side as ``T @ P``.
"""

from __future__ import annotations

import operator

import numpy as np
from scipy.sparse import csr_matrix, issparse

from ..radial.base import RadialFE
from ._boundary3 import (
    BoundaryCondition as _Solver3BoundaryCondition,
    boundary_basis_values,
)


class BoundaryCondition(_Solver3BoundaryCondition):
    """Solver3 sparse-nullspace BC with one-sided trial reduction."""

    def reduce_sparse_trial_matrix(self, matrix, tol: float | None = None) -> csr_matrix:
        r"""Apply a boundary-condition prolongation on the trial side.

        For a rectangular mixed operator ``T`` and a sparse prolongation ``P``
        satisfying ``C @ P = 0``, return ``T @ P``.  Equation rows are not
        projected because they already live in the solver4 lower test space.
        """
        P = self.get_sparse_prolongation(tol=tol)
        mat = matrix.tocsr() if issparse(matrix) else csr_matrix(matrix)
        if mat.shape[1] != P.shape[0]:
            raise ValueError(
                f"Expected matrix with {P.shape[0]} trial columns, got shape {mat.shape}."
            )
        return (mat @ P).tocsr()

    def __mul__(self, other: _Solver3BoundaryCondition) -> "BoundaryCondition":
        """Stack two compatible constraint matrices in left-to-right order."""
        if not isinstance(other, _Solver3BoundaryCondition):
            raise ValueError("Can only combine with another BoundaryCondition.")
        if self.C is None or other.C is None:
            raise ValueError("Both boundary conditions must have their C matrices defined.")
        if self.C.shape[1] != other.C.shape[1]:
            raise ValueError(
                "Boundary conditions must act on the same trial space, got "
                f"{self.C.shape[1]} and {other.C.shape[1]} columns."
            )
        return BoundaryCondition(C=np.vstack([self.C, other.C]))

    def __rmul__(self, other: _Solver3BoundaryCondition) -> "BoundaryCondition":
        """Stack two compatible constraint matrices in right-to-left order."""
        if not isinstance(other, _Solver3BoundaryCondition):
            raise ValueError("Can only combine with another BoundaryCondition.")
        if self.C is None or other.C is None:
            raise ValueError("Both boundary conditions must have their C matrices defined.")
        if self.C.shape[1] != other.C.shape[1]:
            raise ValueError(
                "Boundary conditions must act on the same trial space, got "
                f"{other.C.shape[1]} and {self.C.shape[1]} columns."
            )
        return BoundaryCondition(C=np.vstack([other.C, self.C]))


def _validate_mixed_spaces(
    high_radialfe: RadialFE,
    low_radialfe: RadialFE,
) -> tuple[int, int]:
    """Return compatible high/low dimensions for the solver4 layout."""
    n_high = high_radialfe.get_ndofs()
    n_low = low_radialfe.get_ndofs()
    if n_high < 2:
        raise ValueError(f"The high radial space must have at least 2 DoFs, got {n_high}.")
    if n_low != n_high - 1:
        raise ValueError(
            "Compatible mixed spaces require n_low=n_high-1, got "
            f"n_high={n_high}, n_low={n_low}."
        )
    return n_high, n_low


class _XiPsiEndpointBC(BoundaryCondition):
    """Common high-space ``xi^psi`` endpoint trace constraint."""

    def __init__(
        self,
        high_radialfe: RadialFE,
        low_radialfe: RadialFE,
        n_modes: int,
        side: str,
    ):
        try:
            n_modes = operator.index(n_modes)
        except TypeError as exc:
            raise ValueError("n_modes must be an integer.") from exc
        if n_modes < 1:
            raise ValueError("At least one mode is required.")

        n_high, n_low = _validate_mixed_spaces(high_radialfe, low_radialfe)
        _, endpoint_trace = boundary_basis_values(high_radialfe, side=side)
        endpoint_trace = np.asarray(endpoint_trace, dtype=np.complex128)
        if endpoint_trace.shape != (n_high,):
            raise ValueError(
                f"Expected high-space endpoint trace shape {(n_high,)}, "
                f"got {endpoint_trace.shape}."
            )
        if not np.all(np.isfinite(endpoint_trace)):
            raise ValueError("High-space endpoint trace contains non-finite values.")
        if not np.any(endpoint_trace != 0.0):
            raise ValueError(f"High-space {side} endpoint trace is identically zero.")

        self.high_radialfe = high_radialfe
        self.low_radialfe = low_radialfe
        self.n_high = n_high
        self.n_low = n_low
        self.n_modes = n_modes
        self.n_rows = 4 * n_low * n_modes
        self.n_trial = 2 * n_modes * (n_high + n_low)

        self.p1_offset = 0
        self.xipsi_offset = n_high * n_modes
        self.xis_offset = 2 * n_high * n_modes
        self.divxi_offset = (2 * n_high + n_low) * n_modes
        self.field_offsets = (
            self.p1_offset,
            self.xipsi_offset,
            self.xis_offset,
            self.divxi_offset,
        )

        self.side = side
        self.endpoint_trace = endpoint_trace
        self.endpoint_support = np.flatnonzero(endpoint_trace != 0.0)

        C = np.zeros((n_modes, self.n_trial), dtype=np.complex128)
        harmonic_rows = np.arange(n_modes)
        for radial_index, value in enumerate(endpoint_trace):
            if value == 0.0:
                continue
            columns = self.xipsi_offset + radial_index * n_modes + harmonic_rows
            C[harmonic_rows, columns] = value

        # For clamped B-splines this contains exactly one column per harmonic.
        self.endpoint_columns = np.concatenate(
            [
                self.xipsi_offset
                + radial_index * n_modes
                + np.arange(n_modes, dtype=np.int64)
                for radial_index in self.endpoint_support
            ]
        )
        super().__init__(C=C)


class regularityAxisBC(_XiPsiEndpointBC):
    r"""Enforce ``xi^psi_m(0)=0`` in the high trial space."""

    def __init__(
        self,
        high_radialfe: RadialFE,
        low_radialfe: RadialFE,
        m_numbers: np.ndarray,
    ):
        m_numbers = np.asarray(m_numbers)
        if m_numbers.ndim != 1:
            raise ValueError("m_numbers must be a 1D array of poloidal mode numbers.")
        if m_numbers.size < 1:
            raise ValueError("At least one mode is required.")

        self.m_numbers = m_numbers
        super().__init__(
            high_radialfe=high_radialfe,
            low_radialfe=low_radialfe,
            n_modes=m_numbers.size,
            side="left",
        )


class idealSeparatrixBC(_XiPsiEndpointBC):
    r"""Enforce ``xi^psi_m(1)=0`` in the high trial space."""

    def __init__(
        self,
        high_radialfe: RadialFE,
        low_radialfe: RadialFE,
        n_modes: int,
    ):
        super().__init__(
            high_radialfe=high_radialfe,
            low_radialfe=low_radialfe,
            n_modes=n_modes,
            side="right",
        )


__all__ = [
    "BoundaryCondition",
    "idealSeparatrixBC",
    "regularityAxisBC",
]
