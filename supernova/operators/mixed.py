r"""Compatible mixed radial spaces and their field-major layout.

Two B-spline spaces of adjacent degree satisfy ``d(Phi_H)/dx = Phi_L D``
exactly.  Putting ``p1`` and ``xi^psi`` in the high space and ``xi^s`` and
``div_xi`` in the low one makes the weak derivative exact rather than
approximated, at the cost of a rectangular operator: every equation is tested
in the low space, so only the trial side carries the boundary reduction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from supernova.radial import BSplineFE, CompatibleBSplinePair


@dataclass(frozen=True)
class MixedFieldLayout:
    """Dimensions and field-major index maps for the solver4 operator."""

    n_high: int
    n_low: int
    n_modes: int

    field_names = ("p1", "xi_psi", "xi_s", "div_xi")

    def __post_init__(self):
        if self.n_high < 2:
            raise ValueError("n_high must be at least 2.")
        if self.n_low != self.n_high - 1:
            raise ValueError(
                "Compatible mixed spaces require n_low=n_high-1, got "
                f"n_high={self.n_high}, n_low={self.n_low}."
            )
        if self.n_modes < 1:
            raise ValueError("n_modes must be positive.")

    @property
    def n_rows(self) -> int:
        return 4 * self.n_low * self.n_modes

    @property
    def n_trial(self) -> int:
        return 2 * self.n_modes * (self.n_high + self.n_low)

    @property
    def trial_offsets(self) -> tuple[int, int, int, int]:
        nm = self.n_modes
        return (
            0,
            self.n_high * nm,
            2 * self.n_high * nm,
            (2 * self.n_high + self.n_low) * nm,
        )

    def row_index(self, field: int, radial: int, mode: int) -> int:
        """Return the low-test row for one field/radial/harmonic index."""
        self._validate_field_mode(field, mode)
        if not 0 <= radial < self.n_low:
            raise IndexError(f"Low-space radial index {radial} is out of range.")
        return field * self.n_low * self.n_modes + radial * self.n_modes + mode

    def trial_index(self, field: int, radial: int, mode: int) -> int:
        """Return the mixed trial column for one field/radial/harmonic index."""
        self._validate_field_mode(field, mode)
        n_radial = self.n_high if field < 2 else self.n_low
        if not 0 <= radial < n_radial:
            raise IndexError(
                f"Trial radial index {radial} is out of range for field {field}."
            )
        return self.trial_offsets[field] + radial * self.n_modes + mode

    def trial_slice(self, field: int) -> slice:
        """Return the full coefficient slice for one trial field."""
        if not 0 <= field < 4:
            raise IndexError(f"Field index {field} is out of range.")
        start = self.trial_offsets[field]
        n_radial = self.n_high if field < 2 else self.n_low
        return slice(start, start + n_radial * self.n_modes)

    def _validate_field_mode(self, field: int, mode: int):
        if not 0 <= field < 4:
            raise IndexError(f"Field index {field} is out of range.")
        if not 0 <= mode < self.n_modes:
            raise IndexError(f"Mode index {mode} is out of range.")


@dataclass(frozen=True)
class MixedRadialData:
    """Cached compatible bases and their shared quadrature representation."""

    pair: CompatibleBSplinePair
    points_q: np.ndarray
    weights_q: np.ndarray
    phi_high_q: np.ndarray
    phi_low_q: np.ndarray
    mass_low: csr_matrix
    G_low_high: csr_matrix


def _build_mixed_radial_data(
    high: BSplineFE,
    points_q: np.ndarray | None = None,
    weights_q: np.ndarray | None = None,
    validate: bool = True,
) -> MixedRadialData:
    """Build compatible basis data on native or externally supplied quadrature.

    Supplying ``points_q`` and ``weights_q`` avoids generating a native
    quadrature for the compatible low space.  This is the multigrid path: all
    nested levels are evaluated on the identical finest-grid rule.

    ``validate`` runs two dense identity checks on the compatible derivative.
    They are worth their cost once per basis but not once per multigrid level,
    so the nested-level construction turns them off after the first level has
    proved the construction sound.
    """
    external_quadrature = points_q is not None or weights_q is not None
    if external_quadrature:
        if points_q is None or weights_q is None:
            raise ValueError("points_q and weights_q must be supplied together.")
        points_q = np.asarray(points_q, dtype=np.float64)
        weights_q = np.asarray(weights_q, dtype=np.float64)
        if points_q.ndim != 1 or weights_q.shape != points_q.shape:
            raise ValueError(
                "External quadrature points and weights must be equal-length "
                "one-dimensional arrays."
            )
        if points_q.size == 0:
            raise ValueError("External quadrature must contain at least one point.")
        if np.any(~np.isfinite(points_q)) or np.any(~np.isfinite(weights_q)):
            raise ValueError("External quadrature points and weights must be finite.")
        if np.any(np.diff(points_q) <= 0.0) or np.any(weights_q <= 0.0):
            raise ValueError(
                "External quadrature points must be strictly increasing and "
                "weights must be positive."
            )
        pair = CompatibleBSplinePair.from_high(
            high,
            build_low_matrices=False,
            verification_points=points_q,
        )
    else:
        pair = CompatibleBSplinePair.from_high(high)
        points_q, weights_q = pair.high.get_quadrature_points()
        points_low, weights_low = pair.low.get_quadrature_points()
        if not np.array_equal(points_q, points_low) or not np.array_equal(
            weights_q, weights_low
        ):
            raise RuntimeError("Compatible high and low spaces must share quadrature.")

    phi_high_q = pair.high.eval_basis(points_q).T
    phi_low_q = pair.low.eval_basis(points_q).T
    mass_low = csr_matrix(phi_low_q.T @ (weights_q[:, None] * phi_low_q))
    G_low_high = (mass_low @ pair.derivative).tocsr()

    if validate:
        dphi_high_q = pair.high.eval_basis_derivative(points_q).T
        compatible_derivative = phi_low_q @ pair.derivative.toarray()
        if not np.allclose(
            dphi_high_q,
            compatible_derivative,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise RuntimeError("The compatible high/low derivative identity failed.")

        direct_G = phi_low_q.T @ (weights_q[:, None] * dphi_high_q)
        if not np.allclose(
            G_low_high.toarray(),
            direct_G,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise RuntimeError("The mixed strong-derivative matrix identity failed.")

    return MixedRadialData(
        pair=pair,
        points_q=np.asarray(points_q),
        weights_q=np.asarray(weights_q),
        phi_high_q=np.asarray(phi_high_q),
        phi_low_q=np.asarray(phi_low_q),
        mass_low=mass_low,
        G_low_high=G_low_high,
    )
