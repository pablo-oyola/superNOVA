"""NOVA-specific symmetry contracts and mixed-layout adapters.

The numerical checks live in :mod:`supernova.verification.symmetry`.  This
module contains only conventions that must be derived from the NOVA equations
and the live solver layout.

For the stored basis ``exp(i*(n*zeta-m*Theta))`` the equilibrium inputs below
are real surface fields.  A coefficient with ``EVEN`` parity has cosine-only
poloidal amplitudes; an ``ODD`` coefficient has sine-only amplitudes.

The displacement component used by the four-field state follows Cheng and
Chance (1987), Eq. (15):

    xi = xi^psi grad(psi)/|grad(psi)|^2
       + xi^s (B x grad(psi))/B^2
       + xi^b B/B^2,

so

    xi^s = xi dot (B x grad(psi))/|grad(psi)|^2.

It is a rescaled binormal scalar projection, not a Cartesian component and not
a covariant/contravariant coordinate component.  The repository does not
define how the *perturbed vector* is reflected through the midplane, including
the polar/axial-vector convention.  Consequently no intrinsic reflection sign
is assigned to ``xi_s`` here.  A full doubled-m reflection commutator must stay
unclassified until that missing physical transformation is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .symmetry import Parity


NOVA_FIELD_NAMES = ("p1", "xi_psi", "xi_s", "div_xi")

# Derived from the explicit i factors in the live C/D/F blocks:
# C is real; D11 and D21 are imaginary while D12 and D22 are real; F11 and
# F12 are imaginary while F21 and F22 are real.  Multiplying xi_s by +i makes
# every one of those paired row/trial blocks real on an up-down-symmetric
# equilibrium.  The inference diagnostic independently re-derives this tuple.
NOVA_FIELD_PHASES = {
    "p1": 0.0,
    "xi_psi": 0.0,
    "xi_s": 0.5 * math.pi,
    "div_xi": 0.0,
}

XI_S_REFLECTION_STATUS = "UNKNOWN"
XI_S_REFLECTION_REASON = (
    "xi_s is the projection onto (B x grad(psi))/|grad(psi)|^2. "
    "The code does not specify the reflected perturbation-vector convention, "
    "so a scalar sign or field-mixing matrix cannot be derived without "
    "introducing an unsupported assumption."
)


@dataclass(frozen=True)
class NovaCoefficientSpec:
    """Expected parity and provenance for one assembly input."""

    name: str
    expression: str
    owner: str
    attribute: str
    parity: Parity
    rationale: str


def nova_coefficient_registry(gamma_s: float = 5.0 / 3.0) -> tuple[NovaCoefficientSpec, ...]:
    """Return every angular equilibrium coefficient consumed by solver2-5.

    Aliases that enter different equations (for example ``D12=C11`` and the
    continuum/solver ``K`` object) are named explicitly so the generated table
    matches the assembly nomenclature.
    """
    even_scalar = "Scalar made only from reflection-even contractions or flux functions."
    odd_binormal = (
        "Contains one projection on B x grad(psi), which reverses its poloidal "
        "surface parity while the remaining scalar factors are even."
    )
    theta_metric = (
        "Contravariant metric component with exactly one Theta index; "
        "Theta -> -Theta changes its sign."
    )
    toroidal_gauge_metric = (
        "In axisymmetry zeta=phi+nu(psi,Theta), with odd Boozer gauge shift "
        "nu.  Since grad(psi).grad(phi)=0, g^{psi zeta} inherits the odd "
        "parity of the gauge contribution."
    )
    even_local_shear = (
        "For the axisymmetric component convention used by the live code, "
        "s_N=B cross grad(psi)/g^{psi psi} has (odd,even,even) cylindrical "
        "components.  Its curl has the same component parities, so "
        "s_N dot curl(s_N), J dot B, and their even-scalar products are even."
    )
    gamma = repr(float(gamma_s))
    return (
        NovaCoefficientSpec(
            "C11",
            "kappa_Psi/gpsi_psi",
            "solver",
            "C11",
            Parity.EVEN,
            even_scalar,
        ),
        NovaCoefficientSpec(
            "O",
            "B0**2/gpsi_psi/h",
            "solver",
            "O",
            Parity.EVEN,
            even_scalar,
        ),
        NovaCoefficientSpec(
            "C12_2",
            "B0**2/h",
            "solver",
            "C12_2",
            Parity.EVEN,
            even_scalar,
        ),
        NovaCoefficientSpec(
            "C12_3",
            "(J_B - shear*gpsi_psi) * shear / B0**2",
            "solver",
            "C12_3",
            Parity.EVEN,
            even_local_shear,
        ),
        NovaCoefficientSpec(
            "C22",
            "-divPsiVec",
            "solver",
            "C22",
            Parity.EVEN,
            even_scalar,
        ),
        NovaCoefficientSpec(
            "D11",
            "(shear*gpsi_psi - J_B)/h",
            "solver",
            "D11",
            Parity.EVEN,
            even_local_shear,
        ),
        NovaCoefficientSpec(
            "D12_alias_C11",
            "kappa_Psi/gpsi_psi",
            "solver",
            "C11",
            Parity.EVEN,
            even_scalar,
        ),
        NovaCoefficientSpec(
            "D21_1_kappa_s",
            "kappa_s",
            "solver",
            "D21_1",
            Parity.ODD,
            odd_binormal,
        ),
        NovaCoefficientSpec(
            "F11_2",
            "1/B0**2",
            "solver",
            "F11_2",
            Parity.EVEN,
            even_scalar,
        ),
        NovaCoefficientSpec(
            "F12_1",
            "gpsi_psi/B0**2 * shear",
            "solver",
            "F12_1",
            Parity.EVEN,
            even_local_shear,
        ),
        NovaCoefficientSpec(
            "F12_2",
            "-J_B/B0**2",
            "solver",
            "F12_2",
            Parity.EVEN,
            even_scalar,
        ),
        NovaCoefficientSpec(
            "F21",
            "-1/B0**4*h",
            "solver",
            "F21",
            Parity.EVEN,
            even_scalar,
        ),
        NovaCoefficientSpec(
            "F22",
            "-2*kappa_Psi/(gpsi_psi*B0**2)*h",
            "solver",
            "F22",
            Parity.EVEN,
            even_scalar,
        ),
        NovaCoefficientSpec(
            "inv_gpsi_psi",
            "1/gpsi_psi",
            "solver",
            "inv_gpsi_psi",
            Parity.EVEN,
            even_scalar,
        ),
        NovaCoefficientSpec(
            "gpsi_theta_over_gpsi_psi",
            "gpsi_theta/gpsi_psi",
            "solver",
            "gpsi_theta_over_gpsi_psi",
            Parity.ODD,
            theta_metric,
        ),
        NovaCoefficientSpec(
            "gpsi_zeta_over_gpsi_psi",
            "gpsi_zeta/gpsi_psi",
            "solver",
            "gpsi_zeta_over_gpsi_psi",
            Parity.ODD,
            toroidal_gauge_metric,
        ),
        NovaCoefficientSpec(
            "continuum_H",
            "gpsi_psi/h",
            "continuum",
            "H",
            Parity.EVEN,
            even_scalar,
        ),
        NovaCoefficientSpec(
            "continuum_J",
            "gpsi_psi/B0**4*h",
            "continuum",
            "J",
            Parity.EVEN,
            even_scalar,
        ),
        NovaCoefficientSpec(
            "continuum_K",
            "2.0*kappa_s/B0**4*h",
            "continuum",
            "K",
            Parity.ODD,
            odd_binormal,
        ),
        NovaCoefficientSpec(
            "continuum_L",
            f"({gamma}*pressure + B0**2)*h/B0**4",
            "continuum",
            "L",
            Parity.EVEN,
            even_scalar,
        ),
        NovaCoefficientSpec(
            "continuum_N",
            (
                f"h/B0**4*(4*{gamma}*pressure*kappa_s**2)"
                f"/({gamma}*pressure+B0**2)"
            ),
            "continuum",
            "N",
            Parity.EVEN,
            even_scalar,
        ),
        NovaCoefficientSpec(
            "continuum_M",
            "1.0/h",
            "continuum",
            "M",
            Parity.EVEN,
            even_scalar,
        ),
        NovaCoefficientSpec(
            "h",
            "h",
            "solver",
            "hprof",
            Parity.EVEN,
            "Flux function.",
        ),
        NovaCoefficientSpec(
            "I",
            "I",
            "solver",
            "Iprof",
            Parity.EVEN,
            "Flux function.",
        ),
        NovaCoefficientSpec(
            "g",
            "g",
            "solver",
            "gprof",
            Parity.EVEN,
            "Flux function.",
        ),
        NovaCoefficientSpec(
            "q",
            "q",
            "solver",
            "qprof",
            Parity.EVEN,
            "Flux function.",
        ),
        NovaCoefficientSpec(
            "pressure",
            "pressure",
            "solver",
            "pressure",
            Parity.EVEN,
            "Flux function.",
        ),
        NovaCoefficientSpec(
            "pressure_deriv",
            "pressure",
            "solver",
            "pressure_deriv",
            Parity.EVEN,
            "Radial derivative of a flux function.",
        ),
        NovaCoefficientSpec(
            "density",
            "density",
            "solver",
            "density",
            Parity.EVEN,
            "Flux function.",
        ),
    )


def local_field_labels(n_modes: int) -> np.ndarray:
    """Return field labels for a local ``[p1,xi_psi,xi_s,div_xi]`` tensor."""
    if int(n_modes) < 1:
        raise ValueError("n_modes must be positive.")
    return np.repeat(np.asarray(NOVA_FIELD_NAMES, dtype=object), int(n_modes))


def mixed_row_labels(layout) -> tuple[np.ndarray, np.ndarray]:
    """Return field and mode-slot labels for solver4/5 equation rows."""
    fields = np.concatenate(
        [
            np.full(layout.n_low * layout.n_modes, name, dtype=object)
            for name in NOVA_FIELD_NAMES
        ]
    )
    modes = np.concatenate(
        [
            np.tile(np.arange(layout.n_modes, dtype=np.int64), layout.n_low)
            for _ in NOVA_FIELD_NAMES
        ]
    )
    return fields, modes


def mixed_trial_labels(layout) -> tuple[np.ndarray, np.ndarray]:
    """Return field and mode-slot labels for solver4/5 full trial columns."""
    radial_counts = (layout.n_high, layout.n_high, layout.n_low, layout.n_low)
    fields = np.concatenate(
        [
            np.full(n_radial * layout.n_modes, name, dtype=object)
            for name, n_radial in zip(NOVA_FIELD_NAMES, radial_counts)
        ]
    )
    modes = np.concatenate(
        [
            np.tile(np.arange(layout.n_modes, dtype=np.int64), n_radial)
            for n_radial in radial_counts
        ]
    )
    return fields, modes


def reduced_trial_labels(bc, layout) -> tuple[np.ndarray, np.ndarray]:
    """Label sparse reduced columns and verify BCs do not mix fields/modes."""
    full_fields, full_modes = mixed_trial_labels(layout)
    prolongation = bc.get_sparse_prolongation()
    free = np.asarray(bc.sparse_free, dtype=np.int64)
    if free.size != prolongation.shape[1]:
        raise RuntimeError("Sparse prolongation/free-column metadata disagree.")

    reduced_fields = full_fields[free]
    reduced_modes = full_modes[free]
    csc = prolongation.tocsc()
    for column in range(csc.shape[1]):
        support = csc.indices[csc.indptr[column] : csc.indptr[column + 1]]
        if support.size == 0:
            raise RuntimeError(f"Reduced column {column} has empty support.")
        if not np.all(full_fields[support] == reduced_fields[column]):
            raise RuntimeError("A boundary prolongation column mixes physical fields.")
        if not np.all(full_modes[support] == reduced_modes[column]):
            raise RuntimeError("A boundary prolongation column mixes poloidal modes.")
    return reduced_fields, reduced_modes


def labels_to_phases(
    labels: np.ndarray,
    phases: dict[str, float] | None = None,
) -> np.ndarray:
    """Expand one phase per NOVA field to an arbitrary layout."""
    phases = NOVA_FIELD_PHASES if phases is None else phases
    return np.asarray([phases[str(label)] for label in labels], dtype=np.float64)


__all__ = [
    "NOVA_FIELD_NAMES",
    "NOVA_FIELD_PHASES",
    "XI_S_REFLECTION_REASON",
    "XI_S_REFLECTION_STATUS",
    "NovaCoefficientSpec",
    "labels_to_phases",
    "local_field_labels",
    "mixed_row_labels",
    "mixed_trial_labels",
    "nova_coefficient_registry",
    "reduced_trial_labels",
]
