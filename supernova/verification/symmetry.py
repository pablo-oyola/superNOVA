"""Generic numerical checks for parity and reflection symmetry.

The routines in this module deliberately know nothing about NOVA's equations.
They operate on sampled fields, explicit signed harmonic lists, and matrices
whose degree-of-freedom layout is supplied by the caller.  NOVA-specific
parities, field phases, and layout adapters live in :mod:`.nova`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import json
import math
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy import sparse


class Parity(str, Enum):
    """Scalar parity under ``theta -> -theta``."""

    EVEN = "EVEN"
    ODD = "ODD"
    GENERAL = "GENERAL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Tolerance:
    """Combined absolute/relative tolerance with a relative-error floor."""

    atol: float = 0.0
    rtol: float = 1.0e-12
    floor: float = np.finfo(np.float64).tiny

    def __post_init__(self) -> None:
        for name in ("atol", "rtol", "floor"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number.")

    def scale(self, reference: float) -> float:
        """Return the denominator used for a normalized error."""
        return max(float(abs(reference)), float(self.floor))

    def threshold(self, reference: float) -> float:
        """Return ``atol + rtol * max(reference, floor)``."""
        return float(self.atol) + float(self.rtol) * self.scale(reference)

    def to_dict(self) -> dict[str, float]:
        return {
            "atol": float(self.atol),
            "rtol": float(self.rtol),
            "floor": float(self.floor),
        }


INDEX_TOLERANCE = Tolerance(atol=0.0, rtol=0.0)
ALGEBRAIC_TOLERANCE = Tolerance(atol=1.0e-14, rtol=1.0e-12)
QUADRATURE_TOLERANCE = Tolerance(atol=1.0e-13, rtol=1.0e-10)
FOURIER_TOLERANCE = Tolerance(atol=1.0e-13, rtol=1.0e-10)
ASSEMBLY_TOLERANCE = Tolerance(atol=1.0e-13, rtol=1.0e-10)
PETSC_TOLERANCE = Tolerance(atol=1.0e-12, rtol=1.0e-9)
EIGENVECTOR_TOLERANCE = Tolerance(atol=1.0e-12, rtol=1.0e-7)

DEFAULT_TOLERANCES = {
    "index": INDEX_TOLERANCE,
    "algebraic": ALGEBRAIC_TOLERANCE,
    "quadrature": QUADRATURE_TOLERANCE,
    "fourier": FOURIER_TOLERANCE,
    "assembly": ASSEMBLY_TOLERANCE,
    "petsc": PETSC_TOLERANCE,
    "eigenvector": EIGENVECTOR_TOLERANCE,
}


def _json_safe(value: Any) -> Any:
    """Convert numpy, complex, enum, and dataclass values to strict JSON data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0.0 else "-inf"
        return value
    if isinstance(value, complex):
        return {"real": _json_safe(value.real), "imag": _json_safe(value.imag)}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return {
            item.name: _json_safe(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    return str(value)


@dataclass(frozen=True)
class VerificationResult:
    """Machine-readable result shared by all verification gates.

    ``passed=None`` represents an explicitly unclassified diagnostic.  This is
    used for ``GENERAL`` and ``UNKNOWN`` coefficient parity rather than
    silently treating those entries as passing.
    """

    name: str
    absolute_error: float
    relative_error: float
    tolerance: Tolerance
    passed: Optional[bool]
    reference_scale: float = 0.0
    worst_location: Optional[Mapping[str, Any]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def effective_tolerance(self) -> float:
        return self.tolerance.threshold(self.reference_scale)

    @property
    def status(self) -> str:
        if self.passed is None:
            return "UNCLASSIFIED"
        return "PASSED" if self.passed else "FAILED"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "name": self.name,
                "status": self.status,
                "passed": self.passed,
                "absolute_error": self.absolute_error,
                "relative_error": self.relative_error,
                "reference_scale": self.reference_scale,
                "tolerance": self.tolerance.to_dict(),
                "effective_tolerance": self.effective_tolerance,
                "worst_location": self.worst_location,
                "metadata": self.metadata,
            }
        )

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    def format(self) -> str:
        lines = [
            f"{self.status}: {self.name}",
            f"  absolute error: {self.absolute_error:.6e}",
            f"  relative error: {self.relative_error:.6e}",
            (
                "  tolerance:      "
                f"{self.effective_tolerance:.6e} "
                f"(atol={self.tolerance.atol:.3e}, "
                f"rtol={self.tolerance.rtol:.3e})"
            ),
        ]
        if self.worst_location:
            rendered = ", ".join(
                f"{key}={_json_safe(value)}"
                for key, value in self.worst_location.items()
            )
            lines.append(f"  worst location: {rendered}")
        return "\n".join(lines)


def _make_result(
    *,
    name: str,
    absolute_error: float,
    reference_scale: float,
    tolerance: Tolerance,
    worst_location: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    classified: bool = True,
) -> VerificationResult:
    absolute_error = float(abs(absolute_error))
    reference_scale = float(abs(reference_scale))
    denominator = tolerance.scale(reference_scale)
    relative_error = absolute_error / denominator
    passed = absolute_error <= tolerance.threshold(reference_scale) if classified else None
    return VerificationResult(
        name=name,
        absolute_error=absolute_error,
        relative_error=float(relative_error),
        tolerance=tolerance,
        passed=passed,
        reference_scale=reference_scale,
        worst_location=worst_location,
        metadata={} if metadata is None else dict(metadata),
    )


def _flatten_results(items: Iterable[Any]) -> list[VerificationResult]:
    flattened: list[VerificationResult] = []
    for item in items:
        if isinstance(item, VerificationResult):
            flattened.append(item)
        elif hasattr(item, "results"):
            flattened.extend(_flatten_results(item.results))
        elif isinstance(item, Iterable) and not isinstance(item, (str, bytes, Mapping)):
            flattened.extend(_flatten_results(item))
        else:
            raise TypeError(f"Cannot format verification result of type {type(item)!r}.")
    return flattened


def format_summary(results: Iterable[Any]) -> str:
    """Format a collection of results with a deterministic status tally."""
    flat = _flatten_results(results)
    counts = {
        "PASSED": sum(result.passed is True for result in flat),
        "FAILED": sum(result.passed is False for result in flat),
        "UNCLASSIFIED": sum(result.passed is None for result in flat),
    }
    heading = (
        f"Verification summary: {counts['PASSED']} passed, "
        f"{counts['FAILED']} failed, {counts['UNCLASSIFIED']} unclassified"
    )
    return "\n\n".join([heading, *(result.format() for result in flat)])


def results_to_json(results: Iterable[Any], *, indent: Optional[int] = 2) -> str:
    """Serialize a result collection without numpy or non-finite JSON values."""
    flat = _flatten_results(results)
    return json.dumps(
        [result.to_dict() for result in flat],
        indent=indent,
        sort_keys=True,
        allow_nan=False,
    )


def _coerce_parity(parity: Parity | str) -> Parity:
    if isinstance(parity, Parity):
        return parity
    try:
        return Parity(str(parity).upper())
    except ValueError as exc:
        raise ValueError(f"Unknown parity {parity!r}.") from exc


def _periodic_reflection_indices(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 1 or theta.size == 0:
        raise ValueError("theta must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta contains non-finite values.")
    phase = np.mod(theta, 2.0 * np.pi)
    if np.unique(np.round(phase, decimals=14)).size != phase.size:
        raise ValueError("theta contains duplicate periodic nodes.")

    reflected = np.empty(theta.size, dtype=np.int64)
    worst_distance = 0.0
    for index, value in enumerate(phase):
        target = np.mod(-value, 2.0 * np.pi)
        distance = np.abs(np.angle(np.exp(1j * (phase - target))))
        reflected[index] = int(np.argmin(distance))
        worst_distance = max(worst_distance, float(distance[reflected[index]]))
    matching_tolerance = 128.0 * np.finfo(np.float64).eps * max(1.0, theta.size)
    if worst_distance > matching_tolerance:
        raise ValueError(
            "theta grid is not closed under periodic reflection: "
            f"worst angular mismatch {worst_distance:.6e} exceeds "
            f"{matching_tolerance:.6e}."
        )
    if not np.array_equal(reflected[reflected], np.arange(theta.size)):
        raise ValueError("theta reflection map is not an involution.")
    return reflected


def _worst_array_location(
    error: np.ndarray,
    *,
    angle_axis: Optional[int] = None,
    theta: Optional[np.ndarray] = None,
    radial_axis: Optional[int] = None,
    radial_coordinates: Optional[np.ndarray] = None,
    harmonic_axis: Optional[int] = None,
    harmonics: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    if error.size == 0:
        return {}
    index = tuple(int(i) for i in np.unravel_index(np.argmax(np.abs(error)), error.shape))
    location: dict[str, Any] = {
        "index": index,
        "magnitude": float(np.abs(error[index])),
    }
    ndim = error.ndim
    if angle_axis is not None:
        axis = np.core.numeric.normalize_axis_index(angle_axis, ndim)
        location["angle_index"] = index[axis]
        if theta is not None:
            location["theta"] = float(theta[index[axis]])
    if radial_axis is not None:
        axis = np.core.numeric.normalize_axis_index(radial_axis, ndim)
        location["radial_index"] = index[axis]
        if radial_coordinates is not None:
            coords = np.asarray(radial_coordinates)
            location["radial_location"] = coords[index[axis]]
    if harmonic_axis is not None:
        axis = np.core.numeric.normalize_axis_index(harmonic_axis, ndim)
        location["harmonic_index"] = index[axis]
        if harmonics is not None:
            location["harmonic"] = int(harmonics[index[axis]])
    return location


def check_grid_parity(
    values: np.ndarray,
    parity: Parity | str,
    *,
    theta: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    angle_axis: int = -1,
    radial_axis: Optional[int] = 0,
    radial_coordinates: Optional[np.ndarray] = None,
    tolerance: Tolerance = QUADRATURE_TOLERANCE,
    name: str = "grid parity",
) -> VerificationResult:
    """Check scalar parity on a periodic quadrature grid.

    The grid must contain the reflected partner of every angular node.  When
    ``theta`` is omitted, nodes are assumed to be ordered as an endpoint-free
    uniform grid beginning at zero, so slot ``j`` reflects to ``(-j) % N``.
    """
    data = np.asarray(values)
    if data.ndim == 0:
        raise ValueError("values must have an angular dimension.")
    angle_axis = np.core.numeric.normalize_axis_index(angle_axis, data.ndim)
    ntheta = data.shape[angle_axis]
    if ntheta == 0:
        raise ValueError("values has an empty angular dimension.")

    if theta is None:
        theta_values = 2.0 * np.pi * np.arange(ntheta) / ntheta
        reflected_indices = (-np.arange(ntheta)) % ntheta
    else:
        theta_values = np.asarray(theta, dtype=np.float64)
        if theta_values.shape != (ntheta,):
            raise ValueError(
                f"theta shape {theta_values.shape} does not match angular size {ntheta}."
            )
        reflected_indices = _periodic_reflection_indices(theta_values)

    parity = _coerce_parity(parity)
    classified = parity in (Parity.EVEN, Parity.ODD)
    sign = 1.0 if parity is Parity.EVEN else -1.0
    reflected = np.take(data, reflected_indices, axis=angle_axis)
    difference = data - sign * reflected if classified else np.zeros_like(data)

    if weights is None:
        angular_weights = np.ones(ntheta, dtype=np.float64)
    else:
        angular_weights = np.asarray(weights, dtype=np.float64)
        if angular_weights.shape != (ntheta,):
            raise ValueError(
                f"weights shape {angular_weights.shape} does not match angular size {ntheta}."
            )
        if not np.all(np.isfinite(angular_weights)) or np.any(angular_weights < 0.0):
            raise ValueError("weights must be finite and non-negative.")
        if not np.any(angular_weights > 0.0):
            raise ValueError("at least one quadrature weight must be positive.")

    weight_shape = [1] * data.ndim
    weight_shape[angle_axis] = ntheta
    broadcast_weights = angular_weights.reshape(weight_shape)
    reference = float(np.sqrt(np.sum(broadcast_weights * np.abs(data) ** 2)))
    absolute = float(
        np.sqrt(np.sum(broadcast_weights * np.abs(difference) ** 2))
    )
    worst = _worst_array_location(
        difference,
        angle_axis=angle_axis,
        theta=theta_values,
        radial_axis=radial_axis,
        radial_coordinates=radial_coordinates,
    )
    return _make_result(
        name=name,
        absolute_error=absolute,
        reference_scale=reference,
        tolerance=tolerance,
        worst_location=worst,
        metadata={
            "parity": parity.value,
            "angular_points": ntheta,
            "weighted": weights is not None,
        },
        classified=classified,
    )


def check_fourier_parity(
    coefficients: np.ndarray,
    m_values: Sequence[int],
    parity: Parity | str,
    *,
    harmonic_axis: int = -1,
    radial_axis: Optional[int] = 0,
    radial_coordinates: Optional[np.ndarray] = None,
    tolerance: Tolerance = FOURIER_TOLERANCE,
    name: str = "Fourier parity",
) -> VerificationResult:
    """Check ``c[-m] = +/- c[m]`` for explicit signed Fourier coefficients."""
    coeffs = np.asarray(coefficients, dtype=np.complex128)
    if coeffs.ndim == 0:
        raise ValueError("coefficients must have a harmonic dimension.")
    harmonic_axis = np.core.numeric.normalize_axis_index(harmonic_axis, coeffs.ndim)
    modes = np.asarray(m_values)
    if modes.ndim != 1 or modes.size != coeffs.shape[harmonic_axis]:
        raise ValueError("m_values must match the coefficient harmonic dimension.")
    if not np.issubdtype(modes.dtype, np.integer):
        if not np.all(np.isfinite(modes)) or not np.all(modes == np.rint(modes)):
            raise ValueError("m_values must contain finite integers.")
    modes = modes.astype(np.int64, copy=False)
    if np.unique(modes).size != modes.size:
        raise ValueError("m_values must not contain duplicates.")
    mode_to_slot = {int(mode): slot for slot, mode in enumerate(modes)}
    missing = [int(mode) for mode in modes if int(-mode) not in mode_to_slot]
    if missing:
        raise ValueError(
            "m_values is not closed under reflection; missing partners for "
            f"{missing}."
        )
    reflection = np.asarray([mode_to_slot[int(-mode)] for mode in modes])

    parity = _coerce_parity(parity)
    classified = parity in (Parity.EVEN, Parity.ODD)
    sign = 1.0 if parity is Parity.EVEN else -1.0
    reflected = np.take(coeffs, reflection, axis=harmonic_axis)
    difference = reflected - sign * coeffs if classified else np.zeros_like(coeffs)
    reference = float(np.linalg.norm(coeffs.ravel()))
    absolute = float(np.linalg.norm(difference.ravel()))
    worst = _worst_array_location(
        difference,
        radial_axis=radial_axis,
        radial_coordinates=radial_coordinates,
        harmonic_axis=harmonic_axis,
        harmonics=modes,
    )
    return _make_result(
        name=name,
        absolute_error=absolute,
        reference_scale=reference,
        tolerance=tolerance,
        worst_location=worst,
        metadata={"parity": parity.value, "harmonics": modes},
        classified=classified,
    )


@dataclass(frozen=True)
class FourierParityReport:
    """Parity relation plus the independent forbidden-component gate."""

    parity: VerificationResult
    forbidden_component: VerificationResult

    @property
    def results(self) -> tuple[VerificationResult, VerificationResult]:
        return (self.parity, self.forbidden_component)

    @property
    def passed(self) -> Optional[bool]:
        statuses = [result.passed for result in self.results]
        if any(status is False for status in statuses):
            return False
        if all(status is True for status in statuses):
            return True
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "parity": self.parity.to_dict(),
            "forbidden_component": self.forbidden_component.to_dict(),
        }

    def format(self) -> str:
        return "\n\n".join(result.format() for result in self.results)


def check_profile_fourier_parity(
    profile: Any,
    parity: Parity | str,
    *,
    ntor: int = 0,
    tolerance: Tolerance = FOURIER_TOLERANCE,
    name: Optional[str] = None,
) -> FourierParityReport:
    """Check parity of a one-sided real-field :class:`~supernova.Profile`.

    For the stored ``A_m = a_m + i b_m`` convention, a real even field has no
    sine amplitude and a real odd field has no cosine amplitude (including a
    zero ``m=0`` coefficient).  Only ``ntor=0`` is accepted: for nonzero
    toroidal mode number, theta reflection requires ``(-m,+n)``, which is not
    represented by the rectangular one-sided Profile storage.
    """
    if int(ntor) != 0:
        raise ValueError(
            "One-sided Profile theta-parity is only representable for ntor=0; "
            "nonzero ntor requires an explicit signed (m,n) mode set."
        )
    if not hasattr(profile, "data"):
        raise TypeError("profile must expose an xarray-like .data member.")
    data = profile.data
    for dimension in ("x", "mpol", "ntor"):
        if dimension not in data.dims:
            raise ValueError(f"profile.data is missing the {dimension!r} dimension.")
    if ntor not in set(int(value) for value in data.ntor.values):
        raise ValueError(f"Profile does not contain ntor={ntor}.")
    m_values = np.asarray(data.mpol.values, dtype=np.int64)
    if np.any(m_values < 0) or np.unique(m_values).size != m_values.size:
        raise ValueError("One-sided Profile storage must contain unique mpol >= 0.")
    order = np.argsort(m_values)
    m_values = m_values[order]
    amplitudes = np.asarray(
        data.sel(ntor=ntor).transpose("x", "mpol").values[:, order],
        dtype=np.complex128,
    )
    radial = np.asarray(data.x.values)

    positive_mask = m_values > 0
    negative_modes = -m_values[positive_mask][::-1]
    signed_modes = np.concatenate([negative_modes, m_values])
    negative_coefficients = np.conjugate(
        amplitudes[:, positive_mask][:, ::-1]
    ) / 2.0
    nonnegative_coefficients = amplitudes.copy()
    nonnegative_coefficients[:, positive_mask] *= 0.5
    signed_coefficients = np.concatenate(
        [negative_coefficients, nonnegative_coefficients],
        axis=1,
    )

    base_name = "Profile" if name is None else str(name)
    parity_result = check_fourier_parity(
        signed_coefficients,
        signed_modes,
        parity,
        harmonic_axis=1,
        radial_axis=0,
        radial_coordinates=radial,
        tolerance=tolerance,
        name=f"{base_name} Fourier parity",
    )

    parity_value = _coerce_parity(parity)
    classified = parity_value in (Parity.EVEN, Parity.ODD)
    if parity_value is Parity.EVEN:
        forbidden = np.imag(amplitudes)
        component_name = "imaginary"
    elif parity_value is Parity.ODD:
        forbidden = np.real(amplitudes)
        component_name = "real"
    else:
        forbidden = np.zeros_like(amplitudes.real)
        component_name = "unclassified"
    absolute = float(np.linalg.norm(forbidden.ravel()))
    reference = float(np.linalg.norm(amplitudes.ravel()))
    worst = _worst_array_location(
        forbidden,
        radial_axis=0,
        radial_coordinates=radial,
        harmonic_axis=1,
        harmonics=m_values,
    )
    forbidden_result = _make_result(
        name=f"{base_name} forbidden {component_name} Fourier component",
        absolute_error=absolute,
        reference_scale=reference,
        tolerance=tolerance,
        worst_location=worst,
        metadata={
            "parity": parity_value.value,
            "forbidden_component": component_name,
            "one_sided_amplitudes": True,
        },
        classified=classified,
    )
    return FourierParityReport(parity_result, forbidden_result)


def reflection_permutation(m_values: Sequence[int]) -> np.ndarray:
    """Return harmonic slots implementing ``m -> -m``.

    The returned array satisfies ``reflected[..., i] = original[..., perm[i]]``.
    """
    modes = np.asarray(m_values)
    if modes.ndim != 1 or modes.size == 0:
        raise ValueError("m_values must be a nonempty one-dimensional array.")
    if not np.issubdtype(modes.dtype, np.integer):
        if not np.all(np.isfinite(modes)) or not np.all(modes == np.rint(modes)):
            raise ValueError("m_values must contain finite integers.")
    modes = modes.astype(np.int64, copy=False)
    if np.unique(modes).size != modes.size:
        raise ValueError("m_values must not contain duplicates.")
    slots = {int(mode): index for index, mode in enumerate(modes)}
    missing = [int(mode) for mode in modes if int(-mode) not in slots]
    if missing:
        raise ValueError(
            "m_values is not closed under reflection; missing partners for "
            f"{missing}."
        )
    permutation = np.asarray([slots[int(-mode)] for mode in modes], dtype=np.int64)
    if not np.array_equal(permutation[permutation], np.arange(modes.size)):
        raise RuntimeError("Internal reflection permutation is not an involution.")
    return permutation


def _field_names_and_counts(
    radial_dofs: int | Sequence[int] | Mapping[str, int],
    field_names: Optional[Sequence[str]],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if isinstance(radial_dofs, Mapping):
        if field_names is None:
            names = tuple(str(name) for name in radial_dofs)
        else:
            names = tuple(str(name) for name in field_names)
        counts = tuple(int(radial_dofs[name]) for name in names)
    elif np.isscalar(radial_dofs):
        names = ("field0",) if field_names is None else tuple(map(str, field_names))
        counts = tuple(int(radial_dofs) for _ in names)
    else:
        counts = tuple(int(value) for value in radial_dofs)
        names = (
            tuple(f"field{index}" for index in range(len(counts)))
            if field_names is None
            else tuple(map(str, field_names))
        )
        if len(names) != len(counts):
            raise ValueError("field_names and radial_dofs must have the same length.")
    if not names:
        raise ValueError("At least one field is required.")
    if len(set(names)) != len(names):
        raise ValueError("field_names must be unique.")
    if any(count < 1 for count in counts):
        raise ValueError("Every field must have at least one radial degree of freedom.")
    return names, counts


def build_reflection_operator(
    m_values: Sequence[int],
    *,
    field_names: Optional[Sequence[str]] = None,
    radial_dofs: int | Sequence[int] | Mapping[str, int] = 1,
    field_signs: Optional[Sequence[complex] | Mapping[str, complex]] = None,
    field_reflection: Optional[np.ndarray] = None,
) -> sparse.csr_matrix:
    r"""Build ``(P x)[a,m,j] = sum_b S[a,b] x[b,-m,j]``.

    The layout is field-major, then radial-major, with the explicit harmonic
    list as the fastest index.  Non-diagonal field mixing is supported when
    every coupled field pair has the same radial-space size.
    """
    permutation = reflection_permutation(m_values)
    n_modes = permutation.size
    names, radial_counts = _field_names_and_counts(radial_dofs, field_names)
    n_fields = len(names)
    if field_reflection is not None and field_signs is not None:
        raise ValueError("Specify field_reflection or field_signs, not both.")
    if field_reflection is not None:
        S = np.asarray(field_reflection, dtype=np.complex128)
        if S.shape != (n_fields, n_fields):
            raise ValueError(
                f"field_reflection must have shape {(n_fields, n_fields)}."
            )
    else:
        if field_signs is None:
            signs = np.ones(n_fields, dtype=np.complex128)
        elif isinstance(field_signs, Mapping):
            signs = np.asarray(
                [field_signs[name] for name in names],
                dtype=np.complex128,
            )
        else:
            signs = np.asarray(field_signs, dtype=np.complex128)
            if signs.shape != (n_fields,):
                raise ValueError(f"field_signs must have shape {(n_fields,)}.")
        S = np.diag(signs)
    if not np.all(np.isfinite(S)):
        raise ValueError("field reflection contains non-finite values.")

    offsets = np.cumsum((0, *(count * n_modes for count in radial_counts)))
    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    for output_field in range(n_fields):
        for input_field in range(n_fields):
            coefficient = S[output_field, input_field]
            if coefficient == 0.0:
                continue
            if radial_counts[output_field] != radial_counts[input_field]:
                raise ValueError(
                    "Field reflection mixes fields with different radial sizes: "
                    f"{names[output_field]} has {radial_counts[output_field]}, "
                    f"{names[input_field]} has {radial_counts[input_field]}."
                )
            for radial_index in range(radial_counts[output_field]):
                output_start = offsets[output_field] + radial_index * n_modes
                input_start = offsets[input_field] + radial_index * n_modes
                rows.extend((output_start + np.arange(n_modes)).tolist())
                columns.extend((input_start + permutation).tolist())
                values.extend([complex(coefficient)] * n_modes)
    size = int(offsets[-1])
    return sparse.coo_matrix(
        (np.asarray(values), (np.asarray(rows), np.asarray(columns))),
        shape=(size, size),
        dtype=np.complex128,
    ).tocsr()


def _matrix_frobenius(matrix: Any) -> float:
    if sparse.issparse(matrix):
        return float(np.sqrt(np.sum(np.abs(matrix.data) ** 2)))
    array = np.asarray(matrix)
    return float(np.linalg.norm(array.ravel()))


def _matrix_worst_location(
    difference: Any,
    *,
    actual: Optional[Any] = None,
    expected: Optional[Any] = None,
) -> dict[str, Any]:
    if sparse.issparse(difference):
        coo = difference.tocoo()
        if coo.nnz == 0:
            row = column = 0
            magnitude = 0.0
        else:
            slot = int(np.argmax(np.abs(coo.data)))
            row, column = int(coo.row[slot]), int(coo.col[slot])
            magnitude = float(abs(coo.data[slot]))
    else:
        array = np.asarray(difference)
        if array.size == 0:
            return {}
        row, column = (
            int(value)
            for value in np.unravel_index(np.argmax(np.abs(array)), array.shape)
        )
        magnitude = float(abs(array[row, column]))
    location: dict[str, Any] = {
        "row": row,
        "column": column,
        "magnitude": magnitude,
    }
    if actual is not None:
        location["actual"] = complex(actual[row, column])
    if expected is not None:
        location["expected"] = complex(expected[row, column])
    return location


def _identity_like(matrix: Any) -> Any:
    size = matrix.shape[0]
    if sparse.issparse(matrix):
        return sparse.eye(size, format="csr", dtype=np.complex128)
    return np.eye(size, dtype=np.complex128)


def check_reflection_involution(
    reflection: Any,
    *,
    tolerance: Tolerance = ALGEBRAIC_TOLERANCE,
    name: str = "reflection involution",
) -> VerificationResult:
    """Check ``P @ P = I``."""
    if len(reflection.shape) != 2 or reflection.shape[0] != reflection.shape[1]:
        raise ValueError("reflection must be square.")
    product = reflection @ reflection
    identity = _identity_like(reflection)
    difference = product - identity
    reference = _matrix_frobenius(identity)
    return _make_result(
        name=name,
        absolute_error=_matrix_frobenius(difference),
        reference_scale=reference,
        tolerance=tolerance,
        worst_location=_matrix_worst_location(
            difference,
            actual=product,
            expected=identity,
        ),
        metadata={"matrix_shape": reflection.shape},
    )


def check_reflection_covariance(
    matrix: Any,
    trial_reflection: Any,
    row_reflection: Optional[Any] = None,
    *,
    tolerance: Tolerance = ASSEMBLY_TOLERANCE,
    name: str = "reflection covariance",
) -> VerificationResult:
    """Check the two-sided relation ``A P_trial = P_row A``.

    ``row_reflection`` defaults to ``trial_reflection`` for square operators.
    Supplying both makes the gate valid for rectangular mixed test/trial spaces.
    """
    if row_reflection is None:
        row_reflection = trial_reflection
    if matrix.shape[1] != trial_reflection.shape[0]:
        raise ValueError("trial reflection size does not match matrix columns.")
    if matrix.shape[0] != row_reflection.shape[0]:
        raise ValueError("row reflection size does not match matrix rows.")
    actual = matrix @ trial_reflection
    expected = row_reflection @ matrix
    difference = actual - expected
    return _make_result(
        name=name,
        absolute_error=_matrix_frobenius(difference),
        reference_scale=_matrix_frobenius(matrix),
        tolerance=tolerance,
        worst_location=_matrix_worst_location(
            difference,
            actual=actual,
            expected=expected,
        ),
        metadata={"matrix_shape": matrix.shape, "two_sided": True},
    )


def check_operator_commutator(
    matrix: Any,
    reflection: Any,
    *,
    tolerance: Tolerance = ASSEMBLY_TOLERANCE,
    name: str = "operator-reflection commutator",
) -> VerificationResult:
    """Square-operator alias for :func:`check_reflection_covariance`."""
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "A commutator requires a square matrix; use "
            "check_reflection_covariance with row/trial reflections."
        )
    return check_reflection_covariance(
        matrix,
        reflection,
        tolerance=tolerance,
        name=name,
    )


def check_operator_intertwining(
    matrix: Any,
    row_action: Any,
    trial_action: Any,
    *,
    tolerance: Tolerance = ASSEMBLY_TOLERANCE,
    name: str = "operator-reflection intertwining",
) -> VerificationResult:
    """Check ``A P_trial = P_row A`` with explicit test/trial actions."""
    return check_reflection_covariance(
        matrix,
        trial_action,
        row_reflection=row_action,
        tolerance=tolerance,
        name=name,
    )


def parity_projectors(reflection: Any) -> tuple[Any, Any]:
    """Return ``(I+P)/2`` and ``(I-P)/2``."""
    if len(reflection.shape) != 2 or reflection.shape[0] != reflection.shape[1]:
        raise ValueError("reflection must be square.")
    identity = _identity_like(reflection)
    return 0.5 * (identity + reflection), 0.5 * (identity - reflection)


def check_parity_sector_mixing(
    matrix: Any,
    trial_reflection: Any,
    row_reflection: Optional[Any] = None,
    *,
    tolerance: Tolerance = ASSEMBLY_TOLERANCE,
    name: str = "parity-sector mixing",
) -> VerificationResult:
    """Check both off-diagonal parity sectors of a square or rectangular map."""
    if row_reflection is None:
        row_reflection = trial_reflection
    row_plus, row_minus = parity_projectors(row_reflection)
    trial_plus, trial_minus = parity_projectors(trial_reflection)
    plus_from_minus = row_plus @ matrix @ trial_minus
    minus_from_plus = row_minus @ matrix @ trial_plus
    norm_first = _matrix_frobenius(plus_from_minus)
    norm_second = _matrix_frobenius(minus_from_plus)
    worst_matrix = (
        plus_from_minus if norm_first >= norm_second else minus_from_plus
    )
    worst_sector = "+<--" if norm_first >= norm_second else "-<-+"
    return _make_result(
        name=name,
        absolute_error=norm_first + norm_second,
        reference_scale=_matrix_frobenius(matrix),
        tolerance=tolerance,
        worst_location={
            **_matrix_worst_location(worst_matrix),
            "sector": worst_sector,
        },
        metadata={
            "plus_from_minus_norm": norm_first,
            "minus_from_plus_norm": norm_second,
            "matrix_shape": matrix.shape,
        },
    )


def _harmonic_layout_indices(
    source_m_values: Sequence[int],
    selected_m_values: Sequence[int],
    radial_counts: Sequence[int],
) -> np.ndarray:
    source = np.asarray(source_m_values, dtype=np.int64)
    selected = np.asarray(selected_m_values, dtype=np.int64)
    slots = {int(mode): index for index, mode in enumerate(source)}
    missing = [int(mode) for mode in selected if int(mode) not in slots]
    if missing:
        raise ValueError(f"Selected harmonics are absent from source list: {missing}.")
    harmonic_slots = np.asarray([slots[int(mode)] for mode in selected])
    n_source = source.size
    offsets = np.cumsum((0, *(int(count) * n_source for count in radial_counts)))
    indices: list[np.ndarray] = []
    for field_index, radial_count in enumerate(radial_counts):
        for radial_index in range(int(radial_count)):
            indices.append(
                offsets[field_index] + radial_index * n_source + harmonic_slots
            )
    return np.concatenate(indices).astype(np.int64, copy=False)


def compare_positive_subblock(
    production_matrix: Any,
    doubled_matrix: Any,
    *,
    production_m_values: Sequence[int],
    doubled_m_values: Sequence[int],
    row_radial_dofs: int | Sequence[int] | Mapping[str, int] = 1,
    trial_radial_dofs: Optional[
        int | Sequence[int] | Mapping[str, int]
    ] = None,
    field_names: Optional[Sequence[str]] = None,
    row_field_names: Optional[Sequence[str]] = None,
    trial_field_names: Optional[Sequence[str]] = None,
    tolerance: Tolerance = ASSEMBLY_TOLERANCE,
    name: str = "positive-m subblock consistency",
) -> VerificationResult:
    """Compare a production matrix with the positive block of doubled assembly.

    Radial counts may be scalars, per-field sequences, or field-name mappings.
    Row and trial counts are independent so the function supports the mixed operator's
    rectangular mixed B-spline layout.
    """
    production_modes = np.asarray(production_m_values, dtype=np.int64)
    if production_modes.ndim != 1 or production_modes.size == 0:
        raise ValueError("production_m_values must be nonempty and one-dimensional.")
    if np.any(production_modes <= 0):
        raise ValueError("production_m_values must contain only positive harmonics.")
    if np.unique(production_modes).size != production_modes.size:
        raise ValueError("production_m_values must not contain duplicates.")
    doubled_modes = np.asarray(doubled_m_values, dtype=np.int64)
    if doubled_modes.ndim != 1 or doubled_modes.size == 0:
        raise ValueError("doubled_m_values must be nonempty and one-dimensional.")

    row_names = row_field_names if row_field_names is not None else field_names
    trial_names = (
        trial_field_names if trial_field_names is not None else field_names
    )
    row_names_resolved, row_counts = _field_names_and_counts(
        row_radial_dofs,
        row_names,
    )
    if trial_radial_dofs is None:
        trial_radial_dofs = row_radial_dofs
        if trial_names is None:
            trial_names = row_names_resolved
    _, trial_counts = _field_names_and_counts(
        trial_radial_dofs,
        trial_names,
    )
    row_indices = _harmonic_layout_indices(
        doubled_modes,
        production_modes,
        row_counts,
    )
    trial_indices = _harmonic_layout_indices(
        doubled_modes,
        production_modes,
        trial_counts,
    )
    expected_production_shape = (
        sum(row_counts) * production_modes.size,
        sum(trial_counts) * production_modes.size,
    )
    expected_doubled_shape = (
        sum(row_counts) * doubled_modes.size,
        sum(trial_counts) * doubled_modes.size,
    )
    if tuple(production_matrix.shape) != expected_production_shape:
        raise ValueError(
            f"production matrix shape {production_matrix.shape} does not match "
            f"layout shape {expected_production_shape}."
        )
    if tuple(doubled_matrix.shape) != expected_doubled_shape:
        raise ValueError(
            f"doubled matrix shape {doubled_matrix.shape} does not match "
            f"layout shape {expected_doubled_shape}."
        )
    if sparse.issparse(doubled_matrix):
        extracted = doubled_matrix.tocsr()[row_indices][:, trial_indices]
    else:
        extracted = np.asarray(doubled_matrix)[np.ix_(row_indices, trial_indices)]
    difference = production_matrix - extracted
    return _make_result(
        name=name,
        absolute_error=_matrix_frobenius(difference),
        reference_scale=_matrix_frobenius(production_matrix),
        tolerance=tolerance,
        worst_location=_matrix_worst_location(
            difference,
            actual=production_matrix,
            expected=extracted,
        ),
        metadata={
            "production_shape": production_matrix.shape,
            "doubled_shape": doubled_matrix.shape,
            "production_m_values": production_modes,
            "doubled_m_values": doubled_modes,
            "row_indices": row_indices,
            "trial_indices": trial_indices,
        },
    )


def _matvec(matrix: Optional[Any], vector: np.ndarray) -> np.ndarray:
    return vector if matrix is None else np.asarray(matrix @ vector)


def _phase_transform_weight(
    weight: Optional[Any],
    phases: np.ndarray,
) -> Optional[Any]:
    """Transform a metric consistently with ``y = D^{-1} x``."""
    if weight is None:
        return None
    left = np.exp(-1j * phases)
    right = np.exp(1j * phases)
    if tuple(weight.shape) != (phases.size, phases.size):
        raise ValueError("weight shape does not match the phase-vector size.")
    if sparse.issparse(weight):
        return (
            sparse.diags(left, format="csr")
            @ weight
            @ sparse.diags(right, format="csr")
        ).tocsr()
    return left[:, None] * np.asarray(weight) * right[None, :]


def _positive_quadratic_norm(vector: np.ndarray, weight: Optional[Any]) -> float:
    vector = np.asarray(vector, dtype=np.complex128)
    value = np.vdot(vector, _matvec(weight, vector))
    scale = max(1.0, float(np.linalg.norm(vector) ** 2))
    if abs(value.imag) > 1.0e-11 * scale:
        raise ValueError("Weight quadratic form is not real; W must be Hermitian.")
    if value.real < -1.0e-12 * scale:
        raise ValueError("Weight quadratic form is negative; W must be positive.")
    return float(np.sqrt(max(value.real, 0.0)))


def _weighted_column_norm(matrix: np.ndarray, weight: Optional[Any]) -> float:
    total = 0.0
    for column in range(matrix.shape[1]):
        norm = _positive_quadratic_norm(matrix[:, column], weight)
        total += norm * norm
    return float(np.sqrt(total))


def _vector_phase_array(
    size: int,
    *,
    phases: Optional[Sequence[float]],
    labels: Optional[Sequence[str]],
    field_phases: Optional[Mapping[str, float]],
) -> np.ndarray:
    if phases is not None and (labels is not None or field_phases is not None):
        raise ValueError("Specify explicit phases or labels/field_phases, not both.")
    if phases is not None:
        result = np.asarray(phases, dtype=np.float64)
        if result.shape != (size,):
            raise ValueError(f"phases must have shape {(size,)}.")
        return result
    if labels is None and field_phases is None:
        return np.zeros(size, dtype=np.float64)
    if labels is None or field_phases is None:
        raise ValueError("labels and field_phases must be supplied together.")
    label_array = np.asarray(labels, dtype=object)
    if label_array.shape != (size,):
        raise ValueError(f"labels must have shape {(size,)}.")
    try:
        return np.asarray(
            [field_phases[str(label)] for label in label_array],
            dtype=np.float64,
        )
    except KeyError as exc:
        raise ValueError(f"No phase was supplied for field {exc.args[0]!r}.") from exc


def check_eigenvector_transformed_realness(
    vector: np.ndarray,
    weight: Optional[Any] = None,
    *,
    phases: Optional[Sequence[float]] = None,
    labels: Optional[Sequence[str]] = None,
    field_phases: Optional[Mapping[str, float]] = None,
    phase_floor: float = 1.0e-12,
    tolerance: Tolerance = EIGENVECTOR_TOLERANCE,
    name: str = "eigenvector transformed realness",
) -> VerificationResult:
    """Remove field phases and one robust global phase, then test imaginary norm."""
    x = np.asarray(vector, dtype=np.complex128)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("vector must be a nonempty one-dimensional array.")
    dof_phases = _vector_phase_array(
        x.size,
        phases=phases,
        labels=labels,
        field_phases=field_phases,
    )
    transformed = np.exp(-1j * dof_phases) * x
    transformed_weight = _phase_transform_weight(weight, dof_phases)
    transformed_weighted = _matvec(transformed_weight, transformed)
    bilinear = transformed.T @ transformed_weighted
    norm = _positive_quadratic_norm(transformed, transformed_weight)
    if abs(bilinear) > float(phase_floor) * max(norm * norm, np.finfo(float).tiny):
        global_phase = 0.5 * float(np.angle(bilinear))
        phase_method = "bilinear"
    else:
        largest = int(np.argmax(np.abs(transformed)))
        global_phase = float(np.angle(transformed[largest]))
        phase_method = "largest-component"
    rotated = np.exp(-1j * global_phase) * transformed
    imaginary = np.imag(rotated).astype(np.complex128)
    absolute = _positive_quadratic_norm(imaginary, transformed_weight)
    largest_imaginary = int(np.argmax(np.abs(np.imag(rotated))))
    return _make_result(
        name=name,
        absolute_error=absolute,
        reference_scale=norm,
        tolerance=tolerance,
        worst_location={
            "dof": largest_imaginary,
            "imaginary_component": float(np.imag(rotated[largest_imaginary])),
        },
        metadata={
            "global_phase": global_phase,
            "phase_method": phase_method,
            "bilinear_form": complex(bilinear),
            "weight_transformed": weight is not None,
        },
    )


def cluster_eigenvalues(
    eigenvalues: Sequence[complex],
    *,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-8,
) -> tuple[np.ndarray, ...]:
    """Return deterministic transitive clusters of eigenvalue indices.

    Two values are linked when
    ``abs(a-b) <= atol + rtol * max(abs(a), abs(b))``.  Connected components
    are returned in first-input-index order, with indices sorted inside each
    component.
    """
    values = np.asarray(eigenvalues, dtype=np.complex128)
    if values.ndim != 1:
        raise ValueError("eigenvalues must be one-dimensional.")
    if not np.all(np.isfinite(values)):
        raise ValueError("eigenvalues contains non-finite values.")
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0.0 or rtol < 0.0:
        raise ValueError("atol and rtol must be finite and non-negative.")
    parents = np.arange(values.size, dtype=np.int64)

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parents[right_root] = left_root
        else:
            parents[left_root] = right_root

    for left in range(values.size):
        for right in range(left + 1, values.size):
            threshold = float(atol) + float(rtol) * max(
                abs(values[left]),
                abs(values[right]),
            )
            if abs(values[left] - values[right]) <= threshold:
                union(left, right)
    components: dict[int, list[int]] = {}
    for index in range(values.size):
        components.setdefault(find(index), []).append(index)
    ordered = sorted(components.values(), key=lambda component: component[0])
    return tuple(np.asarray(component, dtype=np.int64) for component in ordered)


@dataclass(frozen=True)
class RealEigenspaceResult:
    """Conjugation-invariance and real-basis gates for an eigenvalue cluster."""

    basis: np.ndarray
    conjugation_invariance: VerificationResult
    real_basis: VerificationResult
    real_rank: int
    singular_values: np.ndarray

    @property
    def results(self) -> tuple[VerificationResult, VerificationResult]:
        return (self.conjugation_invariance, self.real_basis)

    @property
    def passed(self) -> bool:
        return all(result.passed is True for result in self.results)

    def to_dict(self, *, include_basis: bool = False) -> dict[str, Any]:
        output = {
            "passed": self.passed,
            "real_rank": self.real_rank,
            "singular_values": self.singular_values,
            "conjugation_invariance": self.conjugation_invariance.to_dict(),
            "real_basis": self.real_basis.to_dict(),
        }
        if include_basis:
            output["basis"] = self.basis
        return _json_safe(output)


def check_eigenspace_transformed_realness(
    vectors: np.ndarray,
    weight: Optional[Any] = None,
    *,
    phases: Optional[Sequence[float]] = None,
    labels: Optional[Sequence[str]] = None,
    field_phases: Optional[Mapping[str, float]] = None,
    rank_tolerance: float = 1.0e-10,
    tolerance: Tolerance = EIGENVECTOR_TOLERANCE,
    name: str = "transformed-real eigenspace",
) -> RealEigenspaceResult:
    """Test whether a clustered eigenspace admits a real W-orthonormal basis.

    The field transform is applied first.  The resulting column space must be
    invariant under complex conjugation, and the real rank of
    ``[Re(Q), Im(Q)]`` must equal the complex subspace dimension.  This avoids
    rejecting arbitrary complex rotations of a genuinely real degenerate
    eigenspace.
    """
    X = np.asarray(vectors, dtype=np.complex128)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("vectors must be a nonempty two-dimensional column basis.")
    if (
        not np.isfinite(rank_tolerance)
        or rank_tolerance < 0.0
        or rank_tolerance >= 1.0
    ):
        raise ValueError("rank_tolerance must be finite and in [0, 1).")
    dof_phases = _vector_phase_array(
        X.shape[0],
        phases=phases,
        labels=labels,
        field_phases=field_phases,
    )
    transformed = np.exp(-1j * dof_phases)[:, None] * X
    transformed_weight = _phase_transform_weight(weight, dof_phases)
    orthonormal = w_orthonormalize(
        transformed,
        transformed_weight,
        rank_tolerance=max(rank_tolerance, np.finfo(float).eps),
    )

    conjugate_basis = np.conjugate(orthonormal)
    weighted_conjugate = _matvec(transformed_weight, conjugate_basis)
    conjugate_coordinates = orthonormal.conj().T @ weighted_conjugate
    conjugation_residual = conjugate_basis - orthonormal @ conjugate_coordinates
    conjugation_absolute = _weighted_column_norm(
        conjugation_residual,
        transformed_weight,
    )
    basis_norm = _weighted_column_norm(orthonormal, transformed_weight)
    worst_column = int(
        np.argmax(np.sqrt(np.sum(np.abs(conjugation_residual) ** 2, axis=0)))
    )
    worst_dof = int(np.argmax(np.abs(conjugation_residual[:, worst_column])))
    conjugation_result = _make_result(
        name=f"{name}: conjugation invariance",
        absolute_error=conjugation_absolute,
        reference_scale=basis_norm,
        tolerance=tolerance,
        worst_location={
            "vector": worst_column,
            "dof": worst_dof,
            "magnitude": float(abs(conjugation_residual[worst_dof, worst_column])),
        },
        metadata={"subspace_dimension": X.shape[1]},
    )

    real_span = np.concatenate(
        [np.real(orthonormal), np.imag(orthonormal)],
        axis=1,
    )
    left_vectors, singular_values, _ = np.linalg.svd(
        real_span,
        full_matrices=False,
    )
    leading = float(singular_values[0]) if singular_values.size else 0.0
    real_rank = int(
        np.count_nonzero(
            singular_values > float(rank_tolerance) * max(leading, 1.0)
        )
    )
    subspace_dimension = X.shape[1]
    forbidden_singular_values = singular_values[subspace_dimension:]
    tail_norm = float(np.linalg.norm(forbidden_singular_values))
    singular_norm = float(np.linalg.norm(singular_values))
    real_basis_result = _make_result(
        name=f"{name}: real-basis rank",
        absolute_error=tail_norm,
        reference_scale=singular_norm,
        tolerance=tolerance,
        worst_location={
            "singular_value_index": (
                subspace_dimension
                if forbidden_singular_values.size
                else None
            ),
            "largest_forbidden_singular_value": (
                float(forbidden_singular_values[0])
                if forbidden_singular_values.size
                else 0.0
            ),
        },
        metadata={
            "real_rank": real_rank,
            "required_rank": subspace_dimension,
            "rank_tolerance": rank_tolerance,
            "singular_values": singular_values,
        },
    )

    candidate = left_vectors[:, :subspace_dimension].astype(np.complex128)
    real_basis = w_orthonormalize(
        candidate,
        transformed_weight,
        rank_tolerance=max(rank_tolerance, np.finfo(float).eps),
    )
    return RealEigenspaceResult(
        basis=real_basis,
        conjugation_invariance=conjugation_result,
        real_basis=real_basis_result,
        real_rank=real_rank,
        singular_values=singular_values,
    )


@dataclass(frozen=True)
class EigenvectorParityResult:
    """Both parity residuals and the normalized parity expectation value."""

    even: VerificationResult
    odd: VerificationResult
    expectation: complex
    classification: str

    @property
    def results(self) -> tuple[VerificationResult, VerificationResult]:
        return (self.even, self.odd)

    @property
    def passed(self) -> bool:
        return self.even.passed is True or self.odd.passed is True

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "passed": self.passed,
                "classification": self.classification,
                "expectation": self.expectation,
                "even": self.even.to_dict(),
                "odd": self.odd.to_dict(),
            }
        )


def check_eigenvector_parity(
    vector: np.ndarray,
    reflection: Any,
    weight: Optional[Any] = None,
    *,
    tolerance: Tolerance = EIGENVECTOR_TOLERANCE,
    name: str = "eigenvector reflection parity",
) -> EigenvectorParityResult:
    """Measure even and odd reflection residuals in a positive metric."""
    x = np.asarray(vector, dtype=np.complex128)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("vector must be a nonempty one-dimensional array.")
    if reflection.shape != (x.size, x.size):
        raise ValueError("reflection shape does not match vector size.")
    reflected = np.asarray(reflection @ x)
    norm = _positive_quadratic_norm(x, weight)
    denominator_scale = 2.0 * norm
    even_difference = x - reflected
    odd_difference = x + reflected
    even_absolute = _positive_quadratic_norm(even_difference, weight)
    odd_absolute = _positive_quadratic_norm(odd_difference, weight)
    even = _make_result(
        name=f"{name}: even",
        absolute_error=even_absolute,
        reference_scale=denominator_scale,
        tolerance=tolerance,
        worst_location={
            "dof": int(np.argmax(np.abs(even_difference))),
            "magnitude": float(np.max(np.abs(even_difference))),
        },
    )
    odd = _make_result(
        name=f"{name}: odd",
        absolute_error=odd_absolute,
        reference_scale=denominator_scale,
        tolerance=tolerance,
        worst_location={
            "dof": int(np.argmax(np.abs(odd_difference))),
            "magnitude": float(np.max(np.abs(odd_difference))),
        },
    )
    denominator = np.vdot(x, _matvec(weight, x))
    expectation = np.vdot(x, _matvec(weight, reflected)) / denominator
    if even.passed:
        classification = "EVEN"
    elif odd.passed:
        classification = "ODD"
    else:
        classification = "MIXED"
    return EigenvectorParityResult(
        even=even,
        odd=odd,
        expectation=complex(expectation),
        classification=classification,
    )


def w_orthonormalize(
    vectors: np.ndarray,
    weight: Optional[Any] = None,
    *,
    rank_tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Return a W-orthonormal basis spanning the input column space."""
    X = np.asarray(vectors, dtype=np.complex128)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("vectors must be a nonempty two-dimensional column basis.")
    weighted = X if weight is None else np.asarray(weight @ X)
    gram = X.conj().T @ weighted
    gram = 0.5 * (gram + gram.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    largest = float(np.max(eigenvalues))
    if largest <= 0.0:
        raise ValueError("Input eigenspace has no positive W norm.")
    keep = eigenvalues > float(rank_tolerance) * largest
    if not np.all(keep):
        raise ValueError(
            "Input eigenspace is rank deficient in W: "
            f"retained {np.count_nonzero(keep)} of {eigenvalues.size} vectors."
        )
    inverse_sqrt = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues))
        @ eigenvectors.conj().T
    )
    return X @ inverse_sqrt


@dataclass(frozen=True)
class ParityAdaptedEigenspace:
    """W-orthonormal basis rotated to diagonalize reflection in a cluster."""

    basis: np.ndarray
    parity_eigenvalues: np.ndarray
    orthonormality: VerificationResult
    parity: VerificationResult
    subspace_reflection: np.ndarray

    @property
    def results(self) -> tuple[VerificationResult, VerificationResult]:
        return (self.orthonormality, self.parity)

    @property
    def passed(self) -> bool:
        return all(result.passed is True for result in self.results)

    def to_dict(self, *, include_basis: bool = False) -> dict[str, Any]:
        output = {
            "passed": self.passed,
            "parity_eigenvalues": self.parity_eigenvalues,
            "subspace_reflection": self.subspace_reflection,
            "orthonormality": self.orthonormality.to_dict(),
            "parity": self.parity.to_dict(),
        }
        if include_basis:
            output["basis"] = self.basis
        return _json_safe(output)


def parity_adapt_eigenspace(
    vectors: np.ndarray,
    reflection: Any,
    weight: Optional[Any] = None,
    *,
    rank_tolerance: float = 1.0e-12,
    tolerance: Tolerance = EIGENVECTOR_TOLERANCE,
    name: str = "parity-adapted eigenspace",
) -> ParityAdaptedEigenspace:
    """W-orthonormalize a cluster and rotate it into reflection eigenvectors."""
    basis = w_orthonormalize(
        vectors,
        weight,
        rank_tolerance=rank_tolerance,
    )
    if reflection.shape != (basis.shape[0], basis.shape[0]):
        raise ValueError("reflection shape does not match eigenspace row dimension.")
    weighted_basis = basis if weight is None else np.asarray(weight @ basis)
    gram = basis.conj().T @ weighted_basis
    identity = np.eye(basis.shape[1], dtype=np.complex128)
    gram_difference = gram - identity
    orthonormality = _make_result(
        name=f"{name}: W orthonormality",
        absolute_error=float(np.linalg.norm(gram_difference)),
        reference_scale=float(np.linalg.norm(identity)),
        tolerance=tolerance,
        worst_location=_matrix_worst_location(
            gram_difference,
            actual=gram,
            expected=identity,
        ),
    )

    reflected_basis = np.asarray(reflection @ basis)
    weighted_reflected = (
        reflected_basis
        if weight is None
        else np.asarray(weight @ reflected_basis)
    )
    subspace_reflection = basis.conj().T @ weighted_reflected
    hermitian_subspace = 0.5 * (
        subspace_reflection + subspace_reflection.conj().T
    )
    parity_values, rotation = np.linalg.eigh(hermitian_subspace)
    order = np.argsort(-parity_values)
    parity_values = parity_values[order]
    rotation = rotation[:, order]
    adapted = basis @ rotation
    residual = np.asarray(reflection @ adapted) - adapted * parity_values[None, :]
    residual_norm = _weighted_column_norm(residual, weight)
    adapted_norm = _weighted_column_norm(adapted, weight)
    eigenvalue_distance = float(
        np.max(np.minimum(np.abs(parity_values - 1.0), np.abs(parity_values + 1.0)))
    )
    combined_relative = max(
        residual_norm / max(adapted_norm, tolerance.floor),
        eigenvalue_distance,
    )
    combined_absolute = combined_relative * max(adapted_norm, tolerance.floor)
    parity_result = _make_result(
        name=f"{name}: reflection diagonalization",
        absolute_error=combined_absolute,
        reference_scale=adapted_norm,
        tolerance=tolerance,
        worst_location={
            "vector": int(
                np.argmax(
                    np.sqrt(np.sum(np.abs(residual) ** 2, axis=0))
                )
            ),
            "largest_parity_eigenvalue_distance": eigenvalue_distance,
        },
        metadata={
            "weighted_residual_norm": residual_norm,
            "parity_eigenvalues": parity_values,
            "subspace_antihermitian_norm": float(
                np.linalg.norm(
                    subspace_reflection - subspace_reflection.conj().T
                )
            ),
        },
    )
    return ParityAdaptedEigenspace(
        basis=adapted,
        parity_eigenvalues=parity_values,
        orthonormality=orthonormality,
        parity=parity_result,
        subspace_reflection=subspace_reflection,
    )


__all__ = [
    "ALGEBRAIC_TOLERANCE",
    "ASSEMBLY_TOLERANCE",
    "DEFAULT_TOLERANCES",
    "EIGENVECTOR_TOLERANCE",
    "FOURIER_TOLERANCE",
    "INDEX_TOLERANCE",
    "PETSC_TOLERANCE",
    "QUADRATURE_TOLERANCE",
    "EigenvectorParityResult",
    "FourierParityReport",
    "Parity",
    "ParityAdaptedEigenspace",
    "RealEigenspaceResult",
    "Tolerance",
    "VerificationResult",
    "build_reflection_operator",
    "check_eigenvector_parity",
    "check_eigenvector_transformed_realness",
    "check_eigenspace_transformed_realness",
    "check_fourier_parity",
    "check_grid_parity",
    "check_operator_intertwining",
    "check_operator_commutator",
    "check_parity_sector_mixing",
    "check_profile_fourier_parity",
    "check_reflection_covariance",
    "check_reflection_involution",
    "cluster_eigenvalues",
    "compare_positive_subblock",
    "format_summary",
    "parity_adapt_eigenspace",
    "parity_projectors",
    "reflection_permutation",
    "results_to_json",
    "w_orthonormalize",
]
