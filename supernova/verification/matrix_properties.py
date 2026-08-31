"""Adjointness and phase-realness checks for dense and SciPy sparse matrices."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy import sparse

from supernova.verification.symmetry import (
    ALGEBRAIC_TOLERANCE,
    ASSEMBLY_TOLERANCE,
    Tolerance,
    VerificationResult,
    _json_safe,
    _make_result,
    _matrix_frobenius,
    _matrix_worst_location,
)


def _adjoint(matrix: Any) -> Any:
    if sparse.issparse(matrix):
        return matrix.conjugate().transpose().tocsr()
    array = np.asarray(matrix)
    return array.conj().T


def check_matrix_adjointness(
    matrix: Any,
    *,
    kind: str = "hermitian",
    tolerance: Tolerance = ASSEMBLY_TOLERANCE,
    name: Optional[str] = None,
) -> VerificationResult:
    """Check ordinary Hermitian or skew-Hermitian adjointness."""
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Adjointness requires a square matrix.")
    normalized = str(kind).lower().replace("_", "-")
    if normalized in ("hermitian", "self-adjoint", "selfadjoint"):
        expected = _adjoint(matrix)
        default_name = "matrix Hermiticity"
        relation = "A = A^H"
    elif normalized in (
        "skew-hermitian",
        "anti-hermitian",
        "skew-adjoint",
        "antihermitian",
    ):
        expected = -_adjoint(matrix)
        default_name = "matrix skew-Hermiticity"
        relation = "A = -A^H"
    else:
        raise ValueError(
            "kind must be 'hermitian' or 'skew-hermitian', "
            f"got {kind!r}."
        )
    difference = matrix - expected
    return _make_result(
        name=default_name if name is None else str(name),
        absolute_error=_matrix_frobenius(difference),
        reference_scale=_matrix_frobenius(matrix),
        tolerance=tolerance,
        worst_location=_matrix_worst_location(
            difference,
            actual=matrix,
            expected=expected,
        ),
        metadata={"kind": normalized, "relation": relation, "shape": matrix.shape},
    )


def check_hermitian(
    matrix: Any,
    *,
    tolerance: Tolerance = ASSEMBLY_TOLERANCE,
    name: str = "matrix Hermiticity",
) -> VerificationResult:
    return check_matrix_adjointness(
        matrix,
        kind="hermitian",
        tolerance=tolerance,
        name=name,
    )


def check_skew_hermitian(
    matrix: Any,
    *,
    tolerance: Tolerance = ASSEMBLY_TOLERANCE,
    name: str = "matrix skew-Hermiticity",
) -> VerificationResult:
    return check_matrix_adjointness(
        matrix,
        kind="skew-hermitian",
        tolerance=tolerance,
        name=name,
    )


def check_weighted_adjointness(
    matrix: Any,
    weight: Any,
    *,
    tolerance: Tolerance = ASSEMBLY_TOLERANCE,
    name: str = "weighted adjointness",
) -> VerificationResult:
    r"""Check ``A^H W = W A`` using the Frobenius norm."""
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Weighted adjointness requires a square matrix.")
    if tuple(weight.shape) != tuple(matrix.shape):
        raise ValueError("weight shape must match the square matrix shape.")
    actual = _adjoint(matrix) @ weight
    expected = weight @ matrix
    difference = actual - expected
    return _make_result(
        name=name,
        absolute_error=_matrix_frobenius(difference),
        reference_scale=_matrix_frobenius(expected),
        tolerance=tolerance,
        worst_location=_matrix_worst_location(
            difference,
            actual=actual,
            expected=expected,
        ),
        metadata={
            "relation": "A^H W = W A",
            "shape": matrix.shape,
            "weight_shape": weight.shape,
        },
    )


def _resolve_phase_vector(
    size: int,
    *,
    phases: Optional[Sequence[float]],
    labels: Optional[Sequence[str]],
    field_phases: Optional[Mapping[str, float]],
    side: str,
) -> Optional[np.ndarray]:
    if phases is not None and labels is not None:
        raise ValueError(f"Specify {side}_phases or {side}_labels, not both.")
    if phases is not None:
        result = np.asarray(phases, dtype=np.float64)
        if result.shape != (size,):
            raise ValueError(f"{side}_phases must have shape {(size,)}.")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{side}_phases contains non-finite values.")
        return result
    if labels is None:
        return None
    if field_phases is None:
        raise ValueError("field_phases is required when field labels are supplied.")
    label_array = np.asarray(labels, dtype=object)
    if label_array.shape != (size,):
        raise ValueError(f"{side}_labels must have shape {(size,)}.")
    try:
        result = np.asarray(
            [field_phases[str(label)] for label in label_array],
            dtype=np.float64,
        )
    except KeyError as exc:
        raise ValueError(f"No phase was supplied for field {exc.args[0]!r}.") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"Resolved {side} phases contain non-finite values.")
    return result


def phase_transform_matrix(
    matrix: Any,
    *,
    row_phases: Optional[Sequence[float]] = None,
    trial_phases: Optional[Sequence[float]] = None,
    row_labels: Optional[Sequence[str]] = None,
    trial_labels: Optional[Sequence[str]] = None,
    field_phases: Optional[Mapping[str, float]] = None,
) -> Any:
    r"""Return ``D_row^{-1} A D_trial``.

    Independent row and trial phase vectors make the transformation valid for
    rectangular mixed test/trial operators.  For a square operator only,
    omitted trial phases/labels default to the row representation.
    """
    if len(matrix.shape) != 2:
        raise ValueError("matrix must be two-dimensional.")
    n_rows, n_columns = matrix.shape
    resolved_rows = _resolve_phase_vector(
        n_rows,
        phases=row_phases,
        labels=row_labels,
        field_phases=field_phases,
        side="row",
    )
    resolved_trials = _resolve_phase_vector(
        n_columns,
        phases=trial_phases,
        labels=trial_labels,
        field_phases=field_phases,
        side="trial",
    )
    if resolved_rows is None:
        resolved_rows = np.zeros(n_rows, dtype=np.float64)
    if resolved_trials is None:
        if n_rows != n_columns:
            raise ValueError(
                "Rectangular matrices require explicit trial_phases or trial_labels."
            )
        resolved_trials = resolved_rows.copy()

    left = np.exp(-1j * resolved_rows)
    right = np.exp(1j * resolved_trials)
    if sparse.issparse(matrix):
        transformed = (
            sparse.diags(left, format="csr")
            @ matrix
            @ sparse.diags(right, format="csr")
        )
        return transformed.tocsr()
    return left[:, None] * np.asarray(matrix) * right[None, :]


def _imaginary_matrix(matrix: Any) -> Any:
    if sparse.issparse(matrix):
        output = matrix.copy().astype(np.complex128)
        output.data = np.imag(output.data).astype(np.complex128)
        output.eliminate_zeros()
        return output
    return np.imag(np.asarray(matrix))


def check_transformed_realness(
    matrix: Any,
    *,
    row_phases: Optional[Sequence[float]] = None,
    trial_phases: Optional[Sequence[float]] = None,
    row_labels: Optional[Sequence[str]] = None,
    trial_labels: Optional[Sequence[str]] = None,
    field_phases: Optional[Mapping[str, float]] = None,
    tolerance: Tolerance = ASSEMBLY_TOLERANCE,
    name: str = "transformed matrix realness",
) -> VerificationResult:
    """Check the imaginary Frobenius norm after a two-sided phase transform."""
    transformed = phase_transform_matrix(
        matrix,
        row_phases=row_phases,
        trial_phases=trial_phases,
        row_labels=row_labels,
        trial_labels=trial_labels,
        field_phases=field_phases,
    )
    imaginary = _imaginary_matrix(transformed)
    zero = (
        sparse.csr_matrix(transformed.shape, dtype=np.complex128)
        if sparse.issparse(transformed)
        else np.zeros_like(np.asarray(transformed), dtype=np.complex128)
    )
    return _make_result(
        name=name,
        absolute_error=_matrix_frobenius(imaginary),
        reference_scale=_matrix_frobenius(transformed),
        tolerance=tolerance,
        worst_location=_matrix_worst_location(
            imaginary,
            actual=imaginary,
            expected=zero,
        ),
        metadata={
            "shape": transformed.shape,
            "sparse": bool(sparse.issparse(transformed)),
            "two_sided": True,
        },
    )


def _matrix_mapping(matrices: Any) -> dict[str, Any]:
    if isinstance(matrices, Mapping):
        if not matrices:
            raise ValueError("At least one matrix is required.")
        return {str(name): matrix for name, matrix in matrices.items()}
    if hasattr(matrices, "shape") and len(matrices.shape) == 2:
        return {"matrix": matrices}
    try:
        sequence = list(matrices)
    except TypeError as exc:
        raise TypeError("matrices must be a matrix, mapping, or sequence.") from exc
    if not sequence:
        raise ValueError("At least one matrix is required.")
    return {f"matrix_{index}": matrix for index, matrix in enumerate(sequence)}


def _labels_for_matrix(
    specification: Any,
    matrix_name: str,
    expected_size: int,
    *,
    side: str,
) -> np.ndarray:
    if isinstance(specification, Mapping):
        if matrix_name not in specification:
            raise ValueError(f"No {side} labels supplied for {matrix_name!r}.")
        labels = specification[matrix_name]
    else:
        labels = specification
    output = np.asarray(labels, dtype=object)
    if output.shape != (expected_size,):
        raise ValueError(
            f"{side} labels for {matrix_name!r} must have shape "
            f"{(expected_size,)}, got {output.shape}."
        )
    return output


@dataclass(frozen=True)
class PhaseInferenceResult:
    """Best common quarter-turn field phases found across multiple matrices."""

    result: VerificationResult
    field_phases: Mapping[str, float]
    quarter_turns: Mapping[str, int]
    matrix_results: Mapping[str, VerificationResult]
    tested_assignments: int
    equivalent_best_assignments: int

    @property
    def results(self) -> tuple[VerificationResult, ...]:
        return (self.result, *self.matrix_results.values())

    @property
    def passed(self) -> bool:
        return self.result.passed is True

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "passed": self.passed,
                "field_phases": self.field_phases,
                "quarter_turns": self.quarter_turns,
                "tested_assignments": self.tested_assignments,
                "equivalent_best_assignments": self.equivalent_best_assignments,
                "result": self.result.to_dict(),
                "matrices": {
                    name: result.to_dict()
                    for name, result in self.matrix_results.items()
                },
            }
        )


def infer_field_phase_transform(
    matrices: Any,
    *,
    row_labels: Sequence[str] | Mapping[str, Sequence[str]],
    trial_labels: Optional[
        Sequence[str] | Mapping[str, Sequence[str]]
    ] = None,
    reference_field: Optional[str] = None,
    tolerance: Tolerance = ASSEMBLY_TOLERANCE,
    max_assignments: int = 65536,
    name: str = "quarter-turn field-phase inference",
) -> PhaseInferenceResult:
    """Exhaustively infer a common diagonal phase transform.

    Each field phase is restricted to ``0, pi/2, pi, 3*pi/2`` and one field is
    fixed at zero to remove the global gauge.  All supplied matrices contribute
    to one aggregate imaginary Frobenius norm, so a phase choice must work
    consistently across independently assembled operator coefficients.
    """
    named_matrices = _matrix_mapping(matrices)
    rows_by_name: dict[str, np.ndarray] = {}
    trials_by_name: dict[str, np.ndarray] = {}
    fields_in_order: list[str] = []
    for matrix_name, matrix in named_matrices.items():
        if len(matrix.shape) != 2:
            raise ValueError(f"Matrix {matrix_name!r} is not two-dimensional.")
        row = _labels_for_matrix(
            row_labels,
            matrix_name,
            matrix.shape[0],
            side="row",
        )
        if trial_labels is None:
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError(
                    f"Rectangular matrix {matrix_name!r} requires trial_labels."
                )
            trial = row.copy()
        else:
            trial = _labels_for_matrix(
                trial_labels,
                matrix_name,
                matrix.shape[1],
                side="trial",
            )
        rows_by_name[matrix_name] = row
        trials_by_name[matrix_name] = trial
        for label in (*row, *trial):
            label_string = str(label)
            if label_string not in fields_in_order:
                fields_in_order.append(label_string)

    if reference_field is None:
        reference = fields_in_order[0]
    else:
        reference = str(reference_field)
        if reference not in fields_in_order:
            raise ValueError(f"reference_field {reference!r} is absent from labels.")
    free_fields = [field for field in fields_in_order if field != reference]
    assignment_count = 4 ** len(free_fields)
    if assignment_count > int(max_assignments):
        raise ValueError(
            f"Quarter-turn inference requires {assignment_count} assignments, "
            f"exceeding max_assignments={int(max_assignments)}."
        )

    total_norm_squared = sum(
        _matrix_frobenius(matrix) ** 2 for matrix in named_matrices.values()
    )
    best_imaginary_squared = np.inf
    best_turns: Optional[dict[str, int]] = None
    scores: list[float] = []
    for assignment in itertools.product(range(4), repeat=len(free_fields)):
        turns = {reference: 0}
        turns.update(zip(free_fields, assignment))
        phase_map = {
            field: 0.5 * np.pi * turn for field, turn in turns.items()
        }
        imaginary_squared = 0.0
        for matrix_name, matrix in named_matrices.items():
            transformed = phase_transform_matrix(
                matrix,
                row_labels=rows_by_name[matrix_name],
                trial_labels=trials_by_name[matrix_name],
                field_phases=phase_map,
            )
            imaginary_squared += _matrix_frobenius(
                _imaginary_matrix(transformed)
            ) ** 2
        scores.append(imaginary_squared)
        if imaginary_squared < best_imaginary_squared:
            best_imaginary_squared = imaginary_squared
            best_turns = turns
    assert best_turns is not None
    best_phases = {
        field: 0.5 * np.pi * turn for field, turn in best_turns.items()
    }
    absolute = float(np.sqrt(max(best_imaginary_squared, 0.0)))
    reference_norm = float(np.sqrt(total_norm_squared))
    score_tolerance = max(
        np.finfo(float).eps * max(total_norm_squared, 1.0) * 64.0,
        tolerance.threshold(reference_norm) ** 2,
    )
    equivalent = sum(
        abs(score - best_imaginary_squared) <= score_tolerance for score in scores
    )

    matrix_results = {
        matrix_name: check_transformed_realness(
            matrix,
            row_labels=rows_by_name[matrix_name],
            trial_labels=trials_by_name[matrix_name],
            field_phases=best_phases,
            tolerance=tolerance,
            name=f"{matrix_name} transformed realness",
        )
        for matrix_name, matrix in named_matrices.items()
    }
    if matrix_results:
        worst_matrix_name, worst_matrix_result = max(
            matrix_results.items(),
            key=lambda item: item[1].relative_error,
        )
        worst_location = {
            "matrix": worst_matrix_name,
            **(worst_matrix_result.worst_location or {}),
        }
    else:
        worst_location = None
    aggregate = _make_result(
        name=name,
        absolute_error=absolute,
        reference_scale=reference_norm,
        tolerance=tolerance,
        worst_location=worst_location,
        metadata={
            "field_phases": best_phases,
            "quarter_turns": best_turns,
            "matrix_names": tuple(named_matrices),
            "tested_assignments": assignment_count,
            "equivalent_best_assignments": equivalent,
        },
    )
    return PhaseInferenceResult(
        result=aggregate,
        field_phases=best_phases,
        quarter_turns=best_turns,
        matrix_results=matrix_results,
        tested_assignments=assignment_count,
        equivalent_best_assignments=equivalent,
    )


__all__ = [
    "PhaseInferenceResult",
    "check_hermitian",
    "check_matrix_adjointness",
    "check_skew_hermitian",
    "check_transformed_realness",
    "check_weighted_adjointness",
    "infer_field_phase_transform",
    "phase_transform_matrix",
]
