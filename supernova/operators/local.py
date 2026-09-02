r"""The point-local split of the ideal-MHD operator.

``M(s)`` is separated into the three coefficient tensors of

    M(s) = Mminus1 / s + M0 + s * M1,

each holding the four-field local state ``[p1, xi^psi, xi^s, div_xi]`` at every
quadrature point, with ``lambda = mu_0 * density * s``.  The split is what lets
one assembly serve both the quadratic PEP and, at exact ``gamma_s == 0`` where
``Mminus1`` vanishes, the linear GEVP.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_array, csr_matrix

from supernova.operators.blocks import (
    mu_0,
    SolverBlocks,
    _as_coo_tensor,
    _filter_zero_data,
    _combine_coo_tensors,
    _scale_coo_axis0,
    _offset_coo_tensor,
)


def _matrix_stats(name: str, matrix: csr_matrix) -> str:
    """
    Return a one-line sparse matrix summary.
    """
    density = matrix.nnz / max(1, matrix.shape[0] * matrix.shape[1])
    return f"{name}: shape={matrix.shape}, nnz={matrix.nnz}, density={density:.3e}"


def _validate_split_block_shapes(sb: SolverBlocks):
    """
    Check the local block shapes expected by the four-field solver state.
    """
    nmode = sb.m_range.size
    npsi = sb.C0.shape[0]
    nstates_xip1 = 2 * nmode

    xip1_shape = (npsi, nstates_xip1, nstates_xip1)
    harmonic_shape = (npsi, nmode, nmode)
    expected_shapes = {
        "C0": xip1_shape,
        "D0": xip1_shape,
        "F": xip1_shape,
        "A": xip1_shape,
        "B": xip1_shape,
        "C1": harmonic_shape,
        "D1": harmonic_shape,
        "gtp_over_gpp": harmonic_shape,
        "gpz_over_gpp": harmonic_shape,
    }
    for block_name, expected_shape in expected_shapes.items():
        tensor = _as_coo_tensor(getattr(sb, block_name), block_name)
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{block_name} must have shape {expected_shape}, got {tensor.shape}."
            )


def _select_coo_rows(
    tensor,
    row_start: int | None = None,
    row_stop: int | None = None,
    name: str = "tensor",
) -> coo_array:
    """
    Keep only entries whose local row is in ``[row_start, row_stop)``.
    """
    tensor = _as_coo_tensor(tensor, name)
    q, rows, cols = tensor.coords
    mask = np.ones(tensor.nnz, dtype=bool)
    if row_start is not None:
        mask &= rows >= int(row_start)
    if row_stop is not None:
        mask &= rows < int(row_stop)

    if not np.any(mask):
        return coo_array(tensor.shape, dtype=np.complex128)

    data = tensor.data.astype(np.complex128, copy=True)[mask]
    coords = (q[mask].copy(), rows[mask].copy(), cols[mask].copy())
    data, coords = _filter_zero_data(data, coords)
    return coo_array((data, coords), shape=tensor.shape, dtype=np.complex128)


def _slice_to_quadrature(tensor, nq: int, name: str) -> coo_array:
    """
    Drop the optional edge operator point and keep quadrature slices only.
    """
    tensor = _as_coo_tensor(tensor, name)
    if tensor.shape[0] == nq:
        return tensor
    if tensor.shape[0] == nq + 1:
        return tensor[:nq]
    raise ValueError(
        f"{name} must use {nq} quadrature points, optionally plus one edge point. "
        f"Got {tensor.shape[0]} local points."
    )


def _build_local_M_split(
    sb: SolverBlocks,
    regularizer: float = 0.0,
) -> tuple[coo_array, coo_array, coo_array]:
    r"""
    Build local COO tensors ``M0``, ``M1`` and ``Mminus1`` for ``M(s)``.

    The split follows ``lambda = mu_0 * density * s`` and preserves the
    four-field local state ``[p1, xi^psi, xi^s, div_xi]``.
    """
    _validate_split_block_shapes(sb)

    nmode = sb.m_range.size
    nstates_xip1 = 2 * nmode
    nstates = 4 * nmode
    npsi = sb.C0.shape[0]

    rho = mu_0 * np.asarray(sb.density, dtype=np.complex128)
    if rho.shape != (npsi,):
        raise ValueError(f"density must have shape {(npsi,)}, got {rho.shape}.")
    if np.any(rho == 0.0):
        raise ValueError("density cannot contain zero values in the split 1/s operator.")

    xip1_shape = (npsi, nstates_xip1, nstates_xip1)
    harmonic_shape = (npsi, nmode, nmode)
    full_shape = (npsi, nstates, nstates)
    ones = np.ones(npsi, dtype=np.complex128)

    phase = _combine_coo_tensors(harmonic_shape, sb.gtp_over_gpp, sb.gpz_over_gpp)
    C0 = _combine_coo_tensors(
        xip1_shape,
        sb.C0,
        _offset_coo_tensor(phase, 0, 0, xip1_shape, "phase", scale=-1.0),
        _offset_coo_tensor(
            phase,
            nmode,
            nmode,
            xip1_shape,
            "phase",
            scale=-1.0,
        ),
    )
    C1 = _offset_coo_tensor(
        _scale_coo_axis0(sb.C1, rho, "C1"),
        0,
        nmode,
        xip1_shape,
        "C1",
    )

    D0 = _as_coo_tensor(sb.D0, "D0")
    Dminus1 = _offset_coo_tensor(
        _scale_coo_axis0(sb.D1, 1.0 / rho, "D1"),
        nmode,
        nmode,
        xip1_shape,
        "D1",
    )
    F = _as_coo_tensor(sb.F, "F")

    r = (1.0 + 1.0j * regularizer) ** 2
    A_const = _scale_coo_axis0(sb.A, ones, "A", scale=-1.0)
    A_inverse = _scale_coo_axis0(sb.A, 1.0 / rho, "A", scale=-1.0)
    B_linear = _scale_coo_axis0(sb.B, rho, "B", scale=r)
    B_const = _scale_coo_axis0(sb.B, ones, "B", scale=r)

    E0 = _combine_coo_tensors(
        xip1_shape,
        _select_coo_rows(A_const, row_stop=nmode, name="A"),
        _select_coo_rows(B_const, row_start=nmode, name="B"),
    )
    E1 = _select_coo_rows(B_linear, row_stop=nmode, name="B")
    Eminus1 = _select_coo_rows(A_inverse, row_start=nmode, name="A")

    M0 = _combine_coo_tensors(
        full_shape,
        _offset_coo_tensor(C0, 0, 0, full_shape, "C0"),
        _offset_coo_tensor(D0, 0, nstates_xip1, full_shape, "D0"),
        _offset_coo_tensor(F, nstates_xip1, 0, full_shape, "F"),
        _offset_coo_tensor(E0, nstates_xip1, nstates_xip1, full_shape, "E0", scale=-1.0),
    )
    M1 = _combine_coo_tensors(
        full_shape,
        _offset_coo_tensor(C1, 0, 0, full_shape, "C1"),
        _offset_coo_tensor(E1, nstates_xip1, nstates_xip1, full_shape, "E1", scale=-1.0),
    )
    Mminus1 = _combine_coo_tensors(
        full_shape,
        _offset_coo_tensor(Dminus1, 0, nstates_xip1, full_shape, "Dminus1"),
        _offset_coo_tensor(
            Eminus1,
            nstates_xip1,
            nstates_xip1,
            full_shape,
            "Eminus1",
            scale=-1.0,
        ),
    )
    return M0, M1, Mminus1
