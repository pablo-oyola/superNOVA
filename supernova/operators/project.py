r"""Galerkin projection of a local tensor onto the mixed radial spaces.

This is index bookkeeping, not physics: the tensor arrives already built and
leaves mapped onto global rows and columns.  It contracts one element at a
time, summing over that element's quadrature points before emitting any COO
triplet, which is both far faster than working per nonzero and far lighter on
memory -- the emitted triplet count drops by a factor of the quadrature order.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from supernova.operators.blocks import (
    _as_coo_tensor,
    _validate_dpsi_hat_dx,
    solver_blocks2,
)
from supernova.operators.mixed import MixedFieldLayout


def _project_local_tensor_mixed(
    sb: solver_blocks2,
    M,
    phi_high_q: np.ndarray,
    phi_low_q: np.ndarray,
    weights_q: np.ndarray,
    G_low_high,
    include_derivative: bool = False,
) -> csr_matrix:
    r"""Project one local split tensor into the mixed field-major spaces.

    The Galerkin contribution is ``-<Phi_L, M Phi_trial>`` after multiplying
    its first two differential-equation rows by ``sb.dpsi_hat_dx``. For the
    first two state groups, the optional derivative contribution is
    ``+<Phi_L, d(Phi_H)/dx> = +G_low_high``.
    """
    M = _as_coo_tensor(M, "M")
    phi_high_q = np.asarray(phi_high_q)
    phi_low_q = np.asarray(phi_low_q)
    weights_q = np.asarray(weights_q)
    G_low_high = csr_matrix(G_low_high)

    if phi_high_q.ndim != 2:
        raise ValueError("phi_high_q must have shape (n_quad, n_high).")
    if phi_low_q.ndim != 2:
        raise ValueError("phi_low_q must have shape (n_quad, n_low).")

    nq = M.shape[0]
    n_modes = np.asarray(sb.m_range).size
    dpsi_hat_dx = _validate_dpsi_hat_dx(
        getattr(sb, "dpsi_hat_dx", 1.0)
    )
    n_state = 4 * n_modes
    if M.shape[1:] != (n_state, n_state):
        raise ValueError(
            f"M state dimensions must be {(n_state, n_state)}, got {M.shape[1:]}."
        )
    if phi_high_q.shape[0] != nq or phi_low_q.shape[0] != nq:
        raise ValueError("M and both radial bases must use the same quadrature points.")
    if weights_q.shape != (nq,):
        raise ValueError(f"weights_q must have shape {(nq,)}, got {weights_q.shape}.")

    n_high = phi_high_q.shape[1]
    n_low = phi_low_q.shape[1]
    layout = MixedFieldLayout(n_high=n_high, n_low=n_low, n_modes=n_modes)
    if G_low_high.shape != (n_low, n_high):
        raise ValueError(
            f"G_low_high must have shape {(n_low, n_high)}, "
            f"got {G_low_high.shape}."
        )

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []

    if include_derivative:
        derivative = coo_matrix(G_low_high)
        if derivative.nnz:
            fields, modes = np.divmod(
                np.arange(2 * n_modes, dtype=np.int64), n_modes
            )
            offsets = np.asarray(layout.trial_offsets, dtype=np.int64)
            row_parts.append(
                (
                    fields[:, None] * n_low * n_modes
                    + derivative.row.astype(np.int64, copy=False)[None, :] * n_modes
                    + modes[:, None]
                ).ravel()
            )
            col_parts.append(
                (
                    offsets[fields][:, None]
                    + derivative.col.astype(np.int64, copy=False)[None, :] * n_modes
                    + modes[:, None]
                ).ravel()
            )
            data_parts.append(
                np.broadcast_to(
                    derivative.data.astype(np.complex128, copy=False),
                    (fields.size, derivative.nnz),
                ).ravel()
            )

    q_coords, local_rows, local_cols = M.coords
    local_data = M.data.astype(np.complex128, copy=True)
    local_data[local_rows < 2 * n_modes] *= dpsi_hat_dx
    nonzero = local_data != 0.0
    q_coords = q_coords[nonzero].astype(np.int64, copy=False)
    local_rows = local_rows[nonzero].astype(np.int64, copy=False)
    local_cols = local_cols[nonzero].astype(np.int64, copy=False)
    local_data = local_data[nonzero]

    if local_data.size:
        row_field, row_mode = np.divmod(local_rows, n_modes)
        col_field, col_mode = np.divmod(local_cols, n_modes)
        trial_offsets = np.asarray(layout.trial_offsets, dtype=np.int64)

        # Quadrature points that see the same basis functions form one element.
        # Grouping by the support signature rather than by position keeps this
        # correct for any point ordering, and the element blocks below then sum
        # over quadrature before any global index is materialised.
        support_signature = np.concatenate(
            [phi_low_q != 0.0, phi_high_q != 0.0], axis=1
        )
        _, element_of_q = np.unique(support_signature, axis=0, return_inverse=True)
        element_of_q = np.ravel(element_of_q)
        order = np.argsort(element_of_q[q_coords], kind="stable")
        entry_bounds = np.searchsorted(
            element_of_q[q_coords][order], np.arange(element_of_q.max() + 2)
        )

        for element in range(entry_bounds.size - 1):
            first, last = entry_bounds[element], entry_bounds[element + 1]
            if first == last:
                continue
            entries = order[first:last]
            quad = np.flatnonzero(element_of_q == element)

            # One support per element: the union over its quadrature points.
            # Any spurious index contributes an exact zero and is dropped by
            # eliminate_zeros below, so a slightly wide support stays correct.
            test_support = np.flatnonzero(np.any(phi_low_q[quad] != 0.0, axis=0))
            if test_support.size == 0:
                continue
            test_values = phi_low_q[np.ix_(quad, test_support)]

            local_q = np.searchsorted(quad, q_coords[entries])
            weighted = local_data[entries] * weights_q[q_coords[entries]]

            for low_trial in (False, True):
                selected = (col_field[entries] >= 2) == low_trial
                if not selected.any():
                    continue
                trial_basis = phi_low_q if low_trial else phi_high_q
                trial_support = np.flatnonzero(
                    np.any(trial_basis[quad] != 0.0, axis=0)
                )
                if trial_support.size == 0:
                    continue
                trial_values = trial_basis[np.ix_(quad, trial_support)]

                chosen = entries[selected]
                # Distinct (row, col) pairs in this element and trial space.
                pair_key = local_rows[chosen] * n_state + local_cols[chosen]
                pairs, pair_of_entry = np.unique(pair_key, return_inverse=True)
                accumulated = np.zeros(
                    (pairs.size, quad.size), dtype=np.complex128
                )
                np.add.at(
                    accumulated,
                    (pair_of_entry, local_q[selected]),
                    weighted[selected],
                )

                block = -np.einsum(
                    "gq,qi,qj->gij",
                    accumulated,
                    test_values,
                    trial_values,
                    optimize=True,
                )
                pair_rows, pair_cols = np.divmod(pairs, n_state)
                pair_row_field, pair_row_mode = np.divmod(pair_rows, n_modes)
                pair_col_field, pair_col_mode = np.divmod(pair_cols, n_modes)
                row_parts.append(
                    np.broadcast_to(
                        pair_row_field[:, None, None] * n_low * n_modes
                        + test_support[None, :, None] * n_modes
                        + pair_row_mode[:, None, None],
                        block.shape,
                    ).ravel()
                )
                col_parts.append(
                    np.broadcast_to(
                        trial_offsets[pair_col_field][:, None, None]
                        + trial_support[None, None, :] * n_modes
                        + pair_col_mode[:, None, None],
                        block.shape,
                    ).ravel()
                )
                data_parts.append(block.ravel())

    if not data_parts:
        return csr_matrix((layout.n_rows, layout.n_trial), dtype=np.complex128)

    projected = coo_matrix(
        (
            np.concatenate(data_parts),
            (np.concatenate(row_parts), np.concatenate(col_parts)),
        ),
        shape=(layout.n_rows, layout.n_trial),
        dtype=np.complex128,
    ).tocsr()
    projected.sum_duplicates()
    projected.eliminate_zeros()
    return projected
