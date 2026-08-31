"""Pin the mixed Galerkin projection against the superNOVA_2 baseline.

``_project_local_tensor_mixed`` was rewritten to contract per element rather
than per nonzero.  These references were captured from the implementation at
f8b246e, so they hold the rewrite -- and any later one -- to the numbers the
original produced.

The checksum is a phase-weighted sum over the CSR values.  It is sensitive to
value *and* to ordering, so a projection that produced the right entries in
the wrong places would still fail.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import coo_array

from supernova.radial import BSplineFE
from supernova.core.solver4 import (
    _build_mixed_radial_data,
    _project_local_tensor_mixed,
)


# Ten of the sixteen field blocks are populated, with banded mode coupling --
# the structure _build_local_M_split produces for a real equilibrium.
_BLOCKS = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 3),
           (2, 0), (2, 2), (3, 1), (3, 3))


def build_inputs(spans, nmode, dm, order=3, quad_order=6, holes=0.0,
                 ragged=False, derivative=True):
    """Deterministic solver blocks, local tensor and mixed bases for one case."""
    high = BSplineFE(knots=np.linspace(0.0, 1.0, spans + 1), order=order,
                     domain=(0.0, 1.0), quad_order=quad_order)
    mixed = _build_mixed_radial_data(high)
    nq = mixed.weights_q.size
    rng = np.random.default_rng(spans * 1000 + nmode)

    rows, cols = [], []
    for row_field, col_field in _BLOCKS:
        for m in range(nmode):
            for mm in range(max(0, m - dm), min(nmode, m + dm + 1)):
                rows.append(row_field * nmode + m)
                cols.append(col_field * nmode + mm)
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    q = np.repeat(np.arange(nq), rows.size)
    r = np.tile(rows, nq)
    c = np.tile(cols, nq)
    data = rng.standard_normal(q.size) + 1j * rng.standard_normal(q.size)
    if holes:            # exact zeros make the pattern vary between points
        data[rng.random(data.size) < holes] = 0.0
    if ragged:           # dropped entries make it vary more aggressively
        keep = rng.random(q.size) > 0.15
        q, r, c, data = q[keep], r[keep], c[keep], data[keep]

    M = coo_array((data, (q, r, c)), shape=(nq, 4 * nmode, 4 * nmode))
    sb = SimpleNamespace(m_range=np.arange(nmode), dpsi_hat_dx=1.3)
    args = (mixed.phi_high_q, mixed.phi_low_q, mixed.weights_q, mixed.G_low_high)
    return sb, M, args, derivative


CASES = {
    "tiny": dict(spans=12, nmode=6, dm=2),
    "tiny_no_derivative": dict(spans=12, nmode=6, dm=2, derivative=False),
    "small": dict(spans=20, nmode=12, dm=5),
    "exact_zeros": dict(spans=20, nmode=12, dm=5, holes=0.30),
    "ragged_pattern": dict(spans=20, nmode=12, dm=5, ragged=True),
    "coarse_quadrature": dict(spans=20, nmode=12, dm=5, quad_order=4),
    "quintic": dict(spans=20, nmode=12, dm=5, order=5, quad_order=9),
    "diagonal_only": dict(spans=20, nmode=12, dm=0),
}

# name -> (n_rows, n_trial, nnz, phase-weighted checksum of the CSR values)
REFERENCE: dict[str, tuple[int, int, int, complex]] = {
    # Captured from supernova/core/solver4.py at f8b246e, before the rewrite.
    "tiny": (312, 324, 10752, complex(10.601583589342871, -21.33923951109286)),
    "tiny_no_derivative": (312, 324, 10752, complex(-1.3539437572482176, -1.0566174548508804)),
    "small": (1008, 1032, 75072, complex(1.505247396526396, -2.3294749889681308)),
    "exact_zeros": (1008, 1032, 75040, complex(3.987259532657669, 7.069731602110518)),
    "ragged_pattern": (1008, 1032, 75072, complex(2.452074436483597, -2.273860640885547)),
    "coarse_quadrature": (1008, 1032, 75072, complex(5.7873616344180565, -4.01651712720522)),
    "quintic": (1104, 1128, 166056, complex(0.7881928846841633, 8.284996376940615)),
    "diagonal_only": (1008, 1032, 8832, complex(0.7181674454396316, -0.74563221860582)),
}


def _checksum(matrix) -> complex:
    return complex(matrix.data @ np.exp(1j * np.arange(matrix.nnz)))


@pytest.mark.parametrize("name", sorted(CASES))
def test_projection_matches_the_baseline_reference(name):
    sb, M, args, derivative = build_inputs(**CASES[name])
    projected = _project_local_tensor_mixed(sb, M, *args, include_derivative=derivative)

    rows, trial, nnz, checksum = REFERENCE[name]
    assert projected.shape == (rows, trial)
    assert projected.nnz == nnz, (
        "The projection changed its sparsity pattern, which no reassociation "
        "of the quadrature sums can explain."
    )
    got = _checksum(projected)
    assert abs(got - checksum) <= 1e-11 * max(abs(checksum), 1.0), (
        f"Projection checksum drifted for {name!r}: {got!r} vs {checksum!r}."
    )
