"""The COO combine must equal scipy's coalesced sum, sorted or not.

``_coo_tensor_from_parts`` skips the duplicate sort when the parts sit at
disjoint row/column offsets and coalesces with one integer key otherwise.
Both paths are checked against a reference built with scipy's own
``sum_duplicates`` on the flattened tensor.
"""

import numpy as np
import pytest
from scipy.sparse import coo_array, csr_matrix

from supernova.operators.blocks import (
    _coo_tensor_from_parts,
    _combine_coo_tensors,
    _offset_coo_tensor,
    _parts_are_disjoint,
)


def _canonical(tensor) -> csr_matrix:
    """Flatten a 3-D COO tensor to a canonical 2-D CSR matrix."""
    q, rows, cols = tensor.coords
    flat = csr_matrix(
        (tensor.data, (q * tensor.shape[1] + rows, cols)),
        shape=(tensor.shape[0] * tensor.shape[1], tensor.shape[2]),
        dtype=np.complex128,
    )
    flat.sum_duplicates()
    flat.eliminate_zeros()
    flat.sort_indices()
    return flat


def _reference(shape, data_parts, coord_parts) -> csr_matrix:
    if not data_parts:
        return csr_matrix((shape[0] * shape[1], shape[2]), dtype=np.complex128)
    data = np.concatenate(data_parts)
    coords = tuple(np.concatenate([part[axis] for part in coord_parts]) for axis in range(3))
    out = coo_array((data, coords), shape=shape, dtype=np.complex128)
    out.sum_duplicates()
    out.eliminate_zeros()
    return _canonical(out)


def _assert_same(result, reference):
    canonical = _canonical(result)
    assert canonical.shape == reference.shape
    np.testing.assert_array_equal(canonical.indptr, reference.indptr)
    np.testing.assert_array_equal(canonical.indices, reference.indices)
    # Duplicates may be summed in a different order than scipy sums them;
    # the values agree to rounding, the pattern exactly.
    np.testing.assert_allclose(canonical.data, reference.data, rtol=0, atol=1e-14)


def _random_part(rng, shape, row_range, col_range, nnz, zeros=0):
    q = rng.integers(0, shape[0], nnz)
    rows = rng.integers(row_range[0], row_range[1], nnz)
    cols = rng.integers(col_range[0], col_range[1], nnz)
    data = rng.normal(size=nnz) + 1.0j * rng.normal(size=nnz)
    if zeros:
        data[rng.choice(nnz, zeros, replace=False)] = 0.0
    return data, (q, rows, cols)


@pytest.mark.parametrize("seed", range(3))
def test_disjoint_parts_are_concatenated_without_duplicates(seed):
    rng = np.random.default_rng(seed)
    shape = (7, 12, 12)
    parts = [
        _random_part(rng, shape, (0, 6), (0, 6), 40, zeros=3),
        _random_part(rng, shape, (6, 12), (6, 12), 40),
        _random_part(rng, shape, (0, 6), (6, 12), 25),
    ]
    data_parts = [p[0] for p in parts]
    coord_parts = [p[1] for p in parts]
    assert _parts_are_disjoint(coord_parts)
    result = _coo_tensor_from_parts(shape, data_parts, coord_parts)
    _assert_same(result, _reference(shape, data_parts, coord_parts))
    # the within-part duplicates random coordinates can create are coalesced by
    # the reference; the result must therefore equal it only after canonicalising,
    # which _assert_same does, and must contain no exact zeros.
    assert np.all(result.data != 0.0)


@pytest.mark.parametrize("seed", range(3))
def test_overlapping_parts_are_coalesced_in_canonical_order(seed):
    rng = np.random.default_rng(100 + seed)
    shape = (5, 9, 9)
    parts = [
        _random_part(rng, shape, (0, 9), (0, 9), 120, zeros=5),
        _random_part(rng, shape, (2, 7), (2, 7), 80),
    ]
    data_parts = [p[0] for p in parts]
    coord_parts = [p[1] for p in parts]
    assert not _parts_are_disjoint(coord_parts)
    result = _coo_tensor_from_parts(shape, data_parts, coord_parts)
    _assert_same(result, _reference(shape, data_parts, coord_parts))
    key = (result.coords[0] * shape[1] + result.coords[1]) * shape[2] + result.coords[2]
    assert np.all(np.diff(key) > 0), "coalesced output must be sorted and unique"
    assert result.has_canonical_format


def test_cancelling_duplicates_are_dropped():
    shape = (2, 3, 3)
    data_parts = [np.array([1.0 + 1.0j, 2.0]), np.array([-1.0 - 1.0j, 3.0])]
    coord_parts = [
        (np.array([0, 1]), np.array([1, 2]), np.array([1, 0])),
        (np.array([0, 1]), np.array([1, 2]), np.array([1, 1])),
    ]
    result = _coo_tensor_from_parts(shape, data_parts, coord_parts)
    assert result.nnz == 3
    _assert_same(result, _reference(shape, data_parts, coord_parts))


def test_empty_inputs():
    shape = (3, 4, 4)
    assert _coo_tensor_from_parts(shape, [], []).nnz == 0
    zero = _coo_tensor_from_parts(
        shape, [np.zeros(2)], [(np.array([0, 1]), np.array([1, 2]), np.array([1, 2]))]
    )
    assert zero.nnz == 0 and zero.shape == shape


def test_combine_of_offset_blocks_matches_dense_sum():
    rng = np.random.default_rng(7)
    nq, nm = 4, 3
    small = (nq, nm, nm)
    full = (nq, 2 * nm, 2 * nm)
    a = coo_array(rng.normal(size=small) + 1.0j * rng.normal(size=small))
    b = coo_array(rng.normal(size=small))
    combined = _combine_coo_tensors(
        full,
        _offset_coo_tensor(a, 0, 0, full, "a"),
        _offset_coo_tensor(b, nm, nm, full, "b", scale=-2.0),
        _offset_coo_tensor(a, 0, 0, full, "a again"),  # overlaps the first block
    )
    expected = np.zeros(full, dtype=np.complex128)
    expected[:, :nm, :nm] = 2.0 * a.toarray()
    expected[:, nm:, nm:] = -2.0 * b.toarray()
    dense = np.zeros(full, dtype=np.complex128)
    np.add.at(dense, combined.coords, combined.data)
    np.testing.assert_allclose(dense, expected, rtol=0, atol=1e-14)
