"""Acceptance: the new projection must reproduce the baseline exactly."""
import importlib.util, sys, time
import numpy as np
from types import SimpleNamespace
from scipy.sparse import coo_array

from supernova.radial import BSplineFE
from supernova.core.solver4 import _build_mixed_radial_data, _project_local_tensor_mixed

import supernova.core.solver4 as _sn4
spec = importlib.util.spec_from_file_location(
    "solver4_baseline",
    "/tmp/claude-1000/-home-pablooyola-Desktop-Pablo-codes-superNOVA-2/"
    "e71cdc67-2f0f-4cf5-b030-0cb72b5bafbf/scratchpad/solver4_baseline.py")
baseline = importlib.util.module_from_spec(spec)
sys.modules["solver4_baseline"] = baseline
spec.loader.exec_module(baseline)


def make_tensor(nq, nmode, dm, rng, *, holes=0.0, ragged=False):
    n_state = 4 * nmode
    blocks = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,3),(2,0),(2,2),(3,1),(3,3)]
    rows, cols = [], []
    for fr, fc in blocks:
        for m in range(nmode):
            for mm in range(max(0, m - dm), min(nmode, m + dm + 1)):
                rows.append(fr * nmode + m); cols.append(fc * nmode + mm)
    rows = np.asarray(rows); cols = np.asarray(cols)
    q = np.repeat(np.arange(nq), rows.size)
    r = np.tile(rows, nq); c = np.tile(cols, nq)
    data = rng.standard_normal(q.size) + 1j * rng.standard_normal(q.size)
    if holes:                       # exact zeros -> pattern varies per point
        data[rng.random(data.size) < holes] = 0.0
    if ragged:                      # drop entries so patterns differ per point
        keep = rng.random(q.size) > 0.15
        q, r, c, data = q[keep], r[keep], c[keep], data[keep]
    return coo_array((data, (q, r, c)), shape=(nq, n_state, n_state))


def compare(spans, nmode, dm, *, order=3, quad_order=6, holes=0.0, ragged=False,
            derivative=True, label=""):
    high = BSplineFE(knots=np.linspace(0, 1, spans + 1), order=order,
                     domain=(0.0, 1.0), quad_order=quad_order)
    mixed = _build_mixed_radial_data(high)
    rng = np.random.default_rng(spans * 1000 + nmode)
    M = make_tensor(mixed.weights_q.size, nmode, dm, rng, holes=holes, ragged=ragged)
    sb = SimpleNamespace(m_range=np.arange(nmode), dpsi_hat_dx=1.3)
    args = (mixed.phi_high_q, mixed.phi_low_q, mixed.weights_q, mixed.G_low_high)

    t0 = time.perf_counter()
    new = _project_local_tensor_mixed(sb, M, *args, include_derivative=derivative)
    t1 = time.perf_counter()
    old = baseline._project_local_tensor_mixed(sb, M, *args, include_derivative=derivative)
    t2 = time.perf_counter()

    same_pattern = (new.shape == old.shape and new.nnz == old.nnz
                    and np.array_equal(new.indptr, old.indptr)
                    and np.array_equal(new.indices, old.indices))
    diff = abs(new - old)
    maxdiff = float(diff.max()) if diff.nnz else 0.0
    scale = float(abs(old).max()) if old.nnz else 1.0
    rel = maxdiff / scale if scale else 0.0
    ok = same_pattern and rel <= 1e-14
    print(f"{'PASS' if ok else 'FAIL'}  {label:32s} nnz={old.nnz:>9,}  "
          f"pattern={'same' if same_pattern else 'DIFFERENT'}  "
          f"rel={rel:.2e}  old={t2-t1:7.2f}s new={t1-t0:6.2f}s  "
          f"speedup={(t2-t1)/(t1-t0):6.1f}x")
    return ok


cases = [
    dict(spans=12,  nmode=6,  dm=2, label="tiny"),
    dict(spans=12,  nmode=6,  dm=2, derivative=False, label="tiny, no derivative"),
    dict(spans=20,  nmode=12, dm=5, label="small"),
    dict(spans=20,  nmode=12, dm=5, holes=0.30, label="30% exact zeros"),
    dict(spans=20,  nmode=12, dm=5, ragged=True, label="ragged per-point pattern"),
    dict(spans=20,  nmode=12, dm=5, order=3, quad_order=4, label="order 3, quad 4"),
    dict(spans=20,  nmode=12, dm=5, order=5, quad_order=9, label="order 5, quad 9"),
    dict(spans=20,  nmode=12, dm=5, order=4, quad_order=8, label="order 4, quad 8"),
    dict(spans=20,  nmode=12, dm=0, label="diagonal only (dm=0)"),
    dict(spans=50,  nmode=24, dm=5, label="medium"),
    dict(spans=100, nmode=30, dm=5, label="large"),
]
results = [compare(**c) for c in cases]
print()
print(f"{sum(results)}/{len(results)} cases pass")
sys.exit(0 if all(results) else 1)
