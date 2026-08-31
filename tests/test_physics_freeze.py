"""Byte-level freeze on the numba kernels that carry the discretised physics.

These kernels encode the MHD operators themselves.  The consolidation work is
allowed to *relocate* them -- ``git mv`` plus an edit to the import block --
but never to rewrite them.  Making that a test rather than a convention means
a rename cannot quietly change a loop bound.

Discovery is deliberately done at runtime rather than by walking the AST for
``@njit`` decorators: three of the continuum kernels are wrapped with call
syntax (``fill_block_kernel = njit(...)(_fill_block_kernel)``) and a decorator
scan misses them.  Keying the table on the bare qualname rather than on
``module:qualname`` is also deliberate -- moving a kernel to a new module must
not require touching this file, while editing its body must.

To retire a kernel, delete its row in the same commit that deletes the code.
To add one, add its row with a hash you have reviewed.  Never regenerate the
whole table to make a failure go away.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import pkgutil

import pytest

import supernova


# Optional third-party backends.  A missing DESC install must not silently
# hide a kernel, so failures here are reported rather than ignored.
OPTIONAL_MODULE_PREFIXES = (
    "supernova.core.desc_helper",
    "supernova.core.desc_loader",
)

# qualname -> sha256 of inspect.getsource(fn.py_func)
# Captured from superNOVA_2 at f8b246e; 22 kernels, 1019 lines.
FROZEN_KERNELS: dict[str, str] = {
    # supernova/core/_solver_kernel.py
    "_build_static_blocks": "5b4ddfdd6abe84810a973ca0567563f7c70402af850938ae803394f00e943c47",
    "scale_rows": "eb1c5e10615ce12acce394dd1e99a9317c1ca53d4d1d35707a840748d0bb1dc3",
    "scale_cols": "38c7ce14be7974d29dd1f8171806ad23c2736e01a6b0325b9e0d8dba6373ca7a",
    "apply_dm_cutoff": "6299e18603420c0f7bc1b50c0311e9ceec21ff860efb322af74f4cbbc228bae3",
    "_apply_E_inv": "05d2e426e6cdf541e27d4d65214a303ae0306402fd81e87aca9d6a4e2aa0fdec",
    "_apply_Eprime_inv": "0ffa0d5dae381e5408daee3edde71bd65925be8efa84f6afbb120249a0ad8f47",
    # supernova/core/_continuum_kernel.py
    "_fill_block_kernel": "c8a9d2f49d1891632dce9336987b7b506a3bb4fd2ff69b7cda35f864d7bec4a8",
    "_fill_convolution_block_kernel": "5f64ebd3159da3e779f982f1decee403d67bf0f927de533fed7f22680f83f622",
    "_apply_ghg_scaling": "516525ccacebb0f35741c65d7997542987390b0300f8853807f0c08ae2ebd9bd",
    "assemble_full_matrices": "9650d4a9dd1894073562c1f4b1851da063a4e89176ee65dfe9d746d89d98bddd",
    "assembly_slow_sound": "be8dd3a796eece31a3331d86269e6c8983e6f8684238220e637f09e820c37b90",
    "get_active_bounds": "896b54c7c7959ba5fea71bdd75dfc0af1878559c10c7b92aaecdb885a2176dbb",
    "get_harmonic_value": "85cbbfaa3ca5bc71fd4a8dbbe2a4f8b34297e3af64d071d42e63d0c11c6e10d4",
    # supernova/core/_solver_kernel2.py
    "assemble_continuum": "2f22578864457e90d6f7a337264e323b2a0a177ee6b90957516fdcc812fb7da1",
    # supernova/core/_classify_mode.py
    "_quadratic_forms_real": "5d249dc4e6461638941f23ee50dfb041c794a489706b4318673865cc4c70ca2f",
    # supernova/radial/_harmonics_kernels.py
    "_decompose2d_impl": "878a16900acbcca952514e6ade7c54d84049d091e3813474565e8aede5c90580",
    "_decompose3d_impl": "ad14cba8106f1a14223bafa973eaec0cd40af982e05d5a0736ca332390b48a32",
    "recompose2d": "e670106c6abd5ade7963f9ab019cfefd0334524b39bd6fdb6952720f829ffddf",
    "recompose3d": "199df291a1ff5bcc08a7a6ec802de92399a93f147803aabbc18f45aca5912b1b",
    "_solve_trigonometric_gram": "ae91e41d86904c0946fb02cc53302f20fe5ffdcc2462024c028ab59957af789c",
    "_periodic_weights": "ab457b72a7827757dc4650b81362403ba5dc1f6e4d79314e9438792ef09cce0f",
    "_periodic_integral_1d": "d52a63b8e25750ec8fab0942111199b672a36e557908f814a9333636ce4d3657",
}


def _discover_kernels() -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Return ``{qualname: (module, sha256)}`` and the list of skipped modules."""
    kernels: dict[str, tuple[str, str]] = {}
    skipped: list[str] = []
    for info in pkgutil.walk_packages(supernova.__path__, supernova.__name__ + "."):
        try:
            module = importlib.import_module(info.name)
        except ImportError:
            skipped.append(info.name)
            continue
        for value in vars(module).values():
            py_func = getattr(value, "py_func", None)
            if py_func is None or not callable(py_func):
                continue
            if getattr(py_func, "__module__", None) != info.name:
                continue  # imported from elsewhere; hashed at its definition site
            qualname = py_func.__qualname__
            digest = hashlib.sha256(inspect.getsource(py_func).encode()).hexdigest()
            previous = kernels.get(qualname)
            if previous is not None and previous[1] != digest:
                raise AssertionError(
                    f"Two different kernels share the qualname {qualname!r}: "
                    f"{previous[0]} and {info.name}. The freeze table keys on "
                    "the bare qualname, so one of them must be renamed."
                )
            kernels[qualname] = (info.name, digest)
    return kernels, skipped


@pytest.fixture(scope="module")
def discovered():
    kernels, skipped = _discover_kernels()
    unexpected = [
        name
        for name in skipped
        if not name.startswith(OPTIONAL_MODULE_PREFIXES)
    ]
    assert not unexpected, (
        f"Modules failed to import, so their kernels were not checked: {unexpected}"
    )
    return kernels


def test_numba_is_active():
    """The freeze is meaningless if ``njit`` degraded to the no-op fallback."""
    from supernova.core._continuum_kernel import get_active_bounds

    assert hasattr(get_active_bounds, "py_func"), (
        "numba is not installed, so no kernel exposes .py_func and the freeze "
        "silently passes. Install numba before trusting this test."
    )


def test_no_kernel_added_or_removed(discovered):
    """Exact set equality: additions and deletions are both explicit edits."""
    found = set(discovered)
    frozen = set(FROZEN_KERNELS)
    added = sorted(found - frozen)
    removed = sorted(frozen - found)
    assert not added, (
        f"New numba kernels are not covered by the freeze: {added}. "
        "Add each one to FROZEN_KERNELS with a reviewed hash."
    )
    assert not removed, (
        f"Frozen kernels have disappeared: {removed}. If the removal is "
        "intended, delete the rows in the same commit that deletes the code."
    )


@pytest.mark.parametrize("qualname", sorted(FROZEN_KERNELS))
def test_kernel_source_is_unchanged(discovered, qualname):
    """A kernel body must be byte-identical to the superNOVA_2 baseline."""
    assert qualname in discovered, f"{qualname} was not found in the package."
    module, digest = discovered[qualname]
    assert digest == FROZEN_KERNELS[qualname], (
        f"{qualname} in {module} no longer matches its frozen source.\n"
        "These kernels carry the discretised MHD operators and may be moved "
        "but not rewritten. If the change is deliberate physics work, update "
        "the hash in a commit that does nothing else."
    )


if __name__ == "__main__":  # pragma: no cover - convenience for a quick check
    found, skipped = _discover_kernels()
    bad = [q for q, (_, d) in found.items() if FROZEN_KERNELS.get(q) != d]
    missing = sorted(set(FROZEN_KERNELS) - set(found))
    extra = sorted(set(found) - set(FROZEN_KERNELS))
    print(f"discovered {len(found)} kernels; skipped modules: {skipped}")
    print(f"mismatched: {bad or 'none'}")
    print(f"missing:    {missing or 'none'}")
    print(f"unfrozen:   {extra or 'none'}")
    raise SystemExit(1 if (bad or missing or extra) else 0)
