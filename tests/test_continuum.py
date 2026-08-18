import numpy as np
import pytest

from supernova.core.continuum import _eigh_signed_definite


def test_eigh_signed_definite_preserves_common_sign_reversal():
    A = np.array(
        [[3.0, 0.25 - 0.1j], [0.25 + 0.1j, 1.0]],
        dtype=np.complex128,
    )
    B = np.array(
        [[2.0, 0.1j], [-0.1j, 1.5]],
        dtype=np.complex128,
    )

    positive = _eigh_signed_definite(A, B)
    negative = _eigh_signed_definite(-A, -B)

    np.testing.assert_allclose(negative, positive, rtol=1.0e-13, atol=1.0e-13)


def test_eigh_signed_definite_rejects_indefinite_mass_matrix():
    A = np.diag([1.0, 2.0])
    B = np.diag([1.0, -1.0])

    with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
        _eigh_signed_definite(A, B)
