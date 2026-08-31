"""Focused tests for solver5 polynomial mapping and exact-zero EPS dispatch."""

from types import SimpleNamespace

import numpy as np
import pytest
import unyt
import xarray as xr
from scipy.sparse import csr_matrix

import supernova
import supernova.core.solver2 as solver2_module
import supernova.core.solver5 as solver5_module
import no_sync.solver5_itpa_fitted_eqdsk.solver5_plots as solver5_plots_module
from supernova.core.solver5 import (
    _linear_residuals,
    _polynomial_backward_error,
    _polynomial_residual_norm,
    _validate_gamma_zero_pencil,
    superSolver5,
)
from supernova.core.solver4 import MixedFieldLayout
from supernova.operators.blocks import _build_static_blocks
from supernova.equilibrium import (
    _active_poloidal_flux_span,
    _canonicalize_boozer_flux_profiles,
    _validate_internal_equilibrium_convention,
    _orient_realspace_boozer_components,
    _outward_boozer_interpolation_grid,
    _reflect_boozer_theta,
)
from no_sync.solver5_itpa_fitted_eqdsk.solver5_plots import (
    check_xipsi_mirror_symmetry,
)


def test_static_f12_matches_nova_left_right_derivative_order_with_sine_terms():
    """Regress Eq. (29) for G=i(nq-m) and one-sided amplitudes."""
    npsi = 1
    m_range = np.array([10, 11], dtype=np.int32)
    ntor = 6
    q = 1.75
    harmonic_shape = (npsi, 2, 1)

    def zeros():
        return np.zeros(harmonic_shape, dtype=np.complex128)

    f12_shear = zeros()
    f12_current = zeros()
    f12_shear[0, :, 0] = np.array([0.8, 0.35 - 0.20j])
    f12_current[0, :, 0] = np.array([-0.3, -0.15 - 0.10j])

    scalar_one = np.ones(npsi, dtype=np.complex128)
    _, _, _, _, f_block, _, _ = _build_static_blocks(
        C11=zeros(),
        O=zeros(),
        C12_2=zeros(),
        C12_3=zeros(),
        C22=zeros(),
        D11=zeros(),
        D12=zeros(),
        kappa_s=zeros(),
        K=zeros(),
        M=zeros(),
        F11_2=zeros(),
        F12_1=f12_shear,
        F12_2=f12_current,
        F21=zeros(),
        F22=zeros(),
        hprof=scalar_one,
        Iprof=scalar_one,
        gprof=scalar_one,
        qprof=np.array([q], dtype=np.complex128),
        pressure=np.zeros(npsi, dtype=np.complex128),
        pressure_deriv=np.zeros(npsi, dtype=np.complex128),
        density=scalar_one,
        gamma_s=5.0 / 3.0,
        one_gpp=zeros(),
        gtp_over_gpp=zeros(),
        gpz_over_gpp=zeros(),
        ntor=ntor,
        m_range=m_range,
        dm_max=1,
        dn_max=0,
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)

    def real_profile(coefficients):
        return (
            coefficients[0].real
            + np.real(coefficients[1] * np.exp(-1j * theta))
        )

    shear_values = real_profile(f12_shear[0, :, 0])
    current_values = real_profile(f12_current[0, :, 0])
    g_diag = 1j * (ntor * q - m_range)
    expected = np.empty((m_range.size, m_range.size), dtype=np.complex128)
    for row, m_row in enumerate(m_range):
        for col, m_col in enumerate(m_range):
            phase = np.exp(1j * (m_row - m_col) * theta)
            expected[row, col] = np.mean(
                phase
                * (
                    g_diag[row] * shear_values
                    + current_values * g_diag[col]
                )
            )

    nmode = m_range.size
    np.testing.assert_allclose(
        f_block[0, :nmode, nmode:],
        expected,
        atol=2.0e-13,
    )


def test_polynomial_residual_uses_tminus1_t0_t1_order():
    Tminus1 = csr_matrix(np.diag([2.0, 3.0]))
    T0 = csr_matrix(np.diag([-3.0, 1.0]))
    T1 = csr_matrix(np.diag([1.0, -1.0]))
    vector = np.array([1.0, 0.0])

    absolute, relative = _polynomial_residual_norm(
        Tminus1, T0, T1, 1.0, vector
    )

    assert absolute == pytest.approx(0.0)
    assert relative == pytest.approx(0.0)


def test_polynomial_backward_error_is_operator_scaled():
    A0 = csr_matrix(np.eye(2) * 2.0)
    A1 = csr_matrix(np.eye(2) * 3.0)
    A2 = csr_matrix(np.eye(2) * 4.0)
    eigenvalue = 0.5
    vector = np.array([3.0, 4.0])

    backward = _polynomial_backward_error(
        A0,
        A1,
        A2,
        eigenvalue,
        vector,
    )

    residual_scale = 2.0 + 3.0 * eigenvalue + 4.0 * eigenvalue**2
    matrix_scale = (
        np.linalg.norm(A0.data)
        + eigenvalue * np.linalg.norm(A1.data)
        + eigenvalue**2 * np.linalg.norm(A2.data)
    )
    expected = residual_scale * np.linalg.norm(vector) / (
        matrix_scale * np.linalg.norm(vector)
    )
    assert backward == pytest.approx(expected)


def test_gamma_zero_pencil_requires_exact_zero_inverse_coefficient():
    T0 = csr_matrix(np.diag([-1.0, -2.0]))
    T1 = csr_matrix(np.eye(2))
    stored_zero = csr_matrix(
        (np.array([0.0]), (np.array([0]), np.array([0]))),
        shape=(2, 2),
    )

    actual_T0, actual_T1 = _validate_gamma_zero_pencil(
        stored_zero,
        T0,
        T1,
        gamma_s=0.0,
    )

    assert actual_T0.nnz == 2
    assert actual_T1.nnz == 2
    with pytest.raises(ValueError, match="exact gamma_s"):
        _validate_gamma_zero_pencil(
            csr_matrix((2, 2)),
            T0,
            T1,
            gamma_s=1.0e-15,
        )
    with pytest.raises(ValueError, match="Tminus1 to be exactly zero"):
        _validate_gamma_zero_pencil(
            csr_matrix(np.diag([1.0e-30, 0.0])),
            T0,
            T1,
            gamma_s=0.0,
        )


def test_linear_residual_uses_native_generalized_pencil():
    T0 = csr_matrix(np.diag([-1.0, -2.0]))
    T1 = csr_matrix(np.eye(2))

    absolute, relative, backward = _linear_residuals(
        T0,
        T1,
        2.0,
        np.array([0.0, 1.0]),
    )

    assert absolute == 0.0
    assert relative == 0.0
    assert backward == 0.0


def test_build_pep_maps_split_matrices_to_polynomial_coefficients(monkeypatch):
    solver = object.__new__(superSolver5)
    solver.gamma_s = 5.0 / 3.0
    split = tuple(csr_matrix(np.eye(2) * scale) for scale in (2.0, 3.0, 5.0))
    solver.assemble_split_matrices = lambda **kwargs: split
    captured = {}

    def fake_pep(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(solver5_module, "QuadraticPEPSolver", fake_pep)
    solver.build_pep_solver(
        ntor=4,
        target_omega=2.0,
        region={"center": 4.0, "radius": 0.5, "vscale": 0.25},
    )

    np.testing.assert_allclose(captured["A0"].toarray(), split[2].toarray())
    np.testing.assert_allclose(captured["A1"].toarray(), split[0].toarray())
    np.testing.assert_allclose(captured["A2"].toarray(), split[1].toarray())
    assert captured["target_s"] == 4.0
    assert captured["st_type"] == "SINVERT"


def test_build_pep_forwards_explicit_signed_harmonics(monkeypatch):
    solver = object.__new__(superSolver5)
    solver.gamma_s = 5.0 / 3.0
    split = tuple(csr_matrix(np.eye(2) * scale) for scale in (2.0, 3.0, 5.0))
    assembled = {}
    solver.assemble_split_matrices = lambda **kwargs: (
        assembled.update(kwargs) or split
    )
    monkeypatch.setattr(
        solver5_module,
        "QuadraticPEPSolver",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    m_values = np.array([-2, -1, 1, 2], dtype=np.int64)

    solver.build_pep_solver(
        ntor=6,
        target_omega=2.0,
        region={"center": 4.0, "radius": 0.5, "vscale": 0.25},
        m_values=m_values,
    )

    np.testing.assert_array_equal(assembled["m_values"], m_values)
    assert assembled["mmax"] is None
    with pytest.raises(ValueError, match="either mmax or m_values"):
        solver.build_pep_solver(
            ntor=6,
            target_omega=2.0,
            region={"center": 4.0, "radius": 0.5, "vscale": 0.25},
            mmax=2,
            m_values=m_values,
        )


def test_build_pep_transparently_selects_eps_at_exact_gamma_zero(monkeypatch):
    solver = object.__new__(superSolver5)
    solver.gamma_s = 0.0
    T0 = csr_matrix(np.diag([-1.0, -2.0]))
    T1 = csr_matrix(np.eye(2))
    Tminus1 = csr_matrix((2, 2), dtype=np.complex128)
    solver.assemble_split_matrices = lambda **kwargs: (T0, T1, Tminus1)
    captured = {}

    def fake_eps(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(backend="EPS")

    monkeypatch.setattr(solver5_module, "LinearGeneralizedEPSSolver", fake_eps)
    monkeypatch.setattr(
        solver5_module,
        "QuadraticPEPSolver",
        lambda **kwargs: pytest.fail("gamma_s=0 unexpectedly selected PEP"),
    )

    built = solver.build_pep_solver(
        ntor=4,
        target_omega=2.0,
        region={"center": 4.0, "radius": 0.5, "vscale": 0.25},
    )

    assert built.backend == "EPS"
    np.testing.assert_allclose(captured["T0"].toarray(), T0.toarray())
    np.testing.assert_allclose(captured["T1"].toarray(), T1.toarray())
    assert "pep_type" not in captured
    assert captured["target_s"] == 4.0
    assert captured["st_type"] == "SINVERT"


def test_build_pep_keeps_small_nonzero_gamma_on_quadratic_path(monkeypatch):
    solver = object.__new__(superSolver5)
    solver.gamma_s = 1.0e-15
    split = tuple(csr_matrix(np.eye(2) * scale) for scale in (2.0, 3.0, 5.0))
    solver.assemble_split_matrices = lambda **kwargs: split
    captured = {}

    monkeypatch.setattr(
        solver5_module,
        "QuadraticPEPSolver",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(backend="PEP"),
    )
    monkeypatch.setattr(
        solver5_module,
        "LinearGeneralizedEPSSolver",
        lambda **kwargs: pytest.fail("nonzero gamma_s unexpectedly selected EPS"),
    )

    built = solver.build_pep_solver(
        ntor=4,
        target_omega=2.0,
        region={"center": 4.0, "radius": 0.5, "vscale": 0.25},
    )

    assert built.backend == "PEP"
    np.testing.assert_allclose(captured["A0"].toarray(), split[2].toarray())


@pytest.mark.parametrize(
    ("coefficient", "bad_value"),
    [
        ("A0=Tminus1", np.nan),
        ("A1=T0", np.inf),
        ("A2=T1", complex(0.0, -np.inf)),
    ],
)
def test_quadratic_pep_rejects_nonfinite_coefficients_before_petsc(
    monkeypatch,
    coefficient,
    bad_value,
):
    coefficient_index = ("A0=Tminus1", "A1=T0", "A2=T1").index(coefficient)
    arrays = [np.eye(2, dtype=np.complex128) for _ in range(3)]
    arrays[coefficient_index][1, 0] = bad_value
    matrices = [csr_matrix(array) for array in arrays]

    def fail_if_petsc_is_requested():
        pytest.fail("PETSc was requested before coefficient validation")

    monkeypatch.setattr(
        solver5_module,
        "_require_petsc",
        fail_if_petsc_is_requested,
    )

    with pytest.raises(ValueError, match=rf"{coefficient} contains 1 non-finite"):
        solver5_module.QuadraticPEPSolver(
            A0=matrices[0],
            A1=matrices[1],
            A2=matrices[2],
            target_s=1.0,
            region={"center": 1.0, "radius": 0.1},
        )


def test_quadratic_pep_retarget_resets_and_reattaches_existing_operators(
    monkeypatch,
):
    events = []

    class FakeST:
        def setType(self, kind):
            events.append(("st.setType", kind))

    class FakeRG:
        def setEllipseParameters(self, center, radius, vscale):
            events.append(
                ("rg.setEllipseParameters", center, radius, vscale)
            )

    class FakePEP:
        st = FakeST()

        def reset(self):
            events.append(("pep.reset",))

        def setOperators(self, matrices):
            events.append(("pep.setOperators", matrices))

        def setRG(self, region):
            events.append(("pep.setRG", region))

        def setTarget(self, target):
            events.append(("pep.setTarget", target))

        def getST(self):
            return self.st

    wrapper = object.__new__(solver5_module.QuadraticPEPSolver)
    wrapper.pep = FakePEP()
    wrapper.rg = FakeRG()
    wrapper.mats = [object(), object(), object()]
    wrapper.target_s = 1.0
    wrapper.region = {"type": "ellipse", "center": 1.0, "radius": 0.1}
    wrapper.st_type = "SINVERT"
    wrapper.verbose = False
    wrapper.last_residual_norms = np.array([1.0])
    wrapper.last_absolute_residual_norms = np.array([2.0])
    wrapper.last_backward_errors = np.array([3.0])
    wrapper.last_slepc_error_estimates = np.array([4.0])
    wrapper.last_slepc_backward_errors = np.array([5.0])
    wrapper._configure_mixed_shift_solver = lambda: events.append(("configure",))
    monkeypatch.setattr(
        solver5_module,
        "_require_slepc",
        lambda: SimpleNamespace(
            ST=SimpleNamespace(Type=SimpleNamespace(SINVERT="sinvert"))
        ),
    )

    wrapper.retarget(
        9.0,
        {"center": 9.0, "radius": 0.75, "vscale": 0.2},
    )

    assert events == [
        ("pep.reset",),
        ("pep.setOperators", wrapper.mats),
        ("rg.setEllipseParameters", 9.0, 0.75, 0.2),
        ("pep.setRG", wrapper.rg),
        ("pep.setTarget", 9.0),
        ("st.setType", "sinvert"),
        ("configure",),
    ]
    assert wrapper.target_s == 9.0
    assert wrapper.region == {
        "type": "ellipse",
        "center": 9.0,
        "radius": 0.75,
        "vscale": 0.2,
    }
    assert wrapper.last_residual_norms is None
    assert wrapper.last_absolute_residual_norms is None
    assert wrapper.last_backward_errors is None
    assert wrapper.last_slepc_error_estimates is None
    assert wrapper.last_slepc_backward_errors is None


def test_quadratic_pep_retarget_validates_before_reset():
    wrapper = object.__new__(solver5_module.QuadraticPEPSolver)
    wrapper.pep = SimpleNamespace(reset=lambda: pytest.fail("unexpected reset"))
    wrapper.rg = object()
    wrapper.mats = [object(), object(), object()]
    wrapper.target_s = 1.0
    wrapper.region = {"type": "ellipse", "center": 1.0, "radius": 0.1}

    with pytest.raises(ValueError, match="radius must be positive"):
        wrapper.retarget(4.0, {"center": 4.0, "radius": 0.0})

    assert wrapper.target_s == 1.0
    assert wrapper.region == {
        "type": "ellipse",
        "center": 1.0,
        "radius": 0.1,
    }


def test_solve_pep_preserves_primary_error_when_destroy_also_fails():
    class FailingPEP:
        destroy_calls = 0

        def solve(self, max_eigenpairs=None):
            raise ValueError("primary PEPSolve failure")

        def destroy(self):
            self.destroy_calls += 1
            raise RuntimeError("secondary PEPDestroy failure")

    solver = object.__new__(superSolver5)
    pep = FailingPEP()
    solver.build_pep_solver = lambda **kwargs: pep

    with pytest.raises(ValueError, match="primary PEPSolve failure") as excinfo:
        solver.solve_pep(
            ntor=4,
            target_omega=1.0,
            region={"center": 1.0, "radius": 0.1},
        )

    assert pep.destroy_calls == 1
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == "secondary PEPDestroy failure"


def test_solve_pep_destroys_pep_after_success(monkeypatch):
    expected = xr.Dataset({"eigenvalue": ("eigenpair", [1.0 + 0.0j])})

    class SuccessfulPEP:
        destroy_calls = 0

        def solve(self, max_eigenpairs=None):
            return SimpleNamespace()

        def destroy(self):
            self.destroy_calls += 1

    solver = object.__new__(superSolver5)
    pep = SuccessfulPEP()
    solver.build_pep_solver = lambda **kwargs: pep
    monkeypatch.setattr(solver, "_split_result_dataset", lambda **kwargs: expected)

    actual = solver.solve_pep(
        ntor=4,
        target_omega=1.0,
        region={"center": 1.0, "radius": 0.1},
    )

    assert actual is expected
    assert pep.destroy_calls == 1


def test_solve_pep_packs_exact_zero_eps_metadata_without_api_change(monkeypatch):
    expected = xr.Dataset({"eigenvalue": ("eigenpair", [1.0 + 0.0j])})

    class SuccessfulEPS:
        backend = "EPS"
        eps_type = "KRYLOVSCHUR"
        destroy_calls = 0

        def solve(self, max_eigenpairs=None):
            return SimpleNamespace()

        def destroy(self):
            self.destroy_calls += 1

    solver = object.__new__(superSolver5)
    eps = SuccessfulEPS()
    solver.build_pep_solver = lambda **kwargs: eps
    packed = {}
    monkeypatch.setattr(
        solver,
        "_split_result_dataset",
        lambda **kwargs: packed.update(kwargs) or expected,
    )

    actual = solver.solve_pep(
        ntor=4,
        target_omega=1.0,
        region={"center": 1.0, "radius": 0.1},
    )

    assert actual is expected
    assert eps.destroy_calls == 1
    assert packed["nep_type"] == "EPS"
    assert packed["extra_attrs"]["problem_type"] == "GENERALIZED_LINEAR"
    assert packed["extra_attrs"]["gamma_zero_reduction"] is True
    assert packed["extra_attrs"]["linear_coefficients"] == "[-T0, T1]"


def test_linear_eps_extracts_known_roots_and_native_residuals(monkeypatch):
    class FakeVector:
        def __init__(self):
            self.values = np.zeros(2, dtype=np.complex128)

        def destroy(self):
            pass

    class FakeMatrix:
        @staticmethod
        def createVecRight():
            return FakeVector()

    class FakeEPS:
        roots = (1.0, 2.0)

        def solve(self):
            pass

        def getConverged(self):
            return 2

        def getConvergedReason(self):
            return "converged"

        def getIterationNumber(self):
            return 3

        def getEigenpair(self, index, vector, imaginary):
            vector.values[:] = 0.0
            vector.values[index] = 1.0
            return self.roots[index]

        def getErrorEstimate(self, index):
            return 1.0e-13

        def computeError(self, index, error_type):
            return 2.0e-13

        @staticmethod
        def getST():
            return SimpleNamespace(getType=lambda: "sinvert")

    wrapper = object.__new__(solver5_module.LinearGeneralizedEPSSolver)
    wrapper.T0 = csr_matrix(np.diag([-1.0, -2.0]))
    wrapper.T1 = csr_matrix(np.eye(2))
    wrapper.eps = FakeEPS()
    wrapper.mats = [FakeMatrix(), FakeMatrix()]
    wrapper.size = 2
    wrapper.scalar_type = np.dtype(np.complex128)
    wrapper.compute_residuals = True
    wrapper.target_s = 1.5
    wrapper.verbose = False
    monkeypatch.setattr(
        solver5_module,
        "_vec_read_array",
        lambda vector: vector.values,
    )
    monkeypatch.setattr(
        solver5_module,
        "_require_slepc",
        lambda: SimpleNamespace(
            EPS=SimpleNamespace(ErrorType=SimpleNamespace(BACKWARD="backward"))
        ),
    )

    result = wrapper.solve()

    np.testing.assert_allclose(result.eigenvalues, [1.0, 2.0])
    np.testing.assert_allclose(wrapper.last_absolute_residual_norms, 0.0)
    np.testing.assert_allclose(wrapper.last_residual_norms, 0.0)
    np.testing.assert_allclose(wrapper.last_backward_errors, 0.0)
    np.testing.assert_allclose(wrapper.last_slepc_error_estimates, 1.0e-13)
    np.testing.assert_allclose(wrapper.last_slepc_backward_errors, 2.0e-13)


def test_quadratic_pep_allow_empty_supports_empty_frequency_windows(monkeypatch):
    class FakeVector:
        def destroy(self):
            pass

    class FakeMatrix:
        def createVecRight(self):
            return FakeVector()

    class FakePEP:
        def solve(self):
            pass

        def getConverged(self):
            return 0

        def getConvergedReason(self):
            return "empty"

        def getIterationNumber(self):
            return 2

        def getST(self):
            return SimpleNamespace(getType=lambda: "sinvert")

    wrapper = object.__new__(solver5_module.QuadraticPEPSolver)
    wrapper.pep = FakePEP()
    wrapper.mats = [FakeMatrix()]
    wrapper.size = 3
    wrapper.scalar_type = np.dtype(np.complex128)
    wrapper.compute_residuals = True
    wrapper.target_s = 1.0
    wrapper.verbose = False
    monkeypatch.setattr(solver5_module, "_require_petsc", lambda: SimpleNamespace())
    monkeypatch.setattr(solver5_module, "_require_slepc", lambda: SimpleNamespace())

    result = wrapper.solve(allow_empty=True)

    assert result.nconv == 0
    assert result.eigenvalues.shape == (0,)
    assert result.eigenvectors.shape == (0, 3)
    assert wrapper.last_backward_errors.shape == (0,)
    assert wrapper.last_slepc_backward_errors.shape == (0,)

    with pytest.raises(RuntimeError, match="converged no eigenpairs"):
        wrapper.solve()


def test_linear_eps_allow_empty_supports_empty_frequency_windows(monkeypatch):
    class FakeVector:
        def destroy(self):
            pass

    class FakeMatrix:
        def createVecRight(self):
            return FakeVector()

    class FakeEPS:
        def solve(self):
            pass

        def getConverged(self):
            return 0

        def getConvergedReason(self):
            return -1

        def getIterationNumber(self):
            return 40

        def getST(self):
            return SimpleNamespace(getType=lambda: "sinvert")

    wrapper = object.__new__(solver5_module.LinearGeneralizedEPSSolver)
    wrapper.eps = FakeEPS()
    wrapper.mats = [FakeMatrix(), FakeMatrix()]
    wrapper.size = 3
    wrapper.scalar_type = np.dtype(np.complex128)
    wrapper.compute_residuals = True
    wrapper.target_s = 1.0
    wrapper.verbose = False
    monkeypatch.setattr(solver5_module, "_require_slepc", lambda: SimpleNamespace())

    result = wrapper.solve(allow_empty=True)

    assert result.nconv == 0
    assert result.reason == -1
    assert result.iterations == 40
    assert result.eigenvalues.shape == (0,)
    assert result.eigenvectors.shape == (0, 3)
    assert wrapper.last_backward_errors.shape == (0,)
    assert wrapper.last_slepc_backward_errors.shape == (0,)

    with pytest.raises(RuntimeError, match="converged no eigenpairs"):
        wrapper.solve()


def test_solve_pep_forwards_signed_harmonics_to_result_packing(monkeypatch):
    expected = xr.Dataset({"eigenvalue": ("eigenpair", [1.0 + 0.0j])})

    class SuccessfulPEP:
        last_residual_norms = None
        last_absolute_residual_norms = None

        def solve(self, max_eigenpairs=None):
            return SimpleNamespace()

        def destroy(self):
            pass

    solver = object.__new__(superSolver5)
    build_kwargs = {}
    packing_kwargs = {}
    solver.build_pep_solver = lambda **kwargs: (
        build_kwargs.update(kwargs) or SuccessfulPEP()
    )
    monkeypatch.setattr(
        solver,
        "_split_result_dataset",
        lambda **kwargs: packing_kwargs.update(kwargs) or expected,
    )
    m_values = np.array([-2, -1, 1, 2], dtype=np.int64)

    actual = solver.solve_pep(
        ntor=6,
        target_omega=1.0,
        region={"center": 1.0, "radius": 0.1},
        m_values=m_values,
    )

    assert actual is expected
    np.testing.assert_array_equal(build_kwargs["m_values"], m_values)
    np.testing.assert_array_equal(packing_kwargs["m_values"], m_values)


def test_split_result_dataset_packs_explicit_signed_harmonic_coordinates():
    solver = object.__new__(superSolver5)
    solver.M = SimpleNamespace(m_pol=np.arange(-2, 3, dtype=np.int64))
    solver._species_mass_amu = lambda: 2.0
    m_values = np.array([-2, -1, 1, 2], dtype=np.int64)
    n_modes = m_values.size
    solver._mixed_layout = lambda sb: MixedFieldLayout(
        n_high=2,
        n_low=1,
        n_modes=np.asarray(sb.m_range).size,
    )
    n_trial = 2 * n_modes * (2 + 1)
    result = SimpleNamespace(
        eigenvalues=np.array([1.5 + 0.0j]),
        eigenvectors=np.arange(n_trial, dtype=np.complex128)[None, :],
        nconv=1,
        reason=1,
        iterations=3,
    )
    nep = SimpleNamespace(
        last_residual_norms=np.array([1.0e-9]),
        last_absolute_residual_norms=np.array([2.0e-9]),
        last_backward_errors=np.array([3.0e-9]),
        last_slepc_error_estimates=np.array([4.0e-9]),
        last_slepc_backward_errors=np.array([5.0e-9]),
    )

    packed = solver._split_result_dataset(
        result=result,
        nep=nep,
        ntor=6,
        target_omega=np.sqrt(1.5),
        target_s=1.5,
        region={"type": "ellipse", "center": 1.5, "radius": 0.1, "vscale": 0.1},
        mmax=None,
        max_dm=2,
        regularizer=0.0,
        apply_bcs=False,
        bc_kind="none",
        bc_tol=None,
        verbose=False,
        compute_residuals=False,
        which="TARGET_MAGNITUDE",
        nep_type="PEP",
        description="signed test",
        m_values=m_values,
    )

    np.testing.assert_array_equal(packed.coords["m_pol"], m_values)
    assert packed.attrs["harmonic_selection"] == "explicit_diagnostic"
    assert packed.sizes["m_pol"] == n_modes
    assert packed["xi_psi_coefficient"].shape == (1, 2, n_modes)
    assert packed["residual_norm"].item() == pytest.approx(1.0e-9)
    assert packed["backward_error"].item() == pytest.approx(3.0e-9)
    assert packed["slepc_error_estimate"].item() == pytest.approx(4.0e-9)
    assert packed["slepc_backward_error"].item() == pytest.approx(5.0e-9)


class _MirrorTestBasis:
    def get_ndofs(self):
        return 2

    def get_mesh(self):
        return np.array([0.0, 1.0])

    def eval_basis(self, points):
        points = np.asarray(points, dtype=float)
        return np.vstack((1.0 - points, points))


def _mirror_test_result():
    m_values = np.array([-2, -1, 1, 2], dtype=np.int64)
    coefficients = np.zeros((2, 2, m_values.size), dtype=np.complex128)
    coefficients[1, :, 0] = np.array([0.2 + 0.1j, -0.1 + 0.05j])
    coefficients[1, :, 3] = coefficients[1, :, 0]
    coefficients[1, :, 1] = np.array([1.0 - 0.2j, 0.4 + 0.3j])
    coefficients[1, :, 2] = coefficients[1, :, 1]
    return xr.Dataset(
        {
            "s": ("eigenpair", np.array([1.0, 2.0], dtype=np.complex128)),
            "omega": (
                "eigenpair",
                np.sqrt(np.array([1.0, 2.0], dtype=np.complex128)),
            ),
            "xi_psi_coefficient": (
                ("eigenpair", "radial_dof_high", "m_pol"),
                coefficients,
            ),
        },
        coords={
            "eigenpair": np.arange(2),
            "radial_dof_high": np.arange(2),
            "m_pol": m_values,
        },
    )


def test_xipsi_mirror_check_selects_input_eigenvalue_and_compares_signed_pairs():
    result = _mirror_test_result()
    pair = SimpleNamespace(high=_MirrorTestBasis(), low=_MirrorTestBasis())
    check = check_xipsi_mirror_symmetry(
        result,
        SimpleNamespace(),
        2.01,
        mpol=(1, 2),
        pair=pair,
        n_plot=33,
        rtol=1.0e-12,
    )

    assert check.passed
    assert check.selected_eigenpair == 1
    assert check.selected_eigenvalue == pytest.approx(2.0)
    assert check.eigenvalue_distance == pytest.approx(0.01)
    assert check.relative_error == pytest.approx(0.0)
    assert all(item.passed for item in check.pair_results)
    assert "xi^psi_m = xi^psi_-m" in check.format()

    broken = result.copy(deep=True)
    broken["xi_psi_coefficient"].loc[
        dict(eigenpair=1, radial_dof_high=0, m_pol=-1)
    ] += 0.1
    failure = check_xipsi_mirror_symmetry(
        broken,
        SimpleNamespace(),
        2.0,
        mpol=(1, 2),
        pair=pair,
        n_plot=33,
        rtol=1.0e-12,
    )
    assert not failure.passed
    assert failure.pair_results[0].m == 1
    assert not failure.pair_results[0].passed


def test_xipsi_mirror_check_rejects_positive_only_production_result():
    result = _mirror_test_result().sel(m_pol=[1, 2])
    pair = SimpleNamespace(high=_MirrorTestBasis(), low=_MirrorTestBasis())

    with pytest.raises(ValueError, match="normal ITPA production result"):
        check_xipsi_mirror_symmetry(
            result,
            SimpleNamespace(),
            2.0,
            pair=pair,
        )


def test_solve_and_plot_xipsi_mirror_symmetry_builds_signed_itpa_diagnostic(
    monkeypatch,
):
    solve_call = {}
    plot_call = {}
    signed_result = object()
    fake_check = object()

    class FakeSolver:
        def solve_pep(self, **kwargs):
            solve_call.update(kwargs)
            return signed_result

    def fake_plot(result, solver, eigenvalue, **kwargs):
        plot_call.update(
            {
                "result": result,
                "solver": solver,
                "eigenvalue": eigenvalue,
                **kwargs,
            }
        )
        return "figure", "axes", fake_check

    monkeypatch.setattr(
        solver5_plots_module,
        "plot_xipsi_mirror_symmetry",
        fake_plot,
    )
    solver = FakeSolver()
    output = solver5_plots_module.solve_and_plot_xipsi_mirror_symmetry(
        solver,
        4.0,
        positive_m_values=np.array([7, 8, 9]),
        ntor=6,
        max_dm=2,
        search_radius=0.2,
        target_relative_offset=1.0e-5,
        pep_kwargs={"nev": 9},
        plot_kwargs={"mpol": (8, 9), "rtol": 1.0e-7},
    )

    assert output == (signed_result, "figure", "axes", fake_check)
    np.testing.assert_array_equal(
        solve_call["m_values"],
        np.array([-9, -8, -7, 7, 8, 9]),
    )
    assert "mmax" not in solve_call
    assert solve_call["ntor"] == 6
    assert solve_call["max_dm"] == 2
    assert solve_call["target_omega"] ** 2 == pytest.approx(4.0 * 1.00001)
    assert solve_call["region"]["center"] == pytest.approx(4.0)
    assert solve_call["region"]["radius"] == pytest.approx(0.2)
    assert solve_call["nev"] == 9
    assert plot_call["result"] is signed_result
    assert plot_call["eigenvalue"] == pytest.approx(4.0)
    assert plot_call["max_eigenvalue_distance"] == pytest.approx(0.2)
    assert plot_call["mpol"] == (8, 9)


@pytest.mark.parametrize(
    "bc_options",
    [
        {"apply_bcs": False, "bc_kind": "full"},
        {"apply_bcs": True, "bc_kind": "axis"},
        {"apply_bcs": True, "bc_kind": "none"},
    ],
)
def test_solver5_rejects_rectangular_pep_inputs_before_petsc(bc_options):
    solver = object.__new__(superSolver5)
    with pytest.raises(ValueError, match="require apply_bcs=True"):
        solver.build_pep_solver(
            ntor=2,
            target_omega=1.0,
            region={"center": 1.0, "radius": 0.1},
            **bc_options,
        )


def test_supernova_exports_solver5_without_replacing_solver4():
    assert supernova.superSolver5 is superSolver5
    assert supernova.superSolver4 is not supernova.superSolver5


def test_quadratic_pep_configures_mumps_before_internal_st_options():
    events = []

    class FakePC:
        kind = "none"
        factor_solver = None

        def setType(self, kind):
            events.append(("pc.setType", kind))
            self.kind = kind

        def getType(self):
            return self.kind

        def setFactorSolverType(self, kind):
            events.append(("pc.setFactorSolverType", kind))
            self.factor_solver = kind

        def getFactorSolverType(self):
            return self.factor_solver

    class FakeKSP:
        pc = FakePC()
        kind = "gmres"

        def setType(self, kind):
            events.append(("ksp.setType", kind))
            self.kind = kind

        def getPC(self):
            return self.pc

        def setFromOptions(self):
            events.append(("ksp.setFromOptions", None))

        def getOptionsPrefix(self):
            return "nova_quadratic_pep_st_"

        def getType(self):
            return self.kind

    ksp = FakeKSP()
    wrapper = object.__new__(solver5_module.QuadraticPEPSolver)
    wrapper.pep = SimpleNamespace(
        getST=lambda: SimpleNamespace(getKSP=lambda: ksp)
    )
    wrapper.verbose = False

    wrapper._configure_mixed_shift_solver()

    assert events == [
        ("ksp.setType", "preonly"),
        ("pc.setType", "lu"),
        ("pc.setFactorSolverType", "mumps"),
        ("ksp.setFromOptions", None),
    ]
    assert ksp.getType() == "preonly"
    assert ksp.pc.getType() == "lu"
    assert ksp.pc.getFactorSolverType() == "mumps"


def test_negative_q_profiles_are_oriented_without_losing_flux_identity():
    q = -np.array([1.5, 1.7, 1.9])
    F = np.array([3.0, 3.1, 3.2])
    I = -np.array([0.1, 0.15, 0.2])
    h = q * F + I

    h_nova, q_nova, F_nova, I_nova, orientation = (
        _canonicalize_boozer_flux_profiles(h, q, F, I)
    )

    assert orientation == -1
    np.testing.assert_allclose(q_nova, -q)
    np.testing.assert_allclose(F_nova, F)
    np.testing.assert_allclose(I_nova, -I)
    np.testing.assert_allclose(h_nova, -h)
    np.testing.assert_allclose(h_nova, q_nova * F_nova + I_nova)


def test_boozer_profiles_reject_an_inconsistent_flux_identity():
    q = np.array([1.5, 1.7])
    F = np.array([3.0, 3.1])
    I = np.array([0.1, 0.2])
    h = q * F + I

    with pytest.raises(ValueError, match=r"h = q\*F \+ I"):
        _canonicalize_boozer_flux_profiles(h + 0.5, q, F, I)


def test_boozer_orientation_rejects_a_q_sign_change():
    q = np.array([-1.0, 0.5])
    F = np.full(2, 3.0)
    I = np.array([-0.1, 0.1])
    h = q * F + I

    with pytest.raises(ValueError, match="one sign"):
        _canonicalize_boozer_flux_profiles(h, q, F, I)


def test_positive_q_gauge_preserves_a_negative_signed_jacobian_factor():
    q = np.array([1.2, 1.4, 1.6])
    F = -np.array([3.0, 3.1, 3.2])
    I = -np.array([0.1, 0.15, 0.2])
    h = q * F + I

    h_nova, q_nova, F_nova, I_nova, orientation = (
        _canonicalize_boozer_flux_profiles(h, q, F, I)
    )

    assert orientation == 1
    assert np.all(q_nova > 0.0)
    assert np.all(h_nova < 0.0)
    np.testing.assert_allclose(F_nova, F)
    np.testing.assert_allclose(I_nova, I)
    np.testing.assert_allclose(h_nova, q_nova * F_nova + I_nova)


def test_theta_reflection_reindexes_scalars_and_transforms_cross_metrics():
    theta = np.linspace(0.0, 2.0 * np.pi, 5)
    psi = np.array([0.25, 0.75])
    base = np.arange(psi.size * theta.size, dtype=float).reshape(
        psi.size, theta.size
    )
    dataset = xr.Dataset(
        {
            "J_B": (("psi", "theta"), base),
            "gpsi_psi": (("psi", "theta"), base + 10.0),
            "gpsi_theta": (("psi", "theta"), base + 20.0),
            "gtheta_zeta": (("psi", "theta"), base + 30.0),
            "q": (("psi",), np.array([1.5, 1.8])),
        },
        coords={"psi": psi, "theta": theta},
    )
    indices = np.array([0, 3, 2, 1, 4])

    reflected = _reflect_boozer_theta(dataset, -1)

    np.testing.assert_allclose(reflected.J_B, dataset.J_B.values[:, indices])
    np.testing.assert_allclose(
        reflected.gpsi_psi, dataset.gpsi_psi.values[:, indices]
    )
    np.testing.assert_allclose(
        reflected.gpsi_theta, -dataset.gpsi_theta.values[:, indices]
    )
    np.testing.assert_allclose(
        reflected.gtheta_zeta, -dataset.gtheta_zeta.values[:, indices]
    )
    np.testing.assert_allclose(reflected.q, dataset.q)
    np.testing.assert_allclose(reflected.theta, theta)
    np.testing.assert_allclose(dataset.J_B, base)


def test_active_poloidal_flux_span_is_offset_invariant():
    assert _active_poloidal_flux_span(np.array([2.0, 6.0])) == pytest.approx(4.0)
    assert _active_poloidal_flux_span(np.array([-8.0, -4.0])) == pytest.approx(4.0)


def test_solver5_normalizes_cocos1_flux_per_radian(monkeypatch):
    from supernova.cases.itpa import ITPAEq

    equ = ITPAEq(ns=16, ntheta=32, m_max=4)
    continuum = supernova.superContinuum(equ, dense=True)
    radialfe = supernova.BSplineFE(
        knots=np.linspace(0.0, 1.0, 5),
        order=2,
        domain=(0.0, 1.0),
        quad_order=3,
    )

    solver = superSolver5(equ, continuum, radialfe=radialfe)

    assert np.all(np.isfinite(solver.pressure_deriv.data.values))
    physical_span = _active_poloidal_flux_span(equ.boozer.coords.psi0)
    normalized_span = (
        physical_span * unyt.Unit("Wb/rad", registry=equ.registry)
    ).in_base(equ.my_norm).to("psi0/rad").v
    assert normalized_span == pytest.approx(
        physical_span / (equ.Baxis * equ.aminor**2)
    )
    assert solver.dpsi_hat_dx == pytest.approx(normalized_span)

    monkeypatch.setattr(
        solver2_module,
        "build_solver_blocks2",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    blocks = solver._assemble_solver_blocks_for_m_range(
        ntor=1,
        m_range=np.array([1], dtype=np.int64),
        max_dm=0,
    )
    assert blocks.dpsi_hat_dx == pytest.approx(normalized_span)


@pytest.mark.parametrize("psi0", [np.array([1.0, 1.0]), np.array([0.0, np.nan])])
def test_active_poloidal_flux_span_rejects_invalid_grids(psi0):
    with pytest.raises(ValueError):
        _active_poloidal_flux_span(psi0)


@pytest.mark.parametrize(
    ("psi_axis", "psi_boundary", "psi0", "expected"),
    [
        (1.0, 2.0, [1.05, 1.50, 1.95], [1.05, 1.50, 1.95]),
        (2.0, 1.0, [1.05, 1.50, 1.95], [1.95, 1.50, 1.05]),
    ],
)
def test_outward_boozer_grid_respects_physical_axis_to_boundary_direction(
    psi_axis,
    psi_boundary,
    psi0,
    expected,
):
    coordinate = xr.DataArray(
        psi0,
        dims=("psi0",),
        attrs={"psi_axis": psi_axis, "psi_boundary": psi_boundary},
    )
    target = _outward_boozer_interpolation_grid(
        coordinate,
        np.array([0.0, 0.5, 1.0]),
    )
    np.testing.assert_allclose(target, expected)


def test_outward_boozer_grid_requires_physical_endpoint_metadata():
    coordinate = xr.DataArray([1.0, 2.0], dims=("psi0",))

    with pytest.raises(ValueError, match="psi_axis and psi_boundary"):
        _outward_boozer_interpolation_grid(coordinate, np.array([0.0, 1.0]))


def test_realspace_theta_metric_components_change_sign_without_reindexing():
    base = np.arange(6, dtype=float).reshape(2, 3)
    dataset = xr.Dataset(
        {
            "gpsi_theta": (("R", "z"), base.copy()),
            "gtheta_zeta": (("R", "z"), base + 10.0),
            "gpsi_psi": (("R", "z"), base + 20.0),
        }
    )
    oriented = _orient_realspace_boozer_components(dataset, -1)
    np.testing.assert_allclose(oriented.gpsi_theta, -base)
    np.testing.assert_allclose(oriented.gtheta_zeta, -(base + 10.0))
    np.testing.assert_allclose(oriented.gpsi_psi, base + 20.0)
    assert oriented.attrs["poloidal_coordinate_orientation"] == -1


def test_supernova_rejects_explicitly_noncanonical_internal_equilibria():
    _validate_internal_equilibrium_convention(
        SimpleNamespace(
            cocos_internal=1,
            phiclockwise_internal=False,
            flux_normalization_internal="Wb/rad",
        )
    )
    with pytest.raises(ValueError, match="internal COCOS 1"):
        _validate_internal_equilibrium_convention(
            SimpleNamespace(
                cocos_internal=2,
                phiclockwise_internal=False,
                flux_normalization_internal="Wb",
            )
        )
    with pytest.raises(ValueError, match="phiclockwise_internal=False"):
        _validate_internal_equilibrium_convention(
            SimpleNamespace(
                cocos_internal=1,
                phiclockwise_internal=True,
                flux_normalization_internal="Wb/rad",
            )
        )
