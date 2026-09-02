from types import SimpleNamespace

import numpy as np
import xarray as xr

import supernova.solve.stability as solver2_module
from supernova.solve.stability import StabilitySolver
from supernova.radial.profile import Profile


class _NormalizedQuantity:
    v = 1.0

    def in_base(self, _unit_system):
        return self

    def to(self, _unit):
        return self


class _NormalizedUnit:
    def __rmul__(self, _value):
        return _NormalizedQuantity()


class _Equilibrium:
    def __init__(self, x):
        self.x = np.asarray(x, dtype=np.float64)
        self.registry = object()
        self.my_norm = object()
        self.boozer = SimpleNamespace(coords=SimpleNamespace(psi0=object()))

    def _profile(self, name):
        scale = 1.0 + sum(map(ord, name)) % 11
        values = scale + self.x**2
        return Profile(self.x, values, mpol=[0], ntor=[0])

    def __call__(self, *names, normalize=False):
        assert normalize
        profiles = [self._profile(name) for name in names]
        return profiles if len(profiles) > 1 else profiles[0]


class _RadialFE:
    def __init__(self, points):
        self._points = np.asarray(points, dtype=np.float64)

    def get_quadrature_points(self):
        return self._points.copy(), np.ones(self._points.size)

    def get_mesh(self):
        return np.array([0.0, 1.0])

    def eval_basis(self, points):
        return np.ones((1, np.asarray(points).size))

    @property
    def weak_derivative_matrix(self):
        return np.ones((1, 1))


def _continuum(equilibrium):
    continuum = SimpleNamespace(
        gamma_s=5.0 / 3.0,
        mass=_NormalizedQuantity(),
    )
    for index, name in enumerate(("H", "J", "K", "L", "M"), start=1):
        values = index + equilibrium.x**2
        setattr(
            continuum,
            name,
            Profile(equilibrium.x, values, mpol=[0], ntor=[0]),
        )
    return continuum


def _profile_snapshots(owner):
    return {
        name: getattr(owner, name).data.copy(deep=True)
        for name in ("H", "J", "K", "L", "M")
    }


def _assert_profiles_unchanged(owner, snapshots):
    for name, expected in snapshots.items():
        xr.testing.assert_identical(getattr(owner, name).data, expected)


def test_solver_resampling_owns_continuum_profiles_across_radial_grids(monkeypatch):
    monkeypatch.setattr(
        solver2_module.unyt,
        "Unit",
        lambda *_args, **_kwargs: _NormalizedUnit(),
    )
    monkeypatch.setattr(
        solver2_module,
        "_active_poloidal_flux_span",
        lambda _psi0: 1.0,
    )

    equilibrium = _Equilibrium(np.linspace(0.0, 1.0, 9))
    continuum = _continuum(equilibrium)
    continuum_before = _profile_snapshots(continuum)

    first = StabilitySolver(equilibrium, continuum, _RadialFE([0.2, 0.6]))

    _assert_profiles_unchanged(continuum, continuum_before)
    for name in continuum_before:
        assert getattr(first, name) is not getattr(continuum, name)
    assert first.D21_2 is first.M
    first_before = _profile_snapshots(first)

    second = StabilitySolver(equilibrium, continuum, _RadialFE([0.15, 0.45, 0.75]))

    _assert_profiles_unchanged(continuum, continuum_before)
    _assert_profiles_unchanged(first, first_before)
    for name in continuum_before:
        assert getattr(second, name) is not getattr(continuum, name)
        assert getattr(second, name) is not getattr(first, name)
    assert second.D21_2 is second.M


def test_solver_block_assembly_uses_solver_owned_continuum_profiles(monkeypatch):
    monkeypatch.setattr(
        solver2_module.unyt,
        "Unit",
        lambda *_args, **_kwargs: _NormalizedUnit(),
    )
    monkeypatch.setattr(
        solver2_module,
        "_active_poloidal_flux_span",
        lambda _psi0: 1.0,
    )
    monkeypatch.setattr(
        solver2_module,
        "build_solver_blocks",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    equilibrium = _Equilibrium(np.linspace(0.0, 1.0, 9))
    continuum = _continuum(equilibrium)
    solver = StabilitySolver(equilibrium, continuum, _RadialFE([0.2, 0.6]))

    for name in ("H", "J", "K", "L", "M"):
        getattr(continuum, name).data.values[...] = -1000.0

    blocks = solver._assemble_solver_blocks_for_m_range(
        ntor=0,
        m_range=np.array([0], dtype=np.int64),
        max_dm=0,
    )

    for name in ("H", "J", "K", "L", "M"):
        expected = solver._profile_harmonics(getattr(solver, name), 0, 0)
        np.testing.assert_array_equal(getattr(blocks, name), expected)
