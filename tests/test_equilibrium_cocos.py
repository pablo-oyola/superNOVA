from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from supernova.core.equilibrium import (
    _normalize_active_magnetic_radial_coordinate,
    _validate_internal_equilibrium_convention,
    superEquilibrium,
)


_CANONICAL_METADATA = {
    "cocos_internal": 1,
    "phiclockwise_internal": False,
    "flux_normalization_internal": "Wb/rad",
}


def _equilibrium_metadata(**updates):
    metadata = dict(_CANONICAL_METADATA)
    metadata.update(updates)
    return SimpleNamespace(**metadata)


def test_canonical_internal_cocos_metadata_is_accepted():
    _validate_internal_equilibrium_convention(_equilibrium_metadata())


def test_active_physical_flux_is_relabelled_for_solver_without_losing_provenance():
    physical = np.array([0.05, 0.20, 0.95])
    dataset = xr.Dataset(
        {"metric": ("psi", [1.0, 2.0, 3.0])},
        coords={
            "psi": xr.DataArray(
                physical,
                dims=("psi",),
                attrs={"name": "psi_norm", "units": ""},
            )
        },
    )

    result = _normalize_active_magnetic_radial_coordinate(dataset)

    np.testing.assert_allclose(result.psi, [0.0, 1.0 / 6.0, 1.0])
    np.testing.assert_allclose(result.psi_physical_norm, physical)
    assert result.psi.attrs["physical_psi_norm_min"] == pytest.approx(0.05)
    assert result.psi.attrs["physical_psi_norm_max"] == pytest.approx(0.95)
    assert result.psi_physical_norm.attrs["name"] == "psi_norm"


@pytest.mark.parametrize(
    "psi",
    ([0.5], [0.0, np.nan], [0.0, 0.0], [0.5, 0.4]),
)
def test_active_solver_flux_rejects_invalid_physical_coordinate(psi):
    dataset = xr.Dataset(
        {"metric": ("psi", np.ones(len(psi)))},
        coords={"psi": psi},
    )

    with pytest.raises(ValueError, match="active physical normalized-flux"):
        _normalize_active_magnetic_radial_coordinate(dataset)


def test_solver_geometry_bypasses_native_grid_round_trip():
    class FakeBoozer:
        def __init__(self):
            self.metric_calls = []
            self.projected = None
            self.R = np.linspace(1.0, 2.0, 9)
            self.z = np.linspace(-0.5, 0.5, 11)

        def metric(self, ii, jj, *, tensor):
            self.metric_calls.append((ii, jj, tensor))
            values = np.full((self.R.size, self.z.size), 10.0)
            values[[0, -1], :] = np.nan
            values[:, [0, -1]] = np.nan
            return xr.DataArray(
                values,
                dims=("R", "z"),
                coords={"R": self.R, "z": self.z},
            )

        def jacobian(self):
            return xr.DataArray(
                np.full((self.R.size, self.z.size), 5.0),
                dims=("R", "z"),
                coords={"R": self.R, "z": self.z},
            )

        def cyl2mag_scalar(self, *, field, return_psi_norm):
            assert return_psi_norm
            self.projected = field.copy(deep=True)
            shape = (3, 8, 1)
            return xr.Dataset(
                {
                    name: (
                        ("psi", "theta", "zeta"),
                        np.full(shape, index + 1.0),
                    )
                    for index, name in enumerate(field.data_vars)
                },
                coords={
                    "psi": [0.05, 0.5, 0.95],
                    "theta": np.linspace(0.0, 2.0 * np.pi, shape[1]),
                    "zeta": [0.0],
                },
            )

    fake = FakeBoozer()
    equilibrium = superEquilibrium.__new__(superEquilibrium)
    equilibrium.boozer = fake
    equilibrium._poloidal_orientation = -1

    geometry = equilibrium._boozer_geometry_on_solver_grid()

    assert len(fake.metric_calls) == 9
    assert set(geometry.data_vars) == {
        "gpsi_psi",
        "gpsi_theta",
        "gpsi_zeta",
        "gtheta_psi",
        "gtheta_theta",
        "gtheta_zeta",
        "gzeta_psi",
        "gzeta_theta",
        "gzeta_zeta",
        "jac",
    }
    assert fake.projected.sizes == {"R": 9, "z": 11}
    np.testing.assert_allclose(fake.projected["jac"], -5.0)
    np.testing.assert_allclose(geometry.psi, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(
        geometry.psi_physical_norm,
        [0.05, 0.5, 0.95],
    )
    assert np.all(np.isfinite(geometry.to_array()))
    assert geometry.gpsi_theta.attrs["projection"].startswith(
        "pyCOCOS coordinate R-Z"
    )


@pytest.mark.parametrize("missing_name", tuple(_CANONICAL_METADATA))
def test_missing_internal_cocos_metadata_is_rejected(missing_name):
    metadata = dict(_CANONICAL_METADATA)
    metadata.pop(missing_name)

    with pytest.raises(ValueError, match=missing_name):
        _validate_internal_equilibrium_convention(SimpleNamespace(**metadata))


@pytest.mark.parametrize("value", [2, True, 1.0, "1"])
def test_noncanonical_or_malformed_internal_cocos_is_rejected(value):
    with pytest.raises(ValueError, match="internal COCOS 1"):
        _validate_internal_equilibrium_convention(
            _equilibrium_metadata(cocos_internal=value)
        )


@pytest.mark.parametrize("value", [True, 0, "False"])
def test_clockwise_or_malformed_internal_orientation_is_rejected(value):
    with pytest.raises(ValueError, match="phiclockwise_internal=False"):
        _validate_internal_equilibrium_convention(
            _equilibrium_metadata(phiclockwise_internal=value)
        )


@pytest.mark.parametrize("value", ["Wb", None, 1])
def test_noncanonical_internal_flux_normalization_is_rejected(value):
    with pytest.raises(ValueError, match="flux_normalization_internal='Wb/rad'"):
        _validate_internal_equilibrium_convention(
            _equilibrium_metadata(flux_normalization_internal=value)
        )


def test_super_equilibrium_rejects_before_accessing_profiles():
    with pytest.raises(ValueError, match="internal COCOS 1"):
        superEquilibrium(
            equ=_equilibrium_metadata(cocos_internal=2),
            density=None,
            pressure=None,
        )


def test_solver_equilibrium_contract_is_explicitly_cocos_1():
    assert superEquilibrium.cocos_internal == 1
    assert superEquilibrium.phiclockwise_internal is False
    assert superEquilibrium.flux_normalization_internal == "Wb/rad"
