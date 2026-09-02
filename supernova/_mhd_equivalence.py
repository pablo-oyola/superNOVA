from unyt.equivalencies import Equivalence
from unyt.exceptions import InvalidUnitEquivalence
from unyt.dimensions import (
    Rational,
    pressure as u_pressure,
    magnetic_field as u_magnetic_field,
    current_mks as u_current_mks,
    length as u_length,
    magnetic_flux as u_magnetic_flux,
    mass as u_mass,
    time as u_time,
    density as u_mass_density,
)
from unyt import unyt_quantity
import xarray as xr
from typing import Any, Tuple
import unyt
import numpy as np


_MHD_PRESSURE_DIMS = u_pressure
_MHD_B2_DIMS = u_magnetic_field**2

_MHD_RAW_JB_DIMS = u_current_mks * u_magnetic_field / u_length**2
_MHD_MU0_JB_DIMS = u_magnetic_field**2 / u_length

_MHD_RAW_JDIAG_DIMS = u_current_mks * u_magnetic_field**2 / u_length
_MHD_MU0_JDIAG_DIMS = u_magnetic_field**3


# Registered with unyt by subclassing Equivalence: the class body's
# ``type_name = "mhd_mu0"`` is what ``to_equivalent(..., "mhd_mu0")`` below
# resolves. Nothing names this class directly, so a reference-based dead-code
# sweep will call it unused -- it is not.
class MHDMu0Equivalence(Equivalence):
    """
    MHD SI equivalence that absorbs or restores a factor of mu_0.

    This is meant for normalized-MHD expression algebra, where:
        pressure -> mu_0 * pressure ~ B**2
        J.B      -> mu_0 * J.B      ~ B**2 / L
        J.s      -> mu_0 * J.s      ~ B**3
    """
    type_name = "mhd_mu0"

    _dims = (
        _MHD_PRESSURE_DIMS,
        _MHD_B2_DIMS,
        _MHD_RAW_JB_DIMS,
        _MHD_MU0_JB_DIMS,
        _MHD_RAW_JDIAG_DIMS,
        _MHD_MU0_JDIAG_DIMS,
    )

    def _convert(self, x, new_dims):
        old_dims = x.units.dimensions
        mu0 = _rebind_unyt_registry(unyt.mu_0, reg=x.units.registry)

        # pressure -> B**2
        if old_dims == _MHD_PRESSURE_DIMS and new_dims == _MHD_B2_DIMS:
            return x * mu0

        # B**2 -> pressure
        if old_dims == _MHD_B2_DIMS and new_dims == _MHD_PRESSURE_DIMS:
            return x / mu0

        # J.B -> mu0 J.B
        if old_dims == _MHD_RAW_JB_DIMS and new_dims == _MHD_MU0_JB_DIMS:
            return x * mu0

        # mu0 J.B -> J.B
        if old_dims == _MHD_MU0_JB_DIMS and new_dims == _MHD_RAW_JB_DIMS:
            return x / mu0

        # J.s -> mu0 J.s
        if old_dims == _MHD_RAW_JDIAG_DIMS and new_dims == _MHD_MU0_JDIAG_DIMS:
            return x * mu0

        # mu0 J.s -> J.s
        if old_dims == _MHD_MU0_JDIAG_DIMS and new_dims == _MHD_RAW_JDIAG_DIMS:
            return x / mu0

        raise InvalidUnitEquivalence(self, x.units, new_dims)

    def __str__(self):
        return "mhd_mu0: p <-> B**2, J.B <-> B**2/L, J.s <-> B**3"
    

def _has_unyt_units(x: Any) -> bool:
    """
    Return True for unyt arrays/quantities without depending on exact subclasses.
    """
    return hasattr(x, "units") and hasattr(x, "in_base")


def _rebind_unyt_registry(x: Any, reg: unyt.UnitRegistry=None):
    """
    Recreate a unyt value in the local registry.
    """
    if reg is None or not _has_unyt_units(x):
        return x
    return x.value * unyt.Unit(str(x.units), registry=reg)


def _sync_xarray_units_attr(x: xr.DataArray) -> xr.DataArray:
    """
    Make the xarray units attribute follow the actual unyt data units.
    """
    if _has_unyt_units(x.data):
        x.attrs["units"] = str(x.data.units)
    elif "units" not in x.attrs:
        x.attrs["units"] = "dimensionless"
    return x


def _unytfy_my_xarray(x: xr.DataArray, reg: unyt.UnitRegistry=None) -> xr.DataArray:
    """
    Using unyt, we attach the units.
    """
    x = x.copy(deep=True)

    if _has_unyt_units(x.data):
        x.data = _rebind_unyt_registry(x.data, reg=reg)
        return _sync_xarray_units_attr(x)

    if 'units' not in x.attrs:
        x.attrs['units'] = 'dimensionless'
        x.data = x.data * unyt.Unit('dimensionless', registry=reg)
        return x

    try:
        x.data = x.data * unyt.Unit(x.attrs['units'], registry=reg)
    except KeyError as e:
        print(f"Cannot find the units of {x.name} "
              f"with units {x.attrs['units']}. Following the exception {e}")
        raise e
    return _sync_xarray_units_attr(x)


def _in_unit_system(x: Any, unit_system, reg: unyt.UnitRegistry=None):
    """
    Convert xarray fields, unyt constants, or plain scalars to a unit system.
    """
    if isinstance(x, xr.DataArray):
        out = _unytfy_my_xarray(x, reg=reg)
        out.data = out.data.in_base(unit_system)
        return _sync_xarray_units_attr(out)

    if _has_unyt_units(x):
        return _rebind_unyt_registry(x, reg=reg).in_base(unit_system)

    if isinstance(x, (int, float, complex, np.number)):
        return x

    raise TypeError(f"Cannot convert object of type {type(x).__name__} to a unit system.")


def _deunytfy_my_xarray(x: xr.DataArray, reg: unyt.UnitRegistry=None) -> xr.DataArray:
    """
    Using unyt, we dequantify the units.
    """
    x = x.copy(deep=True)

    if _has_unyt_units(x.data):
        x.attrs['units'] = str(x.data.units)
        x.data = x.data.value
    elif 'units' not in x.attrs:
        x.attrs['units'] = 'dimensionless'
    return x


def _to_equivalent_xarray_or_unyt(x: Any, target_unit: str, reg: unyt.UnitRegistry):
    """
    Convert x to target_unit using the MHD mu0 equivalence.
    """
    target = unyt.Unit(target_unit, registry=reg)

    if isinstance(x, xr.DataArray):
        out = x.copy(deep=True)
        if _has_unyt_units(out.data):
            out.data = out.data.to_equivalent(target, "mhd_mu0")
            return _sync_xarray_units_attr(out)
        return out

    if _has_unyt_units(x):
        return x.to_equivalent(target, "mhd_mu0")

    return x


def _to_normalized_expression_units(name: str, x: Any, reg: unyt.UnitRegistry):
    """
    Recast normalized operands into canonical MHD expression units.

    In normalized expression mode:
        pressure -> Baxis**2
        J_B      -> Baxis**2 / aminor
        J_diag   -> Baxis**3
        mu_0     -> 1
    """
    if name == "mu_0":
        return 1.0

    def dims_of(obj):
        if isinstance(obj, xr.DataArray) and _has_unyt_units(obj.data):
            return obj.data.units.dimensions
        if _has_unyt_units(obj):
            return obj.units.dimensions
        return None

    dims = dims_of(x)

    if dims == _MHD_PRESSURE_DIMS:
        return _to_equivalent_xarray_or_unyt(
            x,
            "Baxis**2",
            reg,
        )

    if dims == _MHD_RAW_JB_DIMS:
        return _to_equivalent_xarray_or_unyt(
            x,
            "Baxis**2/aminor",
            reg,
        )

    if dims == _MHD_RAW_JDIAG_DIMS:
        return _to_equivalent_xarray_or_unyt(
            x,
            "Baxis**3",
            reg,
        )

    return x


def _expression_binary_op(left: Any, right: Any, op):
    """
    Apply a binary expression operation while preserving xarray coordinates.
    """
    if isinstance(left, xr.DataArray) and isinstance(right, xr.DataArray):
        return _sync_xarray_units_attr(op(left, right))

    if isinstance(left, xr.DataArray):
        out = left.copy(deep=True)
        out.data = op(left.data, right)
        return _sync_xarray_units_attr(out)

    if isinstance(right, xr.DataArray):
        out = right.copy(deep=True)
        out.data = op(left, right.data)
        return _sync_xarray_units_attr(out)

    return op(left, right)


def create_nova_registry(name: str, aminor: unyt_quantity, 
                         B0: unyt_quantity, 
                         n0: unyt_quantity) -> Tuple[unyt.UnitSystem, unyt.UnitRegistry]:
    """
    From the user-provided normalizations, we create a unyt registry with the appropriate equivalences.
    """
    # We need to create our own registry first:
    reg = unyt.UnitRegistry()

    aminor = aminor.to("m").v * unyt.Unit("m", registry=reg)
    B0 = B0.to("T").v * unyt.Unit("T", registry=reg)
    n0 = n0.to("m**-3").v * unyt.Unit("m**-3", registry=reg)
    mass_ion = 1*unyt.Unit('amu', registry=reg)
    rho0 = (n0 * mass_ion).to('kg/m**3').v * unyt.Unit('kg/m**3', registry=reg)
    mu0 = unyt.mu_0
    vA = (B0 / np.sqrt(mu0 * rho0)).to('m/s').v * unyt.Unit('m/s', registry=reg)
    tauA = (aminor / vA).to('s')

    M0 = rho0 * aminor**3
    I0 = B0 * aminor / mu0
    p0 = (B0**2 / mu0).to("Pa")
    psi0 = (B0 * aminor**2).to("Wb")

    # We will create new symbols to store the normalization factors.
    # All base_value entries must be in MKS values for the given dimension.
    reg.add(
        "aminor",
        base_value=float(aminor.to("m").value),
        dimensions=u_length,
        tex_repr=r"a",
    )

    reg.add(
        "M0",
        base_value=float(M0.to("kg").value),
        dimensions=u_mass,
        tex_repr=r"M_0",
    )

    reg.add(
        "tauA",
        base_value=float(tauA.to("s").value),
        dimensions=u_time,
        tex_repr=r"\tau_A",
    )

    reg.add(
        "I0",
        base_value=float(I0.to("A").value),
        dimensions=u_current_mks,
        tex_repr=r"I_0",
    )

    # Useful derived named units.
    reg.add(
        "Baxis",
        base_value=float(B0.to("T").value),
        dimensions=u_magnetic_field,
        tex_repr=r"B_{\rm axis}",
    )

    reg.add(
        "rhoaxis",
        base_value=float(rho0.to("kg/m**3").value),
        dimensions=u_mass_density,
        tex_repr=r"\rho_\rm{axis}",
    )

    reg.add(
        "p0",
        base_value=float(p0.to("Pa").value),
        dimensions=u_pressure,
        tex_repr=r"\frac{B_{\rm axis}^2}{\mu_0}",
    )

    reg.add(
        "sqrt_p0",
        base_value=np.sqrt(float(p0.to("Pa").value)),
        dimensions=u_pressure**Rational(1, 2),
        tex_repr=r"\sqrt{p_0}",
    )

    reg.add(
        "psi0",
        base_value=float(psi0.to("Wb").value),
        dimensions=u_magnetic_flux,
        tex_repr=r"\psi_\rm{wall}",
    )

    # Optional but useful.
    reg.add(
        "vA",
        base_value=float(vA.to("m/s").value),
        dimensions=u_length / u_time,
        tex_repr=r"v_A^\rm{axis}",
    )

    reg.add(
        "omegaA",
        base_value=float((1.0 / tauA).to("Hz").value),
        dimensions=1 / u_time,
        tex_repr=r"\omega_A",
    )

    reg.add(
        "naxis",
        base_value=float(n0.to("m**-3").value),
        dimensions=1 / u_length**3,
        tex_repr=r"n_{\rm axis}",
    )

    my_norm_name = f"nova_{name}"
    my_norm = unyt.UnitSystem(my_norm_name,
                              length_unit=unyt.Unit("aminor", registry=reg),
                              mass_unit=unyt.Unit("M0", registry=reg),
                              time_unit=unyt.Unit("tauA", registry=reg),
                              temperature_unit="K",
                              angle_unit="rad",
                              current_mks_unit=unyt.Unit("I0", registry=reg),
                              luminous_intensity_unit="cd",
                              registry=reg,
                            )
    my_norm["magnetic_field"] = unyt.Unit("Baxis", registry=reg)
    my_norm["pressure"] = unyt.Unit("p0", registry=reg)
    my_norm["magnetic_flux"] = unyt.Unit("psi0", registry=reg)

    # Depending on unyt version, this may or may not be accepted as a string
    # dimension key. If it fails, you can simply normalize rho_mass explicitly
    # with .to("rhoaxis") instead of relying on .in_base().
    try:
        my_norm["density"] = unyt.Unit("rhoaxis", registry=reg)
    except Exception:
        pass

    return my_norm, reg