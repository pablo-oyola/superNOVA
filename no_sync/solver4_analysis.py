"""Import-safe analysis helpers for compatible mixed-spline solver4 results."""

import numpy as np


_FIELD_SPECS = {
    "p1": ("p1_coefficient", "radial_dof_high", "high"),
    "xi_psi": ("xi_psi_coefficient", "radial_dof_high", "high"),
    "xi_s": ("xi_s_coefficient", "radial_dof_low", "low"),
    "div_xi": ("div_xi_coefficient", "radial_dof_low", "low"),
}
_FIELD_ALIASES = {
    "xi^psi": "xi_psi",
    "xi^s": "xi_s",
    "divxi": "div_xi",
}


def _field_spec(field):
    key = _FIELD_ALIASES.get(str(field), str(field))
    if key not in _FIELD_SPECS:
        raise ValueError(f"field must be one of {list(_FIELD_SPECS)}, got {field!r}.")
    return key, _FIELD_SPECS[key]


def _compatible_pair(solver, pair=None):
    """Resolve solver4's cached pair, with a lazy public-constructor fallback."""
    if pair is not None:
        return pair

    for attribute in ("compatible_pair", "mixed_pair"):
        candidate = getattr(solver, attribute, None)
        if candidate is not None:
            return candidate() if callable(candidate) else candidate

    for method_name in ("_get_compatible_pair", "_get_mixed_pair"):
        method = getattr(solver, method_name, None)
        if method is not None:
            return method()

    mixed_data_method = getattr(solver, "_get_mixed_radial_data", None)
    if mixed_data_method is not None:
        return mixed_data_method().pair

    # Keep importing this optional diagnostic module cheap and side-effect free.
    from supernova.radial import CompatibleBSplinePair

    return CompatibleBSplinePair.from_high(solver.radialfe)


def _named_coefficients(result, field, eigenpair_index):
    key, (variable, radial_dim, _) = _field_spec(field)
    if variable not in result:
        raise ValueError(
            f"solver4 result is missing named coefficient array {variable!r}."
        )

    coefficients = result[variable]
    required_dims = {"eigenpair", radial_dim, "m_pol"}
    if not required_dims.issubset(coefficients.dims):
        raise ValueError(
            f"{variable} must contain dimensions {sorted(required_dims)}, "
            f"got {coefficients.dims}."
        )
    coefficients = coefficients.isel(eigenpair=int(eigenpair_index)).transpose(
        radial_dim, "m_pol"
    )
    values = np.asarray(coefficients.values, dtype=np.complex128)
    m_pol = np.asarray(coefficients.coords["m_pol"].values)
    return key, values, m_pol


def reconstruct_solver4_field(
    result,
    solver,
    field="xi_psi",
    eigenpair_index=0,
    points=None,
    n_plot=512,
    pair=None,
):
    """Reconstruct one named solver4 field without discarding complex phase."""
    pair = _compatible_pair(solver, pair=pair)
    key, coefficients, m_pol = _named_coefficients(
        result, field, eigenpair_index
    )
    basis_kind = _FIELD_SPECS[key][2]
    basis = pair.high if basis_kind == "high" else pair.low

    if coefficients.shape[0] != basis.get_ndofs():
        raise ValueError(
            f"{_FIELD_SPECS[key][0]} has {coefficients.shape[0]} radial "
            f"coefficients, but the {basis_kind} basis has {basis.get_ndofs()}."
        )

    if points is None:
        mesh = np.asarray(pair.high.get_mesh(), dtype=float)
        points = np.linspace(mesh[0], mesh[-1], int(n_plot))
    else:
        points = np.asarray(points, dtype=float)
    if points.ndim != 1:
        raise ValueError("points must be a one-dimensional array.")

    # BSplineFE.evaluate currently casts coefficients to float.  Matrix
    # multiplication with the real basis keeps the complex eigenmode intact.
    phi = np.asarray(basis.eval_basis(points), dtype=float).T
    values = phi @ coefficients
    return points, m_pol, values


def solver4_field_envelope(
    result,
    solver,
    field="xi_psi",
    eigenpair_index=0,
    points=None,
    n_plot=512,
    pair=None,
):
    """Return the harmonic Euclidean envelope of a reconstructed field."""
    points, _, values = reconstruct_solver4_field(
        result,
        solver,
        field=field,
        eigenpair_index=eigenpair_index,
        points=points,
        n_plot=n_plot,
        pair=pair,
    )
    return points, np.linalg.norm(values, axis=1)


def _radial_coordinate(equ, psi):
    psi = np.asarray(psi, dtype=float)
    if hasattr(equ, "_psi") and hasattr(equ, "_rho"):
        return np.interp(psi, equ._psi, equ._rho)
    return np.sqrt(np.clip(psi, 0.0, None))


def xipsi_effective_kpsi(result, solver, sb, eigenpair_index=None, pair=None):
    """Return the weighted effective radial wavevector of named ``xi_psi``."""
    pair = _compatible_pair(solver, pair=pair)
    psi = np.asarray(sb.points_q, dtype=float)
    weights = np.asarray(sb.weights_q, dtype=float)
    phi = np.asarray(pair.high.eval_basis(psi), dtype=float).T
    dphi = np.asarray(pair.high.eval_basis_derivative(psi), dtype=float).T

    density = solver.equ.density.data.sel(mpol=0, ntor=0).real.interp(x=psi).values
    density = density * solver.cont.mass.to("kg").v
    h = solver.equ("h").data.sel(mpol=0, ntor=0).real.interp(x=psi).values
    q = solver.equ("q").data.sel(mpol=0, ntor=0).real.interp(x=psi).values
    radius = _radial_coordinate(solver.equ, psi) * solver.equ.aminor
    with np.errstate(divide="ignore", invalid="ignore"):
        physical_weight = np.abs(
            density * h * q**2 / solver.equ.Baxis**2 / radius**2
        )
    radial_weight = weights * physical_weight

    indices = range(result.sizes.get("eigenpair", 0))
    if eigenpair_index is not None:
        indices = [int(eigenpair_index)]

    values = []
    for index in indices:
        _, coefficients, _ = _named_coefficients(result, "xi_psi", index)
        xipsi = phi @ coefficients
        dxipsi = dphi @ coefficients
        numerator = np.sum(radial_weight[:, None] * np.abs(dxipsi) ** 2)
        denominator = np.sum(radial_weight[:, None] * np.abs(xipsi) ** 2)
        if (
            not np.isfinite(numerator)
            or not np.isfinite(denominator)
            or denominator <= 0
        ):
            values.append(np.nan)
        else:
            values.append(float(np.sqrt(numerator / denominator)))

    values = np.asarray(values)
    return float(values[0]) if eigenpair_index is not None else values


def xipsi_radial_location(result, solver, sb=None, n_plot=512, pair=None):
    """Return peak position and RMS width of each named ``xi_psi`` envelope."""
    del sb  # Retained for compatibility with solver3 diagnostic call sites.
    pair = _compatible_pair(solver, pair=pair)
    mesh = np.asarray(pair.high.get_mesh(), dtype=float)
    psi = np.linspace(mesh[0], mesh[-1], int(n_plot))
    n_eigenpairs = result.sizes.get("eigenpair", 0)
    peak_psi = np.empty(n_eigenpairs)
    radial_width = np.empty(n_eigenpairs)

    for index in range(n_eigenpairs):
        _, envelope = solver4_field_envelope(
            result,
            solver,
            field="xi_psi",
            eigenpair_index=index,
            points=psi,
            pair=pair,
        )
        envelope2 = envelope**2
        norm = np.trapezoid(envelope2, psi)
        peak_psi[index] = psi[int(np.argmax(envelope2))]
        if norm > 0.0:
            center = np.trapezoid(psi * envelope2, psi) / norm
            variance = np.trapezoid((psi - center) ** 2 * envelope2, psi) / norm
            radial_width[index] = np.sqrt(max(variance, 0.0))
        else:
            radial_width[index] = np.nan

    return peak_psi, radial_width


__all__ = [
    "reconstruct_solver4_field",
    "solver4_field_envelope",
    "xipsi_effective_kpsi",
    "xipsi_radial_location",
]
