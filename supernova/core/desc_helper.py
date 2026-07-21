"""DESC-backed equilibrium solver for superNOVA."""

from types import SimpleNamespace
from typing import Optional

import numpy as np
import xarray as xr
from scipy.constants import mu_0
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from ..radial.profile import Profile
from .desc_loader import *
from .equilibrium import (
    _validate_internal_equilibrium_convention,
    superEquilibrium,
)


def _profile_values(profile: Profile):
    return np.asarray(
        profile.data.sel(mpol=0, ntor=0).values.real,
        dtype=np.float64,
    )


def _reflect_boozer_surface_arrays(data, metric, shear, div_psi, orientation):
    """Express DESC surface arrays in the solver's positive-q Boozer angle."""
    if orientation not in (-1, 1):
        raise ValueError("The Boozer poloidal orientation must be +1 or -1.")
    if orientation == 1:
        return data, metric, shear, div_psi

    ntheta = metric.shape[1]
    indices = np.concatenate(([0], np.arange(ntheta - 1, 0, -1)))
    reflected = {
        name: np.take(values, indices, axis=1)
        for name, values in data.items()
    }
    metric = np.take(metric, indices, axis=1)
    shear = np.take(shear, indices, axis=1)
    div_psi = np.take(div_psi, indices, axis=1)

    # Theta'=-Theta: a contravariant metric component changes sign exactly
    # when it carries one theta index.
    metric[..., 1, :] *= -1.0
    metric[..., :, 1] *= -1.0
    return reflected, metric, shear, div_psi


def _rho_tor_from_q(psipol, q):
    """Map normalized poloidal flux to DESC rho."""
    psipol = np.asarray(psipol, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if np.any(~np.isfinite(q)) or np.any(np.abs(q) < 1e-12):
        raise ValueError("q must be finite and non-zero.")
    if np.any(np.sign(q) != np.sign(q[0])):
        raise ValueError("q must keep the same sign.")

    psitor = cumulative_trapezoid(q, psipol, initial=0.0)
    psitor /= psitor[-1]
    if np.any(np.diff(psitor) <= 0.0):
        raise ValueError("q does not produce a monotonic flux mapping.")
    psitor[0], psitor[-1] = 0.0, 1.0
    return np.sqrt(psitor)


def _miller_angle(R, Z):
    """Fit a nonlinear Miller angle to a closed R-Z contour."""
    Rcenter = 0.5 * (R.max() + R.min())
    Zcenter = 0.5 * (Z.max() + Z.min())
    aminor = 0.5 * (R.max() - R.min())
    bminor = 0.5 * (Z.max() - Z.min())
    theta = np.mod(
        np.arctan2((Z - Zcenter) / bminor, (R - Rcenter) / aminor),
        2 * np.pi,
    )
    order = np.argsort(theta)
    Rfit, Zfit, theta_fit = R[order], Z[order], theta[order]

    def residual(values):
        R0, Z0, a, kappa, delta = values[:5]
        angle = values[5:]
        alpha = np.arcsin(delta)
        return np.r_[
            R0 + a * np.cos(angle + alpha * np.sin(angle)) - Rfit,
            Z0 + kappa * a * np.sin(angle) - Zfit,
        ]

    initial = np.r_[Rcenter, Zcenter, aminor, bminor / aminor, 0.3, theta_fit]
    lower = np.r_[R.min(), Z.min(), 0.3 * aminor, 0.5, -0.9, theta_fit - 0.8]
    upper = np.r_[R.max(), Z.max(), 1.7 * aminor, 3.0, 0.9, theta_fit + 0.8]
    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=3000,
    )
    angle = np.empty_like(theta)
    angle[order] = result.x[5:]
    return np.mod(angle, 2 * np.pi)


def _spectral_condensation_angle(R, Z, theta, mmax, kmax):
    """Minimize the boundary Fourier residual with a monotonic angle map."""
    if kmax == 0:
        return theta

    def design(angle):
        columns = [np.ones_like(angle)]
        for m in range(1, mmax + 1):
            columns.extend((np.cos(m * angle), np.sin(m * angle)))
        return np.column_stack(columns)

    dense_angle = np.linspace(0.0, 2 * np.pi, 2048, endpoint=False)

    def corrected(coefficients, angle):
        output = angle.copy()
        for k in range(1, kmax + 1):
            output += coefficients[2 * k - 2] * np.sin(k * angle)
            output += coefficients[2 * k - 1] * np.cos(k * angle)
        return output

    def derivative(coefficients, angle):
        output = np.ones_like(angle)
        for k in range(1, kmax + 1):
            output += k * coefficients[2 * k - 2] * np.cos(k * angle)
            output -= k * coefficients[2 * k - 1] * np.sin(k * angle)
        return output

    def residual(coefficients):
        angle = corrected(coefficients, theta)
        matrix = design(angle)
        Rfit = matrix @ np.linalg.lstsq(matrix, R, rcond=None)[0]
        Zfit = matrix @ np.linalg.lstsq(matrix, Z, rcond=None)[0]
        monotonicity = 10.0 * np.minimum(
            derivative(coefficients, dense_angle) - 0.15,
            0.0,
        )
        return np.r_[Rfit - R, Zfit - Z, monotonicity]

    result = least_squares(
        residual,
        np.zeros(2 * kmax),
        max_nfev=1500,
    )
    return np.mod(corrected(result.x, theta), 2 * np.pi)


def _fit_lcfs(R, Z, theta, mmax=10, kmax=4):
    """Fit the LCFS using a spectrally condensed clockwise angle."""
    condensed_theta = _spectral_condensation_angle(
        R,
        Z,
        _miller_angle(R, Z),
        mmax,
        kmax,
    )
    theta_origin = np.argmin(np.abs(np.angle(np.exp(1j * theta))))
    condensed_theta = np.mod(
        condensed_theta - condensed_theta[theta_origin],
        2 * np.pi,
    )
    theta_desc = np.mod(-condensed_theta, 2 * np.pi)
    order = np.argsort(theta_desc)
    theta_desc = theta_desc[order]
    coords = np.column_stack(
        (R[order], np.zeros_like(R[order]), Z[order])
    )
    surface = FourierRZToroidalSurface.from_values(
        coords=coords,
        theta=theta_desc,
        M=mmax,
        N=0,
        NFP=1,
        sym=False,
    )
    return surface, condensed_theta


def _make_guesses(
    equ,
    psipol,
    rhotor,
    theta_lcfs,
    theta_condensed,
    psipol_scale=1.0,
):
    """Use a few pyCOCOS surfaces as the DESC initial guess."""
    Raxis = float(equ.fluxdata.attrs.get("Raxis", equ.R_axis))
    Zaxis = float(equ.fluxdata.attrs.get("zaxis", equ.z_axis))
    nodes, R_values, Z_values = [], [], []

    order = np.argsort(theta_lcfs)
    theta_lcfs = theta_lcfs[order]
    theta_condensed = np.unwrap(theta_condensed[order])
    theta_lcfs = np.r_[theta_lcfs, theta_lcfs[0] + 2 * np.pi]
    theta_condensed = np.r_[theta_condensed, theta_condensed[0] + 2 * np.pi]

    for x, rho in zip(psipol, rhotor):
        surface = equ.flux_surface(np.sqrt(psipol_scale * x))
        R = np.asarray(surface["R"].values, dtype=np.float64)
        Z = np.asarray(surface["z"].values, dtype=np.float64)
        if R.size < 4 or Z.size < 4:
            raise ValueError(f"Could not construct the flux surface at psi_pol={x}.")
        if np.allclose([R[0], Z[0]], [R[-1], Z[-1]]):
            R, Z = R[:-1], Z[:-1]
        theta_geom = np.mod(np.arctan2(Z - Zaxis, R - Raxis), 2 * np.pi)
        theta_geom = np.where(
            theta_geom < theta_lcfs[0],
            theta_geom + 2 * np.pi,
            theta_geom,
        )
        theta = np.mod(
            -np.interp(theta_geom, theta_lcfs, theta_condensed),
            2 * np.pi,
        )
        order = np.argsort(theta)
        theta, R, Z = theta[order], R[order], Z[order]
        nodes.append(np.column_stack((np.full_like(theta, rho), theta, np.zeros_like(theta))))
        R_values.append(R)
        Z_values.append(Z)

    return (
        Grid(np.vstack(nodes), NFP=1, sort=False),
        np.concatenate(R_values),
        np.concatenate(Z_values),
    )


def _reshape(values, nrho, ntheta):
    return np.asarray(values, dtype=np.float64).reshape(nrho, ntheta)


def _local_magnetic_shear_desc(eq, grid):
    """Compute NOVA local shear using DESC poloidal flux chi in Webers."""
    data = eq.compute(
        ["B", "J", "e^rho", "g^rr", "(B*grad) grad(rho)", "chi_r"],
        grid=grid,
        basis="xyz",
    )
    geometric_part = (
        mu_0 * np.einsum("...i,...i->...", data["B"], data["J"])
        / data["g^rr"]
        - 2.0
        * np.einsum(
            "...i,...i->...",
            np.cross(data["B"], data["e^rho"]),
            data["(B*grad) grad(rho)"],
        )
        / data["g^rr"] ** 2
    )
    return -geometric_part / data["chi_r"] ** 2


def _to_boozer(values, theta_B, theta):
    """Interpolate rows from DESC theta to a uniform Boozer theta."""
    out = np.empty((values.shape[0], theta.size))
    for i in range(values.shape[0]):
        angle = np.mod(theta_B[i], 2 * np.pi)
        order = np.argsort(angle)
        angle = angle[order]
        row = values[i, order]
        keep = np.concatenate(([True], np.diff(angle) > 1e-10))
        angle, row = angle[keep], row[keep]
        angle = np.concatenate((angle - 2 * np.pi, angle, angle + 2 * np.pi))
        row = np.concatenate((row, row, row))
        out[i] = np.interp(theta, angle, row)
    return out


def _dtheta(values):
    ntheta = values.shape[1]
    m = np.fft.fftfreq(ntheta, d=1.0 / ntheta)
    return np.fft.ifft(1j * m[None, :] * np.fft.fft(values, axis=1), axis=1).real


class DESCSolver(superEquilibrium):
    """Solve an axisymmetric equilibrium with DESC.

    All input Profile objects use normalized poloidal flux as their x coordinate.
    """

    def __init__(
        self,
        LCFS: dict,
        Baxis: float,
        q: Profile,
        density: Profile,
        pressure: Profile,
        resistivity: Optional[Profile] = None,
        vtor: Optional[Profile] = None,
        vpol: Optional[Profile] = None,
        m_max: int = 128,
        boundary_m_max: int = 10,
        boundary_k_max: int = 4,
        desc_resolution: tuple[int, int] = (6, 6),
        mag_resolution: tuple[int, int] = (129, 256),
        psi_domain: tuple[float, float] = (0.005, 0.995),
        initial_guess=None,
        solve_kwargs: Optional[dict] = None,
    ):
        if "R" not in LCFS or "theta" not in LCFS or not ("Z" in LCFS or "z" in LCFS):
            raise ValueError("LCFS must contain R, Z, and theta.")

        def values_in(value, unit):
            if hasattr(value, "to"):
                value = value.to(unit).value
            return np.asarray(value, dtype=np.float64)

        self.Rlcfs = values_in(LCFS["R"], "m")
        self.Zlcfs = values_in(LCFS.get("Z", LCFS.get("z")), "m")
        self.theta_lcfs = values_in(LCFS["theta"], "rad")
        if not (self.Rlcfs.ndim == self.Zlcfs.ndim == self.theta_lcfs.ndim == 1):
            raise ValueError("LCFS arrays must be one-dimensional.")
        if not (self.Rlcfs.size == self.Zlcfs.size == self.theta_lcfs.size):
            raise ValueError("LCFS arrays must have the same size.")
        if np.allclose(
            [self.Rlcfs[0], self.Zlcfs[0]],
            [self.Rlcfs[-1], self.Zlcfs[-1]],
        ):
            self.Rlcfs = self.Rlcfs[:-1]
            self.Zlcfs = self.Zlcfs[:-1]
            self.theta_lcfs = self.theta_lcfs[:-1]
        self.theta_lcfs = np.mod(self.theta_lcfs, 2 * np.pi)
        order = np.argsort(self.theta_lcfs)
        self.Rlcfs = self.Rlcfs[order]
        self.Zlcfs = self.Zlcfs[order]
        self.theta_lcfs = self.theta_lcfs[order]

        self.Baxis_target = float(values_in(Baxis, "T"))
        if self.Baxis_target <= 0.0:
            raise ValueError("Baxis must be positive.")

        for name, profile in [("q", q), ("density", density), ("pressure", pressure)]:
            if not isinstance(profile, Profile):
                raise TypeError(f"{name} must be a Profile.")
            if profile.m_pol.tolist() != [0] or profile.n_tor.tolist() != [0]:
                raise ValueError(f"{name} must be an axisymmetric flux function.")

        self.qin = q.copy()
        self.density = density.copy()
        self.pressure = pressure.copy()
        x = self.qin.x
        zero = lambda name, units: Profile(
            x=x,
            F_nm=np.zeros_like(x),
            attrs={"name": name, "units": units},
        )
        self.eta = resistivity.copy() if resistivity is not None else zero("eta", "ohm*m")
        self.resistivity = self.eta
        self.vtor = vtor.copy() if vtor is not None else zero("vtor", "m/s")
        self.vpol = vpol.copy() if vpol is not None else zero("vpol", "m/s")
        for profile in [self.density, self.pressure, self.eta, self.vtor, self.vpol]:
            if not np.allclose(profile.x, x):
                raise ValueError("All profiles must use the same x grid.")
        if x.size < 4 or not np.isclose(x[0], 0.0) or not np.isclose(x[-1], 1.0):
            raise ValueError("Profile x must contain at least four points including 0 and 1.")

        # Input profiles use normalized poloidal flux. DESC profiles use
        # rho_tor = sqrt(normalized toroidal flux), obtained by integrating q.
        self.rho_tor_input = _rho_tor_from_q(x, _profile_values(self.qin))
        self.lcfs = {"R": self.Rlcfs, "Z": self.Zlcfs, "theta": self.theta_lcfs}
        self.lcfs_fit, self.theta_lcfs_condensed = _fit_lcfs(
            self.Rlcfs,
            self.Zlcfs,
            self.theta_lcfs,
            boundary_m_max,
            boundary_k_max,
        )
        self.boundary_m_max = boundary_m_max
        self.boundary_k_max = boundary_k_max
        self.desc_pressure = SplineProfile(
            _profile_values(self.pressure), knots=self.rho_tor_input, name="p"
        )
        self.desc_density = SplineProfile(
            _profile_values(self.density), knots=self.rho_tor_input, name="density"
        )
        self.desc_iota = SplineProfile(
            1.0 / _profile_values(self.qin), knots=self.rho_tor_input, name="iota"
        )
        self.desc_resolution = desc_resolution
        self.mag_resolution = mag_resolution
        self.psi_domain = psi_domain
        self.initial_guess = initial_guess
        self.initial_guess_psipol_scale = 1.0
        self.rhop_boundary = 1.0
        self.psipol_boundary = 1.0
        self.solve_kwargs = solve_kwargs or {}
        self.m_max = abs(m_max)
        self.n_max = 0
        self.realspace = xr.Dataset()
        self._cache = {}

    @classmethod
    def from_eqdsk(
        cls,
        equ,
        density: Profile,
        use_surface_guess=True,
        rhop_boundary=1.0,
        **kwargs,
    ):
        """Build the same solver from an EQDSK object."""
        _validate_internal_equilibrium_convention(equ)
        if "q" not in equ.gs_profs or "pres" not in equ.gs_profs:
            raise ValueError("The EQDSK must contain q and pressure profiles.")
        if not 0.0 < rhop_boundary <= 1.0:
            raise ValueError("rhop_boundary must satisfy 0 < rhop_boundary <= 1.")

        # EQDSK profiles are tabulated on rho_pol. Profile.x is normalized
        # poloidal flux, so x = rho_pol**2 before the q-based rho_tor mapping.
        q_eqdsk = equ.gs_profs["q"]
        pressure_eqdsk = equ.gs_profs["pres"]
        q_psipol = np.asarray(q_eqdsk.coords["rhop"], dtype=np.float64) ** 2
        pressure_psipol = np.asarray(
            pressure_eqdsk.coords["rhop"], dtype=np.float64
        ) ** 2
        x = density.x
        psipol_boundary = rhop_boundary**2
        x_eqdsk = psipol_boundary * x
        q = Profile(
            x=x,
            F_nm=np.interp(x_eqdsk, q_psipol, q_eqdsk.values),
            attrs={"name": "q", "units": "1"},
        )
        pressure = Profile(
            x=x,
            F_nm=np.interp(x_eqdsk, pressure_psipol, pressure_eqdsk.values),
            attrs={"name": "pressure", "units": "Pa"},
        )

        def truncate(profile):
            if profile is None:
                return None
            return Profile(
                x=x,
                F_nm=np.interp(x_eqdsk, profile.x, _profile_values(profile)),
                attrs=dict(profile.data.attrs),
            )

        density = truncate(density)
        resistivity = truncate(kwargs.pop("resistivity", None))
        vtor = truncate(kwargs.pop("vtor", None))
        vpol = truncate(kwargs.pop("vpol", None))

        if rhop_boundary < 1.0:
            boundary = equ.flux_surface(rhop_boundary)
            R = np.asarray(boundary["R"].values, dtype=np.float64)
            Z = np.asarray(boundary["z"].values, dtype=np.float64)
        else:
            R = np.asarray(equ.boundary.R)
            Z = np.asarray(equ.boundary.z)
        Raxis = float(equ.fluxdata.attrs.get("Raxis", equ.R_axis))
        Zaxis = float(equ.fluxdata.attrs.get("zaxis", equ.z_axis))
        theta = np.unwrap(np.arctan2(Z - Zaxis, R - Raxis))
        solver = cls(
            {"R": R, "Z": Z, "theta": theta},
            float(equ.Bdata.Baxis),
            q,
            density,
            pressure,
            resistivity=resistivity,
            vtor=vtor,
            vpol=vpol,
            initial_guess=equ if use_surface_guess else None,
            **kwargs,
        )
        solver.initial_guess_psipol_scale = psipol_boundary
        solver.rhop_boundary = rhop_boundary
        solver.psipol_boundary = psipol_boundary
        return solver

    def solve(
        self,
        initial_guess=None,
        psi_tor_guess=None,
        max_psi_iterations=6,
        rtol=1e-3,
        fast=False,
        **kwargs,
    ):
        """Solve DESC, match Baxis, and build magspace."""
        if psi_tor_guess is None:
            aminor = 0.5 * (self.Rlcfs.max() - self.Rlcfs.min())
            psitor = self.Baxis_target * np.pi * aminor**2
        else:
            psitor = float(psi_tor_guess.to("Wb").value) if hasattr(psi_tor_guess, "to") else float(psi_tor_guess)

        guess = self.initial_guess if initial_guess is None else initial_guess
        if guess is not None and hasattr(guess, "flux_surface"):
            idx = np.unique(np.linspace(1, self.qin.x.size - 2, min(16, self.qin.x.size - 2), dtype=int))
            guess = _make_guesses(
                guess,
                self.qin.x[idx],
                self.rho_tor_input[idx],
                self.theta_lcfs,
                self.theta_lcfs_condensed,
                self.initial_guess_psipol_scale,
            )

        L, M = self.desc_resolution
        solve_kwargs = dict(self.solve_kwargs)
        solve_kwargs.update(kwargs)
        verbose = solve_kwargs.pop("verbose", 0)
        self.history = []

        for iteration in range(max(max_psi_iterations, 1)):
            eq = DESC_Equilibrium(
                L=L,
                M=M,
                N=0,
                L_grid=max(2 * L, L + 3),
                M_grid=max(2 * M, M + 3),
                N_grid=0,
                surface=self.lcfs_fit,
                pressure=self.desc_pressure,
                iota=self.desc_iota,
                Psi=psitor,
                NFP=1,
                sym=False,
            )
            if guess is not None:
                if isinstance(guess, tuple):
                    eq.set_initial_guess(*guess)
                else:
                    eq.set_initial_guess(guess)
            if verbose:
                print(f"--- Solving DESC equilibrium, iteration {iteration + 1} ---")
            if fast:
                solved, _ = eq.solve(verbose=0, copy=True, **solve_kwargs)
            else:
                family = solve_continuation_automatic(eq, verbose=0, **solve_kwargs)
                try:
                    solved = family[-1]
                except TypeError:
                    solved = family

            Baxis = float(np.asarray(solved.compute("<|B|>_axis")["<|B|>_axis"]))
            error = abs(Baxis - self.Baxis_target) / self.Baxis_target
            self.history.append({"Psi": psitor, "Baxis": Baxis, "error": error})
            if error < rtol:
                break
            psitor *= self.Baxis_target / Baxis
            guess = solved
        else:
            raise RuntimeError(f"DESC did not match Baxis. Last relative error: {error:.3e}")

        self.desc_equilibrium = solved
        self.last_equilibrium = solved
        self.Baxis = Baxis
        self.Psi_tor_total = float(solved.Psi)
        data = solved.compute(["R0", "a"])
        self.Rmajor = float(np.asarray(data["R0"]))
        self.aminor = float(np.asarray(data["a"]))
        self._prepare_unit_system()
        self.build_magspace()
        return self

    def build_magspace(self, mag_resolution=None, psi_domain=None):
        """Build the superNOVA magnetic database directly from DESC."""
        npsi, ntheta = mag_resolution or self.mag_resolution
        xmin, xmax = psi_domain or self.psi_domain
        if npsi < 4 or ntheta < 8 or not 0 < xmin < xmax < 1:
            raise ValueError("Use npsi>=4, ntheta>=8, and 0<psi_min<psi_max<1.")

        # Find rho(x) from DESC's own poloidal flux chi.
        rho_map = np.linspace(0.0, 1.0, max(4 * npsi, 257))
        grid_map = LinearGrid(rho=rho_map, theta=[0.0], zeta=[0.0], NFP=1)
        chi_map = np.asarray(self.desc_equilibrium.compute("chi", grid=grid_map)["chi"])
        chi_edge = float(chi_map[-1])
        x = np.linspace(xmin, xmax, npsi)
        rho = np.interp(x, chi_map / chi_edge, rho_map)

        theta_desc = np.linspace(0.0, 2 * np.pi, ntheta, endpoint=False)
        theta = np.linspace(0.0, 2 * np.pi, ntheta, endpoint=False)
        grid = LinearGrid(rho=rho, theta=theta_desc, zeta=[0.0], NFP=1)
        names = [
            "R", "Z", "chi", "chi_r", "psi_r", "psi_rr", "q", "iota",
            "iota_r", "G", "I",
            "|B|", "theta_B", "zeta_B", "theta_PEST_r", "theta_PEST_t",
            "theta_PEST_z", "phi_r", "phi_t", "phi_z", "nu", "nu_t", "nu_z",
            "sqrt(g)_Boozer", "g^rr", "g^rt", "g^rz", "g^tt", "g^tz", "g^zz",
            "g^rr_r", "g^rr_t", "g^rr_z", "curvature_H_rho", "J*B", "p_r",
            "kappa_n", "kappa_g", "|grad(rho)|",
        ]
        raw = self.desc_equilibrium.compute(names, grid=grid)
        data = {name: _reshape(raw[name], npsi, ntheta) for name in names}
        data["nu_r"] = CubicSpline(rho, data["nu"], axis=0).derivative()(rho)

        shear = _reshape(
            _local_magnetic_shear_desc(self.desc_equilibrium, grid),
            npsi,
            ntheta,
        )

        chi_rr = (
            data["psi_rr"] * data["iota"]
            + data["psi_r"] * data["iota_r"]
        )
        # Keep the full rho/theta/zeta contraction, including for N=0 solves.
        grad_g_dot_grad_rho = (
            data["g^rr"] * data["g^rr_r"]
            + data["g^rt"] * data["g^rr_t"]
            + data["g^rz"] * data["g^rr_z"]
        )
        divPsiVec = (
            -2 * data["curvature_H_rho"]
            / (data["chi_r"] * np.sqrt(data["g^rr"]))
            - chi_rr / data["chi_r"] ** 2
            - grad_g_dot_grad_rho
            / (2 * data["chi_r"] * data["g^rr"] ** 2)
        )

        # Transform the DESC contravariant metric to (chi, theta_B, zeta_B).
        A = np.zeros((npsi, ntheta, 3, 3))
        A[..., 0, 0] = data["chi_r"]
        A[..., 1, 0] = data["theta_PEST_r"] + data["iota_r"] * data["nu"] + data["iota"] * data["nu_r"]
        A[..., 1, 1] = data["theta_PEST_t"] + data["iota"] * data["nu_t"]
        A[..., 1, 2] = data["theta_PEST_z"] + data["iota"] * data["nu_z"]
        A[..., 2, 0] = data["phi_r"] + data["nu_r"]
        A[..., 2, 1] = data["phi_t"] + data["nu_t"]
        A[..., 2, 2] = data["phi_z"] + data["nu_z"]

        gdesc = np.zeros_like(A)
        gdesc[..., 0, 0] = data["g^rr"]
        gdesc[..., 0, 1] = gdesc[..., 1, 0] = data["g^rt"]
        gdesc[..., 0, 2] = gdesc[..., 2, 0] = data["g^rz"]
        gdesc[..., 1, 1] = data["g^tt"]
        gdesc[..., 1, 2] = gdesc[..., 2, 1] = data["g^tz"]
        gdesc[..., 2, 2] = data["g^zz"]
        g = np.einsum("...ai,...ij,...bj->...ab", A, gdesc, A)

        theta_B = data["theta_B"]
        fields = [
            "R", "Z", "chi", "chi_r", "q", "G", "I", "|B|", "zeta_B",
            "sqrt(g)_Boozer", "J*B", "p_r", "kappa_n", "kappa_g", "|grad(rho)|",
        ]
        data = {name: _to_boozer(data[name], theta_B, theta) for name in fields}
        shear = _to_boozer(shear, theta_B, theta)
        divPsiVec = _to_boozer(divPsiVec, theta_B, theta)
        theta_desc_2d = np.broadcast_to(theta_desc[None, :], (npsi, ntheta))
        data["theta_DESC"] = _to_boozer(theta_desc_2d, theta_B, theta)
        for i in range(3):
            for j in range(3):
                g[..., i, j] = _to_boozer(g[..., i, j], theta_B, theta)

        q_raw = np.asarray(data["q"][:, 0], dtype=np.float64)
        if (
            not np.all(np.isfinite(q_raw))
            or np.any(np.isclose(q_raw, 0.0))
            or np.any(np.sign(q_raw) != np.sign(q_raw[0]))
        ):
            raise ValueError("DESC Boozer q must be finite, nonzero, and one-signed.")
        orientation = int(np.sign(q_raw[0]))
        data, g, shear, divPsiVec = _reflect_boozer_surface_arrays(
            data,
            g,
            shear,
            divPsiVec,
            orientation,
        )

        chi = data["chi"][:, 0]
        chi_r = data["chi_r"][:, 0]
        q = orientation * data["q"][:, 0]
        G = data["G"][:, 0]
        I = orientation * data["I"][:, 0]
        B = data["|B|"]
        h = q * G + I
        jac = h[:, None] / B**2
        gpp = g[..., 0, 0]

        kappa_Psi = data["chi_r"] * data["|grad(rho)|"] * data["kappa_n"]
        kappa_s = -data["chi_r"] * B * data["|grad(rho)|"] * data["kappa_g"]
        J_diag = data["p_r"] / data["chi_r"] * gpp

        coords = {"psi": x, "theta": theta}
        self.magspace = xr.Dataset(coords=coords)
        self.magspace.attrs["poloidal_coordinate_orientation"] = orientation

        def add(name, values, units):
            dims = ("psi",) if np.ndim(values) == 1 else ("psi", "theta")
            self.magspace[name] = xr.DataArray(values, dims=dims, coords={d: coords[d] for d in dims})
            self.magspace[name].attrs["units"] = units

        add("psi_pol_norm", x, "1")
        add("psi_pol", chi, "Wb/rad")
        add("rho_tor", rho, "1")
        add("dpsi_pol_drho", chi_r, "Wb/rad")
        add("theta_B", np.broadcast_to(theta[None, :], (npsi, ntheta)), "rad")
        add("theta_DESC", data["theta_DESC"], "rad")
        add("zeta_B", data["zeta_B"], "rad")
        add("R", data["R"], "m")
        add("z", data["Z"], "m")
        add("B0", B, "T")
        add("jac", jac, "m**3/Wb")
        add("jac_from_desc_sqrtg", orientation * data["sqrt(g)_Boozer"] / data["chi_r"], "m**3/Wb")

        metric_names = [
            ["gpsi_psi", "gpsi_theta", "gpsi_zeta"],
            ["gtheta_psi", "gtheta_theta", "gtheta_zeta"],
            ["gzeta_psi", "gzeta_theta", "gzeta_zeta"],
        ]
        for i in range(3):
            for j in range(3):
                units = "Wb**2/m**2" if i == j == 0 else "Wb/m**2" if i == 0 or j == 0 else "1/m**2"
                add(metric_names[i][j], g[..., i, j], units)

        add("kappa_Psi", kappa_Psi, "Wb/m**2")
        add("kappa_s", kappa_s, "T**2")
        add("J_B", data["J*B"], "A*T/m**2")
        add("J_diag", J_diag, "A*T**2/m")
        add("shear", shear, "1/m**3")
        add("divPsiVec", divPsiVec, "1/Wb")
        add("q", q, "1")
        add("G", G, "T*m")
        add("F", G, "T*m")
        add("g", G, "T*m")
        add("I", I, "T*m")
        add("h", h, "T*m")
        add("density", np.interp(x, self.density.x, _profile_values(self.density)), "m**-3")
        add("pressure", np.interp(x, self.pressure.x, _profile_values(self.pressure)), "Pa")
        add("vtor", np.interp(x, self.vtor.x, _profile_values(self.vtor)), "m/s")
        add("vpol", np.interp(x, self.vpol.x, _profile_values(self.vpol)), "m/s")
        add("eta", np.interp(x, self.eta.x, _profile_values(self.eta)), "ohm*m")

        self.chi_edge = chi_edge
        self._psi, self._rho = x, rho
        self.qprof = self.qin.copy()
        self.qprof.data *= orientation
        self.Fprof = Profile(x=x, F_nm=G, attrs={"units": "T*m", "name": "F"})
        self.gprof = Profile(x=x, F_nm=G, attrs={"units": "T*m", "name": "g"})
        self.Iprof = Profile(x=x, F_nm=I, attrs={"units": "T*m", "name": "I"})
        self.hprof = Profile(x=x, F_nm=h, attrs={"units": "T*m", "name": "h"})
        x_full = np.unique(np.concatenate(([0.0], x, [1.0])))
        self.hprof.data = self.hprof.data.interp(
            x=x_full,
            kwargs={"fill_value": "extrapolate"},
        )
        self._cache.clear()

        psi0 = xr.DataArray(
            chi_edge * self.qin.x,
            dims="psi",
            coords={"psi": self.qin.x},
            attrs={
                "units": "Wb/rad",
                "psi_axis": 0.0,
                "psi_boundary": chi_edge,
                "normalization": (
                    "psi_N = (psi - psi_axis) / (psi_boundary - psi_axis)"
                ),
            },
        )
        self.boozer = SimpleNamespace(coords=SimpleNamespace(psi0=psi0))
        self.equ = SimpleNamespace(
            desc_equilibrium=self.desc_equilibrium,
            fluxdata=SimpleNamespace(Raxis=self.Rmajor, aminor=self.aminor),
        )
        return self.magspace

    def __call__(self, *args, **kwargs):
        """Use the parent interface and extend interior profiles to x=0 and x=1."""
        output = super().__call__(*args, **kwargs)
        profiles = [output] if isinstance(output, Profile) else output
        for profile in profiles:
            if profile.x[0] > 0 or profile.x[-1] < 1:
                x = np.unique(np.concatenate(([0.0], profile.x, [1.0])))
                profile.data = profile.data.interp(x=x, kwargs={"fill_value": "extrapolate"})
        return output

    def solution_summary(self):
        jac = self.magspace.jac.values
        jac_desc = self.magspace.jac_from_desc_sqrtg.values
        return {
            "kind": "DESCSolver",
            "desc_resolution": {
                "L": int(self.desc_equilibrium.L),
                "M": int(self.desc_equilibrium.M),
                "N": int(self.desc_equilibrium.N),
            },
            "boundary_fit": {
                "M": int(self.boundary_m_max),
                "K": int(self.boundary_k_max),
            },
            "computational_boundary": {
                "rhop_eqdsk": self.rhop_boundary,
                "psipol_norm_eqdsk": self.psipol_boundary,
            },
            "magnetic_grid": {
                "npsi": self.magspace.sizes["psi"],
                "ntheta": self.magspace.sizes["theta"],
                "theta_coordinate": "theta_B",
            },
            "field_and_flux": {
                "Baxis_target_T": self.Baxis_target,
                "Baxis_T": self.Baxis,
                "Psi_tor_total_Wb": self.Psi_tor_total,
                "chi_edge_Wb": self.chi_edge,
            },
            "jacobian_check": {
                "max_relative_error": float(
                    np.max(np.abs(jac_desc - jac) / np.maximum(np.abs(jac), 1e-30))
                )
            },
            "available_magspace_variables": list(self.magspace.data_vars),
        }

    def print_summary(self):
        summary = self.solution_summary()
        print("DESCSolver equilibrium")
        print("----------------------")
        print(
            f"Baxis = {self.Baxis:.6g} T "
            f"(target {self.Baxis_target:.6g} T)"
        )
        print(f"Psi_tor = {self.Psi_tor_total:.6g} Wb, chi_edge = {self.chi_edge:.6g} Wb")
        print(
            f"Boozer grid = ({self.magspace.sizes['psi']}, "
            f"{self.magspace.sizes['theta']})"
        )
        return summary


__all__ = ["DESCSolver", "_rho_tor_from_q", "_dtheta", "_local_magnetic_shear_desc"]
