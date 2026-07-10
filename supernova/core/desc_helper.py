"""DESC-backed equilibrium solver for superNOVA."""

from types import SimpleNamespace
from typing import Optional

import numpy as np
import xarray as xr
from scipy.constants import mu_0
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline

from ..radial.profile import Profile, decompose2d
from .desc_loader import *
from .equilibrium import superEquilibrium


def _profile_values(profile: Profile):
    return np.asarray(
        profile.data.sel(mpol=0, ntor=0).values.real,
        dtype=np.float64,
    )


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


def _fit_lcfs(R, Z, theta, mmax=10):
    """Fit the LCFS using DESC's clockwise poloidal angle."""
    theta_desc = np.mod(-theta, 2 * np.pi)
    order = np.argsort(theta_desc)
    theta_desc = theta_desc[order]
    Rcos, Rsin = decompose2d(R[order][None, :], m_max=mmax, angle=theta_desc)
    Zcos, Zsin = decompose2d(Z[order][None, :], m_max=mmax, angle=theta_desc)

    R_lmn = np.concatenate((Rcos[0], Rsin[0, 1:]))
    Z_lmn = np.concatenate((Zcos[0], Zsin[0, 1:]))
    modes = [(m, 0) for m in range(mmax + 1)]
    modes += [(-m, 0) for m in range(1, mmax + 1)]
    return FourierRZToroidalSurface(
        R_lmn=R_lmn,
        Z_lmn=Z_lmn,
        modes_R=modes,
        modes_Z=modes,
        NFP=1,
        sym=False,
        M=mmax,
        N=0,
    )


def _make_guesses(equ, psipol, rhotor):
    """Use a few pyCOCOS surfaces as the DESC initial guess."""
    Raxis = float(equ.fluxdata.attrs.get("Raxis", equ.R_axis))
    Zaxis = float(equ.fluxdata.attrs.get("zaxis", equ.z_axis))
    nodes, R_values, Z_values = [], [], []

    for x, rho in zip(psipol, rhotor):
        R, Z = equ.flux_surface(np.sqrt(x))
        R, Z = np.asarray(R), np.asarray(Z)
        if np.allclose([R[0], Z[0]], [R[-1], Z[-1]]):
            R, Z = R[:-1], Z[:-1]
        theta = np.mod(-np.unwrap(np.arctan2(Z - Zaxis, R - Raxis)), 2 * np.pi)
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
        self.lcfs_fit = _fit_lcfs(
            self.Rlcfs,
            self.Zlcfs,
            self.theta_lcfs,
            boundary_m_max,
        )
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
        self.solve_kwargs = solve_kwargs or {}
        self.m_max = abs(m_max)
        self.n_max = 0
        self.realspace = xr.Dataset()
        self._cache = {}

    @classmethod
    def from_eqdsk(cls, equ, density: Profile, use_surface_guess=True, **kwargs):
        """Build the same solver from an EQDSK object."""
        if "q" not in equ.gs_profs or "pres" not in equ.gs_profs:
            raise ValueError("The EQDSK must contain q and pressure profiles.")

        # EQDSK profiles are tabulated on rho_pol. Profile.x is normalized
        # poloidal flux, so x = rho_pol**2 before the q-based rho_tor mapping.
        q_eqdsk = equ.gs_profs["q"]
        pressure_eqdsk = equ.gs_profs["pres"]
        q_psipol = np.asarray(q_eqdsk.coords["rhop"], dtype=np.float64) ** 2
        pressure_psipol = np.asarray(
            pressure_eqdsk.coords["rhop"], dtype=np.float64
        ) ** 2
        x = density.x
        q = Profile(
            x=x,
            F_nm=np.interp(x, q_psipol, q_eqdsk.values),
            attrs={"name": "q", "units": "1"},
        )
        pressure = Profile(
            x=x,
            F_nm=np.interp(x, pressure_psipol, pressure_eqdsk.values),
            attrs={"name": "pressure", "units": "Pa"},
        )
        R = np.asarray(equ.boundary.R)
        Z = np.asarray(equ.boundary.z)
        Raxis = float(equ.fluxdata.attrs.get("Raxis", equ.R_axis))
        Zaxis = float(equ.fluxdata.attrs.get("zaxis", equ.z_axis))
        theta = np.unwrap(np.arctan2(Z - Zaxis, R - Raxis))
        return cls(
            {"R": R, "Z": Z, "theta": theta},
            float(equ.Bdata.Baxis),
            q,
            density,
            pressure,
            initial_guess=equ if use_surface_guess else None,
            **kwargs,
        )

    def solve(
        self,
        initial_guess=None,
        psi_tor_guess=None,
        max_psi_iterations=4,
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
            idx = np.unique(np.linspace(1, self.qin.x.size - 1, min(16, self.qin.x.size - 1), dtype=int))
            guess = _make_guesses(guess, self.qin.x[idx], self.rho_tor_input[idx])

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
                eq.set_initial_guess(*guess)
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
            guess = None
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
        raw = self.desc_equilibrium.compute(
            names + ["B", "grad(rho)", "(B*grad) grad(rho)"], grid=grid
        )
        data = {name: _reshape(raw[name], npsi, ntheta) for name in names}
        data["nu_r"] = CubicSpline(rho, data["nu"], axis=0).derivative()(rho)

        B_vector = np.asarray(raw["B"]).reshape(npsi, ntheta, 3)
        grad_rho = np.asarray(raw["grad(rho)"]).reshape(npsi, ntheta, 3)
        B_grad_grad_rho = np.asarray(raw["(B*grad) grad(rho)"]).reshape(
            npsi, ntheta, 3
        )
        B_cross_grad_rho = np.cross(B_vector, grad_rho)
        torsion_numerator = np.sum(
            B_cross_grad_rho * B_grad_grad_rho, axis=-1
        )
        shear = (
            mu_0 * data["J*B"] / data["g^rr"]
            - 2 * torsion_numerator / data["g^rr"] ** 2
        ) / data["chi_r"] ** 2

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

        chi = data["chi"][:, 0]
        chi_r = data["chi_r"][:, 0]
        q = data["q"][:, 0]
        G = data["G"][:, 0]
        I_boozer = data["I"][:, 0]
        B = data["|B|"]
        h = q * G + I_boozer
        jac = h[:, None] / B**2
        gpp = g[..., 0, 0]

        kappa_Psi = data["chi_r"] * data["|grad(rho)|"] * data["kappa_n"]
        kappa_s = -data["chi_r"] * B * data["|grad(rho)|"] * data["kappa_g"]
        J_diag = data["p_r"] / data["chi_r"] * gpp

        coords = {"psi": x, "theta": theta}
        self.magspace = xr.Dataset(coords=coords)

        def add(name, values, units):
            dims = ("psi",) if np.ndim(values) == 1 else ("psi", "theta")
            self.magspace[name] = xr.DataArray(values, dims=dims, coords={d: coords[d] for d in dims})
            self.magspace[name].attrs["units"] = units

        add("psi_pol_norm", x, "1")
        add("psi_pol", chi, "Wb")
        add("rho_tor", rho, "1")
        add("dpsi_pol_drho", chi_r, "Wb")
        add("theta_B", np.broadcast_to(theta[None, :], (npsi, ntheta)), "rad")
        add("theta_DESC", data["theta_DESC"], "rad")
        add("zeta_B", data["zeta_B"], "rad")
        add("R", data["R"], "m")
        add("z", data["Z"], "m")
        add("B0", B, "T")
        add("jac", jac, "m**3/Wb")
        add("jac_from_desc_sqrtg", data["sqrt(g)_Boozer"] / data["chi_r"], "m**3/Wb")

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
        add("I_boozer", I_boozer, "T*m")
        add("I", 2 * np.pi * I_boozer, "T*m")
        add("h", h, "T*m")
        add("density", np.interp(x, self.density.x, _profile_values(self.density)), "m**-3")
        add("pressure", np.interp(x, self.pressure.x, _profile_values(self.pressure)), "Pa")
        add("vtor", np.interp(x, self.vtor.x, _profile_values(self.vtor)), "m/s")
        add("vpol", np.interp(x, self.vpol.x, _profile_values(self.vpol)), "m/s")
        add("eta", np.interp(x, self.eta.x, _profile_values(self.eta)), "ohm*m")

        self.chi_edge = chi_edge
        self._psi, self._rho = x, rho
        self.qprof = self.qin.copy()
        self.Fprof = Profile(x=x, F_nm=G, attrs={"units": "T*m", "name": "F"})
        self.gprof = Profile(x=x, F_nm=G, attrs={"units": "T*m", "name": "g"})
        self.Iprof = Profile(x=x, F_nm=2 * np.pi * I_boozer, attrs={"units": "T*m", "name": "I"})
        self.hprof = Profile(x=x, F_nm=h, attrs={"units": "T*m", "name": "h"})
        self._cache.clear()

        psi0 = xr.DataArray(chi_edge * self.qin.x, dims="psi", coords={"psi": self.qin.x})
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


__all__ = ["DESCSolver", "_rho_tor_from_q", "_dtheta"]
