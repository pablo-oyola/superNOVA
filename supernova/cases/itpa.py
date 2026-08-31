from types import SimpleNamespace

import numpy as np
import unyt
import xarray as xr

from supernova import Profile, superEquilibrium
from supernova.equilibrium import _axisymmetric_cylindrical_local_shear


class ITPAEq(superEquilibrium):
    """
    Analytical circular equilibrium for the ITPA TAE benchmark.

    The solver-facing radial coordinate is normalized poloidal flux ``psi`` in
    ``[0, 1]``. The benchmark geometry is defined in physical minor radius
    ``r``, so ``Psi(r)`` is integrated from the appendix relation and inverted
    onto the normalized ``psi`` grid. The benchmark profile coordinate ``s`` is
    the normalized toroidal flux, computed from ``dPsi_tor = q(r) dPsi_pol``.
    The internal geometry is first built in geometric circular angle and then
    interpolated to a Boozer/straight-field-line poloidal angle.

    ``Bt_sign`` selects the physical toroidal-field direction while ``B0``
    remains the positive magnetic-field normalization magnitude.
    """

    def __init__(
        self,
        aminor: float = 1.0 * unyt.m,
        Rmajor: float = 10.0 * unyt.m,
        q0: float = 1.71,
        qedge: float = 1.87,
        B0: float = 3.0 * unyt.T,
        ns: int = 100,
        ntheta: int = 256,
        m_max: int = 128,
        Bt_sign: int = 1,
    ):
        if ns < 4:
            raise ValueError("ns must be at least 4.")
        if ntheta < 8:
            raise ValueError("ntheta must be at least 8.")

        self.Rmajor = self._as_mks_value(Rmajor, "m")
        self.aminor = self._as_mks_value(aminor, "m")
        self.Baxis = self._as_mks_value(B0, "T")
        if isinstance(Bt_sign, (bool, np.bool_)) or Bt_sign not in (-1, 1):
            raise ValueError("Bt_sign must be either +1 or -1.")
        self.Bt_sign = int(Bt_sign)
        self.q0 = float(q0)
        self.qedge = float(qedge)
        self.m_max = abs(int(m_max))
        self.n_max = 0

        if self.aminor <= 0.0:
            raise ValueError("aminor must be positive.")
        if self.Rmajor <= self.aminor:
            raise ValueError("Rmajor must be larger than aminor for this circular model.")
        if self.Baxis <= 0.0:
            raise ValueError("B0 must be positive.")

        psi, r_minor, psi_edge, s_tor, psitor_edge = self._build_flux_grid(ns)
        theta_geom = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
        theta = theta_geom.copy()

        self._s = s_tor
        self._psi = psi
        self._psi_edge = psi_edge
        self._psitor = s_tor * psitor_edge
        self._psitor_edge = psitor_edge
        self._rminor = r_minor
        self._rho = r_minor / self.aminor
        self._theta_geom = theta_geom
        self._theta = theta
        # With psi increasing outwards and zeta=phi, the raw geometric q has
        # sign -Bt_sign in internal COCOS 1.  Reflect the poloidal coordinate
        # only when needed so the solver-facing q remains positive.
        self._poloidal_orientation = -self.Bt_sign

        q = self._q_of_r(r_minor)
        qbar = self._qbar_of_r(r_minor)
        dpsi_dr = self._dpsi_dr(r_minor)
        d2psi_dr2 = np.gradient(dpsi_dr, r_minor, edge_order=2)

        self.pressure = self._make_profile(
            psi,
            self._pressure_profile(s_tor),
            units="Pa",
            name="pressure",
        )
        self.density = self._make_profile(
            psi,
            self._density_profile(s_tor),
            units="m**-3",
            name="density",
        )
        self.vtor = self._make_profile(psi, np.zeros_like(psi), units="m/s", name="vtor")
        self.vpol = self._make_profile(psi, np.zeros_like(psi), units="m/s", name="vpol")
        self.eta = self._make_profile(psi, np.zeros_like(psi), units="ohm*m", name="eta")

        self._prepare_unit_system()

        self._cache = {}
        self.realspace = xr.Dataset()
        self.magspace = self._build_magspace(
            psi=psi,
            theta=theta_geom,
            r_minor=r_minor,
            s_tor=s_tor,
            q=q,
            qbar=qbar,
            dpsi_dr=dpsi_dr,
            d2psi_dr2=d2psi_dr2,
        )

        F = np.full_like(psi, self.Bt_sign * self.Baxis * self.Rmajor)
        I_raw = self._I_of_r(r_minor, dpsi_dr, theta_geom)
        I = self._poloidal_orientation * I_raw
        h = F * q + I
        self.hprof = self._make_profile(psi, h, units="T*m", name="h")
        self.qprof = self._make_profile(psi, q, units="1", name="q")
        self.Fprof = self._make_profile(psi, F, units="T*m", name="F")
        self.Iprof = self._make_profile(psi, I, units="T*m", name="I")
        self.gprof = self._make_profile(psi, F, units="T*m", name="g")
        self.qnought = qbar

        self._add_flux_profiles_to_magspace()
        self._add_jacobian_diagnostics(r_minor, dpsi_dr)
        self._transform_to_magnetic_theta(psi, theta_geom, theta)
        self._set_compatibility_handles(psi)

    @staticmethod
    def _as_mks_value(value, unit: str) -> float:
        if hasattr(value, "to"):
            return float(value.to(unit).value)
        return float(value)

    @staticmethod
    def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=np.float64)
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
        return out

    @staticmethod
    def _periodic_theta_derivative(values: np.ndarray, theta: np.ndarray) -> np.ndarray:
        dtheta = theta[1] - theta[0]
        return (np.roll(values, -1, axis=1) - np.roll(values, 1, axis=1)) / (2.0 * dtheta)

    @staticmethod
    def _regularize_axis(values: np.ndarray) -> np.ndarray:
        out = np.array(values, copy=True)
        if out.shape[0] > 1:
            out[0] = out[1]
        return out

    def _q_of_r(self, r_minor: np.ndarray) -> np.ndarray:
        rho = r_minor / self.aminor
        return self.q0 + (self.qedge - self.q0) * rho**2

    def _qbar_of_r(self, r_minor: np.ndarray) -> np.ndarray:
        aspect_factor = np.sqrt(np.clip(1.0 - (r_minor / self.Rmajor) ** 2, 0.0, None))
        return self._q_of_r(r_minor) * aspect_factor

    def _dpsi_dr(self, r_minor: np.ndarray) -> np.ndarray:
        # qbar is the positive solver-gauge magnitude.  The COCOS-1 sign of
        # the physical poloidal field is applied separately in _build_magspace.
        qbar = self._qbar_of_r(r_minor)
        return self.Baxis * r_minor / qbar

    def _I_of_r(
        self,
        r_minor: np.ndarray,
        dpsi_dr: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        r"""
        Return the raw geometric-angle Boozer current function.

        For a circular flux surface, ``I = (1 / 2 pi) int B . dl`` reduces to
        ``r <B_theta>_theta``.  Internal COCOS 1 with outward-increasing
        ``Psi`` has ``B_theta = -(dPsi / dr) / R(theta)``, so this raw current
        and raw q are negative.  Both become positive under the solver-facing
        coordinate reflection ``Theta=-theta_raw``.
        """
        r = r_minor[:, None]
        R = self.Rmajor + r * np.cos(theta)[None, :]
        Btheta = -dpsi_dr[:, None] / R
        return r_minor * np.mean(Btheta, axis=1)

    def _current_coordinate_jacobian(
        self,
        r_minor: np.ndarray,
        dpsi_dr: np.ndarray,
        R: np.ndarray,
    ) -> np.ndarray:
        r"""
        Estimate ``sqrt(g)`` for the current geometric coordinates.

        The current coordinates are physical poloidal flux ``Psi``, geometric
        poloidal angle, and the toroidal angle. For circular surfaces,
        the signed triple product is
        ``J = -R r / (dPsi / dr)`` for ``Z=r sin(theta)`` and
        counterclockwise ``zeta=phi``.  The subsequent positive-q reflection
        changes the sign of both ``theta`` and ``J``.
        """
        r = r_minor[:, None]
        psi_prime = dpsi_dr[:, None]
        psi_prime_safe = np.where(psi_prime == 0.0, dpsi_dr[1], psi_prime)
        return self._regularize_axis(-R * r / psi_prime_safe)

    @staticmethod
    def _interp_periodic_rows(
        values: np.ndarray,
        theta_source: np.ndarray,
        theta_target: np.ndarray,
    ) -> np.ndarray:
        theta_ext = np.concatenate([theta_source, [2.0 * np.pi]])
        out = np.empty_like(theta_target, dtype=np.asarray(values).dtype)
        for ipsi, row in enumerate(values):
            row_ext = np.concatenate([row, row[:1]])
            out[ipsi] = np.interp(theta_target[ipsi], theta_ext, row_ext)
        return out

    def _build_magnetic_theta_map(
        self,
        psi: np.ndarray,
        theta_geom: np.ndarray,
        theta_magnetic: np.ndarray,
    ) -> dict[str, np.ndarray | int]:
        jac_current = self.magspace["jac_current"].values
        jac_boozer = self.magspace["jac_boozer"].values
        signed_ratio = np.divide(
            jac_current,
            jac_boozer,
            out=np.ones_like(jac_current),
            where=jac_boozer != 0.0,
        )
        if not np.all(np.isfinite(signed_ratio)):
            raise ValueError("Cannot build magnetic angle from non-finite Jacobian ratio.")
        ratio_sign = np.sign(signed_ratio)
        if np.any(ratio_sign == 0.0) or not np.all(ratio_sign == ratio_sign.flat[0]):
            raise ValueError("Magnetic-angle Jacobian ratio must be nonzero and one-signed.")
        orientation = int(ratio_sign.flat[0])
        if orientation != self._poloidal_orientation:
            raise ValueError(
                "Analytic ITPA poloidal orientation disagrees with its signed Jacobian."
            )
        weight = orientation * signed_ratio
        if np.any(weight <= 0.0):
            raise ValueError("Cannot build magnetic angle from non-positive Jacobian magnitude ratio.")

        theta_ext = np.concatenate([theta_geom, [2.0 * np.pi]])
        weight_ext = np.concatenate([weight, weight[:, :1]], axis=1)
        increments = 0.5 * (weight_ext[:, 1:] + weight_ext[:, :-1]) * np.diff(theta_ext)
        total = np.sum(increments, axis=1)
        if np.any(total <= 0.0):
            raise ValueError("Magnetic-angle map has non-positive poloidal integral.")

        theta_of_geom_ext = np.zeros((weight.shape[0], weight.shape[1] + 1), dtype=np.float64)
        theta_of_geom_ext[:, 1:] = np.cumsum(increments, axis=1)
        theta_of_geom_ext *= (2.0 * np.pi / total)[:, None]
        theta_of_geom_ext *= orientation
        theta_of_geom = theta_of_geom_ext[:, :-1]
        dtheta_dtheta_geom = (
            orientation * weight * (2.0 * np.pi / total)[:, None]
        )

        theta_geom_of_theta = np.empty((weight.shape[0], theta_magnetic.size), dtype=np.float64)
        alpha_of_geom_ext = orientation * theta_of_geom_ext
        alpha_target = np.mod(orientation * theta_magnetic, 2.0 * np.pi)
        for ipsi in range(weight.shape[0]):
            theta_geom_of_theta[ipsi] = np.interp(
                alpha_target,
                alpha_of_geom_ext[ipsi],
                theta_ext,
            )

        psi_phys = psi * self._psi_edge
        dtheta_dpsi_geom = np.gradient(theta_of_geom, psi_phys, axis=0, edge_order=2)
        dtheta_dtheta_geom_on_theta = self._interp_periodic_rows(
            dtheta_dtheta_geom,
            theta_geom,
            theta_geom_of_theta,
        )
        dtheta_dpsi_geom_on_theta = self._interp_periodic_rows(
            dtheta_dpsi_geom,
            theta_geom,
            theta_geom_of_theta,
        )

        return {
            "theta_of_geom": theta_of_geom,
            "theta_geom_of_theta": theta_geom_of_theta,
            "dtheta_dtheta_geom": dtheta_dtheta_geom_on_theta,
            "dtheta_dpsi_geom": dtheta_dpsi_geom_on_theta,
            "coordinate_orientation": orientation,
        }

    @staticmethod
    def _replace_attrs(description: str, units: str) -> dict[str, str]:
        return {"description": description, "units": units}

    def _transform_metric_to_magnetic_theta(
        self,
        values_on_theta: dict[str, np.ndarray],
        dtheta_dpsi_geom: np.ndarray,
        dtheta_dtheta_geom: np.ndarray,
    ) -> dict[str, np.ndarray]:
        gpp = values_on_theta["gpsi_psi"]
        gpt = values_on_theta["gpsi_theta"]
        gpz = values_on_theta["gpsi_zeta"]
        gtp = values_on_theta["gtheta_psi"]
        gtt = values_on_theta["gtheta_theta"]
        gtz = values_on_theta["gtheta_zeta"]
        gzp = values_on_theta["gzeta_psi"]
        gzt = values_on_theta["gzeta_theta"]
        gzz = values_on_theta["gzeta_zeta"]

        dpsi = dtheta_dpsi_geom
        dtheta = dtheta_dtheta_geom

        return {
            "gpsi_psi": gpp,
            "gpsi_theta": dpsi * gpp + dtheta * gpt,
            "gpsi_zeta": gpz,
            "gtheta_psi": dpsi * gpp + dtheta * gtp,
            "gtheta_theta": dpsi**2 * gpp + dpsi * dtheta * (gpt + gtp) + dtheta**2 * gtt,
            "gtheta_zeta": dpsi * gpz + dtheta * gtz,
            "gzeta_psi": gzp,
            "gzeta_theta": dpsi * gzp + dtheta * gzt,
            "gzeta_zeta": gzz,
        }

    def _transform_to_magnetic_theta(
        self,
        psi: np.ndarray,
        theta_geom: np.ndarray,
        theta_magnetic: np.ndarray,
    ):
        mapping = self._build_magnetic_theta_map(psi, theta_geom, theta_magnetic)
        theta_geom_of_theta = mapping["theta_geom_of_theta"]
        dtheta_dtheta_geom = mapping["dtheta_dtheta_geom"]
        dtheta_dpsi_geom = mapping["dtheta_dpsi_geom"]
        coordinate_orientation = int(mapping["coordinate_orientation"])

        coords = {"psi": psi, "theta": theta_magnetic}
        values_on_theta: dict[str, np.ndarray] = {}
        attrs: dict[str, dict] = {}

        for name, data_array in self.magspace.data_vars.items():
            attrs[name] = dict(data_array.attrs)
            if data_array.dims == ("psi", "theta"):
                values_on_theta[name] = self._interp_periodic_rows(
                    data_array.values,
                    theta_geom,
                    theta_geom_of_theta,
                )
            elif data_array.dims == ("psi",):
                values_on_theta[name] = np.asarray(data_array.values)

        metric_values = self._transform_metric_to_magnetic_theta(
            values_on_theta,
            dtheta_dpsi_geom,
            dtheta_dtheta_geom,
        )
        values_on_theta.update(metric_values)
        for metric_name in metric_values:
            attrs[metric_name] = self._replace_attrs(
                attrs[metric_name].get("description", metric_name)
                + " Transformed to magnetic poloidal angle.",
                attrs[metric_name].get("units", "1"),
            )

        jac_current = values_on_theta["jac_current"]
        jac_boozer = values_on_theta["h"][:, None] / values_on_theta["B0"]**2
        jac_magnetic = jac_current / dtheta_dtheta_geom
        # jac_current belongs to the raw geometric angle, while jac_boozer is
        # in the positive-q solver gauge.  Align their coordinate orientation
        # before diagnosing the pre-remap magnitude error.
        geom_error = coordinate_orientation * jac_current - jac_boozer
        geom_rel_error = np.divide(
            geom_error,
            jac_boozer,
            out=np.zeros_like(geom_error),
            where=jac_boozer != 0.0,
        )
        magnetic_error = jac_magnetic - jac_boozer
        magnetic_rel_error = np.divide(
            magnetic_error,
            jac_boozer,
            out=np.zeros_like(magnetic_error),
            where=jac_boozer != 0.0,
        )

        values_on_theta["jac_current"] = jac_current
        values_on_theta["jac_boozer"] = jac_boozer
        values_on_theta["jac_magnetic"] = jac_magnetic
        values_on_theta["jac_geometric_boozer_error"] = geom_error
        values_on_theta["jac_geometric_boozer_rel_error"] = geom_rel_error
        values_on_theta["jac_boozer_error"] = magnetic_error
        values_on_theta["jac_boozer_rel_error"] = magnetic_rel_error
        values_on_theta["theta_geom"] = theta_geom_of_theta
        values_on_theta["theta_boozer"] = np.broadcast_to(
            theta_magnetic[None, :],
            theta_geom_of_theta.shape,
        )
        values_on_theta["dtheta_boozer_dtheta_geom"] = dtheta_dtheta_geom
        values_on_theta["dtheta_boozer_dpsi_geom"] = dtheta_dpsi_geom

        attrs.update(
            {
                "jac_current": self._replace_attrs(
                    "Signed raw geometric-coordinate Jacobian sampled on the magnetic-angle grid.",
                    "m**3/Wb",
                ),
                "jac_boozer": self._replace_attrs(
                    "Boozer Jacobian implied by h / |B|**2 on the magnetic-angle grid.",
                    "m**3/Wb",
                ),
                "jac_magnetic": self._replace_attrs(
                    "Jacobian after transforming to magnetic poloidal angle.",
                    "m**3/Wb",
                ),
                "jac_geometric_boozer_error": self._replace_attrs(
                    "Orientation-aligned geometric-coordinate Jacobian minus h / |B|**2.",
                    "m**3/Wb",
                ),
                "jac_geometric_boozer_rel_error": self._replace_attrs(
                    "Relative geometric-coordinate Jacobian error.",
                    "1",
                ),
                "jac_boozer_error": self._replace_attrs(
                    "Magnetic-coordinate Jacobian minus h / |B|**2.",
                    "m**3/Wb",
                ),
                "jac_boozer_rel_error": self._replace_attrs(
                    "Relative magnetic-coordinate Jacobian error.",
                    "1",
                ),
                "theta_geom": self._replace_attrs(
                    "Geometric poloidal angle corresponding to the magnetic-angle grid.",
                    "rad",
                ),
                "theta_boozer": self._replace_attrs(
                    "Boozer/straight-field-line poloidal angle.",
                    "rad",
                ),
                "dtheta_boozer_dtheta_geom": self._replace_attrs(
                    "Derivative of Boozer angle with respect to geometric angle.",
                    "1",
                ),
                "dtheta_boozer_dpsi_geom": self._replace_attrs(
                    "Derivative of Boozer angle with respect to physical poloidal flux at fixed geometric angle.",
                    "1/Wb",
                ),
            }
        )

        data_vars = {}
        for name, values in values_on_theta.items():
            if np.ndim(values) == 1:
                data_vars[name] = xr.DataArray(
                    values,
                    coords={"psi": psi},
                    dims=("psi",),
                    attrs=attrs.get(name, {}),
                )
            else:
                data_vars[name] = xr.DataArray(
                    values,
                    coords=coords,
                    dims=("psi", "theta"),
                    attrs=attrs.get(name, {}),
                )

        self.magspace = xr.Dataset(data_vars)
        self.magspace.attrs.update(
            {
                "poloidal_coordinate_orientation": coordinate_orientation,
                "coordinate_convention": (
                    "Internal COCOS 1; solver Theta is the positive-q reflection "
                    "of the raw geometric circular angle."
                ),
            }
        )
        self._theta = theta_magnetic
        self._rmajor = self.magspace["R"]
        self._z = self.magspace["z"]
        self._B_R = self.magspace["BR"]
        self._B_phi = self.magspace["Bphi"]
        self._B_z = self.magspace["Bz"]

    def _build_flux_grid(self, ns: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
        r_int = np.linspace(0.0, self.aminor, max(8 * ns, 2048))
        dpsi_dr = self._dpsi_dr(r_int)
        q_int = self._q_of_r(r_int)
        psi_phys = self._cumtrapz(dpsi_dr, r_int)
        psitor_phys = self._cumtrapz(q_int * dpsi_dr, r_int)
        psi_edge = float(psi_phys[-1])
        psitor_edge = float(psitor_phys[-1])
        if psi_edge <= 0.0:
            raise ValueError("Integrated edge flux must be positive.")
        if psitor_edge <= 0.0:
            raise ValueError("Integrated edge toroidal flux must be positive.")

        psi_norm_int = psi_phys / psi_edge
        s_tor_int = psitor_phys / psitor_edge
        psi = np.linspace(0.0, 1.0, ns)
        r_minor = np.interp(psi, psi_norm_int, r_int)
        s_tor = np.interp(psi, psi_norm_int, s_tor_int)
        s_tor[0] = 0.0
        s_tor[-1] = 1.0
        return psi, r_minor, psi_edge, s_tor, psitor_edge

    def _pressure_profile(self, s_tor: np.ndarray) -> np.ndarray:
        pressure = 7.17e3 - 6.811e3 * s_tor - 3.585e2 * s_tor**2
        return np.maximum(pressure, 0.0)

    def _density_profile(self, s_tor: np.ndarray) -> np.ndarray:
        # n0 = 1.44131e19
        # c0 = 0.49123
        # c1 = 0.298228
        # c2 = 0.198739
        # c3 = 0.521298
        # return n0 * c3 * np.exp(-(c2 / c1) * np.tanh((np.sqrt(s_tor) - c0) / c2))
        return np.ones_like(s_tor) * 2.0e19

    def _make_profile(self, psi: np.ndarray, values: np.ndarray, units: str, name: str) -> Profile:
        return Profile(
            x=psi,
            F_nm=np.asarray(values, dtype=np.float64),
            attrs={"units": units, "name": name},
        )

    def _build_magspace(
        self,
        psi: np.ndarray,
        theta: np.ndarray,
        r_minor: np.ndarray,
        s_tor: np.ndarray,
        q: np.ndarray,
        qbar: np.ndarray,
        dpsi_dr: np.ndarray,
        d2psi_dr2: np.ndarray,
    ) -> xr.Dataset:
        sin_theta = np.sin(theta)[None, :]
        cos_theta = np.cos(theta)[None, :]
        r = r_minor[:, None]
        r_safe = np.where(r == 0.0, r_minor[1], r)

        R = self.Rmajor + r * cos_theta
        Z = r * sin_theta
        F0 = self.Bt_sign * self.Baxis * self.Rmajor

        psi_prime = dpsi_dr[:, None]
        psi_prime_safe = np.where(psi_prime == 0.0, dpsi_dr[1], psi_prime)
        psi_second = d2psi_dr2[:, None]
        # Internal COCOS 1: B=grad(phi) x grad(Psi) + F grad(phi).
        Btheta = -psi_prime / R
        Bphi = F0 / R
        BR = -Btheta * sin_theta
        BZ = Btheta * cos_theta
        Bmag = np.sqrt(Btheta**2 + Bphi**2)

        btheta = Btheta / Bmag
        bphi = Bphi / Bmag
        dbtheta_dtheta = self._periodic_theta_derivative(btheta, theta)
        dbphi_dtheta = self._periodic_theta_derivative(bphi, theta)

        kappa_r = -(btheta**2) / r_safe - (bphi**2) * cos_theta / R
        kappa_theta = (btheta / r_safe) * dbtheta_dtheta + (bphi**2) * sin_theta / R
        kappa_phi = (btheta / r_safe) * dbphi_dtheta - (bphi * btheta) * sin_theta / R

        kappa_Psi = kappa_r * psi_prime
        kappa_s = psi_prime * (- kappa_theta * Bphi + kappa_phi * Btheta)

        r_btheta = r * Btheta
        d_r_btheta_dr = np.gradient(r_btheta, r_minor, axis=0, edge_order=2)
        mu0 = unyt.mu_0.to("N/A**2").value
        jR = np.zeros_like(R)
        jphi = -d_r_btheta_dr / (mu0 * r_safe)
        jZ = np.zeros_like(R)
        sR = Bphi * psi_prime * sin_theta
        sphi = psi_prime * Btheta
        sZ = -Bphi * psi_prime * cos_theta
        J_B = jR * BR + jphi * Bphi + jZ * BZ
        J_diag = jR * sR + jphi * sphi + jZ * sZ

        A = np.array(
            [
                Bphi / psi_prime_safe * sin_theta,
                Btheta / psi_prime_safe,
                -Bphi / psi_prime_safe * cos_theta,
            ]
        )

        def d_dR(values):
            d_dr = np.gradient(values, r_minor, axis=0, edge_order=2)
            d_dtheta = self._periodic_theta_derivative(values, theta)
            return cos_theta * d_dr - sin_theta / r_safe * d_dtheta

        def d_dZ(values):
            d_dr = np.gradient(values, r_minor, axis=0, edge_order=2)
            d_dtheta = self._periodic_theta_derivative(values, theta)
            return sin_theta * d_dr + cos_theta / r_safe * d_dtheta

        shear = _axisymmetric_cylindrical_local_shear(A, R, d_dR, d_dZ)

        d_rR_over_psi_prime_dr = (
            (R + r * cos_theta) * psi_prime_safe - r * R * psi_second
        ) / psi_prime_safe**2
        divPsiVec = d_rR_over_psi_prime_dr / (r_safe * R)

        gpsi_psi = np.broadcast_to(psi_prime**2, R.shape)
        gtheta_theta = np.broadcast_to(1.0 / r_safe**2, R.shape)
        gzeta_zeta = 1.0 / R**2
        zeros = np.zeros_like(R)

        gpsi_psi = self._regularize_axis(gpsi_psi)
        gtheta_theta = self._regularize_axis(gtheta_theta)
        divPsiVec = self._regularize_axis(divPsiVec)
        shear = self._regularize_axis(shear)
        kappa_Psi = self._regularize_axis(kappa_Psi)
        kappa_s = self._regularize_axis(kappa_s)
        J_B = self._regularize_axis(J_B)
        J_diag = self._regularize_axis(J_diag)

        coords = {"psi": psi, "theta": theta}

        def da(values, units: str, description: str) -> xr.DataArray:
            return xr.DataArray(
                values,
                coords=coords,
                dims=("psi", "theta"),
                attrs={"units": units, "description": description},
            )

        self._rmajor = da(R, "m", "Cylindrical major radius R(psi, theta).")
        self._z = da(Z, "m", "Vertical coordinate z(psi, theta).")
        self._B_R = da(BR, "T", "Radial cylindrical magnetic-field component.")
        self._B_phi = da(Bphi, "T", "Toroidal magnetic-field component.")
        self._B_z = da(BZ, "T", "Vertical magnetic-field component.")

        magspace = xr.Dataset(
            {
                "R": self._rmajor,
                "z": self._z,
                "B0": da(Bmag, "T", "Norm of the equilibrium magnetic field."),
                "BR": self._B_R,
                "Bphi": self._B_phi,
                "Bz": self._B_z,
                "s": xr.DataArray(
                    s_tor,
                    coords={"psi": psi},
                    dims=("psi",),
                    attrs={
                        "units": "1",
                        "description": "Normalized toroidal flux used by benchmark profiles.",
                    },
                ),
                "gpsi_psi": da(gpsi_psi, "Wb**2/m**2", "Contravariant metric g^{psi psi}."),
                "gpsi_theta": da(zeros, "Wb/m**2", "Contravariant metric g^{psi theta}."),
                "gpsi_zeta": da(zeros, "Wb/m**2", "Contravariant metric g^{psi zeta}."),
                "gtheta_psi": da(zeros, "Wb/m**2", "Contravariant metric g^{theta psi}."),
                "gtheta_theta": da(gtheta_theta, "1/m**2", "Contravariant metric g^{theta theta}."),
                "gtheta_zeta": da(zeros, "1/m**2", "Contravariant metric g^{theta zeta}."),
                "gzeta_psi": da(zeros, "Wb/m**2", "Contravariant metric g^{zeta psi}."),
                "gzeta_theta": da(zeros, "1/m**2", "Contravariant metric g^{zeta theta}."),
                "gzeta_zeta": da(gzeta_zeta, "1/m**2", "Contravariant metric g^{zeta zeta}."),
                "kappa_Psi": da(kappa_Psi, "Wb/m**2", "Curvature projected along grad(Psi)."),
                "kappa_s": da(kappa_s, "T**2", "Curvature projected along B cross grad(Psi)."),
                "J_B": da(J_B, "A*T/m**2", "Parallel-current projection J dot B."),
                "J_diag": da(J_diag, "A*T**2/m", "Diamagnetic-current projection J dot s."),
                "shear": da(shear, "1/m**3", "Local shear s_N dot curl(s_N)."),
                "divPsiVec": da(divPsiVec, "1/Wb", "div(grad(Psi) / |grad(Psi)|**2)."),
                "qbar": xr.DataArray(
                    qbar,
                    coords={"psi": psi},
                    dims=("psi",),
                    attrs={"units": "1", "description": "Appendix qbar profile."},
                ),
            }
        )
        return magspace

    def _add_flux_profiles_to_magspace(self):
        profiles = [
            self.hprof,
            self.qprof,
            self.Fprof,
            self.Iprof,
            self.gprof,
            self.density,
            self.pressure,
            self.vtor,
            self.vpol,
            self.eta,
        ]
        names = ["h", "q", "F", "I", "g", "density", "pressure", "vtor", "vpol", "eta"]
        descriptions = [
            "h profile",
            "q profile",
            "F profile",
            "I profile",
            "g profile",
            "Density profile",
            "Pressure profile",
            "Toroidal flow velocity profile",
            "Poloidal flow velocity profile",
            "Resistivity profile",
        ]

        for name, profile, description in zip(names, profiles, descriptions):
            data = profile.data.sel(mpol=0, ntor=0).real
            self.magspace[name] = xr.DataArray(
                data.values,
                coords={"psi": data.x.values},
                dims=("psi",),
                attrs={
                    "units": profile.data.attrs.get("units", "dimensionless"),
                    "description": description,
                },
            )
    def _add_jacobian_diagnostics(self, r_minor: np.ndarray, dpsi_dr: np.ndarray):
        jac_current = self._current_coordinate_jacobian(
            r_minor,
            dpsi_dr,
            self.magspace["R"].values,
        )
        jac_boozer = self.magspace["h"].values[:, None] / self.magspace["B0"].values**2
        jac_error = self._poloidal_orientation * jac_current - jac_boozer
        jac_rel_error = np.divide(
            jac_error,
            jac_boozer,
            out=np.zeros_like(jac_error),
            where=jac_boozer != 0.0,
        )

        coords = {
            "psi": self.magspace["R"].coords["psi"].values,
            "theta": self.magspace["R"].coords["theta"].values,
        }

        def da(values, units: str, description: str) -> xr.DataArray:
            return xr.DataArray(
                values,
                coords=coords,
                dims=("psi", "theta"),
                attrs={"units": units, "description": description},
            )

        self.magspace["jac_current"] = da(
            jac_current,
            "m**3/Wb",
            "Signed coordinate Jacobian for the raw geometric coordinates.",
        )
        self.magspace["jac_boozer"] = da(
            jac_boozer,
            "m**3/Wb",
            "Boozer Jacobian implied by h / |B|**2.",
        )
        self.magspace["jac_boozer_error"] = da(
            jac_error,
            "m**3/Wb",
            "Orientation-aligned geometric Jacobian minus h / |B|**2.",
        )
        self.magspace["jac_boozer_rel_error"] = da(
            jac_rel_error,
            "1",
            "Relative orientation-aligned geometric Boozer-Jacobian error.",
        )

    def _relative_error_summary(self, name: str, skip_axis: bool = True) -> dict[str, float]:
        rel_error = self.magspace[name].values
        mask = np.isfinite(rel_error)
        if skip_axis and mask.shape[0] > 1:
            mask[0, :] = False
        values = rel_error[mask]
        if values.size == 0:
            return {
                "max_abs_rel_error": np.nan,
                "rms_rel_error": np.nan,
                "mean_rel_error": np.nan,
            }
        return {
            "max_abs_rel_error": float(np.max(np.abs(values))),
            "rms_rel_error": float(np.sqrt(np.mean(values**2))),
            "mean_rel_error": float(np.mean(values)),
        }

    def jacobian_error_summary(self, skip_axis: bool = True) -> dict[str, float]:
        """
        Return the residual Boozer-Jacobian mismatch after the magnetic transform.
        """
        return self._relative_error_summary("jac_boozer_rel_error", skip_axis=skip_axis)

    def geometric_jacobian_error_summary(self, skip_axis: bool = True) -> dict[str, float]:
        """
        Return the original geometric-coordinate Boozer-Jacobian mismatch.
        """
        return self._relative_error_summary(
            "jac_geometric_boozer_rel_error",
            skip_axis=skip_axis,
        )

    def plot_jacobian_diagnostics(self, cmap: str = "coolwarm"):
        """
        Plot geometric, magnetic, and Boozer Jacobian estimates.

        Returns
        -------
        fig, axes
            Matplotlib figure and the 2x2 axes array.
        """
        import matplotlib.pyplot as plt

        R = self.magspace["R"].values
        Z = self.magspace["z"].values
        jac_current = self.magspace["jac_current"].values
        jac_magnetic = self.magspace["jac_magnetic"].values
        jac_boozer = self.magspace["jac_boozer"].values
        rel_error = self.magspace["jac_boozer_rel_error"].values

        fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)
        panels = [
            (jac_current, "geometric-coordinate Jacobian", "m^3/Wb", "viridis"),
            (jac_magnetic, "magnetic-coordinate Jacobian", "m^3/Wb", "viridis"),
            (jac_boozer, "Boozer h / |B|^2", "m^3/Wb", "viridis"),
            (rel_error, "magnetic-coordinate relative error", "1", cmap),
        ]
        for ax, (values, title, label, panel_cmap) in zip(axes.ravel(), panels):
            pcm = ax.tricontourf(
                R[1:].ravel(),
                Z[1:].ravel(),
                values[1:].ravel(),
                levels=32,
                cmap=panel_cmap,
            )
            fig.colorbar(pcm, ax=ax, label=label)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("R [m]")
            ax.set_ylabel("z [m]")
            ax.set_title(title)
        return fig, axes

    def plot_summary(self, n_surfaces: int = 9, cmap: str = "viridis"):
        """
        Plot the main analytic-equilibrium profiles and magnetic geometry.

        Returns
        -------
        fig, axes
            Matplotlib figure and the 2x3 axes array.
        """
        import matplotlib.pyplot as plt

        rho = self._rho
        theta = self._theta
        R = self.magspace["R"].values
        Z = self.magspace["z"].values
        Bmag = self.magspace["B0"].values
        Bphi = self.magspace["Bphi"].values
        Bpol = np.hypot(self.magspace["BR"].values, self.magspace["Bz"].values)
        F = self.magspace["F"].values
        I = self.magspace["I"].values
        h = self.magspace["h"].values

        def close_theta(values):
            return np.concatenate([values, values[:, :1]], axis=1)

        theta0 = 0
        theta90 = theta.size // 4
        theta180 = theta.size // 2
        theta270 = (3 * theta.size) // 4

        fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.5), constrained_layout=True)

        ax = axes[0, 0]
        q = self.qprof.data.sel(mpol=0, ntor=0).real.values
        ax.plot(rho, q, label="q")
        ax.plot(rho, self.magspace["qbar"].values, "--", label="qbar")
        ax.set_xlabel("r / a")
        ax.set_ylabel("safety factor")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[0, 1]
        pressure = self.pressure.data.sel(mpol=0, ntor=0).real.values
        density = self.density.data.sel(mpol=0, ntor=0).real.values
        ax.plot(rho, pressure / 1.0e3, color="tab:red", label="pressure")
        ax.set_xlabel("r / a")
        ax.set_ylabel("pressure [kPa]", color="tab:red")
        ax.tick_params(axis="y", labelcolor="tab:red")
        ax.grid(True, alpha=0.3)
        ax_density = ax.twinx()
        ax_density.plot(rho, density / 1.0e17, color="tab:green", label="density")
        ax_density.set_ylabel("density [1e17 m^-3]", color="tab:green")
        ax_density.tick_params(axis="y", labelcolor="tab:green")

        ax = axes[0, 2]
        pcm = ax.tricontourf(
            R[1:].ravel(),
            Z[1:].ravel(),
            Bmag[1:].ravel(),
            levels=32,
            cmap=cmap,
        )
        fig.colorbar(pcm, ax=ax, label="|B| [T]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("z [m]")
        ax.set_title("magnetic-field strength")

        ax = axes[1, 0]
        surface_ids = np.unique(
            np.linspace(0, R.shape[0] - 1, min(n_surfaces, R.shape[0]), dtype=int)
        )
        spoke_ids = [theta0, theta90, theta180, theta270]
        for ii in surface_ids:
            ax.plot(close_theta(R[ii : ii + 1])[0], close_theta(Z[ii : ii + 1])[0], color="0.35", lw=0.9)
        for jj in spoke_ids:
            ax.plot(R[:, jj], Z[:, jj], color="tab:orange", lw=0.8, alpha=0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("z [m]")
        ax.set_title("magnetic-coordinate grid")

        ax = axes[1, 1]
        ax.plot(rho, Bphi[:, theta0], label="B_phi, outboard")
        ax.plot(rho, Bphi[:, theta180], "--", label="B_phi, inboard")
        ax.plot(rho, Bpol[:, theta0], label="B_pol, outboard")
        ax.plot(rho, Bpol[:, theta180], ":", label="B_pol, inboard")
        ax.set_xlabel("r / a")
        ax.set_ylabel("magnetic field [T]")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1, 2]
        ax.plot(rho, F, label="F = B0 R0")
        ax.plot(rho, I, label="I")
        ax.plot(rho, h, label="h = F q + I")
        ax.set_xlabel("r / a")
        ax.set_ylabel("T m")
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig, axes

    def _set_compatibility_handles(self, psi: np.ndarray):
        psi0 = xr.DataArray(
            psi * self._psi_edge,
            coords={"psi": psi},
            dims=("psi",),
            attrs={
                "units": "Wb/rad",
                "psi_axis": 0.0,
                "psi_boundary": self._psi_edge,
                "normalization": (
                    "psi_N = (psi - psi_axis) / (psi_boundary - psi_axis)"
                ),
            },
        )
        self.boozer = SimpleNamespace(coords=SimpleNamespace(psi0=psi0))
        self.equ = SimpleNamespace(
            fluxdata=SimpleNamespace(Raxis=self.Rmajor, aminor=self.aminor),
            Bdata=SimpleNamespace(Baxis=xr.DataArray(self.Baxis)),
        )


def plot_itpa_equilibrium(equilibrium: ITPAEq | None = None, **kwargs):
    """
    Convenience plotting routine for the analytic ITPA benchmark equilibrium.
    """
    if equilibrium is None:
        equilibrium = ITPAEq()
    return equilibrium.plot_summary(**kwargs)
