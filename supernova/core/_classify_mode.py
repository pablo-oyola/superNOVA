"""
Classification of modes in the continuum.
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import xarray as xr

from .._utils import njit
from ._continuum_kernel import fill_block_kernel


@njit(cache=True, nogil=True)
def _quadratic_forms_real(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Return real(x_k^H M x_k) for the columns of X.

    This is the only small numeric kernel worth compiling here. The xarray
    packaging and plotting paths are not performance critical.
    """
    nrow, ncol = X.shape
    out = np.empty(ncol, dtype=np.float64)

    for k in range(ncol):
        total = 0.0 + 0.0j
        for i in range(nrow):
            row_sum = 0.0 + 0.0j
            for j in range(nrow):
                row_sum += M[i, j] * X[j, k]
            total += np.conjugate(X[i, k]) * row_sum
        out[k] = total.real

    return out


def _quadratic_forms_real_safe(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    # if hasattr(_quadratic_forms_real, "signatures"):
    #     return _quadratic_forms_real(X, M)

    return np.real(
        np.einsum("ik,ij,jk->k", np.conjugate(X), M, X, optimize=True)
    )


def _positive_diag_metric(M: np.ndarray, size: int) -> np.ndarray:
    """
    Return a nonnegative diagonal weight for a Hermitian-positive-ish metric block.

    For mode classification we need positive per-m weights. The full J/L
    quadratic form is used for the global Alfven/sound fraction, but the
    diagonal is more robust for assigning a dominant m.
    """
    if M is None:
        return np.ones(size, dtype=np.float64)

    d = np.real(np.diag(M)).astype(np.float64)

    good = np.isfinite(d) & (d > 0.0)
    if not np.any(good):
        return np.ones(size, dtype=np.float64)

    floor = np.nanmedian(d[good]) * 1e-14
    return np.maximum(d, floor)

def classify_full_modes_at_radius(
    vr: np.ndarray,
    B: np.ndarray,
    m_range: np.ndarray,
    family_threshold: float = 0.65,
    eps: float = 1e-300,
) -> dict:
    """
    Classify full sound-Alfven continuum eigenvectors at one radial point.

    Parameters
    ----------
    vr:
        Right eigenvectors returned by scipy.linalg.eig(A, B), shape
        (2*nmode, 2*nmode). Columns are eigenmodes.
    B:
        Generalized mass matrix for this radial point.
    m_range:
        Poloidal mode array, e.g. np.arange(-mmax, mmax + 1).
    family_threshold:
        Threshold for hard family label. If sound_fraction >= threshold,
        label as sound. If alfven_fraction >= threshold, label as Alfven.
        Otherwise label as mixed.

    Returns
    -------
    dict of arrays, each with shape (nstate,).
    family_code convention:
        -1 = invalid
         0 = Alfvenic
         1 = sound/acoustic
         2 = mixed
    alcon_family_code convention:
         0 = Alfvenic by raw largest component
         1 = sound/acoustic by raw largest component
    """
    m_range = np.asarray(m_range)
    nmode = m_range.size
    nstate = 2 * nmode

    if vr.shape != (nstate, nstate):
        raise ValueError(
            f"Expected vr.shape == {(nstate, nstate)}, got {vr.shape}."
        )

    # Split the full state vector into the two physical sectors.
    xA = vr[:nmode, :]      # Alfvenic/shear block
    xS = vr[nmode:, :]      # sound/acoustic block

    J = B[:nmode, :nmode]
    L = B[nmode:, nmode:]

    # ------------------------------------------------------------
    # ALCON-compatible raw classification.
    # ------------------------------------------------------------
    raw_power = np.abs(vr) ** 2
    raw_total = np.sum(raw_power, axis=0)
    raw_idx = np.argmax(raw_power, axis=0)

    alcon_family_code = (raw_idx >= nmode).astype(np.int8)
    alcon_m = m_range[raw_idx % nmode]
    alcon_purity = np.max(raw_power, axis=0) / np.maximum(raw_total, eps)

    # ------------------------------------------------------------
    # Metric-weighted global Alfven/sound participation.
    # ------------------------------------------------------------
    # Sector quadratic forms:
    #   E_A = a^H J a
    #   E_S = s^H L s
    #
    # J and L should be positive Hermitian in a clean ideal-MHD continuum
    # discretization. Small negative values can appear from roundoff or
    # mild non-Hermitian contamination, so we fall back to diagonal weights
    # when needed.
    EA_full = _quadratic_forms_real_safe(xA, J)
    ES_full = _quadratic_forms_real_safe(xS, L)

    Jdiag = _positive_diag_metric(J, nmode)
    Ldiag = _positive_diag_metric(L, nmode)

    WA = Jdiag[:, None] * np.abs(xA) ** 2
    WS = Ldiag[:, None] * np.abs(xS) ** 2

    EA_diag = np.sum(WA, axis=0)
    ES_diag = np.sum(WS, axis=0)

    bad_metric = (
        ~np.isfinite(EA_full)
        | ~np.isfinite(ES_full)
        | (EA_full < 0.0)
        | (ES_full < 0.0)
    )

    EA = np.where(bad_metric, EA_diag, EA_full)
    ES = np.where(bad_metric, ES_diag, ES_full)

    EA = np.maximum(EA, 0.0)
    ES = np.maximum(ES, 0.0)

    Etot = EA + ES
    alfven_fraction = EA / np.maximum(Etot, eps)
    sound_fraction = ES / np.maximum(Etot, eps)

    family_code = np.full(nstate, 2, dtype=np.int8)
    family_code[alfven_fraction >= family_threshold] = 0
    family_code[sound_fraction >= family_threshold] = 1
    family_code[Etot <= eps] = -1

     # ------------------------------------------------------------
    # Dominant m classification.
    # ------------------------------------------------------------
    # Use diagonal metric weights for per-m participation. This is robust and
    # positive. The full quadratic form is not cleanly separable by m because
    # off-diagonal Fourier couplings contain cross terms.
    Wm = WA + WS
    Wm_total = np.sum(Wm, axis=0)

    imain = np.argmax(Wm, axis=0)
    dominant_m = m_range[imain]
    m_purity = Wm[imain, np.arange(nstate)] / np.maximum(Wm_total, eps)

    dominant_m_alfven = m_range[np.argmax(WA, axis=0)]
    dominant_m_sound = m_range[np.argmax(WS, axis=0)]

    alfven_m_purity = np.max(WA, axis=0) / np.maximum(np.sum(WA, axis=0), eps)
    sound_m_purity = np.max(WS, axis=0) / np.maximum(np.sum(WS, axis=0), eps)

    return {
        "family_code": family_code,
        "alfven_fraction": alfven_fraction,
        "sound_fraction": sound_fraction,
        "dominant_m": dominant_m.astype(np.int32),
        "m_purity": m_purity,
        "dominant_m_alfven": dominant_m_alfven.astype(np.int32),
        "dominant_m_sound": dominant_m_sound.astype(np.int32),
        "alfven_m_purity": alfven_m_purity,
        "sound_m_purity": sound_m_purity,
        "alcon_family_code": alcon_family_code,
        "alcon_m": alcon_m.astype(np.int32),
        "alcon_purity": alcon_purity,
        "bad_metric": bad_metric,
    }


@dataclass
class ContinuumClassifier:
    vr: np.ndarray
    vl: np.ndarray
    w: np.ndarray
    J: np.ndarray
    L: np.ndarray
    m_range: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None
    ntor: int = 0
    family_threshold: float = 0.65
    eps: float = 1e-300
    n_top_modes: int = 8
    data: Optional[xr.Dataset] = field(default=None, init=False, repr=False)

    _FAMILY_LABELS = {
        -1: "invalid",
         0: "Alfvenic",
         1: "sound",
         2: "mixed",
    }
    _ALCON_LABELS = {
        0: "Alfvenic",
        1: "sound",
    }
    _FAMILY_ALIASES = {
        "alfven": "Alfvenic",
        "alfvenic": "Alfvenic",
        "alvfenic": "Alfvenic",
        "alvenic": "Alfvenic",
        "sound": "sound",
        "acoustic": "sound",
        "mixed": "mixed",
        "invalid": "invalid",
    }

    def __post_init__(self):
        self.vr = self._as_vector_stack("vr", self.vr)
        nx, nstate, _ = self.vr.shape
        if nstate % 2 != 0:
            raise ValueError(
                "Full continuum eigenvectors must have an even number of states."
            )

        nmode = nstate // 2
        self.m_range = self._infer_m_range(nmode)
        if self.x is None:
            self.x = self._infer_x_from_profile_inputs(nx, self.J, self.L)
        self.x = self._x_coord(nx)
        self.w = self._as_eigenvalue_stack(self.w, nx, nstate)
        self.J = self._as_metric_stack(
            "J", self.J, nx, nmode, self.m_range, self.ntor, self.x
        )
        self.L = self._as_metric_stack(
            "L", self.L, nx, nmode, self.m_range, self.ntor, self.x
        )

        if self.vl is not None:
            self.vl = self._as_vector_stack("vl", self.vl)
            if self.vl.shape != self.vr.shape:
                raise ValueError(f"vl must have shape {self.vr.shape}, got {self.vl.shape}.")
        
        self.classify(family_threshold=self.family_threshold, eps=self.eps)

    @staticmethod
    def _profile_data(value):
        if isinstance(value, xr.DataArray):
            return value
        data = getattr(value, "data", None)
        if isinstance(data, xr.DataArray):
            return data
        return None

    @classmethod
    def _infer_x_from_profile_inputs(cls, nx: int, *values) -> Optional[np.ndarray]:
        for value in values:
            data = cls._profile_data(value)
            if data is None or "x" not in data.coords:
                continue
            x = np.asarray(data.coords["x"].values)
            if x.ndim == 1 and x.size == nx:
                return x
        return None

    @staticmethod
    def _harmonics_to_metric_stack(
        harmonics: np.ndarray,
        nx: int,
        m_range: np.ndarray,
        ntor: int,
    ) -> np.ndarray:
        harmonics = np.asarray(harmonics)
        if harmonics.ndim != 3:
            raise ValueError(
                f"Harmonic metric data must have shape (x, mpol, ntor), "
                f"got {harmonics.shape}."
            )

        nmode = m_range.size
        if harmonics.shape[0] == 1 and nx != 1:
            harmonics = np.broadcast_to(harmonics, (nx,) + harmonics.shape[1:])
        if harmonics.shape[0] != nx:
            raise ValueError(
                f"Harmonic metric data has {harmonics.shape[0]} radial points "
                f"but vr has {nx}."
            )

        out = np.zeros((nx, nmode, nmode), dtype=np.complex128)
        n_range = np.full(nmode, ntor, dtype=np.int32)
        m_range = np.asarray(m_range, dtype=np.int32)

        for ix in range(nx):
            fill_block_kernel(
                np.ascontiguousarray(harmonics[ix]),
                m_range,
                n_range,
                out[ix],
            )

        return out

    @classmethod
    def _profile_to_metric_stack(
        cls,
        name: str,
        value,
        nx: int,
        m_range: np.ndarray,
        ntor: int,
        x: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        data = cls._profile_data(value)
        if data is None:
            return None
        if "mpol" not in data.dims:
            return None

        if "x" in data.dims:
            if x is None:
                raise ValueError(
                    f"{name} was passed as profile data with an x coordinate, "
                    "but classifier x could not be inferred. Pass x explicitly."
                )
            data = data.interp(x=x)
        else:
            data = data.expand_dims(x=np.arange(1, dtype=np.int32))

        if "ntor" not in data.dims:
            data = data.expand_dims(ntor=[0])

        mmax = int(np.max(np.abs(m_range)))
        harmonics = (
            data.reindex(mpol=np.arange(mmax + 1), ntor=np.arange(1), fill_value=0)
                .transpose("x", "mpol", "ntor")
                .values
        )
        return cls._harmonics_to_metric_stack(harmonics, nx, m_range, ntor)

    @classmethod
    def _array_harmonics_to_metric_stack(
        cls,
        value: np.ndarray,
        nx: int,
        m_range: np.ndarray,
        ntor: int,
    ) -> Optional[np.ndarray]:
        arr = np.asarray(value)
        mmax = int(np.max(np.abs(m_range)))
        mharm = mmax + 1

        if arr.ndim == 1:
            if arr.size < mharm:
                return None
            harmonics = arr[None, :, None]

        elif arr.ndim == 2:
            if arr.shape[0] == nx and arr.shape[1] >= mharm:
                harmonics = arr[:, :, None]
            elif arr.shape[0] >= mharm:
                harmonics = arr[None, :, :]
            else:
                return None

        elif arr.ndim == 3:
            if arr.shape[1] < mharm:
                return None
            harmonics = arr

        else:
            return None

        return cls._harmonics_to_metric_stack(harmonics, nx, m_range, ntor)

    @staticmethod
    def _as_vector_stack(name: str, value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        if arr.ndim != 3:
            raise ValueError(
                f"{name} must have shape (state, mode) or "
                f"(x, state, mode), got {arr.shape}."
            )
        if arr.shape[-2] != arr.shape[-1]:
            raise ValueError(
                f"{name} must contain square eigenvector matrices, got {arr.shape}."
            )
        return arr

    @staticmethod
    def _as_eigenvalue_stack(
        value: np.ndarray,
        nx: int,
        nstate: int,
    ) -> np.ndarray:
        arr = np.asarray(value)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape != (nx, nstate):
            raise ValueError(
                f"w must have shape {(nstate,)} or {(nx, nstate)}, got {arr.shape}."
            )
        return arr

    @staticmethod
    def _as_metric_stack(
        name: str,
        value: np.ndarray,
        nx: int,
        nmode: int,
        m_range: Optional[np.ndarray] = None,
        ntor: int = 0,
        x: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if m_range is not None:
            profile_stack = ContinuumClassifier._profile_to_metric_stack(
                name, value, nx, m_range, ntor, x
            )
            if profile_stack is not None:
                return profile_stack

        arr = np.asarray(value)

        if arr.ndim == 1:
            if arr.size != nmode and m_range is not None:
                harmonic_stack = ContinuumClassifier._array_harmonics_to_metric_stack(
                    arr, nx, m_range, ntor
                )
                if harmonic_stack is not None:
                    return harmonic_stack
            if arr.size != nmode:
                raise ValueError(
                    f"{name} diagonal must have length {nmode}, got {arr.size}."
                )
            arr = np.diag(arr)[None, :, :]

        elif arr.ndim == 2:
            if arr.shape == (nmode, nmode):
                arr = arr[None, :, :]
            elif arr.shape == (nx, nmode):
                arr = np.asarray([np.diag(row) for row in arr])
            elif arr.shape == (1, nmode):
                arr = np.diag(arr[0])[None, :, :]
            elif m_range is not None:
                harmonic_stack = ContinuumClassifier._array_harmonics_to_metric_stack(
                    arr, nx, m_range, ntor
                )
                if harmonic_stack is not None:
                    return harmonic_stack
                raise ValueError(
                    f"{name} must be a metric matrix, metric stack, diagonal "
                    f"weights, or profile harmonics with nmode={nmode}; got "
                    f"{arr.shape}."
                )
            else:
                raise ValueError(
                    f"{name} must be a metric matrix, metric stack, or diagonal "
                    f"weights with nmode={nmode}; got {arr.shape}."
                )

        elif arr.ndim == 3:
            if arr.shape[-2:] != (nmode, nmode) and m_range is not None:
                harmonic_stack = ContinuumClassifier._array_harmonics_to_metric_stack(
                    arr, nx, m_range, ntor
                )
                if harmonic_stack is not None:
                    return harmonic_stack
        else:
            raise ValueError(
                f"{name} must have shape (m, m), (x, m, m), (m,), "
                f"or (x, m), got {arr.shape}."
            )

        if arr.shape[-2:] != (nmode, nmode):
            raise ValueError(
                f"{name} metric blocks must have shape {(nmode, nmode)}, "
                f"got {arr.shape[-2:]}."
            )
        if arr.shape[0] == 1 and nx != 1:
            arr = np.broadcast_to(arr, (nx, nmode, nmode))
        if arr.shape[0] != nx:
            raise ValueError(
                f"{name} has {arr.shape[0]} radial points but vr has {nx}."
            )
        return arr

    @staticmethod
    def _block_metric(J: np.ndarray, L: np.ndarray) -> np.ndarray:
        nmode = J.shape[0]
        B = np.zeros((2 * nmode, 2 * nmode), dtype=np.result_type(J, L, 1j))
        B[:nmode, :nmode] = J
        B[nmode:, nmode:] = L
        return B

    def _infer_m_range(self, nmode: int) -> np.ndarray:
        if self.m_range is not None:
            m_range = np.asarray(self.m_range, dtype=np.int32)
            if m_range.ndim != 1 or m_range.size != nmode:
                raise ValueError(
                    f"m_range must be one-dimensional with length {nmode}, "
                    f"got {m_range.shape}."
                )
            return m_range

        if nmode % 2 == 0:
            raise ValueError(
                "m_range cannot be inferred from an even number of poloidal "
                "modes. Pass m_range explicitly."
            )
        mmax = nmode // 2
        return np.arange(-mmax, mmax + 1, dtype=np.int32)

    def _x_coord(self, nx: int) -> np.ndarray:
        if self.x is None:
            return np.arange(nx, dtype=np.int32)

        x = np.asarray(self.x)
        if x.ndim != 1 or x.size != nx:
            raise ValueError(f"x must have length {nx}, got {x.shape}.")
        return x

    @staticmethod
    def _labels(codes: np.ndarray, mapping: dict) -> np.ndarray:
        labels = np.full(codes.shape, "unknown", dtype=object)
        for code, label in mapping.items():
            labels[codes == code] = label
        return labels

    @classmethod
    def _normalize_family_name(cls, family: str) -> str:
        key = str(family).lower().replace("é", "e").replace("è", "e")
        key = key.replace("-", "").replace("_", "").replace(" ", "")
        if key not in cls._FAMILY_ALIASES:
            allowed = ", ".join(["Alfvenic", "sound", "mixed", "invalid"])
            raise ValueError(f"Unknown family '{family}'. Expected one of {allowed}.")
        return cls._FAMILY_ALIASES[key]

    @classmethod
    def _family_selection(
        cls,
        families: Optional[Iterable[str]] = None,
        exclude_families: Optional[Iterable[str]] = None,
        only: Optional[str] = None,
        include_invalid: bool = False,
    ) -> set[str]:
        all_families = {"Alfvenic", "sound", "mixed"}
        if include_invalid:
            all_families.add("invalid")

        if only is not None:
            if families is not None:
                raise ValueError("Use either only or families, not both.")
            selected = {cls._normalize_family_name(only)}
        elif families is None:
            selected = set(all_families)
        else:
            selected = {cls._normalize_family_name(family) for family in families}

        if exclude_families is not None:
            selected -= {
                cls._normalize_family_name(family)
                for family in exclude_families
            }

        return selected

    @staticmethod
    def _plot_values(eigenvalues: np.ndarray, y: str) -> np.ndarray:
        if y == "sqrt_real":
            with np.errstate(invalid="ignore"):
                return np.sqrt(np.where(eigenvalues.real >= 0.0,
                                        eigenvalues.real, np.nan))
        if y == "real":
            return eigenvalues.real
        if y == "imag":
            return eigenvalues.imag
        if y == "abs":
            return np.abs(eigenvalues)
        raise ValueError(
            "y must be one of 'sqrt_real', 'real', 'imag', or 'abs'."
        )

    @staticmethod
    def _family_labels_from_code(code: np.ndarray) -> np.ndarray:
        code = np.asarray(code)
        labels = np.full(code.shape, "invalid", dtype=object)
        labels[code == 0] = "Alfvenic"
        labels[code == 1] = "sound"
        labels[code == 2] = "mixed"
        return labels

    @staticmethod
    def _format_top_m_text(
        top_m_row,
        top_w_row,
        top_a_row,
        top_s_row,
        min_weight: float = 0.005,
    ) -> str:
        parts = []
        for m, wt, wa, ws in zip(top_m_row, top_w_row, top_a_row, top_s_row):
            if not np.isfinite(wt) or wt < min_weight:
                continue

            parts.append(
                f"m={int(m)}: {100.0 * wt:5.1f}% "
                f"(A {100.0 * wa:4.1f}%, S {100.0 * ws:4.1f}%)"
            )

        if not parts:
            return "no resolved m content"

        return "<br>".join(parts)

    def _top_m_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nx, nstate, _ = self.vr.shape
        nmode = nstate // 2
        nrank = min(max(int(self.n_top_modes), 1), nmode)
        m_range = np.asarray(self.m_range, dtype=np.int32)

        top_m = np.zeros((nx, nstate, nrank), dtype=np.int32)
        top_weight = np.full((nx, nstate, nrank), np.nan, dtype=np.float64)
        top_alfven = np.full((nx, nstate, nrank), np.nan, dtype=np.float64)
        top_sound = np.full((nx, nstate, nrank), np.nan, dtype=np.float64)

        for ix in range(nx):
            xA = self.vr[ix, :nmode, :]
            xS = self.vr[ix, nmode:, :]
            Jdiag = _positive_diag_metric(self.J[ix], nmode)
            Ldiag = _positive_diag_metric(self.L[ix], nmode)

            WA = Jdiag[:, None] * np.abs(xA) ** 2
            WS = Ldiag[:, None] * np.abs(xS) ** 2
            Wtot = WA + WS
            mode_totals = np.sum(Wtot, axis=0)

            for imode in range(nstate):
                denom = max(mode_totals[imode], self.eps)
                total_fraction = Wtot[:, imode] / denom
                alfven_fraction = WA[:, imode] / denom
                sound_fraction = WS[:, imode] / denom
                order = np.argsort(total_fraction)[::-1][:nrank]

                top_m[ix, imode, :] = m_range[order]
                top_weight[ix, imode, :] = total_fraction[order]
                top_alfven[ix, imode, :] = alfven_fraction[order]
                top_sound[ix, imode, :] = sound_fraction[order]

        return top_m, top_weight, top_alfven, top_sound

    @classmethod
    def _classification_to_tables(
        cls,
        eigenvalues: np.ndarray,
        classification: xr.Dataset,
        y: str = "omega",
        min_top_weight: float = 0.005,
    ):
        """
        Flatten eigenvalues and classification into point and distribution tables.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "ContinuumClassifier.make_dash_app requires pandas, plotly, "
                "and dash. Install the optional dashboard dependencies to use it."
            ) from exc

        vals = np.asarray(eigenvalues)
        x = np.asarray(classification.coords["x"].values)

        npsi, nstate = vals.shape
        point_id = np.arange(npsi * nstate, dtype=np.int64).reshape(npsi, nstate)

        lambda_real = np.real(vals)
        lambda_imag = np.imag(vals)

        if y == "omega":
            with np.errstate(invalid="ignore"):
                yvals = np.where(lambda_real >= 0.0, np.sqrt(lambda_real), np.nan)
            y_label = "omega"
        elif y in {"omega2", "lambda"}:
            yvals = lambda_real
            y_label = "omega^2"
        else:
            raise ValueError("y must be either 'omega' or 'omega2'.")

        family_code = classification["family_code"].values
        family = cls._family_labels_from_code(family_code)

        top_m = classification["top_m"].values
        top_weight = classification["top_weight"].values
        top_alfven = classification["top_alfven_weight"].values
        top_sound = classification["top_sound_weight"].values

        npoints = npsi * nstate
        nrank = top_m.shape[2]

        top_text = []
        for ix in range(npsi):
            for imode in range(nstate):
                top_text.append(
                    cls._format_top_m_text(
                        top_m[ix, imode, :],
                        top_weight[ix, imode, :],
                        top_alfven[ix, imode, :],
                        top_sound[ix, imode, :],
                        min_weight=min_top_weight,
                    )
                )

        points = pd.DataFrame(
            {
                "point_id": point_id.ravel(),
                "x": np.repeat(x, nstate),
                "mode": np.tile(np.arange(nstate), npsi),
                "y": yvals.ravel(),
                "lambda_real": lambda_real.ravel(),
                "lambda_imag": lambda_imag.ravel(),
                "family_code": family_code.ravel(),
                "family": family.ravel(),
                "dominant_m": classification["dominant_m"].values.ravel(),
                "m_purity": classification["m_purity"].values.ravel(),
                "alfven_fraction": classification["alfven_fraction"].values.ravel(),
                "sound_fraction": classification["sound_fraction"].values.ravel(),
                "top_text": np.asarray(top_text, dtype=object),
            }
        )
        points = points[np.isfinite(points["y"])].copy()

        distribution = pd.DataFrame(
            {
                "point_id": np.repeat(point_id.ravel(), nrank),
                "rank": np.tile(np.arange(1, nrank + 1), npoints),
                "m": top_m.reshape(npoints * nrank),
                "weight": top_weight.reshape(npoints * nrank),
                "alfven_weight": top_alfven.reshape(npoints * nrank),
                "sound_weight": top_sound.reshape(npoints * nrank),
            }
        )
        distribution = distribution[
            np.isfinite(distribution["weight"]) & (distribution["weight"] > 0.0)
        ].copy()

        return points, distribution, y_label

    @staticmethod
    def _make_interactive_continuum_figure(
        points,
        y_label: str = "omega",
        title: str = "Full sound-Alfven continuum",
        use_webgl: bool = True,
        families: Optional[Iterable[str]] = None,
        exclude_families: Optional[Iterable[str]] = None,
    ):
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError(
                "ContinuumClassifier.make_dash_app requires plotly. Install "
                "the optional dashboard dependencies to use it."
            ) from exc

        trace_cls = go.Scattergl if use_webgl else go.Scatter

        symbol_map = {
            "Alfvenic": "circle",
            "sound": "diamond",
            "mixed": "x",
            "invalid": "cross",
        }

        fig = go.Figure()

        families_to_plot = ContinuumClassifier._family_selection(
            families=families,
            exclude_families=exclude_families,
            include_invalid=False,
        )
        ordered_families = [
            family for family in ["Alfvenic", "sound", "mixed"]
            if family in families_to_plot
        ]
        mode_color = np.abs(points["dominant_m"].to_numpy(dtype=np.int32))
        max_color_m = int(np.max(mode_color)) if mode_color.size else 0
        color_max = max(max_color_m, 1)
        first_colorbar = True

        for family in ordered_families:
            df = points[points["family"] == family].copy()
            if df.empty:
                continue
            color_m = np.abs(df["dominant_m"].to_numpy(dtype=np.int32))

            customdata = np.column_stack(
                [
                    df["point_id"].to_numpy(),
                    df["mode"].to_numpy(),
                    df["dominant_m"].to_numpy(),
                    df["family"].to_numpy(dtype=object),
                    df["alfven_fraction"].to_numpy(),
                    df["sound_fraction"].to_numpy(),
                    df["m_purity"].to_numpy(),
                    df["lambda_real"].to_numpy(),
                    df["lambda_imag"].to_numpy(),
                    df["top_text"].to_numpy(dtype=object),
                ]
            )

            fig.add_trace(
                trace_cls(
                    x=df["x"],
                    y=df["y"],
                    mode="markers",
                    name=family,
                    customdata=customdata,
                    marker={
                        "size": 4.0 + 5.0 * np.clip(
                            df["m_purity"].to_numpy(), 0.0, 1.0
                        ),
                        "symbol": symbol_map[family],
                        "color": color_m,
                        "colorscale": "Jet",
                        "cmin": 0,
                        "cmax": color_max,
                        "showscale": first_colorbar,
                        "colorbar": (
                            {
                                "title": {"text": "|dominant m|", "side": "right"},
                                "x": 1.015,
                                "y": 0.48,
                                "len": 0.78,
                                "thickness": 18,
                                "outlinewidth": 0,
                            }
                            if first_colorbar else None
                        ),
                    },
                    hovertemplate=(
                        "<b>%{customdata[3]}</b><br>"
                        "x = %{x:.5f}<br>"
                        f"{y_label} = "
                        + "%{y:.6g}<br>"
                        "mode index = %{customdata[1]}<br>"
                        "dominant m = %{customdata[2]}<br>"
                        "m purity = %{customdata[6]:.1%}<br>"
                        "Alfven fraction = %{customdata[4]:.1%}<br>"
                        "sound fraction = %{customdata[5]:.1%}<br>"
                        "Re(omega^2) = %{customdata[7]:.6g}<br>"
                        "Im(omega^2) = %{customdata[8]:.3e}<br>"
                        "<br><b>top m content</b><br>"
                        "%{customdata[9]}"
                        "<extra></extra>"
                    ),
                )
            )

            first_colorbar = False

        fig.update_layout(
            title=title,
            xaxis_title="radial coordinate x",
            yaxis_title=y_label,
            legend_title="mode family",
            legend={
                "orientation": "h",
                "x": 0.0,
                "xanchor": "left",
                "y": 1.09,
                "yanchor": "bottom",
            },
            hovermode="closest",
            margin={"l": 65, "r": 105, "t": 95, "b": 55},
        )

        return fig

    def classify(
        self,
        family_threshold: Optional[float] = None,
        eps: Optional[float] = None,
    ) -> xr.Dataset:
        """
        Classify every continuum eigenmode and store the result as an xarray Dataset.

        The returned dataset is ordered by radial coordinate ``x`` and eigenmode
        index ``mode``. The class also keeps it in ``self.data``.
        """
        threshold = self.family_threshold if family_threshold is None else family_threshold
        eps_value = self.eps if eps is None else eps

        vr = self._as_vector_stack("vr", self.vr)
        nx, nstate, _ = vr.shape
        if nstate % 2 != 0:
            raise ValueError(
                "Full continuum eigenvectors must have an even number of states."
            )

        nmode = nstate // 2
        m_range = self._infer_m_range(nmode)
        x = self._x_coord(nx)
        w = self._as_eigenvalue_stack(self.w, nx, nstate)
        J = self._as_metric_stack("J", self.J, nx, nmode)
        L = self._as_metric_stack("L", self.L, nx, nmode)

        if self.vl is not None:
            vl = self._as_vector_stack("vl", self.vl)
            if vl.shape != vr.shape:
                raise ValueError(f"vl must have shape {vr.shape}, got {vl.shape}.")

        results = {}
        for ix in range(nx):
            B = self._block_metric(J[ix], L[ix])
            out = classify_full_modes_at_radius(
                vr[ix],
                B,
                m_range,
                family_threshold=threshold,
                eps=eps_value,
            )
            for key, value in out.items():
                results.setdefault(key, []).append(value)

        results = {
            key: np.stack(value, axis=0)
            for key, value in results.items()
        }

        dset = xr.Dataset(
            coords={
                "x": x,
                "mode": np.arange(nstate, dtype=np.int32),
                "m_pol": m_range,
            },
            attrs={
                "description": "Metric-weighted continuum mode classification.",
                "family_threshold": threshold,
                "eps": eps_value,
            },
        )
        dset["eigenvalue"] = (("x", "mode"), w)

        for key, value in results.items():
            dset[key] = (("x", "mode"), value)

        top_m, top_weight, top_alfven, top_sound = self._top_m_arrays()
        dset = dset.assign_coords(
            rank=np.arange(1, top_m.shape[2] + 1, dtype=np.int32)
        )
        dset["top_m"] = (("x", "mode", "rank"), top_m)
        dset["top_weight"] = (("x", "mode", "rank"), top_weight)
        dset["top_alfven_weight"] = (("x", "mode", "rank"), top_alfven)
        dset["top_sound_weight"] = (("x", "mode", "rank"), top_sound)

        dset["family"] = (
            ("x", "mode"),
            self._labels(results["family_code"], self._FAMILY_LABELS),
        )
        dset["alcon_family"] = (
            ("x", "mode"),
            self._labels(results["alcon_family_code"], self._ALCON_LABELS),
        )

        dset["family_code"].attrs["legend"] = (
            "-1 invalid, 0 Alfvenic, 1 sound, 2 mixed"
        )
        dset["alcon_family_code"].attrs["legend"] = (
            "0 Alfvenic, 1 sound, based on the largest raw eigenvector component"
        )
        dset["dominant_m"].attrs["description"] = (
            "Dominant poloidal mode from diagonal metric-weighted participation."
        )
        dset["m_purity"].attrs["description"] = (
            "Fraction of metric-weighted power in the dominant poloidal mode."
        )
        dset["top_m"].attrs["description"] = (
            "Top-ranked poloidal modes by diagonal metric-weighted participation."
        )
        dset["top_weight"].attrs["description"] = (
            "Top-ranked total Alfvenic plus sound participation fraction."
        )

        self.data = dset
        return dset

    def to_xarray(self) -> xr.Dataset:
        """Return the cached classification dataset, computing it if needed."""
        if self.data is None:
            return self.classify()
        return self.data

    def plot(
        self,
        ax=None,
        y: str = "sqrt_real",
        x: np.ndarray | None = None,
        scale: float = 1.0,
        xlabel: str = "Radial coordinate",
        ylabel: Optional[str] = None,
        title: Optional[str] = "Continuum mode classification",
        colorbar: bool = True,
        legend: bool = True,
        only: Optional[str] = None,
        families: Optional[Iterable[str]] = None,
        exclude_families: Optional[Iterable[str]] = None,
        **scatter_kwargs,
    ):
        """
        Plot the classified continuum branches.

        Points are colored by dominant poloidal mode ``m`` and marker shape
        identifies the Alfvenic, sound, or mixed family.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm

        dset = self.to_xarray()
        eigenvalues = dset["eigenvalue"].values
        yvals = self._plot_values(eigenvalues, y) * scale
        if x is not None:
            xvals = np.asarray(x)
            if xvals.ndim != 1 or xvals.size != yvals.shape[0]:
                raise ValueError(
                    f"x must be one-dimensional with length {yvals.shape[0]}, "
                    f"got {xvals.shape}."
                )
            xvals = np.broadcast_to(xvals[:, None], yvals.shape)
        else:
            xvals = np.broadcast_to(dset["x"].values[:, None], yvals.shape)
        family = dset["family_code"].values
        dominant_m = dset["dominant_m"].values
        color_m = np.abs(dominant_m)
        purity = np.clip(dset["m_purity"].values, 0.0, 1.0)
        m_range = dset["m_pol"].values

        if ax is None:
            fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
        else:
            fig = ax.figure

        cmap = plt.get_cmap("jet")
        max_color_m = int(np.max(np.abs(m_range)))
        bounds = np.arange(-0.5, max_color_m + 1.5)
        norm = BoundaryNorm(bounds, cmap.N)

        base_kwargs = {
            "cmap": cmap,
            "norm": norm,
            "linewidths": 0.35,
            "alpha": 0.88,
        }
        base_kwargs.update(scatter_kwargs)
        sizes = base_kwargs.pop("s", 10.0 + 26.0 * purity**1.5)

        family_specs = [
            (-1, "invalid", "X", "#64748b"),
            (0, "Alfvenic", "o", "#1d4ed8"),
            (1, "sound", "^", "#c2410c"),
            (2, "mixed", "s", "#475569"),
        ]
        finite = np.isfinite(yvals)
        mappable = None

        selected_families = self._family_selection(
            families=families,
            exclude_families=exclude_families,
            only=only,
            include_invalid=True,
        )
        family_specs = [
            item for item in family_specs if item[1] in selected_families
        ]

        for code, label, marker, edgecolor in family_specs:
            mask = finite & (family == code)
            if not np.any(mask):
                continue
            isizes = sizes[mask] if np.ndim(sizes) else sizes
            artist = ax.scatter(
                xvals[mask],
                yvals[mask],
                c=color_m[mask],
                s=isizes,
                marker=marker,
                edgecolors=edgecolor,
                label=label,
                **base_kwargs,
            )
            if mappable is None:
                mappable = artist

        if colorbar and mappable is not None:
            cbar = fig.colorbar(mappable, ax=ax, pad=0.02)
            cbar.set_label("|Dominant m|")
            ticks = np.arange(max_color_m + 1, dtype=np.int32)
            if ticks.size > 21:
                ticks = np.unique(
                    np.round(np.linspace(ticks.min(), ticks.max(), 9)).astype(int)
                )
            cbar.set_ticks(ticks)

        if ylabel is None:
            ylabel = {
                "sqrt_real": r"$\sqrt{\mathrm{Re}(\lambda)}$",
                "real": r"$\mathrm{Re}(\lambda)$",
                "imag": r"$\mathrm{Im}(\lambda)$",
                "abs": r"$|\lambda|$",
            }[y]

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if legend and mappable is not None:
            ax.legend(frameon=False, loc="best")

        ax.grid(True, color="0.88", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.015)

        return fig, ax

    def make_dash_app(
        self,
        y: str = "omega",
        min_top_weight: float = 0.005,
        use_webgl: bool = False,
        families: Optional[Iterable[str]] = None,
        exclude_families: Optional[Iterable[str]] = None,
    ):
        """
        Build a Dash app for inspecting continuum mode classification.

        Use ``families`` or ``exclude_families`` to systematically include or
        remove Alfvenic, sound, or mixed branches from the dashboard.
        """
        try:
            import plotly.graph_objects as go
            from dash import Dash, Input, Output, dcc, html
        except ImportError as exc:
            raise ImportError(
                "ContinuumClassifier.make_dash_app requires dash and plotly. "
                "Install the optional dashboard dependencies to use it."
            ) from exc

        points, distribution, y_label = self._classification_to_tables(
            self.w,
            self.to_xarray(),
            y=y,
            min_top_weight=min_top_weight,
        )
        selected_families = self._family_selection(
            families=families,
            exclude_families=exclude_families,
            include_invalid=False,
        )
        points = points[points["family"].isin(selected_families)].copy()
        distribution = distribution[
            distribution["point_id"].isin(points["point_id"])
        ].copy()
        fig = self._make_interactive_continuum_figure(
            points,
            y_label=y_label,
            title="Full sound-Alfven continuum mode classifier",
            use_webgl=use_webgl,
            families=selected_families,
        )

        points_by_id = points.set_index("point_id", drop=False)
        dist_by_id = {
            int(pid): group.sort_values("rank").copy()
            for pid, group in distribution.groupby("point_id")
        }

        app = Dash(__name__)

        app.layout = html.Div(
            [
                html.H3("Full sound-Alfven continuum mode classifier"),
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Graph(
                                    id="continuum-plot",
                                    figure=fig,
                                    style={"height": "760px"},
                                ),
                            ],
                            style={
                                "width": "68%",
                                "display": "inline-block",
                                "verticalAlign": "top",
                            },
                        ),
                        html.Div(
                            [
                                html.H4("Selected mode"),
                                html.Div(
                                    id="selected-mode-summary",
                                    children=(
                                        "Click a continuum point to inspect "
                                        "its top-m distribution."
                                    ),
                                ),
                                dcc.Graph(
                                    id="mode-distribution",
                                    figure=go.Figure(),
                                    style={"height": "360px"},
                                ),
                            ],
                            style={
                                "width": "30%",
                                "display": "inline-block",
                                "verticalAlign": "top",
                                "paddingLeft": "1rem",
                            },
                        ),
                    ]
                ),
            ]
        )

        def _distribution_figure(point_id: int):
            row = points_by_id.loc[point_id]
            dist = dist_by_id.get(point_id)

            if dist is None or dist.empty:
                return go.Figure()

            dist = dist[dist["weight"] >= min_top_weight].copy()
            fig_bar = go.Figure()

            fig_bar.add_bar(
                x=dist["m"].astype(str),
                y=dist["alfven_weight"],
                name="Alfvenic part",
            )
            fig_bar.add_bar(
                x=dist["m"].astype(str),
                y=dist["sound_weight"],
                name="sound part",
            )

            fig_bar.update_layout(
                title=f"top-m content: point {point_id}, mode {int(row['mode'])}",
                xaxis_title="m",
                yaxis_title="fraction of total diagonal metric weight",
                yaxis_tickformat=".0%",
                barmode="stack",
                template="plotly_white",
                margin={"l": 50, "r": 15, "t": 55, "b": 45},
            )

            return fig_bar

        def _summary_children(point_id: int):
            row = points_by_id.loc[point_id]

            return html.Div(
                [
                    html.Div([html.B("point id: "), f"{point_id}"]),
                    html.Div([html.B("mode index: "), f"{int(row['mode'])}"]),
                    html.Div([html.B("x: "), f"{row['x']:.6g}"]),
                    html.Div([html.B(y_label + ": "), f"{row['y']:.6g}"]),
                    html.Div([html.B("family: "), f"{row['family']}"]),
                    html.Div([html.B("dominant m: "), f"{int(row['dominant_m'])}"]),
                    html.Div([html.B("m purity: "), f"{row['m_purity']:.1%}"]),
                    html.Div(
                        [html.B("Alfven fraction: "), f"{row['alfven_fraction']:.1%}"]
                    ),
                    html.Div([html.B("sound fraction: "), f"{row['sound_fraction']:.1%}"]),
                    html.Br(),
                    html.Div([html.B("top m content:")]),
                    html.Div(
                        row["top_text"].replace("<br>", "\n"),
                        style={"whiteSpace": "pre-wrap", "fontFamily": "monospace"},
                    ),
                ]
            )

        @app.callback(
            Output("selected-mode-summary", "children"),
            Output("mode-distribution", "figure"),
            Input("continuum-plot", "clickData"),
        )
        def _on_click(click_data):
            if click_data is None:
                return (
                    "Click a continuum point to inspect its top-m distribution.",
                    go.Figure(),
                )

            point = click_data["points"][0]
            point_id = int(point["customdata"][0])
            return _summary_children(point_id), _distribution_figure(point_id)

        return app
