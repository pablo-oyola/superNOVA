"""
Classification of modes in the continuum.
"""

from dataclasses import dataclass

import numpy as np


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
    EA_full = np.real(
        np.einsum("ik,ij,jk->k", np.conjugate(xA), J, xA, optimize=True)
    )
    ES_full = np.real(
        np.einsum("ik,ij,jk->k", np.conjugate(xS), L, xS, optimize=True)
    )

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
class ContinuumClassification:
    vr: np.ndarray
    vl: np.ndarray
    w: np.ndarray
    J: np.ndarray
    L: np.ndarray
    def classify(self):
        pass


