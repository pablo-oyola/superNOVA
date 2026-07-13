"""Import-safe plotting helpers for named mixed-spline solver4 fields."""

import numpy as np

from no_sync.solver4_analysis import (
    reconstruct_solver4_field,
    solver4_field_envelope,
)


_FIELD_LABELS = {
    "p1": r"p_1",
    "xi_psi": r"\xi^\psi",
    "xi^psi": r"\xi^\psi",
    "xi_s": r"\xi^s",
    "xi^s": r"\xi^s",
    "div_xi": r"\nabla\!\cdot\!\xi",
}


def _radial_coordinate(equ, psi):
    psi = np.asarray(psi, dtype=float)
    if hasattr(equ, "_psi") and hasattr(equ, "_rho"):
        return np.interp(psi, equ._psi, equ._rho)
    return np.sqrt(np.clip(psi, 0.0, None))


def plot_xipsi_eigenpair(
    result,
    solver,
    sb=None,
    eigenpair_index=0,
    mpol=None,
    n_plot=512,
    field="xi_psi",
    max_modes=None,
    radial_label=r"$r/a$",
    pair=None,
):
    """Plot a named solver4 field using its compatible high or low basis."""
    del sb  # Harmonic labels are carried by the named coefficient arrays.
    if result.sizes.get("eigenpair", 0) == 0:
        raise ValueError("No converged eigenpairs are available to plot.")
    if field not in _FIELD_LABELS:
        raise ValueError(f"field must be one of {list(_FIELD_LABELS)}, got {field!r}.")

    # Matplotlib remains optional until a plotting function is actually called.
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    psi, m_values, field_values = reconstruct_solver4_field(
        result,
        solver,
        field=field,
        eigenpair_index=eigenpair_index,
        n_plot=n_plot,
        pair=pair,
    )
    rho = _radial_coordinate(solver.equ, psi)

    mode_indices = None
    if mpol is not None:
        requested = np.atleast_1d(mpol).astype(int)
        mode_indices = []
        for m_value in requested:
            matches = np.flatnonzero(m_values == m_value)
            if matches.size == 0:
                raise ValueError(f"m={m_value} is outside the solver harmonic range.")
            mode_indices.append(int(matches[0]))
        mode_indices = np.asarray(mode_indices, dtype=int)
    elif max_modes is not None:
        norms = np.linalg.norm(field_values, axis=0)
        mode_indices = np.argsort(norms)[-int(max_modes):]

    if mode_indices is None:
        mode_indices = np.arange(m_values.size)
    if mode_indices.size == 0:
        raise ValueError("At least one harmonic must be selected for plotting.")

    field_label = _FIELD_LABELS[field]
    fig, ax = plt.subplots()
    if mode_indices.size == 1:
        mode_index = int(mode_indices[0])
        values = field_values[:, mode_index]
        ax.plot(rho, values.real, label=rf"$\Re\,({field_label})$")
        ax.plot(rho, values.imag, "--", label=rf"$\Im\,({field_label})$")
        ax.plot(rho, np.abs(values), ":", label=rf"$|{field_label}|$")
        ax.legend()
        mode_summary = rf"$m={int(m_values[mode_index])}$"
    else:
        selected_m = m_values[mode_indices]
        norm = mpl.colors.Normalize(
            vmin=float(np.min(selected_m)),
            vmax=float(np.max(selected_m)),
        )
        cmap = plt.get_cmap("viridis")
        norms = np.linalg.norm(field_values[:, mode_indices], axis=0)
        dominant = int(mode_indices[np.argmax(norms)])
        for mode_index in mode_indices:
            color = cmap(norm(float(m_values[mode_index])))
            width = 2.0 if mode_index == dominant else 0.9
            ax.plot(
                rho,
                field_values[:, mode_index].real,
                color=color,
                lw=width,
                label=rf"$m={int(m_values[mode_index])}$",
            )
            ax.plot(
                rho,
                field_values[:, mode_index].imag,
                color=color,
                lw=width,
                ls="--",
            )
        ax.legend(title=r"$m$", loc="upper left", bbox_to_anchor=(1.02, 1.0))
        mode_summary = rf"dominant $m={int(m_values[dominant])}$"

    s_value = result["s"].isel(eigenpair=eigenpair_index).item()
    omega_value = result["omega"].isel(eigenpair=eigenpair_index).item()
    residual = ""
    if "residual_norm" in result:
        residual_value = result["residual_norm"].isel(
            eigenpair=eigenpair_index
        ).item()
        residual = rf", res={residual_value:.1e}"

    ax.set_xlabel(radial_label)
    ax.set_ylabel(rf"$({field_label})_m$")
    ax.set_title(
        rf"${field_label}$, eigenpair {eigenpair_index}, {mode_summary}; "
        rf"$s={s_value:.4g}$, $\omega={omega_value:.4g}${residual}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def _omega_norm_to_frequency_khz(omega_norm, solver):
    import unyt

    omega_norm = np.asarray(omega_norm, dtype=np.complex128)
    omega_unit = unyt.Unit("omegaA", registry=solver.equ.registry)
    return (omega_norm.real * omega_unit).to("kHz").v / (2 * np.pi)


def plot_slow_sound_with_eigenmode_extensions(
    result,
    solver,
    sb,
    psi_continuum,
    slow_sound,
    region=None,
    extension_threshold=0.2,
    n_plot=512,
    max_eigenpairs=None,
    radial_label=r"$r/a$",
    pair=None,
):
    """Plot slow-sound branches and named ``xi_psi`` mode extensions."""
    del sb
    if result.sizes.get("eigenpair", 0) == 0:
        raise ValueError("No converged eigenpairs are available to plot.")
    if not 0.0 < extension_threshold <= 1.0:
        raise ValueError("extension_threshold must lie in (0, 1].")

    import matplotlib.pyplot as plt

    psi_continuum = np.asarray(psi_continuum, dtype=float)
    rho_continuum = _radial_coordinate(solver.equ, psi_continuum)
    slow_sound_khz = np.sqrt(slow_sound.real).to("kHz").v / (2 * np.pi)
    fig, ax = plt.subplots()
    for branch in range(slow_sound_khz.shape[1]):
        ax.plot(
            rho_continuum,
            slow_sound_khz[:, branch],
            color="0.65",
            lw=0.8,
            alpha=0.75,
            label="slow sound continuum" if branch == 0 else None,
        )

    if region is not None:
        center = complex(region["center"])
        radius = float(region["radius"])
        omega_band = np.sqrt(
            [max(0.0, center.real - radius), max(0.0, center.real + radius)]
        )
        fmin, fmax = _omega_norm_to_frequency_khz(omega_band, solver)
        ax.axhspan(
            fmin,
            fmax,
            color="tab:orange",
            alpha=0.18,
            label="NEP search band",
        )

    n_eigenpairs = result.sizes["eigenpair"]
    if max_eigenpairs is not None:
        n_eigenpairs = min(n_eigenpairs, int(max_eigenpairs))
    colors = plt.get_cmap("tab10")(
        np.linspace(0.0, 1.0, max(1, n_eigenpairs))
    )

    for index in range(n_eigenpairs):
        psi, envelope = solver4_field_envelope(
            result,
            solver,
            field="xi_psi",
            eigenpair_index=index,
            n_plot=n_plot,
            pair=pair,
        )
        peak = float(np.max(envelope))
        if peak == 0.0:
            continue
        active = envelope >= extension_threshold * peak
        rho = _radial_coordinate(solver.equ, psi)
        active_rho = rho[active]
        frequency = float(
            _omega_norm_to_frequency_khz(
                result["omega"].isel(eigenpair=index).item(), solver
            )
        )
        residual_label = ""
        if "residual_norm" in result:
            residual = result["residual_norm"].isel(eigenpair=index).item()
            residual_label = rf", res={residual:.1e}"
        ax.hlines(
            frequency,
            active_rho[0],
            active_rho[-1],
            color=colors[index],
            lw=2.5,
            label=rf"NEP $k={index}${residual_label}",
        )
        ax.plot(rho[int(np.argmax(envelope))], frequency, "o", color=colors[index])

    ax.set_xlabel(radial_label)
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(
        rf"Slow-sound continuum and solver4 mode extensions "
        rf"($|\xi^\psi|>{100 * extension_threshold:.0f}\%$ peak)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return fig, ax


__all__ = [
    "plot_slow_sound_with_eigenmode_extensions",
    "plot_xipsi_eigenpair",
]
