"""The static overview figure written at the end of a scan.

This is what replaced the Dash explorer: matplotlib, no server, written
straight into the artifact directory beside the datasets. Interactive
inspection is deferred to a desktop front end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from supernova.app.schema import LABEL_ORDER, LABEL_STYLE


def _overview_figure(
    modes: xr.Dataset,
    continuum: xr.Dataset,
    frequency_band: tuple[float, float],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(11.5, 6.8))
    psi = np.asarray(continuum.coords["psi"].values, dtype=float)
    continuum_khz = np.asarray(continuum["frequency_khz"].values, dtype=float)
    axis.plot(psi, continuum_khz, color="#708090", lw=0.45, alpha=0.34)
    labeled_modes = False
    for label in LABEL_ORDER:
        selected = np.flatnonzero(np.asarray(modes["label"].values) == label)
        if selected.size == 0:
            continue
        labeled_modes = True
        style = LABEL_STYLE[label]
        axis.errorbar(
            modes["center"].values[selected],
            modes["frequency_khz"].values[selected],
            xerr=modes["radial_width"].values[selected],
            fmt="o",
            color=style["color"],
            ms=5.5,
            elinewidth=1.1,
            capsize=0,
            alpha=0.9,
            label=f"{style['title']} ({selected.size})",
        )
    axis.axhspan(*frequency_band, color="#3B82F6", alpha=0.045)
    axis.set(
        xlim=(0.0, 1.0),
        ylim=(max(0.0, frequency_band[0] - 5.0), frequency_band[1] + 5.0),
        xlabel=r"Normalized poloidal flux $\psi_N$",
        ylabel="Cyclic frequency [kHz]",
        title="multigrid classifications over the slow-sound continuum",
    )
    axis.grid(True, alpha=0.18)
    if labeled_modes:
        axis.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
