"""The ITPA TAE benchmark, end to end: equilibrium, continuum, eigenmodes.

Needs the compiled PETSc/SLEPc stack (``tools/install_stack.py``) and takes a
couple of seconds::

    python examples/itpa_tae.py [figure.png]

It prints every root in the band with its MODELOCATE class and saves a figure
of the gap mode on its continuum and of its poloidal harmonics.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import unyt

import supernova as sn
from supernova.cases.itpa import ITPAEq

# MUMPS's default workspace headroom, ICNTL(14) = 20 %, runs out on these
# pencils under SLEPc 3.25; the GUI uses 200 % as well.
MUMPS = sn.PetscOptions(extra={"st_mat_mumps_icntl_14": "200"})


def main(figure="itpa_tae.png"):
    # The published configuration: circular, R0/a = 10, q from 1.71 to 1.87.
    equilibrium = ITPAEq(aminor=1.0 * unyt.m, Rmajor=10.0 * unyt.m,
                         B0=3.0 * unyt.T, q0=1.71, qedge=1.87,
                         ns=256, ntheta=256, m_max=20)
    # Hydrogen, and no sound coupling, as the benchmark defines it.
    continuum = sn.superContinuum(equilibrium, gamma_s=0.0, species="H")
    case = sn.Case(equilibrium, continuum)

    # The continuum on its own, in kHz: no eigensolver involved.
    omega2 = continuum.solve(ntor=6, mmax=20, which="alfven",
                             normalize_eigvals=True,
                             psi_vals=np.linspace(0.0, 1.0, 256))
    f_khz = np.sqrt(omega2.real).to("kHz").v / (2.0 * np.pi)
    print(f"continuum: {f_khz.shape[1]} branches on {f_khz.shape[0]} radii")

    scan = sn.Scan(case, sn.RadialGrid.uniform(spans=100), ntor=6, mmax=20,
                   max_dm=5, regularizer=1e-5)
    modes = scan.run(sn.Band(60.0, 76.0, units="kHz"), window=16.0, nev=4,
                     tol=1e-7, maxit=80, options=MUMPS)

    located = modes.locate()
    for f, label, rho in zip(modes.frequency_khz, located["mode_label"].values,
                             located["peak_rho"].values):
        print(f"{f:9.3f} kHz  {label:<10} peak at rho_pol = {rho:.3f}")

    tae = modes.gap_modes()
    print(f"gap mode: {tae.frequency_khz[0]:.3f} kHz, centroid psi_N = "
          f"{tae.centroid[0]:.3f}, width {tae.width[0]:.3f}")

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4),
                                      constrained_layout=True)
    psi = scan.continuum.coords["psi"].values
    branches = scan.band_for(scan.continuum["omega"].values.real)
    left.plot(psi, branches, "k.", ms=1)
    left.axhline(tae.frequency_khz[0], color="C3", lw=1,
                 label=f"gap mode, {tae.frequency_khz[0]:.2f} kHz")
    left.set(xlabel=r"$\psi_N$", ylabel="f [kHz]", ylim=(0, 150),
             title="n = 6 continuum")
    left.legend(loc="upper right")

    xi = tae.field("xi_psi").isel(eigenpair=0)
    strongest = np.argsort(np.abs(xi).max("x").values)[::-1][:4]
    for index in sorted(strongest):
        m = int(xi.m_pol.values[index])
        right.plot(xi.x, xi.isel(m_pol=index).real, label=f"m = {m}")
    right.set(xlabel=r"$\psi_N$", ylabel=r"Re $\xi^\psi$",
              title="the four strongest harmonics")
    right.legend()
    fig.savefig(figure, dpi=110)
    print(f"saved {figure}")


if __name__ == "__main__":
    main(*sys.argv[1:])
