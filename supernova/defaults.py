"""The defaults every front end shares.

The script API, the solver classes, the CLI and the GUI read their defaults
from here, so changing a value here changes it everywhere at once.
"""

from __future__ import annotations

import numpy as np

# the problem
MAX_DM = 20                  # widest |m - m'| coupling kept in the operator
MMAX_RHO_POL = 0.95          # mmax defaults to ceil(n q) at this rho_pol
REGULARIZER = 0.0
GAMMA_S = 5.0 / 3.0
SPECIES = "D"
SOUND_MODEL = "ideal"
CONTINUUM_BRANCHES = "slow_sound"

# the frequency search
NEV = 4
TOL = 1.0e-8
MAXIT = 100
REGION_VSCALE = 0.25

# the radial grid
ORDER = 3
QUAD_ORDER = 6

# the shift-and-invert factorization
FACTORIZATION = "mumps"
# MUMPS's own ICNTL(14), 20 % workspace headroom, runs out on these pencils
# under SLEPc 3.25 with INFOG(1) = -9.
MUMPS_ICNTL_14 = 200


def default_mmax(equilibrium, ntor: int, rho_pol: float = MMAX_RHO_POL) -> int | None:
    """The default mmax for toroidal mode number ``ntor``: ``ceil(abs(n) * q)`` at ``rho_pol``.

    That is the poloidal harmonic resonant at ``rho_pol`` (0.95 by default).
    For n = 0 no harmonic is resonant, so this returns None and the solvers
    keep their own limit, the q band plus its sidebands.
    """
    ntor = abs(int(ntor))
    if ntor == 0:
        return None
    x = np.asarray(equilibrium.qprof.x, dtype=float)
    q = np.abs(np.asarray(equilibrium.qprof.data).squeeze().real)
    # the solver's x spans the Boozer domain; it is psi_N only when that is (0, 1)
    psi_n = x
    magspace = equilibrium.magspace
    if "psi_physical_norm" in magspace.coords:
        psi_n = np.interp(x, magspace.coords["psi"].values,
                          magspace.coords["psi_physical_norm"].values)
    return int(np.ceil(ntor * np.interp(rho_pol**2, psi_n, q)))
