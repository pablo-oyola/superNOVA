# Equilibria

Every calculation starts from a `superEquilibrium`. It holds the flux
surfaces, the Boozer coordinates and their metric, and the density and
pressure profiles. A {py:class}`~supernova.Case` pairs it with a continuum
(next page).

Building the equilibrium is the slow step: tens of seconds for the Boozer
transform of an eqdsk at production resolution. Nothing in it depends on the
toroidal mode number or the frequency band, so build it once and reuse it for
every scan.

## From a g-EQDSK file

{py:meth}`Case.from_eqdsk <supernova.Case.from_eqdsk>` reads the file
through [pyCOCOS](https://github.com/pablo-oyola/pyCOCOS) and converts it to
the internal COCOS 1. It then computes the Boozer coordinates and builds the
continuum:

```python
import supernova as sn

case = sn.Case.from_eqdsk(
    "shot.geqdsk", cocos_in=5,
    density=sn.profile_from_columns("ne.dat", units="m^-3"),
    boozer_domain=(0.005, 0.995),
    gamma_s=5 / 3, species="D",
    m_max=128, res_boozer=(511, 512),
)
```

`cocos_in`
: The COCOS convention the file is written in. It is required: superNOVA
  never guesses it.

`density`
: A {py:class}`~supernova.Profile` on normalised poloidal flux ψ_N.
  {py:func}`~supernova.profile_from_columns` reads a two-column text file. It
  takes two choices that give different profiles for the same data:
  - `x` is what the first column holds, `"psi_n"` (the default) or
    `"rho_pol"`;
  - `interpolate_in` is the variable the interpolation runs in, with
    `"rho_pol"` as the default.

  `skiprows=1` skips one header line.

`pressure`
: `"eqdsk"`, the default, takes p(ψ) from the file. Pass a `Profile` to
  override it. {py:func}`~supernova.profile_from_eqdsk` turns any of the
  file's 1-D profiles into one.

`boozer_domain`
: The (inner, outer) range of the Boozer transform, in ρ_pol. It keeps the
  transform away from the magnetic axis, where the coordinates are singular,
  and from the separatrix, where q diverges.

Other keywords
: These go to {py:class}`~supernova.superEquilibrium`:
  - `m_max`: the poloidal harmonics kept in the metric;
  - `res_boozer=(npsi, ntheta)`: the Boozer grid, where the GUI's default is
    (511, 512);
  - optional `vtor`, `vpol` and `resistivity` profiles.

  The profiles must depend on the flux surface only.

## The ITPA benchmark

{py:class}`~supernova.cases.itpa.ITPAEq` is the analytic equilibrium of the
ITPA TAE benchmark: a circular, large-aspect-ratio plasma. It is built
directly in straight-field-line coordinates, with no file and no transform,
and carries the benchmark's own density and pressure. Since it already *is*
a `superEquilibrium`, pair it with a continuum yourself:

```python
import unyt
import supernova as sn
from supernova.cases.itpa import ITPAEq

equilibrium = ITPAEq(aminor=1.0 * unyt.m, Rmajor=10.0 * unyt.m, B0=3.0 * unyt.T,
                     q0=1.71, qedge=1.87, ns=256, ntheta=256, m_max=20)
case = sn.Case(equilibrium, sn.superContinuum(equilibrium, gamma_s=0.0, species="H"))
```

`ns` and `ntheta` set the radial and poloidal resolution of the coordinates.
`Bt_sign` flips the toroidal field while `B0` stays its magnitude. It builds
in a fraction of a second, which makes it the case to try things on.

## DESC

{py:class}`~supernova.interop.desc.DESCSolver` re-solves the equilibrium
with DESC from a boundary, a q profile and a pressure profile. It needs the
optional `desc` extra (`tools/install_stack.py --desc`). From an eqdsk, the
boundary, q and pressure come from the file:

```python
import pycocos as pc
from supernova.interop.desc import DESCSolver

eqdsk = pc.EQDSK("shot.geqdsk", cocos_in=5)
density = sn.profile_from_columns("ne.dat", units="m^-3")
equilibrium = DESCSolver.from_eqdsk(eqdsk, density)
equilibrium.solve()            # DESC, then the field and the metric
case = sn.Case(equilibrium, sn.superContinuum(equilibrium, gamma_s=0.0, species="D"))
```

## TokaMaker refinement

A tabulated eqdsk is rarely an exact Grad-Shafranov solution, and the error
shows up in the q profile that the Boozer transform derives from the
geometry. {py:mod}`supernova.refine` re-solves the file as a fixed-boundary
TokaMaker equilibrium whose q matches that geometric q. It needs
OpenFUSIONToolkit, which is not on PyPI, plus the `tokamaker` extra.
Refinement is a separate pre-processing step: it writes a new eqdsk, so it
does not have to share an environment with the solver.

```python
from pathlib import Path
from supernova.refine import RefinementConfig, refine_equilibrium

refine_equilibrium(RefinementConfig(
    source_eqdsk=Path("shot.geqdsk"), source_cocos=5,
    output_dir=Path("refined"), stem="shot_tokamaker",
))
case = sn.Case.from_refined(
    "refined", density=sn.profile_from_columns("ne.dat", units="m^-3"))
```

The refinement writes `<stem>.eqdsk`, a `<stem>.json` sidecar with its
diagnostics and output COCOS, and a comparison figure `<stem>.png`. It raises
an error instead of writing a degraded equilibrium when an acceptance
threshold in the config is missed. {py:meth}`Case.from_refined
<supernova.Case.from_refined>` reads the sidecar, so you never pass the
refined file's COCOS by hand. The GUI runs the same refinement from the
Equilibrium tab ([GUI](gui.md)).
