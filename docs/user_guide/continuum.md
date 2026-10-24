# The continuum

The shear-Alfvén and sound continua locate the gaps where global modes can
live, and the radii where the rest of the spectrum resonates.
{py:class}`~supernova.superContinuum` builds them from an equilibrium:

```python
continuum = sn.superContinuum(equilibrium, gamma_s=5 / 3, species="D")
```

`gamma_s`
: The adiabatic index. Exactly 0 removes the sound coupling. It is a
  different problem, not a limit: the eigenproblem becomes generalized
  linear (SLEPc EPS) instead of polynomial (SLEPc PEP).

`species`
: The ion species. With the density it sets the mass density, and so the
  Alfvén speed and every frequency in kHz.

The continuum depends on γ_s and the species but not on the geometry's
expensive part. {py:meth}`Case.with_continuum <supernova.Case.with_continuum>`
therefore swaps it in about 0.1 s without repeating the Boozer transform:

```python
case_no_sound = case.with_continuum(gamma_s=0.0)
```

## Solving it on its own

{py:meth}`superContinuum.solve <supernova.superContinuum.solve>` gives the
continuum at any set of radii, without the eigensolver:

```python
import numpy as np

omega2 = continuum.solve(ntor=6, mmax=20, which="alfven", normalize_eigvals=True,
                         psi_vals=np.linspace(0.0, 1.0, 256))
f_khz = np.sqrt(omega2.real).to("kHz").v / (2 * np.pi)    # (radius, branch)
```

`which` picks the branches:
- `"alfven"`: shear Alfvén only, no sound;
- `"slow_sound"`: the slow sound branches;
- `"full"`: the coupled sound-Alfvén spectrum.

`classify=True` labels each root Alfvénic, sound or mixed from its
eigenvector. The labels only mean something for `which="full"`, since the
pure branches have one family by construction.

## The continuum the modes see

The continuum to compare eigenmodes against is
{py:attr}`Scan.continuum <supernova.Scan.continuum>`. It is built from the
solver's own blocks, harmonics and `max_dm` coupling cut-off, on its
quadrature points, so its crossings are the ones the modes actually see. It
is an `xarray.Dataset` with normalised `omega(psi, branch)`, which
{py:meth}`Scan.band_for <supernova.Scan.band_for>` converts to kHz:

```python
psi = scan.continuum.coords["psi"].values
f_khz = scan.band_for(scan.continuum["omega"].values.real)
```

The left panel of the figure on [Eigenmode scans](eigenmodes.md) is drawn
this way.

## The sound model

With γ_s > 0 the full sound coupling is the `"ideal"` sound model.
`sound_model="nova"` instead drops the parallel sound, as NOVA-K's
`ksound = 1` does. Pass it to the eigensolve
(`scan.run(..., sound_model="nova")`), or choose it in the *sound* row of the
GUI's Solver tab. The full γ_s = 5/3 problem is not yet ready for production
use, so use the NOVA model or γ_s = 0 for now.
