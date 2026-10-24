# Eigenmode scans

superNOVA solves the linear ideal-MHD eigenvalue problem for one toroidal
mode number n. The displacement is expanded in B-spline finite elements in
the radius and in Fourier harmonics in the poloidal angle. The eigenvalue is
s = ω², with ω normalised to the on-axis Alfvén frequency. For γ_s > 0 the
problem is a polynomial eigenvalue problem (SLEPc PEP); for γ_s = 0 it
reduces to a generalized linear one (SLEPc EPS).
{py:attr}`Scan.backend <supernova.Scan.backend>` says which one a scan will
use.

Each solve is shift-and-invert around a target frequency. It finds the roots
nearest the target inside an elliptical region of the complex s-plane, so a
frequency band is searched window by window.

## The complete example

[`examples/itpa_tae.py`](https://github.com/pablo-oyola/superNOVA/blob/master/examples/itpa_tae.py)
runs the ITPA case end to end in a couple of seconds:

```{literalinclude} ../../examples/itpa_tae.py
:language: python
:pyobject: main
```

It prints:

```text
continuum: 9 branches on 256 radii
   69.954 kHz  gap        peak at rho_pol = 0.500
   70.150 kHz  continuum  peak at rho_pol = 0.510
   70.336 kHz  continuum  peak at rho_pol = 0.500
   70.659 kHz  continuum  peak at rho_pol = 0.520
   71.018 kHz  continuum  peak at rho_pol = 0.490
gap mode: 69.954 kHz, centroid psi_N = 0.254, width 0.012
```

```{image} ../img/itpa_tae.png
:alt: The n = 6 continuum of the ITPA case with the gap mode's frequency, and the mode's four strongest poloidal harmonics.
```

The rest of this page goes through each piece.

## The radial grid

{py:class}`~supernova.RadialGrid` describes the finite elements:

```python
grid = sn.RadialGrid.uniform(spans=150)                  # cubic, 150 equal elements
grid = sn.RadialGrid.graded(150, centers=[0.25], local=2e-3, growth=0.15)
```

`order=3` (cubic) and `quad_order=6` are the defaults. `graded` keeps the
base element size of `spans` equal elements, but refines to `local` around
each of `centers`. The element size relaxes back at a rate set by `growth`.
Both are in the solver's radial coordinate, which spans the Boozer domain
in ψ_N. The radial displacement and the pressure live in this space. The
tangential displacement and the compression live in the compatible
lower-order space built with it.

## The problem

```python
scan = sn.Scan(case, grid, ntor=6, mmax=20, max_dm=5, regularizer=1e-5)
```

`ntor`
: The toroidal mode number.

`mmax`
: The highest poloidal harmonic allowed. The scan keeps the harmonics that
  can resonate across the q range, from ⌊n·q_min⌋ − 3 to ⌈n·q_max⌉ + 3,
  capped at `mmax`. For the ITPA case at n = 6 that is m = 7–15. The
  results list them in `m_pol`.

`max_dm`
: The widest coupling |m − m′| kept in the operator.

`regularizer`
: ε scales the inertia by (1 + iε)². This is a small imaginary frequency
  shift that keeps continuum resonances finite. 0 is the ideal problem; the
  ITPA benchmark uses 10⁻⁵.

A `Scan` assembles the operator blocks once. Every window after the first is
only a new shift-and-invert solve.

## Bands and windows

```python
band = sn.Band(60.0, 76.0, units="kHz")         # vscale=0.25 by default
modes = scan.run(band, window=2.0, overlap=0.0, nev=4, tol=1e-7, maxit=200,
                 options=sn.PetscOptions(extra={"st_mat_mumps_icntl_14": "200"}))
```

{py:class}`~supernova.Band` holds the band in physical units. It converts
the band to s = ω² through the case's unit registry, so no script needs its
own kHz-to-ω conversion. {py:meth}`Scan.run <supernova.Scan.run>` cuts the
band into windows of `window` kHz, with an optional fractional `overlap`.
Each window is solved for `nev` roots, targeted at its centre, inside an
ellipse whose imaginary semi-axis is `vscale` times its half-width. The roots
of all windows are concatenated. `on_window(index, total, low, high, modes)`
is called after each window, which is enough to drive a progress bar.
{py:meth}`Scan.solve_window <supernova.Scan.solve_window>` solves a single
window.

The window size is a trade-off:
- **Too wide for `nev`** (a *saturated* window) returns `nev` roots near
  its centre and can miss the rest. The quick start's single 16 kHz window is
  saturated, and the GUI says so in its status bar.
- **Too narrow** costs one factorization per window. A window with no root
  near it may also converge nothing.

````{warning}
When SLEPc converges no eigenpair in a window, the solve raises a
`RuntimeError` and `Scan.run` stops there. In the ITPA case at 100 spans,
for instance, the 2 kHz windows inside the gap (64–68 kHz) converge nothing.
Until that is handled, sweep such a band yourself and skip the empty
windows:

```python
found = []
for low, high in band.windows(2.0):
    try:
        found.append(scan.solve_window(band, low=low, high=high, nev=4,
                                       maxit=200, options=options))
    except RuntimeError:
        pass
```

The GUI keeps going instead and lists each window's outcome on its
*Windows* page.
````

## Solver options

{py:class}`~supernova.PetscOptions` names the shift-and-invert
factorization:
- `"mumps"` (the default) is an LU that copes with the indefinite shifted
  operator;
- `"spqr"` and `"petsc_qr"` are the QR alternatives, worth trying when MUMPS
  runs out of memory or reports a zero pivot.

`extra` takes raw PETSc options. On SLEPc 3.25, MUMPS's default workspace
headroom (ICNTL(14) = 20 %) runs out on these pencils with
`INFOG(1) = -9`. Pass `"st_mat_mumps_icntl_14": "200"`, which is what the GUI
uses.

`sound_model="nova"` and the other keywords of
{py:meth}`StabilitySolver.solve_pep <supernova.StabilitySolver.solve_pep>`
pass through `run` and `solve_window` unchanged.

## Working with the modes

{py:class}`~supernova.ModeSet` wraps the solver's `xarray.Dataset`
(`modes.dataset`) and adds what scripts used to recompute:

| | |
|---|---|
| `len(modes)`, `modes.omega` | number of roots; complex normalised ω |
| `modes.frequency_khz`, `modes.growth_rate_khz` | Re ω and Im ω in kHz |
| `modes.m_pol` | the poloidal harmonics |
| `modes.field(name, n=512)` | a field on `n` radii, dims `(eigenpair, x, m_pol)` |
| `modes.centroid`, `modes.width` | power-weighted centre and RMS width of the ξ^ψ envelope, in the solver's radial coordinate |
| `modes.locate()`, `modes.gap_modes()` | MODELOCATE classes (below) and the gap modes alone |
| `modes.sort_by_frequency()` | the same roots in frequency order |
| `modes.to_netcdf(path)` | the raw dataset, complex arrays split into `_real`/`_imag` |

The four fields are:
- `xi_psi`: the radial displacement ξ^ψ;
- `p1`: the pressure perturbation;
- `xi_s`: the tangential displacement ξ^s;
- `div_xi`: the compression ∇·ξ.

`field` picks the right basis for each, so you never handle the
finite-element coefficients yourself.

### Gap modes and continuum modes

{py:meth}`ModeSet.locate <supernova.ModeSet.locate>` is a port of Gerrit
Kramer's `modelocate`, the filter NOVA users ran on every scan. It flags each
mode from where its ξ^ψ envelope peaks, with radial thresholds in ρ_pol:

| class | when |
|---|---|
| `spike` | the peak is one grid point: both neighbours are below `spike_level` (0.7) of it |
| `core` | the peak is inside `core_edge` (0.05) |
| `edge` | the peak is outside `boundary_edge` (0.96) |
| `continuum` | the envelope exceeds `continuum_level` (0.9) at a continuum crossing |
| `ideal` | Re ω² < 0, a purely growing mode, which is not tested further |
| `gap` | none of the above |

Pass a {py:class}`~supernova.ModeLocateThresholds` to change the levels. The
spike and continuum tests compare neighbouring grid points, so the verdicts
depend on the radial grid the modes were solved on. `gap_modes()` keeps the
`gap` class.
