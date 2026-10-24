# The GUI

```bash
pip install -e ".[gui]"       # PySide6, once
supernova gui                 # or: python -m supernova gui
supernova gui session.h5      # reopen a saved session
```

The window has three tabs: **Equilibrium**, **Continuum** and **Solver**.
Each has its own toolbar and status bar. The Equilibrium tab loads an
equilibrium and passes it on to the other two, which have no equilibrium
inputs of their own: all three use the same object. Every slow step runs in
one compute process that starts with the window: it holds the loaded
equilibria, solves the continuum and runs the solver's worker pool. The
window stays responsive through a solve of any length, and *working…* shows
in the top-right corner while the process is busy. Each tab also opens on
its own with `python -m supernova.gui.equilibrium`, `.continuum` or
`.solver`.

The screenshots below follow the ITPA case of the
[quick start](../getting_started.md).

## Equilibrium

```{image} ../img/gui_equilibrium.png
:alt: The Equilibrium tab with the ITPA benchmark loaded, showing its Boozer grid.
```

The *Equilibrium* box chooses the source:

**g-EQDSK file**
: The form takes:
  - the file and its *COCOS*;
  - an optional density file, with its *column / interp.* choices (see
    {py:func}`~supernova.profile_from_columns`) and *units*;
  - *or uniform [m^-3]*, a flat density instead of a file;
  - *Boozer rho_pol*, the domain of the transform;
  - *surfaces x angles*, its resolution;
  - *equilibrium m_max*.

  Without a density the geometry still loads. The other tabs then mark their
  kHz scale as a placeholder, and the Solver refuses to run until a density
  is given.

**ITPA TAE benchmark (analytic)**
: The fields hold the published configuration: a, R0, B0, q0, q edge, the
  sign of B_phi, the resolution and m_max.

*ion species* applies to every tab. Changing it does not reload, because
the geometry does not depend on it. **Load equilibrium** runs the Boozer
transform. That takes tens of seconds for an eqdsk at production
resolution, and a fraction of a second for ITPA.

The pages show what the solver will see:
- **Shape**: the Boozer grid over the flux surfaces, with the selected
  surface in red.
- **Profiles**: the 1-D profiles.
- **Surface**: any metric element along one flux surface against the
  solver-facing angle. The quantities are grouped as contravariant metric,
  covariant metric, Jacobian and field, and curvature, shear and current.
  The surface is set by the *Flux surface* slider, and *divide each by its
  mean over Theta* compares shapes.

**Re-solve with TokaMaker** runs the fixed-boundary refinement of
[Equilibria](equilibria.md) as a separate process, when OpenFUSIONToolkit is
installed. It streams its output into the *TokaMaker log* page and loads the
refined file when it finishes, keeping the previous profiles on the plots
for comparison. The refined equilibrium replaces the one the other tabs use
only if *pass it on* was ticked when the re-solve finished.

## Continuum

```{image} ../img/gui_continuum.png
:alt: The Continuum tab showing the n = 6 Alfvén continuum of the ITPA case with the 60–76 kHz band shaded.
```

- ***n***, ***gamma_s*** and ***mmax*** set the problem. *species* is the
  Equilibrium tab's, shown greyed.
- ***branches*** offers *slow sound only*, *no sound (Alfven only)* and
  *full sound-Alfven*. *label families* colours the full spectrum's roots as
  Alfvénic, sound or mixed, which costs about a second instead of 0.1 s.
- ***radial points*** is what the solve time scales with.

The continuum is solved again on every edit; *Solve now* forces it. Under
*View*:
- *f max* caps the axis (0 autoscales);
- *shade band* marks a search range;
- *mode f* places a dashed line you can drag, to read off where a frequency
  crosses the continuum.

## Solver

```{image} ../img/gui_solver_spectrum.png
:alt: The Solver tab's Spectrum page after a solve, with the gap mode marked over the continuum and the table of roots.
```

The left column holds every setting of one pass:

*Problem*
: n and mmax, max dm and regularizer, gamma_s, and the *sound* model: *ideal
  MHD*, or *NOVA ksound=1* without the parallel sound. See
  [Eigenmode scans](eigenmodes.md) for what each one means.

*Frequency search*
: The band, the window width and overlap, nev and the region's vscale, and
  tol and maxit. A line below says how the band will be cut.

*Radial grid*
: The spans, the order and quadrature, and the walls in ρ_pol. The
  *resolution guide* lists each gap of the continuum, the width its modes
  should have, and the element size that resolves them against the current
  grid. Hover a gap on the spectrum for the same numbers.

*Eigensolver and linear solves*
: Collapsed by default. It holds the PEP type, the spectral transform, the
  factorization, *MUMPS ICNTL(14)* (200 by default) and raw PETSc options.

*Mode filter*
: *MODELOCATE (Kramer)*, as in {py:meth}`ModeSet.locate
  <supernova.ModeSet.locate>`, or the *global path alpha* test. The path
  test re-solves each root on a deformed complex radial path: a gap mode
  stays put and a continuum mode moves. Both keep MODELOCATE's spike, core
  and edge tests. *show gap modes only* hides the rest.

*Refine selected modes*
: Re-solves the modes selected in the table on meshes graded towards them
  (*finest [mm] / meshes*, *growth*) until their frequency converges.
  *path-test on the finest mesh* is optional.

*Parallel windows*
: *workers / threads each*: the windows are spread over worker processes,
  each with its own OpenMP threads.

**Solve** runs the pass. The pages then show:
- **Spectrum**: every root at its radial centre, with a bar for its width,
  over the continuum. Gap modes are marked apart from the roots the filter
  rejected, and the gap estimates are drawn too. The table lists frequency,
  Im/Re ω, the filter's verdict, peak, centre, width, dominant m, backward
  error and window. Roots that converged but miss the 10⁻⁸ backward-error
  gate are dropped.
- **Mode structure**: the selected root's field (ξ^ψ, p1, ξ^s or ∇·ξ) and
  part. The *view* is either *on the continuum*, with its harmonics drawn
  around the mode's frequency, or the harmonics alone. A line below explains
  the filter's verdict.
- **Windows**: what each window did, how many roots converged against how
  many were asked for, and why SLEPc stopped. This is where a search goes
  wrong. A window that returns nev roots (*saturated*) may be hiding more;
  one that hits maxit found nothing.
- **Refinement** and **Log**: the refinement's meshes and frequencies, and
  the solver output, including any traceback.

```{image} ../img/gui_solver_modes.png
:alt: The Mode structure page with the gap mode's harmonics drawn on the continuum.
```

## Sessions

*File → Save session…* (Ctrl+Shift+S) writes one HDF5 file. It holds every
tab's settings, the continuum, and the solver's pass and refinement. The
tabs' own *Save session* / *Load session* buttons save and open the same
whole session. *File → Open session…* (Ctrl+Shift+O), or
`supernova gui session.h5`, loads its equilibrium again and puts the roots
back once that load is done. JSON sessions from earlier versions still open,
with their settings only.
