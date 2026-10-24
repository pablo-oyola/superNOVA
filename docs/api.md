# API reference

The first two sections import from the top-level package,
`import supernova as sn`; the last one names its submodules.

## Scripting interface

The objects the [Getting started](getting_started.md) examples use.

```{eval-rst}
.. currentmodule:: supernova

.. autosummary::
   :toctree: generated
   :nosignatures:

   Case
   Scan
   Band
   RadialGrid
   ModeSet
   PetscOptions
   profile_from_columns
   profile_from_eqdsk
```

## Equilibria, continuum and solvers

```{eval-rst}
.. currentmodule:: supernova

.. autosummary::
   :toctree: generated
   :nosignatures:

   superEquilibrium
   superContinuum
   StabilitySolver
   MultigridSolver
   locate_modes
   ModeLocateThresholds
   BSplineFE
   Profile
   ContinuumClassifier
```

## Other equilibrium sources

These live in their own modules: DESC and TokaMaker are optional
dependencies, and ITPA is a benchmark case rather than part of the solver.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   cases.itpa.ITPAEq
   interop.desc.DESCSolver
   refine.RefinementConfig
   refine.refine_equilibrium
   refine.load_diagnostics
```
