# Command line

```bash
supernova solve      ...    # the multigrid scan, saved to an artifact directory
supernova reclassify ...    # re-sort saved roots, without SLEPc
supernova reacquire  ...    # continue roots lost between levels
supernova gui [session.h5]  # the window (see GUI)
```

`python -m supernova` is the same command, and `supernova <command> --help`
lists every option with its default.

## `supernova solve`

```bash
supernova solve --equilibrium shot.geqdsk --cocos-in 5 --density ne.dat \
                --species D --ntor 4 --mmax 68 --frequency-khz 50 150 \
                --artifact-dir runs/n4
```

The script API solves on one grid. `solve` instead runs
{py:class}`~supernova.MultigridSolver`: the same problem on several radial
levels, each twice as fine as the last. Every root is tracked across the
levels and classified from that evidence. Always pass `--equilibrium` and
`--density`, because their defaults point at a development case that is not
in the repository.

| Options | What they set |
|---|---|
| `--equilibrium`, `--cocos-in`, `--density` | the eqdsk, its COCOS, and a two-column ψ_N density file |
| `--ntor`, `--mmax`, `--max-dm`, `--regularizer` | the problem, as in [Eigenmode scans](eigenmodes.md) |
| `--gamma-s`, `--species` | the continuum |
| `--frequency-khz LOW HIGH` | the band |
| `--window-width-khz`, `--window-padding`, `--region-vscale` | fixed core windows of at most this width, each padded on both sides by a fraction of a core width, and the ellipse's imaginary semi-axis |
| `--nev`, `--tol`, `--maxit` | roots per window and SLEPc's stopping rule |
| `--base-spans`, `--levels`, `--order`, `--quadrature-order` | the radial levels: the default 38 spans × 3 levels gives 38, 76 and 152 |
| `--factorization {mumps,spqr,petsc_qr}` | the shift-and-invert factorization |
| `--adaptive-discovery` | extra windows (K = 4, 8, 16, and centre and quarter-window offsets) on the coarse and medium grids |
| `--no-reacquisition`, `--reacquisition-max-candidates` | turn off, or bound, the narrow follow-up searches for roots missing on the next level |
| `--save-raw-levels` | also write every level's eigenpairs |
| `--profile-points`, `--reconstruction-points`, `--display-continuum-points` | sampling of the profiles, the saved mode structures and the plotted continuum |

The artifact directory receives:

| File | Contents |
|---|---|
| `modes.nc` | the classified modes and their reconstructed structure |
| `continuum.nc`, `display_continuum.nc` | the solver's continuum, and a finer one for plots |
| `window_diagnostics.nc` | what every window on every level did |
| `level_<n>.nc` | each level's raw eigenpairs (with `--save-raw-levels`) |
| `summary.csv` | one row per mode |
| `overview.png` | the modes over the continuum |
| `manifest.json` | the inputs, the options and the git commit |

## `supernova reclassify`

```bash
supernova reclassify --artifact-dir runs/n4 [--output-dir runs/n4_v2] [--source auto|raw|legacy]
```

`reclassify` sorts the saved eigenpairs again with the current classifier
without running SLEPc. `--source raw` rebuilds the tracks from the
`level_<n>.nc` catalogs; `legacy` keeps `modes.nc`'s old assignments; `auto`
prefers the raw catalogs when they are there. It writes new
`*_evidence_v2.nc` datasets, a manifest and a `reclassification_report.json`
next to the originals, and never overwrites them.

## `supernova reacquire`

```bash
supernova reacquire --artifact-dir runs/n4             # dry run: what it would do
supernova reacquire --artifact-dir runs/n4 --execute   # run the narrow windows
```

`reacquire` continues saved roots that are missing on the next radial level.
It selects them and runs one narrow SLEPc window around each. That window's
half-width is `--relative-half-width` of the root's frequency, with
`--nev-multiplier` times the roots, and a match needs `--minimum-overlap`.
Without `--execute` it only writes `reacquisition_dry_run.json`, so you can
check the selection first. With it, it writes `level_<n>_reacquired.nc`.
`reclassify` uses those once every level has one.
