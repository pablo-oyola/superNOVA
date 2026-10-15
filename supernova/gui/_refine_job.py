"""Run one TokaMaker refinement in its own process, for the equilibrium tab.

    python -m supernova.gui._refine_job spec.json

The spec is :meth:`~supernova.gui._equilibrium.EquilibriumState.
refinement_spec`. A separate process keeps OpenFUSIONToolkit's Fortran state
and its console output out of the window, lets the user cancel by killing it,
and means a crash in the solver cannot take the window down with it.

Progress is whatever TokaMaker and the q fit print. The last line is the
result, prefixed with :data:`RESULT_MARKER`, as JSON: the diagnostics sidecar
on success, or ``{"error": ...}`` on failure.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import traceback

RESULT_MARKER = "SUPERNOVA_REFINE_RESULT "


def config_from_spec(spec: dict):
    """Turn the plain-data spec into a RefinementConfig."""
    from supernova.refine import DEFAULT_FIT, RefinementConfig

    fit = replace(DEFAULT_FIT, **spec.get("fit", {}))
    return RefinementConfig(
        source_eqdsk=Path(spec["source_eqdsk"]),
        output_dir=Path(spec["output_dir"]),
        stem=str(spec["stem"]),
        source_cocos=int(spec["source_cocos"]),
        output_cocos=int(spec["output_cocos"]),
        run_info=str(spec.get("run_info", "")),
        coarse_dx=float(spec["coarse_dx"]),
        fine_dx=float(spec["fine_dx"]),
        refinement_dx_sequence=tuple(float(v) for v in spec["refinement_dx_sequence"]),
        fe_order=int(spec["fe_order"]),
        ffp_knot_count=int(spec["ffp_knot_count"]),
        export_grid=tuple(int(v) for v in spec["export_grid"]),
        nthreads=int(spec["nthreads"]),
        fit=fit,
    )


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if len(argv) != 2:
        print(__doc__)
        return 2
    try:
        spec = json.loads(Path(argv[1]).read_text())
        from supernova.refine import refine_equilibrium

        config = config_from_spec(spec)
        print(f"Refining {config.source_eqdsk} -> {config.eqdsk_path}", flush=True)
        diagnostics = refine_equilibrium(config, plot=True)
        diagnostics["figure"] = str(config.figure_path)
    except Exception as error:
        traceback.print_exc()
        print(RESULT_MARKER + json.dumps({"error": f"{type(error).__name__}: {error}"}),
              flush=True)
        return 1
    print(RESULT_MARKER + json.dumps(diagnostics, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
