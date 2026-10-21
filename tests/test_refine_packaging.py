"""The refinement package's contract with the solver environment.

These tests deliberately never run TokaMaker.  What they protect is the seam:
``import supernova`` must not need OpenFUSIONToolkit, and the diagnostics
sidecar must stay readable, because that sidecar is how a refined equilibrium
tells the solver which COCOS it is in.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_importing_supernova_does_not_require_openfusiontoolkit():
    """The solver environment must not have to install OFT."""
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys, supernova;"
         " sys.exit(1 if 'OpenFUSIONToolkit' in sys.modules else 0)"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        "importing supernova pulled in OpenFUSIONToolkit:\n" + result.stderr
    )


def test_refinement_config_names_its_three_outputs():
    refine = pytest.importorskip("supernova.refine")
    config = refine.RefinementConfig(
        source_eqdsk=Path("/src/MAST53678.03200.eqdsk"),
        output_dir=Path("/out"),
        stem="MAST53678.03200_tokamaker",
        source_cocos=5,
    )
    assert config.eqdsk_path == Path("/out/MAST53678.03200_tokamaker.eqdsk")
    assert config.diagnostics_path == Path("/out/MAST53678.03200_tokamaker.json")
    assert config.figure_path == Path("/out/MAST53678.03200_tokamaker.png")
    assert config.output_cocos == 7
    # The fit defaults preserve the measured current and fit only the FF' shape.
    assert config.fit.fit_plasma_current is False


def test_load_diagnostics_reads_a_sidecar_from_a_directory(tmp_path):
    refine = pytest.importorskip("supernova.refine")
    sidecar = {
        "source_eqdsk": "/src/a.eqdsk",
        "output_eqdsk": str(tmp_path / "a.eqdsk"),
        "source_cocos": 5,
        "output_cocos": 7,
        "fine_q_relative_rms": 0.026,
    }
    (tmp_path / "a.json").write_text(json.dumps(sidecar))

    assert refine.load_diagnostics(tmp_path / "a.json") == sidecar
    assert refine.load_diagnostics(tmp_path) == sidecar, (
        "a directory holding one sidecar should resolve to it"
    )


def test_load_diagnostics_rejects_a_file_that_is_not_a_sidecar(tmp_path):
    refine = pytest.importorskip("supernova.refine")
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 3}))
    with pytest.raises(ValueError, match="not a refinement sidecar"):
        refine.load_diagnostics(tmp_path / "manifest.json")


def test_load_diagnostics_refuses_an_ambiguous_directory(tmp_path):
    refine = pytest.importorskip("supernova.refine")
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text(json.dumps({"output_eqdsk": "x", "output_cocos": 7}))
    with pytest.raises(ValueError, match="exactly one diagnostics sidecar"):
        refine.load_diagnostics(tmp_path)


@pytest.mark.skipif(
    not (REPO_ROOT / "no_sync_ref/NSTX/tokamaker_refined").is_dir(),
    reason="the NSTX refinement artifacts are not available here",
)
def test_the_recorded_nstx_sidecar_still_parses():
    """A sidecar produced before packaging must still be readable."""
    refine = pytest.importorskip("supernova.refine")
    diagnostics = refine.load_diagnostics(
        REPO_ROOT / "no_sync_ref/NSTX/tokamaker_refined"
        / "NSTX_128741_00217_tokamaker_diagnostics.json"
    )
    assert diagnostics["source_cocos"] == 5
    assert diagnostics["output_cocos"] == 7
    assert diagnostics["fit_converged"] is True


def test_the_ffp_bounds_grow_to_contain_a_sign_changing_source_shape():
    """A DIII-D F F' changes sign; its knots must start inside the fit bounds."""
    from types import SimpleNamespace

    import numpy as np

    refine = pytest.importorskip("supernova.refine")
    cfg = refine.QFitConfig(ffp_bounds=(0.02, 3.0), profile_fd_step_abs=0.008,
                            verbose=False)
    within = SimpleNamespace(fit_config=cfg, ffp_y=np.array([1.0, 0.6, 0.2, 0.05]))
    assert refine.FixedBoundaryQFitter._covering_ffp_bounds(within) == (0.02, 3.0)

    d3d = np.array([1.0, 0.004, -0.266, -0.368, -0.2, -0.02, 0.183])
    lower, upper = refine.FixedBoundaryQFitter._covering_ffp_bounds(
        SimpleNamespace(fit_config=cfg, ffp_y=d3d))
    assert upper == 3.0
    assert np.isclose(lower, -0.368 - 0.25 * 1.368)


@pytest.mark.skipif(
    not (REPO_ROOT / "no_sync/D3D/202020/202020C20_t0800.geqdsk").is_file(),
    reason="the DIII-D g-file is not available here",
)
def test_the_header_signs_fix_the_cocos_up_to_orientation_and_units():
    refine = pytest.importorskip("supernova.refine")
    from supernova.refine.fixed_boundary import header_cocos_candidates

    path = REPO_ROOT / "no_sync/D3D/202020/202020C20_t0800.geqdsk"
    assert header_cocos_candidates(path) == (5, 6, 15, 16)


def test_a_stalled_solve_resumes_at_a_looser_tolerance_up_to_the_ceiling():
    from types import SimpleNamespace

    refine = pytest.importorskip("supernova.refine")

    class Stalling:
        """Exceeds maxits until nl_tol reaches ``floor``, as TokaMaker does."""

        def __init__(self, floor):
            self.floor, self.settings, self.pushed = floor, SimpleNamespace(nl_tol=1e-7), []

        def update_settings(self):
            self.pushed.append(self.settings.nl_tol)

        def solve(self, return_its):
            if self.settings.nl_tol < self.floor:
                raise ValueError('Error in solve: Exceeded "maxits"')
            return "equilibrium", 3

    fitter = SimpleNamespace(nonlinear_tolerance=1e-7, max_nonlinear_tolerance=1e-5)
    solver = Stalling(floor=2.4e-6)
    assert refine.FixedBoundaryQFitter._solve(fitter, solver) == ("equilibrium", 3)
    assert solver.pushed == pytest.approx([1e-6, 1e-5])
    assert fitter.nonlinear_tolerance == pytest.approx(1e-5)

    fitter = SimpleNamespace(nonlinear_tolerance=1e-7, max_nonlinear_tolerance=1e-5)
    with pytest.raises(ValueError, match="maxits"):
        refine.FixedBoundaryQFitter._solve(fitter, Stalling(floor=1e-3))
