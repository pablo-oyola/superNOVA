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
