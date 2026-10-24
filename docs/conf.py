"""Sphinx configuration for the superNOVA documentation.

The pages are Markdown (MyST), so the files under ``docs/`` read the same on
GitHub. The README and CONVENTIONS.md are pulled in with ``{include}`` rather
than copied, and the API reference is autodoc on the pure-Python layers: the
PETSc/SLEPc backend is imported lazily and never reached here.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import supernova  # noqa: E402


project = "superNOVA"
author = "superNOVA contributors"
copyright = "2026, superNOVA contributors"
release = supernova.__version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]

myst_enable_extensions = ["dollarmath", "colon_fence"]
myst_heading_anchors = 3

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True}
# supernova.refine needs OpenFUSIONToolkit at import, which is not on PyPI, and
# supernova.interop.desc needs DESC, an optional extra the docs do not install
autodoc_mock_imports = ["OpenFUSIONToolkit", "desc"]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

html_theme = "sphinx_rtd_theme"
html_title = "superNOVA"
html_logo = "img/supernova_logo.png"
html_theme_options = {
    "navigation_depth": 3,
    "logo_only": True,
    # the logo is navy on transparent: it needs a light header
    "style_nav_header_background": "#ffffff",
}
