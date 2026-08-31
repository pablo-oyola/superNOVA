"""The artifact schema: storage version, classification labels, field titles.

Shared by everything in :mod:`supernova.app`, and kept in its own module so the
report, the reconstruction and the artifact I/O do not have to import each
other just to agree on a label colour.
"""

from __future__ import annotations


SCHEMA_VERSION = 2


SUPPORTED_SCHEMA_VERSIONS = (1, 2)


LABEL_ORDER = (
    "solver_rejected",
    "physical_gap",
    "continuum_discretization",
    "continuum_coupled_candidate",
    "discretization_artifact",
    "unresolved",
)


LABEL_STYLE = {
    "solver_rejected": {
        "title": "Solver rejected",
        "color": "#758295",
        "symbol": "x",
    },
    "physical_gap": {
        "title": "Physical gap",
        "color": "#159A8A",
        "symbol": "circle",
    },
    "continuum_discretization": {
        "title": "Continuum discretization",
        "color": "#D64B5F",
        "symbol": "diamond",
    },
    "continuum_coupled_candidate": {
        "title": "Continuum-coupled candidate",
        "color": "#7C3AED",
        "symbol": "star",
    },
    "discretization_artifact": {
        "title": "Discretization artifact",
        "color": "#475569",
        "symbol": "cross",
    },
    "unresolved": {
        "title": "Unresolved",
        "color": "#E39A31",
        "symbol": "triangle-up",
    },
}


FIELD_TITLES = {
    "xi_psi": "Radial displacement xi^psi",
    "p1": "Pressure perturbation p1",
    "xi_s": "Tangential displacement xi^s",
    "div_xi": "Compression div xi",
}
