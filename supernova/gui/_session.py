"""Sessions: every tab's settings, and what it computed, in one HDF5 file.

Each tab keeps everything the user can set in one dataclass; a session holds
those side by side and, next to them, the results that took time to get: the
continuum, the solver's pass -- one group per solver window, with the
eigenpairs it kept -- and the refinement.

The equilibrium itself is not stored. It is loaded again from its inputs when
the session opens: that takes seconds, where pickled it is ~170 MB. Instead
each section's results carry the ``equilibrium_key`` of the load they were
computed on (:meth:`~supernova.gui._equilibrium.EquilibriumState.
equilibrium_key`), and a tab takes them back only once that equilibrium is
loaded.

The layout; every part but the root's attributes is optional::

    /                             format = "supernova-session", version
    /equilibrium                  state (JSON)
    /continuum                    state (JSON), equilibrium_key (JSON)
    /continuum/result             the continuum solved
    /solver                       state (JSON), equilibrium_key (JSON), selected
    /solver/log                   the solver tab's log
    /solver/result                per-mode arrays, fields/, located/ (MODELOCATE)
    /solver/result/windows/wNNN   one solver window: its row (JSON) and the
                                  eigenpairs/ it kept
    /solver/refinement            rows and path verdicts (JSON), x, envelopes/

xarray datasets go in variable by variable, each with its dims, rather than
as netCDF, which takes neither complex values nor bool attributes.

Sessions used to be JSON, and those still open, settings only: a three-tab
one, one tab's bare state, or one saved from the old single-window continuum
tool -- a :class:`~supernova.gui._state.ContinuumState` that may carry the
equilibrium file it loaded.
"""

from __future__ import annotations

from dataclasses import asdict, fields
import json
from pathlib import Path

import numpy as np


class JsonState:
    """Mixin for a state dataclass: dict and JSON round trips, strict on keys."""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown session keys: {', '.join(unknown)}")
        return cls(**data)

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def copy(self):
        return type(self).from_dict(json.loads(json.dumps(self.to_dict())))


# the tab each section of an application session belongs to
SECTIONS = ("equilibrium", "continuum", "solver")
FORMAT = "supernova-session"
VERSION = 1
COMPRESS_BYTES = 1 << 16          # arrays larger than this go in compressed


def write_session(path, *, results: dict | None = None, **states) -> None:
    """Write the tabs' states and what they computed to one HDF5 file.

    ``states`` are the sections' state dataclasses. ``results`` maps a section
    to what it computed, every entry optional: ``{"continuum": {"result",
    "equilibrium_key"}, "solver": {"result", "refinement", "selected", "log",
    "equilibrium_key"}}``. The file is written next to ``path`` and moved
    over it at the end, so a save that fails leaves an older session whole.
    """
    import h5py

    unknown = sorted(set(states) - set(SECTIONS)) + sorted(set(results or {}) - set(SECTIONS))
    if unknown:
        raise ValueError(f"Unknown session sections: {', '.join(unknown)}")
    path = Path(path)
    partial = path.with_name(path.name + ".part")
    try:
        with h5py.File(partial, "w", track_order=True) as h5:
            h5.attrs["format"] = FORMAT
            h5.attrs["version"] = VERSION
            for name, state in states.items():
                h5.create_group(name, track_order=True).attrs["state"] = _json(asdict(state))
            for name, computed in (results or {}).items():
                computed = {key: value for key, value in computed.items() if value is not None}
                if not computed:
                    continue
                group = h5.require_group(name)
                if "equilibrium_key" in computed:
                    group.attrs["equilibrium_key"] = _json(list(computed["equilibrium_key"]))
                if "selected" in computed:
                    group.attrs["selected"] = int(computed["selected"])
                if computed.get("log"):
                    _put_text(group, "log", computed["log"])
                if "result" in computed:
                    writer = _write_continuum if name == "continuum" else _write_solver_result
                    writer(group.create_group("result", track_order=True), computed["result"])
                if "refinement" in computed:
                    _write_refinement(group.create_group("refinement", track_order=True),
                                      computed["refinement"])
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def read_session(path) -> dict:
    """Return ``{section: state}`` for whatever the file holds.

    An HDF5 session's results come back under ``"results"``, as
    :func:`write_session` takes them, each ``equilibrium_key`` a tuple again.
    A JSON session has no results. One whose keys are not section names is a
    single tab's bare state; that of the old single-window continuum tool may
    carry its equilibrium inputs, which come back as ``"continuum_equilibrium"``,
    an Equilibrium tab state, unless the session has an equilibrium section of
    its own.
    """
    import h5py

    from supernova.gui._equilibrium import EquilibriumState
    from supernova.gui._solver import SolverState
    from supernova.gui._state import ContinuumState

    if Path(path).is_file() and h5py.is_hdf5(str(path)):
        return _read_hdf5(path)
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a session.")
    if not set(data) & set(SECTIONS):
        return _read_bare_state(data)

    unknown = sorted(set(data) - set(SECTIONS))
    if unknown:
        raise ValueError(f"Unknown session sections: {', '.join(unknown)}")
    loaded = {}
    if "equilibrium" in data:
        loaded["equilibrium"] = EquilibriumState.from_dict(data["equilibrium"])
    if "continuum" in data:
        loaded["continuum"] = ContinuumState.from_dict(data["continuum"])
        legacy = _legacy_equilibrium(data["continuum"])
        if legacy is not None and "equilibrium" not in data:
            loaded["continuum_equilibrium"] = legacy
    if "solver" in data:
        loaded["solver"] = SolverState.from_dict(data["solver"])
    return loaded


def _read_bare_state(data: dict) -> dict:
    """One tab's state on its own, as the tabs' buttons used to save them."""
    from supernova.gui._equilibrium import EquilibriumState
    from supernova.gui._solver import SolverState
    from supernova.gui._state import ContinuumState

    try:
        loaded = {"continuum": ContinuumState.from_dict(data)}
    except (ValueError, TypeError) as continuum_error:
        for name, cls in (("equilibrium", EquilibriumState), ("solver", SolverState)):
            try:
                return {name: cls.from_dict(data)}
            except (ValueError, TypeError):
                pass
        raise continuum_error
    legacy = _legacy_equilibrium(data)
    if legacy is not None:
        loaded["continuum_equilibrium"] = legacy
    return loaded


def _legacy_equilibrium(data: dict):
    """The Equilibrium tab's state for an old continuum session's inputs, or None.

    Before the continuum tab took its equilibrium from the Equilibrium tab it
    loaded its own, and its sessions carry the file, COCOS, Boozer domain and
    density for that.
    """
    from supernova.gui._equilibrium import EquilibriumState
    from supernova.gui._state import split_legacy

    _, legacy = split_legacy(data)
    if not legacy.get("eqdsk"):
        return None
    if "species" in data:
        legacy["species"] = data["species"]       # the ions belong with the density now
    return EquilibriumState(source="eqdsk", **legacy)


# ----------------------------------------------------------------------------
# HDF5
# ----------------------------------------------------------------------------

def _read_hdf5(path) -> dict:
    import h5py

    from supernova.gui._equilibrium import EquilibriumState
    from supernova.gui._solver import SolverState
    from supernova.gui._state import ContinuumState

    classes = {"equilibrium": EquilibriumState, "continuum": ContinuumState,
               "solver": SolverState}
    loaded, results = {}, {}
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("format") != FORMAT:
            raise ValueError(f"{path} is an HDF5 file but not a superNOVA session.")
        if int(h5.attrs.get("version", 0)) > VERSION:
            raise ValueError(f"{path} was written by a newer superNOVA.")
        unknown = sorted(set(h5) - set(SECTIONS))
        if unknown:
            raise ValueError(f"Unknown session sections: {', '.join(unknown)}")
        for name, group in h5.items():
            if "state" in group.attrs:
                loaded[name] = classes[name].from_dict(json.loads(group.attrs["state"]))
            computed = {}
            if "equilibrium_key" in group.attrs:
                computed["equilibrium_key"] = tuple(json.loads(group.attrs["equilibrium_key"]))
            if "selected" in group.attrs:
                computed["selected"] = int(group.attrs["selected"])
            if "log" in group:
                computed["log"] = _get_text(group["log"])
            if "result" in group:
                reader = _read_continuum if name == "continuum" else _read_solver_result
                computed["result"] = reader(group["result"])
            if "refinement" in group:
                computed["refinement"] = _read_refinement(group["refinement"])
            if computed:
                results[name] = computed
    if results:
        loaded["results"] = results
    return loaded


def _json(value) -> str:
    return json.dumps(value, default=_plain)


def _plain(value):
    """What json cannot write itself: numpy scalars and arrays."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return [value.real, value.imag]
    raise TypeError(f"{type(value).__name__} does not go into a session")


def _put(group, name: str, values):
    """An array as a dataset; text as variable-length strings, big ones compressed."""
    import h5py

    values = np.asarray(values)
    if values.dtype.kind == "U":
        stored = group.create_dataset(name, data=values.astype(object),
                                      dtype=h5py.string_dtype())
        stored.attrs["text"] = True
        return stored
    options = {}
    if values.ndim and values.nbytes > COMPRESS_BYTES:
        options = dict(compression="gzip", compression_opts=4, shuffle=True)
    return group.create_dataset(name, data=values, **options)


def _get(stored) -> np.ndarray:
    if stored.attrs.get("text"):
        return np.asarray(stored.asstr()[()], dtype=str)
    return stored[()]


def _put_text(group, name: str, text: str) -> None:
    """Text too long for an attribute (they stop at 64 kB)."""
    import h5py

    group.create_dataset(name, data=str(text), dtype=h5py.string_dtype())


def _get_text(stored) -> str:
    return stored.asstr()[()]


def _write_xarray(group, dataset) -> None:
    group.attrs["attrs"] = _json(dataset.attrs)
    for name, variable in dataset.variables.items():
        stored = _put(group, str(name), variable.values)
        stored.attrs["dims"] = _json(list(variable.dims))
        stored.attrs["attrs"] = _json(variable.attrs)
        stored.attrs["coordinate"] = name in dataset.coords


def _read_xarray(group):
    import xarray as xr

    coords, data = {}, {}
    for name, stored in group.items():
        variable = xr.Variable(tuple(json.loads(stored.attrs["dims"])), _get(stored),
                               attrs=json.loads(stored.attrs["attrs"]))
        (coords if stored.attrs["coordinate"] else data)[name] = variable
    return xr.Dataset(data, coords=coords, attrs=json.loads(group.attrs["attrs"]))


def _write_continuum(group, result) -> None:
    for item in fields(result):
        value = getattr(result, item.name)
        if isinstance(value, np.ndarray):
            _put(group, item.name, value)
        elif value is not None:
            group.attrs[item.name] = value


def _read_continuum(group):
    from supernova.gui._model import ContinuumResult

    values = {name: _get(stored) for name, stored in group.items()}
    values.update({name: value.item() if isinstance(value, np.generic) else value
                   for name, value in group.attrs.items()})
    return ContinuumResult(**values)


# SolverResult's per-mode and grid arrays, stored as they are
_RESULT_ARRAYS = ("frequency_khz", "omega", "center", "width", "dominant_m",
                  "backward_error", "window_of", "x", "m_pol", "envelope_x",
                  "continuum_x", "continuum_khz")


def _write_solver_result(group, result) -> None:
    """A pass, with the eigenpairs split by the solver window that found them."""
    group.attrs["state"] = _json(result.state.to_dict())
    group.attrs["title"] = result.title
    group.attrs["seconds"] = float(result.seconds)
    group.attrs["psi_core"] = float(result.psi_core)
    group.attrs["psi_edge"] = float(result.psi_edge)
    group.attrs["walls"] = np.asarray(result.walls, dtype=float)
    for name in _RESULT_ARRAYS:
        _put(group, name, getattr(result, name))
    counts = np.asarray([np.size(c) for c in result.crossings], dtype=int)
    _put(group, "crossings", np.concatenate([np.ravel(c) for c in result.crossings])
         if counts.size else np.empty(0))
    _put(group, "crossing_counts", counts)
    fields_group = group.create_group("fields", track_order=True)
    for name, values in result.fields.items():
        _put(fields_group, name, values)
    if result.located is not None:
        _write_xarray(group.create_group("located", track_order=True), result.located)

    dataset = result.dataset
    found_by = (np.asarray(dataset.coords["window"].values)
                if dataset is not None and "window" in dataset.coords else None)
    split = found_by is not None and set(found_by.tolist()) <= {row.window for row in result.windows}
    windows = group.create_group("windows", track_order=True)
    for row in result.windows:
        window = windows.create_group(f"w{row.window:03d}", track_order=True)
        window.attrs["row"] = _json(asdict(row))
        mine = np.flatnonzero(found_by == row.window) if split else np.empty(0, dtype=int)
        if mine.size:
            _write_xarray(window.create_group("eigenpairs", track_order=True),
                          dataset.isel(eigenpair=mine))
    if dataset is not None and not split:
        # eigenpairs no listed window claims: keep them whole
        _write_xarray(group.create_group("eigenpairs", track_order=True), dataset)


def _read_solver_result(group):
    import xarray as xr

    from supernova.gui._solver import SolverResult, SolverState, WindowRow

    rows, parts = [], []
    for window in group["windows"].values():
        row = json.loads(window.attrs["row"])
        row["range_khz"] = tuple(row["range_khz"])
        rows.append(WindowRow(**row))
        if "eigenpairs" in window:
            parts.append(_read_xarray(window["eigenpairs"]))
    if "eigenpairs" in group:
        dataset = _read_xarray(group["eigenpairs"])
    elif parts:
        dataset = xr.concat(parts, dim="eigenpair", data_vars="all", coords="minimal",
                            compat="override", join="exact",
                            combine_attrs="override").sortby("eigenpair")
    else:
        dataset = None
    counts = _get(group["crossing_counts"])
    crossings = (tuple(np.split(_get(group["crossings"]), np.cumsum(counts)[:-1]))
                 if counts.size else ())
    return SolverResult(
        state=SolverState.from_dict(json.loads(group.attrs["state"])),
        title=str(group.attrs["title"]), windows=tuple(rows), dataset=dataset,
        located=_read_xarray(group["located"]) if "located" in group else None,
        crossings=crossings,
        fields={name: _get(stored) for name, stored in group["fields"].items()},
        psi_core=float(group.attrs["psi_core"]), psi_edge=float(group.attrs["psi_edge"]),
        walls=tuple(float(wall) for wall in group.attrs["walls"]),
        seconds=float(group.attrs["seconds"]), equilibrium=None,
        **{name: _get(group[name]) for name in _RESULT_ARRAYS})


def _write_refinement(group, refinement) -> None:
    group.attrs["state"] = _json(refinement.state.to_dict())
    group.attrs["title"] = refinement.title
    group.attrs["modes"] = np.asarray(refinement.modes, dtype=int)
    group.attrs["seconds"] = float(refinement.seconds)
    _put_text(group, "rows", _json(list(refinement.rows)))
    _put_text(group, "path", _json({str(mode): test for mode, test in refinement.path.items()}))
    _put(group, "x", refinement.x)
    envelopes = group.create_group("envelopes", track_order=True)
    for mode, values in refinement.envelopes.items():
        _put(envelopes, str(mode), values)


def _read_refinement(group):
    from supernova.gui._solver import RefineResult, SolverState

    return RefineResult(
        state=SolverState.from_dict(json.loads(group.attrs["state"])),
        title=str(group.attrs["title"]),
        modes=tuple(int(mode) for mode in group.attrs["modes"]),
        rows=tuple(json.loads(_get_text(group["rows"]))), x=_get(group["x"]),
        envelopes={int(mode): _get(stored) for mode, stored in group["envelopes"].items()},
        path={int(mode): test for mode, test in json.loads(_get_text(group["path"])).items()},
        seconds=float(group.attrs["seconds"]))
