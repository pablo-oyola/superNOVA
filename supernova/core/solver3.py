r"""
Split-rational sparse NEP solver for the linear ideal MHD problem.

This module keeps the callback/finite-difference NEP path in ``solver2.py``
untouched and introduces a split representation in the variable

    s = omega**2

with

    T(s) = T0 + s*T1 + (1/s)*Tminus1.
"""

from __future__ import annotations


import numpy as np
import xarray as xr
from scipy.sparse import coo_array, csr_matrix

from .solver2 import superSolver2
from ._petsc_helpers import NEPResult
from ._solver_kernel2 import (
    mu_0,
    solver_blocks2,
    _as_coo_tensor,
    _filter_zero_data,
    _combine_coo_tensors,
    _scale_coo_axis0,
    _offset_coo_tensor,
)


def _as_scalar_complex(value, name: str) -> complex:
    """
    Convert a scalar or one-element array to a Python complex scalar.
    """
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"{name} must be a scalar or one-element array, got shape {arr.shape}.")
    return complex(arr.reshape(-1)[0])


def _format_complex(value: complex) -> str:
    """
    Compact scalar formatting for verbose solver diagnostics.
    """
    value = complex(value)
    if abs(value.imag) < 1e-14:
        return f"{value.real:.16g}"
    return f"{value.real:.16g}{value.imag:+.16g}j"


def _matrix_stats(name: str, matrix: csr_matrix) -> str:
    """
    Return a one-line sparse matrix summary.
    """
    density = matrix.nnz / max(1, matrix.shape[0] * matrix.shape[1])
    return f"{name}: shape={matrix.shape}, nnz={matrix.nnz}, density={density:.3e}"


def _validate_split_block_shapes(sb: solver_blocks2):
    """
    Check the local block shapes expected by the four-field solver state.
    """
    nmode = sb.m_range.size
    npsi = sb.C0.shape[0]
    nstates_xip1 = 2 * nmode

    xip1_shape = (npsi, nstates_xip1, nstates_xip1)
    harmonic_shape = (npsi, nmode, nmode)
    expected_shapes = {
        "C0": xip1_shape,
        "D0": xip1_shape,
        "F": xip1_shape,
        "A": xip1_shape,
        "B": xip1_shape,
        "C1": harmonic_shape,
        "D1": harmonic_shape,
        "gtp_over_gpp": harmonic_shape,
        "gpz_over_gpp": harmonic_shape,
    }
    for block_name, expected_shape in expected_shapes.items():
        tensor = _as_coo_tensor(getattr(sb, block_name), block_name)
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{block_name} must have shape {expected_shape}, got {tensor.shape}."
            )


def _select_coo_rows(
    tensor,
    row_start: int | None = None,
    row_stop: int | None = None,
    name: str = "tensor",
) -> coo_array:
    """
    Keep only entries whose local row is in ``[row_start, row_stop)``.
    """
    tensor = _as_coo_tensor(tensor, name)
    q, rows, cols = tensor.coords
    mask = np.ones(tensor.nnz, dtype=bool)
    if row_start is not None:
        mask &= rows >= int(row_start)
    if row_stop is not None:
        mask &= rows < int(row_stop)

    if not np.any(mask):
        return coo_array(tensor.shape, dtype=np.complex128)

    data = tensor.data.astype(np.complex128, copy=True)[mask]
    coords = (q[mask].copy(), rows[mask].copy(), cols[mask].copy())
    data, coords = _filter_zero_data(data, coords)
    return coo_array((data, coords), shape=tensor.shape, dtype=np.complex128)


def _slice_to_quadrature(tensor, nq: int, name: str) -> coo_array:
    """
    Drop the optional edge operator point and keep quadrature slices only.
    """
    tensor = _as_coo_tensor(tensor, name)
    if tensor.shape[0] == nq:
        return tensor
    if tensor.shape[0] == nq + 1:
        return tensor[:nq]
    raise ValueError(
        f"{name} must use {nq} quadrature points, optionally plus one edge point. "
        f"Got {tensor.shape[0]} local points."
    )


def _build_local_M_split(
    sb: solver_blocks2,
    regularizer: float = 0.0,
) -> tuple[coo_array, coo_array, coo_array]:
    r"""
    Build local COO tensors ``M0``, ``M1`` and ``Mminus1`` for ``M(s)``.

    The split follows ``lambda = mu_0 * density * s`` and preserves the
    four-field local state ``[p1, xi^psi, xi^s, div_xi]``.
    """
    _validate_split_block_shapes(sb)

    nmode = sb.m_range.size
    nstates_xip1 = 2 * nmode
    nstates = 4 * nmode
    npsi = sb.C0.shape[0]

    rho = mu_0 * np.asarray(sb.density, dtype=np.complex128)
    if rho.shape != (npsi,):
        raise ValueError(f"density must have shape {(npsi,)}, got {rho.shape}.")
    if np.any(rho == 0.0):
        raise ValueError("density cannot contain zero values in the split 1/s operator.")

    xip1_shape = (npsi, nstates_xip1, nstates_xip1)
    harmonic_shape = (npsi, nmode, nmode)
    full_shape = (npsi, nstates, nstates)
    ones = np.ones(npsi, dtype=np.complex128)

    phase = _combine_coo_tensors(harmonic_shape, sb.gtp_over_gpp, sb.gpz_over_gpp)
    C0 = _combine_coo_tensors(
        xip1_shape,
        sb.C0,
        _offset_coo_tensor(phase, 0, 0, xip1_shape, "phase", scale=-1.0),
        _offset_coo_tensor(
            phase,
            nmode,
            nmode,
            xip1_shape,
            "phase",
            scale=-1.0,
        ),
    )
    C1 = _offset_coo_tensor(
        _scale_coo_axis0(sb.C1, rho, "C1"),
        0,
        nmode,
        xip1_shape,
        "C1",
    )

    D0 = _as_coo_tensor(sb.D0, "D0")
    Dminus1 = _offset_coo_tensor(
        _scale_coo_axis0(sb.D1, 1.0 / rho, "D1"),
        nmode,
        nmode,
        xip1_shape,
        "D1",
    )
    F = _as_coo_tensor(sb.F, "F")

    r = (1.0 + 1.0j * regularizer) ** 2
    A_const = _scale_coo_axis0(sb.A, ones, "A", scale=-1.0)
    A_inverse = _scale_coo_axis0(sb.A, 1.0 / rho, "A", scale=-1.0)
    B_linear = _scale_coo_axis0(sb.B, rho, "B", scale=r)
    B_const = _scale_coo_axis0(sb.B, ones, "B", scale=r)

    E0 = _combine_coo_tensors(
        xip1_shape,
        _select_coo_rows(A_const, row_stop=nmode, name="A"),
        _select_coo_rows(B_const, row_start=nmode, name="B"),
    )
    E1 = _select_coo_rows(B_linear, row_stop=nmode, name="B")
    Eminus1 = _select_coo_rows(A_inverse, row_start=nmode, name="A")

    M0 = _combine_coo_tensors(
        full_shape,
        _offset_coo_tensor(C0, 0, 0, full_shape, "C0"),
        _offset_coo_tensor(D0, 0, nstates_xip1, full_shape, "D0"),
        _offset_coo_tensor(F, nstates_xip1, 0, full_shape, "F"),
        _offset_coo_tensor(E0, nstates_xip1, nstates_xip1, full_shape, "E0", scale=-1.0),
    )
    M1 = _combine_coo_tensors(
        full_shape,
        _offset_coo_tensor(C1, 0, 0, full_shape, "C1"),
        _offset_coo_tensor(E1, nstates_xip1, nstates_xip1, full_shape, "E1", scale=-1.0),
    )
    Mminus1 = _combine_coo_tensors(
        full_shape,
        _offset_coo_tensor(Dminus1, 0, nstates_xip1, full_shape, "Dminus1"),
        _offset_coo_tensor(
            Eminus1,
            nstates_xip1,
            nstates_xip1,
            full_shape,
            "Eminus1",
            scale=-1.0,
        ),
    )
    return M0, M1, Mminus1


def _ellipse_region_dict(region: dict | None, target_s: complex) -> dict:
    """
    Normalize and validate the v1 ellipse region description.
    """
    if target_s == 0.0:
        raise ValueError("target_omega must be nonzero because the split NEP contains 1/s.")

    if region is None:
        region = {
            "type": "ellipse",
            "center": target_s,
            "radius": 0.25 * abs(target_s),
            "vscale": 0.25,
        }

    if not isinstance(region, dict):
        raise ValueError("region must be a dictionary.")
    region_type = region.get("type", "ellipse")
    if str(region_type).lower() != "ellipse":
        raise ValueError("solver3 v1 supports only ellipse regions.")

    center = _as_scalar_complex(region.get("center", target_s), "region['center']")
    radius = float(region.get("radius", 0.25 * abs(target_s)))
    vscale = float(region.get("vscale", 1.0))
    if radius <= 0.0:
        raise ValueError("ellipse region radius must be positive.")
    if vscale <= 0.0:
        raise ValueError("ellipse region vscale must be positive.")

    scaled_zero_distance = (center.real / radius) ** 2 + (center.imag / (radius * vscale)) ** 2
    if scaled_zero_distance <= 1.0:
        raise ValueError("ellipse region must not include the pole at s=0.")

    return {
        "type": "ellipse",
        "center": center,
        "radius": radius,
        "vscale": vscale,
    }


def _split_residual_norm(
    T0: csr_matrix,
    T1: csr_matrix,
    Tminus1: csr_matrix,
    eigenvalue: complex,
    eigenvector: np.ndarray,
) -> tuple[float, float]:
    """
    Return absolute and vector-normalized residual norms for ``T(s) x``.
    """
    if eigenvalue == 0.0:
        return np.inf, np.inf

    residual = T0 @ eigenvector
    residual = residual + eigenvalue * (T1 @ eigenvector)
    residual = residual + (1.0 / eigenvalue) * (Tminus1 @ eigenvector)
    abs_norm = float(np.linalg.norm(residual))
    rel_norm = abs_norm / max(1.0, float(np.linalg.norm(eigenvector)))
    return abs_norm, rel_norm


class superSolver3(superSolver2):
    """
    Solver2 setup with a split-rational sparse NEP path in ``s = omega**2``.
    """

    @staticmethod
    def _normalize_bc_kind(bc_kind: str | None) -> str:
        """
        Normalize the boundary-condition selector used for diagnostics.
        """
        if bc_kind is None:
            return "full"

        aliases = {
            "both": "full",
            "all": "full",
            "separatrix": "edge",
            "edge_only": "edge",
            "axis_only": "axis",
            "false": "none",
            "off": "none",
            "no": "none",
        }
        kind = str(bc_kind).strip().lower()
        kind = aliases.get(kind, kind)
        valid = {"full", "axis", "edge", "none"}
        if kind not in valid:
            raise ValueError(f"bc_kind must be one of {sorted(valid)}, got {bc_kind!r}.")
        return kind

    def _species_mass_amu(self) -> float:
        """
        Species mass factor used in the inherited solver-block density.
        """
        return float(self.cont.mass.to("amu").v)

    def _split_result_dataset(
        self,
        result: NEPResult,
        nep,
        ntor: int,
        target_omega: complex | None,
        target_s: complex,
        region: dict,
        mmax: int | None,
        max_dm: int,
        regularizer: float,
        apply_bcs: bool,
        bc_kind: str | None,
        bc_tol: float | None,
        verbose: bool,
        compute_residuals: bool,
        which: str,
        nep_type: str,
        description: str,
        extra_attrs: dict | None = None,
    ) -> xr.Dataset:
        """
        Pack a split NEP result as an xarray dataset with full-space eigenvectors.
        """
        species_mass_amu = self._species_mass_amu()
        omega = np.sqrt(result.eigenvalues)

        solver_eigenvectors = result.eigenvectors
        slepc_dofs = solver_eigenvectors.shape[1]
        eigenvectors = solver_eigenvectors
        eigenvector_space = "full"
        slepc_eigenvector_space = "full"
        bc_expanded_eigenvectors = False
        bc_kind = self._normalize_bc_kind(bc_kind)
        if apply_bcs and bc_kind != "none":
            slepc_eigenvector_space = "reduced"
            sb = self.build_solver_blocks(ntor=ntor, mmax=mmax, max_dm=max_dm)
            bc = self._build_nep_boundary_condition(sb, bc_kind=bc_kind)
            full_dofs = self.radialfe.get_ndofs() * 4 * sb.m_range.size
            if solver_eigenvectors.shape[1] != full_dofs:
                eigenvectors = np.zeros(
                    (solver_eigenvectors.shape[0], full_dofs),
                    dtype=np.complex128,
                )
                for ii in range(solver_eigenvectors.shape[0]):
                    eigenvectors[ii] = bc.expand_sparse(solver_eigenvectors[ii], tol=bc_tol)
                bc_expanded_eigenvectors = True
                if verbose:
                    print(
                        "[superSolver3] expanded reduced eigenvectors to full space: "
                        f"reduced_dofs={slepc_dofs}, full_dofs={full_dofs}"
                    )
            elif verbose:
                print("[superSolver3] SLEPc eigenvectors already have full-space size.")

        eigenpair = np.arange(result.eigenvalues.size, dtype=np.int64)
        dof = np.arange(eigenvectors.shape[1], dtype=np.int64)
        data_vars = {
            "s": ("eigenpair", result.eigenvalues),
            "omega": ("eigenpair", omega),
            "eigenvector": (("eigenpair", "dof"), eigenvectors),
        }
        if getattr(nep, "last_residual_norms", None) is not None:
            data_vars["residual_norm"] = ("eigenpair", nep.last_residual_norms)
            data_vars["absolute_residual_norm"] = (
                "eigenpair",
                nep.last_absolute_residual_norms,
            )

        attrs = {
            "ntor": int(ntor),
            "split_variable": "omega2",
            "omega_normalization": "omegaA",
            "omega_is_species_mass_normalized": True,
            "density_includes_species_mass": True,
            "species_mass_amu": species_mass_amu,
            "target_omega": target_omega,
            "target_s": target_s,
            "mmax": None if mmax is None else int(mmax),
            "max_dm": int(max_dm),
            "regularizer": regularizer,
            "apply_bcs": bool(apply_bcs),
            "bc_kind": bc_kind,
            "region": str(region),
            "region_type": region["type"],
            "region_center": region["center"],
            "region_radius": region["radius"],
            "region_vscale": region["vscale"],
            "verbose": bool(verbose),
            "compute_residuals": bool(compute_residuals),
            "eigenvector_space": eigenvector_space,
            "slepc_eigenvector_space": slepc_eigenvector_space,
            "bc_expanded_eigenvectors": bool(bc_expanded_eigenvectors),
            "slepc_dofs": int(slepc_dofs),
            "full_dofs": int(eigenvectors.shape[1]),
            "residual_space": "reduced" if apply_bcs and bc_kind != "none" else "full",
            "nep_type": nep_type,
            "problem_type": "RATIONAL",
            "which": which,
            "nconv": int(result.nconv),
            "iterations": int(result.iterations),
            "reason": str(result.reason),
            "description": description,
        }
        if hasattr(nep, "spectral_shift"):
            attrs["spectral_shift"] = nep.spectral_shift
            attrs["slepc_variable"] = "s_minus_spectral_shift"
        else:
            attrs["slepc_variable"] = "s"
        if extra_attrs:
            attrs.update(extra_attrs)

        return xr.Dataset(
            data_vars=data_vars,
            coords={
                "eigenpair": eigenpair,
                "dof": dof,
            },
            attrs=attrs,
        )

