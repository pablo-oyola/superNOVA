r"""
Split-rational sparse NEP solver for the linear ideal MHD problem.

This module keeps the callback/finite-difference NEP path in ``solver2.py``
untouched and introduces a split representation in the variable

    s = omega**2

with

    T(s) = T0 + s*T1 + (1/s)*Tminus1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.sparse import coo_array, coo_matrix, csr_matrix

from .solver2 import superSolver2
from ._boundary3 import BoundaryCondition, idealSeparatrixBC, regularityAxisBC
from ._petsc_helpers import (
    NEPResult,
    _require_petsc,
    _require_slepc,
    _vec_read_array,
    apply_petsc_options,
    scipy_csr_to_petsc,
)
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
        _offset_coo_tensor(phase, 0, 0, xip1_shape, "phase"),
        _offset_coo_tensor(phase, nmode, nmode, xip1_shape, "phase"),
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


def _project_local_tensor(
    sb: solver_blocks2,
    M: coo_array,
    include_derivative: bool = False,
) -> csr_matrix:
    r"""
    Project a local tensor to a global CSR matrix.

    The Galerkin term contributes ``-M``. If ``include_derivative`` is true,
    the weak radial derivative is added for the first two state groups.
    """
    M = _as_coo_tensor(M, "M")
    if M.shape[1] != M.shape[2]:
        raise ValueError("M must be square in its state dimensions.")

    nq, n_state, _ = M.shape
    nmode = sb.m_range.size
    expected_state = 4 * nmode
    if n_state != expected_state:
        raise ValueError(f"M state size must be {expected_state}, got {n_state}.")

    if sb.phi_q is None:
        raise ValueError("sb.phi_q must be set before assembling split matrices.")
    if sb.weights_q is None:
        raise ValueError("sb.weights_q must be set before assembling split matrices.")
    if include_derivative and sb.WD is None:
        raise ValueError("sb.WD must be set before assembling the derivative term.")

    phi_q = np.asarray(sb.phi_q)
    weights_q = np.asarray(sb.weights_q)
    if phi_q.ndim != 2:
        raise ValueError("phi_q must have shape (n_quad, n_radial).")
    if phi_q.shape[0] != nq:
        raise ValueError("M and phi_q must use the same quadrature points.")
    if weights_q.shape != (nq,):
        raise ValueError(f"weights_q must have shape {(nq,)}, got {weights_q.shape}.")

    n_radial = phi_q.shape[1]
    rows = []
    cols = []
    data = []

    if include_derivative:
        if sb.WD.shape != (n_radial, n_radial):
            raise ValueError(f"WD must have shape {(n_radial, n_radial)}, got {sb.WD.shape}.")
        wd = coo_matrix(sb.WD)
        if wd.nnz:
            derivative_states = np.arange(2 * nmode, dtype=np.int64)
            rows.append(
                np.repeat(wd.row, derivative_states.size) * n_state
                + np.tile(derivative_states, wd.nnz)
            )
            cols.append(
                np.repeat(wd.col, derivative_states.size) * n_state
                + np.tile(derivative_states, wd.nnz)
            )
            data.append(np.repeat(wd.data.astype(np.complex128), derivative_states.size))

    q_coords, local_rows, local_cols = M.coords
    local_data = M.data.astype(np.complex128, copy=False)
    local_keep = local_data != 0.0
    if np.any(local_keep):
        q_coords = q_coords[local_keep]
        local_rows = local_rows[local_keep]
        local_cols = local_cols[local_keep]
        local_data = local_data[local_keep]

        for q in np.unique(q_coords):
            radial_support = np.flatnonzero(phi_q[q] != 0.0)
            if radial_support.size == 0:
                continue

            q_mask = q_coords == q
            local_q_rows = local_rows[q_mask]
            local_q_cols = local_cols[q_mask]
            local_q_data = local_data[q_mask]
            if local_q_data.size == 0:
                continue

            radial_rows = np.repeat(radial_support, radial_support.size)
            radial_cols = np.tile(radial_support, radial_support.size)
            radial_data = (
                weights_q[q]
                * phi_q[q, radial_rows]
                * phi_q[q, radial_cols]
            ).astype(np.complex128, copy=False)

            rows.append((radial_rows[:, None] * n_state + local_q_rows[None, :]).ravel())
            cols.append((radial_cols[:, None] * n_state + local_q_cols[None, :]).ravel())
            data.append(-(radial_data[:, None] * local_q_data[None, :]).ravel())

    size = n_radial * n_state
    if not data:
        return csr_matrix((size, size), dtype=np.complex128)

    return coo_matrix(
        (
            np.concatenate(data),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(size, size),
        dtype=np.complex128,
    ).tocsr()


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


@dataclass
class SplitRationalNEPSolver:
    """
    Thin SLEPc NEP wrapper for a split rational sparse operator.
    """

    T0: csr_matrix
    T1: csr_matrix
    Tminus1: csr_matrix
    target_s: complex
    region: dict
    prefix: str = "nova_split_nep"
    which: str = "TARGET_MAGNITUDE"
    nev: int = 6
    tol: float = 1e-8
    maxit: int = 100
    options: dict | None = None
    comm: object | None = None
    spectral_shift: complex | None = None
    verbose: bool = False
    compute_residuals: bool = True

    def __post_init__(self):
        petsc = _require_petsc()
        slepc = _require_slepc()

        self.T0 = self.T0.tocsr().astype(np.complex128, copy=False)
        self.T1 = self.T1.tocsr().astype(np.complex128, copy=False)
        self.Tminus1 = self.Tminus1.tocsr().astype(np.complex128, copy=False)

        if self.T0.shape[0] != self.T0.shape[1]:
            raise ValueError(f"T0 must be square, got {self.T0.shape}.")
        if self.T1.shape != self.T0.shape:
            raise ValueError(f"T1 shape {self.T1.shape} does not match T0 {self.T0.shape}.")
        if self.Tminus1.shape != self.T0.shape:
            raise ValueError(
                f"Tminus1 shape {self.Tminus1.shape} does not match T0 {self.T0.shape}."
            )

        self.target_s = _as_scalar_complex(self.target_s, "target_s")
        self.region = _ellipse_region_dict(self.region, self.target_s)
        if self.spectral_shift is None:
            self.spectral_shift = self.region["center"]
        self.spectral_shift = _as_scalar_complex(self.spectral_shift, "spectral_shift")
        if self.spectral_shift == 0.0 and self.Tminus1.nnz:
            raise ValueError(
                "SLEPc NLEIGS rejects a 1/s pole at the eigenvalue origin. "
                "Use a nonzero spectral_shift."
            )
        self.target_z = self.target_s - self.spectral_shift
        self.slepc_region = dict(self.region)
        self.slepc_region["center"] = self.region["center"] - self.spectral_shift
        self.size = self.T0.shape[0]
        self.comm = petsc.COMM_SELF if self.comm is None else self.comm
        self.scalar_type = np.dtype(petsc.ScalarType)

        self._vprint("[SplitRationalNEPSolver] initializing")
        self._vprint(f"  target_s={_format_complex(self.target_s)}")
        self._vprint(f"  spectral_shift={_format_complex(self.spectral_shift)}")
        self._vprint(f"  slepc_target_z={_format_complex(self.target_z)}")
        self._vprint(
            "  public_region="
            f"type={self.region['type']}, center={_format_complex(self.region['center'])}, "
            f"radius={self.region['radius']:.6g}, vscale={self.region['vscale']:.6g}"
        )
        self._vprint(
            "  slepc_region="
            f"type={self.slepc_region['type']}, center={_format_complex(self.slepc_region['center'])}, "
            f"radius={self.slepc_region['radius']:.6g}, vscale={self.slepc_region['vscale']:.6g}"
        )
        self._vprint(f"  options: prefix={self.prefix!r}, which={self.which}, nev={self.nev}, tol={self.tol}, maxit={self.maxit}")
        self._vprint(f"  {_matrix_stats('T0', self.T0)}")
        self._vprint(f"  {_matrix_stats('T1', self.T1)}")
        self._vprint(f"  {_matrix_stats('Tminus1', self.Tminus1)}")

        # SLEPc sees z = s - spectral_shift. The public split matrices remain
        # T0 + s*T1 + (1/s)*Tminus1, but internally this becomes
        # (T0 + shift*T1) + z*T1 + 1/(z + shift)*Tminus1.
        self.T0_slepc = (self.T0 + self.spectral_shift * self.T1).tocsr()
        self.mats = [
            scipy_csr_to_petsc(self.T0_slepc, comm=self.comm),
            scipy_csr_to_petsc(self.T1, comm=self.comm),
            scipy_csr_to_petsc(self.Tminus1, comm=self.comm),
        ]
        self.fns = [
            self._make_rational_fn([1.0]),
            self._make_rational_fn([1.0, 0.0]),
            self._make_rational_fn([1.0], denominator=[1.0, self.spectral_shift]),
        ]
        self.rg = self._make_ellipse_region(self.slepc_region)

        self.nep = slepc.NEP().create(comm=self.comm)
        self.nep.setOptionsPrefix(self.prefix)
        self.nep.setType(slepc.NEP.Type.NLEIGS)
        self.nep.setProblemType(slepc.NEP.ProblemType.RATIONAL)
        self.nep.setSplitOperator(
            self.mats,
            self.fns,
            petsc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
        )
        self.nep.setRG(self.rg)
        self.nep.setDimensions(int(self.nev))
        self.nep.setTarget(self.target_z)
        self.nep.setWhichEigenpairs(self._resolve_nleigs_which(slepc, self.which))
        self.nep.setTolerances(float(self.tol), int(self.maxit))

        if self.options:
            apply_petsc_options(self.prefix, **self.options)
            self._vprint(f"  petsc_options={self.options}")
        self.nep.setFromOptions()

    def _vprint(self, message: str):
        """
        Print a diagnostic line when verbose mode is enabled.
        """
        if self.verbose:
            print(message)

    @staticmethod
    def _resolve_enum(enum_cls, value):
        """
        Resolve a SLEPc enum from a string or pass enum values through.
        """
        if value is None or not isinstance(value, str):
            return value
        key = value.upper()
        if not hasattr(enum_cls, key):
            choices = ", ".join(name for name in dir(enum_cls) if name.isupper())
            raise ValueError(f"Unknown enum value {value!r}. Available values: {choices}")
        return getattr(enum_cls, key)

    @classmethod
    def _resolve_nleigs_which(cls, slepc, value):
        """
        Resolve and validate NLEIGS eigenvalue selection.
        """
        which = cls._resolve_enum(slepc.NEP.Which, value)
        allowed = (
            slepc.NEP.Which.TARGET_MAGNITUDE,
            slepc.NEP.Which.TARGET_REAL,
            slepc.NEP.Which.TARGET_IMAGINARY,
        )
        if not any(which == item for item in allowed):
            allowed_names = "TARGET_MAGNITUDE, TARGET_REAL, TARGET_IMAGINARY"
            raise ValueError(
                "SplitRationalNEPSolver uses SLEPc NEP.Type.NLEIGS, which "
                "supports only target selection of eigenvalues. Use one of "
                f"{allowed_names}; move the search with target_omega and region, "
                "and request more eigenpairs with nev/max_eigenpairs."
            )
        return which

    def _make_rational_fn(self, numerator, denominator=None):
        """
        Create a SLEPc rational FN with descending polynomial coefficients.
        """
        petsc = _require_petsc()
        slepc = _require_slepc()

        fn = slepc.FN().create(comm=self.comm)
        fn.setType(slepc.FN.Type.RATIONAL)
        fn.setRationalNumerator(np.asarray(numerator, dtype=petsc.ScalarType))
        if denominator is not None:
            fn.setRationalDenominator(np.asarray(denominator, dtype=petsc.ScalarType))
        fn.setFromOptions()
        return fn

    def _make_ellipse_region(self, region: dict):
        """
        Create the nontrivial ellipse region required by NLEIGS.
        """
        slepc = _require_slepc()

        rg = slepc.RG().create(comm=self.comm)
        rg.setType(slepc.RG.Type.ELLIPSE)
        rg.setEllipseParameters(region["center"], region["radius"], region["vscale"])
        rg.setFromOptions()
        return rg

    def solve(self, max_eigenpairs: int | None = None) -> NEPResult:
        """
        Solve the split NEP and return converged eigenpairs.
        """
        petsc = _require_petsc()

        self._vprint("[SplitRationalNEPSolver] calling NEPSolve()")
        self.nep.solve()
        nconv = int(self.nep.getConverged())
        nout = nconv if max_eigenpairs is None else min(nconv, int(max_eigenpairs))
        reason = self.nep.getConvergedReason() if hasattr(self.nep, "getConvergedReason") else None
        iterations = self.nep.getIterationNumber() if hasattr(self.nep, "getIterationNumber") else -1
        self._vprint(
            f"[SplitRationalNEPSolver] done: nconv={nconv}, nout={nout}, "
            f"iterations={int(iterations)}, reason={reason}"
        )

        eigenvalues = np.zeros(nout, dtype=np.complex128)
        eigenvectors = np.zeros((nout, self.size), dtype=np.complex128)
        self.last_residual_norms = np.full(nout, np.nan, dtype=np.float64)
        self.last_absolute_residual_norms = np.full(nout, np.nan, dtype=np.float64)

        vr = self.mats[0].createVecRight()
        real_scalars = not np.issubdtype(np.dtype(petsc.ScalarType), np.complexfloating)
        vi = self.mats[0].createVecRight() if real_scalars else None

        for ii in range(nout):
            eigenvalues[ii] = self.nep.getEigenpair(ii, vr, vi) + self.spectral_shift
            eigenvector = _vec_read_array(vr).astype(np.complex128, copy=True)
            if vi is not None:
                eigenvector += 1.0j * _vec_read_array(vi)
            eigenvectors[ii] = eigenvector

            if self.compute_residuals:
                abs_norm, rel_norm = _split_residual_norm(
                    self.T0,
                    self.T1,
                    self.Tminus1,
                    eigenvalues[ii],
                    eigenvector,
                )
                self.last_absolute_residual_norms[ii] = abs_norm
                self.last_residual_norms[ii] = rel_norm

            self._vprint(
                f"  eigenpair[{ii}]: "
                f"s={_format_complex(eigenvalues[ii])}, "
                f"omega={_format_complex(np.sqrt(eigenvalues[ii]))}, "
                f"residual={self.last_residual_norms[ii]:.3e}"
            )

        return NEPResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            nconv=nconv,
            reason=reason,
            iterations=int(iterations),
        )


@dataclass
class SplitRationalCISSNEPSolver:
    """
    SLEPc CISS wrapper for all split-rational eigenpairs inside a region.
    """

    T0: csr_matrix
    T1: csr_matrix
    Tminus1: csr_matrix
    target_s: complex
    region: dict
    prefix: str = "nova_split_ciss"
    which: str = "ALL"
    nev: int = 32
    tol: float = 1e-8
    maxit: int = 100
    ciss_sizes: dict | None = None
    ciss_extraction: str | None = None
    ciss_refinement: dict | None = None
    ciss_threshold: dict | None = None
    options: dict | None = None
    comm: object | None = None
    verbose: bool = False
    compute_residuals: bool = True

    def __post_init__(self):
        petsc = _require_petsc()
        slepc = _require_slepc()

        self.T0 = self.T0.tocsr().astype(np.complex128, copy=False)
        self.T1 = self.T1.tocsr().astype(np.complex128, copy=False)
        self.Tminus1 = self.Tminus1.tocsr().astype(np.complex128, copy=False)

        if self.T0.shape[0] != self.T0.shape[1]:
            raise ValueError(f"T0 must be square, got {self.T0.shape}.")
        if self.T1.shape != self.T0.shape:
            raise ValueError(f"T1 shape {self.T1.shape} does not match T0 {self.T0.shape}.")
        if self.Tminus1.shape != self.T0.shape:
            raise ValueError(
                f"Tminus1 shape {self.Tminus1.shape} does not match T0 {self.T0.shape}."
            )

        self.target_s = _as_scalar_complex(self.target_s, "target_s")
        self.region = _ellipse_region_dict(self.region, self.target_s)
        self.size = self.T0.shape[0]
        self.comm = petsc.COMM_SELF if self.comm is None else self.comm
        self.scalar_type = np.dtype(petsc.ScalarType)

        self._vprint("[SplitRationalCISSNEPSolver] initializing")
        self._vprint(f"  target_s={_format_complex(self.target_s)}")
        self._vprint(
            "  region="
            f"type={self.region['type']}, center={_format_complex(self.region['center'])}, "
            f"radius={self.region['radius']:.6g}, vscale={self.region['vscale']:.6g}"
        )
        self._vprint(f"  options: prefix={self.prefix!r}, which={self.which}, nev={self.nev}, tol={self.tol}, maxit={self.maxit}")
        self._vprint(f"  {_matrix_stats('T0', self.T0)}")
        self._vprint(f"  {_matrix_stats('T1', self.T1)}")
        self._vprint(f"  {_matrix_stats('Tminus1', self.Tminus1)}")

        self.mats = [
            scipy_csr_to_petsc(self.T0, comm=self.comm),
            scipy_csr_to_petsc(self.T1, comm=self.comm),
            scipy_csr_to_petsc(self.Tminus1, comm=self.comm),
        ]
        self.fns = [
            self._make_rational_fn([1.0]),
            self._make_rational_fn([1.0, 0.0]),
            self._make_rational_fn([1.0], denominator=[1.0, 0.0]),
        ]
        self.rg = self._make_ellipse_region(self.region)

        self.nep = slepc.NEP().create(comm=self.comm)
        self.nep.setOptionsPrefix(self.prefix)
        self.nep.setType(slepc.NEP.Type.CISS)
        self.nep.setProblemType(slepc.NEP.ProblemType.RATIONAL)
        self.nep.setSplitOperator(
            self.mats,
            self.fns,
            petsc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
        )
        self.nep.setRG(self.rg)
        self.nep.setDimensions(int(self.nev))
        self.nep.setTarget(self.target_s)
        self.nep.setWhichEigenpairs(self._resolve_ciss_which(slepc, self.which))
        self.nep.setTolerances(float(self.tol), int(self.maxit))

        if self.ciss_sizes:
            self.nep.setCISSSizes(**self.ciss_sizes)
            self._vprint(f"  ciss_sizes={self.ciss_sizes}")
        if self.ciss_extraction is not None:
            extraction = SplitRationalNEPSolver._resolve_enum(
                slepc.NEP.CISSExtraction,
                self.ciss_extraction,
            )
            self.nep.setCISSExtraction(extraction)
            self._vprint(f"  ciss_extraction={self.ciss_extraction}")
        if self.ciss_refinement:
            self.nep.setCISSRefinement(**self.ciss_refinement)
            self._vprint(f"  ciss_refinement={self.ciss_refinement}")
        if self.ciss_threshold:
            self.nep.setCISSThreshold(**self.ciss_threshold)
            self._vprint(f"  ciss_threshold={self.ciss_threshold}")

        if self.options:
            apply_petsc_options(self.prefix, **self.options)
            self._vprint(f"  petsc_options={self.options}")
        self.nep.setFromOptions()

        try:
            self.ciss_sizes_used = self.nep.getCISSSizes()
        except Exception:  # pragma: no cover - depends on SLEPc runtime
            self.ciss_sizes_used = None

    def _vprint(self, message: str):
        """
        Print a diagnostic line when verbose mode is enabled.
        """
        if self.verbose:
            print(message)

    @classmethod
    def _resolve_ciss_which(cls, slepc, value):
        """
        CISS is a contour method, so only region-wide ALL selection is exposed.
        """
        which = SplitRationalNEPSolver._resolve_enum(slepc.NEP.Which, value)
        if which != slepc.NEP.Which.ALL:
            raise ValueError(
                "SplitRationalCISSNEPSolver is the region-complete path and "
                "uses which='ALL'. Use solve_nep_split() with NLEIGS for "
                "target-ranked searches."
            )
        return which

    def _make_rational_fn(self, numerator, denominator=None):
        """
        Create a SLEPc rational FN with descending polynomial coefficients.
        """
        petsc = _require_petsc()
        slepc = _require_slepc()

        fn = slepc.FN().create(comm=self.comm)
        fn.setType(slepc.FN.Type.RATIONAL)
        fn.setRationalNumerator(np.asarray(numerator, dtype=petsc.ScalarType))
        if denominator is not None:
            fn.setRationalDenominator(np.asarray(denominator, dtype=petsc.ScalarType))
        fn.setFromOptions()
        return fn

    def _make_ellipse_region(self, region: dict):
        """
        Create the ellipse contour used by CISS.
        """
        slepc = _require_slepc()

        rg = slepc.RG().create(comm=self.comm)
        rg.setType(slepc.RG.Type.ELLIPSE)
        rg.setEllipseParameters(region["center"], region["radius"], region["vscale"])
        rg.setFromOptions()
        return rg

    def solve(self, max_eigenpairs: int | None = None) -> NEPResult:
        """
        Solve the split NEP with CISS and return converged eigenpairs.
        """
        petsc = _require_petsc()

        self._vprint("[SplitRationalCISSNEPSolver] calling NEPSolve()")
        self.nep.solve()
        nconv = int(self.nep.getConverged())
        nout = nconv if max_eigenpairs is None else min(nconv, int(max_eigenpairs))
        reason = self.nep.getConvergedReason() if hasattr(self.nep, "getConvergedReason") else None
        iterations = self.nep.getIterationNumber() if hasattr(self.nep, "getIterationNumber") else -1
        self._vprint(
            f"[SplitRationalCISSNEPSolver] done: nconv={nconv}, nout={nout}, "
            f"iterations={int(iterations)}, reason={reason}"
        )

        eigenvalues = np.zeros(nout, dtype=np.complex128)
        eigenvectors = np.zeros((nout, self.size), dtype=np.complex128)
        self.last_residual_norms = np.full(nout, np.nan, dtype=np.float64)
        self.last_absolute_residual_norms = np.full(nout, np.nan, dtype=np.float64)

        vr = self.mats[0].createVecRight()
        real_scalars = not np.issubdtype(np.dtype(petsc.ScalarType), np.complexfloating)
        vi = self.mats[0].createVecRight() if real_scalars else None

        for ii in range(nout):
            eigenvalues[ii] = self.nep.getEigenpair(ii, vr, vi)
            eigenvector = _vec_read_array(vr).astype(np.complex128, copy=True)
            if vi is not None:
                eigenvector += 1.0j * _vec_read_array(vi)
            eigenvectors[ii] = eigenvector

            if self.compute_residuals:
                abs_norm, rel_norm = _split_residual_norm(
                    self.T0,
                    self.T1,
                    self.Tminus1,
                    eigenvalues[ii],
                    eigenvector,
                )
                self.last_absolute_residual_norms[ii] = abs_norm
                self.last_residual_norms[ii] = rel_norm

            self._vprint(
                f"  eigenpair[{ii}]: "
                f"s={_format_complex(eigenvalues[ii])}, "
                f"omega={_format_complex(np.sqrt(eigenvalues[ii]))}, "
                f"residual={self.last_residual_norms[ii]:.3e}"
            )

        return NEPResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            nconv=nconv,
            reason=reason,
            iterations=int(iterations),
        )


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

    def _build_nep_boundary_condition(
        self,
        sb: solver_blocks2,
        bc_kind: str | None = "full",
    ) -> BoundaryCondition:
        """
        Build the sparse-compatible four-field BC for the solver3 state.
        """
        n_modes = sb.m_range.size
        kind = self._normalize_bc_kind(bc_kind)

        if kind == "none":
            n_full = self.radialfe.get_ndofs() * 4 * n_modes
            return BoundaryCondition(np.zeros((0, n_full), dtype=np.complex128))

        if kind == "axis":
            return regularityAxisBC(self.radialfe, sb.m_range)

        if kind == "edge":
            return idealSeparatrixBC(self.radialfe, n_modes)

        bc_axis = regularityAxisBC(self.radialfe, sb.m_range)
        bc_sep = idealSeparatrixBC(self.radialfe, n_modes)
        return bc_axis * bc_sep

    def _species_mass_amu(self) -> float:
        """
        Species mass factor used in the inherited solver-block density.
        """
        return float(self.cont.mass.to("amu").v)

    def assemble_split_matrices(
        self,
        ntor: int,
        mmax: int | None = None,
        max_dm: int = 10,
        regularizer: float = 0.0,
        apply_bcs: bool = True,
        bc_kind: str | None = "full",
        bc_tol: float | None = None,
        verbose: bool = False,
    ) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
        r"""
        Assemble ``T0``, ``T1`` and ``Tminus1`` in ``T(s)`` as CSR matrices.
        """
        sb = self.build_solver_blocks(ntor=ntor, mmax=mmax, max_dm=max_dm)
        M0, M1, Mminus1 = _build_local_M_split(sb, regularizer=regularizer)

        nq = np.asarray(sb.weights_q).shape[0]
        M0_q = _slice_to_quadrature(M0, nq, "M0")
        M1_q = _slice_to_quadrature(M1, nq, "M1")
        Mminus1_q = _slice_to_quadrature(Mminus1, nq, "Mminus1")

        T0 = _project_local_tensor(sb, M0_q, include_derivative=True)
        T1 = _project_local_tensor(sb, M1_q, include_derivative=False)
        Tminus1 = _project_local_tensor(sb, Mminus1_q, include_derivative=False)

        if verbose:
            print("[superSolver3] assembled split matrices before BC reduction")
            print(f"  ntor={ntor}, m_range=[{sb.m_range[0]}, {sb.m_range[-1]}], n_modes={sb.m_range.size}")
            print(f"  n_radial={np.asarray(sb.phi_q).shape[1]}, n_quad={np.asarray(sb.weights_q).shape[0]}")
            print(f"  {_matrix_stats('T0', T0)}")
            print(f"  {_matrix_stats('T1', T1)}")
            print(f"  {_matrix_stats('Tminus1', Tminus1)}")

        bc_kind = self._normalize_bc_kind(bc_kind)
        if not apply_bcs or bc_kind == "none":
            return T0, T1, Tminus1

        bc = self._build_nep_boundary_condition(sb, bc_kind=bc_kind)
        T0_reduced, T1_reduced, Tminus1_reduced = (
            bc.reduce_sparse_matrix(T0, tol=bc_tol),
            bc.reduce_sparse_matrix(T1, tol=bc_tol),
            bc.reduce_sparse_matrix(Tminus1, tol=bc_tol),
        )
        if verbose:
            print("[superSolver3] split matrices after sparse BC reduction")
            print(
                f"  full_dofs={T0.shape[0]}, reduced_dofs={T0_reduced.shape[0]}, "
                f"bc_kind={bc_kind}, bc_tol={bc_tol}"
            )
            print(f"  {_matrix_stats('T0_reduced', T0_reduced)}")
            print(f"  {_matrix_stats('T1_reduced', T1_reduced)}")
            print(f"  {_matrix_stats('Tminus1_reduced', Tminus1_reduced)}")
        return T0_reduced, T1_reduced, Tminus1_reduced

    def build_split_nep_solver(
        self,
        ntor: int,
        target_omega: complex,
        region: dict | None,
        mmax: int | None = None,
        max_dm: int = 10,
        regularizer: float = 0.0,
        apply_bcs: bool = True,
        bc_kind: str | None = "full",
        bc_tol: float | None = None,
        prefix: str = "nova_split_nep",
        which: str = "TARGET_MAGNITUDE",
        nev: int = 6,
        tol: float = 1e-8,
        maxit: int = 100,
        options: dict | None = None,
        spectral_shift: complex | None = None,
        verbose: bool = False,
        compute_residuals: bool = True,
    ) -> SplitRationalNEPSolver:
        """
        Create the SLEPc NLEIGS solver for the split rational operator.
        """
        target_omega = _as_scalar_complex(target_omega, "target_omega")
        target_s = target_omega**2
        region = _ellipse_region_dict(region, target_s)
        T0, T1, Tminus1 = self.assemble_split_matrices(
            ntor=ntor,
            mmax=mmax,
            max_dm=max_dm,
            regularizer=regularizer,
            apply_bcs=apply_bcs,
            bc_kind=bc_kind,
            bc_tol=bc_tol,
            verbose=verbose,
        )
        return SplitRationalNEPSolver(
            T0=T0,
            T1=T1,
            Tminus1=Tminus1,
            target_s=target_s,
            region=region,
            prefix=prefix,
            which=which,
            nev=nev,
            tol=tol,
            maxit=maxit,
            options=options,
            spectral_shift=spectral_shift,
            verbose=verbose,
            compute_residuals=compute_residuals,
        )

    def build_split_ciss_solver(
        self,
        ntor: int,
        region: dict,
        target_omega: complex | None = None,
        mmax: int | None = None,
        max_dm: int = 10,
        regularizer: float = 0.0,
        apply_bcs: bool = True,
        bc_kind: str | None = "full",
        bc_tol: float | None = None,
        prefix: str = "nova_split_ciss",
        which: str = "ALL",
        nev: int = 32,
        tol: float = 1e-8,
        maxit: int = 100,
        ciss_sizes: dict | None = None,
        ciss_extraction: str | None = None,
        ciss_refinement: dict | None = None,
        ciss_threshold: dict | None = None,
        options: dict | None = None,
        verbose: bool = False,
        compute_residuals: bool = True,
    ) -> SplitRationalCISSNEPSolver:
        """
        Create the SLEPc CISS solver for all split-rational eigenpairs in ``region``.
        """
        if target_omega is None:
            if region is None or "center" not in region:
                raise ValueError("target_omega is optional for CISS only when region['center'] is given.")
            target_s = _as_scalar_complex(region["center"], "region['center']")
        else:
            target_omega = _as_scalar_complex(target_omega, "target_omega")
            target_s = target_omega**2
        region = _ellipse_region_dict(region, target_s)
        T0, T1, Tminus1 = self.assemble_split_matrices(
            ntor=ntor,
            mmax=mmax,
            max_dm=max_dm,
            regularizer=regularizer,
            apply_bcs=apply_bcs,
            bc_kind=bc_kind,
            bc_tol=bc_tol,
            verbose=verbose,
        )
        return SplitRationalCISSNEPSolver(
            T0=T0,
            T1=T1,
            Tminus1=Tminus1,
            target_s=target_s,
            region=region,
            prefix=prefix,
            which=which,
            nev=nev,
            tol=tol,
            maxit=maxit,
            ciss_sizes=ciss_sizes,
            ciss_extraction=ciss_extraction,
            ciss_refinement=ciss_refinement,
            ciss_threshold=ciss_threshold,
            options=options,
            verbose=verbose,
            compute_residuals=compute_residuals,
        )

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

    def solve_nep_split(
        self,
        ntor: int,
        target_omega: complex,
        region: dict | None,
        mmax: int | None = None,
        max_dm: int = 10,
        regularizer: float = 0.0,
        apply_bcs: bool = True,
        bc_kind: str | None = "full",
        bc_tol: float | None = None,
        prefix: str = "nova_split_nep",
        which: str = "TARGET_MAGNITUDE",
        nev: int = 6,
        tol: float = 1e-8,
        maxit: int = 100,
        options: dict | None = None,
        max_eigenpairs: int | None = None,
        spectral_shift: complex | None = None,
        verbose: bool = False,
        compute_residuals: bool = True,
    ) -> xr.Dataset:
        """
        Solve ``T(s) x = 0`` with ``s = omega**2`` using SLEPc NLEIGS.
        """
        target_omega = _as_scalar_complex(target_omega, "target_omega")
        target_s = target_omega**2
        region = _ellipse_region_dict(region, target_s)
        nep = self.build_split_nep_solver(
            ntor=ntor,
            target_omega=target_omega,
            region=region,
            mmax=mmax,
            max_dm=max_dm,
            regularizer=regularizer,
            apply_bcs=apply_bcs,
            bc_kind=bc_kind,
            bc_tol=bc_tol,
            prefix=prefix,
            which=which,
            nev=nev,
            tol=tol,
            maxit=maxit,
            options=options,
            spectral_shift=spectral_shift,
            verbose=verbose,
            compute_residuals=compute_residuals,
        )
        result = nep.solve(max_eigenpairs=max_eigenpairs)
        return self._split_result_dataset(
            result=result,
            nep=nep,
            ntor=ntor,
            target_omega=target_omega,
            target_s=target_s,
            region=region,
            mmax=mmax,
            max_dm=max_dm,
            regularizer=regularizer,
            apply_bcs=apply_bcs,
            bc_kind=bc_kind,
            bc_tol=bc_tol,
            verbose=verbose,
            compute_residuals=compute_residuals,
            which=which,
            nep_type="NLEIGS",
            description="SLEPc NLEIGS split-rational NEP solution of sparse solver3 T(s) x = 0.",
        )

    def solve_nep_split_ciss(
        self,
        ntor: int,
        region: dict,
        target_omega: complex | None = None,
        mmax: int | None = None,
        max_dm: int = 10,
        regularizer: float = 0.0,
        apply_bcs: bool = True,
        bc_kind: str | None = "full",
        bc_tol: float | None = None,
        prefix: str = "nova_split_ciss",
        which: str = "ALL",
        nev: int = 32,
        tol: float = 1e-8,
        maxit: int = 100,
        ciss_sizes: dict | None = None,
        ciss_extraction: str | None = None,
        ciss_refinement: dict | None = None,
        ciss_threshold: dict | None = None,
        options: dict | None = None,
        max_eigenpairs: int | None = None,
        verbose: bool = False,
        compute_residuals: bool = True,
    ) -> xr.Dataset:
        """
        Solve ``T(s) x = 0`` for all CISS eigenpairs inside ``region``.

        CISS is the contour-integral pathway.  The region is an ellipse in the
        public variable ``s = omega**2`` and must not enclose the pole at zero.
        """
        if target_omega is None:
            if region is None or "center" not in region:
                raise ValueError("target_omega is optional for CISS only when region['center'] is given.")
            target_s = _as_scalar_complex(region["center"], "region['center']")
        else:
            target_omega = _as_scalar_complex(target_omega, "target_omega")
            target_s = target_omega**2
        region = _ellipse_region_dict(region, target_s)
        nep = self.build_split_ciss_solver(
            ntor=ntor,
            target_omega=target_omega,
            region=region,
            mmax=mmax,
            max_dm=max_dm,
            regularizer=regularizer,
            apply_bcs=apply_bcs,
            bc_kind=bc_kind,
            bc_tol=bc_tol,
            prefix=prefix,
            which=which,
            nev=nev,
            tol=tol,
            maxit=maxit,
            ciss_sizes=ciss_sizes,
            ciss_extraction=ciss_extraction,
            ciss_refinement=ciss_refinement,
            ciss_threshold=ciss_threshold,
            options=options,
            verbose=verbose,
            compute_residuals=compute_residuals,
        )
        result = nep.solve(max_eigenpairs=max_eigenpairs)
        return self._split_result_dataset(
            result=result,
            nep=nep,
            ntor=ntor,
            target_omega=target_omega,
            target_s=target_s,
            region=region,
            mmax=mmax,
            max_dm=max_dm,
            regularizer=regularizer,
            apply_bcs=apply_bcs,
            bc_kind=bc_kind,
            bc_tol=bc_tol,
            verbose=verbose,
            compute_residuals=compute_residuals,
            which=which,
            nep_type="CISS",
            description="SLEPc CISS split-rational NEP solution of sparse solver3 T(s) x = 0.",
            extra_attrs={
                "region_complete_method": True,
                "ciss_sizes": str(ciss_sizes),
                "ciss_sizes_used": str(getattr(nep, "ciss_sizes_used", None)),
                "ciss_extraction": ciss_extraction,
                "ciss_refinement": str(ciss_refinement),
                "ciss_threshold": str(ciss_threshold),
            },
        )
