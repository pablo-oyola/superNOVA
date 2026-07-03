r"""
Small solver3 edge-boundary check.

This mirrors the style of ``no_sync/test_boundary.py`` but isolates the
solver3 four-field ideal-separatrix condition:

    xi^psi_m(psi=1) = 0

The script verifies that:

1. the right B-spline endpoint is represented by one radial DoF,
2. the boundary matrix C reads only the last-radial xi^psi coefficients,
3. every retained poloidal mode is constrained in the correct m ordering,
4. states with nonzero edge xi^psi fail the BC by exactly that edge value,
5. sparse project/expand zeroes only the constrained xi^psi edge DoFs.
"""

from pathlib import Path
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "supernova-matplotlib"))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import spsolve

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from supernova.core._boundary3 import (
    boundary_basis_values,
    endpoint_is_single_dof,
    idealSeparatrixBC,
)
from supernova.radial import BSplineFE

plt.ion()


def eval_fe(fe, coeffs, x):
    """
    Evaluate a possibly complex B-spline coefficient vector.
    """
    return fe.eval_basis(x).T @ coeffs


def eval_component_modes(fe, state, cols, x):
    """
    Evaluate one component block for all retained modes.
    """
    return fe.eval_basis(x).T @ state[:, cols]


# %% Configuration.
mmax = 5
ndofs = 50
korder = 3


# %% Building the finite element space.
knots = np.linspace(0.0, 1.0, ndofs - korder + 2)
fe = BSplineFE(knots=knots, order=korder, domain=[0.0, 1.0], quad_order=6)


# %% Build the solver3 edge boundary condition.
m_range = np.arange(-mmax, mmax + 1)
n_modes = m_range.size
n_radial = fe.get_ndofs()
n_state = 4 * n_modes

p1cols = np.arange(0, n_modes)
xipsicols = np.arange(n_modes, 2 * n_modes)
xiscols = np.arange(2 * n_modes, 3 * n_modes)
divxicols = np.arange(3 * n_modes, 4 * n_modes)

bc_edge = idealSeparatrixBC(fe, n_modes)


# %% Check the endpoint trace and constrained columns.
psi_edge, phi_edge = boundary_basis_values(fe, side="right")
edge_support = np.flatnonzero(np.abs(phi_edge) > 1e-14)
expected_constrained = (n_radial - 1) * n_state + xipsicols

print("right edge psi:", psi_edge)
print("right edge support:", edge_support)
print("right edge basis values:", phi_edge[edge_support])
print("endpoint is single DoF:", endpoint_is_single_dof(fe, side="right"))
print("BC C shape:", bc_edge.C.shape)
print("expected constrained columns:", expected_constrained)
print("actual constrained columns:", bc_edge.constrained)

if not endpoint_is_single_dof(fe, side="right"):
    raise AssertionError("This check expects a clamped right-endpoint B-spline basis.")

if not np.array_equal(bc_edge.constrained, expected_constrained):
    raise AssertionError("idealSeparatrixBC is not constraining last-radial xi^psi columns.")

for row, col in enumerate(expected_constrained):
    row_cols = np.flatnonzero(np.abs(bc_edge.C[row]) > 0.0)
    if row_cols.size != 1 or row_cols[0] != col or bc_edge.C[row, col] != 1.0:
        raise AssertionError(f"Unexpected C row {row}: nonzero columns {row_cols}.")

constrained_radial = bc_edge.constrained // n_state
constrained_state = bc_edge.constrained % n_state
constrained_mode_index = constrained_state - n_modes

print("constrained radial indices:", constrained_radial)
print("constrained state columns:", constrained_state)
print("constrained mode indices:", constrained_mode_index)
print("constrained m values:", m_range[constrained_mode_index])

if not np.array_equal(constrained_radial, np.full(n_modes, n_radial - 1)):
    raise AssertionError("Not every constrained column is on the last radial DoF.")

if not np.array_equal(constrained_mode_index, np.arange(n_modes)):
    raise AssertionError("Edge BC constrained columns are not ordered like m_range.")


# %% Check mode ordering with deliberately unique per-mode edge sentinels.
ordering_state = np.zeros((n_radial, n_state), dtype=np.complex128)
mode_index = np.arange(n_modes, dtype=np.float64)
xipsi_markers = (100.0 + mode_index) + 1.0j * (200.0 - m_range.astype(float))

# Fill other components and nearby radial entries with different sentinels.
# This makes a swapped component, swapped mode, or wrong radial index visible.
ordering_state[-1, p1cols] = (1000.0 + mode_index) + 1.0j * (1100.0 + mode_index)
ordering_state[-1, xipsicols] = xipsi_markers
ordering_state[-1, xiscols] = (2000.0 + mode_index) + 1.0j * (2100.0 + mode_index)
ordering_state[-1, divxicols] = (3000.0 + mode_index) + 1.0j * (3100.0 + mode_index)
ordering_state[-2, xipsicols] = (-100.0 - mode_index) + 1.0j * (-200.0 - mode_index)

ordering_residual = bc_edge.C @ ordering_state.ravel(order="C")
ordering_projected = bc_edge.expand_sparse(
    bc_edge.project_sparse(ordering_state.ravel(order="C"))
).reshape(ordering_state.shape, order="C")

print("mode-ordering check:")
for row, m_value in enumerate(m_range):
    print(
        f"  row={row:02d}, m={int(m_value):+d}, "
        f"residual={ordering_residual[row]}, expected={xipsi_markers[row]}"
    )

if not np.allclose(ordering_residual, xipsi_markers, atol=1e-12, rtol=0.0):
    raise AssertionError("C @ state does not preserve the m_range ordering of xi^psi edge values.")

if not np.allclose(ordering_projected[-1, xipsicols], 0.0, atol=1e-12, rtol=0.0):
    raise AssertionError("Sparse project/expand did not zero all xi^psi edge modes.")

if not np.allclose(
    ordering_projected[bc_edge.free_mask_2d],
    ordering_state[bc_edge.free_mask_2d],
    atol=1e-12,
    rtol=0.0,
):
    raise AssertionError("Sparse project/expand changed unconstrained mode/component entries.")


# %% Build a deliberately bad four-field state with nonzero xi^psi at the edge.
rho = np.linspace(0.0, 1.0, 200)

p1_profile = 0.2 + np.exp(-((rho - 0.45) ** 2) / 0.08**2)
xipsi_profile = 0.25 + 0.55 * rho + 0.15 * np.sin(2 * np.pi * rho)
xis_profile = 0.1 + np.sin(np.pi * rho) ** 2
divxi_profile = 0.3 + (rho - 0.35) ** 2

ak_p1 = fe.project(p1_profile, x=rho)
ak_xipsi = fe.project(xipsi_profile, x=rho)
ak_xis = fe.project(xis_profile, x=rho)
ak_divxi = fe.project(divxi_profile, x=rho)

states_bad = np.zeros((n_radial, n_state), dtype=np.complex128)
for j, m in enumerate(m_range):
    scale = 1.0 + 0.08 * (m - m_range.min())
    states_bad[:, p1cols[j]] = scale * ak_p1
    states_bad[:, xipsicols[j]] = scale * ak_xipsi
    states_bad[:, xiscols[j]] = scale * ak_xis
    states_bad[:, divxicols[j]] = scale * ak_divxi

states_good = states_bad.copy()
states_good[-1, xipsicols] = 0.0


# %% Cx should be exactly the edge xi^psi value for each m.
res_bad = bc_edge.C @ states_bad.ravel(order="C")
res_good = bc_edge.C @ states_good.ravel(order="C")
expected_bad_residual = states_bad[-1, xipsicols]

print("max |bad residual - xi^psi_edge|:", np.max(np.abs(res_bad - expected_bad_residual)))
print("max |good residual|:", np.max(np.abs(res_good)))

if not np.allclose(res_bad, expected_bad_residual, atol=1e-12, rtol=0.0):
    raise AssertionError("C @ state does not equal xi^psi at the edge.")

if not np.allclose(res_good, 0.0, atol=1e-12, rtol=0.0):
    raise AssertionError("A state with xi^psi edge coefficients set to zero fails the BC.")


# %% Check the sparse project/expand path used by solver3.
z_sparse = bc_edge.project_sparse(states_bad.ravel(order="C"))
states_projected = bc_edge.expand_sparse(z_sparse).reshape(states_bad.shape, order="C")

projected_residual = bc_edge.C @ states_projected.ravel(order="C")
unconstrained_mask = bc_edge.free_mask_2d
max_unconstrained_change = np.max(
    np.abs(states_projected[unconstrained_mask] - states_bad[unconstrained_mask])
)
max_constrained_value = np.max(np.abs(states_projected[-1, xipsicols]))

P = bc_edge.get_sparse_prolongation()

print("sparse prolongation shape:", P.shape)
print("sparse prolongation nnz:", P.nnz)
print("sparse prolongation is selector:", P.nnz == P.shape[1])
print("max |projected residual|:", np.max(np.abs(projected_residual)))
print("max unconstrained change after sparse project/expand:", max_unconstrained_change)
print("max constrained xi^psi edge value after sparse project/expand:", max_constrained_value)

if P.nnz != P.shape[1]:
    raise AssertionError("Edge sparse prolongation should be a pure selector for this basis.")

if not np.allclose(projected_residual, 0.0, atol=1e-12, rtol=0.0):
    raise AssertionError("Sparse project/expand did not enforce the edge BC.")

if not np.isclose(max_unconstrained_change, 0.0, atol=1e-12, rtol=0.0):
    raise AssertionError("Sparse project/expand changed unconstrained coefficients.")

if not np.isclose(max_constrained_value, 0.0, atol=1e-12, rtol=0.0):
    raise AssertionError("Sparse project/expand left constrained edge coefficients nonzero.")


# %% Plot the basis trace at the right edge.
fig_basis, ax_basis = plt.subplots()
ax_basis.plot(np.arange(n_radial), phi_edge.real, "ko-", label=r"$\phi_k(1)$")
ax_basis.axvline(n_radial - 1, color="tab:red", ls="--", label="last radial DoF")
ax_basis.set_xlabel("radial basis index")
ax_basis.set_ylabel("basis value at edge")
ax_basis.set_title("Right-edge B-spline trace")
ax_basis.grid(True, alpha=0.3)
ax_basis.legend()


# %% Plot the before/after profiles for one representative mode.
mode_to_plot = 2
mode_index = np.flatnonzero(m_range == mode_to_plot)
if mode_index.size == 0:
    mode_index = np.array([n_modes // 2])
mode_index = int(mode_index[0])

plot_specs = [
    (p1cols[mode_index], r"$p_1$"),
    (xipsicols[mode_index], r"$\xi^\psi$"),
    (xiscols[mode_index], r"$\xi^s$"),
    (divxicols[mode_index], r"$\nabla\cdot\xi$"),
]

fig_state, ax_state = plt.subplots(nrows=2, ncols=2, sharex=True)
for ax, (col, label) in zip(ax_state.ravel(), plot_specs):
    before = eval_fe(fe, states_bad[:, col], rho)
    after = eval_fe(fe, states_projected[:, col], rho)

    ax.plot(rho, before.real, "r-", label="before edge BC")
    ax.plot(rho, after.real, "b--", label="after sparse edge BC")
    ax.plot([1.0], [before[-1].real], "ro")
    ax.plot([1.0], [after[-1].real], "bo")
    ax.set_title(label)
    ax.grid(True, alpha=0.3)

ax_state[1, 0].set_xlabel(r"$\psi$")
ax_state[1, 1].set_xlabel(r"$\psi$")
ax_state[0, 0].set_ylabel("coefficient expansion")
ax_state[1, 0].set_ylabel("coefficient expansion")
ax_state[0, 0].legend()
fig_state.suptitle(
    rf"solver3 edge BC for $m={int(m_range[mode_index])}$: only $\xi^\psi(1)$ is zeroed"
)
fig_state.tight_layout()


# %% Plot all xi^psi edge values before and after applying the BC.
fig_edge, ax_edge = plt.subplots()
ax_edge.plot(m_range, states_bad[-1, xipsicols].real, "ro-", label="before")
ax_edge.plot(m_range, states_projected[-1, xipsicols].real, "bo--", label="after")
ax_edge.axhline(0.0, color="0.4", lw=0.8)
ax_edge.set_xlabel(r"$m$")
ax_edge.set_ylabel(r"$\xi^\psi_m(1)$ coefficient")
ax_edge.set_title(r"All constrained $\xi^\psi$ edge coefficients")
ax_edge.grid(True, alpha=0.3)
ax_edge.legend()


# %% Plot the explicit multi-m ordering sentinel.
fig_ordering, ax_ordering = plt.subplots()
ax_ordering.plot(m_range, ordering_residual.real, "o-", label=r"$\Re(Cx)$")
ax_ordering.plot(m_range, xipsi_markers.real, "k--", label=r"expected $\Re\,\xi^\psi_m(1)$")
ax_ordering.set_xlabel(r"$m$")
ax_ordering.set_ylabel("real sentinel value")
ax_ordering.set_title(r"Mode ordering check for edge $\xi^\psi$ constraint")
ax_ordering.grid(True, alpha=0.3)
ax_ordering.legend()


# %% Mimic the current sparse solver BC workflow with a simple operator T.
#
# Solver3 applies the boundary condition to the sparse operator and vectors as:
#
#     T_reduced = P^H T_full P
#     b_reduced = P^H b_full
#     z         = solve(T_reduced, b_reduced)
#     x_full    = P z
#
# The toy T below is only a diagonally shifted radial tridiagonal operator.  It
# is not a physics operator; it just exercises the same linear-algebra path.
radial_diag = np.full(n_radial, 2.0, dtype=np.complex128)
radial_offdiag = np.full(n_radial - 1, -0.35, dtype=np.complex128)
radial_T = diags(
    [radial_offdiag, radial_diag, radial_offdiag],
    offsets=[-1, 0, 1],
    shape=(n_radial, n_radial),
    dtype=np.complex128,
).tocsr()

state_weight = np.ones(n_state, dtype=np.complex128)
state_weight[p1cols] = 0.5 + 0.02 * np.arange(n_modes)
state_weight[xipsicols] = 0.8 + 0.03 * np.arange(n_modes)
state_weight[xiscols] = 1.1 + 0.04 * np.arange(n_modes)
state_weight[divxicols] = 1.4 + 0.05 * np.arange(n_modes)

T_full = (
    kron(radial_T, identity(n_state, dtype=np.complex128), format="csr")
    + kron(
        identity(n_radial, dtype=np.complex128),
        diags(state_weight, 0, shape=(n_state, n_state), dtype=np.complex128),
        format="csr",
    )
).tocsr()

rho_dof = np.linspace(0.0, 1.0, n_radial)
b_state = np.zeros((n_radial, n_state), dtype=np.complex128)
for j, m in enumerate(m_range):
    width = 0.12 + 0.01 * j
    center = 0.45 + 0.015 * m
    smooth_drive = np.exp(-((rho_dof - center) ** 2) / width**2)
    edge_drive = rho_dof**2
    scale = 1.0 + 0.12 * j

    b_state[:, p1cols[j]] = 0.4 * scale * smooth_drive
    b_state[:, xipsicols[j]] = scale * (smooth_drive + 0.35 * edge_drive)
    b_state[:, xiscols[j]] = 0.25 * scale * np.sin(np.pi * rho_dof)
    b_state[:, divxicols[j]] = 0.15 * scale * (1.0 - rho_dof)

# Add a deliberately nonzero xipsi edge drive for every m.  The constrained
# solution must still have xi^psi_m(1)=0 after the reduced solve.
b_state[-1, xipsicols] += 5.0 + np.arange(n_modes)

x_unconstrained = spsolve(T_full, b_state.ravel(order="C")).reshape(b_state.shape, order="C")

T_reduced = bc_edge.reduce_sparse_matrix(T_full)
b_reduced = bc_edge.project_sparse(b_state.ravel(order="C"))
z_reduced = spsolve(T_reduced, b_reduced)
x_algorithm = bc_edge.expand_sparse(z_reduced).reshape(b_state.shape, order="C")

full_residual = T_full @ x_algorithm.ravel(order="C") - b_state.ravel(order="C")
projected_residual = bc_edge.project_sparse(full_residual)
reduced_residual = T_reduced @ z_reduced - b_reduced
algorithm_bc_residual = bc_edge.C @ x_algorithm.ravel(order="C")

print("toy T full shape:", T_full.shape)
print("toy T reduced shape:", T_reduced.shape)
print("toy reduced residual norm:", np.linalg.norm(reduced_residual))
print("toy projected full residual norm:", np.linalg.norm(projected_residual))
print("toy edge BC residual norm:", np.linalg.norm(algorithm_bc_residual))
print("toy unconstrained xi^psi edge max:", np.max(np.abs(x_unconstrained[-1, xipsicols])))
print("toy constrained xi^psi edge max:", np.max(np.abs(x_algorithm[-1, xipsicols])))

if not np.allclose(reduced_residual, 0.0, atol=1e-10, rtol=0.0):
    raise AssertionError("Toy reduced solve did not satisfy T_reduced z = b_reduced.")

if not np.allclose(projected_residual, 0.0, atol=1e-10, rtol=0.0):
    raise AssertionError("Toy expanded solution does not satisfy the projected full residual.")

if not np.allclose(algorithm_bc_residual, 0.0, atol=1e-12, rtol=0.0):
    raise AssertionError("Toy expanded solution does not satisfy the edge BC.")


# %% Plot every xi^psi mode through the toy solver workflow.
xipsi_unconstrained = eval_component_modes(fe, x_unconstrained, xipsicols, rho)
xipsi_algorithm = eval_component_modes(fe, x_algorithm, xipsicols, rho)

norm = mpl.colors.Normalize(vmin=float(m_range.min()), vmax=float(m_range.max()))
cmap = plt.get_cmap("viridis")

fig_toy, ax_toy = plt.subplots(ncols=2, sharex=True, sharey=True, constrained_layout=True)
for j, m in enumerate(m_range):
    color = cmap(norm(float(m)))
    ax_toy[0].plot(rho, xipsi_unconstrained[:, j].real, color=color, lw=1.2)
    ax_toy[1].plot(rho, xipsi_algorithm[:, j].real, color=color, lw=1.2)

for ax in ax_toy:
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.axvline(1.0, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel(r"$\psi$")
    ax.grid(True, alpha=0.3)

ax_toy[0].set_ylabel(r"$\Re\,\xi^\psi_m$")
ax_toy[0].set_title("Toy T solve without BC")
ax_toy[1].set_title(r"Toy $P^H T P$ solve, expanded")

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig_toy.colorbar(sm, ax=ax_toy.ravel().tolist(), label=r"$m$")
fig_toy.suptitle(r"All-mode $\xi^\psi$ profiles through the sparse edge-BC workflow")


# %% Heatmap version for all xi^psi modes.
fig_toy_heat, ax_toy_heat = plt.subplots(ncols=2, sharex=True, sharey=True, constrained_layout=True)
extent = [rho[0], rho[-1], m_range[0], m_range[-1]]
vmax = max(
    float(np.max(np.abs(xipsi_unconstrained.real))),
    float(np.max(np.abs(xipsi_algorithm.real))),
)

im0 = ax_toy_heat[0].imshow(
    xipsi_unconstrained.real.T,
    origin="lower",
    aspect="auto",
    extent=extent,
    cmap="RdBu_r",
    vmin=-vmax,
    vmax=vmax,
)
ax_toy_heat[1].imshow(
    xipsi_algorithm.real.T,
    origin="lower",
    aspect="auto",
    extent=extent,
    cmap="RdBu_r",
    vmin=-vmax,
    vmax=vmax,
)

ax_toy_heat[0].set_title("without BC")
ax_toy_heat[1].set_title(r"after $P^H T P$ edge BC")
for ax in ax_toy_heat:
    ax.set_xlabel(r"$\psi$")
    ax.set_ylabel(r"$m$")
fig_toy_heat.colorbar(im0, ax=ax_toy_heat.ravel().tolist(), label=r"$\Re\,\xi^\psi_m$")
fig_toy_heat.suptitle(r"All-mode $\xi^\psi$ heatmap")
