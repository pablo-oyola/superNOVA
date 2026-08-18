# Up-down-symmetry verification

This suite verifies the NOVA equilibrium and operator conventions without
changing the production Fourier basis

\[
e^{i(n\zeta-m\Theta)}
\]

or the production q-selected positive-\(m\) band.  The fitted ITPA diagnostic
uses the fixed-boundary TokaMaker EQDSK generated from the original analytic
ITPA functions,

\[
q=1.71+0.16(r/a)^2,\qquad
p(s)=7170-6811s-358.5s^2,
\]

not the separate equilibrium fitted with QSOLVER `outequ` pressure.

## What is tested in the production positive-m space

The following checks do not require negative perturbation harmonics:

- weighted grid parity of every angular equilibrium coefficient consumed by
  the continuum and four-field solver assembly, with explicit uniform
  periodic weights \(w_j=2\pi/N_\Theta\);
- independent parity of the stored one-sided Fourier amplitudes;
- the forbidden Fourier component (imaginary for an even field, real for an
  odd field);
- Hermitian Toeplitz lookup and the expected real-symmetric or
  imaginary-skew-symmetric structure of each equilibrium multiplication
  matrix, with decoded source coefficient, \(m,m'\), \(k=m-m'\), and radial
  location for the worst entry;
- local and global transformed-realness;
- boundary-prolongation identities, including explicit \(m\leftrightarrow-m\)
  intertwining of the mixed-space endpoint constraints and prolongation;
- exact preservation through SciPy sparse assembly and optional serial PETSc
  insertion.

`Profile` stores only the nonnegative representative \(A_m=a_m+i b_m\) of a
real equilibrium field.  For \(m>0\), the literal coefficients in a two-sided
series are \(A_m/2\) and \(A_m^*/2\).  Therefore raw complex Fourier
coefficients are expected.  Up-down symmetry instead requires

- `EVEN`: \(b_m=0\), so the stored amplitude is real;
- `ODD`: \(a_m=0\), so the stored amplitude is imaginary.

Every check records its absolute and normalized error, tolerance, pass/fail
state, and the worst radial/harmonic or matrix location.  `UNKNOWN` entries
remain visible and are never converted to passes.

The registry follows the live cylindrical-component convention.  In
particular, the local shear
\(\mathcal S=\mathbf s_N\mathbin{\cdot}\nabla\times\mathbf s_N\) is even:
\(\mathbf s_N\) and its curl both have `(odd, even, even)` cylindrical
component parities.  The Boozer gauge shift in
\(\zeta=\phi+\nu(\psi,\Theta)\) is odd, so
\(g^{\psi\zeta}=\nabla\psi\mathbin{\cdot}\nabla\zeta\) is odd in an
axisymmetric equilibrium (the \(\nabla\phi\) contribution is orthogonal to
\(\nabla\psi\)).

## Mixed solver4/solver5 operator

Solver4 and solver5 use a compatible mixed Petrov-Galerkin discretization.
The trial ordering is

```text
[p1_H, xi_psi_H, xi_s_L, div_xi_L]
```

while all four equation rows use the lower-order test space.  Before boundary
conditions, each \(T_k\) is rectangular.  Full endpoint constraints apply only
on the trial side as \(T_kP\).  Although this makes the dimensions square, it
does not identify the row basis with the reduced trial basis.

Consequently the suite does **not** claim ordinary Hermiticity,
\(T_k=T_k^\dagger\), or apply a one-space similarity transform to solver5.
Those would be mathematically unjustified for the current discretization.
Phase and reflection relations must instead be written between the two spaces:

\[
D_{\rm row}^{-1}T_kD_{\rm trial},\qquad
R_{\rm row}T_k=T_kR_{\rm trial}.
\]

No positive four-field kinetic-energy symmetrizer is currently implemented.
The positive B-spline overlap matrices are suitable for coefficient and
eigenspace norms, but they do not by themselves prove weighted adjointness of
the nonvariational first-order NOVA operator.  Global adjointness is therefore
reported as `UNKNOWN`, not failed or passed.

The live `_build_static_blocks` kernel also accumulates several analytic
contributions directly into `C0`, `D0`, and `F`.  The present suite checks
every source-coefficient Toeplitz matrix (with decoded \(m,m'\), source
harmonic, and radius) and checks the aggregate local and global operators.
It explicitly reports individual analytic C/D/F contribution-exchange gates
as `NOT_RUN`: exposing those terms without copying the production equations
requires a separate instrumentation refactor of that kernel.

## Real representation

Inspection of the live C/D/F equations gives the field phases

```text
p1       : 0
xi_psi   : 0
xi_s     : pi/2
div_xi   : 0
```

The C blocks are real.  The D11/D21 and F11/F12 blocks contain one explicit
factor of \(i\), while D12/D22 and F21/F22 are real.  Multiplying `xi_s` by
\(i\) makes these paired blocks real on an up-down-symmetric equilibrium.
The diagnostic also searches all quarter-turn field phases rather than
accepting the derived assignment only from documentation.  This inference is
performed on the midpoint local operator tensors and serves only to identify
the phase assignment.  The resulting assignment is checked separately at
every radial local operator point and on the raw and full-BC global mixed
matrices, so midpoint agreement does not replace the full transformed-realness
gates.

Realness checks use `regularizer=0`.  A nonzero solver regularizer intentionally
introduces the complex factor \((1+i\epsilon)^2\) and is not a symmetry error.

For a converged isolated eigenvector \(x\), the generic utility transforms it,
removes the global phase using \(x^T W x\) (with a largest-component fallback),
and measures its remaining imaginary norm.  Nearly degenerate modes are
handled as a \(W\)-orthonormal subspace: the reflection restricted to that
subspace is diagonalized before individual parity vectors are tested.

## Diagnostic doubled-m space

The low-level assembly kernels already accept explicit integer mode lists.
The verification interface exposes an uncached opt-in path for a set such as

```text
[-12, ..., -7, 7, ..., 12]
```

without populating or changing the production solver-block cache.  The primary
gate extracts the positive-positive block before and after boundary handling
and compares it entry-by-entry with the normal production assembly.

The doubled spectrum is not expected to duplicate the positive-\(m\) physical
spectrum: at fixed \(n\), the negative modes generally have remote
\(k_\parallel\).

### Why the full reflection commutator is currently unclassified

Cheng and Chance define

\[
\boldsymbol\xi =
\frac{\xi^\psi}{|\nabla\psi|^2}\nabla\psi+
\frac{\xi^s}{B^2}(\mathbf B\times\nabla\psi)+
\frac{\xi^b}{B^2}\mathbf B,
\]

so

\[
\xi^s=\boldsymbol\xi\cdot
\frac{\mathbf B\times\nabla\psi}{|\nabla\psi|^2}.
\]

Thus `xi_s` is a rescaled binormal projection, not a coordinate component.
The repository does not yet define how the perturbed vector is transformed
under midplane reflection, including the required polar/axial-vector
convention.  A scalar sign for `xi_s`, or a field-mixing matrix, cannot be
derived from the present implementation without guessing.  The suite provides
and unit-tests generic reflection, commutator, sector, eigenvector, and
eigenspace utilities, but the fitted-EQDSK report marks the physical
four-field commutator `UNKNOWN` until that transformation is supplied.

This limitation does not weaken the positive-subblock consistency gate.

## Tolerance classes

Separate tolerances are used for:

- exact index/permutation identities;
- algebraic sparse operations;
- angular-grid quadrature;
- one-sided Fourier parity;
- finite-element assembly;
- PETSc insertion;
- eigenvector/eigenspace diagnostics.

Analytically zero blocks use combined absolute and relative criteria.  The
fitted-EQDSK runner records convergence with angular sampling, radial
B-spline resolution, and quadrature order; it requires the final error to
meet its declared tolerance and checks for systematic growth rather than
demanding strict monotonic decrease at every intermediate resolution.
The growth gate triggers only when every successive level grows by more than
five percent and the final error exceeds the initial error by more than
twenty-five percent.

## Interpreting the fitted-ITPA baseline

In the current fitted-ITPA report, the strict coefficient-symmetry failures
are dominated near the outer fitted Boozer radius.  In contrast, the exact
algebraic checks, boundary/prolongation identities, and raw/full-BC
positive-positive (`T++`) subblock gates pass.  This distinction localizes the
leading discrepancy to the fitted-equilibrium coefficient representation near
the edge rather than to harmonic indexing, boundary reduction, or sparse
assembly.

## Running the fitted ITPA report

From the repository root:

```bash
conda run -n desc-petsc-np2 env \
  MPLBACKEND=Agg \
  MPLCONFIGDIR=/tmp/supernova-matplotlib \
  NUMBA_CACHE_DIR=/tmp/supernova-numba \
  NUMBA_DISABLE_JIT=1 \
  python no_sync/solver5_itpa_fitted_eqdsk/run_up_down_symmetry.py
```

The command writes JSON and CSV data, a Markdown summary, representative
angular traces, a complete parity heatmap, a radial-localization plot, an
operator-gate plot, and a convergence figure under the case-local diagnostics
directory.  The summary records the EQDSK SHA-256 checksum so the
analytic-pressure and QSOLVER-pressure fits cannot be silently confused.
