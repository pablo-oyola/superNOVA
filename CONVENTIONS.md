# superNOVA conventions

## Equilibrium and coordinates

EQDSK equilibria are converted once by pyCOCOS before entering superNOVA.
Both `superEquilibrium` and `DESCSolver.from_eqdsk` require explicit metadata
proving `cocos_internal=1`, `phiclockwise_internal=False`, and
`flux_normalization_internal="Wb/rad"`; noncanonical or metadata-free inputs
are rejected. Internally,

$$
\psi=\Phi_{\mathrm{pol}}/(2\pi),\qquad
\mathbf B=\nabla\phi\times\nabla\psi+F\nabla\phi,
$$

so that

$$
B_R=R^{-1}\partial_Z\psi,\quad
B_Z=-R^{-1}\partial_R\psi,\quad
B_\phi=F/R.
$$

The internal toroidal angle is counterclockwise when viewed from $+Z$, hence
`phiclockwise_internal=False`. Input COCOS, toroidal orientation, and flux
normalization remain loader metadata and are not reused after conversion.
For spline safety, stored `psi0` is numerically increasing; `psi_axis` and
`psi_boundary` retain the physical axis-to-edge direction, and solver profile
coordinates always increase outward even when physical $\psi$ decreases.

The Boozer gauge is

$$
\zeta=\phi+\nu(\psi,\Theta),\qquad
\partial\zeta/\partial\phi=1,
$$

with signed physical Jacobian

$$
J=\sqrt g=
[\nabla\psi\cdot(\nabla\Theta\times\nabla\zeta)]^{-1}.
$$

Solver-facing coordinates use positive $q$ and satisfy

$$
J\mathbf B\cdot\nabla=\partial_\Theta+q\partial_\zeta,
\quad g=F=B_\zeta,\quad I=B_\Theta,\quad h=JB^2=I+qg.
$$

`I` is the sole solver-facing name for the unscaled covariant coefficient
$B_\Theta$.

If an input representation has negative $q$, it is changed by the coordinate
reflection $\Theta'=-\Theta$. This maps $(q,I,h,J)$ to their negatives,
leaves $g$ unchanged, reindexes surface fields, and negates metric components
with exactly one $\Theta$ index. It does not alter the physical equilibrium.

Gradient dot products are the contravariant metric $g^{ij}$; coordinate-tangent
dot products are the covariant metric $g_{ij}$. Cylindrical angular components
use $(1/R)\partial_\phi$ for gradients and $R\partial_i\phi$ for tangents.

## Harmonics and operators

The perturbation basis is

$$
f(\psi,\Theta,\zeta)=\sum_{m,n}f_{mn}(\psi)
e^{i(n\zeta-m\Theta)},
$$

therefore $\partial_\Theta\to-im$ and $\partial_\zeta\to+in$. The
field-line and diamagnetic diagonals are

$$
G=i(nq-m),\qquad
\Sigma=-i(gm+In),
$$

where $\mathbf s=\mathbf B\times\nabla\psi$ and
$J\mathbf s\cdot\nabla=g\partial_\Theta-I\partial_\zeta$.

For a real equilibrium field written as

$$
a_{mn}\cos(m\Theta-n\zeta)+b_{mn}\sin(m\Theta-n\zeta),
$$

`Profile.data` stores the one-sided amplitude $A_{mn}=a_{mn}+ib_{mn}$.
Reconstruction is `Re[A_mn exp(i(n*zeta-m*Theta))]`. Literal two-sided
convolution coefficients are $A_{mn}/2$ for nonzero modes, with
$\hat f_{-m,-n}=\hat f_{mn}^*$; the zero mode is not halved.
