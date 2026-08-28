# Fisher-Lagrangian model

This package implements the parsimonious continuous-time formulation in which only two scientific objects are fitted from the data:

1. tissue-specific neural anatomical coordinate maps `z = alpha_i(x,y)`, and
2. a shared smooth mean-expression state map `E(z,t)`.

Velocity, flow, expression trajectories, and pseudotime are derived after this primary fit. There is no learned velocity, `q`, `rho`, or Stage-2 decoder.

## Observation and expression model

For cell `c` from tissue `i`,

\[
y_{icg}\sim\operatorname{Poisson}(L_{ic}E_g(z_{ic},t_i)),\qquad
z_{ic}=\alpha_i(x_{ic}).
\]

The coordinate maps are smooth neural networks. `E` is deliberately lower-complexity because its derivatives, not only its fitted values, determine the dynamics. On the log-mean scale,

\[
\eta_g(z,t)=\log E_g(z,t)
=\sum_j\beta_{gj}B_j(z)
+\sum_{jk}\theta_{gjk}B_j(z)T_k(t),
\]

where `B_j` are fixed cubic B-splines in anatomy and `T_k` are fixed low-dimensional temporal spline contrasts satisfying `T_k(t0)=0`. Consequently,

\[
\eta_g(z,t_0)=\sum_j\beta_{gj}B_j(z),
\]

so baseline expression along anatomy is isolated without an additional latent decoder. Temporal basis dimension should remain small for sparse time courses and can increase when many time points are observed.

## Anatomical gauge

Each `alpha_i` maps to `(0,1)`. Training retains two geometric constraints:

- a submersion penalty discouraging `||grad alpha_i||` from approaching zero;
- a relative-position/constant-speed penalty

\[
R_\alpha=
\frac{\nabla\alpha^T H_\alpha\nabla\alpha}{\|\nabla\alpha\|^3},
\qquad
\mathcal L_{\rm rel}=\mathbb E[R_\alpha^2].
\]

`R_alpha=0` makes the gradient magnitude constant along each gradient streamline; with the `[0,1]` gauge, equal `z` increments then represent equal fractions of physical arclength along that streamline. This prevents tissue/time-specific monotone coordinate warps from masquerading as flow.

## Block-coordinate MLE

Conditional on `z`, `eta_g(z,t)` is linear in its spline coefficients, so the expression block is a convex Poisson GLM and is fitted with damped Newton/IRLS. Conditional on the expression field, the neural coordinate maps are updated by Adam through the Poisson likelihood plus the anatomical penalties. The package therefore alternates

\[
E\mid\alpha\quad\leftrightarrow\quad\alpha\mid E
\]

and finishes with the conditional expression MLE for the final coordinates.

Coordinate networks can be initialized by pretraining them to a normalized PCA major axis of each tissue. This is only an initialization; expression likelihood and anatomical penalties refine the coordinate afterward.

## Fisher geometry and closed-form local velocity

For independent Poisson means,

\[
ds_I^2=\sum_g\frac{dE_g^2}{E_g}.
\]

The square-root coordinates

\[
\Psi_g=2\sqrt{E_g}
\]

make this metric Euclidean. A population with anatomical velocity `v` has instantaneous expression velocity

\[
\dot\Gamma=E_t+vE_z.
\]

The local expression-persistence principle chooses

\[
V(z,t)=\arg\min_v\|E_t+vE_z\|_I^2,
\]

which has the closed form

\[
V(z,t)=-\frac{\langle E_z,E_t\rangle_I}{\langle E_z,E_z\rangle_I}
=-\frac{\Psi_z^T\Psi_t}{\Psi_z^T\Psi_z}.
\]

`||Psi_z||^2` is returned as an observability diagnostic. Where it is near zero, anatomical flow is not identifiable from expression alone. A tiny `eps` is used only for numerical stabilization.

## Lagrangian flow and pseudotime

The anatomical flow is

\[
\dot F(a,t)=V(F(a,t),t),\qquad F(a,0)=a.
\]

The corresponding expected expression trajectory is

\[
\Gamma_a(t)=E(F(a,t),t).
\]

Pseudotime is the Fisher arclength of that expression trajectory,

\[
\rho_a(t)=\int_0^t\|\dot\Gamma_a(\tau)\|_I\,d\tau.
\]

Thus `F` is integrated movement along the anatomical coordinate and `rho` is integrated movement through expression-information space. Pure anatomical drift can change `F` while leaving `rho` constant if the carried expression state is unchanged.

## Files

- `spline.py`: differentiable fixed spatial and anchored temporal spline bases.
- `model.py`: neural `alpha_i`, spline `E`, Poisson likelihood, and anatomical penalties.
- `training.py`: PCA initialization, Poisson IRLS expression fitting, and block-coordinate MLE.
- `geometry.py`: Fisher embedding, derivatives, observability, and closed-form `V`.
- `trajectory.py`: RK4 integration of `F` and Fisher-arclength `rho`.
