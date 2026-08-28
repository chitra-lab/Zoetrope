# Fisher-Lagrangian model

This package implements the parsimonious Lagrangian formulation in which only two scientific objects are fitted from the data:

1. tissue-specific neural anatomical coordinate maps `z = alpha_i(x,y)`, and
2. a shared smooth mean-expression state map `E(z,t)`.

Velocity, flow, expression trajectories, and pseudotime are derived after this primary fit. There is no learned velocity, `rho`, or Stage-2 decoder.

## Observation and expression model

For cell `c` from tissue `i`,

\[
y_{icg}\sim\operatorname{Poisson}(L_{ic}E_g(z_{ic},t_i)),\qquad
z_{ic}=\alpha_i(x_{ic}).
\]

The native coordinates of different tissue slices need not be directly comparable: cuts can differ in orientation and geometry and can contain different portions of the underlying anatomical axis. Each tissue therefore receives its own flexible coordinate network, while all tissues use one fixed spatial gauge so that increments of `z` have the same local geometric meaning.

The mean-expression field is deliberately lower-complexity because its derivatives, not only its fitted values, determine the dynamics. On the log-mean scale,

\[
\eta_g(z,t)=\log E_g(z,t)
=\sum_j\beta_{gj}B_j(z)
+\sum_{jk}\theta_{gjk}B_j(z)T_k(t),
\]

where `B_j` are fixed cubic B-splines on one common spatial knot sequence and `T_k` are fixed low-dimensional temporal spline contrasts satisfying `T_k(t0)=0`. If `C_1,...,C_{K+1}` denotes the underlying raw temporal B-spline basis, the active contrasts are

\[
T_k(t)=\{C_k(t)-C_k(t_0)\}-\{C_{K+1}(t)-C_{K+1}(t_0)\},
\qquad k=1,\ldots,K.
\]

Consequently,

\[
\eta_g(z,t_0)=\sum_j\beta_{gj}B_j(z),
\]

so baseline expression along anatomy is isolated without an additional latent decoder. Temporal basis dimension should remain small for sparse time courses and can increase when many time points are observed.

## Tissue-specific coordinate maps and fixed gauge

For tissue `i`,

\[
\alpha_i(x)=S_{\Theta_i}(x)+b_i,
\]

where `S_{Theta_i}` is a tissue-specific MLP with architecture

\[
2\longrightarrow64\longrightarrow64\longrightarrow1
\]

and smooth SiLU hidden activations. Networks share an architecture but not parameters. The final MLP layer has no bias; the explicit scalar `b_i` is the tissue-specific translation that determines which window of the common anatomical coordinate is observed.

The coordinate output is linear and unbounded. There is no sigmoid and no intrinsic requirement that `z` lie in `[0,1]`. The `[0,1]` restriction belonged to an earlier Eulerian parameterization and is not needed for the Lagrangian dynamics.

A common unit-speed gauge is fixed by imposing the eikonal condition

\[
\|\nabla_x S_{\Theta_i}(x)\|_2=1
\]

for every tissue. Computationally this is enforced with

\[
\mathcal L_{\rm gauge}
=\sum_i\mathbb E_{x\in T_i}
\left(\|\nabla_x S_{\Theta_i}(x)\|_2-1\right)^2.
\]

This fixes the otherwise irrelevant common monotone spatial gauge to an arclength-like representative. It also implies the submersion condition when satisfied exactly, so the previous separate submersion penalty and Hessian relative-position penalty are removed. Tissue slices retain flexible two-dimensional geometry while no tissue can introduce its own nonlinear ruler for `z`.

Spatial input coordinates must therefore be supplied in common physical units (or under one common global rescaling). Individual tissues must not be min-max normalized independently, because doing so would change the meaning of the unit-gradient gauge.

## Spatial spline support

The theoretical anatomical coordinate takes values in `R`. The finite spline implementation of `E`, however, uses one configurable common support interval `coordinate_domain=(z_min,z_max)` for the `B_j`. This interval is a numerical support choice, not a bound on `alpha_i`. It should cover the learned coordinates from all tissues with appropriate padding. The package raises an error if fitted cell coordinates leave this support rather than silently rescaling or clamping individual tissues.

## Block-coordinate MLE

Conditional on `z`, `eta_g(z,t)` is linear in its spline coefficients, so the expression block is a convex Poisson GLM and is fitted with damped Newton/IRLS. Conditional on the expression field, the neural coordinate maps and offsets are updated by Adam through the Poisson likelihood plus the unit-speed gauge penalty. The package alternates

\[
E\mid\alpha\quad\leftrightarrow\quad\alpha\mid E
\]

and finishes with the conditional expression MLE for the final coordinates.

Coordinate networks can be initialized by pretraining them to the signed projection onto the normalized PCA major axis of each tissue. Because that projection direction has unit norm, this initialization already has the intended unit spatial scale and is not min-max normalized to `[0,1]`. Orientation signs should be chosen consistently across tissues.

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

Thus `F` is integrated movement along the anatomical coordinate and `rho` is integrated movement through expression-information space. Pure anatomical drift can change `F` while leaving `rho` constant if the carried expression state is unchanged. Because the spatial coordinate is unbounded in the Lagrangian formulation, RK4 flow integration does not clamp positions to `[0,1]`.

## Files

- `spline.py`: differentiable fixed spatial and anchored temporal spline bases.
- `model.py`: tissue-specific unbounded coordinate MLPs, explicit offsets, spline `E`, Poisson likelihood, and unit-speed gauge.
- `training.py`: unit-speed PCA initialization, Poisson IRLS expression fitting, and block-coordinate MLE.
- `geometry.py`: Fisher embedding, derivatives, observability, and closed-form `V`.
- `trajectory.py`: unclamped RK4 integration of `F` and Fisher-arclength `rho`.
