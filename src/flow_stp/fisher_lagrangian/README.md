# Fisher-Lagrangian model

This package implements a Lagrangian trajectory-inference model in which the only primary learned objects are

1. tissue-specific anatomical coordinate maps `z = alpha_i(x,y)`, and
2. a shared smooth mean-expression field `lambda(z,t)`.

The flow, residual developmental speed, and developmental progress are then derived from the geometry of the fitted expression field rather than represented by independent neural networks.

## Observation model

For cell `c` from tissue `i` observed at time `t_i`,

\[
y_{icg} \sim \operatorname{Poisson}(L_{ic}\,\lambda_g(z_{ic},t_i)),
\qquad
z_{ic}=\alpha_i(x_{ic}),
\]

where `L_ic` is a known exposure/library-size factor. The model learns `log lambda(z,t)` jointly with the coordinate maps.

## Fisher geometry

For independent Poisson means, the Fisher line element is

\[
ds_F^2 = \sum_g \frac{d\lambda_g^2}{\lambda_g}.
\]

Using the square-root embedding

\[
Y_g(z,t)=2\sqrt{\lambda_g(z,t)},
\]

this becomes ordinary Euclidean geometry:

\[
ds_F^2=\sum_g dY_g^2.
\]

## Derived expression-registration flow

Temporal expression change is decomposed into a component explained by movement along the anatomical axis and an orthogonal residual:

\[
Y_t = -VY_z + R.
\]

The pointwise least-squares velocity is

\[
V(z,t)=-\frac{Y_z^T Y_t}{Y_z^T Y_z+\varepsilon}.
\]

Thus

\[
R(z,t)=Y_t(z,t)+V(z,t)Y_z(z,t).
\]

The regularized denominator is intentionally explicit: where the expression field carries almost no anatomical gradient, the flow is weakly identified from expression alone.

## Lagrangian trajectories

For a material label `a` at the reference time,

\[
F(a,0)=a,
\qquad
\dot F(a,t)=V(F(a,t),t).
\]

The intrinsic developmental speed is

\[
q(z,t)=\|R(z,t)\|_2,
\]

and cumulative developmental progress along the material trajectory is

\[
\rho_a(t)=\int_0^t q(F(a,\tau),\tau)\,d\tau.
\]

So the interpretation is

- `z`: relative anatomical position,
- `F(a,t)`: inferred anatomical location of the material population starting at `a`,
- `q`: Fisher-information rate of expression change not explained by anatomical repositioning,
- `rho_a(t)`: accumulated intrinsic transcriptional development along that trajectory.

The transport PDE

\[
\rho_t+V\rho_z=q
\]

is not fitted directly; it is the Eulerian identity induced by the Lagrangian construction.

## Implementation

- `model.py`: joint coordinate-map + smooth log-mean expression model with Poisson likelihood.
- `geometry.py`: derivatives, Fisher square-root embedding, derived velocity, residual, and speed.
- `trajectory.py`: RK4 integration of the derived flow and accumulated Fisher arclength.
- `__init__.py`: public API.

This package intentionally does not import or reuse the old learned `rho`, `q`, Stage-2 decoder, or neural-PDE model.