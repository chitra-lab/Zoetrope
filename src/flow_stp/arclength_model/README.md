# Arclength model

This directory is a **standalone prototype namespace** for the registered transcriptomic arclength formulation. It is intentionally kept separate from the existing `stage1` and `stage2` implementations so that experiments cannot accidentally mix the old staged latent-variable models with the new formulation.

## Core idea

For tissue `i`, learn a smooth anatomical coordinate

\[
z = h_i(x,y)
\]

with the submersion requirement

\[
\|\nabla h_i(x,y)\| > 0
\]

throughout the tissue interior (implemented with a nonzero-gradient penalty plus the existing coordinate gauge/geometry controls).

Jointly learn a shared spatiotemporal log-mean expression field

\[
M(z,t) \in \mathbb{R}^G,
\qquad
M_g(z,t)=\log \mu_g(z,t),
\]

with raw counts modeled, initially, by a Poisson likelihood.

The primary learned optimization is therefore

\[
(\hat h,\hat M)
=\arg\min_{h,M}
\mathcal L_{\mathrm{counts}}
+\lambda_{\mathrm{sub}}\mathcal L_{\mathrm{submersion}}
+\lambda_{\mathrm{geom}}\mathcal L_{\mathrm{geometry}}.
\]

There is **no separately learned Stage-2 latent `rho` decoder** in this model.

## Derived quantities

After fitting `h` and `M`, compute

\[
M_z=\partial_z M,
\qquad
M_t=\partial_t M.
\]

### No-registration arclength

The unregistered instantaneous transcriptomic-change rate is

\[
q_0(z,t)=\|M_t(z,t)\|_W,
\]

and cumulative arclength is

\[
\rho_0(z,t)=\int_0^t q_0(z,\tau)\,d\tau.
\]

### Smooth registration

Optionally solve for a scalar velocity field `V(z,t)` that explains the component of temporal expression change attributable to repositioning along the anatomical axis:

\[
\hat V
=\arg\min_V
\int
\|M_t+V M_z\|_W^2
+\lambda_V V^2
+\lambda_z (\partial_zV)^2
+\lambda_t (\partial_tV)^2
\,dz\,dt.
\]

The residual developmental expression-change vector is

\[
r(z,t)=M_t(z,t)+V(z,t)M_z(z,t),
\]

with rate

\[
q(z,t)=\|r(z,t)\|_W.
\]

The induced flow map is derived from

\[
\dot Z(a,t)=V(Z(a,t),t),
\qquad Z(a,0)=a,
\]

and the registered transcriptomic arclength is

\[
\rho(a,t)=\int_0^t q(Z(a,\tau),\tau)\,d\tau.
\]

Thus the core interpretation is:

> **Developmental progression is cumulative transcriptomic change through time after accounting for expression change explainable by repositioning along the learned anatomical axis.**

## Intended implementation order

1. `coordinate_expression.py` — jointly fit tissue coordinate maps `h_i(x,y)` and shared expression field `M(z,t)`.
2. `derivatives.py` — evaluate `M`, `M_z`, and `M_t` on a dense `(z,t)` grid.
3. `registration.py` — solve the smooth quadratic registration problem for `V`.
4. `trajectory.py` — integrate the induced flow and cumulative arclength `rho`.
5. `diagnostics.py` — geometry, registration, arclength, and held-out biological diagnostics.

Prototype 1 should always compute the **no-registration** result first. Registration is an optional extension whose value must be demonstrated empirically.

## Separation from the staged models

This package should not silently depend on `flow_stp.stage1` or `flow_stp.stage2`. Shared low-level utilities may be reused only when the dependency is explicit and does not import staged-model assumptions. In particular, this package should not reuse the old learned `rho`, `q`, decoder, flow-aware latent accumulation, or Neural-PDE architectures.
