# Arclength model

This directory is a **standalone implementation** of the registered transcriptomic arclength formulation. It is intentionally separate from `stage1` and `stage2`: it does not import the old learned-rho, learned-q, Stage-2 decoder, flow-aware latent accumulation, or Neural-PDE architectures.

## 1. Learned objects

For tissue `i`, learn a smooth anatomical coordinate

\[
z=\alpha_i(x,y)
\]

with a nonzero-gradient/submersion penalty on the locally gauged coordinate,

\[
\|\nabla \alpha_i(x,y)\|>0.
\]

The coordinate implementation retains the robust detached affine gauge, learned coverage endpoints, and relative normal-scale variation regularizer used in the Stage-1a geometry, but is implemented independently in this namespace.

Jointly learn one shared spatiotemporal log-mean expression field

\[
M(z,t)\in\mathbb R^G,\qquad M_g(z,t)=\log\mu_g(z,t).
\]

Counts are modeled by

\[
Y_{cg}\sim\operatorname{Poisson}(\exp(M_g(z_c,t_c)+\log L_c)).
\]

### Temporal basis for M

To keep time interpolation controlled when only a few ages are observed, `M` does **not** use a generic time MLP. It uses the fixed C2 truncated-cubic basis

\[
B(t)=\left[1,t,t^2,t^3,(t-k_1)_+^3,\ldots,(t-k_K)_+^3\right]
\]

with fixed interior knots. A neural network of `z` predicts the gene-by-basis coefficients `C_gk(z)`, so

\[
M_g(z,t)=\beta_g+\sum_k C_{gk}(z)B_k(t).
\]

The learned optimization is

\[
(\hat\alpha,\hat M)=\arg\min_{\alpha,M}
\mathcal L_{\rm Poisson}
+\lambda_{\rm sub}\mathcal L_{\rm submersion}
+\lambda_{\rm scale}\mathcal L_{\rm scale}
+\lambda_{\rm coverage}\mathcal L_{\rm coverage}.
\]

`training.py` provides a minimal tensor-first joint Adam optimizer. Dataset-specific loading remains outside this package.

## 2. Forward-mode derivatives

After fitting, freeze `alpha` and `M`. `derivatives.py` uses `torch.func.jvp` to evaluate

\[
M_z=\partial_z M,\qquad M_t=\partial_t M
\]

on a dense `(t,z)` grid. Two JVPs provide the full G-dimensional derivatives without looping over genes.

## 3. Smooth registration

The flow-aware model solves

\[
\hat V=\arg\min_V\int
\|M_t+VM_z\|_W^2
+\lambda_VV^2
+\lambda_z(\partial_zV)^2
+\lambda_t(\partial_tV)^2\,dz\,dt
\]

subject to

\[
V(0,t)=V(1,t)=0.
\]

Define

\[
A=\langle M_z,M_z\rangle_W,\qquad
B=\langle M_z,M_t\rangle_W.
\]

Then

\[
\|M_t+VM_z\|_W^2
=\|M_t\|_W^2+2BV+AV^2.
\]

After discretization and restriction to interior z nodes, stationarity gives the sparse linear system

\[
\left[\operatorname{diag}(A+\lambda_V)
+\lambda_zD_z^TD_z
+\lambda_tD_t^TD_t\right]V=-B.
\]

Thus **`-B` is simply the right-hand side created by the cross term** `2 B V`; it is not a separately learned quantity. `registration.py` solves this convex system with `scipy.sparse.linalg.spsolve` or conjugate gradient and inserts exact zero-velocity boundary values.

The registered residual expression velocity and its magnitude are

\[
r=M_t+VM_z,\qquad q=\|r\|_W.
\]

The no-flow baseline is computed from the same fitted `M`:

\[
q_0=\|M_t\|_W.
\]

## 4. Flow and arclength

The registration velocity induces the flow map

\[
F(a,t)=Z(a,t),\qquad
\dot Z=V(Z,t),\quad Z(a,0)=a.
\]

`trajectory.py` integrates the coupled system

\[
\dot Z=V(Z,t),\qquad
\dot\rho=q(Z,t),\quad \rho(a,0)=0
\]

with vectorized RK4. Therefore

\[
\rho(a,t)=\int_0^t q(F(a,\tau),\tau)\,d\tau.
\]

An observed `(z,t)` is assigned a reference coordinate `a` by inverting the monotone one-dimensional map `F_t(a)=z`, then receives the corresponding `rho(a,t)`.

## 5. Interpretation

> **Developmental progression is cumulative transcriptomic change through time after accounting for expression change explainable by smooth repositioning along the learned anatomical axis.**

The core pipeline is

```text
(x,y,t,Y)
    -> jointly fit alpha_i and M
    -> JVPs M_z, M_t
    -> sparse smooth-registration solve for V
    -> flow F(a,t)
    -> residual rate q
    -> registered arclength rho
```

Always retain the no-flow result `rho_0` as a nested baseline so the value of registration can be tested rather than assumed.
