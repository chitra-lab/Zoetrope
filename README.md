# Anatomical Axis Dynamics Model

This repository contains one active scientific pipeline.

## Stage 1 — anatomical axis and Fisher-flat expression field

`flow_stp.stage1` learns tissue-specific anatomical coordinates together with
a shared neural expression field parameterized directly in independent-Poisson
Fisher-flat coordinates:

Psi(z,t) = 2 sqrt(E(z,t)).

The observation model remains Poisson with the cell library size as an exposure:

y_cg ~ Poisson(L_c E_g(z_c,t_c))

with

E_g(z,t) = Psi_g(z,t)^2 / 4.

Thus the fitted neural decoder outputs Psi directly. It does **not** first fit
log E and then transform it into Fisher coordinates after estimation.

The direct-Psi baseline is initialized on the positive branch from the
library-normalized training gene means, with the neural residual head initialized
near zero. `stage1a_optimized.py` is the same scientific model with the
empirically selected 2e-3 learning-rate preset.

## Stage 2 — derived dynamics

`flow_stp.stage2` differentiates the directly fitted Fisher-flat field and
computes

V(z,t) = - <Psi_z, Psi_t> / <Psi_z, Psi_z>.

It then obtains the anatomical flow F by classical RK4 integration of

dF/dt = V(F,t).

There is no separately fitted Stage-2 trajectory model.

Typical use is:

    field = stage1_result.model.fisher_flat
    local = estimate_local_dynamics(field, z, t)
    trajectory = integrate_flow_rk4(field, anchors, t0=0.0, t1=1.0)

## Package layout

- `src/flow_stp/stage1/` — library-offset Poisson estimation of z and direct Psi.
- `src/flow_stp/stage2/` — direct-Psi derivatives, closed-form V, and RK4 F.
- `src/flow_stp/data/` — data loading.
- `src/flow_stp/simulations/benchmark_dgp/` — retained benchmark generator.
- `src/flow_stp/metrics.py` — evaluation metrics.

Competing, archived, finite-time, inverse-flow, neural-PDE, Stage-1b, zoo, and
arclength model implementations remain removed from the active tree.
