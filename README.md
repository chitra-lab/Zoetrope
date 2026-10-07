# Anatomical Axis Dynamics Model

This repository contains one active scientific pipeline.

## Stage 1 — anatomical axis and Fisher-flat expression field

`flow_stp.stage1` learns anatomical coordinates together with
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

Stage 1 has exactly three coordinate-decoder choices; all other scientific
components and defaults are held fixed:

1. **Independent / per-tissue gauge** (existing baseline):
   `z_i = alpha_i(x,y)`, with a robust affine gauge refreshed separately in
   each tissue.
2. **Shared-time / per-tissue gauge**:
   `z = alpha(x,y,t)`, but with the same per-tissue robust affine gauge as the
   baseline.
3. **Shared-time / global gauge**:
   `z = alpha(x,y,t)`, with one robust affine ruler computed from pooled fixed
   reference points across all tissues.

The shared variants are exposed as
`run_stage1a_shared_time_per_tissue` and
`run_stage1a_shared_time_global`. This keeps real-data testing focused on the
coordinate-decoder/gauge structure while leaving the library-offset Poisson
likelihood, direct-Psi expression decoder, optimizer preset, outputs, and Stage 2
interface unchanged.

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
