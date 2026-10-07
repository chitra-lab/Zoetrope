# Anatomical Axis Dynamics Model

This repository now contains one active scientific pipeline.

## Stage 1 — anatomical axis and expression field

flow_stp.stage1 learns tissue-specific anatomical coordinates and a shared
neural expression field E(z,t) under a Poisson observation model with the cell
library size used as an exposure offset:

y_cg ~ Poisson(L_c E_g(z_c,t_c)).

The selected implementation is stage1a.py. stage1a_optimized.py is the same
scientific model with the empirically selected higher-learning-rate
optimization preset.

## Stage 2 — derived dynamics

flow_stp.stage2 differentiates the fitted E(z,t), computes the closed-form
Fisher projection velocity

V(z,t) = - <Psi_z, Psi_t> / <Psi_z, Psi_z>, with Psi = 2 sqrt(E),

and obtains the anatomical flow F by RK4 integration of dF/dt = V(F,t).

No alternative trajectory model is trained in Stage 2.

## Package layout

- src/flow_stp/stage1/ — selected library-offset Poisson Stage 1.
- src/flow_stp/stage2/ — derivative-based closed-form Fisher dynamics + RK4.
- src/flow_stp/data/ — data loading.
- src/flow_stp/simulations/benchmark_dgp/ — retained benchmark data generator.
- src/flow_stp/metrics.py — evaluation metrics.

Competing, archived, finite-time, inverse-flow, neural-PDE, Stage-1b, zoo, and
arclength model implementations have been removed from the active tree.
