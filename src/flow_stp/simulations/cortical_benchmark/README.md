# Cortical crescent synthetic benchmark — frozen v1.0 (2026-10-08)

This **new module** does not replace `benchmark_dgp`, which is retained unchanged. It generates synthetic cortical tissue and transcript counts without calling the Flow-STP estimator. Do not change its primary settings after competitive runs without incrementing the simulator version and reporting the change.

**Geometry (frozen):** ten equally spaced developmental tissues, longitudinal intrinsic coordinate `u`, radial anatomical coordinate `z`, length 1.25, initial thickness 0.17, maximum curvature 3.35, edge thickness growth 0.82, central addition 0.70, center growth width 0.26. Rigid sequencer rotations for zero-indexed tissues 2, 5, 8 are -38°, +57°, -65°; translations independently seeded. `(x,y)=R_i G_{t_i}(u,z)+b_i`.

**Migration (frozen):** `dz/dt = 1.8 z(1-z)[1+0.45(2z-1)][1-0.9(3t²-2t³)]`, `du/dt=0`; physical sequencer-frame migration velocity is `R_i @ partial_z(G_t) * dz/dt`, excluding the geometric deformation term `partial_t G_t`. Do not mistake within-section migration vectors for inter-section absolute displacements without registering those sections.

**Expression:** 48 genes (8 families of 5 signal genes and 8 housekeeping controls), 8 orthogonal gene-space directions, nonlinear anatomical programs plus temporally changing paired molecular programs. Expected biological rate `lambda=Psi²/4`; `Y_cg ~ Poisson(lambda_cg * L_c)`. By default `L=1`, optional lognormal exposure variation. Hidden material position `a = F_t^{-1}(z)`.

**Regimes:** A pure transport; **B default positive-control orthogonal differentiation**; C tunable nonorthogonality `parallel_strength`; D tunable transverse expression `offaxis_strength`. In B the exact Fisher projection recovers the prescribed intrinsic migration under exact derivatives and coordinate knowledge. C and D challenge distinct assumptions; evaluation must distinguish physical velocity from Fisher-projected velocity.

**Sampling:** default uniform **physical** area density using the exact embedding Jacobian and initial 1,800 cells; later counts satisfy `N_t=round(N_0 area_t/area_0)`. `--population constant` keeps N fixed. `--sampling uniform_intrinsic` is a labeled alternative. Area-proportional sampling is *not* a lineage-level stochastic birth process; `growth_rate_continuity_true` is the net source term required by the implied density and given migration/geometry.

Run from an installed Flow-STP environment:

```bash
python -m flow_stp.simulations.cortical_benchmark --output-dir experiments/cortical_v1_B
python -m flow_stp.simulations.cortical_benchmark --output-dir experiments/cortical_v1_A --regime A
python -m flow_stp.simulations.cortical_benchmark --output-dir experiments/cortical_v1_C025 --regime C --parallel-strength 0.25
python -m flow_stp.simulations.cortical_benchmark --output-dir experiments/cortical_v1_D025 --regime D --offaxis-strength 0.25
```

Pass **only** `public/` (observed `x,y`, tissue/time, raw counts, gene names) to Flow-STP, STORIES and stVCR. Do not expose `hidden/` during fitting or hyperparameter selection. The output manifest records configuration and cell counts. The public digest helps detect inadvertent input changes.

**Benchmark protocol:** First validate A/B under the generating assumptions, then sweep C/D; use matched random draws and matching tissue coordinates across expression regimes. Include simple quantile-matching and independent-coordinate controls. Physical velocity claims are appropriate only for methods that actually predict a physical velocity. STORIES should be scored primarily on gene-state trajectories and developmental ordering rather than assigned a fabricated spatial-velocity output.

**Scope:** This is a frozen generator, not yet a finished competitor tournament; stVCR and STORIES adapters, their validated configuration, and full-scale runtime comparisons remain separate tasks.
