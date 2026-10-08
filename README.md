# Anatomical Axis Dynamics Model (Flow-STP)

## Active Stage 1: shared anatomical coordinate + log-expression likelihood

The **only selected Stage-1 runner** is `flow_stp.stage1.run_stage1a`.
It learns one time-dependent anatomical coordinate and a shared log-expression
field using the library-exposure Poisson observation model:

```text
eta_g(z,t) = beta_g + neural_decoder_g(z,t)      # learned log E
Y_cg ~ Poisson(L_c exp(eta_g(z_c,t_c)))          # observed raw counts
Psi_g(z,t) = 2 exp(eta_g(z,t)/2)                 # derived Fisher-flat field
```

The model automatically chooses the coordinate input based on **physical tissue
counts per unique observed developmental time**, not biological sample IDs:

- **Exactly one tissue at every time:** `z = alpha(x,y,t)`. No tissue ID.
- **Two or more tissues at any observed time:** `z = alpha(x,y,t,onehot(tissue_id))`.
  Tissue ID distinguishes physical slices, including replicates from the same
  biological specimen.

The choice is automatic for a run; if even one time is replicated, all slices
use the tissue-conditioned shared network. In both cases the coordinate model
uses one **pooled robust global affine gauge** (0.5%–99.5% fixed spatial
reference quantiles). Coordinates are standardized separately in each tissue
before the neural coordinate map. Counts are not log-transformed or
standardized; the model uses a clipped log library-size exposure offset.

**Frozen settings:** coordinate MLP (64,64) tanh; log-E decoder (128,128) GELU;
AdamW learning rate 2e-3; weight decay 1e-5; 1500 epochs; 1024 observations
per tissue per update; seed 20260622; no coordinate or nuisance penalties. Initial log-expression intercept is zero, matching the
frozen synthetic configuration. Historical methods were moved to
[`stage1/archive`](src/flow_stp/stage1/archive/README.md).

### Example

```python
from flow_stp.stage1 import Stage1ARunConfig, run_stage1a
from flow_stp.stage2 import estimate_local_dynamics, integrate_flow_rk4

result = run_stage1a(
    counts_by_slice=counts_by_slice,
    spatial_by_slice=spatial_by_slice,
    panel_df=panel_df,             # one row per physical tissue, with time
    gene_names=gene_names,
    obs_df=obs_df,
    config=Stage1ARunConfig(output_dir="experiments/selected_stage1"),
)
print(result.model.config.coordinate_decoder)  # shared_time or shared_time_tissue
field = result.model.fisher_flat
local = estimate_local_dynamics(field, z, t)
trajectory = integrate_flow_rk4(field, anchors, t0=0.0, t1=1.0)
```

Use a **new** output directory: archived direct-Psi or historical log-E
checkpoints are not interchangeable with this new model version. The runner
writes a method-versioned manifest, checkpoints, coordinate tables, and
training/validation diagnostics.

## Stage 2: continuous Fisher dynamics

Differentiate the derived Fisher-flat field `Psi(z,t)=2 exp(eta/2)` to obtain

```text
V(z,t) = - <Psi_z,Psi_t> / ||Psi_z||²
q(z,t) = Psi_t + V(z,t) Psi_z
```

Then integrate `dF/dt = V(F,t)` by RK4. There is no independent Stage-2
trajectory training. Physical velocity requires a physical-coordinate
interpretation/inversion of the learned anatomical map; `V` by itself is
velocity in learned `z` units.

## Layout

- `src/flow_stp/stage1/stage1a.py`: active selected log-E shared model
- `src/flow_stp/stage1/archive/`: historical direct-Psi and related variants
- `src/flow_stp/stage2/`: differentiable Fisher projection and RK4 dynamics
- `src/flow_stp/data/`: data loading
- `src/flow_stp/simulations/benchmark_dgp/`: simulation tools
- `tests/`: tests for active code

Legacy neural-PDE, inverse-flow and other previously deleted implementations
remain accessible via Git history rather than being reintroduced into the active
package.


## Installation and notebooks

Install the core package with `pip install -e .`; add `[qian,notebooks]` for
Qian h5ad loading and notebooks, or `[test]` for regression tests.

- `flow_stp_full_stage1_qian_pfc_workflow.ipynb`: current Qian loading, selected
  Stage 1, and derived Stage 2 workflow.
- `notebooks/metrics_and_tissue_displacement.ipynb`: coordinate diagnostics,
  optional RPCE, and the logistic material-flow tissue displacement benchmark.
- `src/flow_stp/simulations/logistic_displacement.py`: reusable generator,
  contour lift, and displacement evaluator. The notebook defaults to a smoke
  check; enable `RUN_FULL` for the original 50 × 100,000-cell scale.

The archive remains unchanged and accessible. The active API no longer accepts
obsolete penalty, nuisance, coverage-endpoint or neural-flow settings.
Existing checkpoints from the selected method are migrated when loading through
`run_stage1a`: zero coverage endpoints and their optimizer group are removed.
Nonzero endpoints and alternative models are rejected. New checkpoints use the
clean state schema; older source versions cannot read them. Network initialization,
log-E likelihood and pooled gauge are preserved. Removing clamped endpoint
parameters also removes their contribution to gradient clipping, so training is
not promised to be bit-for-bit identical to pre-cleanup runs.
