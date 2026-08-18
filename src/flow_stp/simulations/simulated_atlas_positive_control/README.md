# Simulated atlas positive control

This package generates the complete six-time-point spatiotemporal cortical-atlas
positive control used for the bare-bones Stage 2 decoder.

The default run creates:

- six tissues at evenly spaced developmental times;
- 10,000 cells per tissue, sampled uniformly in intrinsic `(s, z)` coordinates;
- progressive block-arc bending from 0 to 120 degrees;
- six neuronal stopping positions at the centers of equal layers spanning
  `z in [0.5, 1.0]`;
- a conservative density and developmental-moment PDE;
- slowly developing low-rho radial glia and neuronal cohorts converging to a
  shared mature rho;
- gene means generated as
  `B_g(z) + D_raw,g(z, rho) - D_raw,g(z, 0)` on the log scale;
- independent Poisson count observations.

## Python API

```python
from flow_stp.simulations.simulated_atlas_positive_control import (
    SimulationConfig,
    run_simulation,
    save_simulation,
)

result = run_simulation(SimulationConfig())
save_simulation(result, "simulated_atlas_positive_control")

obs = result.obs
counts = result.counts
expected_mean = result.expected_mean
rho_truth = result.decoder.rho_decoder
```

## Reusable model-comparison plots

`comparison_plotting.py` contains the reusable versions of the PI-facing plots
used in synthetic confirmatory experiments. They are intentionally model-agnostic:
future experiments supply metric tables or truth/model fields rather than copying
notebook-specific plotting code.

```python
from flow_stp.simulations.simulated_atlas_positive_control import (
    plot_bent_tissue_field_comparison,
    plot_field_comparison,
    plot_flow_trajectories,
    plot_metric_by_time_heatmap,
    plot_metric_grid,
    save_figure,
)
```

Available helpers:

- `plot_metric_grid(...)`: PI-facing metric panels; repeated fit seeds become boxplots.
- `plot_metric_by_time_heatmap(...)`: annotated model-by-time Spearman/RMSE heatmaps.
- `plot_bent_tissue_field_comparison(...)`: truth/model fields rendered over the progressively bent tissue geometry, suitable for `rho`, `z`, `q`, or other scalar fields.
- `plot_field_comparison(...)`: side-by-side time-by-`z` fields, such as recovered `V(z,t)`.
- `plot_flow_trajectories(...)`: selected truth/model `F(a,t)` trajectories.
- `save_figure(...)`: small shared helper for reproducible PNG output.

The plotting functions use `viridis` by default and return Matplotlib figure/axes
objects so notebooks can display, further annotate, or save them. They do not fit
models or impose model-specific calibration; gauge/orientation/scale calibration
should be performed explicitly by the experiment before plotting.

For example, if `rho_fields` is an ordered mapping from display label to an array
with shape `[n_timepoints, len(result.pde.z)]`:

```python
fig, axes = plot_bent_tissue_field_comparison(
    config=result.config,
    z_axis=result.pde.z,
    fields=rho_fields,
    times=result.pde.time,
    title="Ground-truth and recovered rho across bent tissue slices",
    colorbar_label="rho",
    vmin=0.0,
    vmax=1.0,
)
```

## Direct package execution

```bash
python -m flow_stp.simulations.simulated_atlas_positive_control \
  --output-dir simulated_atlas_positive_control \
  --plots
```

Useful smoke-test run:

```bash
python -m flow_stp.simulations.simulated_atlas_positive_control \
  --cells-per-tissue 100 \
  --n-z 101 \
  --output-dir smoke_atlas \
  --no-expected-mean
```

## Saved outputs

- `cell_metadata.csv.gz`
- `counts.csv.gz`
- `expected_mean.csv.gz` unless disabled
- `cohorts.csv`
- `gene_metadata.csv`
- `truth_arrays.npz`
- `config.json`
- `summary.json`
