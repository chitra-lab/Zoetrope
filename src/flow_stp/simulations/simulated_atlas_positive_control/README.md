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
