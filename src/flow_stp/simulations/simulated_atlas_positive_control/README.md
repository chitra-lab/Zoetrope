# Simulated atlas positive control

This package generates the current Fisher-Lagrangian positive control for the
spatiotemporal trajectory-inference model.

The scientific data-generating process now mirrors the current framework:

1. sample cells in intrinsic tissue coordinates `(s,z)`;
2. bend each tissue into a different two-dimensional geometry while retaining
   an exact ground-truth common coordinate map `alpha_i(x,y)=z`;
3. generate a smooth expression state map `E(z,t)` through Fisher-flat
   coordinates `Psi=2*sqrt(E)`;
4. derive `V`, `F`, and trajectory progression `rho` from `E` by the same
   Fisher-Lagrangian definitions used by the method;
5. sample independent Poisson transcript counts with mean `E(z,t)`.

There is no separately generated developmental `rho`, `q`, or Stage-2
expression decoder in the active positive-control path.

## Analytic truth

The default truth contains transported spatial genes

```
Psi_g(z,t) = b_g + s_g (z-Ut) + A_g phi_g(z-Ut)
```

and developmental genes

```
Psi_g(z,t) = b_g + q_g t.
```

For the transported genes,

```
partial_t Psi_g = -U partial_z Psi_g,
```

while developmental genes have zero spatial derivative. Therefore the current
Fisher-Lagrangian identification rule gives exactly

```
V(z,t) = U,
F(a,t) = a + U t.
```

The only residual Fisher motion along a carried trajectory comes from the
developmental genes, so

```
rho_a(t) = t * sqrt(sum_g q_g^2).
```

The simulator checks these identities internally. The default trajectory truth
is evaluated only for initial positions whose full paths remain inside the
observed `z` interval.

## Geometry

Tissues begin as intrinsic rectangles and are progressively bent into annular
block arcs. The simulator saves the observed two-dimensional coordinates and
the true intrinsic coordinate `z`. The function `true_common_coordinate`
analytically inverts the bending transformation and therefore provides exact
ground truth for evaluating learned coordinate maps.

## Python API

```python
from flow_stp.simulations.simulated_atlas_positive_control import (
    SimulationConfig,
    run_simulation,
)

result = run_simulation(SimulationConfig())

obs = result.obs
counts = result.counts
expected_mean = result.expected_mean
truth = result.truth
```

Important truth arrays include

- `truth.psi`
- `truth.mean`
- `truth.psi_z`
- `truth.psi_t`
- `truth.velocity`
- `truth.observability`
- `truth.flow`
- `truth.rho`

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
- `gene_metadata.csv`
- `truth_arrays.npz`
- `config.json`
- `summary.json`

`truth_arrays.npz` contains

- `z_grid`, `time_grid`, and `initial_position_grid`
- `psi_true`, `E_true`, `psi_z_true`, and `psi_t_true`
- `V_true` and `observability_true`
- `F_true` and `rho_true`

## Reusable comparison plots

`comparison_plotting.py` still contains model-agnostic helpers for truth/model
field comparisons, trajectory plots, heatmaps, and metric panels. These can be
used for the upcoming oracle, expression-estimator, and end-to-end coordinate
benchmarks.

The older PDE, cohort, and centered-rho decoder modules remain in the repository
for historical compatibility, but they are no longer called by
`run_simulation`.
