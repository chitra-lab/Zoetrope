# Comparative developmental benchmark DGP

This simulator is the method-independent benchmark for comparing Flow-STP with
more flexible spatial trajectory-inference methods such as STORIES and stVCR.
The simpler `flow_stp.simulations.logistic_displacement` benchmark checks
physical displacement recovery under a known logistic material flow.

## Scientific role

The benchmark tests a conditional claim:

> When developmental spatial transcriptomics is dominated by a low-dimensional
> anatomical program, explicitly exploiting that structure can improve
> trajectory inference.

The simulator does **not** use the Flow-STP neural architecture to generate
expression, and it does **not** define physical truth by applying the Flow-STP
Fisher projection to an arbitrary expression field.

## Latent biological process

Each hidden particle has a Lagrangian coordinate `a` and transverse coordinate
`u`. Its axial position follows an independently specified physical velocity
field

```
dz/dt = V_physical(z,t),   z(0)=a.
```

The population-level expression state is generated in Fisher-flat coordinates
`Psi = 2 sqrt(E)`. Spatial structure is deliberately rich: the initial
expression curve contains multiple nonlinear anatomical programs, and the
physical velocity retains nonlinear dependence on axial position. Temporal
structure is deliberately simple and identifiable from snapshots. Along each
hidden Lagrangian label, `Psi(a,t)` is affine in time, while the explicit
temporal factor of `V_physical(z,t)` is also affine. Thus the benchmark can be
challenging within each tissue without hiding oscillations or transients
between sparsely sampled developmental ages.

In the structured regime the molecular slope occupies gene-space directions
orthogonal to the spatial-expression subspace and has constant Fisher norm
across the anatomical axis. This makes the local molecular-persistence
hypothesis exact while keeping the data generator independent of the Flow-STP
neural estimator.

Observed tissues are curved ribbons embedded independently into two-dimensional
coordinate systems, including sample-specific rotation, reflection, translation
and curvature.

Cells are sampled destructively: the same hidden particle is never exposed in
two observed snapshots, although its latent identity and trajectory are retained
for evaluation.

## Independent assumption-violation knobs

Two parameters break different assumptions one at a time:

- `off_axis_strength`: adds genuine transverse (`u`-dependent) expression
  structure, violating one-dimensional spatial sufficiency.
- `parallel_strength`: adds molecular change parallel to the spatial expression
  tangent, so Fisher-identified effective flow no longer equals physical motion.

At

```
off_axis_strength = 0
parallel_strength = 0
```

the benchmark is in the exact structured regime. Increasing either parameter
creates a controlled departure from that regime.

The simulator therefore stores both

- `V_physical_true`: literal axial movement of hidden particles;
- `V_fisher_true`: effective flow identified from the evolving expression field.

They coincide in the structured regime and intentionally separate under the
parallel-change stress test.

## Observation model

The default benchmark uses gene-specific negative-binomial counts implemented
through a Gamma-Poisson mixture,

```
Var[Y_g] = mu_g + mu_g^2 / theta_g,
```

with log-normal library-size variation. A Poisson option is retained as a
well-specified sanity condition.

## Fairness boundary

Competing methods should receive only

- `observed_cell_metadata.csv.gz`: tissue/time and observed `(x,y)` coordinates;
- `counts.csv.gz`: transcript counts;
- `gene_metadata.csv`: gene identifiers only.

Files prefixed with `hidden_` and all `*_true` fields are evaluation truth and
must never be supplied to a method during fitting.

## Python API

```python
from flow_stp.simulations.benchmark_dgp import (
    BenchmarkSimulationConfig,
    run_benchmark_simulation,
    save_benchmark_simulation,
)

config = BenchmarkSimulationConfig(
    off_axis_strength=0.0,
    parallel_strength=0.0,
)
result = run_benchmark_simulation(config)
save_benchmark_simulation(result, "benchmark_structured")
```

The main paper benchmark should use one canonical configuration and vary one
assumption or sampling axis at a time rather than taking a full factorial over
all simulation parameters.

