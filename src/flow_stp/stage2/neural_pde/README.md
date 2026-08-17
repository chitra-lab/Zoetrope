# Neural PDE Stage 2 — Eulerian confirmatory model

This directory preserves the neural-PDE Stage-2 architecture used in the confirmatory experiments as a sibling of `basic` and `flow_aware`.

With the Stage-1 anatomical coordinate `z` frozen, the model learns

`rho_t + V(z,t) rho_z = q(z,t)`, with `q(z,t) >= 0`.

`rho(z,t)` is evaluated by reverse-characteristic RK4 integration. The same integration returns the material origin `a`, but expression is deliberately decoded at the **current** anatomical coordinate:

`log mu_g(z,t) = B_g(z) + D_g(z,rho(z,t)) - D_g(z,0)`.

Default architecture:

- baseline `B`: `1 -> 64 -> 64 -> genes`, GELU
- developmental decoder `D`: `2 -> 64 -> 64 -> genes`, GELU
- velocity `V`: `2 -> 96 -> 96 -> 96 -> 1`, tanh
- source `q`: `2 -> 96 -> 96 -> 96 -> 1`, SiLU + ReLU output
- differentiable characteristic integration: 16 RK4 steps
- `V` final layer starts at zero
- `q` final bias starts at `0.40`
- decoder final weights start near zero (`std=1e-3`)

Training follows the confirmatory decomposition: fit `B(z)` on the earliest (`t=0`) tissue, freeze it, then optimize `V`, `q`, and `D` with Poisson NLL. Default learning rates are `2e-3` for the baseline, `1e-3` for the dynamic fields, and `1.5e-3` for the developmental decoder.

```python
from flow_stp.stage2 import (
    NeuralPDEStage2Model,
    NeuralPDETrainingConfig,
    make_neural_pde_baseline_optimizer,
    fit_neural_pde,
)

model = NeuralPDEStage2Model(num_genes=300)
config = NeuralPDETrainingConfig()
```

`V` is intentionally unconstrained in this version. Boundary-preserving velocity parameterizations should be treated as separate model variants rather than silently changing this confirmatory architecture.
