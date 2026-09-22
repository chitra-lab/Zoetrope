# Stage-1 competing models

The active comparison keeps three conceptually distinct coordinate families and two controlled Stage1a spline-expression variants for the simulation benchmark.

## Active comparison

| family / variant | implementation | purpose |
|---|---|---|
| Stage1a NN alpha + NN E | flow_stp.stage1.stage1a | flexible joint baseline |
| Stage1a NN alpha + spline E, joint | competing_models/stage1a_spline.py | isolate E representation |
| Stage1a NN alpha + spline E, profiled | competing_models/stage1a_spline.py | isolate optimization strategy |
| Fisher-score / Rayleigh | competing_models/fisher_score.py | local Fisher geometry -> coordinate |
| Fisher principal-curve registration | competing_models/curve_registration.py | expression geometry -> registered coordinate |

The two spline Stage1a variants keep the same scientific estimands as Stage1a. They are not new biological hypotheses.

The multiscale score-Rayleigh estimator remains grouped with the Fisher-score family because it changes only the local Fisher-geometry estimate.

## Removed experimental estimators

The following exploratory estimators remain retired after the frozen Qian PFC ordering experiments:

- RKHS score-Rayleigh;
- Fisher diffusion;
- geometry-anchored low-rank coordinate model;
- multiblock kernel PLS.

## Escalation rule

The simulation benchmark should start with the compact comparison above. More sophisticated expression models, derivative estimators, or dynamical solvers should only be added if the matched simulations reveal a specific algorithmic bottleneck.
