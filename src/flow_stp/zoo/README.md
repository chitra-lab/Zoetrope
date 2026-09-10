# Stage-1 competing models

The active Stage-1 comparison has been narrowed to three modeling families after the frozen Qian PFC ordering benchmarks.

The implementations and family-level documentation now live in [`competing_models/`](./competing_models/).

## Surviving families

| family | implementation | idea |
|---|---|---|
| Stage1a | `flow_stp.stage1.stage1a`, exposed through `competing_models/stage1a.py` | jointly learn tissue-specific spatial coordinates and a shared time-dependent expression decoder |
| Fisher-score / Rayleigh | `competing_models/fisher_score.py` | estimate local Fisher geometry directly from spatial expression scores and fit a coordinate aligned with its leading direction |
| Fisher principal-curve registration | `competing_models/curve_registration.py` | learn a one-dimensional expression curve in Fisher-flat coordinates, then spatially regularize and register it across tissues |

The multiscale score-Rayleigh estimator is grouped with the Fisher-score family because it changes only the local Fisher-geometry estimator: it combines principal-line estimates across neighborhood scales using eigengap and sign-free cross-scale stability. The downstream Rayleigh coordinate construction is the same.

The production Stage1a implementation remains in `flow_stp.stage1.stage1a` for downstream compatibility. The competing-model namespace exposes it alongside the other two families.

## Removed experimental estimators

The following exploratory estimators were removed after failing to improve sufficiently on the retained families in the frozen Qian PFC H1 ordering experiments:

- RKHS score-Rayleigh;
- Fisher diffusion;
- geometry-anchored low-rank coordinate model;
- multiblock kernel PLS.

Their benchmark outputs remain experiment artifacts; they are no longer active estimator implementations in the package.

## Shared infrastructure

`common.py`, `fisher.py`, `trainers.py`, `objectives.py`, and the coordinate-map helpers remain shared infrastructure used by the surviving families.

The next model-selection gate should compare the retained coordinates through the same downstream expression-map and dynamics pipeline rather than continue expanding the estimator zoo.
