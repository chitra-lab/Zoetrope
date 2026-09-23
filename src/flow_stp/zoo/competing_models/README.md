# Retained model families

Model selection is frozen. Two estimator families remain active.

## Primary M1

`primary.py` exposes the production Stage1a neural coordinate / neural
expression pipeline implemented in `flow_stp.stage1.stage1a`.

The selected primary configuration is the stripped estimator:

```text
tissue-specific neural coordinate maps
+ detached robust affine computational gauge
+ shared neural expression field E(z,t)
+ Poisson likelihood
```

The former fuller M1 remains available as
`PRIMARY_BELLS_AND_WHISTLES_CONFIG` for a controlled real-data ablation. It
restores learned coverage, coordinate geometry penalties, rank-2 slice nuisance,
and the total-count library-size offset. It is an ablation of M1, not an
additional model-selection candidate.

## Secondary model

`secondary.py` exposes the Fisher principal-curve registration estimator
implemented in `curve_registration.py`. It is retained as a constrained
geometric alternative for robustness checks on real data.

## Removed exploratory methods

The joint/profiled spline Stage1a variants, Fisher-score/Rayleigh estimators,
and discrete geometric-atlas prototype are no longer active. Their code history
remains in git but they are intentionally excluded from the validation API.

The model-selection phase is considered complete unless a real-data validation
experiment reveals a concrete failure mode that requires revisiting the
estimator.
