# Retained model families

Model selection is frozen. Only two estimators remain active.

## Primary model

`primary.py` exposes the production Stage1a neural coordinate / neural expression
pipeline implemented in `flow_stp.stage1.stage1a`. It is the flexible estimator
selected for the main analysis.

## Secondary model

`secondary.py` exposes the Fisher principal-curve registration estimator implemented
in `curve_registration.py`. It is retained as a constrained geometric alternative
for robustness checks on real data.

## Removed exploratory methods

The joint/profiled spline Stage1a variants, Fisher-score/Rayleigh estimators, and
discrete geometric-atlas prototype are no longer active. Their code history remains
in git but they are intentionally excluded from the validation API.

The model-selection phase is considered complete unless a real-data validation
experiment reveals a concrete failure mode that requires revisiting the estimator.
