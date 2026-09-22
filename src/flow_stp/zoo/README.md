# Stage-1 competing models

The active comparison now contains six first-pass methods. The sixth is a
discrete geometric-atlas formulation motivated by the gauge freedom of the
tissue-specific anatomical coordinates.

## Active comparison

| family / variant | implementation | purpose |
|---|---|---|
| Stage1a NN alpha + NN E | flow_stp.stage1.stage1a | flexible joint baseline |
| Stage1a NN alpha + spline E, joint | competing_models/stage1a_spline.py | isolate E representation |
| Stage1a NN alpha + spline E, profiled | competing_models/stage1a_spline.py | isolate optimization strategy |
| Fisher-score / Rayleigh | competing_models/fisher_score.py | local Fisher geometry -> coordinate |
| Fisher principal-curve registration | competing_models/curve_registration.py | expression geometry -> registered coordinate |
| Discrete geometric atlas | competing_models/discrete_geometric_atlas.py | local tissue charts + direct adjacent-time Fisher registration |

The discrete geometric-atlas method does not impose one common numerical gauge
on all tissues. Each tissue is treated as a local oriented chart of a common
one-dimensional process manifold. Adjacent-time transport is estimated directly
as a monotone map between charts by minimizing Fisher-flat expression
displacement on the learned overlap. The residual displacement defines a
discrete Fisher progression increment. Under tissue-wise monotone
reparameterizations, adjacent maps transform by conjugacy.

The two spline Stage1a variants keep the same scientific estimands as Stage1a.
They are controlled representation/optimization variants rather than separate
biological hypotheses.

## Removed experimental estimators

The following exploratory estimators remain retired after the frozen Qian PFC
ordering experiments:

- RKHS score-Rayleigh;
- Fisher diffusion;
- geometry-anchored low-rank coordinate model;
- multiblock kernel PLS.

## Escalation rule

The simulation benchmark should stay compact. Additional machinery should only
be introduced when a matched simulation diagnoses a specific failure mode.
