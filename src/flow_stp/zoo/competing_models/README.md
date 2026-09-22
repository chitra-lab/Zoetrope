# Competing Stage-1 model families

The active simulation comparison contains six methods spanning coordinate-first,
expression-first, and discrete-atlas formulations.

## Stage1a family

stage1a.py exposes the production neural-coordinate / neural-expression model.

stage1a_spline.py keeps the same Stage1a tissue-specific neural coordinate maps,
coverage model, submersion penalty, and relative-scale penalty, while swapping
the expression decoder for the common tensor-product spline field.

Two spline fitting strategies are available:

- **NN alpha + spline E (joint):** optimize coordinate maps and spline coefficients jointly with AdamW.
- **NN alpha + spline E (profiled):** solve the conditional Poisson spline problem by IRLS, update the coordinate maps with AdamW, and alternate.

## Fisher-score family

fisher_score.py contains the retained direct-score estimators:

- single-scale score-Rayleigh;
- multiscale score-Rayleigh.

## Fisher principal-curve registration

curve_registration.py is the expression-first common-coordinate alternative:
local mean rates -> Fisher-flat coordinates -> shared PCA -> tissue principal
curves -> spatial regularization -> orientation synchronization -> monotone
registration into one numerical gauge.

## Discrete geometric atlas

discrete_geometric_atlas.py keeps tissue coordinates local rather than forcing a
common numerical gauge.

For tissue i it estimates a local oriented chart u_i and a Fisher-flat
expression curve Psi_i(u_i). For adjacent times it estimates a partial monotone
map G_i by minimizing

    ||Psi_{i+1}(G_i(u)) - Psi_i(u)||^2

over the learned overlap. The residual Fisher displacement is stored as the
discrete progression increment. Adjacent maps can be composed to obtain
multi-time transport, and they transform by conjugacy under independent
monotone reparameterizations of the tissue charts.

The current prototype assumes one tissue per distinct developmental time.
Replicate-time atlas synchronization is intentionally deferred until the basic
positive-control benchmark establishes whether the formulation is useful.

## Simulation comparison set

1. Stage1a NN alpha + NN E;
2. Stage1a NN alpha + spline E, joint;
3. Stage1a NN alpha + spline E, profiled/block-coordinate;
4. Fisher-score / Rayleigh + common downstream E;
5. Fisher principal-curve registration + common downstream E;
6. Discrete geometric atlas with direct adjacent-time Fisher registration.

Oracle-z expression fits remain diagnostic controls rather than additional
coordinate competitors.

## Removed experimental estimators

The previous RKHS, Fisher-diffusion, geometry-anchored low-rank, and multiblock
kernel PLS estimators remain retired after the frozen Qian PFC ordering
benchmarks.
