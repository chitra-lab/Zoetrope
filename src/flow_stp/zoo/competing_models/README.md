# Competing Stage-1 model families

The active comparison keeps three conceptually distinct coordinate families and adds two controlled Stage1a variants for the simulation benchmark.

## Stage1a family

stage1a.py exposes the production neural-coordinate / neural-expression model.

stage1a_spline.py keeps the same Stage1a tissue-specific neural coordinate maps, coverage model, submersion penalty, and relative-scale penalty, but swaps the expression decoder for the common tensor-product spline field used by the Fisher-Lagrangian implementation.

Two spline fitting strategies are available:

- **NN alpha + spline E (joint):** optimize coordinate maps and spline coefficients jointly with AdamW.
- **NN alpha + spline E (profiled):** solve the conditional Poisson spline problem by IRLS, update the coordinate maps with AdamW, and alternate.

The two variants estimate the same scientific objects. Their purpose is to separate expression representation from optimization strategy.

## Fisher-score family

fisher_score.py contains the retained direct-score estimators:

- single-scale score-Rayleigh;
- multiscale score-Rayleigh, which combines local Fisher principal-line estimates across neighborhood scales.

These remain one family because they share the same downstream Rayleigh coordinate construction.

## Fisher principal-curve registration

curve_registration.py is the expression-first alternative: local mean rates -> Fisher-flat coordinates -> shared PCA -> tissue principal curves -> spatial regularization -> orientation synchronization -> monotone registration.

## Simulation comparison set

The intended first-pass simulation benchmark is:

1. Stage1a NN alpha + NN E;
2. Stage1a NN alpha + spline E, joint optimization;
3. Stage1a NN alpha + spline E, profiled/block-coordinate optimization;
4. Fisher-score / Rayleigh + common downstream E;
5. Fisher principal-curve registration + common downstream E.

Oracle-z NN and spline expression fits are diagnostic controls rather than additional coordinate competitors.

More sophisticated models should only be introduced after a concrete algorithmic bottleneck is identified on simulation.

## Removed experimental estimators

The previous RKHS, Fisher-diffusion, geometry-anchored low-rank, and multiblock kernel PLS estimators remain retired after the frozen Qian PFC ordering benchmarks.
