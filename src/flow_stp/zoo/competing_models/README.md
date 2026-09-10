# Competing Stage-1 model families

After the Qian PFC model-selection sweeps, three modeling families remain worth carrying into downstream dynamics validation.

## 1. Stage1a

`stage1a.py` exposes the production implementation in `flow_stp.stage1.stage1a`. Stage1a jointly learns tissue-specific spatial coordinate maps and a shared time-dependent expression decoder, with submersion, scale-variation, coverage, and nuisance controls.

The implementation stays in the production `stage1` package because downstream Stage1b code depends on its API; this module is the competing-model namespace entry point.

## 2. Fisher-score family

`fisher_score.py` contains both surviving direct-score variants:

- single-scale score-Rayleigh, using one local neighborhood scale;
- multiscale score-Rayleigh, combining local Fisher principal-line estimates across several neighborhood scales using eigengap and sign-free cross-scale stability weights.

These are treated as one modeling family because the multiscale estimator changes only how local Fisher geometry is estimated; the downstream Rayleigh coordinate model is the same.

## 3. Fisher principal-curve registration

`curve_registration.py` implements the expression-first alternative:

1. estimate local population mean-expression rates;
2. transform to Poisson Fisher-flat coordinates `Y = 2 sqrt(M)`;
3. fit a shared expression PCA basis;
4. fit a one-dimensional principal curve within each tissue;
5. spatially regularize the local curve coordinate;
6. synchronize tissue orientations without H1 labels;
7. register the curves through monotone tissue-specific warps and a smooth common expression surface.

Its final coordinate is `z_i = h_i(u_i)`.

## Model-selection status

The retained families are intentionally conceptually distinct:

- **Stage1a:** joint spatial-coordinate / expression reconstruction;
- **Fisher-score:** local spatial Fisher geometry -> coordinate;
- **curve registration:** expression geometry -> principal curve -> registered coordinate.

The previous RKHS, Fisher-diffusion, geometry-anchored low-rank, and multiblock-kernel-PLS experimental estimators were removed after failing to improve sufficiently on the surviving families in the frozen Qian PFC H1 ordering benchmark.

The next selection gate should compare downstream expression-map quality, finite-time flow/correspondence stability, and Fisher progression using the same downstream model wherever possible.
