# Workflows and evaluation

Use `metrics_and_tissue_displacement.ipynb` from a repository checkout after
`pip install -e ".[notebooks]"`. The root Qian workflow additionally requires
the `[qian]` extra and access to the h5ad files.

The metrics notebook defaults to a reduced execution check. Set `RUN_FULL=True`
to generate the original logistic material-flow benchmark:

| Setting | Full benchmark |
|---|---|
| Observed tissues | 50 times uniformly spaced in [0,1] |
| Cells | 100,000 independently sampled cells per tissue |
| Genes | 8 |
| DGP seed | 20261006 |
| Physical flow | r(a,t) = a exp(0.8t) / (1 − a + a exp(0.8t)) |
| XY geometry | x=r, y=0.15u; a,u independently uniform |
| Expression | Psi=10+3(a−0.5)h1+cos(pi t/2)h2+sin(pi t/2)h3 |
| Gene basis | Normalized Hadamard columns 1,2,3 (zero-based) |
| Counts | Independent Poisson(Psi²/4) |
| Selected Stage 1 | Shared 64×64 coordinate network, global gauge; log-E 128×128 |
| Training | 1,500 epochs; 1,024 cells/tissue/update; seed 20260622 |
| Stage 2 | Differentiate the fitted Stage-1 field; no decoder refit |
| Source | T25; 2,500 cells; evaluation seed 20261010+707 |
| Targets | T0,5,10,15,20,24,26,30,35,40,45,49 |
| RK4 | Direct autodiff field; maximum time step 0.0025 |
| Physical lift | K=316 nearest target z values, then nearest source y |
| Primary metric | Pooled displacement NRMSE |
| Controls | No motion (NRMSE=1), oracle physical contour lift |
| Motion strata | Target true displacement RMS <0.01, 0.01–0.03, ≥0.03 |

Physical source selection reproduces the original exclusions: the fixed
10,000-cell sample and 5,000-cell gauge-calibration sample at each tissue are
excluded, followed by the 0.5%–99.5% observed-x restriction. These legacy sets
serve only to define the evaluation source; the selected Stage-1 runner uses
its own fixed splits, and there is no separate Stage-2 fit. Material labels
are never passed to Stage 1. Truth is not used to orient or calibrate z.

The earlier tournament independently refit a direct-Psi decoder and evaluated
a gridded velocity interpolator. This notebook intentionally uses the selected
model's joint fitted field and direct autodiff RK4. Historical tournament
improvement percentages are not results for this version.

Outputs under `experiments/` include the fitted model, training diagnostics,
per-time coordinate/truth correlations, per-target displacement metrics,
pooled metrics, an oracle-lift resolution control, source indices and endpoint
arrays. The generation seed/configuration and evaluation protocol are recorded
in `benchmark_manifest.json`. No generated data or trained weights are committed.

The separate `benchmark_dgp` package remains the richer method-comparison
generator with assumption-violation controls. RPCE remains an optional
geometric diagnostic for synthetic or real tissues. Its numerical path trace
and common observed z interval should be inspected before interpreting scores.
