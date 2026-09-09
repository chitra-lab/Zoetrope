# Stage-1 estimator zoo

This folder contains alternative estimators of the **same frozen geometric
object**: tissue-specific process maps into a common one-dimensional process
manifold. The production `flow_stp.stage1.stage1a` implementation remains
outside the zoo.

The first Qian PFC benchmark strongly favored Stage1a and the direct
score-Rayleigh family. The older joint-likelihood, neural-mean-field,
independent-registration, eigenfield-integration, and graph-arclength entry
points have therefore been removed from the active zoo. The replacements
below target the failure modes exposed by that benchmark rather than adding
more generic latent-variable machinery.

## Current methods

| module | estimator | question it tests |
|---|---|---|
| `score_rayleigh_warp` | local spatial log-rate scores -> Fisher tensor -> Rayleigh coordinate + tissue warps | surviving direct-score reference |
| `multiscale_score_rayleigh` | score Fisher tensors over several neighborhood scales, combined by eigengap and cross-scale line stability | can local scale selection make score geometry more robust? |
| `rkhs_score_rayleigh` | direct-score Fisher geometry + Gaussian-RBF/RKHS scalar coordinate + shared expression reconstruction | does the coordinate need an MLP? |
| `fisher_diffusion` | Fisher-anisotropic diffusion map using the local principal line | can a global spectral scalar recover the process without vector-field integration? |
| `geometry_anchored_lowrank` | score-Rayleigh geometry anchor + low-rank Fisher-flat expression surface | can expression information refine `z` without overpowering the geometry? |
| `multiblock_kernel_pls` | shared Fisher-flat response loading + tissue-specific RBF spatial blocks | how much of the axis is recoverable by classical spatial-expression covariance? |

`flow_stp.stage1.stage1a` should be evaluated alongside these methods as the
current production baseline.

The Rayleigh/RKHS methods estimate tissue coverage explicitly. By contrast,
`fisher_diffusion` and `multiblock_kernel_pls` are deliberately **independently
quantile-normalized controls**: each tissue is put on its own robust [0,1]
coordinate after fitting. Their scores therefore test axis recovery, not the
more general partial-coverage/common-gauge problem. They should not be used as
evidence that every tissue spans the same biological support.

## Direct-score Fisher geometry

For a smooth local mean-rate field `M_i(s)`, the independent-Poisson Fisher
pullback is

```text
G_i(s) = sum_g M_ig(s) grad log M_ig(s) grad log M_ig(s)^T.
```

The score-first estimators approximate `grad log M_ig` directly by weighted
local linear regression. The Rayleigh objective aligns `grad z_i` to the top
Fisher eigenspace without choosing an eigenvector sign, and weights the loss by
the local eigengap.

The direct score-Rayleigh reference now defaults to `k_neighbors=16`, reflecting
the most coherent setting in the first Qian PFC benchmark. The multiscale
method repeats the estimate at several `k` values and combines the sign-free
projectors `u_1 u_1^T`, weighting each scale by both its Fisher eigengap and its
agreement with the principal line estimated at the other scales. This directly
targets the neighborhood-size sensitivity seen in the first benchmark.

## Dynamics-ready low-rank expression model

`geometry_anchored_lowrank` represents mean expression in the Poisson
Fisher-flat coordinates

```text
Y = 2 sqrt(E)
```

and learns a low-rank surface

```text
Y(z,t) ~= mean + B a(z,t).
```

The basis `B` is estimated by PCA from local Fisher-flat mean-expression
estimates. The latent surface `a(z,t)` is a smooth tensor-product Bernstein
surface. The process coordinate is still strongly anchored by the local
score-Rayleigh geometry, so reconstruction is a secondary refinement signal
rather than a free latent-axis objective. The learned low-rank surface is also
a natural starting point for downstream estimates of cross-time
correspondence and Fisher arc length.

## Classical / spectral controls

`fisher_diffusion` builds a spatial KNN graph whose edge cost is much larger
for motion transverse to the local Fisher principal line than for motion along
it. A diffusion eigenfunction then provides a globally integrable scalar
coordinate. Unlike the retired eigenfield method, no oriented vector field is
integrated.

`multiblock_kernel_pls` is intentionally simpler. Each tissue is represented
by fixed RBF spatial features, while all tissues share one Fisher-flat
expression loading. The tissue spatial weights are the PLS directions most
predictive of that common expression component, which fixes orientation
atlas-wide up to one global sign.

## Shared distortion model

The Rayleigh methods retain the smooth tissue-specific monotone anatomical
warps used by the original zoo. For a common coordinate `z`, tissue `i` has

```text
h_i'(z) = exp(delta_i(z)) > 0,
```

with a low-order Bernstein expansion for `delta_i`. Gauge and roughness
penalties keep these warps interpretable while allowing partial process
coverage.

Reference-normalized neural and RKHS coordinate maps are robustly gauged by
lower/upper quantiles and then clipped to their nominal local [0,1] support
before the learned tissue coverage interval is applied. This keeps the
coordinate seen by the expression and warp models consistent for tail cells.

## Input/output contract

All methods accept an `AtlasData` object. `prepare_atlas(...)` accepts one
count matrix and one spatial coordinate matrix per tissue, plus observation
times and optional gene/tissue names. It retains raw physical coordinates and,
by default, uses only an isotropic similarity rescaling for fitting
coordinates.

```python
from flow_stp.zoo import prepare_atlas, fit_multiscale_score_rayleigh

atlas = prepare_atlas(
    counts_by_slice=counts_by_slice,
    spatial_by_slice=spatial_by_slice,
    times=times,
    gene_names=gene_names,
)
result = fit_multiscale_score_rayleigh(atlas)
```

Every estimator returns a `ZooResult` with `z_by_tissue` in atlas order,
method-specific auxiliary objects, and, when applicable, an optimization
history.
