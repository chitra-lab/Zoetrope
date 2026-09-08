# Stage-1 estimator zoo

This folder contains alternative estimators of the **same frozen geometric object**: tissue-specific process maps into a common one-dimensional process manifold, with a consensus physical ruler and smooth tissue-specific anatomical distortion.

The current production `stage1a` implementation remains in `flow_stp.stage1.stage1a` and is not duplicated here.

## Ranked methods

| rank | module | estimator |
|---:|---|---|
| 1 | `joint_likelihood_fisher_warp` | raw Poisson likelihood + independently estimated Fisher principal tensor + tissue-specific warps |
| 2 | `meanfield_rayleigh_warp` | smooth local mean fields -> Fisher tensor -> direct Rayleigh coordinate + tissue warps |
| 3 | `score_rayleigh_warp` | direct local spatial log-rate scores -> Fisher tensor -> direct Rayleigh coordinate + tissue warps |
| 4 | `flow_stp.stage1.stage1a` | existing direct Stage1a baseline; not duplicated |
| 5 | `independent_register` | independent tissue coordinates -> post-hoc monotone registration |
| 6 | `eigenfield_warp` | mean-field Fisher tensor -> explicit principal-eigenfield integration -> registration |
| 7 | `principal_arclength_warp` | principal-line graph arclength in original physical coordinates -> registration |

## Shared distortion model

For a common process coordinate `z`, tissue `i` has a monotone anatomical warp `h_i` with positive derivative. `MonotoneWarpFamily` represents

```text
h_i'(z) = exp(delta_i(z)) > 0
```

where `delta_i` is a low-order Bernstein expansion. The nonlinear common gauge is centered on overlapping support by encouraging

```text
mean_i log h_i'(z) = 0.
```

The per-tissue constant scale is normalized separately because it is absorbed by the path-specific linear factor in the geometric model.

For jointly learned coordinates the corrected physical ruler `h_i(z_i(s))` is encouraged to have constant speed along process trajectories through the local condition

```text
d/dell log ||grad h_i(z_i)|| = 0.
```

This permits smooth tissue-level expansion/contraction while preventing an independent nonlinear warp for every principal trajectory.

## Fisher geometry

For a smooth local mean-rate field `M_i(s)`, the independent-Poisson Fisher pullback is evaluated as

```text
G_i(s) = sum_g M_ig(s) grad log M_ig(s) grad log M_ig(s)^T.
```

The direct Rayleigh methods align `grad z_i` to the top eigenspace without choosing a sign for the eigenvector. The alignment loss is weighted by the local eigengap.

The score-first method estimates `grad log M_ig` directly by weighted local linear regression. It is score-first in target but is **not** an implementation of SLOPER score matching.

## Input contract

All methods accept an `AtlasData` object. The helper `prepare_atlas(...)` accepts one count matrix and spatial coordinate matrix per tissue, plus observation times and optional gene/tissue names. It keeps both raw physical coordinates and fitting coordinates. By default the latter use only an isotropic similarity rescaling, so Euclidean angles and the physical principal line are preserved.

```python
from flow_stp.zoo import prepare_atlas, fit_meanfield_rayleigh_warp

atlas = prepare_atlas(
    counts_by_slice=counts_by_slice,
    spatial_by_slice=spatial_by_slice,
    times=times,
    gene_names=gene_names,
)
result = fit_meanfield_rayleigh_warp(atlas)
```

Every estimator returns a `ZooResult` with `z_by_tissue` in atlas order, method-specific auxiliary objects, and a lightweight optimization history.
