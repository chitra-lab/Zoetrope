# Frozen model selection

The exploratory estimator zoo is closed. After positive-control model selection
and the subsequent Stage1a ablation study, two model families are retained for
validation and the paper:

| role | model | purpose |
|---|---|---|
| **Primary M1** | stripped Stage1a neural coordinate + neural expression model | production estimator |
| **Secondary** | Fisher principal-curve registration | constrained geometric robustness check |

The selected Primary M1 is deliberately minimal:

```text
tissue-specific neural alpha_i
+ detached robust affine computational gauge
+ shared neural E(z,t)
+ Poisson likelihood
```

Its scientific defaults have fixed full coordinate coverage, no coordinate
geometry penalties, no jointly learned slice-gene nuisance term, and no
total-count library-size offset.

The earlier full/bells-and-whistles M1 is retained explicitly as
`PRIMARY_BELLS_AND_WHISTLES_CONFIG` for real-data ablation. It restores the
submersion penalty, relative scale-variation penalty, learned coverage endpoints
and coverage prior, rank-2 slice nuisance, and total-count library-size offset.
This ablation is intentionally not registered as a third model family: it is
the previous configuration of the same M1 estimator.

The primary and secondary model families deliberately occupy opposite ends of
the flexibility spectrum. Agreement on real data is therefore informative;
disagreement identifies sensitivity to modeling assumptions rather than to
small implementation variants.

All previously explored zoo methods have been removed from the active package
API. They remain available in git history for provenance.

## Frozen defaults

The public wrappers expose `PRIMARY_DEFAULT_CONFIG`,
`PRIMARY_BELLS_AND_WHISTLES_CONFIG`, and `SECONDARY_DEFAULT_CONFIG`.
Dataset-specific file paths and runtime-only settings may be changed, but the
scientific configuration should not be tuned per dataset unless a validation
experiment identifies a specific failure mode.

Known or independently justified technical offsets may be supplied as
observation-model information when appropriate. They are not part of the core
trajectory geometry.

## Reproducibility

Both retained pipelines use explicit seeds. The primary PyTorch runner enables
deterministic algorithms and disables cuDNN benchmarking when deterministic mode
is active. Reproducibility is intended for repeated runs under the same
software, hardware, and backend configuration; exact bitwise equality is not
promised across different PyTorch/CUDA stacks.

## Gauge convention and flow

The numerical anatomical coordinate is a chart, not a biologically unique
scalar. The fitted `(alpha, E)` pair chooses a computational gauge. If an
independently estimated flow map is expressed in another monotone coordinate
`u` with `z = h_t(u)`, express it in the `E` gauge by conjugation:

```text
F_E(s -> t) = h_t o F_u(s -> t) o h_s^{-1}.
```

No scientific gauge penalty is required to make `F` meaningful. If `F` is
derived directly from `E`, it is already expressed in the learned `E` gauge.
The progression quantities `q` and `rho` are invariant to such monotone chart
reparameterizations.
