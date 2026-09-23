# Frozen model selection

The exploratory estimator zoo is closed. After positive-control model selection,
two models are retained for validation and the paper:

| role | model | purpose |
|---|---|---|
| **Primary** | Stage1a neural coordinate + neural expression model | flexible production estimator |
| **Secondary** | Fisher principal-curve registration | constrained geometric robustness check |

The primary and secondary models deliberately occupy opposite ends of the
flexibility spectrum. Agreement on real data is therefore informative; disagreement
identifies sensitivity to modeling assumptions rather than to small implementation
variants.

All previously explored zoo methods have been removed from the active package API.
They remain available in git history for provenance.

## Frozen defaults

The public wrappers expose `PRIMARY_DEFAULT_CONFIG` and
`SECONDARY_DEFAULT_CONFIG`. These are the model-selection defaults used going
forward. Dataset-specific file paths and runtime-only settings may be changed, but
the scientific configuration should not be tuned per dataset unless a validation
experiment identifies a specific failure mode.

## Reproducibility

Both retained pipelines use explicit seeds. The primary PyTorch runner enables
deterministic algorithms and disables cuDNN benchmarking when deterministic mode is
active. Reproducibility is intended for repeated runs under the same software,
hardware, and backend configuration; exact bitwise equality is not promised across
different PyTorch/CUDA stacks.

## Gauge convention and flow

The numerical anatomical coordinate is a chart, not a biologically unique scalar.
The fitted `(alpha, E)` pair chooses a computational gauge. If an independently
estimated flow map is expressed in another monotone coordinate `u` with
`z = h_t(u)`, express it in the `E` gauge by conjugation:

```text
F_E(s -> t) = h_t o F_u(s -> t) o h_s^{-1}.
```

No scientific gauge penalty is required to make `F` meaningful. If `F` is derived
directly from `E`, it is already expressed in the learned `E` gauge. The progression
quantities `q` and `rho` are invariant to such monotone chart reparameterizations.
