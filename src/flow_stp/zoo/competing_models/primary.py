"""Primary trajectory-inference coordinate model.

The selected M1 primary model is the stripped Stage1a neural-coordinate /
neural-expression estimator.  The former bells-and-whistles M1 is retained as
an explicit ablation configuration for real-data comparison.
"""

from dataclasses import replace

from ...stage1.stage1a import (
    FlowSTPConfig,
    FlowSTPModel,
    SliceCoordinateMaps,
    Stage1ARunConfig,
    Stage1AResult,
    run_stage1a,
)

# Selected M1: tissue-specific neural charts + shared neural E + Poisson
# likelihood + detached robust computational gauge.
PRIMARY_DEFAULT_CONFIG = Stage1ARunConfig()

# Historical/full M1 retained for ablation only.  This reproduces the previous
# scientific defaults: learned coverage, geometric penalties, rank-2 slice
# nuisance, and total-count library-size offset.
PRIMARY_BELLS_AND_WHISTLES_CONFIG = replace(
    PRIMARY_DEFAULT_CONFIG,
    output_dir=PRIMARY_DEFAULT_CONFIG.output_dir.parent
    / "flow_stp_stage1a_bells_and_whistles_ablation",
    submersion_weight=0.11364637,
    scale_variation_weight=1e-3,
    coverage_prior_weight=1e-4,
    nuisance_weight=5e-3,
    max_missing_each_end=0.25,
    initial_missing=0.005,
    slice_effect_dim=2,
    use_library_size_offset=True,
)

fit_primary = run_stage1a

__all__ = [
    "FlowSTPConfig",
    "FlowSTPModel",
    "SliceCoordinateMaps",
    "Stage1ARunConfig",
    "Stage1AResult",
    "PRIMARY_DEFAULT_CONFIG",
    "PRIMARY_BELLS_AND_WHISTLES_CONFIG",
    "fit_primary",
]
