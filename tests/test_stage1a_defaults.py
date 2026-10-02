from flow_stp.stage1 import (
    Stage1AOptimizedRunConfig,
    Stage1ARunConfig,
)
from flow_stp.stage1.stage1a import METHOD_VERSION
from flow_stp.stage1.stage1a_optimized import (
    OPTIMIZATION_PRESET_VERSION,
)


def test_stage1a_primary_defaults_are_library_offset_only():
    config = Stage1ARunConfig()

    assert METHOD_VERSION == "stage1a_library_offset_v1"

    # Selected primary observation model.
    assert config.use_library_size_offset is True

    # Optional regularizers / nuisance components are ablations, not defaults.
    assert config.submersion_weight == 0.0
    assert config.scale_variation_weight == 0.0
    assert config.coverage_prior_weight == 0.0
    assert config.nuisance_weight == 0.0
    assert config.slice_effect_dim == 0

    # Primary model uses the complete normalized coordinate interval.
    assert config.max_missing_each_end == 0.0
    assert config.initial_missing == 0.0


def test_stage1a_optimized_preset_changes_only_optimizer_mobility():
    baseline = Stage1ARunConfig()
    optimized = Stage1AOptimizedRunConfig()

    assert OPTIMIZATION_PRESET_VERSION == (
        "stage1a_library_offset_high_lr_v1"
    )
    assert baseline.learning_rate == 1e-3
    assert optimized.learning_rate == 2e-3

    # The complementary preset must not change the scientific model.
    for name in (
        "use_library_size_offset",
        "submersion_weight",
        "scale_variation_weight",
        "coverage_prior_weight",
        "nuisance_weight",
        "slice_effect_dim",
        "max_missing_each_end",
        "initial_missing",
        "coord_hidden",
        "decoder_hidden",
        "batch_size_per_slice",
        "weight_decay",
        "grad_clip_norm",
    ):
        assert getattr(optimized, name) == getattr(baseline, name)
