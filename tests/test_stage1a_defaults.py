from flow_stp.stage1 import Stage1ARunConfig
from flow_stp.stage1.stage1a import METHOD_VERSION


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
