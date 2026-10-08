import torch

from flow_stp.stage1 import (
    Stage1AOptimizedRunConfig,
    Stage1ARunConfig,
)
from flow_stp.stage1.stage1a import (
    METHOD_VERSION,
    FlowSTPConfig,
    FlowSTPModel,
)
from flow_stp.stage1.stage1a_optimized import (
    OPTIMIZATION_PRESET_VERSION,
)


def test_stage1a_primary_defaults_are_library_offset_fisher_flat():
    config = Stage1ARunConfig()

    assert METHOD_VERSION == (
        "stage1a_library_offset_fisher_flat_v1"
    )

    # Selected primary observation model.
    assert config.use_library_size_offset is True

    # Optional regularizers / nuisance components are not active.
    assert config.submersion_weight == 0.0
    assert config.scale_variation_weight == 0.0
    assert config.coverage_prior_weight == 0.0
    assert config.nuisance_weight == 0.0
    assert config.slice_effect_dim == 0

    # Primary model uses the complete normalized coordinate interval.
    assert config.max_missing_each_end == 0.0
    assert config.initial_missing == 0.0


def test_stage1_parameterizes_psi_directly_and_reconstructs_poisson_mean():
    model = FlowSTPModel(
        FlowSTPConfig(
            num_slices=1,
            num_genes=2,
            coord_hidden=(4,),
            decoder_hidden=(4,),
        )
    )

    with torch.no_grad():
        model.initialize_fisher_flat_baseline_(
            torch.tensor([4.0, 6.0])
        )
        # Make the residual field exactly zero for this algebraic test.
        for parameter in model.decoder.parameters():
            parameter.zero_()

    z = torch.tensor([0.2, 0.8])
    t = torch.tensor([0.1, 0.9])

    psi = model.fisher_flat(
        z,
        t,
    )
    expression = model.expression(
        z,
        t,
    )

    assert torch.allclose(
        psi,
        torch.tensor(
            [[4.0, 6.0], [4.0, 6.0]]
        ),
    )
    assert torch.allclose(
        expression,
        torch.tensor(
            [[4.0, 9.0], [4.0, 9.0]]
        ),
    )


def test_library_offset_multiplies_reconstructed_expression_mean():
    model = FlowSTPModel(
        FlowSTPConfig(
            num_slices=1,
            num_genes=2,
            coord_hidden=(4,),
            decoder_hidden=(4,),
        )
    )

    with torch.no_grad():
        model.initialize_fisher_flat_baseline_(
            torch.tensor([4.0, 6.0])
        )
        for parameter in model.decoder.parameters():
            parameter.zero_()

    output = model(
        coords=torch.tensor(
            [[0.0, 0.0], [1.0, 1.0]]
        ),
        slice_idx=torch.zeros(
            2,
            dtype=torch.long,
        ),
        time=torch.tensor([0.2, 0.8]),
        log_library_size=torch.log(
            torch.tensor([2.0, 0.5])
        ),
    )

    expected_mu = torch.tensor(
        [
            [8.0, 18.0],
            [2.0, 4.5],
        ]
    )

    assert torch.allclose(
        torch.exp(
            output["log_mu"]
        ),
        expected_mu,
        atol=1e-6,
        rtol=1e-6,
    )


def test_stage1a_optimized_preset_changes_only_optimizer_mobility():
    baseline = Stage1ARunConfig()
    optimized = Stage1AOptimizedRunConfig()

    assert OPTIMIZATION_PRESET_VERSION == (
        "stage1a_library_offset_fisher_flat_high_lr_v1"
    )
    assert baseline.learning_rate == 1e-3
    assert optimized.learning_rate == 2e-3

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
        assert getattr(
            optimized,
            name,
        ) == getattr(
            baseline,
            name,
        )
