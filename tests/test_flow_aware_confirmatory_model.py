import torch

from flow_stp.stage2.flow_aware import (
    FlowAwareStage2Model,
    FlowAwareTrainingConfig,
)


def test_confirmatory_flow_aware_defaults_and_constraints():
    model = FlowAwareStage2Model(num_genes=24)

    assert model.config.flow_segments == 24
    assert model.config.flow_hidden == (64, 64)
    assert model.config.rate_hidden == (96, 96, 96)
    assert model.config.decoder_hidden == (64, 64)
    assert model.config.r_quadrature == 12
    assert model.config.native_rate_init_bias == 0.40
    assert model.config.flow_spatial_smooth == 0.0
    assert model.config.flow_temporal_smooth == 0.0

    assert abs(float(model.quadrature_weights.sum()) - 1.0) < 1e-6

    z = torch.linspace(0.0, 1.0, 101)[:, None]
    t0 = torch.zeros_like(z)

    # Identity inverse map and exact developmental anchor at t=0.
    assert torch.max(torch.abs(model.inverse_coordinate(z, t0) - z)) < 2e-6
    assert torch.max(torch.abs(model.rho(z, t0))) < 1e-8

    # Hard nonnegative native material rate.
    rate = model.material_rate(z, torch.full_like(z, 0.5))
    assert torch.min(rate) >= 0.0

    # Confirmatory default has no flow regularizer.
    assert float(model.flow_penalty()) == 0.0

    output = model(z, torch.full_like(z, 0.5))
    assert output["log_mu"].shape == (101, 24)

    training = FlowAwareTrainingConfig()
    assert training.baseline_lr == 2e-3
    assert training.rate_lr == 1e-3
    assert training.decoder_lr == 1.5e-3
    assert training.flow_lr == 7.5e-4
    assert training.grad_clip == 5.0
