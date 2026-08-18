import torch

from flow_stp.stage2.neural_pde import (
    NEURAL_PDE_ARCHITECTURE_VERSION,
    NeuralPDEStage2Model,
    NeuralPDETrainingConfig,
    make_neural_pde_optimizer,
)


def test_neural_pde_confirmatory_defaults_and_constraints():
    model = NeuralPDEStage2Model(num_genes=24, train_rk4_steps=4)

    assert NEURAL_PDE_ARCHITECTURE_VERSION == "eulerian-neural-pde-v1-2026-08-17"
    assert model.config.baseline_hidden == (64, 64)
    assert model.config.decoder_hidden == (64, 64)
    assert model.config.dynamic_hidden == (96, 96, 96)
    assert model.config.velocity_activation == "tanh"
    assert model.config.source_activation == "silu"
    assert model.config.source_initial_bias == 0.40

    z = torch.linspace(0.05, 0.95, 17)
    t0 = torch.zeros_like(z)
    t = torch.linspace(0.0, 1.0, len(z))

    assert torch.equal(model.rho(z, t0), torch.zeros(len(z), 1))
    assert torch.equal(
        model.developmental_residual(z, torch.zeros_like(z)),
        torch.zeros(len(z), 24),
    )
    assert torch.min(model.q(z, t)) >= 0.0

    output = model(z, t)
    assert output["a"].shape == (len(z), 1)
    assert output["rho"].shape == (len(z), 1)
    assert output["log_mu"].shape == (len(z), 24)

    torch.testing.assert_close(
        output["log_mu"],
        model.decode_log_mean(z, output["rho"]),
    )


def test_neural_pde_gradients_reach_both_dynamic_fields():
    torch.manual_seed(19)
    model = NeuralPDEStage2Model(num_genes=7, train_rk4_steps=4)

    z = torch.linspace(0.05, 0.95, 12)
    t = torch.linspace(0.1, 1.0, 12)
    output = model(z, t)

    loss = output["log_mu"].square().mean() + output["rho"].mean()
    loss.backward()

    assert any(p.grad is not None for p in model.v_net.parameters())
    assert any(p.grad is not None for p in model.q_net.parameters())
    assert any(p.grad is not None for p in model.decoder_net.parameters())


def test_neural_pde_optimizer_matches_confirmatory_learning_rates():
    model = NeuralPDEStage2Model(num_genes=5, train_rk4_steps=2)
    config = NeuralPDETrainingConfig()
    optimizer = make_neural_pde_optimizer(model, config)

    assert [group["lr"] for group in optimizer.param_groups] == [
        1e-3,
        1e-3,
        1.5e-3,
    ]
