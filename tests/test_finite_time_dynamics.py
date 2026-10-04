import torch
from torch import nn

from flow_stp.fisher_lagrangian import (
    DynamicsConfig,
    fit_finite_time_dynamics,
    fit_finite_time_interval,
    infer_dynamics,
)


class OrthogonalTimeField(nn.Module):
    """Psi=(z+2, t+2): spatial identity flow with orthogonal molecular change."""

    def forward(self, z, time):
        if z.ndim == 1:
            z = z[:, None]
        if time.ndim == 1:
            time = time[:, None]
        psi = torch.cat((z + 2.0, time + 2.0), dim=1)
        return torch.log((psi / 2.0).square())


def test_dynamics_default_mode_is_finite_time():
    assert DynamicsConfig().mode == "finite_time"


def test_finite_time_identity_flow_for_orthogonal_change():
    field = OrthogonalTimeField()
    z_grid = torch.linspace(0.0, 1.0, 81)

    config = DynamicsConfig(
        k_candidates=(4,),
        validation_fraction=0.5,
        optimizer_steps=250,
        learning_rate=5e-2,
        seed=3,
    )

    fit = fit_finite_time_interval(
        field,
        source_time=0.0,
        target_time=0.5,
        z_grid=z_grid,
        config=config,
    )

    predicted = fit.flow(z_grid)
    assert fit.selected_k == 4
    assert torch.allclose(predicted, z_grid, atol=2e-3, rtol=0.0)
    assert torch.allclose(
        fit.flow.interval_velocity(z_grid),
        torch.zeros_like(z_grid),
        atol=4e-3,
        rtol=0.0,
    )


def test_finite_time_rho_is_fisher_chord_sum():
    field = OrthogonalTimeField()
    z_grid = torch.linspace(0.0, 1.0, 81)
    anchors = torch.tensor([0.2, 0.8])

    config = DynamicsConfig(
        k_candidates=(4,),
        validation_fraction=0.5,
        optimizer_steps=250,
        learning_rate=5e-2,
        seed=4,
    )

    traj = fit_finite_time_dynamics(
        field,
        observed_times=(0.0, 0.25, 0.5, 1.0),
        z_grid=z_grid,
        a=anchors,
        config=config,
    )

    expected = torch.tensor([0.0, 0.25, 0.5, 1.0])[:, None].expand(-1, 2)
    assert torch.allclose(traj.position, anchors[None, :].expand(4, -1), atol=2e-3)
    assert torch.allclose(traj.rho, expected, atol=3e-3)
    assert torch.allclose(
        traj.interval_velocity,
        torch.zeros_like(traj.interval_velocity),
        atol=4e-3,
    )


def test_infer_dynamics_can_toggle_to_continuous():
    field = OrthogonalTimeField()
    anchors = torch.tensor([0.2, 0.8])

    result = infer_dynamics(
        field,
        a=anchors,
        observed_times=(0.0, 1.0),
        config=DynamicsConfig(
            mode="continuous",
            continuous_steps=16,
        ),
    )

    assert result.mode == "continuous"
    assert result.flow is None
    assert torch.allclose(result.position[-1], anchors, atol=1e-5)
    assert torch.allclose(result.rho[-1], torch.ones_like(anchors), atol=1e-4)
