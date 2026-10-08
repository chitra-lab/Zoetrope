"""Tests for optional Stage-2 no-flux ODE postprocessing."""
import numpy as np
import pytest
import torch

from flow_stp.stage2 import (
    RefinementConfig, RefinedVelocityField, fit_trajectory_velocity,
    physical_velocity,
)


def test_no_flux_endpoints_and_flow():
    coeff = torch.ones((4, 4), dtype=torch.float64) * 1.7
    field = RefinedVelocityField(coeff, (0., 1.), (0., 1.), True)
    z = torch.tensor([0., .3, 1.], dtype=torch.float64)
    v = field.velocity(z, .25)
    assert v[0] == 0 and v[2] == 0 and v[1] > 0
    flow = field.integrate(z, torch.linspace(0., 1., 9, dtype=torch.float64))
    assert flow.position.shape == (9, 3)
    assert torch.allclose(flow.position[:, [0, 2]], z[[0, 2]].expand(9, -1))
    assert flow.position.min() >= 0 and flow.position.max() <= 1


def test_fit_trajectory_positions_instead_of_true_velocities():
    true_field = RefinedVelocityField(
        torch.ones((4, 4), dtype=torch.float64) * 1.1,
        (0., 1.), (0., 1.), True,
    )
    times = torch.linspace(0., 1., 7, dtype=torch.float64)
    initial = torch.linspace(.1, .9, 18, dtype=torch.float64)
    truth = true_field.integrate(initial, times, steps_per_interval=12).position
    obs = truth.numpy().copy()
    valid = np.ones_like(obs, dtype=bool)
    valid[2, 3] = False
    obs[2, 3] = np.nan
    result = fit_trajectory_velocity(
        obs, times, mask=valid,
        config=RefinementConfig(adam_steps=100, lbfgs_steps=25),
    )
    recovered = result.field.integrate(initial, times, steps_per_interval=12).position
    assert result.observed_count == valid.sum()
    assert result.train_position_rmse < .005
    assert torch.sqrt(torch.mean((recovered - truth).square())) < .005


def test_physical_velocity_pushforward():
    field = RefinedVelocityField(
        torch.ones((2, 2), dtype=torch.float64), (0., 1.), (0., 1.), True
    )
    u = torch.tensor([.1, .6], dtype=torch.float64)
    z = torch.tensor([.25, .75], dtype=torch.float64)
    t = torch.tensor([.3, .6], dtype=torch.float64)

    def gz(u, z, t):
        return torch.stack((torch.zeros_like(z), torch.full_like(z, 2.)), -1)

    def gt(u, z, t):
        return torch.stack((torch.full_like(z, .3), torch.zeros_like(z)), -1)

    migration, total = physical_velocity(field, u, z, t, dG_dz=gz, dG_dt=gt)
    assert torch.allclose(migration[:, 1], 2 * z * (1-z))
    assert torch.allclose(total - migration, gt(u,z,t))


def test_reject_invalid_input():
    with pytest.raises(ValueError, match="initial"):
        fit_trajectory_velocity(np.array([[.2, np.nan], [.3, .4]]), [0., 1.])
    with pytest.raises(ValueError, match="z_bounds"):
        RefinementConfig(z_bounds=(1., 0.))
    with pytest.raises(ValueError, match="outside z_bounds"):
        fit_trajectory_velocity(np.array([[.2], [1.2]]), [0., 1.])
