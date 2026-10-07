import torch

from flow_stp.stage2 import (
    estimate_local_dynamics,
    integrate_flow_rk4,
)


def _linear_log_expression(z, time):
    # One-gene field with eta(z,t) = z - 2t.
    # Because Psi_z and Psi_t share the same positive factor,
    # V = -eta_t / eta_z = 2 everywhere.
    return z - 2.0 * time


def test_closed_form_velocity_from_expression_derivatives():
    z = torch.tensor([[0.1], [0.4], [0.8]], dtype=torch.float64)
    t = torch.tensor([[0.0], [0.3], [0.7]], dtype=torch.float64)

    result = estimate_local_dynamics(
        _linear_log_expression,
        z,
        t,
        eps=0.0,
    )

    assert torch.allclose(
        result.velocity,
        torch.full_like(z, 2.0),
        atol=1e-10,
        rtol=1e-10,
    )

    inner = (result.residual * result.dpsi_dz).sum(dim=1)
    assert torch.allclose(
        inner,
        torch.zeros_like(inner),
        atol=1e-10,
        rtol=1e-10,
    )


def test_rk4_recovers_constant_velocity_flow():
    anchors = torch.tensor(
        [[-0.5], [0.0], [0.75]],
        dtype=torch.float64,
    )
    trajectory = integrate_flow_rk4(
        _linear_log_expression,
        anchors,
        t0=0.0,
        t1=0.6,
        n_steps=12,
        eps=0.0,
    )

    expected = anchors[:, 0] + 1.2
    assert torch.allclose(
        trajectory.final_position,
        expected,
        atol=1e-10,
        rtol=1e-10,
    )
