"""Diagnostics for Stage 2 fields and flow maps."""

from __future__ import annotations

from typing import Dict

import torch

from .model import Stage2DynamicsModel, Tensor


@torch.no_grad()
def evaluate_fields_on_grid(
    model: Stage2DynamicsModel,
    z_grid: Tensor,
    t_grid: Tensor,
) -> Dict[str, Tensor]:
    """Evaluate ``V``, ``q``, ``rho``, and ``F^{-1}`` on a Cartesian grid."""
    model.eval()

    device = next(model.parameters()).device
    z_grid = z_grid.to(device).reshape(-1)
    t_grid = t_grid.to(device).reshape(-1)

    zz, tt = torch.meshgrid(z_grid, t_grid, indexing="xy")
    z_flat = zz.reshape(-1, 1)
    t_flat = tt.reshape(-1, 1)

    velocity = model.velocity(z_flat, t_flat)
    source = model.q(z_flat, t_flat)
    reference, rho = model.inverse_reference_and_rho(z_flat, t_flat)

    shape = (t_grid.numel(), z_grid.numel())
    return {
        "z_grid": z_grid.detach().cpu(),
        "t_grid": t_grid.detach().cpu(),
        "velocity": velocity.reshape(shape).detach().cpu(),
        "q": source.reshape(shape).detach().cpu(),
        "a": reference.reshape(shape).detach().cpu(),
        "rho": rho.reshape(shape).detach().cpu(),
    }


@torch.no_grad()
def initial_condition_diagnostics(
    model: Stage2DynamicsModel,
    z_grid: Tensor,
) -> Dict[str, float]:
    """Check the exact time-zero conditions and report ``q(z,0)``."""
    model.eval()

    device = next(model.parameters()).device
    z = z_grid.to(device).reshape(-1)
    t = torch.zeros_like(z)

    reference, rho = model.inverse_reference_and_rho(z, t)
    source = model.q(z, t)

    return {
        "max_abs_F_inverse_z0_minus_z": float(
            (reference.reshape(-1) - z).abs().max().cpu()
        ),
        "rho_t0_min": float(rho.min().cpu()),
        "rho_t0_max": float(rho.max().cpu()),
        "q_t0_min": float(source.min().cpu()),
        "q_t0_max": float(source.max().cpu()),
    }


@torch.no_grad()
def flow_health_diagnostics(
    model: Stage2DynamicsModel,
    a_grid: Tensor,
    t_grid: Tensor,
) -> Dict[str, float]:
    """Check range and monotonicity of the derived one-dimensional flow."""
    model.eval()

    device = next(model.parameters()).device
    a = a_grid.to(device).reshape(-1)
    t_values = t_grid.to(device).reshape(-1)

    flow_rows = []
    for time_value in t_values:
        times = torch.full_like(a, time_value)
        flow_rows.append(model.flow_forward(a, times).reshape(-1))

    flow = torch.stack(flow_rows)
    increments = flow[:, 1:] - flow[:, :-1]

    return {
        "flow_min": float(flow.min().cpu()),
        "flow_max": float(flow.max().cpu()),
        "minimum_neighbor_increment": float(increments.min().cpu()),
        "monotonicity_violation_fraction": float(
            (increments <= 0.0).float().mean().cpu()
        ),
    }
