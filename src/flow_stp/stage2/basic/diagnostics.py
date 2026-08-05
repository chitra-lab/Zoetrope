\
"""Diagnostics for the bare-bones centered-rho Stage 2 model."""

from __future__ import annotations

from typing import Dict

import torch

from .model import BasicStage2Model, Tensor


@torch.no_grad()
def evaluate_basic_fields_on_grid(
    model: BasicStage2Model,
    z_grid: Tensor,
    t_grid: Tensor,
    *,
    include_expression: bool = False,
) -> Dict[str, Tensor]:
    """Evaluate rho and optionally decoded expression on a Cartesian grid."""
    model.eval()
    device = next(model.parameters()).device
    z_values = z_grid.to(device).reshape(-1)
    t_values = t_grid.to(device).reshape(-1)

    tt, zz = torch.meshgrid(t_values, z_values, indexing="ij")
    z_flat = zz.reshape(-1)
    t_flat = tt.reshape(-1)
    rho = model.rho(z_flat, t_flat)

    result: Dict[str, Tensor] = {
        "z_grid": z_values.detach().cpu(),
        "t_grid": t_values.detach().cpu(),
        "rho": rho.reshape(t_values.numel(), z_values.numel()).detach().cpu(),
        "baseline_log_mean": model.baseline(z_values).detach().cpu(),
    }

    if include_expression:
        output = model(z_flat, t_flat)
        shape = (t_values.numel(), z_values.numel(), model.config.num_genes)
        result["developmental_log_effect"] = (
            output["developmental_log_effect"].reshape(shape).detach().cpu()
        )
        result["log_mu"] = output["log_mu"].reshape(shape).detach().cpu()
        result["mu"] = output["mu"].reshape(shape).detach().cpu()

    return result


@torch.no_grad()
def basic_initial_condition_diagnostics(
    model: BasicStage2Model,
    z_grid: Tensor,
) -> Dict[str, float]:
    """Check the exact rho and decoder anchors at the earliest time."""
    model.eval()
    device = next(model.parameters()).device
    z = z_grid.to(device).reshape(-1)
    zero = torch.zeros_like(z)
    rho_t0 = model.rho(z, zero)
    residual_at_zero = model.developmental_residual(z, zero)
    output_t0 = model(z, zero)
    baseline = model.baseline(z)

    return {
        "rho_t0_max_abs": float(rho_t0.abs().max().cpu()),
        "decoder_residual_at_zero_max_abs": float(
            residual_at_zero.abs().max().cpu()
        ),
        "forward_t0_minus_baseline_max_abs": float(
            (output_t0["log_mu"] - baseline).abs().max().cpu()
        ),
    }


@torch.no_grad()
def evaluate_basic_decoder_derivative(
    model: BasicStage2Model,
    z: Tensor,
    rho: Tensor,
    *,
    delta: float = 1e-3,
) -> Tensor:
    """Finite-difference local derivative dD(z,rho)/d(rho) for every gene."""
    if delta <= 0.0:
        raise ValueError("delta must be positive.")
    model.eval()
    device = next(model.parameters()).device
    z_value = z.to(device)
    rho_value = rho.to(device)
    upper = model.developmental_residual(z_value, rho_value + delta)
    lower = model.developmental_residual(z_value, rho_value - delta)
    return ((upper - lower) / (2.0 * delta)).detach().cpu()
