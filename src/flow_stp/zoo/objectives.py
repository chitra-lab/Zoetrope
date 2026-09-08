"""Geometric losses shared across the Stage-1 estimator zoo."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .common import CoordinateMaps, Tensor
from .fisher import eigengap_weight
from .warps import MonotoneWarpFamily


def coordinate_gradient(maps: CoordinateMaps, coords: Tensor, tissue_id: int, *, create_graph: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    x = coords.detach().clone().requires_grad_(True)
    z = maps.one(x, int(tissue_id))
    grad = torch.autograd.grad(z.sum(), x, create_graph=create_graph, retain_graph=create_graph, only_inputs=True)[0]
    return z, grad, x


def rayleigh_alignment_loss(grad: Tensor, tensor: Tensor, lambda1: Tensor, lambda2: Tensor, *, eps: float = 1e-8) -> tuple[Tensor, dict[str, Tensor]]:
    g2 = grad.square().sum(dim=1)
    rq_num = torch.einsum("bd,bde,be->b", grad, tensor, grad)
    rq = rq_num / (g2 + float(eps))
    normalized = torch.clamp(rq / (lambda1 + float(eps)), min=0.0, max=1.0)
    w = eigengap_weight(lambda1, lambda2, eps=eps)
    loss = (w * (1.0 - normalized)).sum() / torch.clamp(w.sum(), min=1.0)
    return loss, {"rayleigh_alignment": normalized.mean().detach(), "eigengap_weight": w.mean().detach()}


def submersion_loss(grad: Tensor, *, min_grad: float = 0.03, eps: float = 1e-12) -> tuple[Tensor, dict[str, Tensor]]:
    norm = torch.sqrt(grad.square().sum(dim=1) + float(eps))
    threshold = max(float(min_grad), 1e-8)
    deficit = F.relu((threshold - norm) / threshold)
    return deficit.square().mean(), {"grad_norm_mean": norm.mean().detach(), "grad_norm_min": norm.min().detach(), "submersion_violation_fraction": (norm < threshold).float().mean().detach()}


def corrected_physical_scale_loss(maps: CoordinateMaps, warps: MonotoneWarpFamily, coords: Tensor, tissue_id: int, *, eps: float = 1e-8) -> tuple[Tensor, dict[str, Tensor]]:
    """Encourage h_i(z_i) to advance at constant speed along process curves."""
    z, _, x = coordinate_gradient(maps, coords, tissue_id, create_graph=True)
    r = warps.one(z, tissue_id)
    grad_r = torch.autograd.grad(r.sum(), x, create_graph=True, retain_graph=True, only_inputs=True)[0]
    speed = torch.sqrt(grad_r.square().sum(dim=1) + float(eps))
    log_speed = torch.log(speed + float(eps))
    grad_log_speed = torch.autograd.grad(log_speed.sum(), x, create_graph=True, retain_graph=True, only_inputs=True)[0]
    unit = grad_r / speed[:, None]
    directional = (grad_log_speed * unit).sum(dim=1)
    return directional.square().mean(), {"corrected_speed_mean": speed.mean().detach(), "corrected_logspeed_directional_abs": directional.abs().mean().detach()}
