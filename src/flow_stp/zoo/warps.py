"""Monotone tissue-specific anatomical distortion maps."""

from __future__ import annotations

from typing import Optional
import math
import torch
from torch import nn

Tensor = torch.Tensor


def _bernstein_basis(z: Tensor, degree: int) -> Tensor:
    z = torch.clamp(z, 0.0, 1.0)
    pieces = []
    for k in range(degree + 1):
        coeff = math.comb(degree, k)
        pieces.append(coeff * z.pow(k) * (1.0 - z).pow(degree - k))
    return torch.stack(pieces, dim=-1)


class MonotoneWarpFamily(nn.Module):
    """Smooth monotone tissue warps with h_i'(z)=exp(delta_i(z))>0."""

    def __init__(self, n_tissues: int, degree: int = 4, clamp_log_derivative: float = 5.0, quadrature_points: int = 24) -> None:
        super().__init__()
        self.n_tissues = int(n_tissues)
        self.degree = int(degree)
        self.clamp_log_derivative = float(clamp_log_derivative)
        self.quadrature_points = int(quadrature_points)
        self.coefficients = nn.Parameter(torch.zeros(self.n_tissues, self.degree + 1))

    def log_derivative(self, z: Tensor, tissue_id: int) -> Tensor:
        basis = _bernstein_basis(z, self.degree)
        delta = basis @ self.coefficients[int(tissue_id)]
        c = self.clamp_log_derivative
        return torch.clamp(delta, min=-c, max=c)

    def derivative(self, z: Tensor, tissue_id: int) -> Tensor:
        return torch.exp(self.log_derivative(z, tissue_id))

    def one(self, z: Tensor, tissue_id: int) -> Tensor:
        shape = z.shape
        flat = torch.clamp(z.reshape(-1), 0.0, 1.0)
        q = max(self.quadrature_points, 4)
        grid = torch.linspace(0.0, 1.0, q, device=z.device, dtype=z.dtype)
        u = flat[:, None] * grid[None, :]
        deriv = self.derivative(u, tissue_id)
        integral = torch.trapezoid(deriv, grid, dim=1) * flat
        return integral.reshape(shape)

    def normalized_one(self, z: Tensor, tissue_id: int) -> Tensor:
        h = self.one(z, tissue_id)
        one = torch.ones(1, device=z.device, dtype=z.dtype)
        normalizer = torch.clamp(self.one(one, tissue_id)[0], min=1e-6)
        return h / normalizer

    def forward(self, z: Tensor, tissue_idx: Tensor, *, normalized: bool = False) -> Tensor:
        tissue_idx = tissue_idx.long().reshape(-1)
        z = z.reshape(-1)
        out = torch.empty_like(z)
        for tid_tensor in torch.unique(tissue_idx, sorted=True):
            tid = int(tid_tensor.item())
            idx = torch.nonzero(tissue_idx == tid_tensor, as_tuple=False).flatten()
            out[idx] = self.normalized_one(z[idx], tid) if normalized else self.one(z[idx], tid)
        return out

    def roughness_penalty(self, grid_n: int = 64) -> Tensor:
        grid = torch.linspace(0.0, 1.0, int(grid_n), device=self.coefficients.device, dtype=self.coefficients.dtype)
        terms = []
        for i in range(self.n_tissues):
            d = self.log_derivative(grid, i)
            terms.append((d[1:] - d[:-1]).square().mean())
        return torch.stack(terms).mean()

    def magnitude_penalty(self, grid_n: int = 64) -> Tensor:
        grid = torch.linspace(0.0, 1.0, int(grid_n), device=self.coefficients.device, dtype=self.coefficients.dtype)
        return torch.stack([self.log_derivative(grid, i).square().mean() for i in range(self.n_tissues)]).mean()

    def constant_scale_normalization_penalty(self, left: Optional[Tensor] = None, right: Optional[Tensor] = None, grid_n: int = 64) -> Tensor:
        grid = torch.linspace(0.0, 1.0, int(grid_n), device=self.coefficients.device, dtype=self.coefficients.dtype)
        terms = []
        for i in range(self.n_tissues):
            if left is None or right is None:
                z = grid
            else:
                mask = (grid >= left[i].detach()) & (grid <= right[i].detach())
                z = grid[mask]
                if z.numel() < 2:
                    z = grid
            terms.append(self.log_derivative(z, i).mean().square())
        return torch.stack(terms).mean()

    def consensus_gauge_penalty(self, left: Optional[Tensor] = None, right: Optional[Tensor] = None, grid_n: int = 64, min_overlap: int = 2) -> Tensor:
        grid = torch.linspace(0.0, 1.0, int(grid_n), device=self.coefficients.device, dtype=self.coefficients.dtype)
        logd = torch.stack([self.log_derivative(grid, i) for i in range(self.n_tissues)])
        if left is None or right is None:
            return logd.mean(dim=0).square().mean()
        mask = torch.stack([(grid >= left[i].detach()) & (grid <= right[i].detach()) for i in range(self.n_tissues)], dim=0)
        counts = mask.sum(dim=0)
        valid = counts >= int(min_overlap)
        if not bool(valid.any()):
            return torch.zeros((), device=grid.device, dtype=grid.dtype)
        weighted = (logd * mask.to(logd.dtype)).sum(dim=0)
        mean = weighted / torch.clamp(counts.to(logd.dtype), min=1.0)
        return mean[valid].square().mean()
