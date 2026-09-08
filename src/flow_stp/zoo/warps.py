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
    """Smooth monotone tissue warps with ``h_i'(z)=exp(delta_i(z))>0``.

    ``delta_i`` is represented in a Bernstein basis.  The derivative used by
    the warp is smoothly bounded with ``tanh`` rather than hard-clamped.  This
    matters in long optimization runs: a hard clamp gives exactly zero gradient
    once a coefficient drives ``delta`` outside the clipping interval.
    """

    def __init__(
        self,
        n_tissues: int,
        degree: int = 4,
        clamp_log_derivative: float = 5.0,
        quadrature_points: int = 24,
    ) -> None:
        super().__init__()
        self.n_tissues = int(n_tissues)
        self.degree = int(degree)
        self.clamp_log_derivative = float(clamp_log_derivative)
        self.quadrature_points = int(quadrature_points)
        self.coefficients = nn.Parameter(
            torch.zeros(self.n_tissues, self.degree + 1)
        )

    def raw_log_derivative(self, z: Tensor, tissue_id: int) -> Tensor:
        basis = _bernstein_basis(z, self.degree)
        return basis @ self.coefficients[int(tissue_id)]

    def log_derivative(self, z: Tensor, tissue_id: int) -> Tensor:
        delta = self.raw_log_derivative(z, tissue_id)
        c = float(self.clamp_log_derivative)
        if c <= 0:
            return delta
        return c * torch.tanh(delta / c)

    def derivative(self, z: Tensor, tissue_id: int) -> Tensor:
        return torch.exp(self.log_derivative(z, tissue_id))

    def one(self, z: Tensor, tissue_id: int) -> Tensor:
        """Evaluate the forward warp with ``h_i(0)=0`` by quadrature."""
        shape = z.shape
        flat = torch.clamp(z.reshape(-1), 0.0, 1.0)
        q = max(self.quadrature_points, 4)
        grid = torch.linspace(
            0.0,
            1.0,
            q,
            device=z.device,
            dtype=z.dtype,
        )
        u = flat[:, None] * grid[None, :]
        deriv = self.derivative(u, tissue_id)
        integral = torch.trapezoid(deriv, grid, dim=1) * flat
        return integral.reshape(shape)

    def normalizer(self, tissue_id: int, *, like: Tensor) -> Tensor:
        one = torch.ones(1, device=like.device, dtype=like.dtype)
        return torch.clamp(self.one(one, tissue_id)[0], min=1e-6)

    def normalized_one(self, z: Tensor, tissue_id: int) -> Tensor:
        h = self.one(z, tissue_id)
        return h / self.normalizer(tissue_id, like=z)

    def normalized_derivative(self, z: Tensor, tissue_id: int) -> Tensor:
        return self.derivative(z, tissue_id) / self.normalizer(
            tissue_id,
            like=z,
        )

    def inverse_normalized_one(
        self,
        y: Tensor,
        tissue_id: int,
        *,
        iterations: int = 8,
        eps: float = 1e-6,
    ) -> Tensor:
        """Differentiably invert ``normalized_one`` by damped Newton steps."""
        target = torch.clamp(y, 0.0, 1.0)
        x = target.clone()
        for _ in range(max(int(iterations), 1)):
            value = self.normalized_one(x, tissue_id)
            deriv = torch.clamp(
                self.normalized_derivative(x, tissue_id),
                min=float(eps),
            )
            step = (value - target) / deriv
            x = torch.clamp(x - step, 0.0, 1.0)
        return x

    def forward(
        self,
        z: Tensor,
        tissue_idx: Tensor,
        *,
        normalized: bool = False,
    ) -> Tensor:
        tissue_idx = tissue_idx.long().reshape(-1)
        z = z.reshape(-1)
        out = torch.empty_like(z)
        for tid_tensor in torch.unique(tissue_idx, sorted=True):
            tid = int(tid_tensor.item())
            idx = torch.nonzero(
                tissue_idx == tid_tensor,
                as_tuple=False,
            ).flatten()
            if normalized:
                out[idx] = self.normalized_one(z[idx], tid)
            else:
                out[idx] = self.one(z[idx], tid)
        return out

    def roughness_penalty(self, grid_n: int = 64) -> Tensor:
        grid = torch.linspace(
            0.0,
            1.0,
            int(grid_n),
            device=self.coefficients.device,
            dtype=self.coefficients.dtype,
        )
        terms = []
        for i in range(self.n_tissues):
            # Penalize the unconstrained field so saturated warps are still
            # pulled back toward the regular regime.
            d = self.raw_log_derivative(grid, i)
            terms.append((d[1:] - d[:-1]).square().mean())
        return torch.stack(terms).mean()

    def magnitude_penalty(self, grid_n: int = 64) -> Tensor:
        grid = torch.linspace(
            0.0,
            1.0,
            int(grid_n),
            device=self.coefficients.device,
            dtype=self.coefficients.dtype,
        )
        return torch.stack(
            [
                self.raw_log_derivative(grid, i).square().mean()
                for i in range(self.n_tissues)
            ]
        ).mean()

    def constant_scale_normalization_penalty(
        self,
        left: Optional[Tensor] = None,
        right: Optional[Tensor] = None,
        grid_n: int = 64,
    ) -> Tensor:
        """Remove the tissue-wise constant scale absorbed by path scale."""
        grid = torch.linspace(
            0.0,
            1.0,
            int(grid_n),
            device=self.coefficients.device,
            dtype=self.coefficients.dtype,
        )
        terms = []
        for i in range(self.n_tissues):
            if left is None or right is None:
                z = grid
            else:
                mask = (
                    (grid >= left[i].detach())
                    & (grid <= right[i].detach())
                )
                z = grid[mask]
                if z.numel() < 2:
                    z = grid
            terms.append(self.log_derivative(z, i).mean().square())
        return torch.stack(terms).mean()

    def consensus_gauge_penalty(
        self,
        left: Optional[Tensor] = None,
        right: Optional[Tensor] = None,
        grid_n: int = 64,
        min_overlap: int = 2,
    ) -> Tensor:
        """Center *forward* tissue distortions on overlapping common support."""
        grid = torch.linspace(
            0.0,
            1.0,
            int(grid_n),
            device=self.coefficients.device,
            dtype=self.coefficients.dtype,
        )
        logd = torch.stack(
            [self.log_derivative(grid, i) for i in range(self.n_tissues)]
        )
        if left is None or right is None:
            return logd.mean(dim=0).square().mean()
        mask = torch.stack(
            [
                (grid >= left[i].detach()) & (grid <= right[i].detach())
                for i in range(self.n_tissues)
            ],
            dim=0,
        )
        counts = mask.sum(dim=0)
        valid = counts >= int(min_overlap)
        if not bool(valid.any()):
            return torch.zeros((), device=grid.device, dtype=grid.dtype)
        weighted = (logd * mask.to(logd.dtype)).sum(dim=0)
        mean = weighted / torch.clamp(counts.to(logd.dtype), min=1.0)
        return mean[valid].square().mean()

    def inverse_consensus_gauge_penalty(
        self,
        left: Tensor,
        right: Tensor,
        *,
        grid_n: int = 64,
        min_overlap: int = 2,
        inverse_iterations: int = 8,
    ) -> Tensor:
        """Gauge penalty when this family parameterizes local->common warps.

        Post-hoc registration uses ``k_i = h_i^{-1}``: the learned normalized
        warp maps a local physical coordinate to the common coordinate.  The
        biological gauge, however, centers the *forward* distortion ``h_i'``.
        At a common coordinate ``z`` we therefore evaluate

            log h_i'(z) = -log k_i'(h_i(z)).

        This routine performs that conversion numerically on the overlap grid.
        """
        grid = torch.linspace(
            0.0,
            1.0,
            int(grid_n),
            device=self.coefficients.device,
            dtype=self.coefficients.dtype,
        )
        rows = []
        masks = []
        for i in range(self.n_tissues):
            a = left[i].detach()
            b = right[i].detach()
            width = torch.clamp(b - a, min=1e-6)
            mask = (grid >= a) & (grid <= b)
            v = torch.clamp((grid - a) / width, 0.0, 1.0)
            local = self.inverse_normalized_one(
                v,
                i,
                iterations=inverse_iterations,
            )
            log_forward = -torch.log(
                torch.clamp(
                    self.normalized_derivative(local, i),
                    min=1e-8,
                )
            )
            rows.append(log_forward)
            masks.append(mask)
        logd = torch.stack(rows, dim=0)
        mask = torch.stack(masks, dim=0)
        counts = mask.sum(dim=0)
        valid = counts >= int(min_overlap)
        if not bool(valid.any()):
            return torch.zeros((), device=grid.device, dtype=grid.dtype)
        mean = (logd * mask.to(logd.dtype)).sum(dim=0) / torch.clamp(
            counts.to(logd.dtype),
            min=1.0,
        )
        return mean[valid].square().mean()
