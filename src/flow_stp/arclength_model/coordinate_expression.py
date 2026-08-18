"""Joint anatomical-coordinate and mean-expression model.

The arclength model learns only two biological objects from counts:

    z = alpha_i(x, y)
    M(z, t) = log mean expression

Each tissue has its own smooth neural coordinate map ``alpha_i``.  The shared
expression field uses a fixed C2 temporal spline-like basis and a neural map of
``z`` to gene-by-basis coefficients.  There is no learned rho/q/Stage-2 decoder
in this package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Sequence[int], activation: str = "tanh"):
        super().__init__()
        acts = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}
        if activation not in acts:
            raise ValueError(f"unknown activation {activation!r}")
        layers = []
        d = in_dim
        for width in hidden:
            layers += [nn.Linear(d, int(width)), acts[activation]()]
            d = int(width)
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass
class ArclengthConfig:
    num_slices: int
    num_genes: int
    coord_hidden: Tuple[int, ...] = (64, 64)
    coefficient_hidden: Tuple[int, ...] = (96, 96)
    num_interior_time_knots: int = 2
    coord_grad_min: float = 0.05
    coord_grad_eps: float = 1e-12
    gauge_eps: float = 1e-4
    max_missing_each_end: float = 0.25
    initial_missing: float = 0.005
    coverage_prior_sd: float = 0.05
    submersion_weight: float = 0.11364637
    scale_variation_weight: float = 1e-3
    coverage_weight: float = 1e-4


class SliceCoordinateMaps(nn.Module):
    """Tissue-specific neural maps alpha_i: R^2 -> R with a detached gauge."""

    def __init__(self, config: ArclengthConfig):
        super().__init__()
        self.config = config
        self.maps = nn.ModuleList([
            MLP(2, 1, config.coord_hidden, "tanh") for _ in range(config.num_slices)
        ])
        self.missing_left = nn.Parameter(torch.full((config.num_slices,), config.initial_missing))
        self.missing_right = nn.Parameter(torch.full((config.num_slices,), config.initial_missing))
        self.register_buffer("gauge_low", torch.zeros(config.num_slices))
        self.register_buffer("gauge_high", torch.ones(config.num_slices))

    def raw_one_slice(self, coords: Tensor, slice_id: int) -> Tensor:
        return self.maps[int(slice_id)](coords).squeeze(-1)

    def local_and_global_one_slice(self, coords: Tensor, slice_id: int) -> tuple[Tensor, Tensor]:
        u = self.raw_one_slice(coords, slice_id)
        low = self.gauge_low[int(slice_id)]
        high = self.gauge_high[int(slice_id)]
        span = torch.clamp(high - low, min=self.config.gauge_eps)
        z_local = (u - low) / span
        a = self.missing_left[int(slice_id)]
        b = 1.0 - self.missing_right[int(slice_id)]
        z = a + (b - a) * z_local
        return z_local, z

    def forward(self, coords: Tensor, slice_idx: Tensor) -> Tensor:
        slice_idx = slice_idx.long().reshape(-1)
        out = torch.empty(len(coords), dtype=coords.dtype, device=coords.device)
        for s_tensor in torch.unique(slice_idx, sorted=True):
            s = int(s_tensor.item())
            idx = torch.nonzero(slice_idx == s_tensor, as_tuple=False).flatten()
            out = out.index_copy(0, idx, self.local_and_global_one_slice(coords[idx], s)[1])
        return out

    @torch.no_grad()
    def refresh_gauge_(self, coords: Tensor, slice_idx: Tensor, q_low: float = 0.005, q_high: float = 0.995) -> None:
        """Refresh detached robust affine gauges from reference coordinates."""
        slice_idx = slice_idx.long().reshape(-1)
        for s in range(self.config.num_slices):
            idx = torch.nonzero(slice_idx == s, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            u = self.raw_one_slice(coords[idx], s)
            self.gauge_low[s] = torch.quantile(u, q_low)
            self.gauge_high[s] = torch.quantile(u, q_high)

    def coverage_penalty(self) -> Tensor:
        sd = max(self.config.coverage_prior_sd, 1e-12)
        return 0.5 * ((self.missing_left / sd).square() + (self.missing_right / sd).square()).mean()

    @torch.no_grad()
    def project_endpoints_(self) -> None:
        self.missing_left.clamp_(0.0, self.config.max_missing_each_end)
        self.missing_right.clamp_(0.0, self.config.max_missing_each_end)


class TruncatedCubicTimeBasis(nn.Module):
    """Fixed C2 basis: 1,t,t^2,t^3,(t-k_j)_+^3.

    Time is expected to be normalized to [0,1].  The knots are fixed rather
    than learned, which makes M_t substantially more constrained than a generic
    time MLP when only a few developmental ages are observed.
    """

    def __init__(self, num_interior_knots: int = 2):
        super().__init__()
        if num_interior_knots < 0:
            raise ValueError("num_interior_knots must be nonnegative")
        if num_interior_knots:
            knots = torch.linspace(0.0, 1.0, num_interior_knots + 2)[1:-1]
        else:
            knots = torch.empty(0)
        self.register_buffer("knots", knots)

    @property
    def num_basis(self) -> int:
        return 4 + int(self.knots.numel())

    def forward(self, t: Tensor) -> Tensor:
        t = t.reshape(-1, 1)
        parts = [torch.ones_like(t), t, t.square(), t.pow(3)]
        if self.knots.numel():
            hinge = torch.relu(t - self.knots.reshape(1, -1)).pow(3)
            parts.append(hinge)
        return torch.cat(parts, dim=1)


class TemporalBasisExpressionField(nn.Module):
    """Shared M(z,t) with fixed temporal basis and neural z coefficients."""

    def __init__(self, config: ArclengthConfig):
        super().__init__()
        self.num_genes = int(config.num_genes)
        self.time_basis = TruncatedCubicTimeBasis(config.num_interior_time_knots)
        K = self.time_basis.num_basis
        self.coefficient_net = MLP(1, self.num_genes * K, config.coefficient_hidden, "silu")
        self.gene_intercept = nn.Parameter(torch.zeros(self.num_genes))

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        z = z.reshape(-1, 1)
        t = t.reshape(-1)
        B = self.time_basis(t)                                      # [N,K]
        C = self.coefficient_net(z).reshape(len(z), self.num_genes, -1)  # [N,G,K]
        return self.gene_intercept[None, :] + torch.einsum("ngk,nk->ng", C, B)


class ArclengthModel(nn.Module):
    """Jointly learns alpha_i(x,y) and M(z,t)."""

    def __init__(self, config: ArclengthConfig):
        super().__init__()
        self.config = config
        self.coordinate_maps = SliceCoordinateMaps(config)
        self.expression_field = TemporalBasisExpressionField(config)

    def forward(
        self,
        *,
        coords: Tensor,
        slice_idx: Tensor,
        time: Tensor,
        log_library_size: Optional[Tensor] = None,
    ) -> Mapping[str, Tensor]:
        z = self.coordinate_maps(coords, slice_idx)
        M = self.expression_field(z, time)
        log_mu = M
        if log_library_size is not None:
            log_mu = log_mu + log_library_size.reshape(-1, 1).to(log_mu)
        return {"z": z[:, None], "M": M, "log_mu": log_mu}

    @staticmethod
    def poisson_nll(counts: Tensor, log_mu: Tensor, full: bool = True) -> Tensor:
        return F.poisson_nll_loss(log_mu, counts, log_input=True, full=full, reduction="mean")

    def coordinate_submersion_penalty(
        self,
        coords: Tensor,
        slice_idx: Tensor,
        *,
        samples_per_slice: Optional[int] = 128,
        create_graph: bool = True,
    ) -> tuple[Tensor, Mapping[str, Tensor]]:
        """Penalize ||grad alpha_i|| below a fixed threshold on local z."""
        slice_idx = slice_idx.long().reshape(-1)
        penalties, norms = [], []
        for s_tensor in torch.unique(slice_idx, sorted=True):
            s = int(s_tensor.item())
            idx = torch.nonzero(slice_idx == s_tensor, as_tuple=False).flatten()
            if samples_per_slice and idx.numel() > samples_per_slice:
                idx = idx[torch.randperm(idx.numel(), device=idx.device)[:samples_per_slice]]
            xy = coords[idx].detach().clone().requires_grad_(True)
            z_local, _ = self.coordinate_maps.local_and_global_one_slice(xy, s)
            grad = torch.autograd.grad(z_local.sum(), xy, create_graph=create_graph, retain_graph=create_graph)[0]
            norm = torch.sqrt(grad.square().sum(1) + self.config.coord_grad_eps)
            p = torch.relu((self.config.coord_grad_min - norm) / self.config.coord_grad_min).square()
            penalties.append(p)
            norms.append(norm)
        all_p = torch.cat(penalties)
        all_n = torch.cat(norms)
        return all_p.mean(), {"grad_norm": all_n.detach()}

    def coordinate_scale_variation_penalty(
        self,
        coords: Tensor,
        slice_idx: Tensor,
        *,
        samples_per_slice: Optional[int] = 64,
    ) -> Tensor:
        """Relative normal-scale variation (grad z)^T H_z grad z / ||grad z||^3."""
        slice_idx = slice_idx.long().reshape(-1)
        vals = []
        for s_tensor in torch.unique(slice_idx, sorted=True):
            s = int(s_tensor.item())
            idx = torch.nonzero(slice_idx == s_tensor, as_tuple=False).flatten()
            if samples_per_slice and idx.numel() > samples_per_slice:
                idx = idx[torch.randperm(idx.numel(), device=idx.device)[:samples_per_slice]]
            xy = coords[idx].detach().clone().requires_grad_(True)
            z_local, _ = self.coordinate_maps.local_and_global_one_slice(xy, s)
            grad = torch.autograd.grad(z_local.sum(), xy, create_graph=True, retain_graph=True)[0]
            directional = (grad * grad).sum(1)
            hgrad = torch.autograd.grad(directional.sum(), xy, create_graph=True, retain_graph=True)[0] / 2.0
            numerator = (hgrad * grad).sum(1)
            denom = (grad.square().sum(1) + self.config.coord_grad_eps).pow(1.5)
            vals.append((numerator / denom).square())
        return torch.cat(vals).mean()

    def loss(
        self,
        *,
        coords: Tensor,
        slice_idx: Tensor,
        time: Tensor,
        counts: Tensor,
        log_library_size: Optional[Tensor] = None,
        submersion_samples_per_slice: int = 128,
        scale_samples_per_slice: int = 64,
    ) -> Mapping[str, Tensor]:
        out = self(coords=coords, slice_idx=slice_idx, time=time, log_library_size=log_library_size)
        nll = self.poisson_nll(counts, out["log_mu"])
        sub, _ = self.coordinate_submersion_penalty(
            coords, slice_idx, samples_per_slice=submersion_samples_per_slice, create_graph=True
        )
        scale = self.coordinate_scale_variation_penalty(
            coords, slice_idx, samples_per_slice=scale_samples_per_slice
        ) if self.config.scale_variation_weight > 0 else nll.new_zeros(())
        coverage = self.coordinate_maps.coverage_penalty()
        total = (
            nll
            + self.config.submersion_weight * sub
            + self.config.scale_variation_weight * scale
            + self.config.coverage_weight * coverage
        )
        return {"loss": total, "nll": nll, "submersion": sub, "scale_variation": scale, "coverage": coverage, **out}


def normalize_time(time: Tensor, t_min: float, t_max: float) -> Tensor:
    if not t_max > t_min:
        raise ValueError("t_max must be larger than t_min")
    return (time - float(t_min)) / float(t_max - t_min)
