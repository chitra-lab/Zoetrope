"""RBF/RKHS process coordinate fitted against direct-score Fisher geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .common import (
    AtlasData,
    SharedSqrtDecoder,
    ZooResult,
    default_device,
    sample_indices,
    set_seed,
    sqrt_fisher_coordinate,
)
from .fisher import score_first_fisher_geometry
from .objectives import (
    coordinate_gradient,
    corrected_physical_scale_loss,
    rayleigh_alignment_loss,
    submersion_loss,
)
from .trainers import GeometryWeights
from .warps import MonotoneWarpFamily

Tensor = torch.Tensor


class RBFCoordinateMaps(nn.Module):
    """Tissue-specific smooth scalar fields in a fixed Gaussian RBF basis."""

    def __init__(
        self,
        atlas: AtlasData,
        *,
        n_centers: int,
        max_missing_each_end: float,
        initial_missing: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.n_tissues = atlas.n_tissues
        self.max_missing_each_end = float(max_missing_each_end)
        self.n_centers = int(n_centers)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        self.coefficients = nn.ParameterList()

        frac = np.clip(
            float(initial_missing) / max(float(max_missing_each_end), 1e-8),
            1e-5,
            1.0 - 1e-5,
        )
        init = float(math.log(frac / (1.0 - frac)))
        self.left_logits = nn.Parameter(torch.full((self.n_tissues,), init))
        self.right_logits = nn.Parameter(torch.full((self.n_tissues,), init))

        for tid, tissue in enumerate(atlas.tissues):
            x = tissue.coords.detach().cpu()
            m = min(max(self.n_centers, 4), len(x))
            if len(x) <= m:
                centers = x.clone()
            else:
                idx = torch.randperm(len(x), generator=gen)[:m]
                centers = x[idx].clone()
            if len(centers) > 1:
                d = torch.pdist(centers)
                positive = d[d > 1e-6]
                bandwidth = (
                    positive.median()
                    if positive.numel()
                    else torch.tensor(1.0)
                )
            else:
                bandwidth = torch.tensor(1.0)
            bandwidth = torch.clamp(bandwidth, min=1e-2)
            self.register_buffer(f"centers_{tid}", centers)
            self.register_buffer(f"bandwidth_{tid}", bandwidth)
            self.register_buffer(f"gauge_low_{tid}", torch.tensor(0.0))
            self.register_buffer(f"gauge_high_{tid}", torch.tensor(1.0))
            coeff = 0.02 * torch.randn(m + 3, generator=gen)
            coeff[-3] = 0.0
            self.coefficients.append(nn.Parameter(coeff))

    def _features(self, coords: Tensor, tissue_id: int) -> Tensor:
        centers = getattr(self, f"centers_{int(tissue_id)}").to(coords)
        bandwidth = getattr(self, f"bandwidth_{int(tissue_id)}").to(coords)
        sqdist = torch.cdist(coords, centers).square()
        radial = torch.exp(-0.5 * sqdist / bandwidth.square())
        ones = torch.ones((len(coords), 1), device=coords.device, dtype=coords.dtype)
        return torch.cat([radial, coords, ones], dim=1)

    def raw_one(self, coords: Tensor, tissue_id: int) -> Tensor:
        return self._features(coords, tissue_id) @ self.coefficients[int(tissue_id)]

    def local_one(self, coords: Tensor, tissue_id: int) -> Tensor:
        raw = self.raw_one(coords, tissue_id)
        low = getattr(self, f"gauge_low_{int(tissue_id)}").to(raw)
        high = getattr(self, f"gauge_high_{int(tissue_id)}").to(raw)
        local = (raw - low) / torch.clamp(high - low, min=1e-4)
        return torch.clamp(local, 0.0, 1.0)

    def coverage_intervals(self) -> tuple[Tensor, Tensor]:
        left = self.max_missing_each_end * torch.sigmoid(self.left_logits)
        right = 1.0 - self.max_missing_each_end * torch.sigmoid(
            self.right_logits
        )
        return left, right

    def one(self, coords: Tensor, tissue_id: int) -> Tensor:
        local = self.local_one(coords, tissue_id)
        left, right = self.coverage_intervals()
        return left[int(tissue_id)] + (
            right[int(tissue_id)] - left[int(tissue_id)]
        ) * local

    def coverage_prior(self, sd: float = 0.08) -> Tensor:
        left, right = self.coverage_intervals()
        missing = torch.cat([left, 1.0 - right])
        return 0.5 * (missing / max(float(sd), 1e-8)).square().mean()

    def roughness_penalty(self) -> Tensor:
        terms = []
        for tid, coeff in enumerate(self.coefficients):
            m = getattr(self, f"centers_{tid}").shape[0]
            terms.append(coeff[:m].square().mean())
        return torch.stack(terms).mean()

    @torch.no_grad()
    def refresh_gauge_(
        self,
        coords_by_tissue: Sequence[Tensor],
        *,
        max_points: int = 4096,
        lower_quantile: float = 0.005,
        upper_quantile: float = 0.995,
    ) -> None:
        for tid, coords in enumerate(coords_by_tissue):
            x = coords
            if max_points > 0 and len(x) > int(max_points):
                idx = torch.linspace(
                    0,
                    len(x) - 1,
                    int(max_points),
                    device=x.device,
                ).round().long()
                x = x[idx]
            raw = self.raw_one(x, tid)
            low = torch.quantile(raw, float(lower_quantile))
            high = torch.quantile(raw, float(upper_quantile))
            if float(high - low) < 1e-4:
                center = 0.5 * (high + low)
                low = center - 5e-5
                high = center + 5e-5
            getattr(self, f"gauge_low_{tid}").copy_(low)
            getattr(self, f"gauge_high_{tid}").copy_(high)


class Model(nn.Module):
    def __init__(self, atlas: AtlasData, config: "Config") -> None:
        super().__init__()
        self.coordinate_maps = RBFCoordinateMaps(
            atlas,
            n_centers=config.n_centers,
            max_missing_each_end=config.max_missing_each_end,
            initial_missing=config.initial_missing,
            seed=config.seed,
        )
        self.warps = MonotoneWarpFamily(
            atlas.n_tissues, degree=config.warp_degree
        )
        self.decoder = SharedSqrtDecoder(
            atlas.n_genes, hidden=config.decoder_hidden
        )


@dataclass
class Config:
    k_neighbors: int = 16
    ridge: float = 1e-3
    pseudocount: float = 0.5
    n_centers: int = 48
    epochs: int = 900
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    decoder_hidden: tuple[int, ...] = (128, 128)
    warp_degree: int = 4
    min_grad: float = 0.03
    max_missing_each_end: float = 0.30
    initial_missing: float = 0.01
    gauge_reference_points: int = 4096
    gauge_lower_quantile: float = 0.005
    gauge_upper_quantile: float = 0.995
    reconstruction_weight: float = 1.0
    rbf_roughness_weight: float = 1e-3
    weights: GeometryWeights = field(default_factory=GeometryWeights)
    seed: int = 20260908
    print_every: int = 100


def fit(
    atlas: AtlasData,
    config: Config = Config(),
    *,
    device: Optional[torch.device] = None,
) -> ZooResult:
    if device is None:
        device = default_device()
    set_seed(config.seed)
    geometry = score_first_fisher_geometry(
        atlas,
        k_neighbors=config.k_neighbors,
        ridge=config.ridge,
        pseudocount=config.pseudocount,
    )
    atlas_dev = atlas.to(device)
    model = Model(atlas_dev, config).to(device)
    target_y = [
        sqrt_fisher_coordinate(rate).to(device)
        for rate in geometry.mean_rates
    ]
    G_dev = [x.to(device) for x in geometry.tensors]
    l1_dev = [x.to(device) for x in geometry.lambda1]
    l2_dev = [x.to(device) for x in geometry.lambda2]
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    history: list[dict[str, float]] = []
    w = config.weights

    for epoch in range(int(config.epochs)):
        model.coordinate_maps.refresh_gauge_(
            [t.coords for t in atlas_dev.tissues],
            max_points=config.gauge_reference_points,
            lower_quantile=config.gauge_lower_quantile,
            upper_quantile=config.gauge_upper_quantile,
        )
        rec_terms, dir_terms, sub_terms, scale_terms, align_terms = (
            [], [], [], [], []
        )
        for tid, tissue in enumerate(atlas_dev.tissues):
            idx = sample_indices(
                tissue.n_cells, config.batch_size, device=device
            )
            x = tissue.coords[idx]
            z = model.coordinate_maps.one(x, tid)
            t = torch.full_like(z, float(tissue.time))
            rec_terms.append(
                F.mse_loss(model.decoder(z, t), target_y[tid][idx])
            )

            _, grad, _ = coordinate_gradient(
                model.coordinate_maps, x, tid, create_graph=True
            )
            G = G_dev[tid][idx]
            l1 = l1_dev[tid][idx]
            l2 = l2_dev[tid][idx]
            dloss, dstats = rayleigh_alignment_loss(grad, G, l1, l2)
            sloss, _ = submersion_loss(grad, min_grad=config.min_grad)
            closs, _ = corrected_physical_scale_loss(
                model.coordinate_maps, model.warps, x, tid
            )
            dir_terms.append(dloss)
            sub_terms.append(sloss)
            scale_terms.append(closs)
            align_terms.append(dstats["rayleigh_alignment"])

        rec = torch.stack(rec_terms).mean()
        direction = torch.stack(dir_terms).mean()
        sub = torch.stack(sub_terms).mean()
        scale = torch.stack(scale_terms).mean()
        left, right = model.coordinate_maps.coverage_intervals()
        regs = {
            "warp_roughness": model.warps.roughness_penalty(),
            "warp_magnitude": model.warps.magnitude_penalty(),
            "gauge_center": model.warps.consensus_gauge_penalty(left, right),
            "warp_scale_normalization": (
                model.warps.constant_scale_normalization_penalty(left, right)
            ),
            "coverage_prior": model.coordinate_maps.coverage_prior(
                w.coverage_prior_sd
            ),
        }
        rbf_roughness = model.coordinate_maps.roughness_penalty()
        loss = (
            float(config.reconstruction_weight) * rec
            + w.direction * direction
            + w.submersion * sub
            + w.corrected_scale * scale
            + w.warp_roughness * regs["warp_roughness"]
            + w.warp_magnitude * regs["warp_magnitude"]
            + w.gauge_center * regs["gauge_center"]
            + w.warp_scale_normalization * regs["warp_scale_normalization"]
            + w.coverage_prior * regs["coverage_prior"]
            + float(config.rbf_roughness_weight) * rbf_roughness
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (
            epoch % max(int(config.print_every), 1) == 0
            or epoch == int(config.epochs) - 1
        ):
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": float(loss.detach().cpu()),
                    "sqrt_reconstruction": float(rec.detach().cpu()),
                    "direction_loss": float(direction.detach().cpu()),
                    "submersion_loss": float(sub.detach().cpu()),
                    "rayleigh_alignment": float(
                        torch.stack(align_terms).mean().detach().cpu()
                    ),
                    "rbf_roughness": float(rbf_roughness.detach().cpu()),
                }
            )

    model.coordinate_maps.refresh_gauge_(
        [t.coords for t in atlas_dev.tissues],
        max_points=config.gauge_reference_points,
        lower_quantile=config.gauge_lower_quantile,
        upper_quantile=config.gauge_upper_quantile,
    )
    z_by_tissue = []
    with torch.no_grad():
        for tid, tissue in enumerate(atlas_dev.tissues):
            z_by_tissue.append(
                model.coordinate_maps.one(tissue.coords, tid)
                .detach()
                .cpu()
                .numpy()
            )
    result = ZooResult(
        method="rkhs_score_rayleigh",
        z_by_tissue=z_by_tissue,
        model=model.cpu(),
        auxiliary={"geometry": geometry},
        history=history,
    )
    result.check()
    return result
