"""Score-geometry anchored joint coordinate + low-rank expression model."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import math
import torch
from torch import nn
import torch.nn.functional as F

from .common import (
    AtlasData,
    ZooResult,
    default_device,
    sample_indices,
    set_seed,
    sqrt_fisher_coordinate,
)
from .coordinate_maps import ReferenceNormalizedCoordinateMaps
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


def _bernstein_basis(x: Tensor, degree: int) -> Tensor:
    x = torch.clamp(x.reshape(-1), 0.0, 1.0)
    terms = []
    for j in range(int(degree) + 1):
        coeff = float(math.comb(int(degree), j))
        terms.append(coeff * x.pow(j) * (1.0 - x).pow(int(degree) - j))
    return torch.stack(terms, dim=1)


class BernsteinLatentDecoder(nn.Module):
    """Smooth tensor-product polynomial surface a(z,t) in low rank."""

    def __init__(
        self,
        latent_rank: int,
        degree_z: int = 5,
        degree_t: int = 3,
    ) -> None:
        super().__init__()
        self.latent_rank = int(latent_rank)
        self.degree_z = int(degree_z)
        self.degree_t = int(degree_t)
        n_basis = (self.degree_z + 1) * (self.degree_t + 1)
        self.coefficients = nn.Parameter(
            0.01 * torch.randn(n_basis, self.latent_rank)
        )

    def forward(self, z: Tensor, time: Tensor) -> Tensor:
        bz = _bernstein_basis(z, self.degree_z)
        bt = _bernstein_basis(time, self.degree_t)
        features = torch.einsum("bi,bj->bij", bz, bt).reshape(len(z), -1)
        return features @ self.coefficients

    def roughness_penalty(self) -> Tensor:
        c = self.coefficients.reshape(
            self.degree_z + 1,
            self.degree_t + 1,
            self.latent_rank,
        )
        terms = []
        if self.degree_z >= 2:
            terms.append(
                (c[2:] - 2.0 * c[1:-1] + c[:-2]).square().mean()
            )
        if self.degree_t >= 2:
            terms.append(
                (c[:, 2:] - 2.0 * c[:, 1:-1] + c[:, :-2]).square().mean()
            )
        if not terms:
            return c.square().mean() * 0.0
        return torch.stack(terms).mean()


class Model(nn.Module):
    def __init__(self, atlas: AtlasData, config: "Config") -> None:
        super().__init__()
        self.coordinate_maps = ReferenceNormalizedCoordinateMaps(
            atlas.n_tissues,
            hidden=config.coord_hidden,
            max_missing_each_end=config.max_missing_each_end,
            initial_missing=config.initial_missing,
        )
        self.warps = MonotoneWarpFamily(
            atlas.n_tissues, degree=config.warp_degree
        )
        self.latent_decoder = BernsteinLatentDecoder(
            config.latent_rank,
            degree_z=config.degree_z,
            degree_t=config.degree_t,
        )


@dataclass
class Config:
    k_neighbors: int = 16
    ridge: float = 1e-3
    pseudocount: float = 0.5
    latent_rank: int = 12
    basis_cells_per_tissue: int = 4096
    degree_z: int = 5
    degree_t: int = 3
    epochs: int = 1200
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    coord_hidden: tuple[int, ...] = (64, 64)
    warp_degree: int = 4
    min_grad: float = 0.03
    max_missing_each_end: float = 0.30
    initial_missing: float = 0.01
    gauge_reference_points: int = 4096
    gauge_lower_quantile: float = 0.005
    gauge_upper_quantile: float = 0.995
    geometry_anchor_weight: float = 3.0
    reconstruction_weight: float = 0.5
    latent_roughness_weight: float = 0.02
    weights: GeometryWeights = field(default_factory=GeometryWeights)
    seed: int = 20260908
    print_every: int = 100


def _fit_fisher_flat_basis(
    geometry,
    *,
    rank: int,
    cells_per_tissue: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    samples = []
    for rate in geometry.mean_rates:
        y = sqrt_fisher_coordinate(rate.detach().cpu())
        if cells_per_tissue > 0 and len(y) > int(cells_per_tissue):
            idx = torch.randperm(len(y), generator=gen)[: int(cells_per_tissue)]
            y = y[idx]
        samples.append(y)
    Y = torch.cat(samples, dim=0)
    mean = Y.mean(dim=0)
    centered = Y - mean
    q = min(int(rank), centered.shape[1], max(centered.shape[0] - 1, 1))
    _, _, V = torch.pca_lowrank(centered, q=q, center=False)
    return mean, V[:, :q]


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
    mean_y, basis = _fit_fisher_flat_basis(
        geometry,
        rank=config.latent_rank,
        cells_per_tissue=config.basis_cells_per_tissue,
        seed=config.seed,
    )
    latent_rank = basis.shape[1]
    if latent_rank != config.latent_rank:
        config = replace(config, latent_rank=latent_rank)

    atlas_dev = atlas.to(device)
    model = Model(atlas_dev, config).to(device)
    latent_targets = [
        ((sqrt_fisher_coordinate(rate) - mean_y) @ basis).to(device)
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

            target_a = latent_targets[tid][idx]
            pred_a = model.latent_decoder(z, t)
            rec_terms.append(F.mse_loss(pred_a, target_a))

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
        warp_rough = model.warps.roughness_penalty()
        warp_mag = model.warps.magnitude_penalty()
        gauge = model.warps.consensus_gauge_penalty(left, right)
        warp_scale = model.warps.constant_scale_normalization_penalty(left, right)
        coverage = model.coordinate_maps.coverage_prior(w.coverage_prior_sd)
        latent_rough = model.latent_decoder.roughness_penalty()

        loss = (
            float(config.reconstruction_weight) * rec
            + float(config.geometry_anchor_weight) * direction
            + w.submersion * sub
            + w.corrected_scale * scale
            + w.warp_roughness * warp_rough
            + w.warp_magnitude * warp_mag
            + w.gauge_center * gauge
            + w.warp_scale_normalization * warp_scale
            + w.coverage_prior * coverage
            + float(config.latent_roughness_weight) * latent_rough
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
                    "latent_reconstruction": float(rec.detach().cpu()),
                    "direction_loss": float(direction.detach().cpu()),
                    "submersion_loss": float(sub.detach().cpu()),
                    "rayleigh_alignment": float(
                        torch.stack(align_terms).mean().detach().cpu()
                    ),
                    "latent_roughness": float(latent_rough.detach().cpu()),
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
        method="geometry_anchored_lowrank",
        z_by_tissue=z_by_tissue,
        model=model.cpu(),
        auxiliary={
            "geometry": geometry,
            "fisher_flat_mean": mean_y,
            "fisher_flat_basis": basis,
        },
        history=history,
    )
    result.check()
    return result
