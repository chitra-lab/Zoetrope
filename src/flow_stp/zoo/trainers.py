"""Reusable optimization loops for the estimator zoo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .common import AtlasData, CoordinateMaps, LowRankSliceEffect, SharedLogRateDecoder, SharedSqrtDecoder, ZooResult, default_device, evaluate_coordinate_maps, sample_indices, set_seed, sqrt_fisher_coordinate
from .fisher import FisherGeometry
from .objectives import coordinate_gradient, corrected_physical_scale_loss, rayleigh_alignment_loss, submersion_loss
from .warps import MonotoneWarpFamily


@dataclass
class GeometryWeights:
    direction: float = 1.0
    submersion: float = 0.2
    corrected_scale: float = 0.02
    warp_roughness: float = 0.01
    warp_magnitude: float = 0.001
    gauge_center: float = 0.1
    warp_scale_normalization: float = 0.05
    coverage_prior: float = 1e-4
    coverage_prior_sd: float = 0.08


@dataclass
class JointLikelihoodConfig:
    epochs: int = 1000
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    coord_hidden: tuple[int, ...] = (64, 64)
    decoder_hidden: tuple[int, ...] = (128, 128)
    warp_degree: int = 4
    slice_effect_rank: int = 2
    nuisance_weight: float = 5e-3
    min_grad: float = 0.03
    max_missing_each_end: float = 0.30
    initial_missing: float = 0.01
    weights: GeometryWeights = field(default_factory=GeometryWeights)
    seed: int = 20260908
    grad_clip: Optional[float] = 5.0
    print_every: int = 100


class JointLikelihoodWarpModel(nn.Module):
    def __init__(self, atlas: AtlasData, config: JointLikelihoodConfig) -> None:
        super().__init__()
        self.coordinate_maps = CoordinateMaps(atlas.n_tissues, hidden=config.coord_hidden, max_missing_each_end=config.max_missing_each_end, initial_missing=config.initial_missing)
        self.warps = MonotoneWarpFamily(atlas.n_tissues, degree=config.warp_degree)
        self.decoder = SharedLogRateDecoder(atlas.n_genes, hidden=config.decoder_hidden)
        self.slice_effect = LowRankSliceEffect(atlas.n_tissues, atlas.n_genes, rank=config.slice_effect_rank)

    def log_rate(self, coords, tissue_id: int, time: float):
        z = self.coordinate_maps.one(coords, tissue_id)
        t = torch.full_like(z, float(time))
        log_rate = self.decoder(z, t)
        mat = self.slice_effect.matrix()
        if mat is not None:
            log_rate = log_rate + mat[int(tissue_id)][None, :]
        return z, log_rate


def _warp_regularizers(maps: CoordinateMaps, warps: MonotoneWarpFamily, weights: GeometryWeights):
    left, right = maps.coverage_intervals()
    return {
        "warp_roughness": warps.roughness_penalty(),
        "warp_magnitude": warps.magnitude_penalty(),
        "gauge_center": warps.consensus_gauge_penalty(left, right),
        "warp_scale_normalization": warps.constant_scale_normalization_penalty(left, right),
        "coverage_prior": maps.coverage_prior(weights.coverage_prior_sd),
    }


def fit_joint_likelihood_model(atlas: AtlasData, geometry: FisherGeometry, config: JointLikelihoodConfig = JointLikelihoodConfig(), *, device: Optional[torch.device] = None, method_name: str = "joint_likelihood_fisher_warp") -> ZooResult:
    if device is None:
        device = default_device()
    set_seed(config.seed)
    atlas_dev = atlas.to(device)
    model = JointLikelihoodWarpModel(atlas_dev, config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay))
    history = []
    w = config.weights
    for epoch in range(int(config.epochs)):
        nll_terms, dir_terms, sub_terms, scale_terms, align_terms = [], [], [], [], []
        for tid, tissue in enumerate(atlas_dev.tissues):
            idx = sample_indices(tissue.n_cells, config.batch_size, device=device)
            x, counts, log_lib = tissue.coords[idx], tissue.counts[idx], tissue.log_library_size[idx]
            _, log_rate = model.log_rate(x, tid, tissue.time)
            nll_terms.append(F.poisson_nll_loss(log_rate + log_lib[:, None], counts, log_input=True, full=False, reduction="mean"))
            _, grad, _ = coordinate_gradient(model.coordinate_maps, x, tid, create_graph=True)
            G = geometry.tensors[tid].to(device)[idx]
            l1 = geometry.lambda1[tid].to(device)[idx]
            l2 = geometry.lambda2[tid].to(device)[idx]
            dloss, dstats = rayleigh_alignment_loss(grad, G, l1, l2)
            sloss, _ = submersion_loss(grad, min_grad=config.min_grad)
            closs, _ = corrected_physical_scale_loss(model.coordinate_maps, model.warps, x, tid)
            dir_terms.append(dloss); sub_terms.append(sloss); scale_terms.append(closs); align_terms.append(dstats["rayleigh_alignment"])
        nll = torch.stack(nll_terms).mean(); direction = torch.stack(dir_terms).mean(); sub = torch.stack(sub_terms).mean(); scale = torch.stack(scale_terms).mean()
        regs = _warp_regularizers(model.coordinate_maps, model.warps, w)
        nuisance = model.slice_effect.penalty()
        loss = nll + w.direction * direction + w.submersion * sub + w.corrected_scale * scale + w.warp_roughness * regs["warp_roughness"] + w.warp_magnitude * regs["warp_magnitude"] + w.gauge_center * regs["gauge_center"] + w.warp_scale_normalization * regs["warp_scale_normalization"] + w.coverage_prior * regs["coverage_prior"] + float(config.nuisance_weight) * nuisance
        opt.zero_grad(set_to_none=True); loss.backward()
        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip))
        opt.step()
        if epoch % max(int(config.print_every), 1) == 0 or epoch == int(config.epochs) - 1:
            history.append({"epoch": float(epoch), "loss": float(loss.detach().cpu()), "poisson_nll": float(nll.detach().cpu()), "direction_loss": float(direction.detach().cpu()), "submersion_loss": float(sub.detach().cpu()), "corrected_scale_loss": float(scale.detach().cpu()), "rayleigh_alignment": float(torch.stack(align_terms).mean().detach().cpu()), "gauge_center": float(regs["gauge_center"].detach().cpu())})
    z_by_tissue = evaluate_coordinate_maps(model.coordinate_maps, atlas_dev, device)
    result = ZooResult(method=method_name, z_by_tissue=z_by_tissue, model=model.cpu(), auxiliary={"geometry": geometry}, history=history)
    result.check()
    return result


@dataclass
class SmoothedGeometryConfig:
    epochs: int = 900
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    coord_hidden: tuple[int, ...] = (64, 64)
    decoder_hidden: tuple[int, ...] = (128, 128)
    warp_degree: int = 4
    min_grad: float = 0.03
    max_missing_each_end: float = 0.30
    initial_missing: float = 0.01
    reconstruction_weight: float = 1.0
    weights: GeometryWeights = field(default_factory=GeometryWeights)
    seed: int = 20260908
    grad_clip: Optional[float] = 5.0
    print_every: int = 100


class SmoothedGeometryModel(nn.Module):
    def __init__(self, atlas: AtlasData, config: SmoothedGeometryConfig) -> None:
        super().__init__()
        self.coordinate_maps = CoordinateMaps(atlas.n_tissues, hidden=config.coord_hidden, max_missing_each_end=config.max_missing_each_end, initial_missing=config.initial_missing)
        self.warps = MonotoneWarpFamily(atlas.n_tissues, degree=config.warp_degree)
        self.decoder = SharedSqrtDecoder(atlas.n_genes, hidden=config.decoder_hidden)


def fit_smoothed_geometry_model(atlas: AtlasData, geometry: FisherGeometry, config: SmoothedGeometryConfig = SmoothedGeometryConfig(), *, device: Optional[torch.device] = None, method_name: str) -> ZooResult:
    if device is None:
        device = default_device()
    set_seed(config.seed)
    atlas_dev = atlas.to(device)
    model = SmoothedGeometryModel(atlas_dev, config).to(device)
    target_y = [sqrt_fisher_coordinate(x).to(device) for x in geometry.mean_rates]
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay))
    history = []
    w = config.weights
    for epoch in range(int(config.epochs)):
        rec_terms, dir_terms, sub_terms, scale_terms, align_terms = [], [], [], [], []
        for tid, tissue in enumerate(atlas_dev.tissues):
            idx = sample_indices(tissue.n_cells, config.batch_size, device=device)
            x = tissue.coords[idx]
            z = model.coordinate_maps.one(x, tid)
            t = torch.full_like(z, float(tissue.time))
            rec_terms.append(F.mse_loss(model.decoder(z, t), target_y[tid][idx]))
            _, grad, _ = coordinate_gradient(model.coordinate_maps, x, tid, create_graph=True)
            G = geometry.tensors[tid].to(device)[idx]
            l1 = geometry.lambda1[tid].to(device)[idx]
            l2 = geometry.lambda2[tid].to(device)[idx]
            dloss, dstats = rayleigh_alignment_loss(grad, G, l1, l2)
            sloss, _ = submersion_loss(grad, min_grad=config.min_grad)
            closs, _ = corrected_physical_scale_loss(model.coordinate_maps, model.warps, x, tid)
            dir_terms.append(dloss); sub_terms.append(sloss); scale_terms.append(closs); align_terms.append(dstats["rayleigh_alignment"])
        rec = torch.stack(rec_terms).mean(); direction = torch.stack(dir_terms).mean(); sub = torch.stack(sub_terms).mean(); scale = torch.stack(scale_terms).mean()
        regs = _warp_regularizers(model.coordinate_maps, model.warps, w)
        loss = float(config.reconstruction_weight) * rec + w.direction * direction + w.submersion * sub + w.corrected_scale * scale + w.warp_roughness * regs["warp_roughness"] + w.warp_magnitude * regs["warp_magnitude"] + w.gauge_center * regs["gauge_center"] + w.warp_scale_normalization * regs["warp_scale_normalization"] + w.coverage_prior * regs["coverage_prior"]
        opt.zero_grad(set_to_none=True); loss.backward()
        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip))
        opt.step()
        if epoch % max(int(config.print_every), 1) == 0 or epoch == int(config.epochs) - 1:
            history.append({"epoch": float(epoch), "loss": float(loss.detach().cpu()), "sqrt_reconstruction": float(rec.detach().cpu()), "direction_loss": float(direction.detach().cpu()), "submersion_loss": float(sub.detach().cpu()), "corrected_scale_loss": float(scale.detach().cpu()), "rayleigh_alignment": float(torch.stack(align_terms).mean().detach().cpu()), "gauge_center": float(regs["gauge_center"].detach().cpu())})
    z_by_tissue = evaluate_coordinate_maps(model.coordinate_maps, atlas_dev, device)
    result = ZooResult(method=method_name, z_by_tissue=z_by_tissue, model=model.cpu(), auxiliary={"geometry": geometry}, history=history)
    result.check()
    return result
