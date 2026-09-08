"""Independent coordinate fitting and post-hoc one-dimensional registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .common import (
    AtlasData,
    MLP,
    SharedSqrtDecoder,
    Tensor,
    default_device,
    sample_indices,
    set_seed,
    sqrt_fisher_coordinate,
)
from .fisher import FisherGeometry
from .objectives import rayleigh_alignment_loss, submersion_loss
from .warps import MonotoneWarpFamily


@dataclass
class IndependentFitConfig:
    epochs: int = 700
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden: tuple[int, ...] = (64, 64)
    direction_weight: float = 1.0
    submersion_weight: float = 0.2
    scale_weight: float = 0.02
    range_weight: float = 0.1
    min_grad: float = 0.03
    target_std: float = 0.20
    seed: int = 20260908


class IndependentScalarMap(nn.Module):
    def __init__(self, hidden: Sequence[int]) -> None:
        super().__init__()
        self.net = MLP(2, 1, hidden=hidden, activation="tanh")

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.net(x).squeeze(-1))


def _scale_variation_loss(
    model: nn.Module,
    x: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    x_req = x.detach().clone().requires_grad_(True)
    z = model(x_req)
    grad = torch.autograd.grad(
        z.sum(),
        x_req,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    speed = torch.sqrt(grad.square().sum(dim=1) + eps)
    log_speed = torch.log(speed + eps)
    grad_log = torch.autograd.grad(
        log_speed.sum(),
        x_req,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    unit = grad / speed[:, None]
    return ((grad_log * unit).sum(dim=1)).square().mean()


def fit_independent_coordinates(
    atlas: AtlasData,
    geometry: FisherGeometry,
    config: IndependentFitConfig = IndependentFitConfig(),
    *,
    device: Optional[torch.device] = None,
):
    if device is None:
        device = default_device()
    set_seed(config.seed)
    models, raw, history = [], [], []

    for tid, tissue in enumerate(atlas.tissues):
        x_all = tissue.coords.to(device)
        model = IndependentScalarMap(config.hidden).to(device)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        last_stats = {}

        for epoch in range(int(config.epochs)):
            idx = sample_indices(
                len(x_all),
                config.batch_size,
                device=device,
            )
            x = x_all[idx].detach().clone().requires_grad_(True)
            z = model(x)
            grad = torch.autograd.grad(
                z.sum(),
                x,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            G = geometry.tensors[tid].to(device)[idx]
            l1 = geometry.lambda1[tid].to(device)[idx]
            l2 = geometry.lambda2[tid].to(device)[idx]
            dloss, dstats = rayleigh_alignment_loss(grad, G, l1, l2)
            sloss, _ = submersion_loss(grad, min_grad=config.min_grad)
            vloss = _scale_variation_loss(model, x_all[idx])
            range_loss = F.relu(
                float(config.target_std) - z.std(unbiased=False)
            ).square()
            loss = (
                float(config.direction_weight) * dloss
                + float(config.submersion_weight) * sloss
                + float(config.scale_weight) * vloss
                + float(config.range_weight) * range_loss
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_stats = {
                "tissue_id": float(tid),
                "epoch": float(epoch),
                "loss": float(loss.detach().cpu()),
                "direction_loss": float(dloss.detach().cpu()),
                "rayleigh_alignment": float(
                    dstats["rayleigh_alignment"].cpu()
                ),
            }

        with torch.no_grad():
            z_all = model(x_all).detach().cpu()
        models.append(model.cpu())
        raw.append(z_all)
        history.append(last_stats)

    return models, raw, history


class CoverageParameters(nn.Module):
    def __init__(
        self,
        n_tissues: int,
        max_missing: float = 0.30,
        initial_missing: float = 0.01,
    ):
        super().__init__()
        self.n_tissues = int(n_tissues)
        self.max_missing = float(max_missing)
        frac = np.clip(
            float(initial_missing) / max(float(max_missing), 1e-8),
            1e-5,
            1 - 1e-5,
        )
        init = float(math.log(frac / (1.0 - frac)))
        self.left_logits = nn.Parameter(
            torch.full((self.n_tissues,), init)
        )
        self.right_logits = nn.Parameter(
            torch.full((self.n_tissues,), init)
        )

    def intervals(self):
        left = self.max_missing * torch.sigmoid(self.left_logits)
        right = 1.0 - self.max_missing * torch.sigmoid(self.right_logits)
        return left, right

    def prior(self, sd: float = 0.08):
        left, right = self.intervals()
        miss = torch.cat([left, 1.0 - right])
        return 0.5 * (miss / max(float(sd), 1e-8)).square().mean()


@dataclass
class RegistrationConfig:
    epochs_soft_orientation: int = 500
    epochs_hard_refine: int = 400
    batch_size: int = 2048
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    decoder_hidden: tuple[int, ...] = (128, 128)
    warp_degree: int = 4
    max_missing_each_end: float = 0.30
    initial_missing: float = 0.01
    reconstruction_weight: float = 1.0
    warp_roughness_weight: float = 0.01
    warp_magnitude_weight: float = 0.001
    gauge_weight: float = 0.1
    coverage_weight: float = 1e-4
    orientation_binary_weight: float = 0.05
    seed: int = 20260908


class PosthocRegistrationModel(nn.Module):
    """Register local tissue coordinates into the common process coordinate.

    The learned monotone map ``k_i`` sends a local tissue coordinate to the
    common coordinate.  In the geometric notation used in the manuscript,
    ``k_i = h_i^{-1}``, where ``h_i`` is the forward anatomical distortion.
    """

    def __init__(self, atlas: AtlasData, config: RegistrationConfig) -> None:
        super().__init__()
        self.n_tissues = atlas.n_tissues
        self.warps = MonotoneWarpFamily(
            atlas.n_tissues,
            degree=config.warp_degree,
        )
        self.coverage = CoverageParameters(
            atlas.n_tissues,
            max_missing=config.max_missing_each_end,
            initial_missing=config.initial_missing,
        )
        self.decoder = SharedSqrtDecoder(
            atlas.n_genes,
            hidden=config.decoder_hidden,
        )
        self.orientation_logits = nn.Parameter(
            torch.full((max(atlas.n_tissues - 1, 0),), 2.0)
        )

    def orientation_strength(self, tissue_id: int, *, hard: bool):
        if int(tissue_id) == 0:
            return torch.ones((), device=self.orientation_logits.device)
        value = torch.tanh(self.orientation_logits[int(tissue_id) - 1])
        if hard:
            return torch.where(
                value >= 0,
                torch.ones_like(value),
                -torch.ones_like(value),
            )
        return value

    def orient(self, raw: Tensor, tissue_id: int, *, hard: bool):
        s = self.orientation_strength(tissue_id, hard=hard)
        return torch.clamp(0.5 + s * (raw - 0.5), 0.0, 1.0)

    def z(self, raw: Tensor, tissue_id: int, *, hard: bool = False):
        oriented = self.orient(raw, tissue_id, hard=hard)
        common_local = self.warps.normalized_one(oriented, tissue_id)
        left, right = self.coverage.intervals()
        return (
            left[tissue_id]
            + (right[tissue_id] - left[tissue_id]) * common_local
        )

    def orientation_penalty(self):
        if self.orientation_logits.numel() == 0:
            return torch.zeros((), device=self.warps.coefficients.device)
        s = torch.tanh(self.orientation_logits)
        return (1.0 - s.square()).mean()


def fit_posthoc_registration(
    atlas: AtlasData,
    raw_by_tissue: Sequence[Tensor],
    mean_rates: Sequence[Tensor],
    config: RegistrationConfig = RegistrationConfig(),
    *,
    device: Optional[torch.device] = None,
):
    if (
        len(raw_by_tissue) != atlas.n_tissues
        or len(mean_rates) != atlas.n_tissues
    ):
        raise ValueError(
            "raw coordinates and mean rates must have one entry per tissue"
        )
    if device is None:
        device = default_device()
    set_seed(config.seed)

    model = PosthocRegistrationModel(atlas, config).to(device)
    raw = [x.to(device).reshape(-1) for x in raw_by_tissue]
    targets = [sqrt_fisher_coordinate(x).to(device) for x in mean_rates]
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    history = []

    def train_phase(n_epochs: int, hard: bool, phase: str):
        for epoch in range(int(n_epochs)):
            rec_terms = []
            for tid, tissue in enumerate(atlas.tissues):
                idx = sample_indices(
                    len(raw[tid]),
                    config.batch_size,
                    device=device,
                )
                z = model.z(raw[tid][idx], tid, hard=hard)
                t = torch.full_like(z, float(tissue.time))
                rec_terms.append(
                    F.mse_loss(
                        model.decoder(z, t),
                        targets[tid][idx],
                    )
                )

            rec = torch.stack(rec_terms).mean()
            left, right = model.coverage.intervals()
            rough = model.warps.roughness_penalty()
            mag = model.warps.magnitude_penalty()

            # The post-hoc warp is local->common, i.e. the inverse of the
            # forward anatomical distortion.  Gauge centering must therefore
            # be applied after converting its derivative back to h_i'.
            gauge = model.warps.inverse_consensus_gauge_penalty(
                left,
                right,
            )
            cov = model.coverage.prior()
            orient = (
                model.orientation_penalty()
                if not hard
                else torch.zeros_like(rec)
            )

            loss = (
                float(config.reconstruction_weight) * rec
                + float(config.warp_roughness_weight) * rough
                + float(config.warp_magnitude_weight) * mag
                + float(config.gauge_weight) * gauge
                + float(config.coverage_weight) * cov
                + float(config.orientation_binary_weight) * orient
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if epoch == int(n_epochs) - 1:
                history.append(
                    {
                        "phase": 0.0 if phase == "soft" else 1.0,
                        "epoch": float(epoch),
                        "loss": float(loss.detach().cpu()),
                        "sqrt_reconstruction": float(rec.detach().cpu()),
                        "gauge_center": float(gauge.detach().cpu()),
                    }
                )

    train_phase(
        config.epochs_soft_orientation,
        hard=False,
        phase="soft",
    )
    train_phase(
        config.epochs_hard_refine,
        hard=True,
        phase="hard",
    )

    out = []
    model.eval()
    with torch.no_grad():
        for tid in range(atlas.n_tissues):
            out.append(model.z(raw[tid], tid, hard=True).cpu().numpy())
    return model.cpu(), out, history
