"""Multiscale direct-score Fisher geometry followed by Rayleigh coordinate fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from .common import AtlasData, ZooResult, default_device
from .fisher import FisherGeometry, eigengap_weight, score_first_fisher_geometry
from .trainers import SmoothedGeometryConfig, fit_smoothed_geometry_model


@dataclass
class Config:
    k_neighbors: tuple[int, ...] = (8, 16, 24, 32)
    ridge: float = 1e-3
    pseudocount: float = 0.5
    stability_power: float = 2.0
    reconstruction_scale: int = 16
    geometry_fit: SmoothedGeometryConfig = field(default_factory=SmoothedGeometryConfig)


def _multiscale_geometry(
    atlas: AtlasData,
    *,
    k_neighbors: Sequence[int],
    ridge: float,
    pseudocount: float,
    stability_power: float,
    reconstruction_scale: int,
) -> FisherGeometry:
    scales = tuple(dict.fromkeys(int(k) for k in k_neighbors))
    if not scales:
        raise ValueError("k_neighbors must contain at least one scale")

    lines_by_scale: list[list[torch.Tensor]] = []
    gap_by_scale: list[list[torch.Tensor]] = []
    reconstruction_rates: Optional[list[torch.Tensor]] = None

    nearest_reconstruction_scale = min(
        scales, key=lambda k: abs(k - int(reconstruction_scale))
    )

    for k in scales:
        geom = score_first_fisher_geometry(
            atlas,
            k_neighbors=k,
            ridge=ridge,
            pseudocount=pseudocount,
        )
        lines_by_scale.append([u.detach().cpu() for u in geom.principal_lines])
        gap_by_scale.append(
            [
                eigengap_weight(l1, l2).detach().cpu()
                for l1, l2 in zip(geom.lambda1, geom.lambda2)
            ]
        )
        if k == nearest_reconstruction_scale:
            reconstruction_rates = [x.detach().cpu() for x in geom.mean_rates]
        del geom

    if reconstruction_rates is None:
        raise RuntimeError("failed to retain reconstruction rates")

    tensors: list[torch.Tensor] = []
    lambda1: list[torch.Tensor] = []
    lambda2: list[torch.Tensor] = []
    principal_lines: list[torch.Tensor] = []

    n_scales = len(scales)
    for tid in range(atlas.n_tissues):
        lines = torch.stack(
            [lines_by_scale[j][tid] for j in range(n_scales)], dim=0
        )  # [K,N,2]
        gaps = torch.stack(
            [gap_by_scale[j][tid] for j in range(n_scales)], dim=0
        )  # [K,N]

        # Sign-free cross-scale agreement. A perfectly stable line has score 1.
        dots2 = torch.einsum("knd,lnd->kln", lines, lines).square()
        stability = dots2.mean(dim=1)
        weights = gaps * torch.clamp(
            stability, min=0.0, max=1.0
        ).pow(float(stability_power))
        weights = weights + 1e-6

        projectors = torch.einsum("knd,kne->knde", lines, lines)
        G = (
            weights[..., None, None] * projectors
        ).sum(dim=0) / weights.sum(dim=0)[:, None, None]

        evals, evecs = torch.linalg.eigh(G)
        tensors.append(G)
        lambda2.append(evals[:, 0])
        lambda1.append(evals[:, 1])
        principal_lines.append(evecs[:, :, 1])

    return FisherGeometry(
        tensors=tensors,
        lambda1=lambda1,
        lambda2=lambda2,
        principal_lines=principal_lines,
        mean_rates=reconstruction_rates,
        metadata={
            "source": "multiscale_local_log_rate_scores",
            "k_neighbors": scales,
            "ridge": float(ridge),
            "pseudocount": float(pseudocount),
            "stability_power": float(stability_power),
            "reconstruction_scale": int(nearest_reconstruction_scale),
        },
    )


def fit(
    atlas: AtlasData,
    config: Config = Config(),
    *,
    device: Optional[torch.device] = None,
) -> ZooResult:
    if device is None:
        device = default_device()
    geometry = _multiscale_geometry(
        atlas,
        k_neighbors=config.k_neighbors,
        ridge=config.ridge,
        pseudocount=config.pseudocount,
        stability_power=config.stability_power,
        reconstruction_scale=config.reconstruction_scale,
    )
    return fit_smoothed_geometry_model(
        atlas,
        geometry,
        config.geometry_fit,
        device=device,
        method_name="multiscale_score_rayleigh",
    )
