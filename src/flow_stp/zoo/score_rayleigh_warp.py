"""Direct spatial-score Fisher geometry + Rayleigh coordinate + tissue warps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .common import AtlasData, ZooResult, default_device
from .fisher import score_first_fisher_geometry
from .trainers import SmoothedGeometryConfig, fit_smoothed_geometry_model


@dataclass
class Config:
    # k=16 was the most coherent direct-score setting in the first Qian PFC
    # benchmark and is therefore the reference default for new runs.
    k_neighbors: int = 16
    ridge: float = 1e-3
    pseudocount: float = 0.5
    geometry_fit: SmoothedGeometryConfig = field(
        default_factory=SmoothedGeometryConfig
    )


def fit(
    atlas: AtlasData,
    config: Config = Config(),
    *,
    device: Optional[torch.device] = None,
) -> ZooResult:
    if device is None:
        device = default_device()
    geometry = score_first_fisher_geometry(
        atlas,
        k_neighbors=config.k_neighbors,
        ridge=config.ridge,
        pseudocount=config.pseudocount,
    )
    return fit_smoothed_geometry_model(
        atlas,
        geometry,
        config.geometry_fit,
        device=device,
        method_name="score_fisher_rayleigh_warp",
    )
