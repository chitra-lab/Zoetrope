"""Zoo rank 2: smoothed mean field + Fisher-Rayleigh coordinate + tissue warps."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch
from .common import AtlasData, ZooResult, default_device
from .fisher import MeanFieldFitConfig, fisher_geometry_from_mean_fields, fit_local_mean_fields
from .trainers import SmoothedGeometryConfig, fit_smoothed_geometry_model

@dataclass
class Config:
    mean_field: MeanFieldFitConfig = field(default_factory=MeanFieldFitConfig)
    geometry_fit: SmoothedGeometryConfig = field(default_factory=SmoothedGeometryConfig)
    fisher_batch_size: int = 2048

def fit(atlas: AtlasData, config: Config = Config(), *, device: Optional[torch.device] = None) -> ZooResult:
    if device is None: device = default_device()
    mean_models, mean_history = fit_local_mean_fields(atlas, config.mean_field, device=device)
    geometry = fisher_geometry_from_mean_fields(atlas, mean_models, batch_size=config.fisher_batch_size, device=device)
    result = fit_smoothed_geometry_model(atlas, geometry, config.geometry_fit, device=device, method_name="meanfield_fisher_rayleigh_warp")
    result.auxiliary["mean_field_models"] = [m.cpu() for m in mean_models]
    result.auxiliary["mean_field_history"] = mean_history
    return result
