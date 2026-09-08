"""Zoo rank 1: joint count likelihood + Fisher direction + tissue warps."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch
from .common import AtlasData, ZooResult, default_device
from .fisher import MeanFieldFitConfig, fisher_geometry_from_mean_fields, fit_local_mean_fields
from .trainers import JointLikelihoodConfig, fit_joint_likelihood_model

@dataclass
class Config:
    mean_field: MeanFieldFitConfig = field(default_factory=MeanFieldFitConfig)
    joint: JointLikelihoodConfig = field(default_factory=JointLikelihoodConfig)
    fisher_batch_size: int = 2048

def fit(atlas: AtlasData, config: Config = Config(), *, device: Optional[torch.device] = None) -> ZooResult:
    if device is None: device = default_device()
    mean_models, mean_history = fit_local_mean_fields(atlas, config.mean_field, device=device)
    geometry = fisher_geometry_from_mean_fields(atlas, mean_models, batch_size=config.fisher_batch_size, device=device)
    result = fit_joint_likelihood_model(atlas, geometry, config.joint, device=device, method_name="joint_likelihood_fisher_warp")
    result.auxiliary["mean_field_models"] = [m.cpu() for m in mean_models]
    result.auxiliary["mean_field_history"] = mean_history
    return result
