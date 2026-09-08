"""Zoo rank 5: independent tissue coordinates + post-hoc monotone registration."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch
from .common import AtlasData, ZooResult, default_device
from .fisher import MeanFieldFitConfig, fisher_geometry_from_mean_fields, fit_local_mean_fields
from .registration import IndependentFitConfig, RegistrationConfig, fit_independent_coordinates, fit_posthoc_registration

@dataclass
class Config:
    mean_field: MeanFieldFitConfig = field(default_factory=MeanFieldFitConfig)
    independent: IndependentFitConfig = field(default_factory=IndependentFitConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    fisher_batch_size: int = 2048

def fit(atlas: AtlasData, config: Config = Config(), *, device: Optional[torch.device] = None) -> ZooResult:
    if device is None: device = default_device()
    mean_models, mean_history = fit_local_mean_fields(atlas, config.mean_field, device=device)
    geometry = fisher_geometry_from_mean_fields(atlas, mean_models, batch_size=config.fisher_batch_size, device=device)
    coord_models, raw, independent_history = fit_independent_coordinates(atlas, geometry, config.independent, device=device)
    registration, z, registration_history = fit_posthoc_registration(atlas, raw, geometry.mean_rates, config.registration, device=device)
    result = ZooResult(method="independent_coordinates_posthoc_warp", z_by_tissue=z, raw_coordinate_by_tissue=[x.numpy() for x in raw], model=registration, auxiliary={"coordinate_models": coord_models, "mean_field_models": [m.cpu() for m in mean_models], "geometry": geometry, "mean_field_history": mean_history, "independent_history": independent_history}, history=registration_history)
    result.check()
    return result
