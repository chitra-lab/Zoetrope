"""Zoo rank 6: mean-field Fisher tensor + explicit eigenfield integration + warp."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch
from .common import AtlasData, ZooResult, default_device, fit_coordinate_regressor
from .fisher import MeanFieldFitConfig, fisher_geometry_from_mean_fields, fit_local_mean_fields
from .graph_geometry import integrate_eigenfield_nodes
from .registration import RegistrationConfig, fit_posthoc_registration

@dataclass
class Config:
    mean_field: MeanFieldFitConfig = field(default_factory=MeanFieldFitConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    fisher_batch_size: int = 2048
    k_neighbors: int = 10
    integration_epochs: int = 800
    integration_learning_rate: float = 5e-2
    integration_alignment_power: float = 2.0
    regressor_epochs: int = 300
    seed: int = 20260908

def fit(atlas: AtlasData, config: Config = Config(), *, device: Optional[torch.device] = None) -> ZooResult:
    if device is None: device = default_device()
    mean_models, mean_history = fit_local_mean_fields(atlas, config.mean_field, device=device)
    geometry = fisher_geometry_from_mean_fields(atlas, mean_models, batch_size=config.fisher_batch_size, device=device)
    raw, regressors = [], []
    for tid, tissue in enumerate(atlas.tissues):
        r = integrate_eigenfield_nodes(tissue.coords, geometry.principal_lines[tid], k_neighbors=config.k_neighbors, epochs=config.integration_epochs, learning_rate=config.integration_learning_rate, alignment_power=config.integration_alignment_power, seed=config.seed + tid)
        raw.append(r)
        regressors.append(fit_coordinate_regressor(tissue.coords, r, epochs=config.regressor_epochs, device=device).cpu())
    registration, z, registration_history = fit_posthoc_registration(atlas, raw, geometry.mean_rates, config.registration, device=device)
    result = ZooResult(method="meanfield_eigenfield_integration_warp", z_by_tissue=z, raw_coordinate_by_tissue=[x.numpy() for x in raw], model=registration, auxiliary={"coordinate_regressors": regressors, "mean_field_models": [m.cpu() for m in mean_models], "geometry": geometry, "mean_field_history": mean_history}, history=registration_history)
    result.check(); return result
