"""Zoo rank 7: principal-path physical arclength + monotone registration."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch
from .common import AtlasData, ZooResult, default_device, fit_coordinate_regressor
from .fisher import MeanFieldFitConfig, fisher_geometry_from_mean_fields, fit_local_mean_fields
from .graph_geometry import principal_graph_arclength
from .registration import RegistrationConfig, fit_posthoc_registration

@dataclass
class Config:
    mean_field: MeanFieldFitConfig = field(default_factory=MeanFieldFitConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    fisher_batch_size: int = 2048
    k_neighbors: int = 12
    alignment_penalty: float = 4.0
    regressor_epochs: int = 300

def fit(atlas: AtlasData, config: Config = Config(), *, device: Optional[torch.device] = None) -> ZooResult:
    if device is None: device = default_device()
    mean_models, mean_history = fit_local_mean_fields(atlas, config.mean_field, device=device)
    geometry = fisher_geometry_from_mean_fields(atlas, mean_models, batch_size=config.fisher_batch_size, device=device)
    raw, regressors = [], []
    for tid, tissue in enumerate(atlas.tissues):
        u_std = geometry.principal_lines[tid]
        u_raw = u_std * tissue.scale.reshape(1, 2)
        u_raw = u_raw / torch.clamp(torch.linalg.norm(u_raw, dim=1, keepdim=True), min=1e-8)
        r = principal_graph_arclength(tissue.raw_coords, u_raw, k_neighbors=config.k_neighbors, alignment_penalty=config.alignment_penalty)
        raw.append(r)
        regressors.append(fit_coordinate_regressor(tissue.coords, r, epochs=config.regressor_epochs, device=device).cpu())
    registration, z, registration_history = fit_posthoc_registration(atlas, raw, geometry.mean_rates, config.registration, device=device)
    result = ZooResult(method="principal_path_arclength_warp", z_by_tissue=z, raw_coordinate_by_tissue=[x.numpy() for x in raw], model=registration, auxiliary={"coordinate_regressors": regressors, "mean_field_models": [m.cpu() for m in mean_models], "geometry": geometry, "mean_field_history": mean_history}, history=registration_history)
    result.check(); return result
