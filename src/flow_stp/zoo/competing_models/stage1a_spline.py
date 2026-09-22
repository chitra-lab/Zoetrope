"""Stage1a neural-coordinate / spline-expression competitors.

These variants preserve the Stage1a tissue-specific neural coordinate maps,
coverage model, and geometry penalties while replacing the neural expression
decoder with the low-complexity tensor-product spline expression field used by
the current Fisher-Lagrangian implementation.

Two fitting strategies are provided:

* fit_stage1a_spline_joint jointly optimizes alpha and E with AdamW.
* fit_stage1a_spline_profiled alternates conditional Poisson IRLS fits of E
  with AdamW updates of alpha.

The scientific estimands are unchanged. These are algorithmic competitors for
recovering the same common coordinate and expression state map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ...fisher_lagrangian.model import SplineLogMeanExpressionField
from ...fisher_lagrangian.training import (
    fit_expression_irls,
    initialize_expression_from_global_means,
)
from ...stage1.stage1a import FlowSTPConfig, SliceCoordinateMaps
from ..common import AtlasData, ZooResult, default_device, set_seed

Tensor = torch.Tensor


@dataclass(frozen=True)
class Stage1aSplineModelConfig:
    """Shared statistical model for the two Stage1a spline variants."""

    coord_hidden: tuple[int, ...] = (64, 64)
    spatial_basis_dim: int = 10
    temporal_basis_dim: int = 3
    spline_degree: int = 3
    coordinate_domain: tuple[float, float] = (-0.25, 1.25)
    time_domain: tuple[float, float] = (0.0, 1.0)
    reference_time: float = 0.0

    slice_effect_dim: int = 0
    coord_grad_min: float = 0.05
    coord_grad_eps: float = 1e-12
    scale_denominator_eps: float = 1e-8
    coverage_prior_sd: float = 0.05
    max_missing_each_end: float = 0.25
    initial_missing: float = 0.005


@dataclass(frozen=True)
class Stage1aSplineJointConfig:
    model: Stage1aSplineModelConfig = field(
        default_factory=Stage1aSplineModelConfig
    )
    epochs: int = 1000
    batch_size_per_tissue: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float | None = 5.0

    submersion_weight: float = 0.11364637
    scale_variation_weight: float = 1e-3
    coverage_prior_weight: float = 1e-4
    nuisance_weight: float = 5e-3

    geometry_cells_per_tissue: int = 128
    gauge_lower_quantile: float = 0.005
    gauge_upper_quantile: float = 0.995
    seed: int = 20260922
    print_every: int = 100


@dataclass(frozen=True)
class Stage1aSplineProfiledConfig:
    model: Stage1aSplineModelConfig = field(
        default_factory=Stage1aSplineModelConfig
    )
    outer_iterations: int = 8
    coordinate_steps_per_outer: int = 100
    batch_size_per_tissue: int = 1024
    coordinate_learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float | None = 5.0

    submersion_weight: float = 0.11364637
    scale_variation_weight: float = 1e-3
    coverage_prior_weight: float = 1e-4

    geometry_cells_per_tissue: int = 128
    gauge_lower_quantile: float = 0.005
    gauge_upper_quantile: float = 0.995
    gauge_refresh_every: int = 10

    irls_max_iter: int = 25
    irls_tol: float = 1e-6
    irls_ridge: float = 1e-6
    irls_gene_chunk_size: int = 32

    seed: int = 20260922


class Stage1aSplineModel(nn.Module):
    """Stage1a neural coordinate maps with a shared spline log-mean field."""

    def __init__(
        self,
        *,
        num_tissues: int,
        num_genes: int,
        config: Stage1aSplineModelConfig,
    ) -> None:
        super().__init__()
        if num_tissues <= 0 or num_genes <= 0:
            raise ValueError("num_tissues and num_genes must be positive")

        self.model_config = config
        coord_config = FlowSTPConfig(
            num_slices=int(num_tissues),
            num_genes=int(num_genes),
            coord_hidden=tuple(config.coord_hidden),
            slice_effect_dim=int(config.slice_effect_dim),
            coord_grad_min=float(config.coord_grad_min),
            coord_grad_eps=float(config.coord_grad_eps),
            scale_denominator_eps=float(config.scale_denominator_eps),
            coverage_prior_sd=float(config.coverage_prior_sd),
            max_missing_each_end=float(config.max_missing_each_end),
            initial_missing=float(config.initial_missing),
        )
        self.coordinate_maps = SliceCoordinateMaps(coord_config)
        self.coord_config = coord_config

        self.expression_field = SplineLogMeanExpressionField(
            num_genes=int(num_genes),
            spatial_basis_dim=int(config.spatial_basis_dim),
            temporal_basis_dim=int(config.temporal_basis_dim),
            spline_degree=int(config.spline_degree),
            coordinate_domain=tuple(config.coordinate_domain),
            time_domain=tuple(config.time_domain),
            reference_time=float(config.reference_time),
        )

        rank = int(config.slice_effect_dim)
        if rank > 0:
            self.slice_embedding = nn.Embedding(int(num_tissues), rank)
            nn.init.zeros_(self.slice_embedding.weight)
            self.nuisance_loadings = nn.Parameter(
                0.01 * torch.randn(int(num_genes), rank)
            )
        else:
            self.slice_embedding = None
            self.nuisance_loadings = None

    @property
    def coord_maps(self) -> SliceCoordinateMaps:
        return self.coordinate_maps

    def nuisance_effect_matrix(self) -> Tensor | None:
        if self.slice_embedding is None or self.nuisance_loadings is None:
            return None
        effect = self.slice_embedding.weight @ self.nuisance_loadings.T
        return effect - effect.mean(dim=0, keepdim=True)

    def nuisance_penalty(self) -> Tensor:
        effect = self.nuisance_effect_matrix()
        if effect is None:
            return self.expression_field.baseline_coeff.new_zeros(())
        return effect.square().mean()

    def forward(
        self,
        *,
        coords: Tensor,
        slice_idx: Tensor,
        time: Tensor,
        log_library_size: Tensor | None = None,
    ) -> dict[str, Tensor]:
        slice_idx = slice_idx.long().reshape(-1)
        time = time.reshape(-1).to(device=coords.device, dtype=coords.dtype)
        z = self.coordinate_maps(coords, slice_idx).reshape(-1)
        log_rate = self.expression_field(z, time)

        log_mu = log_rate
        if log_library_size is not None:
            log_mu = log_mu + log_library_size.reshape(-1, 1).to(
                device=log_mu.device,
                dtype=log_mu.dtype,
            )

        nuisance = self.nuisance_effect_matrix()
        if nuisance is not None:
            log_mu = log_mu + nuisance[slice_idx]

        return {
            "z": z[:, None],
            "log_rate": log_rate,
            "log_mu": log_mu,
        }

    def coordinate_geometry_penalties(
        self,
        coords_by_tissue: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Use the selected Stage1a submersion and relative-scale penalties."""
        sub_terms: list[Tensor] = []
        scale_terms: list[Tensor] = []

        for tissue_id, coords in enumerate(coords_by_tissue):
            x = coords.detach().clone().requires_grad_(True)
            z_local, z_global = self.coordinate_maps.local_and_global_one_slice(
                x,
                tissue_id,
            )

            grad_local = torch.autograd.grad(
                z_local.sum(),
                x,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            local_norm = torch.sqrt(
                grad_local.square().sum(dim=1)
                + float(self.coord_config.coord_grad_eps)
            )
            threshold = max(float(self.coord_config.coord_grad_min), 1e-12)
            deficit = F.relu((threshold - local_norm) / threshold)
            sub_terms.append(deficit.square().mean())

            grad_global = torch.autograd.grad(
                z_global.sum(),
                x,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grad_sq = grad_global.square().sum(dim=1)
            grad_norm = torch.sqrt(
                grad_sq + float(self.coord_config.coord_grad_eps)
            )
            hessian_times_grad = torch.autograd.grad(
                0.5 * grad_sq.sum(),
                x,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            relative_scale = (
                (grad_global * hessian_times_grad).sum(dim=1)
                / (
                    grad_norm.pow(3)
                    + float(self.coord_config.scale_denominator_eps)
                )
            )
            scale_terms.append(relative_scale.square().mean())

        return torch.stack(sub_terms).mean(), torch.stack(scale_terms).mean()


def _atlas_on_device(
    atlas: AtlasData,
    device: torch.device,
) -> AtlasData:
    if atlas.n_tissues <= 0:
        raise ValueError("atlas contains no tissues")
    if atlas.n_genes <= 0:
        raise ValueError("atlas contains no genes")
    return atlas.to(device)


@torch.no_grad()
def _refresh_gauge(
    model: Stage1aSplineModel,
    atlas: AtlasData,
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> None:
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("gauge quantiles must satisfy 0 <= low < high <= 1")

    model.eval()
    for tissue_id, tissue in enumerate(atlas.tissues):
        raw = model.coordinate_maps.raw_one_slice(tissue.coords, tissue_id)
        low = torch.quantile(raw, float(lower_quantile))
        high = torch.quantile(raw, float(upper_quantile))
        if (
            not bool(torch.isfinite(low))
            or not bool(torch.isfinite(high))
            or float((high - low).detach().cpu()) <= 1e-5
        ):
            raise RuntimeError(f"Degenerate robust gauge for tissue {tissue_id}.")
        model.coordinate_maps.gauge_low[tissue_id] = low
        model.coordinate_maps.gauge_high[tissue_id] = high


def _balanced_batch(
    atlas: AtlasData,
    *,
    batch_size_per_tissue: int,
    device: torch.device,
) -> dict[str, Tensor]:
    coords = []
    counts = []
    log_library = []
    tissue_idx = []
    times = []

    for tissue_id, tissue in enumerate(atlas.tissues):
        n = tissue.n_cells
        m = min(max(int(batch_size_per_tissue), 1), n)
        if m == n:
            idx = torch.arange(n, device=device)
        else:
            idx = torch.randperm(n, device=device)[:m]

        coords.append(tissue.coords[idx])
        counts.append(tissue.counts[idx])
        log_library.append(tissue.log_library_size[idx])
        tissue_idx.append(
            torch.full((m,), tissue_id, dtype=torch.long, device=device)
        )
        times.append(
            torch.full(
                (m,),
                float(tissue.time),
                dtype=tissue.coords.dtype,
                device=device,
            )
        )

    return {
        "coords": torch.cat(coords, dim=0),
        "counts": torch.cat(counts, dim=0),
        "log_library_size": torch.cat(log_library, dim=0),
        "slice_idx": torch.cat(tissue_idx, dim=0),
        "time": torch.cat(times, dim=0),
    }


def _geometry_sample(
    atlas: AtlasData,
    *,
    cells_per_tissue: int,
    device: torch.device,
) -> list[Tensor]:
    samples: list[Tensor] = []
    for tissue in atlas.tissues:
        n = tissue.n_cells
        m = min(max(int(cells_per_tissue), 1), n)
        if m == n:
            idx = torch.arange(n, device=device)
        else:
            idx = torch.randperm(n, device=device)[:m]
        samples.append(tissue.coords[idx])
    return samples


def _full_data(
    atlas: AtlasData,
    *,
    device: torch.device,
) -> dict[str, Tensor]:
    coords = []
    counts = []
    log_exposure = []
    tissue_index = []
    times = []

    for tissue_id, tissue in enumerate(atlas.tissues):
        n = tissue.n_cells
        coords.append(tissue.coords)
        counts.append(tissue.counts)
        log_exposure.append(tissue.log_library_size)
        tissue_index.append(
            torch.full((n,), tissue_id, dtype=torch.long, device=device)
        )
        times.append(
            torch.full(
                (n,),
                float(tissue.time),
                dtype=tissue.coords.dtype,
                device=device,
            )
        )

    return {
        "xy": torch.cat(coords, dim=0),
        "counts": torch.cat(counts, dim=0),
        "log_exposure": torch.cat(log_exposure, dim=0),
        "tissue_index": torch.cat(tissue_index, dim=0),
        "time": torch.cat(times, dim=0),
    }


def _loss_terms(
    model: Stage1aSplineModel,
    batch: dict[str, Tensor],
    geometry_coords: list[Tensor],
    *,
    submersion_weight: float,
    scale_variation_weight: float,
    coverage_prior_weight: float,
    nuisance_weight: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    output = model(
        coords=batch["coords"],
        slice_idx=batch["slice_idx"],
        time=batch["time"],
        log_library_size=batch["log_library_size"],
    )
    data_nll = F.poisson_nll_loss(
        output["log_mu"],
        batch["counts"].float(),
        log_input=True,
        full=False,
        reduction="mean",
    )
    submersion, scale_variation = model.coordinate_geometry_penalties(
        geometry_coords
    )
    coverage = model.coordinate_maps.coverage_penalty()
    nuisance = model.nuisance_penalty()

    total = (
        data_nll
        + float(submersion_weight) * submersion
        + float(scale_variation_weight) * scale_variation
        + float(coverage_prior_weight) * coverage
        + float(nuisance_weight) * nuisance
    )
    return total, {
        "data_nll": data_nll.detach(),
        "submersion": submersion.detach(),
        "scale_variation": scale_variation.detach(),
        "coverage_prior": coverage.detach(),
        "nuisance": nuisance.detach(),
    }


@torch.no_grad()
def _result(
    method: str,
    model: Stage1aSplineModel,
    atlas: AtlasData,
    history: list[dict[str, float]],
) -> ZooResult:
    model.eval()
    z_by_tissue = [
        model.coordinate_maps.one(tissue.coords, tissue_id)
        .detach()
        .cpu()
        .numpy()
        for tissue_id, tissue in enumerate(atlas.tissues)
    ]
    result = ZooResult(
        method=method,
        z_by_tissue=z_by_tissue,
        model=model,
        auxiliary={
            "expression_field": model.expression_field,
            "expression_backend": "tensor_product_bspline",
            "coordinate_backend": "stage1a_neural",
        },
        history=history,
    )
    result.check()
    return result


def fit_stage1a_spline_joint(
    atlas: AtlasData,
    config: Stage1aSplineJointConfig = Stage1aSplineJointConfig(),
    *,
    device: Optional[torch.device] = None,
) -> ZooResult:
    """Joint AdamW fit of Stage1a neural alpha maps and spline E."""
    if device is None:
        device = default_device()
    set_seed(config.seed)
    atlas_device = _atlas_on_device(atlas, device)

    model = Stage1aSplineModel(
        num_tissues=atlas.n_tissues,
        num_genes=atlas.n_genes,
        config=config.model,
    ).to(device)
    initialize_expression_from_global_means(
        model,
        torch.cat([t.counts for t in atlas_device.tissues], dim=0),
        torch.cat([t.log_library_size for t in atlas_device.tissues], dim=0),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    history: list[dict[str, float]] = []

    for epoch in range(1, int(config.epochs) + 1):
        _refresh_gauge(
            model,
            atlas_device,
            lower_quantile=config.gauge_lower_quantile,
            upper_quantile=config.gauge_upper_quantile,
        )
        model.train()
        batch = _balanced_batch(
            atlas_device,
            batch_size_per_tissue=config.batch_size_per_tissue,
            device=device,
        )
        geometry = _geometry_sample(
            atlas_device,
            cells_per_tissue=config.geometry_cells_per_tissue,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        loss, terms = _loss_terms(
            model,
            batch,
            geometry,
            submersion_weight=config.submersion_weight,
            scale_variation_weight=config.scale_variation_weight,
            coverage_prior_weight=config.coverage_prior_weight,
            nuisance_weight=config.nuisance_weight,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite joint loss at epoch {epoch}.")
        loss.backward()
        if config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config.grad_clip_norm),
            )
        optimizer.step()
        model.coordinate_maps.project_endpoints_()

        if (
            epoch == 1
            or epoch == int(config.epochs)
            or epoch % max(int(config.print_every), 1) == 0
        ):
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": float(loss.detach().cpu()),
                    **{k: float(v.cpu()) for k, v in terms.items()},
                }
            )

    _refresh_gauge(
        model,
        atlas_device,
        lower_quantile=config.gauge_lower_quantile,
        upper_quantile=config.gauge_upper_quantile,
    )
    return _result(
        "stage1a_nn_spline_joint",
        model,
        atlas_device,
        history,
    )


def fit_stage1a_spline_profiled(
    atlas: AtlasData,
    config: Stage1aSplineProfiledConfig = Stage1aSplineProfiledConfig(),
    *,
    device: Optional[torch.device] = None,
) -> ZooResult:
    """Profile E by Poisson IRLS while alternating updates of Stage1a alpha."""
    if device is None:
        device = default_device()
    if int(config.model.slice_effect_dim) != 0:
        raise ValueError(
            "Profiled Stage1a spline currently requires slice_effect_dim=0."
        )

    set_seed(config.seed)
    atlas_device = _atlas_on_device(atlas, device)
    model = Stage1aSplineModel(
        num_tissues=atlas.n_tissues,
        num_genes=atlas.n_genes,
        config=config.model,
    ).to(device)

    full = _full_data(atlas_device, device=device)
    initialize_expression_from_global_means(
        model,
        full["counts"],
        full["log_exposure"],
    )

    coordinate_parameters = list(model.coordinate_maps.parameters())
    optimizer = torch.optim.AdamW(
        coordinate_parameters,
        lr=float(config.coordinate_learning_rate),
        weight_decay=float(config.weight_decay),
    )
    history: list[dict[str, float]] = []

    for outer in range(int(config.outer_iterations)):
        _refresh_gauge(
            model,
            atlas_device,
            lower_quantile=config.gauge_lower_quantile,
            upper_quantile=config.gauge_upper_quantile,
        )
        with torch.no_grad():
            z = model.coordinate_maps(
                full["xy"],
                full["tissue_index"],
            )[:, None]

        irls_stats = fit_expression_irls(
            model,
            z=z,
            time=full["time"],
            counts=full["counts"],
            log_exposure=full["log_exposure"],
            max_iter=config.irls_max_iter,
            tol=config.irls_tol,
            ridge=config.irls_ridge,
            gene_chunk_size=config.irls_gene_chunk_size,
        )

        for parameter in model.expression_field.parameters():
            parameter.requires_grad_(False)

        accum: dict[str, float] = {}
        steps = max(int(config.coordinate_steps_per_outer), 1)
        for step in range(steps):
            if step % max(int(config.gauge_refresh_every), 1) == 0:
                _refresh_gauge(
                    model,
                    atlas_device,
                    lower_quantile=config.gauge_lower_quantile,
                    upper_quantile=config.gauge_upper_quantile,
                )

            batch = _balanced_batch(
                atlas_device,
                batch_size_per_tissue=config.batch_size_per_tissue,
                device=device,
            )
            geometry = _geometry_sample(
                atlas_device,
                cells_per_tissue=config.geometry_cells_per_tissue,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            loss, terms = _loss_terms(
                model,
                batch,
                geometry,
                submersion_weight=config.submersion_weight,
                scale_variation_weight=config.scale_variation_weight,
                coverage_prior_weight=config.coverage_prior_weight,
                nuisance_weight=0.0,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite profiled coordinate loss at outer {outer}, step {step}."
                )
            loss.backward()
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    coordinate_parameters,
                    float(config.grad_clip_norm),
                )
            optimizer.step()
            model.coordinate_maps.project_endpoints_()

            accum["loss"] = accum.get("loss", 0.0) + float(loss.detach().cpu())
            for key, value in terms.items():
                accum[key] = accum.get(key, 0.0) + float(value.cpu())

        for parameter in model.expression_field.parameters():
            parameter.requires_grad_(True)

        history.append(
            {
                "outer_iteration": float(outer),
                **irls_stats,
                **{k: v / steps for k, v in accum.items()},
            }
        )

    _refresh_gauge(
        model,
        atlas_device,
        lower_quantile=config.gauge_lower_quantile,
        upper_quantile=config.gauge_upper_quantile,
    )
    with torch.no_grad():
        final_z = model.coordinate_maps(
            full["xy"],
            full["tissue_index"],
        )[:, None]
    final_stats = fit_expression_irls(
        model,
        z=final_z,
        time=full["time"],
        counts=full["counts"],
        log_exposure=full["log_exposure"],
        max_iter=config.irls_max_iter,
        tol=config.irls_tol,
        ridge=config.irls_ridge,
        gene_chunk_size=config.irls_gene_chunk_size,
    )
    history.append(
        {
            "outer_iteration": float(config.outer_iterations),
            **final_stats,
        }
    )

    return _result(
        "stage1a_nn_spline_profiled",
        model,
        atlas_device,
        history,
    )


__all__ = [
    "Stage1aSplineModelConfig",
    "Stage1aSplineJointConfig",
    "Stage1aSplineProfiledConfig",
    "Stage1aSplineModel",
    "fit_stage1a_spline_joint",
    "fit_stage1a_spline_profiled",
]
