"""Block-coordinate maximum-likelihood fitting for the Fisher-Lagrangian model.

Conditional on anatomical coordinates, the spline expression model is a Poisson
GLM and is fit by damped Newton/IRLS. Conditional on that expression field, the
tissue-specific coordinate maps are updated with Adam through the Poisson
likelihood plus the unit-speed/eikonal gauge penalty. Dynamics never enter the
fitting objective; they are derived after the primary MLE is fitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .model import FisherLagrangianModel, Tensor, _as_col


@dataclass(frozen=True)
class BlockFitConfig:
    outer_iterations: int = 8
    coordinate_steps_per_outer: int = 100
    coordinate_learning_rate: float = 1e-3
    coordinate_batch_size: int = 2048
    gauge_weight: float = 1.0
    grad_clip_norm: float | None = 5.0
    irls_max_iter: int = 25
    irls_tol: float = 1e-6
    irls_ridge: float = 1e-6
    irls_gene_chunk_size: int = 32


@dataclass(frozen=True)
class CoordinateInitConfig:
    steps: int = 300
    learning_rate: float = 3e-3
    max_points_per_tissue: int = 10000


def _poisson_nll_from_design(
    design: Tensor,
    coef: Tensor,
    counts: Tensor,
    log_exposure: Tensor,
) -> Tensor:
    eta = design @ coef.T
    log_mu = eta + log_exposure
    return (torch.exp(log_mu.clamp(max=30.0)) - counts * log_mu).sum()


@torch.no_grad()
def initialize_expression_from_global_means(
    model: FisherLagrangianModel,
    counts: Tensor,
    log_exposure: Tensor | None = None,
    *,
    floor: float = 1e-8,
) -> None:
    """Initialize the anchored baseline at the exposure-adjusted global gene mean."""
    y = counts.to(dtype=model.expression_field.baseline_coeff.dtype)
    if log_exposure is None:
        exposure = torch.ones((y.shape[0], 1), dtype=y.dtype, device=y.device)
    else:
        exposure = torch.exp(_as_col(log_exposure).to(dtype=y.dtype, device=y.device))
    mean = y.sum(dim=0) / exposure.sum().clamp_min(floor)
    log_mean = mean.clamp_min(floor).log()
    model.expression_field.baseline_coeff.copy_(
        log_mean[:, None].expand_as(model.expression_field.baseline_coeff)
    )
    model.expression_field.temporal_coeff.zero_()


@torch.no_grad()
def fit_expression_irls(
    model: FisherLagrangianModel,
    *,
    z: Tensor,
    time: Tensor,
    counts: Tensor,
    log_exposure: Tensor | None = None,
    max_iter: int = 25,
    tol: float = 1e-6,
    ridge: float = 1e-6,
    gene_chunk_size: int = 32,
) -> dict[str, float]:
    """Conditional Poisson MLE for spline coefficients by damped IRLS/Newton."""
    field = model.expression_field
    dtype = field.baseline_coeff.dtype
    device = field.baseline_coeff.device
    design = field.design_matrix(z, time).to(dtype=dtype, device=device)
    y = counts.to(dtype=dtype, device=device)
    le = (
        torch.zeros((y.shape[0], 1), dtype=dtype, device=device)
        if log_exposure is None
        else _as_col(log_exposure).to(dtype=dtype, device=device)
    )
    coef = field.coefficient_matrix().detach().clone()
    totals = y.sum(dim=0)
    active = totals > 0
    if (~active).any():
        j = field.spatial_basis_dim
        coef[~active] = 0.0
        coef[~active, :j] = -20.0

    previous = float(_poisson_nll_from_design(design, coef, y, le).cpu())
    p = design.shape[1]
    eye = torch.eye(p, dtype=dtype, device=device)
    iterations = 0
    max_update = 0.0

    for iteration in range(max_iter):
        iterations = iteration + 1
        proposal = coef.clone()
        max_update = 0.0
        active_idx = torch.nonzero(active, as_tuple=False).flatten()
        for start in range(0, active_idx.numel(), gene_chunk_size):
            idx = active_idx[start : start + gene_chunk_size]
            c = coef[idx]
            eta = design @ c.T
            log_mu = eta + le
            mu = torch.exp(log_mu.clamp(max=30.0))
            residual = mu - y[:, idx]
            grad = torch.einsum("np,ng->gp", design, residual)
            hess = torch.einsum("np,ng,nq->gpq", design, mu, design)
            hess = hess + float(ridge) * eye[None, :, :]
            step = torch.linalg.solve(hess, grad.unsqueeze(-1)).squeeze(-1)
            proposal[idx] = c - step
            if step.numel():
                max_update = max(max_update, float(step.abs().max().cpu()))

        accepted = False
        step_scale = 1.0
        candidate_nll = previous
        for _ in range(12):
            candidate = coef + step_scale * (proposal - coef)
            candidate_nll = float(
                _poisson_nll_from_design(design, candidate, y, le).cpu()
            )
            if candidate_nll <= previous + 1e-8:
                coef = candidate
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            break
        relative_improvement = (previous - candidate_nll) / max(abs(previous), 1.0)
        previous = candidate_nll
        if relative_improvement < tol or step_scale * max_update < tol:
            break

    field.set_coefficient_matrix(coef)
    return {
        "irls_iterations": float(iterations),
        "expression_nll_sum": float(previous),
        "irls_max_update": float(max_update),
    }


def _pca_targets(xy: Tensor, *, orientation: float) -> Tensor:
    """Signed unit-speed PCA coordinate in the same units as ``xy``.

    The leading right singular vector has unit Euclidean norm, so projection
    onto it has spatial gradient norm one.  Unlike the previous initialization,
    this target is not min-max normalized to ``[0,1]``.
    """
    centered = xy - xy.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    if orientation < 0:
        direction = -direction
    return centered @ direction


def initialize_coordinate_maps_from_pca(
    model: FisherLagrangianModel,
    xy: Tensor,
    tissue_index: Tensor,
    *,
    orientation_signs: Sequence[float] | None = None,
    config: CoordinateInitConfig = CoordinateInitConfig(),
) -> dict[int, float]:
    """Pretrain each coordinate NN to a signed unit-speed physical major axis."""
    if orientation_signs is None:
        orientation_signs = [1.0] * len(model.coordinate_maps)
    if len(orientation_signs) != len(model.coordinate_maps):
        raise ValueError("orientation_signs must have one value per tissue")
    losses: dict[int, float] = {}
    for idx, mapping in enumerate(model.coordinate_maps):
        mask = tissue_index == idx
        points = xy[mask]
        if points.shape[0] < 2:
            raise ValueError(f"Tissue {idx} has fewer than two cells")
        if points.shape[0] > config.max_points_per_tissue:
            choose = torch.randperm(points.shape[0], device=points.device)[
                : config.max_points_per_tissue
            ]
            points = points[choose]
        targets = _pca_targets(
            points,
            orientation=float(orientation_signs[idx]),
        )[:, None]
        optimizer = torch.optim.Adam(mapping.parameters(), lr=config.learning_rate)
        last = 0.0
        for _ in range(config.steps):
            optimizer.zero_grad(set_to_none=True)
            pred = mapping(points)
            loss = torch.mean((pred - targets) ** 2)
            loss.backward()
            optimizer.step()
            last = float(loss.detach().cpu())
        losses[idx] = last
    return losses


def coordinate_train_step(
    model: FisherLagrangianModel,
    batch: Mapping[str, Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    gauge_weight: float,
    grad_clip_norm: float | None,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, diag = model.loss(
        xy=batch["xy"],
        tissue_index=batch["tissue_index"],
        time=batch["time"],
        counts=batch["counts"],
        log_exposure=batch.get("log_exposure"),
        gauge_weight=gauge_weight,
    )
    loss.backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            model.coordinate_maps.parameters(), float(grad_clip_norm)
        )
    optimizer.step()
    return {
        "coordinate_loss": float(loss.detach().cpu()),
        **{k: float(v.cpu()) for k, v in diag.items()},
    }


def train_step(
    model: FisherLagrangianModel,
    batch: Mapping[str, Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    gauge_weight: float = 1.0,
    grad_clip_norm: float | None = None,
) -> dict[str, float]:
    """Backward-compatible single coordinate-network update."""
    return coordinate_train_step(
        model,
        batch,
        optimizer,
        gauge_weight=gauge_weight,
        grad_clip_norm=grad_clip_norm,
    )


def _slice_batch(data: Mapping[str, Tensor], indices: Tensor) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    n = data["xy"].shape[0]
    for key, value in data.items():
        if isinstance(value, torch.Tensor) and value.shape[0] == n:
            out[key] = value[indices]
    return out


def _check_coordinate_support(model: FisherLagrangianModel, z: Tensor) -> None:
    """Fail early if learned coordinates leave the fixed spline support."""
    lo, hi = model.expression_field.coordinate_domain
    zmin = float(z.min().detach().cpu())
    zmax = float(z.max().detach().cpu())
    if zmin < lo or zmax > hi:
        raise ValueError(
            "Learned anatomical coordinates fall outside the common spatial "
            f"B-spline support [{lo:g}, {hi:g}] (observed [{zmin:g}, {zmax:g}]). "
            "Increase FisherLagrangianConfig.coordinate_domain; do not rescale "
            "individual tissues independently."
        )


def fit_block_coordinate(
    model: FisherLagrangianModel,
    data: Mapping[str, Tensor],
    *,
    config: BlockFitConfig = BlockFitConfig(),
) -> list[dict[str, float]]:
    """Alternate conditional Poisson MLE for ``E`` and Adam updates for ``alpha``."""
    required = {"xy", "tissue_index", "time", "counts"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Missing required data keys: {sorted(missing)}")
    n = data["xy"].shape[0]
    if n == 0:
        raise ValueError("data contain no cells")

    initialize_expression_from_global_means(
        model, data["counts"], data.get("log_exposure")
    )
    coordinate_optimizer = torch.optim.Adam(
        model.coordinate_maps.parameters(), lr=config.coordinate_learning_rate
    )
    history: list[dict[str, float]] = []

    for outer in range(config.outer_iterations):
        model.eval()
        with torch.no_grad():
            z = model.coordinate(data["xy"], data["tissue_index"])
        _check_coordinate_support(model, z)
        expression_stats = fit_expression_irls(
            model,
            z=z,
            time=data["time"],
            counts=data["counts"],
            log_exposure=data.get("log_exposure"),
            max_iter=config.irls_max_iter,
            tol=config.irls_tol,
            ridge=config.irls_ridge,
            gene_chunk_size=config.irls_gene_chunk_size,
        )

        for parameter in model.expression_field.parameters():
            parameter.requires_grad_(False)
        coord_totals: dict[str, float] = {}
        for _ in range(config.coordinate_steps_per_outer):
            batch_size = min(config.coordinate_batch_size, n)
            idx = torch.randint(0, n, (batch_size,), device=data["xy"].device)
            stats = coordinate_train_step(
                model,
                _slice_batch(data, idx),
                coordinate_optimizer,
                gauge_weight=config.gauge_weight,
                grad_clip_norm=config.grad_clip_norm,
            )
            for key, value in stats.items():
                coord_totals[key] = coord_totals.get(key, 0.0) + value
        for parameter in model.expression_field.parameters():
            parameter.requires_grad_(True)

        denom = max(config.coordinate_steps_per_outer, 1)
        history.append(
            {
                "outer_iteration": float(outer),
                **expression_stats,
                **{k: v / denom for k, v in coord_totals.items()},
            }
        )

    model.eval()
    with torch.no_grad():
        final_z = model.coordinate(data["xy"], data["tissue_index"])
    _check_coordinate_support(model, final_z)
    final_stats = fit_expression_irls(
        model,
        z=final_z,
        time=data["time"],
        counts=data["counts"],
        log_exposure=data.get("log_exposure"),
        max_iter=config.irls_max_iter,
        tol=config.irls_tol,
        ridge=config.irls_ridge,
        gene_chunk_size=config.irls_gene_chunk_size,
    )
    history.append({"outer_iteration": float(config.outer_iterations), **final_stats})
    return history


# Backward-compatible names for the earlier prototype API.
TrainConfig = BlockFitConfig
fit = fit_block_coordinate
