"""Discrete geometric-atlas competitor.

Each tissue is treated as a local one-dimensional chart of the same oriented
process manifold. The numerical coordinate gauges of different tissues are not
forced to agree.

The estimator first extracts local Fisher principal-curve charts, then fits a
partial monotone transport map between each pair of adjacent developmental
times by minimizing Fisher-flat expression displacement. The residual
displacement gives a discrete progression increment. No temporal derivative of
a global expression surface is required.

Under independent monotone reparameterizations of the tissue charts, adjacent
maps transform by conjugacy. The fitted atlas therefore represents transport
modulo chart gauge rather than imposing an arbitrary common numerical gauge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from ..common import AtlasData, ZooResult, default_device, set_seed
from ..warps import MonotoneWarpFamily
from .curve_registration import (
    _fit_principal_curve,
    _fit_shared_pca,
    _fit_spline_curve,
    _local_fisher_coordinates,
    _synchronize_orientations,
)


Tensor = torch.Tensor


@dataclass
class Config:
    """Configuration for the discrete geometric-atlas estimator."""

    local_k: int = 16
    pseudocount: float = 0.5
    pca_rank: int = 12
    pca_fit_cells_per_tissue: int = 10000

    curve_knots: int = 12
    curve_degree: int = 3
    curve_rounds: int = 3
    curve_grid_size: int = 256
    curve_ridge: float = 1e-3

    spatial_graph_k: int = 8
    spatial_lambda: float = 0.3
    lower_quantile: float = 0.005
    upper_quantile: float = 0.995

    registration_grid_size: int = 192
    registration_steps: int = 1200
    registration_lr: float = 3e-2
    warp_degree: int = 4
    warp_quadrature_points: int = 32
    max_missing_each_end: float = 0.25
    initial_missing: float = 0.01
    warp_roughness_weight: float = 1e-3
    warp_magnitude_weight: float = 1e-3
    overlap_penalty_weight: float = 5e-2
    grad_clip_norm: float | None = 5.0
    print_every: int = 200

    workers: int = 1
    seed: int = 20260922


@dataclass(frozen=True)
class AdjacentTransportMap:
    """Monotone transport between two adjacent local tissue charts."""

    source_tissue: int
    target_tissue: int
    source_time: float
    target_time: float
    source_coordinate: np.ndarray
    target_coordinate: np.ndarray
    progression_increment: np.ndarray
    progression_rate: np.ndarray
    diagnostics: dict[str, float]

    @property
    def source_interval(self) -> tuple[float, float]:
        return (
            float(self.source_coordinate[0]),
            float(self.source_coordinate[-1]),
        )

    @property
    def target_interval(self) -> tuple[float, float]:
        return (
            float(self.target_coordinate[0]),
            float(self.target_coordinate[-1]),
        )

    def forward(self, coordinate: Any) -> np.ndarray:
        x = np.asarray(coordinate, dtype=np.float64)
        out = np.full(x.shape, np.nan, dtype=np.float64)
        lo, hi = self.source_interval
        valid = np.isfinite(x) & (x >= lo) & (x <= hi)
        out[valid] = np.interp(
            x[valid],
            self.source_coordinate,
            self.target_coordinate,
        )
        return out

    def inverse(self, coordinate: Any) -> np.ndarray:
        y = np.asarray(coordinate, dtype=np.float64)
        out = np.full(y.shape, np.nan, dtype=np.float64)
        lo, hi = self.target_interval
        valid = np.isfinite(y) & (y >= lo) & (y <= hi)
        out[valid] = np.interp(
            y[valid],
            self.target_coordinate,
            self.source_coordinate,
        )
        return out

    def progression(self, coordinate: Any) -> np.ndarray:
        x = np.asarray(coordinate, dtype=np.float64)
        out = np.full(x.shape, np.nan, dtype=np.float64)
        lo, hi = self.source_interval
        valid = np.isfinite(x) & (x >= lo) & (x <= hi)
        out[valid] = np.interp(
            x[valid],
            self.source_coordinate,
            self.progression_increment,
        )
        return out


@dataclass(frozen=True)
class DiscreteGeometricAtlas:
    """Fitted local charts and adjacent gauge-equivariant transport maps."""

    times: np.ndarray
    ordered_tissues: tuple[int, ...]
    adjacent_maps: tuple[AdjacentTransportMap, ...]

    def _position(self, tissue_id: int) -> int:
        try:
            return self.ordered_tissues.index(int(tissue_id))
        except ValueError as exc:
            raise KeyError(f"unknown tissue id {tissue_id}") from exc

    def transport(
        self,
        coordinate: Any,
        *,
        source_tissue: int,
        target_tissue: int,
    ) -> np.ndarray:
        """Compose adjacent chart maps without extrapolating overlaps."""
        x = np.asarray(coordinate, dtype=np.float64)
        source_pos = self._position(source_tissue)
        target_pos = self._position(target_tissue)
        if source_pos == target_pos:
            return x.copy()

        out = x.copy()
        if source_pos < target_pos:
            for pair in self.adjacent_maps[source_pos:target_pos]:
                out = pair.forward(out)
        else:
            for pair in reversed(self.adjacent_maps[target_pos:source_pos]):
                out = pair.inverse(out)
        return out

    def progression(
        self,
        coordinate: Any,
        *,
        source_tissue: int,
        target_tissue: int,
    ) -> np.ndarray:
        """Accumulate discrete Fisher progression along a forward atlas path."""
        source_pos = self._position(source_tissue)
        target_pos = self._position(target_tissue)
        if target_pos < source_pos:
            raise ValueError("progression is defined here only in forward time")

        x = np.asarray(coordinate, dtype=np.float64)
        rho = np.zeros_like(x, dtype=np.float64)
        current = x.copy()
        for pair in self.adjacent_maps[source_pos:target_pos]:
            increment = pair.progression(current)
            rho = rho + increment
            current = pair.forward(current)
        rho[~np.isfinite(current)] = np.nan
        return rho


class _PartialMonotoneMap(nn.Module):
    """Monotone map between learned source and target overlap intervals."""

    def __init__(
        self,
        *,
        degree: int,
        quadrature_points: int,
        max_missing_each_end: float,
        initial_missing: float,
    ) -> None:
        super().__init__()
        max_missing = float(max_missing_each_end)
        if not 0.0 < max_missing < 0.5:
            raise ValueError("max_missing_each_end must lie in (0, 0.5)")
        initial = float(initial_missing)
        if not 0.0 <= initial < max_missing:
            raise ValueError("initial_missing must lie in [0, max_missing)")

        self.max_missing_each_end = max_missing
        fraction = np.clip(initial / max_missing, 1e-5, 1.0 - 1e-5)
        logit = float(np.log(fraction / (1.0 - fraction)))
        self.endpoint_logits = nn.Parameter(torch.full((4,), logit))
        self.warp = MonotoneWarpFamily(
            1,
            degree=int(degree),
            quadrature_points=int(quadrature_points),
        )

    def missing_fractions(self) -> Tensor:
        return self.max_missing_each_end * torch.sigmoid(self.endpoint_logits)

    def intervals(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        missing = self.missing_fractions()
        source_left = missing[0]
        source_right = 1.0 - missing[1]
        target_left = missing[2]
        target_right = 1.0 - missing[3]
        return source_left, source_right, target_left, target_right

    def paired_coordinates(self, r: Tensor) -> tuple[Tensor, Tensor]:
        r = r.reshape(-1)
        source_left, source_right, target_left, target_right = self.intervals()
        source = source_left + (source_right - source_left) * r
        warped = self.warp.normalized_one(r, 0)
        target = target_left + (target_right - target_left) * warped
        return source, target

    def overlap_penalty(self) -> Tensor:
        scale = max(self.max_missing_each_end, 1e-8)
        return (self.missing_fractions() / scale).square().mean()


def _interp_uniform_curve(curve: Tensor, coordinate: Tensor) -> Tensor:
    """Differentiable piecewise-linear interpolation on a unit grid."""
    if curve.ndim != 2:
        raise ValueError("curve must have shape [grid, features]")
    x = torch.clamp(coordinate.reshape(-1), 0.0, 1.0)
    m = int(curve.shape[0])
    if m < 2:
        raise ValueError("curve grid needs at least two points")

    position = x * float(m - 1)
    left = torch.floor(position).long().clamp(0, m - 2)
    fraction = (position - left.to(position.dtype))[:, None]
    return curve[left] + fraction * (curve[left + 1] - curve[left])


def fit_adjacent_transport(
    source_curve: np.ndarray,
    target_curve: np.ndarray,
    *,
    source_tissue: int,
    target_tissue: int,
    source_time: float,
    target_time: float,
    config: Config,
    device: torch.device,
) -> tuple[AdjacentTransportMap, list[dict[str, float]]]:
    """Fit one partial monotone adjacent-time transport map."""
    dt = float(target_time) - float(source_time)
    if not dt > 0.0:
        raise ValueError("adjacent transport requires strictly increasing times")

    source = torch.as_tensor(
        np.asarray(source_curve),
        dtype=torch.float32,
        device=device,
    )
    target = torch.as_tensor(
        np.asarray(target_curve),
        dtype=torch.float32,
        device=device,
    )
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target curves must have matching [grid,G] shape")

    model = _PartialMonotoneMap(
        degree=config.warp_degree,
        quadrature_points=config.warp_quadrature_points,
        max_missing_each_end=config.max_missing_each_end,
        initial_missing=config.initial_missing,
    ).to(device)

    r = torch.linspace(
        0.0,
        1.0,
        int(config.registration_grid_size),
        device=device,
        dtype=source.dtype,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.registration_lr),
    )

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None

    for step in range(1, int(config.registration_steps) + 1):
        source_u, target_u = model.paired_coordinates(r)
        psi_source = _interp_uniform_curve(source, source_u)
        psi_target = _interp_uniform_curve(target, target_u)

        displacement = psi_target - psi_source
        data_loss = displacement.square().sum(dim=1).mean()
        roughness = model.warp.roughness_penalty()
        magnitude = model.warp.magnitude_penalty()
        overlap = model.overlap_penalty()

        loss = (
            data_loss
            + float(config.warp_roughness_weight) * roughness
            + float(config.warp_magnitude_weight) * magnitude
            + float(config.overlap_penalty_weight) * overlap
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"non-finite adjacent registration loss at step {step}"
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config.grad_clip_norm),
            )
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }

        if (
            step == 1
            or step == int(config.registration_steps)
            or step % max(int(config.print_every), 1) == 0
        ):
            source_left, source_right, target_left, target_right = model.intervals()
            history.append(
                {
                    "step": float(step),
                    "loss": loss_value,
                    "fisher_displacement_sq": float(data_loss.detach().cpu()),
                    "warp_roughness": float(roughness.detach().cpu()),
                    "warp_magnitude": float(magnitude.detach().cpu()),
                    "overlap_penalty": float(overlap.detach().cpu()),
                    "source_left": float(source_left.detach().cpu()),
                    "source_right": float(source_right.detach().cpu()),
                    "target_left": float(target_left.detach().cpu()),
                    "target_right": float(target_right.detach().cpu()),
                }
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        dense_r = torch.linspace(
            0.0,
            1.0,
            int(config.curve_grid_size),
            dtype=source.dtype,
            device=device,
        )
        source_u, target_u = model.paired_coordinates(dense_r)
        psi_source = _interp_uniform_curve(source, source_u)
        psi_target = _interp_uniform_curve(target, target_u)
        progression = torch.linalg.vector_norm(
            psi_target - psi_source,
            dim=1,
        )

        source_np = source_u.cpu().numpy().astype(np.float64)
        target_np = target_u.cpu().numpy().astype(np.float64)
        progression_np = progression.cpu().numpy().astype(np.float64)

    diagnostics = {
        "best_objective": float(best_loss),
        "mean_progression_increment": float(np.mean(progression_np)),
        "mean_progression_rate": float(np.mean(progression_np) / dt),
        "source_left": float(source_np[0]),
        "source_right": float(source_np[-1]),
        "target_left": float(target_np[0]),
        "target_right": float(target_np[-1]),
        "source_overlap_width": float(source_np[-1] - source_np[0]),
        "target_overlap_width": float(target_np[-1] - target_np[0]),
    }

    transport = AdjacentTransportMap(
        source_tissue=int(source_tissue),
        target_tissue=int(target_tissue),
        source_time=float(source_time),
        target_time=float(target_time),
        source_coordinate=source_np,
        target_coordinate=target_np,
        progression_increment=progression_np,
        progression_rate=progression_np / dt,
        diagnostics=diagnostics,
    )
    return transport, history


def fit(
    atlas: AtlasData,
    config: Config = Config(),
    *,
    device: Optional[torch.device] = None,
) -> ZooResult:
    """Fit local process charts and direct adjacent-time Fisher registrations."""
    if atlas.n_tissues < 2:
        raise ValueError("discrete geometric atlas requires at least two tissues")
    set_seed(config.seed)
    registration_device = default_device() if device is None else torch.device(device)

    times = np.asarray([t.time for t in atlas.tissues], dtype=np.float64)
    order = np.argsort(times, kind="stable")
    ordered_times = times[order]
    if np.any(np.diff(ordered_times) <= 0.0):
        raise ValueError(
            "current discrete-atlas prototype requires one tissue per distinct time"
        )

    Y_by_tissue = [
        _local_fisher_coordinates(
            tissue,
            k=config.local_k,
            pseudocount=config.pseudocount,
            workers=config.workers,
        )
        for tissue in atlas.tissues
    ]

    pca, scores_by_tissue = _fit_shared_pca(
        Y_by_tissue,
        rank=config.pca_rank,
        fit_cells_per_tissue=config.pca_fit_cells_per_tissue,
        seed=config.seed,
    )

    raw_u_by_tissue: list[np.ndarray] = []
    score_curves: list[np.ndarray] = []
    curve_diagnostics: list[dict[str, float]] = []
    common_grid: np.ndarray | None = None

    for tissue, scores in zip(atlas.tissues, scores_by_tissue):
        u, curve, grid, diagnostics = _fit_principal_curve(
            scores,
            np.asarray(tissue.coords.detach().cpu().numpy(), dtype=np.float64),
            config,
        )
        if common_grid is None:
            common_grid = np.asarray(grid, dtype=np.float64)
        elif not np.allclose(common_grid, grid):
            raise RuntimeError("principal-curve grids differ unexpectedly")
        raw_u_by_tissue.append(np.asarray(u, dtype=np.float64))
        score_curves.append(np.asarray(curve, dtype=np.float64))
        curve_diagnostics.append(diagnostics)

    assert common_grid is not None

    signs, orientation_cost = _synchronize_orientations(
        score_curves,
        times,
    )
    oriented_u: list[np.ndarray] = []
    oriented_score_curves: list[np.ndarray] = []
    for sign, u, curve in zip(signs, raw_u_by_tissue, score_curves):
        if int(sign) == 1:
            oriented_u.append(u)
            oriented_score_curves.append(curve)
        else:
            oriented_u.append(1.0 - u)
            oriented_score_curves.append(curve[::-1])

    fisher_curves = [
        _fit_spline_curve(
            oriented_u[i],
            Y_by_tissue[i],
            grid=common_grid,
            n_knots=config.curve_knots,
            degree=config.curve_degree,
            ridge=config.curve_ridge,
        )
        for i in range(atlas.n_tissues)
    ]

    adjacent_maps: list[AdjacentTransportMap] = []
    history: list[dict[str, float]] = []
    pair_diagnostics: list[dict[str, float]] = []

    for pair_index in range(atlas.n_tissues - 1):
        source_tissue = int(order[pair_index])
        target_tissue = int(order[pair_index + 1])
        transport, pair_history = fit_adjacent_transport(
            fisher_curves[source_tissue],
            fisher_curves[target_tissue],
            source_tissue=source_tissue,
            target_tissue=target_tissue,
            source_time=float(times[source_tissue]),
            target_time=float(times[target_tissue]),
            config=config,
            device=registration_device,
        )
        adjacent_maps.append(transport)
        pair_diagnostics.append(
            {
                "pair_index": float(pair_index),
                "source_tissue": float(source_tissue),
                "target_tissue": float(target_tissue),
                **transport.diagnostics,
            }
        )
        for row in pair_history:
            history.append(
                {
                    "pair_index": float(pair_index),
                    "source_tissue": float(source_tissue),
                    "target_tissue": float(target_tissue),
                    **row,
                }
            )

    atlas_model = DiscreteGeometricAtlas(
        times=ordered_times.copy(),
        ordered_tissues=tuple(int(i) for i in order),
        adjacent_maps=tuple(adjacent_maps),
    )

    result = ZooResult(
        method="discrete_geometric_atlas",
        z_by_tissue=[np.asarray(u, dtype=np.float64) for u in oriented_u],
        raw_coordinate_by_tissue=[
            np.asarray(u, dtype=np.float64) for u in raw_u_by_tissue
        ],
        auxiliary={
            "coordinate_semantics": (
                "tissue-local oriented charts; compare transport modulo "
                "monotone chart reparameterization"
            ),
            "atlas_model": atlas_model,
            "adjacent_maps": adjacent_maps,
            "curve_grid": common_grid,
            "fisher_expression_curves": fisher_curves,
            "score_curves": oriented_score_curves,
            "orientation_signs": signs,
            "orientation_cost": float(orientation_cost),
            "curve_diagnostics": curve_diagnostics,
            "pair_diagnostics": pair_diagnostics,
            "pca_components": pca.components_,
            "pca_mean": pca.mean_,
            "pca_explained_variance_ratio": pca.explained_variance_ratio_,
        },
        history=history,
    )
    result.check()
    return result


__all__ = [
    "Config",
    "AdjacentTransportMap",
    "DiscreteGeometricAtlas",
    "fit_adjacent_transport",
    "fit",
]
