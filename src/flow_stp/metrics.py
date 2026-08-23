"""Geometry metrics for learned anatomical coordinates.

RPCE (Relative Positional Consistency Error) measures whether a fixed learned
coordinate value z corresponds to the same fractional physical position along
gradient trajectories, within and across tissues.

For a smooth coordinate map alpha and a gradient trajectory p parameterized by
alpha(p(z)) = z,

    p'(z) = grad(alpha) / ||grad(alpha)||^2,

so physical distance per unit z is ||p'(z)|| = 1 / ||grad(alpha)||.  For each
sampled path we integrate this quantity over a common observed z interval and
normalize by that path's total length.  Path-specific tissue thickness therefore
cancels, while inconsistent relative positioning remains visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass
class RPCEResult:
    """Outputs of :func:`compute_rpce`."""

    summary: dict[str, Any]
    per_z: pd.DataFrame
    per_tissue: pd.DataFrame
    paths: pd.DataFrame


def _device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    for value in module.parameters():
        if torch.is_floating_point(value):
            return value.device, value.dtype
    for value in module.buffers():
        if torch.is_floating_point(value):
            return value.device, value.dtype
    return torch.device("cpu"), torch.float32


def _xy_array(value: Any, tissue_id: int) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    xy = np.asarray(value, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(
            f"Tissue {tissue_id}: coordinates must have shape [N, >=2]."
        )
    xy = xy[:, :2]
    if len(xy) < 3 or not np.isfinite(xy).all():
        raise ValueError(
            f"Tissue {tissue_id}: need >=3 finite physical XY coordinates."
        )
    return xy


def _normalization(
    xy_norm_params: Optional[pd.DataFrame],
    n_tissues: int,
) -> tuple[np.ndarray, np.ndarray]:
    if xy_norm_params is None:
        return (
            np.zeros((n_tissues, 2), dtype=np.float64),
            np.ones((n_tissues, 2), dtype=np.float64),
        )

    frame = pd.DataFrame(xy_norm_params).copy()
    required = {"center_x", "center_y", "scale_x", "scale_y"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "xy_norm_params is missing: " + ", ".join(sorted(missing))
        )
    if "slice_id" in frame.columns:
        frame = frame.sort_values("slice_id").reset_index(drop=True)
    if len(frame) != n_tissues:
        raise ValueError("xy_norm_params must contain one row per tissue.")

    center = frame[["center_x", "center_y"]].to_numpy(dtype=np.float64)
    scale = frame[["scale_x", "scale_y"]].to_numpy(dtype=np.float64)
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError("xy_norm_params contains non-finite values.")
    if np.any(scale <= 0):
        raise ValueError("XY normalization scales must be positive.")
    return center, scale


def _alpha(
    coordinate_maps: nn.Module,
    x_phys: torch.Tensor,
    tissue_id: int,
    center: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    x_model = (x_phys - center) / scale
    if hasattr(coordinate_maps, "one_slice"):
        z = coordinate_maps.one_slice(x_model, int(tissue_id))
    else:
        tissue_idx = torch.full(
            (len(x_model),),
            int(tissue_id),
            dtype=torch.long,
            device=x_model.device,
        )
        z = coordinate_maps(x_model, tissue_idx)
    return z.reshape(-1)


def _alpha_cells(
    coordinate_maps: nn.Module,
    xy: np.ndarray,
    tissue_id: int,
    center_np: np.ndarray,
    scale_np: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> np.ndarray:
    center = torch.as_tensor(center_np, device=device, dtype=dtype)
    scale = torch.as_tensor(scale_np, device=device, dtype=dtype)
    parts = []
    with torch.no_grad():
        for start in range(0, len(xy), int(batch_size)):
            x = torch.as_tensor(
                xy[start : start + int(batch_size)],
                device=device,
                dtype=dtype,
            )
            parts.append(
                _alpha(coordinate_maps, x, tissue_id, center, scale)
                .detach()
                .cpu()
                .numpy()
            )
    return np.concatenate(parts)


def _alpha_grad(
    coordinate_maps: nn.Module,
    x_phys: torch.Tensor,
    tissue_id: int,
    center: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x_phys.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        z = _alpha(coordinate_maps, x, tissue_id, center, scale)
        grad = torch.autograd.grad(z.sum(), x, only_inputs=True)[0]
    return z.detach(), grad.detach()


def _rhs(
    coordinate_maps: nn.Module,
    x: torch.Tensor,
    tissue_id: int,
    center: torch.Tensor,
    scale: torch.Tensor,
    min_grad_norm: float,
) -> torch.Tensor:
    _, grad = _alpha_grad(coordinate_maps, x, tissue_id, center, scale)
    grad_sq = grad.square().sum(dim=1)
    denom = torch.clamp(grad_sq, min=float(min_grad_norm) ** 2)
    return grad / denom[:, None]


def _rk4(
    coordinate_maps: nn.Module,
    x: torch.Tensor,
    h: float,
    tissue_id: int,
    center: torch.Tensor,
    scale: torch.Tensor,
    min_grad_norm: float,
) -> torch.Tensor:
    k1 = _rhs(coordinate_maps, x, tissue_id, center, scale, min_grad_norm)
    k2 = _rhs(
        coordinate_maps,
        x + 0.5 * h * k1,
        tissue_id,
        center,
        scale,
        min_grad_norm,
    )
    k3 = _rhs(
        coordinate_maps,
        x + 0.5 * h * k2,
        tissue_id,
        center,
        scale,
        min_grad_norm,
    )
    k4 = _rhs(
        coordinate_maps,
        x + h * k3,
        tissue_id,
        center,
        scale,
        min_grad_norm,
    )
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _project_level(
    coordinate_maps: nn.Module,
    x: torch.Tensor,
    target_z: float,
    tissue_id: int,
    center: torch.Tensor,
    scale: torch.Tensor,
    min_grad_norm: float,
    steps: int,
) -> torch.Tensor:
    out = x.detach().clone()
    for _ in range(int(steps)):
        z, grad = _alpha_grad(coordinate_maps, out, tissue_id, center, scale)
        grad_sq = grad.square().sum(dim=1)
        denom = torch.clamp(grad_sq, min=float(min_grad_norm) ** 2)
        out = out + ((float(target_z) - z) / denom)[:, None] * grad
    return out.detach()


def _farthest_points(xy: np.ndarray, n_select: int) -> np.ndarray:
    n_select = min(int(n_select), len(xy))
    selected = np.empty(n_select, dtype=int)
    selected[0] = 0
    min_d2 = np.sum((xy - xy[0]) ** 2, axis=1)
    for k in range(1, n_select):
        idx = int(np.argmax(min_d2))
        selected[k] = idx
        min_d2 = np.minimum(
            min_d2,
            np.sum((xy - xy[idx]) ** 2, axis=1),
        )
    return selected


def _trace_tissue(
    coordinate_maps: nn.Module,
    xy: np.ndarray,
    z_cells: np.ndarray,
    tissue_id: int,
    center_np: np.ndarray,
    scale_np: np.ndarray,
    z_grid: np.ndarray,
    n_paths: int,
    seed_candidate_multiplier: int,
    projection_steps: int,
    min_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mid = len(z_grid) // 2
    z_ref = float(z_grid[mid])
    order = np.argsort(np.abs(z_cells - z_ref))
    n_candidates = min(
        len(order),
        max(int(n_paths), int(seed_candidate_multiplier) * int(n_paths)),
    )
    candidate_idx = order[:n_candidates]
    candidate_xy = xy[candidate_idx]

    spread = np.std(xy, axis=0)
    spread = np.where(spread > 1e-12, spread, 1.0)
    chosen = _farthest_points(candidate_xy / spread, int(n_paths))
    seed_idx = candidate_idx[chosen]

    center = torch.as_tensor(center_np, device=device, dtype=dtype)
    scale = torch.as_tensor(scale_np, device=device, dtype=dtype)
    x_ref = torch.as_tensor(xy[seed_idx], device=device, dtype=dtype)
    x_ref = _project_level(
        coordinate_maps,
        x_ref,
        z_ref,
        tissue_id,
        center,
        scale,
        min_grad_norm,
        projection_steps,
    )

    positions = torch.empty(
        (len(x_ref), len(z_grid), 2),
        dtype=dtype,
        device=device,
    )
    positions[:, mid] = x_ref

    x = x_ref
    for k in range(mid + 1, len(z_grid)):
        h = float(z_grid[k] - z_grid[k - 1])
        x = _rk4(
            coordinate_maps,
            x,
            h,
            tissue_id,
            center,
            scale,
            min_grad_norm,
        )
        positions[:, k] = x

    x = x_ref
    for k in range(mid - 1, -1, -1):
        h = float(z_grid[k] - z_grid[k + 1])
        x = _rk4(
            coordinate_maps,
            x,
            h,
            tissue_id,
            center,
            scale,
            min_grad_norm,
        )
        positions[:, k] = x

    flat = positions.reshape(-1, 2)
    z_eval, grad = _alpha_grad(
        coordinate_maps,
        flat,
        tissue_id,
        center,
        scale,
    )
    n_path = len(x_ref)
    z_eval = z_eval.reshape(n_path, len(z_grid)).cpu().numpy()
    grad_norm = (
        torch.linalg.vector_norm(grad, dim=1)
        .reshape(n_path, len(z_grid))
        .cpu()
        .numpy()
    )
    return (
        positions.detach().cpu().numpy(),
        grad_norm,
        z_eval - z_grid[None, :],
        seed_idx,
    )


def _relative_position(
    grad_norm: np.ndarray,
    z_grid: np.ndarray,
    min_grad_norm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(grad_norm).all(axis=1)
    valid &= np.min(grad_norm, axis=1) >= float(min_grad_norm)

    speed = np.full_like(grad_norm, np.nan, dtype=np.float64)
    speed[valid] = 1.0 / grad_norm[valid]
    dz = np.diff(z_grid)
    increments = 0.5 * (speed[:, :-1] + speed[:, 1:]) * dz[None, :]
    length = np.concatenate(
        [
            np.zeros((len(speed), 1), dtype=np.float64),
            np.cumsum(increments, axis=1),
        ],
        axis=1,
    )
    total = length[:, -1]
    valid &= np.isfinite(total) & (total > 0)
    relative = np.full_like(length, np.nan, dtype=np.float64)
    relative[valid] = length[valid] / total[valid, None]
    return relative, total, valid


def compute_rpce(
    coordinate_maps: nn.Module,
    spatial_by_tissue: Sequence[Any],
    *,
    xy_norm_params: Optional[pd.DataFrame] = None,
    z_grid: Optional[Sequence[float]] = None,
    z_quantiles: tuple[float, float] = (0.01, 0.99),
    n_z: int = 101,
    n_paths_per_tissue: int = 32,
    seed_candidate_multiplier: int = 10,
    projection_steps: int = 8,
    min_grad_norm: float = 1e-6,
    eval_batch_size: int = 8192,
) -> RPCEResult:
    """Compute Relative Positional Consistency Error and its variants.

    Parameters
    ----------
    coordinate_maps
        Learned coordinate-map module.  It should implement either
        ``one_slice(coords, tissue_id)`` or ``forward(coords, tissue_idx)``.
    spatial_by_tissue
        Physical XY coordinates for each tissue.
    xy_norm_params
        Optional Stage 1a-style dataframe with ``center_x``, ``center_y``,
        ``scale_x``, and ``scale_y``.  Supplying it makes all derivatives and
        lengths physical even though the neural map consumes standardized XY.

    Returns
    -------
    RPCEResult
        ``summary`` contains ``rpce``, ``rpce_within``, ``rpce_between``,
        ``rpce_max``, ``rpce_mae``, and ``rpce_pairwise_between``.

    Notes
    -----
    Tissues are weighted equally.  Consequently, up to numerical precision,

        rpce**2 = rpce_within**2 + rpce_between**2.

    RPCE is invariant to one positive multiplicative length scale per sampled
    trajectory, so tissue thickness may vary within and across tissues.
    """
    if not isinstance(coordinate_maps, nn.Module):
        raise TypeError("coordinate_maps must be a torch.nn.Module.")
    if len(spatial_by_tissue) < 2:
        raise ValueError("RPCE requires at least two tissues.")
    if int(n_paths_per_tissue) < 2:
        raise ValueError("n_paths_per_tissue must be at least 2.")
    if int(n_z) < 5:
        raise ValueError("n_z must be at least 5.")
    if not (0 <= z_quantiles[0] < z_quantiles[1] <= 1):
        raise ValueError("z_quantiles must satisfy 0 <= low < high <= 1.")

    xy = [_xy_array(v, i) for i, v in enumerate(spatial_by_tissue)]
    n_tissues = len(xy)
    centers, scales = _normalization(xy_norm_params, n_tissues)
    device, dtype = _device_dtype(coordinate_maps)

    was_training = coordinate_maps.training
    coordinate_maps.eval()
    try:
        z_cells = [
            _alpha_cells(
                coordinate_maps,
                xy[i],
                i,
                centers[i],
                scales[i],
                device,
                dtype,
                int(eval_batch_size),
            )
            for i in range(n_tissues)
        ]

        if z_grid is None:
            low = max(
                float(np.quantile(z, z_quantiles[0]))
                for z in z_cells
            )
            high = min(
                float(np.quantile(z, z_quantiles[1]))
                for z in z_cells
            )
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                raise RuntimeError(
                    "No common observed z interval across tissues. "
                    "Inspect orientations or provide z_grid explicitly."
                )
            z_values = np.linspace(low, high, int(n_z), dtype=np.float64)
        else:
            z_values = np.asarray(z_grid, dtype=np.float64)
            if (
                z_values.ndim != 1
                or len(z_values) < 5
                or not np.isfinite(z_values).all()
                or np.any(np.diff(z_values) <= 0)
            ):
                raise ValueError(
                    "z_grid must be finite, increasing, and have >=5 values."
                )

        tissue_rel = []
        tissue_lengths = []
        errors = []
        gradients = []
        outside_all = []
        path_rows = []

        for i in range(n_tissues):
            pos, grad_norm, coordinate_error, seed_idx = _trace_tissue(
                coordinate_maps,
                xy[i],
                z_cells[i],
                i,
                centers[i],
                scales[i],
                z_values,
                int(n_paths_per_tissue),
                int(seed_candidate_multiplier),
                int(projection_steps),
                float(min_grad_norm),
                device,
                dtype,
            )
            relative, total_length, valid = _relative_position(
                grad_norm,
                z_values,
                float(min_grad_norm),
            )
            valid_idx = np.flatnonzero(valid)
            if len(valid_idx) < 2:
                raise RuntimeError(
                    f"Tissue {i}: fewer than two valid gradient trajectories."
                )

            lo = xy[i].min(axis=0)
            hi = xy[i].max(axis=0)
            span = np.maximum(hi - lo, 1e-12)
            outside = np.any(
                (pos < (lo - 0.05 * span)[None, None, :])
                | (pos > (hi + 0.05 * span)[None, None, :]),
                axis=2,
            )

            tissue_rel.append(relative[valid])
            tissue_lengths.append(total_length[valid])
            errors.append(coordinate_error[valid])
            gradients.append(grad_norm[valid])
            outside_all.append(outside[valid])

            for path_id, path_idx in enumerate(valid_idx):
                for k, z_value in enumerate(z_values):
                    path_rows.append(
                        {
                            "tissue_id": i,
                            "path_id": path_id,
                            "seed_cell_index": int(seed_idx[path_idx]),
                            "z": float(z_value),
                            "x": float(pos[path_idx, k, 0]),
                            "y": float(pos[path_idx, k, 1]),
                            "grad_norm_physical": float(grad_norm[path_idx, k]),
                            "relative_position": float(relative[path_idx, k]),
                            "total_path_length": float(total_length[path_idx]),
                            "coordinate_error": float(coordinate_error[path_idx, k]),
                            "outside_bbox_5pct": bool(outside[path_idx, k]),
                        }
                    )

        tissue_means = np.stack([x.mean(axis=0) for x in tissue_rel])
        tissue_vars = np.stack([x.var(axis=0, ddof=0) for x in tissue_rel])
        within_var_z = tissue_vars.mean(axis=0)
        between_var_z = tissue_means.var(axis=0, ddof=0)
        total_var_z = within_var_z + between_var_z
        mean_profile = tissue_means.mean(axis=0)

        rpce_within = float(np.sqrt(within_var_z.mean()))
        rpce_between = float(np.sqrt(between_var_z.mean()))
        rpce = float(np.sqrt(total_var_z.mean()))
        rpce_max = float(np.sqrt(total_var_z.max()))

        mae_z = np.mean(
            np.stack(
                [
                    np.mean(np.abs(x - mean_profile[None, :]), axis=0)
                    for x in tissue_rel
                ]
            ),
            axis=0,
        )
        rpce_mae = float(mae_z.mean())

        pairwise_sq = []
        for i in range(n_tissues):
            for j in range(i + 1, n_tissues):
                pairwise_sq.append(
                    (tissue_means[i] - tissue_means[j]) ** 2
                )
        pairwise_between_z = np.mean(np.stack(pairwise_sq), axis=0)
        rpce_pairwise_between = float(
            np.sqrt(pairwise_between_z.mean())
        )

        per_z = pd.DataFrame(
            {
                "z": z_values,
                "mean_relative_position": mean_profile,
                "rpce_sd": np.sqrt(total_var_z),
                "within_sd": np.sqrt(within_var_z),
                "between_sd": np.sqrt(between_var_z),
                "mean_abs_deviation": mae_z,
                "pairwise_between_rms": np.sqrt(pairwise_between_z),
            }
        )

        per_tissue_rows = []
        for i, relative in enumerate(tissue_rel):
            sd_z = relative.std(axis=0, ddof=0)
            lengths = tissue_lengths[i]
            mean_length = float(np.mean(lengths))
            per_tissue_rows.append(
                {
                    "tissue_id": i,
                    "n_paths": int(len(relative)),
                    "rpce_within": float(np.sqrt(np.mean(sd_z**2))),
                    "rpce_within_max": float(np.max(sd_z)),
                    "mean_path_length": mean_length,
                    "path_length_sd": float(np.std(lengths, ddof=0)),
                    "path_length_cv": float(
                        np.std(lengths, ddof=0) / max(mean_length, 1e-12)
                    ),
                }
            )
        per_tissue = pd.DataFrame(per_tissue_rows)

        all_errors = np.concatenate([x.reshape(-1) for x in errors])
        all_gradients = np.concatenate([x.reshape(-1) for x in gradients])
        all_outside = np.concatenate([x.reshape(-1) for x in outside_all])

        summary = {
            "rpce": rpce,
            "rpce_within": rpce_within,
            "rpce_between": rpce_between,
            "rpce_max": rpce_max,
            "rpce_mae": rpce_mae,
            "rpce_pairwise_between": rpce_pairwise_between,
            "variance_decomposition_error": float(
                abs(
                    rpce**2
                    - rpce_within**2
                    - rpce_between**2
                )
            ),
            "n_tissues": int(n_tissues),
            "n_paths": int(sum(len(x) for x in tissue_rel)),
            "n_z": int(len(z_values)),
            "z_min": float(z_values[0]),
            "z_max": float(z_values[-1]),
            "min_grad_norm_observed": float(np.min(all_gradients)),
            "median_grad_norm_observed": float(np.median(all_gradients)),
            "coordinate_error_rmse": float(
                np.sqrt(np.mean(all_errors**2))
            ),
            "coordinate_error_max_abs": float(
                np.max(np.abs(all_errors))
            ),
            "outside_bbox_5pct_fraction": float(np.mean(all_outside)),
        }

        return RPCEResult(
            summary=summary,
            per_z=per_z,
            per_tissue=per_tissue,
            paths=pd.DataFrame.from_records(path_rows),
        )
    finally:
        coordinate_maps.train(was_training)


def compute_rpce_from_stage1a(
    result: Any,
    **kwargs: Any,
) -> RPCEResult:
    """Convenience wrapper that accepts a ``Stage1AResult`` directly."""
    if not hasattr(result, "model") or not hasattr(
        result.model,
        "coordinate_maps",
    ):
        raise TypeError("result must provide result.model.coordinate_maps.")
    if not hasattr(result, "cells") or not hasattr(
        result,
        "xy_norm_params",
    ):
        raise TypeError("result must provide cells and xy_norm_params.")

    cells = pd.DataFrame(result.cells)
    required = {"slice_id", "x", "y"}
    missing = required.difference(cells.columns)
    if missing:
        raise ValueError(
            "result.cells is missing: " + ", ".join(sorted(missing))
        )

    tissue_ids = np.sort(cells["slice_id"].unique()).astype(int)
    expected = np.arange(len(tissue_ids), dtype=int)
    if not np.array_equal(tissue_ids, expected):
        raise ValueError(
            "result.cells slice_id must be contiguous 0..n_slices-1."
        )

    spatial = [
        cells.loc[
            cells["slice_id"] == i,
            ["x", "y"],
        ].to_numpy(dtype=np.float64)
        for i in expected
    ]
    return compute_rpce(
        result.model.coordinate_maps,
        spatial,
        xy_norm_params=result.xy_norm_params,
        **kwargs,
    )


__all__ = [
    "RPCEResult",
    "compute_rpce",
    "compute_rpce_from_stage1a",
]
