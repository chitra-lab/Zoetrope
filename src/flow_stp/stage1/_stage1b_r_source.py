# %%
# ============================================================
# Flow-STP Stage 1b: deterministic constant-speed r from
# circular-smoothed master-contour-aligned ell
#
# Purpose
# -------
# Use the previously computed globally aligned ell coordinate and the frozen
# Stage 1a coordinate map z to construct the Stage 1b anatomical coordinate r.
#
# This cell does NOT train Phi/Psi/L. It uses a deterministic teacher chart:
#   - the master reference contour from the previous ell cell supplies seeds
#     indexed by ell;
#   - each seed is flowed through the Stage 1a vector field
#
#         dx/dz = grad z(x) / (||grad z(x)||^2 + eps)
#
#     across the observed z range;
#   - physical arclength along each trajectory is accumulated;
#   - r is the normalized cumulative arclength along the trajectory, so speed
#     in physical tissue units is approximately constant along z for fixed ell.
#
# Expected notebook variables
# ---------------------------
#   master_contour_aligned_circular_ell_results
#       Output from stage1b_master_contour_aligned_circular_ell_cell.py.
#
# Optional notebook variables
# ---------------------------
#   model_flow_stp
#       If present and coord_maps.intervals() exists, r_stage1b is scaled to
#       the learned Stage 1a slice interval [a_i,b_i]. Otherwise r_stage1b
#       equals r_normalized.
#
# Main in-memory outputs
# ----------------------
#   deterministic_r_results
#   deterministic_r_cells_df
#   deterministic_r_summary_df
#   deterministic_r_trajectory_points_df
#
# Files written
# -------------
#   stage1b_deterministic_r_from_master_ell/
#       plots/slice_XX_deterministic_r_heatmap.png
#       plots/all_slices_deterministic_r_heatmaps.png
#       tables/slice_XX_deterministic_r_cells.csv.gz
#       tables/deterministic_r_all_cells.csv.gz
#       tables/deterministic_r_summary_by_slice.csv
#       trajectories/slice_XX_deterministic_r_trajectories.csv.gz
#       trajectories/deterministic_r_all_trajectories.csv.gz
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import gc
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import torch
except Exception:  # the cell can still run without model interval scaling
    torch = None

from scipy.spatial import cKDTree


# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------

if "master_contour_aligned_circular_ell_results" not in globals():
    raise NameError(
        "master_contour_aligned_circular_ell_results is missing. "
        "Run the circular-smoothed master-contour-aligned ell cell first."
    )

ELL_RESULTS = master_contour_aligned_circular_ell_results
SLICE_IDS = sorted(int(k) for k in ELL_RESULTS.keys())

OUT_DIR = Path.cwd() / "stage1b_deterministic_r_from_master_ell"
PLOTS_DIR = OUT_DIR / "plots"
TABLES_DIR = OUT_DIR / "tables"
TRAJECTORY_DIR = OUT_DIR / "trajectories"
for _d in [OUT_DIR, PLOTS_DIR, TABLES_DIR, TRAJECTORY_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Trajectory chart used to estimate arclength.
R_N_ELL_TRAJECTORIES = 256
R_N_Z_SAMPLES = 181
R_Z_QUANTILE_RANGE = (0.005, 0.995)
R_FLOW_SUBSTEPS_PER_Z_INTERVAL = 2
R_FLOW_METHOD = "rk2"  # "euler" or "rk2"
R_FLOW_EPS = 1e-10
R_GRADIENT_FLOOR_SCALE = 1.0
R_MAX_SPATIAL_STEP_GRID_STEPS = 3.0
R_REQUIRE_MASTER_CONNECTED_BLOCK = True

# Assign cells from their already-computed ell,z to the deterministic r chart.
R_ASSIGN_K_NEAREST = 8
R_ASSIGN_IDW_POWER = 2.0
R_ASSIGN_Z_FEATURE_WEIGHT = 2.0
R_CLIP_CELL_Z_TO_TRAJECTORY_RANGE = True
R_MAX_ASSIGN_FEATURE_DISTANCE = np.inf  # keep inf for full coverage; lower for conservative masking

# Plotting.
DISPLAY_FIGURES = True
SAVE_PNG = True
SAVE_PDF = False
DPI = 300
MAX_POINTS_PER_PLOT = 140_000
POINT_SIZE = 0.7
POINT_ALPHA = 0.86
R_CMAP = "viridis"
INVALID_COLOR = "0.80"
TRAJECTORY_COLOR = "black"
TRAJECTORY_ALPHA = 0.18
TRAJECTORY_LINEWIDTH = 0.45
TRAJECTORY_PLOT_STRIDE = 10
FIGSIZE_PER_SLICE = (6.4, 5.6)
MAKE_OVERVIEW_FIGURE = True
OVERVIEW_NCOLS = 5

# Storage.
SAVE_PER_SLICE_TABLES = True
SAVE_COMBINED_TABLES = True
SAVE_TRAJECTORY_TABLES = True


# ------------------------------------------------------------
# Containers
# ------------------------------------------------------------

@dataclass
class DeterministicRTrajectoryChart:
    slice_id: int
    ell_grid: np.ndarray
    z_grid: np.ndarray
    xy: np.ndarray                  # [n_ell, n_z, 2]
    valid: np.ndarray               # [n_ell, n_z]
    arclength: np.ndarray           # [n_ell, n_z]
    total_length: np.ndarray        # [n_ell]
    r_normalized_grid: np.ndarray   # [n_ell, n_z]
    z_eval: np.ndarray              # [n_ell, n_z]


@dataclass
class DeterministicRResult:
    slice_id: int
    panel_label: str
    interval_a: float
    interval_b: float
    interval_width: float
    chart: DeterministicRTrajectoryChart
    cell_table: pd.DataFrame
    trajectory_table: pd.DataFrame
    summary: Dict[str, Any]


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------

def _interp_field(grid: Any, name: str, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    yx = np.column_stack([xy[:, 1], xy[:, 0]])
    return np.asarray(grid.interpolators[name](yx), dtype=np.float64).reshape(-1)


def _inside_ambient_box(grid: Any, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return (
        np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= float(grid.x_values[0]))
        & (xy[:, 0] <= float(grid.x_values[-1]))
        & (xy[:, 1] >= float(grid.y_values[0]))
        & (xy[:, 1] <= float(grid.y_values[-1]))
    )


def _vector_field_dx_dz(grid: Any, xy: np.ndarray, dz_abs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return dx/dz = grad z / (||grad z||^2 + eps), grad norm, valid mask."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    gx = _interp_field(grid, "grad_x", xy)
    gy = _interp_field(grid, "grad_y", xy)
    gm = np.sqrt(gx * gx + gy * gy)

    floor = float(getattr(grid, "gradient_floor", 0.0)) * float(R_GRADIENT_FLOOR_SCALE)
    valid = (
        _inside_ambient_box(grid, xy)
        & np.isfinite(gx)
        & np.isfinite(gy)
        & np.isfinite(gm)
        & (gm > floor)
    )

    denom = gm * gm + float(R_FLOW_EPS)
    v = np.column_stack([gx / denom, gy / denom])
    v[~valid] = np.nan

    # Clip the actual spatial step |v dz|, not just |v|.
    dz_abs = max(float(abs(dz_abs)), 1e-12)
    max_norm = float(R_MAX_SPATIAL_STEP_GRID_STEPS) * float(grid.grid_step) / dz_abs
    norm = np.linalg.norm(v, axis=1)
    too_large = np.isfinite(norm) & (norm > max_norm)
    if np.any(too_large):
        v[too_large] *= (max_norm / norm[too_large])[:, None]

    return v, gm, valid


def _rk_step_z(grid: Any, x: np.ndarray, dz: float, active: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One vectorized z-step for active points. Returns x_new, ok for active rows."""
    xa = x[active]
    method = str(R_FLOW_METHOD).lower()

    v1, _, ok1 = _vector_field_dx_dz(grid, xa, abs(dz))
    ok = ok1 & np.isfinite(v1).all(axis=1)

    if method == "rk2":
        x_mid = xa + 0.5 * float(dz) * v1
        v2, _, ok2 = _vector_field_dx_dz(grid, x_mid, abs(dz))
        ok = ok & ok2 & np.isfinite(v2).all(axis=1)
        x_new = xa + float(dz) * v2
    else:
        x_new = xa + float(dz) * v1

    z_new = _interp_field(grid, "z", x_new)
    ok = ok & _inside_ambient_box(grid, x_new) & np.isfinite(z_new)
    return x_new, ok


def _polyline_arclength(points: np.ndarray) -> Tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return np.zeros(len(points), dtype=np.float64), 0.0
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    return s, float(s[-1])


def _resample_open_polyline_by_arclength(points: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        raise ValueError("Need at least two points on the master contour.")
    keep = np.concatenate([[True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12])
    points = points[keep]
    s, total = _polyline_arclength(points)
    if total <= 0:
        raise ValueError("Master contour has zero arclength.")
    s_new = np.linspace(0.0, total, int(n), dtype=np.float64)
    x = np.interp(s_new, s, points[:, 0])
    y = np.interp(s_new, s, points[:, 1])
    ell = s_new / total
    return ell, np.column_stack([x, y])


def _get_slice_interval(slice_id: int, z_cell: np.ndarray) -> Tuple[float, float, float]:
    """Use learned Stage 1a [a_i,b_i] when available; otherwise use [0,1]."""
    if "model_flow_stp" in globals() and torch is not None:
        model = globals()["model_flow_stp"]
        try:
            if hasattr(model, "coord_maps") and hasattr(model.coord_maps, "intervals"):
                with torch.no_grad():
                    a, b, width = model.coord_maps.intervals()
                return (
                    float(a[int(slice_id)].detach().cpu()),
                    float(b[int(slice_id)].detach().cpu()),
                    float(width[int(slice_id)].detach().cpu()),
                )
        except Exception as exc:
            warnings.warn(f"Could not read coord_maps.intervals(); using normalized r. Reason: {exc}")
    return 0.0, 1.0, 1.0


def _feature_from_ell_z(ell: np.ndarray, z: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    ell = np.asarray(ell, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    z_scaled = (z - float(z_min)) / max(float(z_max - z_min), 1e-12)
    return np.column_stack([
        np.cos(2.0 * np.pi * ell),
        np.sin(2.0 * np.pi * ell),
        float(R_ASSIGN_Z_FEATURE_WEIGHT) * z_scaled,
    ])


# ------------------------------------------------------------
# Build deterministic trajectory chart Psi_teacher(ell,z)
# ------------------------------------------------------------

def _seed_points_from_master_contour(result: Any) -> Tuple[np.ndarray, np.ndarray]:
    master = result.master_contour
    # Use the master contour's actual polyline order. Its inherited ell equals
    # the arclength coordinate by construction, so a uniform resample gives
    # uniform ell seeds.
    return _resample_open_polyline_by_arclength(master.xy, int(R_N_ELL_TRAJECTORIES))


def _choose_z_samples(result: Any) -> np.ndarray:
    z_cell = np.asarray(result.grid.z_cell, dtype=np.float64)
    z_cell = z_cell[np.isfinite(z_cell)]
    q0, q1 = R_Z_QUANTILE_RANGE
    z_min, z_max = np.quantile(z_cell, [float(q0), float(q1)])
    z_min = float(z_min)
    z_max = float(z_max)
    master_z = float(result.grid.master_z_level)
    if not (z_min < master_z < z_max):
        z_min = min(z_min, master_z)
        z_max = max(z_max, master_z)
    z_samples = np.linspace(z_min, z_max, int(R_N_Z_SAMPLES), dtype=np.float64)
    z_samples = np.unique(np.sort(np.concatenate([z_samples, np.array([master_z], dtype=np.float64)])))
    return z_samples


def integrate_master_seed_trajectories(result: Any) -> DeterministicRTrajectoryChart:
    grid = result.grid
    ell_grid, seed_xy = _seed_points_from_master_contour(result)
    z_grid = _choose_z_samples(result)
    master_z = float(grid.master_z_level)
    j0 = int(np.argmin(np.abs(z_grid - master_z)))

    n_ell = len(ell_grid)
    n_z = len(z_grid)
    xy = np.full((n_ell, n_z, 2), np.nan, dtype=np.float64)
    z_eval = np.full((n_ell, n_z), np.nan, dtype=np.float64)
    valid = np.zeros((n_ell, n_z), dtype=bool)

    xy[:, j0, :] = seed_xy
    z_eval[:, j0] = _interp_field(grid, "z", seed_xy)
    valid[:, j0] = _inside_ambient_box(grid, seed_xy) & np.isfinite(z_eval[:, j0])

    def integrate_direction(indices: List[int]) -> None:
        if len(indices) == 0:
            return
        x_cur = seed_xy.copy()
        active = valid[:, j0].copy()
        z_cur = float(z_grid[j0])
        for j in indices:
            z_next = float(z_grid[j])
            dz_total = z_next - z_cur
            n_sub = max(1, int(R_FLOW_SUBSTEPS_PER_Z_INTERVAL))
            dz = dz_total / float(n_sub)
            for _ in range(n_sub):
                active_idx = np.where(active)[0]
                if len(active_idx) == 0:
                    break
                x_new, ok = _rk_step_z(grid, x_cur, dz, active_idx)
                # update active rows only when the step succeeded
                succeeded = active_idx[ok]
                failed = active_idx[~ok]
                x_cur[succeeded, :] = x_new[ok, :]
                active[failed] = False
            xy[:, j, :] = x_cur
            z_here = _interp_field(grid, "z", x_cur)
            z_eval[:, j] = z_here
            # Be lenient here; r assignment later uses nearby valid trajectory samples.
            valid[:, j] = active & np.isfinite(z_here) & _inside_ambient_box(grid, x_cur)
            z_cur = z_next

    integrate_direction(list(range(j0 + 1, n_z)))
    integrate_direction(list(range(j0 - 1, -1, -1)))

    # Accumulate physical arclength on the contiguous valid block containing the master contour.
    arclength = np.full((n_ell, n_z), np.nan, dtype=np.float64)
    r_norm = np.full((n_ell, n_z), np.nan, dtype=np.float64)
    total_length = np.full(n_ell, np.nan, dtype=np.float64)

    for i in range(n_ell):
        v = valid[i].copy()
        if not bool(v[j0]):
            continue

        if bool(R_REQUIRE_MASTER_CONNECTED_BLOCK):
            left = j0
            while left > 0 and bool(v[left - 1]):
                left -= 1
            right = j0
            while right < n_z - 1 and bool(v[right + 1]):
                right += 1
            idx = np.arange(left, right + 1, dtype=int)
        else:
            idx = np.where(v)[0]

        if len(idx) < 3:
            continue
        pts = xy[i, idx, :]
        ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        T = float(s[-1])
        if not np.isfinite(T) or T <= 0:
            continue
        arclength[i, idx] = s
        r_norm[i, idx] = s / T
        total_length[i] = T

    return DeterministicRTrajectoryChart(
        slice_id=int(result.slice_id),
        ell_grid=ell_grid.astype(np.float64),
        z_grid=z_grid.astype(np.float64),
        xy=xy.astype(np.float64),
        valid=valid.astype(bool),
        arclength=arclength.astype(np.float64),
        total_length=total_length.astype(np.float64),
        r_normalized_grid=r_norm.astype(np.float64),
        z_eval=z_eval.astype(np.float64),
    )


# ------------------------------------------------------------
# Assign cell r from ell,z by interpolation over the trajectory chart
# ------------------------------------------------------------

def assign_cells_to_r_chart(result: Any, chart: DeterministicRTrajectoryChart) -> pd.DataFrame:
    d = result.cell_table.copy()
    z_min = float(np.nanmin(chart.z_grid))
    z_max = float(np.nanmax(chart.z_grid))

    ell = d["ell_master_aligned"].to_numpy(dtype=np.float64)
    z = d["z_stage1a"].to_numpy(dtype=np.float64)
    ell_valid = d["ell_valid"].to_numpy(dtype=bool)
    finite_cell = ell_valid & np.isfinite(ell) & np.isfinite(z)

    z_query = z.copy()
    z_extrapolated = np.zeros(len(d), dtype=bool)
    if bool(R_CLIP_CELL_Z_TO_TRAJECTORY_RANGE):
        z_extrapolated = finite_cell & ((z_query < z_min) | (z_query > z_max))
        z_query = np.clip(z_query, z_min, z_max)

    sample_ell = np.repeat(chart.ell_grid.reshape(-1, 1), len(chart.z_grid), axis=1)
    sample_z = np.repeat(chart.z_grid.reshape(1, -1), len(chart.ell_grid), axis=0)
    sample_r = chart.r_normalized_grid
    sample_xy = chart.xy
    sample_valid = np.isfinite(sample_r) & np.isfinite(sample_ell) & np.isfinite(sample_z)

    if not np.any(sample_valid):
        raise RuntimeError(f"Slice {int(result.slice_id)}: no valid r trajectory samples.")

    feat_samples = _feature_from_ell_z(
        sample_ell[sample_valid],
        sample_z[sample_valid],
        z_min,
        z_max,
    )
    r_samples = sample_r[sample_valid]
    xy_samples = sample_xy[sample_valid]
    tree = cKDTree(feat_samples)

    r_out = np.full(len(d), np.nan, dtype=np.float64)
    assign_dist = np.full(len(d), np.nan, dtype=np.float64)
    nearest_x = np.full(len(d), np.nan, dtype=np.float64)
    nearest_y = np.full(len(d), np.nan, dtype=np.float64)
    assign_valid = np.zeros(len(d), dtype=bool)

    idx_cells = np.where(finite_cell)[0]
    if len(idx_cells) > 0:
        feat_cells = _feature_from_ell_z(ell[idx_cells], z_query[idx_cells], z_min, z_max)
        k = min(int(R_ASSIGN_K_NEAREST), len(feat_samples))
        dist, nn = tree.query(feat_cells, k=k)
        if k == 1:
            dist = dist[:, None]
            nn = nn[:, None]
        weights = 1.0 / np.maximum(dist, 1e-12) ** float(R_ASSIGN_IDW_POWER)
        weights[~np.isfinite(weights)] = 0.0
        denom = weights.sum(axis=1)
        ok = denom > 0
        r_vals = np.full(len(idx_cells), np.nan, dtype=np.float64)
        r_vals[ok] = (weights[ok] * r_samples[nn[ok]]).sum(axis=1) / denom[ok]
        min_dist = np.min(dist, axis=1)
        nearest = nn[np.arange(len(idx_cells)), np.argmin(dist, axis=1)]
        valid_here = ok & np.isfinite(r_vals) & (min_dist <= float(R_MAX_ASSIGN_FEATURE_DISTANCE))
        r_out[idx_cells] = r_vals
        assign_dist[idx_cells] = min_dist
        nearest_x[idx_cells] = xy_samples[nearest, 0]
        nearest_y[idx_cells] = xy_samples[nearest, 1]
        assign_valid[idx_cells] = valid_here

    a, b, width = _get_slice_interval(int(result.slice_id), result.grid.z_cell)
    r_stage1b = a + width * r_out

    out = pd.DataFrame(
        {
            "slice_id": int(result.slice_id),
            "cell_index": d["cell_index"].to_numpy(dtype=int),
            "x": d["x"].to_numpy(dtype=np.float64),
            "y": d["y"].to_numpy(dtype=np.float64),
            "z_stage1a": z,
            "ell_master_aligned": ell,
            "ell_valid": ell_valid,
            "r_normalized": r_out,
            "r_stage1b": r_stage1b,
            "r_valid": assign_valid,
            "r_assignment_feature_distance": assign_dist,
            "r_nearest_chart_x": nearest_x,
            "r_nearest_chart_y": nearest_y,
            "r_z_query": z_query,
            "r_z_extrapolated_or_clipped": z_extrapolated,
            "interval_a": float(a),
            "interval_b": float(b),
            "interval_width": float(width),
        }
    )
    return out


# ------------------------------------------------------------
# Tables and plots
# ------------------------------------------------------------

def trajectory_points_dataframe(result: Any, chart: DeterministicRTrajectoryChart) -> pd.DataFrame:
    ell_grid = np.repeat(chart.ell_grid.reshape(-1, 1), len(chart.z_grid), axis=1)
    z_grid = np.repeat(chart.z_grid.reshape(1, -1), len(chart.ell_grid), axis=0)
    return pd.DataFrame(
        {
            "slice_id": int(result.slice_id),
            "trajectory_id": np.repeat(np.arange(len(chart.ell_grid), dtype=int), len(chart.z_grid)),
            "ell": ell_grid.reshape(-1),
            "z_target": z_grid.reshape(-1),
            "x": chart.xy[:, :, 0].reshape(-1),
            "y": chart.xy[:, :, 1].reshape(-1),
            "z_eval": chart.z_eval.reshape(-1),
            "trajectory_valid": chart.valid.reshape(-1),
            "arc_length_from_low": chart.arclength.reshape(-1),
            "trajectory_total_length": np.repeat(chart.total_length, len(chart.z_grid)),
            "r_normalized": chart.r_normalized_grid.reshape(-1),
        }
    )


def compute_deterministic_r_for_slice(slice_id: int) -> DeterministicRResult:
    result = ELL_RESULTS[int(slice_id)]
    chart = integrate_master_seed_trajectories(result)
    cell_table = assign_cells_to_r_chart(result, chart)
    traj_table = trajectory_points_dataframe(result, chart)

    a = float(cell_table["interval_a"].iloc[0])
    b = float(cell_table["interval_b"].iloc[0])
    width = float(cell_table["interval_width"].iloc[0])

    valid = cell_table["r_valid"].to_numpy(bool)
    r = cell_table.loc[valid, "r_normalized"].to_numpy(float)
    z = cell_table.loc[valid, "z_stage1a"].to_numpy(float)
    ell = cell_table.loc[valid, "ell_master_aligned"].to_numpy(float)
    dist = cell_table.loc[valid, "r_assignment_feature_distance"].to_numpy(float)

    good_traj = np.isfinite(chart.total_length) & (chart.total_length > 0)
    summary = {
        "slice_id": int(slice_id),
        "panel_label": str(result.panel_label),
        "n_cells": int(len(cell_table)),
        "r_valid_fraction": float(cell_table["r_valid"].mean()),
        "ell_valid_fraction": float(cell_table["ell_valid"].mean()),
        "n_ell_trajectories": int(len(chart.ell_grid)),
        "n_z_samples": int(len(chart.z_grid)),
        "n_valid_trajectories": int(good_traj.sum()),
        "valid_trajectory_fraction": float(good_traj.mean()),
        "z_min_chart": float(np.nanmin(chart.z_grid)),
        "z_max_chart": float(np.nanmax(chart.z_grid)),
        "master_z_level": float(result.grid.master_z_level),
        "interval_a": a,
        "interval_b": b,
        "interval_width": width,
        "r_normalized_mean": float(np.nanmean(r)) if r.size else np.nan,
        "r_normalized_std": float(np.nanstd(r)) if r.size else np.nan,
        "r_normalized_q01": float(np.nanquantile(r, 0.01)) if r.size else np.nan,
        "r_normalized_q99": float(np.nanquantile(r, 0.99)) if r.size else np.nan,
        "trajectory_total_length_median": float(np.nanmedian(chart.total_length[good_traj])) if good_traj.any() else np.nan,
        "trajectory_total_length_q10": float(np.nanquantile(chart.total_length[good_traj], 0.10)) if good_traj.any() else np.nan,
        "trajectory_total_length_q90": float(np.nanquantile(chart.total_length[good_traj], 0.90)) if good_traj.any() else np.nan,
        "assignment_feature_distance_median": float(np.nanmedian(dist)) if dist.size else np.nan,
        "assignment_feature_distance_q90": float(np.nanquantile(dist, 0.90)) if dist.size else np.nan,
        "r_z_extrapolated_or_clipped_fraction": float(cell_table.loc[valid, "r_z_extrapolated_or_clipped"].mean()) if valid.any() else np.nan,
    }
    # Spearman-like rank correlation with z is just a sanity diagnostic. r need not equal z.
    if r.size > 10 and np.isfinite(z).sum() > 10:
        rz = pd.Series(r).rank(method="average").to_numpy(float)
        zz = pd.Series(z).rank(method="average").to_numpy(float)
        rz -= rz.mean(); zz -= zz.mean()
        den = float(np.sqrt((rz * rz).sum() * (zz * zz).sum()))
        summary["r_z_spearman"] = float((rz * zz).sum() / den) if den > 0 else np.nan
    else:
        summary["r_z_spearman"] = np.nan

    return DeterministicRResult(
        slice_id=int(slice_id),
        panel_label=str(result.panel_label),
        interval_a=a,
        interval_b=b,
        interval_width=width,
        chart=chart,
        cell_table=cell_table,
        trajectory_table=traj_table,
        summary=summary,
    )


def plot_deterministic_r_heatmap(
    result: DeterministicRResult,
    ax: Optional[plt.Axes] = None,
    show_colorbar: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    d = result.cell_table
    valid = d["r_valid"].to_numpy(bool)
    rng = np.random.default_rng(20260806 + int(result.slice_id))
    if len(d) > int(MAX_POINTS_PER_PLOT):
        idx = rng.choice(len(d), size=int(MAX_POINTS_PER_PLOT), replace=False)
        dp = d.iloc[idx].copy()
        valid_p = dp["r_valid"].to_numpy(bool)
    else:
        dp = d
        valid_p = valid

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_PER_SLICE, constrained_layout=True)
    else:
        fig = ax.figure

    if np.any(~valid_p):
        bad = dp.loc[~valid_p]
        ax.scatter(
            bad["x"],
            bad["y"],
            s=POINT_SIZE,
            c=INVALID_COLOR,
            alpha=0.35,
            linewidths=0,
            zorder=1,
        )

    good = dp.loc[valid_p]
    sc = ax.scatter(
        good["x"],
        good["y"],
        c=good["r_normalized"],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        linewidths=0,
        cmap=R_CMAP,
        vmin=0.0,
        vmax=1.0,
        zorder=3,
    )

    # Overlay a sparse set of deterministic depth trajectories.
    ch = result.chart
    for i in range(0, len(ch.ell_grid), max(1, int(TRAJECTORY_PLOT_STRIDE))):
        ok = np.isfinite(ch.r_normalized_grid[i]) & np.isfinite(ch.xy[i, :, 0]) & np.isfinite(ch.xy[i, :, 1])
        if ok.sum() >= 3:
            ax.plot(
                ch.xy[i, ok, 0],
                ch.xy[i, ok, 1],
                color=TRAJECTORY_COLOR,
                alpha=TRAJECTORY_ALPHA,
                linewidth=TRAJECTORY_LINEWIDTH,
                zorder=2,
            )

    ax.set_title(
        f"Slice {result.slice_id}: deterministic constant-speed r\n"
        f"valid={result.summary['r_valid_fraction']:.1%} | "
        f"traj={result.summary['n_valid_trajectories']}/{result.summary['n_ell_trajectories']} | "
        f"{result.panel_label}",
        fontsize=10,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if show_colorbar:
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="r normalized arclength")
    return fig, ax


# ------------------------------------------------------------
# Run all requested slices
# ------------------------------------------------------------

print("Deterministic Stage 1b r from master-contour-aligned circular ell")
print("Slices:", SLICE_IDS)
print("Output directory:", OUT_DIR)
print("Trajectory grid:", R_N_ELL_TRAJECTORIES, "ell x", R_N_Z_SAMPLES, "z samples")


deterministic_r_results: Dict[int, DeterministicRResult] = {}
_summary_rows: List[Dict[str, Any]] = []
_cell_frames: List[pd.DataFrame] = []
_traj_frames: List[pd.DataFrame] = []
_failures: List[Dict[str, Any]] = []

for _slice_id in SLICE_IDS:
    _slice_id = int(_slice_id)
    print("\n" + "=" * 76)
    print(f"Computing deterministic r for slice {_slice_id}")
    print("=" * 76)
    try:
        _result = compute_deterministic_r_for_slice(_slice_id)
        deterministic_r_results[_slice_id] = _result
        _summary_rows.append(_result.summary)
        _cell_frames.append(_result.cell_table)
        _traj_frames.append(_result.trajectory_table)

        if SAVE_PER_SLICE_TABLES:
            _result.cell_table.to_csv(
                TABLES_DIR / f"slice_{_slice_id:02d}_deterministic_r_cells.csv.gz",
                index=False,
                compression="gzip",
            )
        if SAVE_TRAJECTORY_TABLES:
            _result.trajectory_table.to_csv(
                TRAJECTORY_DIR / f"slice_{_slice_id:02d}_deterministic_r_trajectories.csv.gz",
                index=False,
                compression="gzip",
            )

        fig, _ = plot_deterministic_r_heatmap(_result)
        if SAVE_PNG:
            fig.savefig(
                PLOTS_DIR / f"slice_{_slice_id:02d}_deterministic_r_heatmap.png",
                dpi=DPI,
                bbox_inches="tight",
            )
        if SAVE_PDF:
            fig.savefig(
                PLOTS_DIR / f"slice_{_slice_id:02d}_deterministic_r_heatmap.pdf",
                bbox_inches="tight",
            )
        if DISPLAY_FIGURES:
            plt.show()
        else:
            plt.close(fig)

        print(
            f"slice {_slice_id}: r valid={_result.summary['r_valid_fraction']:.1%}, "
            f"valid trajectories={_result.summary['n_valid_trajectories']}/"
            f"{_result.summary['n_ell_trajectories']}, "
            f"median trajectory length={_result.summary['trajectory_total_length_median']:.4g}"
        )

    except Exception as exc:
        _failures.append(
            {
                "slice_id": int(_slice_id),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        warnings.warn(f"Slice {_slice_id} failed: {type(exc).__name__}: {exc}")
    finally:
        gc.collect()

if not deterministic_r_results:
    raise RuntimeError("No slices succeeded. Inspect _failures for details.")


deterministic_r_summary_df = (
    pd.DataFrame(_summary_rows)
    .sort_values("slice_id")
    .reset_index(drop=True)
)
deterministic_r_summary_df.to_csv(
    TABLES_DIR / "deterministic_r_summary_by_slice.csv",
    index=False,
)


deterministic_r_cells_df = (
    pd.concat(_cell_frames, ignore_index=True)
    .sort_values(["slice_id", "cell_index"])
    .reset_index(drop=True)
)
deterministic_r_trajectory_points_df = (
    pd.concat(_traj_frames, ignore_index=True)
    .sort_values(["slice_id", "trajectory_id", "z_target"])
    .reset_index(drop=True)
)

if SAVE_COMBINED_TABLES:
    deterministic_r_cells_df.to_csv(
        TABLES_DIR / "deterministic_r_all_cells.csv.gz",
        index=False,
        compression="gzip",
    )
    deterministic_r_trajectory_points_df.to_csv(
        TRAJECTORY_DIR / "deterministic_r_all_trajectories.csv.gz",
        index=False,
        compression="gzip",
    )

with open(OUT_DIR / "deterministic_r_config.json", "w") as handle:
    json.dump(
        {
            "slice_ids": [int(x) for x in SLICE_IDS],
            "r_n_ell_trajectories": int(R_N_ELL_TRAJECTORIES),
            "r_n_z_samples": int(R_N_Z_SAMPLES),
            "r_z_quantile_range": list(R_Z_QUANTILE_RANGE),
            "r_flow_substeps_per_z_interval": int(R_FLOW_SUBSTEPS_PER_Z_INTERVAL),
            "r_flow_method": str(R_FLOW_METHOD),
            "r_flow_eps": float(R_FLOW_EPS),
            "r_gradient_floor_scale": float(R_GRADIENT_FLOOR_SCALE),
            "r_max_spatial_step_grid_steps": float(R_MAX_SPATIAL_STEP_GRID_STEPS),
            "r_assign_k_nearest": int(R_ASSIGN_K_NEAREST),
            "r_assign_idw_power": float(R_ASSIGN_IDW_POWER),
            "r_assign_z_feature_weight": float(R_ASSIGN_Z_FEATURE_WEIGHT),
            "r_clip_cell_z_to_trajectory_range": bool(R_CLIP_CELL_Z_TO_TRAJECTORY_RANGE),
            "r_max_assign_feature_distance": float(R_MAX_ASSIGN_FEATURE_DISTANCE),
            "failures": _failures,
        },
        handle,
        indent=2,
    )

# Overview figure.
if MAKE_OVERVIEW_FIGURE:
    n = len(deterministic_r_results)
    ncols = max(1, min(int(OVERVIEW_NCOLS), n))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.8 * ncols, 3.4 * nrows),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    for ax, sid in zip(axes, sorted(deterministic_r_results)):
        plot_deterministic_r_heatmap(
            deterministic_r_results[sid],
            ax=ax,
            show_colorbar=False,
        )
    for ax in axes[len(deterministic_r_results):]:
        ax.axis("off")
    fig.suptitle("Deterministic Stage 1b constant-speed r", fontsize=14)
    fig.savefig(
        PLOTS_DIR / "all_slices_deterministic_r_heatmaps.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    if SAVE_PDF:
        fig.savefig(
            PLOTS_DIR / "all_slices_deterministic_r_heatmaps.pdf",
            bbox_inches="tight",
        )
    if DISPLAY_FIGURES:
        plt.show()
    else:
        plt.close(fig)

print("\nDeterministic r computation complete.")
print("Successful slices:", sorted(deterministic_r_results.keys()))
print("Failures:", _failures if _failures else "none")
print("Summary table:", TABLES_DIR / "deterministic_r_summary_by_slice.csv")
print("Cell tables:", TABLES_DIR)
print("Trajectory tables:", TRAJECTORY_DIR)
print("Plots:", PLOTS_DIR)

display(deterministic_r_summary_df)
