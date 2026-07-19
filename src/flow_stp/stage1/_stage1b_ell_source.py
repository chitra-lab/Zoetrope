# %%
# ============================================================
# Flow-STP Stage 1b: circular-smoothed master-contour-aligned ambient multi-contour ell
#
# Purpose
# -------
# Assign an ell coordinate using many Stage 1a level-set contours, while
# forcing every contour's ell coordinate to be inherited from ONE master
# reference contour. This version also treats ell as circular through
# (cos(2*pi*ell), sin(2*pi*ell)) and replaces hard best-contour assignment
# by a score-weighted circular average over the top candidate assignments.
#
# Geometry
# --------
# 1. Evaluate frozen Stage 1a z on a padded ambient raw-XY grid.
# 2. Extract many ambient z-level contours WITHOUT tissue masking.
# 3. Choose one master contour C0, usually near the median cell z.
# 4. Parameterize C0 by arclength ell in [0,1].
# 5. For every point q on every other contour Ck:
#        flow q from z_k to z0 under dx/dz = grad z / (||grad z||^2 + eps)
#        project the endpoint to C0
#        inherit ell(q) from C0.
# 6. For each cell x:
#        try nearby z-level contours
#        flow x to that z-level
#        project to that contour
#        interpolate the inherited master ell on that contour
#        collect the top-scoring candidates
#        return a score-weighted circular mean ell.
#
# This cell does NOT learn r and does NOT train Phi/Psi/L. It computes a
# deterministic, globally-aligned pseudo-label ell field intended for
# diagnostics and later supervision of L(x), Phi(ell,z), or both.
#
# Expected notebook variables
# ---------------------------
#   model_flow_stp
#   spatial_by_slice
#
# Optional notebook variables
# ---------------------------
#   xy_norm_params   # preferred, from Stage 1a loader
#   panel_df         # used only for plot labels
#
# Main in-memory outputs
# ----------------------
#   master_contour_aligned_circular_ell_results
#   master_contour_aligned_circular_ell_cells_df
#   master_contour_aligned_circular_ell_summary_df
#
# Files written
# -------------
#   stage1b_master_contour_aligned_circular_ell/
#       plots/slice_XX_master_contour_aligned_circular_ell_heatmap.png
#       plots/all_slices_master_contour_aligned_circular_ell_heatmaps.png
#       plots/slice_XX_ambient_z_master_aligned_contours.png
#       tables/slice_XX_master_contour_aligned_circular_ell_cells.csv.gz
#       tables/master_contour_aligned_circular_ell_all_cells.csv.gz
#       tables/master_contour_aligned_circular_ell_summary_by_slice.csv
#       contours/slice_XX_master_aligned_contours.csv.gz
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import gc
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    distance_transform_edt,
    label as connected_components,
)
from scipy.spatial import cKDTree
from skimage.measure import find_contours


# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------

if "spatial_by_slice" not in globals():
    raise NameError("spatial_by_slice is missing. Run the PFC loader first.")
if "model_flow_stp" not in globals():
    raise NameError("model_flow_stp is missing. Run the Stage 1a coordinate-only loader first.")

SLICE_IDS = list(range(len(spatial_by_slice)))

OUT_DIR = Path.cwd() / "stage1b_master_contour_aligned_circular_ell"
PLOTS_DIR = OUT_DIR / "plots"
TABLES_DIR = OUT_DIR / "tables"
CONTOURS_DIR = OUT_DIR / "contours"
for _d in [OUT_DIR, PLOTS_DIR, TABLES_DIR, CONTOURS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Ambient Stage 1a grid.
GRID_MAX_DIM = 360
AMBIENT_GRID_PADDING_FRACTION = 0.20
GRID_EVAL_BATCH_SIZE = 65536
CELL_Z_EVAL_BATCH_SIZE = 65536

# Tissue mask is NOT used for contour extraction or backflow validity.
# It is used only to score/select contour components and for diagnostics.
TISSUE_MASK_SPACING_QUANTILE = 0.95
TISSUE_MASK_RADIUS_FACTOR = 2.75
TISSUE_MASK_CLOSING_ITERATIONS = 2

# Multi-contour levels.
N_REFERENCE_LEVELS = 100
LEVEL_QUANTILE_RANGE = (0.01, 0.99)
MASTER_LEVEL_QUANTILE = 0.50
MIN_CONTOUR_POINTS = 25
CONTOUR_RESAMPLE_N = 2048
CONTOUR_ORIENT_BY = "x"  # "x", "pca", or "keep"
KEEP_ALL_AMBIENT_COMPONENTS = False
# False selects one best ambient contour component per z level. True aligns
# all components to the master contour; useful for diagnostics but may still
# create multi-branch ambiguity.

# Select best contour component by tissue contact + length.
CONTOUR_SCORE_NEAR_CELL_GRID_STEPS = 5.0
CONTOUR_SCORE_MAX_POINTS = 800

# Backflow integration: dx/dz = grad z / (||grad z||^2 + eps).
BACKFLOW_N_STEPS = 64
BACKFLOW_METHOD = "rk2"  # "euler" or "rk2"
FLOW_EPS = 1e-10
GRADIENT_FLOOR_QUANTILE = 0.001
ABSOLUTE_GRADIENT_FLOOR = 0.0
MAX_VECTOR_NORM_GRID_STEPS = 10.0
BACKFLOW_CHUNK_SIZE = 60_000
BACKFLOW_Z_TOL_FRACTION = 0.05
CONTOUR_ALIGNMENT_Z_TOL_FRACTION = 0.08

# For each cell, try the nearest z-level contours and keep the best.
N_NEAREST_LEVELS_TO_TRY = 5
PROJECTION_K_SEGMENTS = 16
MAX_PROJECTION_DISTANCE_GRID_STEPS = 8.0
SCORE_Z_ERROR_WEIGHT = 2.0
SCORE_TRAVEL_WEIGHT = 0.02
SCORE_ALIGNMENT_CONFIDENCE_WEIGHT = 0.50

# Seam-safe ell handling and soft candidate assignment.
# The inherited ell lives on a circle, so 0.99 and 0.01 should be close.
# We therefore interpolate and average ell through (cos(2*pi*ell), sin(2*pi*ell)).
USE_CIRCULAR_ELL_EMBEDDING = True
CIRCULAR_AVERAGE_TOP_K = 8
CIRCULAR_AVERAGE_TEMPERATURE = 2.0
CIRCULAR_AVERAGE_MAX_SCORE_GAP = 8.0

# Plotting.
DISPLAY_FIGURES = True
SAVE_PNG = True
SAVE_PDF = False
DPI = 300
MAX_POINTS_PER_PLOT = 140_000
POINT_SIZE = 0.7
POINT_ALPHA = 0.86
ELL_CMAP = "turbo"
INVALID_COLOR = "0.80"
CONTOUR_COLOR = "black"
CONTOUR_ALPHA = 0.18
CONTOUR_LINEWIDTH = 0.50
MASTER_CONTOUR_COLOR = "black"
MASTER_CONTOUR_LINEWIDTH = 1.50
CONTOUR_PLOT_STRIDE = 5
FIGSIZE_PER_SLICE = (6.4, 5.6)
MAKE_OVERVIEW_FIGURE = True
OVERVIEW_NCOLS = 5

# Storage.
SAVE_PER_SLICE_TABLES = True
SAVE_COMBINED_TABLE = True
SAVE_CONTOUR_TABLES = True


# ------------------------------------------------------------
# Small utilities
# ------------------------------------------------------------

ArrayLike = Union[np.ndarray, torch.Tensor]
SpatialCollection = Union[Sequence[ArrayLike], Mapping[int, ArrayLike]]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_float_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(_to_numpy(x), dtype=np.float64)


def _get_slice_xy(spatial_by_slice: SpatialCollection, slice_id: int) -> np.ndarray:
    xy = _as_float_array(spatial_by_slice[int(slice_id)])
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"spatial_by_slice[{slice_id}] must have shape [N,2+], got {xy.shape}")
    xy = xy[:, :2]
    good = np.isfinite(xy).all(axis=1)
    if not np.all(good):
        warnings.warn(f"Slice {slice_id}: dropping {(~good).sum():,} non-finite XY rows.")
        xy = xy[good]
    return xy


def _get_xy_norm_for_slice(xy_normalization: Optional[Any], slice_id: int, raw_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if xy_normalization is None:
        center = raw_xy.mean(axis=0)
        scale = raw_xy.std(axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        return center.astype(np.float64), scale.astype(np.float64)

    if isinstance(xy_normalization, pd.DataFrame):
        df = xy_normalization
        row = None
        if "slice_id" in df.columns:
            hit = df[df["slice_id"].astype(int) == int(slice_id)]
            if len(hit) > 0:
                row = hit.iloc[0]
        elif len(df) > int(slice_id):
            row = df.iloc[int(slice_id)]

        if row is not None:
            center_pairs = [
                ("center_x", "center_y"),
                ("x_center", "y_center"),
                ("mean_x", "mean_y"),
                ("x_mean", "y_mean"),
                ("mu_x", "mu_y"),
                ("x_mu", "y_mu"),
            ]
            scale_pairs = [
                ("scale_x", "scale_y"),
                ("x_scale", "y_scale"),
                ("std_x", "std_y"),
                ("x_std", "y_std"),
                ("sigma_x", "sigma_y"),
                ("x_sigma", "y_sigma"),
            ]
            center = None
            scale = None
            for a, b in center_pairs:
                if a in row.index and b in row.index:
                    center = np.array([row[a], row[b]], dtype=np.float64)
                    break
            for a, b in scale_pairs:
                if a in row.index and b in row.index:
                    scale = np.array([row[a], row[b]], dtype=np.float64)
                    break
            if center is not None and scale is not None:
                scale = np.where(scale > 1e-6, scale, 1.0)
                return center, scale

    if isinstance(xy_normalization, Mapping):
        obj = xy_normalization
        if slice_id in obj:
            obj = obj[slice_id]
        elif str(slice_id) in obj:
            obj = obj[str(slice_id)]
        if isinstance(obj, Mapping):
            center = None
            scale = None
            for k in ["center", "mean", "mu", "xy_center", "xy_mean"]:
                if k in obj:
                    center = np.asarray(obj[k], dtype=np.float64).reshape(-1)[:2]
                    break
            for k in ["scale", "std", "sigma", "xy_scale", "xy_std"]:
                if k in obj:
                    scale = np.asarray(obj[k], dtype=np.float64).reshape(-1)[:2]
                    break
            if center is not None and scale is not None:
                scale = np.where(scale > 1e-6, scale, 1.0)
                return center, scale

    warnings.warn("Could not parse xy_norm_params; using per-slice raw XY mean/std.")
    center = raw_xy.mean(axis=0)
    scale = raw_xy.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return center.astype(np.float64), scale.astype(np.float64)


def _freeze_model(model: nn.Module) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _call_stage1a_alpha(model: nn.Module, xy_stage_norm: torch.Tensor, slice_id: int) -> torch.Tensor:
    sid_vec = torch.full(
        (xy_stage_norm.shape[0],),
        int(slice_id),
        dtype=torch.long,
        device=xy_stage_norm.device,
    )
    if hasattr(model, "coord_maps"):
        coord_maps = getattr(model, "coord_maps")
        try:
            out = coord_maps(xy_stage_norm, sid_vec)
        except Exception:
            out = coord_maps(xy_stage_norm, int(slice_id))
    else:
        try:
            out = model(xy_stage_norm, sid_vec)
        except Exception:
            out = model(xy_stage_norm, int(slice_id))

    if isinstance(out, Mapping):
        for k in ["alpha", "z", "coord", "r"]:
            if k in out:
                out = out[k]
                break
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.reshape(-1)


def _panel_label(slice_id: int) -> str:
    if "panel_df" in globals():
        try:
            row = panel_df.loc[panel_df["slice_id"].astype(int) == int(slice_id)].iloc[0]
            if "panel_key" in row.index:
                return str(row["panel_key"])
        except Exception:
            pass
    return f"slice {int(slice_id)}"


def _polyline_arclength(points: np.ndarray) -> Tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return np.zeros(len(points), dtype=np.float64), 0.0
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    return s, float(s[-1])


def _remove_duplicate_and_closing(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return points
    keep = np.concatenate([[True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12])
    points = points[keep]
    if len(points) > 3 and np.linalg.norm(points[0] - points[-1]) < 1e-9:
        points = points[:-1]
    return points


def _orient_contour(points: np.ndarray, orient_by: str = "x") -> np.ndarray:
    points = _remove_duplicate_and_closing(points)
    if len(points) < 3 or str(orient_by).lower() == "keep":
        return points.copy()

    orient_by = str(orient_by).lower()
    if orient_by == "pca":
        centered = points - points.mean(axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            score = centered @ vt[0]
        except Exception:
            score = points[:, 0]
    else:
        score = points[:, 0]

    if float(score[-1]) < float(score[0]):
        return points[::-1].copy()
    return points.copy()


def _resample_polyline(points: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    points = _remove_duplicate_and_closing(points)
    if len(points) < 2:
        raise ValueError("Need at least two contour points.")
    s, total = _polyline_arclength(points)
    if total <= 0:
        raise ValueError("Contour has zero arclength.")
    s_new = np.linspace(0.0, total, int(n), dtype=np.float64)
    x = np.interp(s_new, s, points[:, 0])
    y = np.interp(s_new, s, points[:, 1])
    ell = s_new / total
    return ell, np.column_stack([x, y])


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    labels, n = connected_components(mask)
    if n <= 1:
        return mask.astype(bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def _fill_nan_along_polyline(values: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fill NaNs in a 1D contour value array by index interpolation.

    Returns filled values and the fraction originally finite.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(v)
    frac = float(np.mean(finite)) if len(v) else 0.0
    if len(v) == 0:
        return v, frac
    if finite.all():
        return v, frac
    if finite.sum() == 0:
        # Last-resort fallback: local arclength. This should be rare and is
        # reported through contour alignment_valid_fraction.
        return np.linspace(0.0, 1.0, len(v), dtype=np.float64), 0.0
    idx = np.arange(len(v), dtype=np.float64)
    filled = np.interp(idx, idx[finite], v[finite])
    return np.clip(filled, 0.0, 1.0), frac


# ------------------------------------------------------------
# Data containers
# ------------------------------------------------------------

@dataclass
class AmbientGrid:
    slice_id: int
    panel_label: str
    raw_xy: np.ndarray
    x_values: np.ndarray
    y_values: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    z_grid: np.ndarray
    z_cell: np.ndarray
    mask: np.ndarray
    grad_x: np.ndarray
    grad_y: np.ndarray
    grad_magnitude: np.ndarray
    gradient_floor: float
    grid_step: float
    z_range_masked: float
    z_range_cell: float
    levels: np.ndarray
    master_level_index: int
    master_z_level: float
    interpolators: Dict[str, RegularGridInterpolator]


@dataclass
class AmbientContour:
    slice_id: int
    level_index: int
    z_level: float
    component_index: int
    local_ell: np.ndarray
    inherited_ell: np.ndarray
    xy: np.ndarray
    length: float
    tissue_contact_fraction: float
    score: float
    alignment_valid_fraction: float
    is_master: bool = False


@dataclass
class MasterContourAlignedEllResult:
    slice_id: int
    panel_label: str
    grid: AmbientGrid
    master_contour: AmbientContour
    contours_by_level: Dict[int, List[AmbientContour]]
    cell_table: pd.DataFrame
    summary: Dict[str, Any]


# ------------------------------------------------------------
# Stage 1a ambient grid evaluation
# ------------------------------------------------------------

def evaluate_stage1a_on_points(
    model: nn.Module,
    raw_points: np.ndarray,
    slice_id: int,
    stage_center: np.ndarray,
    stage_scale: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    raw_points = np.asarray(raw_points, dtype=np.float64)
    out_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(raw_points), int(batch_size)):
            end = min(start + int(batch_size), len(raw_points))
            xy_stage = (raw_points[start:end] - stage_center.reshape(1, 2)) / stage_scale.reshape(1, 2)
            xy_t = torch.tensor(xy_stage, dtype=torch.float32, device=device)
            z = _call_stage1a_alpha(model, xy_t, slice_id)
            out_chunks.append(z.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(out_chunks, axis=0)


def build_tissue_mask(raw_xy: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> Tuple[np.ndarray, float]:
    tree = cKDTree(raw_xy)
    nn_distance, _ = tree.query(raw_xy, k=min(2, len(raw_xy)))
    if nn_distance.ndim == 2 and nn_distance.shape[1] > 1:
        spacing = nn_distance[:, 1]
    else:
        spacing = nn_distance
    spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
    if spacing.size == 0:
        radius = 1.0
    else:
        radius = float(TISSUE_MASK_RADIUS_FACTOR) * float(np.quantile(spacing, float(TISSUE_MASK_SPACING_QUANTILE)))
    radius = max(radius, 1e-9)
    pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    d, _ = tree.query(pts, k=1)
    mask = (d.reshape(grid_x.shape) <= radius)
    if int(TISSUE_MASK_CLOSING_ITERATIONS) > 0:
        mask = binary_closing(mask, iterations=int(TISSUE_MASK_CLOSING_ITERATIONS))
    mask = binary_fill_holes(mask)
    mask = _largest_connected_component(mask)
    return mask.astype(bool), radius


def _make_interpolator(x_values: np.ndarray, y_values: np.ndarray, values: np.ndarray, fill_value: float = np.nan) -> RegularGridInterpolator:
    return RegularGridInterpolator(
        (np.asarray(y_values, dtype=np.float64), np.asarray(x_values, dtype=np.float64)),
        np.asarray(values, dtype=np.float64),
        bounds_error=False,
        fill_value=fill_value,
    )


def build_ambient_grid(slice_id: int) -> AmbientGrid:
    device = _model_device(model_flow_stp)
    _freeze_model(model_flow_stp)

    raw_xy = _get_slice_xy(spatial_by_slice, slice_id)
    stage_center, stage_scale = _get_xy_norm_for_slice(
        globals().get("xy_norm_params", None),
        slice_id,
        raw_xy,
    )

    panel_label = _panel_label(slice_id)

    z_cell = evaluate_stage1a_on_points(
        model=model_flow_stp,
        raw_points=raw_xy,
        slice_id=slice_id,
        stage_center=stage_center,
        stage_scale=stage_scale,
        device=device,
        batch_size=CELL_Z_EVAL_BATCH_SIZE,
    )

    xmin, ymin = raw_xy.min(axis=0)
    xmax, ymax = raw_xy.max(axis=0)
    width = max(float(xmax - xmin), 1e-9)
    height = max(float(ymax - ymin), 1e-9)
    pad = float(AMBIENT_GRID_PADDING_FRACTION)
    x0 = xmin - pad * width
    x1 = xmax + pad * width
    y0 = ymin - pad * height
    y1 = ymax + pad * height

    if width >= height:
        nx = int(GRID_MAX_DIM)
        ny = max(32, int(round(float(GRID_MAX_DIM) * height / width)))
    else:
        ny = int(GRID_MAX_DIM)
        nx = max(32, int(round(float(GRID_MAX_DIM) * width / height)))

    x_values = np.linspace(x0, x1, nx, dtype=np.float64)
    y_values = np.linspace(y0, y1, ny, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="xy")
    raw_grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    z_grid = evaluate_stage1a_on_points(
        model=model_flow_stp,
        raw_points=raw_grid_points,
        slice_id=slice_id,
        stage_center=stage_center,
        stage_scale=stage_scale,
        device=device,
        batch_size=GRID_EVAL_BATCH_SIZE,
    ).reshape(grid_x.shape)

    mask, _ = build_tissue_mask(raw_xy, grid_x, grid_y)

    grad_y, grad_x = np.gradient(z_grid, y_values, x_values, edge_order=1)
    grad_magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    dx = float(np.nanmedian(np.diff(x_values)))
    dy = float(np.nanmedian(np.diff(y_values)))
    grid_step = float(min(abs(dx), abs(dy)))

    good_grad = grad_magnitude[np.isfinite(grad_magnitude) & (grad_magnitude > 0)]
    if good_grad.size == 0:
        raise ValueError(f"Slice {slice_id}: no positive finite gradients on ambient grid.")
    gradient_floor = max(
        float(ABSOLUTE_GRADIENT_FLOOR),
        float(np.nanquantile(good_grad, float(GRADIENT_FLOOR_QUANTILE))),
    )

    z_masked = z_grid[mask & np.isfinite(z_grid)]
    z_range_masked = float(np.nanmax(z_masked) - np.nanmin(z_masked)) if z_masked.size else float(np.nanmax(z_grid) - np.nanmin(z_grid))
    z_range_masked = max(z_range_masked, 1e-12)
    z_range_cell = float(np.nanmax(z_cell) - np.nanmin(z_cell))
    z_range_cell = max(z_range_cell, 1e-12)

    q0, q1 = LEVEL_QUANTILE_RANGE
    base_levels = np.quantile(z_cell[np.isfinite(z_cell)], np.linspace(float(q0), float(q1), int(N_REFERENCE_LEVELS)))
    master_z = float(np.quantile(z_cell[np.isfinite(z_cell)], float(MASTER_LEVEL_QUANTILE)))
    levels = np.unique(np.concatenate([base_levels.astype(np.float64), np.array([master_z], dtype=np.float64)]))
    if len(levels) < 2:
        raise ValueError(f"Slice {slice_id}: fewer than two unique reference z levels.")
    master_level_index = int(np.argmin(np.abs(levels - master_z)))
    master_z_level = float(levels[master_level_index])

    interpolators = {
        "z": _make_interpolator(x_values, y_values, z_grid, fill_value=np.nan),
        "grad_x": _make_interpolator(x_values, y_values, grad_x, fill_value=np.nan),
        "grad_y": _make_interpolator(x_values, y_values, grad_y, fill_value=np.nan),
        "grad_magnitude": _make_interpolator(x_values, y_values, grad_magnitude, fill_value=np.nan),
    }

    return AmbientGrid(
        slice_id=int(slice_id),
        panel_label=panel_label,
        raw_xy=raw_xy.astype(np.float64),
        x_values=x_values,
        y_values=y_values,
        grid_x=grid_x,
        grid_y=grid_y,
        z_grid=z_grid.astype(np.float64),
        z_cell=z_cell.astype(np.float64),
        mask=mask.astype(bool),
        grad_x=grad_x.astype(np.float64),
        grad_y=grad_y.astype(np.float64),
        grad_magnitude=grad_magnitude.astype(np.float64),
        gradient_floor=float(gradient_floor),
        grid_step=float(grid_step),
        z_range_masked=float(z_range_masked),
        z_range_cell=float(z_range_cell),
        levels=levels.astype(np.float64),
        master_level_index=int(master_level_index),
        master_z_level=float(master_z_level),
        interpolators=interpolators,
    )


# ------------------------------------------------------------
# Ambient backflow and projection helpers
# ------------------------------------------------------------

def _interp_field(grid: AmbientGrid, name: str, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    yx = np.column_stack([xy[:, 1], xy[:, 0]])
    return np.asarray(grid.interpolators[name](yx), dtype=np.float64).reshape(-1)


def _inside_ambient_box(grid: AmbientGrid, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return (
        np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= grid.x_values[0])
        & (xy[:, 0] <= grid.x_values[-1])
        & (xy[:, 1] >= grid.y_values[0])
        & (xy[:, 1] <= grid.y_values[-1])
    )


def _vector_field_dx_dz(grid: AmbientGrid, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    gx = _interp_field(grid, "grad_x", xy)
    gy = _interp_field(grid, "grad_y", xy)
    gm = np.sqrt(gx * gx + gy * gy)

    valid = (
        _inside_ambient_box(grid, xy)
        & np.isfinite(gx)
        & np.isfinite(gy)
        & np.isfinite(gm)
        & (gm > float(grid.gradient_floor))
    )

    denom = gm * gm + float(FLOW_EPS)
    v = np.column_stack([gx / denom, gy / denom])
    v[~valid] = np.nan

    approx_dz_step = float(grid.z_range_cell) / max(int(BACKFLOW_N_STEPS), 1)
    max_norm = float(MAX_VECTOR_NORM_GRID_STEPS) * float(grid.grid_step) / max(approx_dz_step, 1e-12)
    norm = np.linalg.norm(v, axis=1)
    too_large = np.isfinite(norm) & (norm > max_norm)
    if np.any(too_large):
        v[too_large] *= (max_norm / norm[too_large])[:, None]

    return v, gm, valid


def backflow_points_to_levels(
    grid: AmbientGrid,
    xy_start: np.ndarray,
    z_start: np.ndarray,
    z_target: np.ndarray,
    *,
    z_tol_fraction: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate each point from z_start to a possibly point-specific z_target."""
    x = np.asarray(xy_start, dtype=np.float64).copy()
    z_start = np.asarray(z_start, dtype=np.float64).reshape(-1)
    z_target = np.asarray(z_target, dtype=np.float64).reshape(-1)
    if len(x) != len(z_start) or len(x) != len(z_target):
        raise ValueError("xy_start, z_start, and z_target lengths do not match.")

    valid = _inside_ambient_box(grid, x) & np.isfinite(z_start) & np.isfinite(z_target)
    dz_total = (z_target - z_start).reshape(-1, 1)
    dz = dz_total / float(max(int(BACKFLOW_N_STEPS), 1))
    method = str(BACKFLOW_METHOD).lower()

    for _ in range(int(BACKFLOW_N_STEPS)):
        active = np.where(valid)[0]
        if len(active) == 0:
            break
        xa = x[active]
        dza = dz[active]

        v1, _, ok1 = _vector_field_dx_dz(grid, xa)
        step_ok = ok1 & np.isfinite(v1).all(axis=1)

        if method == "rk2":
            x_mid = xa + 0.5 * dza * v1
            v2, _, ok2 = _vector_field_dx_dz(grid, x_mid)
            step_ok = step_ok & ok2 & np.isfinite(v2).all(axis=1)
            x_new = xa + dza * v2
        else:
            x_new = xa + dza * v1

        z_new = _interp_field(grid, "z", x_new)
        step_ok = step_ok & _inside_ambient_box(grid, x_new) & np.isfinite(z_new)

        x[active[step_ok]] = x_new[step_ok]
        valid[active[~step_ok]] = False

    z_end = _interp_field(grid, "z", x)
    z_error = z_end - z_target
    if z_tol_fraction is None:
        z_tol_fraction = float(BACKFLOW_Z_TOL_FRACTION)
    z_tol = float(z_tol_fraction) * float(grid.z_range_cell)
    valid = valid & np.isfinite(z_end) & (np.abs(z_error) <= z_tol)
    return x, z_end, z_error, valid



def _ell_to_angle(ell: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * np.asarray(ell, dtype=np.float64)


def _uv_to_ell(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    ell = np.arctan2(v, u) / (2.0 * np.pi)
    ell = np.mod(ell, 1.0)
    return ell.astype(np.float64)


def _circular_interpolate_ell(ell0: np.ndarray, ell1: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate ell values on S^1 and return ell, cos, sin."""
    ell0 = np.asarray(ell0, dtype=np.float64)
    ell1 = np.asarray(ell1, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    a0 = _ell_to_angle(ell0)
    a1 = _ell_to_angle(ell1)
    c = (1.0 - u) * np.cos(a0) + u * np.cos(a1)
    s = (1.0 - u) * np.sin(a0) + u * np.sin(a1)
    r = np.sqrt(c * c + s * s)
    safe = r > 1e-12
    c_norm = np.full_like(c, np.nan, dtype=np.float64)
    s_norm = np.full_like(s, np.nan, dtype=np.float64)
    c_norm[safe] = c[safe] / r[safe]
    s_norm[safe] = s[safe] / r[safe]
    ell = _uv_to_ell(c_norm, s_norm)
    return ell, c_norm, s_norm


def _weighted_circular_mean_ell(ell: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score-weighted circular mean over candidate ell assignments.

    Args:
        ell: [N,K] candidate ell values.
        score: [N,K] lower-is-better assignment scores.

    Returns:
        ell_mean: [N] circular mean in [0,1].
        u_mean: [N] normalized cos component.
        v_mean: [N] normalized sin component.
        resultant: [N] mean resultant length, confidence/consistency in [0,1].
        n_used: [N] number of candidates used in the weighted average.
    """
    ell = np.asarray(ell, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    finite = np.isfinite(ell) & np.isfinite(score)

    min_score = np.min(np.where(finite, score, np.inf), axis=1)
    has_any = np.isfinite(min_score)

    use = finite.copy()
    if CIRCULAR_AVERAGE_MAX_SCORE_GAP is not None:
        use &= score <= (min_score[:, None] + float(CIRCULAR_AVERAGE_MAX_SCORE_GAP))

    tau = max(float(CIRCULAR_AVERAGE_TEMPERATURE), 1e-12)
    weights = np.zeros_like(score, dtype=np.float64)
    delta = score - min_score[:, None]
    weights[use] = np.exp(-delta[use] / tau)

    ell_safe = np.where(finite, ell, 0.0)
    angles = _ell_to_angle(ell_safe)
    sum_w = np.sum(weights, axis=1)
    sum_c = np.sum(weights * np.cos(angles), axis=1)
    sum_s = np.sum(weights * np.sin(angles), axis=1)

    u_mean = np.full(ell.shape[0], np.nan, dtype=np.float64)
    v_mean = np.full(ell.shape[0], np.nan, dtype=np.float64)
    resultant = np.full(ell.shape[0], np.nan, dtype=np.float64)
    ell_mean = np.full(ell.shape[0], np.nan, dtype=np.float64)
    n_used = np.sum(use, axis=1).astype(float)

    ok = has_any & (sum_w > 0)
    c_bar = np.zeros_like(sum_w)
    s_bar = np.zeros_like(sum_w)
    c_bar[ok] = sum_c[ok] / sum_w[ok]
    s_bar[ok] = sum_s[ok] / sum_w[ok]
    r = np.sqrt(c_bar * c_bar + s_bar * s_bar)
    ok &= r > 1e-12

    u_mean[ok] = c_bar[ok] / r[ok]
    v_mean[ok] = s_bar[ok] / r[ok]
    resultant[ok] = r[ok]
    ell_mean[ok] = _uv_to_ell(u_mean[ok], v_mean[ok])
    return ell_mean, u_mean, v_mean, resultant, n_used


def _insert_top_candidate(
    top_score: np.ndarray,
    top_ell: np.ndarray,
    top_level_index: np.ndarray,
    top_component_index: np.ndarray,
    top_z_level: np.ndarray,
    top_xy_back: np.ndarray,
    top_z_back: np.ndarray,
    top_z_error: np.ndarray,
    top_proj_xy: np.ndarray,
    top_proj_dist: np.ndarray,
    top_segment_index: np.ndarray,
    top_backflow_valid: np.ndarray,
    top_projection_valid: np.ndarray,
    top_alignment_valid_fraction: np.ndarray,
    global_idx: np.ndarray,
    score: np.ndarray,
    ell: np.ndarray,
    level_index: int,
    component_index: int,
    z_level: float,
    xy_back: np.ndarray,
    z_back: np.ndarray,
    z_error: np.ndarray,
    proj_xy: np.ndarray,
    proj_dist: np.ndarray,
    segment_index: np.ndarray,
    backflow_valid: np.ndarray,
    projection_valid: np.ndarray,
    alignment_valid_fraction: float,
) -> None:
    good = np.isfinite(score) & np.isfinite(ell)
    if not np.any(good):
        return
    g = np.asarray(global_idx, dtype=int)[good]
    local = np.where(good)[0]
    if len(g) == 0:
        return
    worst_slot = np.argmax(top_score[g], axis=1)
    worst_score = top_score[g, worst_slot]
    improve = score[good] < worst_score
    if not np.any(improve):
        return
    g2 = g[improve]
    slot = worst_slot[improve]
    l2 = local[improve]

    top_score[g2, slot] = score[l2]
    top_ell[g2, slot] = ell[l2]
    top_level_index[g2, slot] = int(level_index)
    top_component_index[g2, slot] = int(component_index)
    top_z_level[g2, slot] = float(z_level)
    top_xy_back[g2, slot, :] = xy_back[l2, :]
    top_z_back[g2, slot] = z_back[l2]
    top_z_error[g2, slot] = z_error[l2]
    top_proj_xy[g2, slot, :] = proj_xy[l2, :]
    top_proj_dist[g2, slot] = proj_dist[l2]
    top_segment_index[g2, slot] = segment_index[l2]
    top_backflow_valid[g2, slot] = backflow_valid[l2]
    top_projection_valid[g2, slot] = projection_valid[l2]
    top_alignment_valid_fraction[g2, slot] = float(alignment_valid_fraction)

def project_points_to_contour_values(
    points: np.ndarray,
    contour_xy: np.ndarray,
    contour_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project points to a polyline and interpolate a scalar value on it.

    Returns:
        value: interpolated contour value at projection
        distance: Euclidean projection distance
        proj_xy: projected point
        segment_index: selected segment
        segment_u: local coordinate in selected segment [0,1]
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    contour = np.asarray(contour_xy, dtype=np.float64)
    values = np.asarray(contour_values, dtype=np.float64).reshape(-1)
    if len(contour) < 2 or len(values) != len(contour):
        raise ValueError("Need contour_xy and contour_values with matching length >= 2.")

    a = contour[:-1]
    b = contour[1:]
    ab = b - a
    denom = np.sum(ab * ab, axis=1)
    mid = 0.5 * (a + b)
    tree = cKDTree(mid)

    k = min(int(PROJECTION_K_SEGMENTS), len(mid))
    _, cand_idx = tree.query(points, k=k)
    if k == 1:
        cand_idx = cand_idx[:, None]

    a_c = a[cand_idx]
    ab_c = ab[cand_idx]
    denom_c = np.sum(ab_c * ab_c, axis=2)
    denom_c = np.where(denom_c > 1e-20, denom_c, 1.0)
    pt = points[:, None, :]
    u = np.sum((pt - a_c) * ab_c, axis=2) / denom_c
    u = np.clip(u, 0.0, 1.0)
    proj = a_c + u[:, :, None] * ab_c
    dist = np.linalg.norm(pt - proj, axis=2)

    best_local = np.argmin(dist, axis=1)
    row = np.arange(len(points))
    best_seg = cand_idx[row, best_local]
    best_u = u[row, best_local]
    best_proj = proj[row, best_local, :]
    best_dist = dist[row, best_local]
    if bool(USE_CIRCULAR_ELL_EMBEDDING):
        value, _, _ = _circular_interpolate_ell(values[best_seg], values[best_seg + 1], best_u)
    else:
        value = (1.0 - best_u) * values[best_seg] + best_u * values[best_seg + 1]
        value = np.clip(value, 0.0, 1.0)
    return (
        value.astype(np.float64),
        best_dist.astype(np.float64),
        best_proj.astype(np.float64),
        best_seg.astype(int),
        best_u.astype(np.float64),
    )


def _nearest_level_indices(levels: np.ndarray, z_values: np.ndarray, k: int) -> np.ndarray:
    levels = np.asarray(levels, dtype=np.float64).reshape(1, -1)
    z_values = np.asarray(z_values, dtype=np.float64).reshape(-1, 1)
    dist = np.abs(levels - z_values)
    k = min(int(k), levels.shape[1])
    return np.argpartition(dist, kth=k-1, axis=1)[:, :k]


# ------------------------------------------------------------
# Ambient contour extraction and master alignment
# ------------------------------------------------------------

def _contour_component_score(contour_xy: np.ndarray, raw_xy_tree: cKDTree, near_radius: float) -> Tuple[float, float]:
    contour_xy = np.asarray(contour_xy, dtype=np.float64)
    if len(contour_xy) == 0:
        return 0.0, 0.0
    if len(contour_xy) > int(CONTOUR_SCORE_MAX_POINTS):
        idx = np.linspace(0, len(contour_xy) - 1, int(CONTOUR_SCORE_MAX_POINTS)).round().astype(int)
        pts = contour_xy[idx]
    else:
        pts = contour_xy
    d, _ = raw_xy_tree.query(pts, k=1)
    contact = float(np.mean(np.asarray(d) <= float(near_radius)))
    _, length = _polyline_arclength(contour_xy)
    score = float(length) * (0.05 + contact) + float(len(pts)) * contact
    return contact, score


def extract_raw_ambient_contours(grid: AmbientGrid) -> Dict[int, List[AmbientContour]]:
    row_axis = np.arange(len(grid.y_values), dtype=np.float64)
    col_axis = np.arange(len(grid.x_values), dtype=np.float64)
    raw_xy_tree = cKDTree(grid.raw_xy)
    near_radius = float(CONTOUR_SCORE_NEAR_CELL_GRID_STEPS) * float(grid.grid_step)

    out: Dict[int, List[AmbientContour]] = {}

    zmin = float(np.nanmin(grid.z_grid))
    zmax = float(np.nanmax(grid.z_grid))

    for li, level in enumerate(grid.levels):
        level = float(level)
        if not (zmin <= level <= zmax):
            out[li] = []
            continue

        contours_rc = find_contours(grid.z_grid, level=level, mask=None)
        candidates: List[Tuple[float, AmbientContour]] = []

        for ci, contour in enumerate(contours_rc):
            if len(contour) < int(MIN_CONTOUR_POINTS):
                continue
            x = np.interp(contour[:, 1], col_axis, grid.x_values)
            y = np.interp(contour[:, 0], row_axis, grid.y_values)
            xy = np.column_stack([x, y])
            xy = _orient_contour(xy, CONTOUR_ORIENT_BY)
            try:
                local_ell, xy_res = _resample_polyline(xy, int(CONTOUR_RESAMPLE_N))
            except Exception:
                continue
            _, length = _polyline_arclength(xy_res)
            if length <= 0:
                continue
            contact, score = _contour_component_score(xy_res, raw_xy_tree, near_radius)
            ambient = AmbientContour(
                slice_id=int(grid.slice_id),
                level_index=int(li),
                z_level=float(level),
                component_index=int(ci),
                local_ell=local_ell.astype(np.float64),
                inherited_ell=np.full_like(local_ell, np.nan, dtype=np.float64),
                xy=xy_res.astype(np.float64),
                length=float(length),
                tissue_contact_fraction=float(contact),
                score=float(score),
                alignment_valid_fraction=0.0,
                is_master=False,
            )
            candidates.append((float(score), ambient))

        candidates.sort(key=lambda x: x[0], reverse=True)
        if bool(KEEP_ALL_AMBIENT_COMPONENTS):
            out[li] = [c[1] for c in candidates]
        else:
            out[li] = [candidates[0][1]] if candidates else []

    return out


def choose_master_contour(grid: AmbientGrid, contours_by_level: Dict[int, List[AmbientContour]]) -> AmbientContour:
    """Choose the master contour near MASTER_LEVEL_QUANTILE with available contours."""
    preferred = int(grid.master_level_index)
    available = [li for li, contours in contours_by_level.items() if len(contours) > 0]
    if not available:
        raise RuntimeError(f"Slice {grid.slice_id}: no contours were extracted.")
    li = min(available, key=lambda j: abs(float(grid.levels[j]) - float(grid.levels[preferred])))
    contours = contours_by_level[li]
    master = max(contours, key=lambda c: c.score)
    master.inherited_ell = master.local_ell.copy()
    master.alignment_valid_fraction = 1.0
    master.is_master = True
    return master


def align_contour_to_master(grid: AmbientGrid, contour: AmbientContour, master: AmbientContour) -> AmbientContour:
    """Assign inherited master ell to one contour by flowing to master z."""
    if contour.level_index == master.level_index and contour.component_index == master.component_index:
        contour.inherited_ell = contour.local_ell.copy()
        contour.alignment_valid_fraction = 1.0
        contour.is_master = True
        return contour

    n = len(contour.xy)
    inherited = np.full(n, np.nan, dtype=np.float64)
    valid_all = np.zeros(n, dtype=bool)

    z_start = np.full(n, float(contour.z_level), dtype=np.float64)
    z_target = np.full(n, float(master.z_level), dtype=np.float64)

    # Process in chunks for safety; each contour is usually only 2048 points.
    for start in range(0, n, int(BACKFLOW_CHUNK_SIZE)):
        end = min(start + int(BACKFLOW_CHUNK_SIZE), n)
        xy_back, z_back, z_error, ok = backflow_points_to_levels(
            grid=grid,
            xy_start=contour.xy[start:end],
            z_start=z_start[start:end],
            z_target=z_target[start:end],
            z_tol_fraction=float(CONTOUR_ALIGNMENT_Z_TOL_FRACTION),
        )
        ell_master, proj_dist, proj_xy, seg_idx, seg_u = project_points_to_contour_values(
            xy_back,
            master.xy,
            master.inherited_ell,
        )
        max_proj = float(MAX_PROJECTION_DISTANCE_GRID_STEPS) * float(grid.grid_step) * 1.5
        good = ok & np.isfinite(ell_master) & np.isfinite(proj_dist) & (proj_dist <= max_proj)
        inherited[start:end][good] = ell_master[good]
        valid_all[start:end][good] = True

    inherited_filled, frac = _fill_nan_along_polyline(inherited)
    contour.inherited_ell = inherited_filled.astype(np.float64)
    contour.alignment_valid_fraction = float(frac)
    contour.is_master = False
    return contour


def align_all_contours_to_master(grid: AmbientGrid, contours_by_level: Dict[int, List[AmbientContour]]) -> Tuple[AmbientContour, Dict[int, List[AmbientContour]]]:
    master = choose_master_contour(grid, contours_by_level)

    aligned: Dict[int, List[AmbientContour]] = {}
    for li, contours in contours_by_level.items():
        aligned_list: List[AmbientContour] = []
        for contour in contours:
            aligned_list.append(align_contour_to_master(grid, contour, master))
        aligned[li] = aligned_list

    # Fetch the aligned master object from the dictionary to keep references consistent.
    master_candidates = [
        c
        for c in aligned.get(master.level_index, [])
        if c.component_index == master.component_index
    ]
    if master_candidates:
        master = master_candidates[0]
    return master, aligned


def ambient_contours_dataframe(contours_by_level: Dict[int, List[AmbientContour]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for _, contours in contours_by_level.items():
        for c in contours:
            frames.append(
                pd.DataFrame(
                    {
                        "slice_id": int(c.slice_id),
                        "level_index": int(c.level_index),
                        "z_level": float(c.z_level),
                        "component_index": int(c.component_index),
                        "local_ell": c.local_ell,
                        "inherited_master_ell": c.inherited_ell,
                        "x": c.xy[:, 0],
                        "y": c.xy[:, 1],
                        "contour_length": float(c.length),
                        "tissue_contact_fraction": float(c.tissue_contact_fraction),
                        "contour_score": float(c.score),
                        "alignment_valid_fraction": float(c.alignment_valid_fraction),
                        "is_master": bool(c.is_master),
                    }
                )
            )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------
# Cell ell assignment
# ------------------------------------------------------------

def compute_master_contour_aligned_ell_for_slice(slice_id: int) -> MasterContourAlignedEllResult:
    grid = build_ambient_grid(slice_id)
    raw_contours_by_level = extract_raw_ambient_contours(grid)
    master_contour, contours_by_level = align_all_contours_to_master(grid, raw_contours_by_level)

    n_levels_with_contours = sum(1 for v in contours_by_level.values() if len(v) > 0)
    if n_levels_with_contours == 0:
        raise RuntimeError(f"Slice {slice_id}: no ambient contours were extracted.")

    raw_xy = grid.raw_xy
    z_cell = grid.z_cell
    n = len(raw_xy)

    max_projection_distance = float(MAX_PROJECTION_DISTANCE_GRID_STEPS) * float(grid.grid_step)
    rows: List[pd.DataFrame] = []
    top_k = max(1, int(CIRCULAR_AVERAGE_TOP_K))

    for start in range(0, n, int(BACKFLOW_CHUNK_SIZE)):
        end = min(start + int(BACKFLOW_CHUNK_SIZE), n)
        xy_chunk = raw_xy[start:end]
        z_start = z_cell[start:end]
        m = len(xy_chunk)

        top_score = np.full((m, top_k), np.inf, dtype=np.float64)
        top_ell = np.full((m, top_k), np.nan, dtype=np.float64)
        top_level_index = np.full((m, top_k), -1, dtype=int)
        top_component_index = np.full((m, top_k), -1, dtype=int)
        top_z_level = np.full((m, top_k), np.nan, dtype=np.float64)
        top_xy_back = np.full((m, top_k, 2), np.nan, dtype=np.float64)
        top_z_back = np.full((m, top_k), np.nan, dtype=np.float64)
        top_z_error = np.full((m, top_k), np.nan, dtype=np.float64)
        top_proj_xy = np.full((m, top_k, 2), np.nan, dtype=np.float64)
        top_proj_dist = np.full((m, top_k), np.nan, dtype=np.float64)
        top_segment_index = np.full((m, top_k), -1, dtype=int)
        top_backflow_valid = np.zeros((m, top_k), dtype=bool)
        top_projection_valid = np.zeros((m, top_k), dtype=bool)
        top_alignment_valid_fraction = np.full((m, top_k), np.nan, dtype=np.float64)

        nearest = _nearest_level_indices(grid.levels, z_start, int(N_NEAREST_LEVELS_TO_TRY))

        for rank in range(nearest.shape[1]):
            level_indices = nearest[:, rank]
            unique_levels = np.unique(level_indices)
            for li in unique_levels:
                li = int(li)
                contours = contours_by_level.get(li, [])
                if not contours:
                    continue
                idx_local = np.where(level_indices == li)[0]
                if len(idx_local) == 0:
                    continue
                z_target = np.full(len(idx_local), float(grid.levels[li]), dtype=np.float64)

                xy_back, z_back, z_error, backflow_valid = backflow_points_to_levels(
                    grid=grid,
                    xy_start=xy_chunk[idx_local],
                    z_start=z_start[idx_local],
                    z_target=z_target,
                    z_tol_fraction=float(BACKFLOW_Z_TOL_FRACTION),
                )

                for contour in contours:
                    ell_global, proj_dist, proj_xy, seg_idx, seg_u = project_points_to_contour_values(
                        xy_back,
                        contour.xy,
                        contour.inherited_ell,
                    )
                    projection_valid = np.isfinite(proj_dist) & (proj_dist <= max_projection_distance)
                    ell_valid = backflow_valid & projection_valid & np.isfinite(ell_global)
                    if not np.any(ell_valid):
                        continue

                    travel = np.abs(z_start[idx_local] - float(grid.levels[li])) / max(float(grid.z_range_cell), 1e-12)
                    zerr_scaled = np.abs(z_error) / max(float(grid.z_range_cell), 1e-12)
                    align_penalty = 1.0 - float(contour.alignment_valid_fraction)
                    score = (
                        proj_dist / max(float(grid.grid_step), 1e-12)
                        + float(SCORE_Z_ERROR_WEIGHT) * zerr_scaled
                        + float(SCORE_TRAVEL_WEIGHT) * travel
                        + float(SCORE_ALIGNMENT_CONFIDENCE_WEIGHT) * align_penalty
                    )
                    score[~ell_valid] = np.inf

                    _insert_top_candidate(
                        top_score=top_score,
                        top_ell=top_ell,
                        top_level_index=top_level_index,
                        top_component_index=top_component_index,
                        top_z_level=top_z_level,
                        top_xy_back=top_xy_back,
                        top_z_back=top_z_back,
                        top_z_error=top_z_error,
                        top_proj_xy=top_proj_xy,
                        top_proj_dist=top_proj_dist,
                        top_segment_index=top_segment_index,
                        top_backflow_valid=top_backflow_valid,
                        top_projection_valid=top_projection_valid,
                        top_alignment_valid_fraction=top_alignment_valid_fraction,
                        global_idx=idx_local,
                        score=score,
                        ell=ell_global,
                        level_index=int(li),
                        component_index=int(contour.component_index),
                        z_level=float(grid.levels[li]),
                        xy_back=xy_back,
                        z_back=z_back,
                        z_error=z_error,
                        proj_xy=proj_xy,
                        proj_dist=proj_dist,
                        segment_index=seg_idx,
                        backflow_valid=backflow_valid,
                        projection_valid=projection_valid,
                        alignment_valid_fraction=float(contour.alignment_valid_fraction),
                    )

        ell_avg, ell_u, ell_v, ell_resultant, n_used = _weighted_circular_mean_ell(top_ell, top_score)
        valid_avg = np.isfinite(ell_avg) & np.isfinite(ell_u) & np.isfinite(ell_v) & (n_used > 0)

        # Hard best candidate is kept only for diagnostics and comparison.
        finite_score = np.isfinite(top_score)
        best_slot = np.argmin(np.where(finite_score, top_score, np.inf), axis=1)
        row_idx = np.arange(m)
        best_score = top_score[row_idx, best_slot]
        best_ell = top_ell[row_idx, best_slot]
        best_level_index = top_level_index[row_idx, best_slot]
        best_component_index = top_component_index[row_idx, best_slot]
        best_z_level = top_z_level[row_idx, best_slot]
        best_xy_back = top_xy_back[row_idx, best_slot, :]
        best_z_back = top_z_back[row_idx, best_slot]
        best_z_error = top_z_error[row_idx, best_slot]
        best_proj_xy = top_proj_xy[row_idx, best_slot, :]
        best_proj_dist = top_proj_dist[row_idx, best_slot]
        best_segment_index = top_segment_index[row_idx, best_slot]
        best_backflow_valid = top_backflow_valid[row_idx, best_slot]
        best_projection_valid = top_projection_valid[row_idx, best_slot]
        best_alignment_valid_fraction = top_alignment_valid_fraction[row_idx, best_slot]

        # For rows with no finite candidate, clean diagnostic fields.
        no_candidate = ~np.isfinite(best_score)
        best_level_index[no_candidate] = -1
        best_component_index[no_candidate] = -1
        best_segment_index[no_candidate] = -1
        best_backflow_valid[no_candidate] = False
        best_projection_valid[no_candidate] = False

        rows.append(
            pd.DataFrame(
                {
                    "slice_id": int(slice_id),
                    "cell_index": np.arange(start, end, dtype=int),
                    "x": xy_chunk[:, 0],
                    "y": xy_chunk[:, 1],
                    "z_stage1a": z_start,
                    "master_z_level": float(grid.master_z_level),
                    "z_selected_level": best_z_level,
                    "selected_level_index": best_level_index,
                    "selected_component_index": best_component_index,
                    "x_backflow": best_xy_back[:, 0],
                    "y_backflow": best_xy_back[:, 1],
                    "z_backflow": best_z_back,
                    "z_backflow_error": best_z_error,
                    "x_projected_contour": best_proj_xy[:, 0],
                    "y_projected_contour": best_proj_xy[:, 1],
                    # Main coordinate: score-weighted circular average over candidates.
                    "ell_master_aligned": ell_avg,
                    "ell_master_aligned_cos": ell_u,
                    "ell_master_aligned_sin": ell_v,
                    "ell_circular_resultant": ell_resultant,
                    "ell_candidate_count": n_used,
                    # Diagnostic hard-best coordinate from the single lowest-score candidate.
                    "ell_master_aligned_best": best_ell,
                    "ell_best_minus_circular": ((best_ell - ell_avg + 0.5) % 1.0) - 0.5,
                    "ell_valid": valid_avg,
                    "ambient_backflow_valid": best_backflow_valid,
                    "ambient_projection_valid": best_projection_valid,
                    "ambient_projection_distance": best_proj_dist,
                    "ambient_projection_segment": best_segment_index,
                    "contour_alignment_valid_fraction": best_alignment_valid_fraction,
                    "ambient_assignment_score": best_score,
                }
            )
        )

        print(
            f"  slice {slice_id:02d} cells {start:,}-{end:,}: "
            f"circular ell valid {float(np.mean(valid_avg)):.1%}, "
            f"mean candidates {float(np.nanmean(n_used)):.2f}"
        )

    cell_table = pd.concat(rows, ignore_index=True)
    valid = cell_table["ell_valid"].to_numpy(bool)
    ell_valid = cell_table.loc[valid, "ell_master_aligned"].to_numpy(float)
    proj_valid = cell_table.loc[valid, "ambient_projection_distance"].to_numpy(float)
    zerr_valid = cell_table.loc[valid, "z_backflow_error"].to_numpy(float)
    align_valid = cell_table.loc[valid, "contour_alignment_valid_fraction"].to_numpy(float)
    resultant_valid = cell_table.loc[valid, "ell_circular_resultant"].to_numpy(float)
    n_used_valid = cell_table.loc[valid, "ell_candidate_count"].to_numpy(float)
    best_delta_valid = cell_table.loc[valid, "ell_best_minus_circular"].to_numpy(float)

    n_selected_components = int(
        cell_table.loc[valid, ["selected_level_index", "selected_component_index"]]
        .drop_duplicates()
        .shape[0]
    ) if np.any(valid) else 0

    contour_df = ambient_contours_dataframe(contours_by_level)
    alignment_fracs = contour_df["alignment_valid_fraction"].to_numpy(float) if len(contour_df) else np.array([])

    summary = {
        "slice_id": int(slice_id),
        "panel_label": grid.panel_label,
        "n_cells": int(n),
        "n_requested_levels": int(N_REFERENCE_LEVELS),
        "n_unique_levels": int(len(grid.levels)),
        "n_levels_with_contours": int(n_levels_with_contours),
        "n_selected_level_components_used": n_selected_components,
        "master_level_index": int(master_contour.level_index),
        "master_z_level": float(master_contour.z_level),
        "master_component_index": int(master_contour.component_index),
        "ambient_padding_fraction": float(AMBIENT_GRID_PADDING_FRACTION),
        "grid_step_raw": float(grid.grid_step),
        "gradient_floor": float(grid.gradient_floor),
        "z_range_cell": float(grid.z_range_cell),
        "ell_valid_fraction": float(cell_table["ell_valid"].mean()),
        "ambient_backflow_valid_fraction": float(cell_table["ambient_backflow_valid"].mean()),
        "ambient_projection_valid_fraction": float(cell_table["ambient_projection_valid"].mean()),
        "ell_mean": float(np.nanmean(ell_valid)) if ell_valid.size else np.nan,
        "ell_std": float(np.nanstd(ell_valid)) if ell_valid.size else np.nan,
        "ell_q01": float(np.nanquantile(ell_valid, 0.01)) if ell_valid.size else np.nan,
        "ell_q99": float(np.nanquantile(ell_valid, 0.99)) if ell_valid.size else np.nan,
        "projection_distance_median": float(np.nanmedian(proj_valid)) if proj_valid.size else np.nan,
        "projection_distance_q90": float(np.nanquantile(proj_valid, 0.90)) if proj_valid.size else np.nan,
        "projection_distance_q99": float(np.nanquantile(proj_valid, 0.99)) if proj_valid.size else np.nan,
        "z_backflow_abs_error_median": float(np.nanmedian(np.abs(zerr_valid))) if zerr_valid.size else np.nan,
        "z_backflow_abs_error_q90": float(np.nanquantile(np.abs(zerr_valid), 0.90)) if zerr_valid.size else np.nan,
        "cell_contour_alignment_valid_fraction_median": float(np.nanmedian(align_valid)) if align_valid.size else np.nan,
        "contour_alignment_valid_fraction_median": float(np.nanmedian(alignment_fracs)) if alignment_fracs.size else np.nan,
        "contour_alignment_valid_fraction_q10": float(np.nanquantile(alignment_fracs, 0.10)) if alignment_fracs.size else np.nan,
        "ell_circular_resultant_median": float(np.nanmedian(resultant_valid)) if resultant_valid.size else np.nan,
        "ell_circular_resultant_q10": float(np.nanquantile(resultant_valid, 0.10)) if resultant_valid.size else np.nan,
        "ell_candidate_count_mean": float(np.nanmean(n_used_valid)) if n_used_valid.size else np.nan,
        "ell_best_circular_abs_delta_median": float(np.nanmedian(np.abs(best_delta_valid))) if best_delta_valid.size else np.nan,
        "ell_best_circular_abs_delta_q90": float(np.nanquantile(np.abs(best_delta_valid), 0.90)) if best_delta_valid.size else np.nan,
    }

    return MasterContourAlignedEllResult(
        slice_id=int(slice_id),
        panel_label=grid.panel_label,
        grid=grid,
        master_contour=master_contour,
        contours_by_level=contours_by_level,
        cell_table=cell_table,
        summary=summary,
    )


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def _plot_aligned_contours(result: MasterContourAlignedEllResult, ax: plt.Axes) -> None:
    levels = sorted(result.contours_by_level.keys())
    levels_to_plot = levels[::max(1, int(CONTOUR_PLOT_STRIDE))]
    for li in levels_to_plot:
        for contour in result.contours_by_level.get(li, []):
            if contour.is_master:
                continue
            ax.plot(
                contour.xy[:, 0],
                contour.xy[:, 1],
                color=CONTOUR_COLOR,
                alpha=CONTOUR_ALPHA,
                linewidth=CONTOUR_LINEWIDTH,
                zorder=2,
            )
    ax.plot(
        result.master_contour.xy[:, 0],
        result.master_contour.xy[:, 1],
        color=MASTER_CONTOUR_COLOR,
        alpha=0.95,
        linewidth=MASTER_CONTOUR_LINEWIDTH,
        zorder=4,
    )


def plot_master_contour_aligned_ell_heatmap(
    result: MasterContourAlignedEllResult,
    ax: Optional[plt.Axes] = None,
    show_colorbar: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    d = result.cell_table
    valid = d["ell_valid"].to_numpy(bool)

    rng = np.random.default_rng(20260805 + int(result.slice_id))
    if len(d) > int(MAX_POINTS_PER_PLOT):
        idx = rng.choice(len(d), size=int(MAX_POINTS_PER_PLOT), replace=False)
        dp = d.iloc[idx].copy()
        valid_p = dp["ell_valid"].to_numpy(bool)
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
        c=good["ell_master_aligned"],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        linewidths=0,
        cmap=ELL_CMAP,
        vmin=0.0,
        vmax=1.0,
        zorder=3,
    )

    _plot_aligned_contours(result, ax)

    ax.set_title(
        f"Slice {result.slice_id}: circular master-contour aligned ell\n"
        f"valid={result.summary['ell_valid_fraction']:.1%} | "
        f"levels={result.summary['n_levels_with_contours']} | {result.panel_label}",
        fontsize=10,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if show_colorbar:
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="score-weighted circular ell")
    return fig, ax


def plot_ambient_z_and_master_aligned_contours(
    result: MasterContourAlignedEllResult,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    grid = result.grid
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_PER_SLICE, constrained_layout=True)
    else:
        fig = ax.figure
    im = ax.pcolormesh(
        grid.grid_x,
        grid.grid_y,
        grid.z_grid,
        shading="auto",
        cmap="viridis",
    )
    try:
        ax.contour(grid.grid_x, grid.grid_y, grid.mask.astype(float), levels=[0.5], colors="white", linewidths=0.8, alpha=0.9)
    except Exception:
        pass
    _plot_aligned_contours(result, ax)
    ax.scatter(grid.raw_xy[:, 0], grid.raw_xy[:, 1], s=0.05, c="white", alpha=0.10, linewidths=0)
    ax.set_title(f"Slice {result.slice_id}: ambient z + master-aligned contours", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Stage 1a z")
    return fig, ax


# ------------------------------------------------------------
# Run all requested slices
# ------------------------------------------------------------

print("Circular-smoothed master-contour-aligned ambient multi-contour ell")
print("Slices:", SLICE_IDS)
print("Output directory:", OUT_DIR)
print("N reference levels:", N_REFERENCE_LEVELS)
print("Master level quantile:", MASTER_LEVEL_QUANTILE)
print("Ambient padding fraction:", AMBIENT_GRID_PADDING_FRACTION)
print("Keep all ambient components:", KEEP_ALL_AMBIENT_COMPONENTS)
print("Backflow allows off-tissue travel inside the padded grid.")

master_contour_aligned_circular_ell_results: Dict[int, MasterContourAlignedEllResult] = {}
_summary_rows: List[Dict[str, Any]] = []
_cell_frames: List[pd.DataFrame] = []
_failures: List[Dict[str, Any]] = []

for _slice_id in SLICE_IDS:
    _slice_id = int(_slice_id)
    print("\n" + "=" * 76)
    print(f"Computing master-contour-aligned ell for slice {_slice_id}")
    print("=" * 76)
    try:
        _result = compute_master_contour_aligned_ell_for_slice(_slice_id)
        master_contour_aligned_circular_ell_results[_slice_id] = _result
        _summary_rows.append(_result.summary)
        _cell_frames.append(_result.cell_table)

        if SAVE_PER_SLICE_TABLES:
            _result.cell_table.to_csv(
                TABLES_DIR / f"slice_{_slice_id:02d}_master_contour_aligned_circular_ell_cells.csv.gz",
                index=False,
                compression="gzip",
            )

        if SAVE_CONTOUR_TABLES:
            _contour_df = ambient_contours_dataframe(_result.contours_by_level)
            _contour_df.to_csv(
                CONTOURS_DIR / f"slice_{_slice_id:02d}_master_aligned_contours.csv.gz",
                index=False,
                compression="gzip",
            )

        fig, _ = plot_master_contour_aligned_ell_heatmap(_result)
        if SAVE_PNG:
            fig.savefig(
                PLOTS_DIR / f"slice_{_slice_id:02d}_master_contour_aligned_circular_ell_heatmap.png",
                dpi=DPI,
                bbox_inches="tight",
            )
        if SAVE_PDF:
            fig.savefig(
                PLOTS_DIR / f"slice_{_slice_id:02d}_master_contour_aligned_circular_ell_heatmap.pdf",
                bbox_inches="tight",
            )
        if DISPLAY_FIGURES:
            plt.show()
        else:
            plt.close(fig)

        fig_z, _ = plot_ambient_z_and_master_aligned_contours(_result)
        if SAVE_PNG:
            fig_z.savefig(
                PLOTS_DIR / f"slice_{_slice_id:02d}_ambient_z_master_aligned_contours.png",
                dpi=DPI,
                bbox_inches="tight",
            )
        if SAVE_PDF:
            fig_z.savefig(
                PLOTS_DIR / f"slice_{_slice_id:02d}_ambient_z_master_aligned_contours.pdf",
                bbox_inches="tight",
            )
        plt.close(fig_z)

        print(
            f"slice {_slice_id}: ell valid={_result.summary['ell_valid_fraction']:.1%}, "
            f"levels={_result.summary['n_levels_with_contours']}, "
            f"master_z={_result.summary['master_z_level']:.4g}, "
            f"ell std={_result.summary['ell_std']:.3f}, "
            f"proj median={_result.summary['projection_distance_median']:.3g}, "
            f"zerr median={_result.summary['z_backflow_abs_error_median']:.3g}, "
            f"contour align med={_result.summary['contour_alignment_valid_fraction_median']:.2f}"
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if not master_contour_aligned_circular_ell_results:
    raise RuntimeError("No slices succeeded. Inspect _failures for details.")

master_contour_aligned_circular_ell_summary_df = (
    pd.DataFrame(_summary_rows)
    .sort_values("slice_id")
    .reset_index(drop=True)
)
master_contour_aligned_circular_ell_summary_df.to_csv(
    TABLES_DIR / "master_contour_aligned_circular_ell_summary_by_slice.csv",
    index=False,
)

if SAVE_COMBINED_TABLE:
    master_contour_aligned_circular_ell_cells_df = (
        pd.concat(_cell_frames, ignore_index=True)
        .sort_values(["slice_id", "cell_index"])
        .reset_index(drop=True)
    )
    master_contour_aligned_circular_ell_cells_df.to_csv(
        TABLES_DIR / "master_contour_aligned_circular_ell_all_cells.csv.gz",
        index=False,
        compression="gzip",
    )
else:
    master_contour_aligned_circular_ell_cells_df = pd.concat(_cell_frames, ignore_index=True)

with open(OUT_DIR / "master_contour_aligned_circular_ell_config.json", "w") as handle:
    json.dump(
        {
            "slice_ids": [int(x) for x in SLICE_IDS],
            "grid_max_dim": int(GRID_MAX_DIM),
            "ambient_grid_padding_fraction": float(AMBIENT_GRID_PADDING_FRACTION),
            "n_reference_levels": int(N_REFERENCE_LEVELS),
            "level_quantile_range": list(LEVEL_QUANTILE_RANGE),
            "master_level_quantile": float(MASTER_LEVEL_QUANTILE),
            "keep_all_ambient_components": bool(KEEP_ALL_AMBIENT_COMPONENTS),
            "contour_resample_n": int(CONTOUR_RESAMPLE_N),
            "contour_orient_by": str(CONTOUR_ORIENT_BY),
            "backflow_n_steps": int(BACKFLOW_N_STEPS),
            "backflow_method": str(BACKFLOW_METHOD),
            "flow_eps": float(FLOW_EPS),
            "gradient_floor_quantile": float(GRADIENT_FLOOR_QUANTILE),
            "absolute_gradient_floor": float(ABSOLUTE_GRADIENT_FLOOR),
            "max_vector_norm_grid_steps": float(MAX_VECTOR_NORM_GRID_STEPS),
            "backflow_z_tol_fraction": float(BACKFLOW_Z_TOL_FRACTION),
            "contour_alignment_z_tol_fraction": float(CONTOUR_ALIGNMENT_Z_TOL_FRACTION),
            "n_nearest_levels_to_try": int(N_NEAREST_LEVELS_TO_TRY),
            "projection_k_segments": int(PROJECTION_K_SEGMENTS),
            "max_projection_distance_grid_steps": float(MAX_PROJECTION_DISTANCE_GRID_STEPS),
            "score_z_error_weight": float(SCORE_Z_ERROR_WEIGHT),
            "score_travel_weight": float(SCORE_TRAVEL_WEIGHT),
            "score_alignment_confidence_weight": float(SCORE_ALIGNMENT_CONFIDENCE_WEIGHT),
            "use_circular_ell_embedding": bool(USE_CIRCULAR_ELL_EMBEDDING),
            "circular_average_top_k": int(CIRCULAR_AVERAGE_TOP_K),
            "circular_average_temperature": float(CIRCULAR_AVERAGE_TEMPERATURE),
            "circular_average_max_score_gap": float(CIRCULAR_AVERAGE_MAX_SCORE_GAP),
            "failures": _failures,
        },
        handle,
        indent=2,
    )

# Overview figure.
if MAKE_OVERVIEW_FIGURE:
    n = len(master_contour_aligned_circular_ell_results)
    ncols = max(1, min(int(OVERVIEW_NCOLS), n))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.8 * ncols, 3.4 * nrows),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    for ax, sid in zip(axes, sorted(master_contour_aligned_circular_ell_results)):
        plot_master_contour_aligned_ell_heatmap(
            master_contour_aligned_circular_ell_results[sid],
            ax=ax,
            show_colorbar=False,
        )
    for ax in axes[len(master_contour_aligned_circular_ell_results):]:
        ax.axis("off")
    fig.suptitle("Circular-smoothed master-contour-aligned ambient multi-contour ell", fontsize=14)
    fig.savefig(
        PLOTS_DIR / "all_slices_master_contour_aligned_circular_ell_heatmaps.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    if SAVE_PDF:
        fig.savefig(
            PLOTS_DIR / "all_slices_master_contour_aligned_circular_ell_heatmaps.pdf",
            bbox_inches="tight",
        )
    if DISPLAY_FIGURES:
        plt.show()
    else:
        plt.close(fig)

print("\nMaster-contour-aligned ell complete.")
print("Successful slices:", sorted(master_contour_aligned_circular_ell_results.keys()))
print("Failures:", _failures if _failures else "none")
print("Summary table:", TABLES_DIR / "master_contour_aligned_circular_ell_summary_by_slice.csv")
print("Per-slice / combined cell tables:", TABLES_DIR)
print("Contour tables:", CONTOURS_DIR)
print("Plots:", PLOTS_DIR)

display(master_contour_aligned_circular_ell_summary_df)
