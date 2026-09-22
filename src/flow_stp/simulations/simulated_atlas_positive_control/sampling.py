"""Uniform intrinsic sampling and Poisson observation generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .geometry import bend_tissue_coordinates, sample_intrinsic_coordinates
from .truth import TruthResult


@dataclass(frozen=True)
class AtlasSample:
    """Cell metadata, expected means, and Poisson count observations."""

    obs: pd.DataFrame
    counts: pd.DataFrame
    expected_mean: pd.DataFrame


def interpolate_time_z(
    values: np.ndarray,
    *,
    time_grid: np.ndarray,
    z_grid: np.ndarray,
    query_time: np.ndarray,
    query_z: np.ndarray,
) -> np.ndarray:
    """Bilinearly interpolate [time,z,...] values at paired cell locations."""
    values = np.asarray(values, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)
    z_grid = np.asarray(z_grid, dtype=float)
    query_time = np.asarray(query_time, dtype=float).reshape(-1)
    query_z = np.asarray(query_z, dtype=float).reshape(-1)

    if values.shape[:2] != (len(time_grid), len(z_grid)):
        raise ValueError("values must begin with dimensions [time,z].")
    if query_time.shape != query_z.shape:
        raise ValueError("query_time and query_z must have equal shape.")
    if np.any((query_time < time_grid[0]) | (query_time > time_grid[-1])):
        raise ValueError("A requested time lies outside the interpolation grid.")
    if np.any((query_z < z_grid[0]) | (query_z > z_grid[-1])):
        raise ValueError("A requested z lies outside the interpolation grid.")

    time_upper = np.searchsorted(time_grid, query_time, side="right")
    time_upper = np.clip(time_upper, 1, len(time_grid) - 1)
    time_lower = time_upper - 1
    z_upper = np.searchsorted(z_grid, query_z, side="right")
    z_upper = np.clip(z_upper, 1, len(z_grid) - 1)
    z_lower = z_upper - 1

    time_fraction = (query_time - time_grid[time_lower]) / (
        time_grid[time_upper] - time_grid[time_lower]
    )
    z_fraction = (query_z - z_grid[z_lower]) / (
        z_grid[z_upper] - z_grid[z_lower]
    )
    trailing_shape = (len(query_time),) + (1,) * (values.ndim - 2)
    time_fraction = time_fraction.reshape(trailing_shape)
    z_fraction = z_fraction.reshape(trailing_shape)

    lower_time = (
        (1.0 - z_fraction) * values[time_lower, z_lower]
        + z_fraction * values[time_lower, z_upper]
    )
    upper_time = (
        (1.0 - z_fraction) * values[time_upper, z_lower]
        + z_fraction * values[time_upper, z_upper]
    )
    return (1.0 - time_fraction) * lower_time + time_fraction * upper_time


def sample_atlas(
    config: SimulationConfig,
    truth: TruthResult,
) -> AtlasSample:
    """Sample tissues uniformly in intrinsic coordinates and draw Poisson counts."""
    config.validate()
    rng = np.random.default_rng(config.random_seed)
    n_total = config.total_cells

    tissue_index = np.repeat(
        np.arange(config.n_timepoints),
        config.n_cells_per_tissue,
    )
    cell_time = np.repeat(config.tissue_times, config.n_cells_per_tissue)
    cell_bend = np.repeat(config.bend_angles_deg, config.n_cells_per_tissue)

    s_intrinsic, z_intrinsic = sample_intrinsic_coordinates(
        n_total,
        width=config.tissue_width,
        rng=rng,
    )
    x_extrinsic, y_extrinsic = bend_tissue_coordinates(
        s_intrinsic,
        z_intrinsic,
        width=config.tissue_width,
        height=config.tissue_height,
        bend_angle_deg=cell_bend,
    )

    expected_mean = interpolate_time_z(
        truth.mean,
        time_grid=truth.time,
        z_grid=truth.z,
        query_time=cell_time,
        query_z=z_intrinsic,
    )
    expected_mean = np.maximum(expected_mean, 0.0)
    counts = rng.poisson(expected_mean).astype(np.int32, copy=False)

    cell_ids = np.asarray(
        [
            f"tissue_{tissue_id:02d}_cell_{within_id:05d}"
            for tissue_id in range(config.n_timepoints)
            for within_id in range(config.n_cells_per_tissue)
        ],
        dtype=object,
    )
    tissue_ids = np.asarray(
        [f"tissue_{index:02d}" for index in tissue_index],
        dtype=object,
    )

    initial_position_true = z_intrinsic - config.flow_speed * cell_time
    rho_true = truth.developmental_speed * cell_time

    obs = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "tissue_id": tissue_ids,
            "tissue_index": tissue_index,
            "time": cell_time,
            "bend_angle_deg": cell_bend,
            "s_intrinsic": s_intrinsic,
            "z_true": z_intrinsic,
            "initial_position_true": initial_position_true,
            "rho_true": rho_true,
            "x_extrinsic": x_extrinsic,
            "y_extrinsic": y_extrinsic,
        }
    ).set_index("cell_id", drop=False)

    count_table = pd.DataFrame(
        counts,
        index=cell_ids,
        columns=truth.gene_names,
    )
    count_table.index.name = "cell_id"
    expected_table = pd.DataFrame(
        expected_mean,
        index=cell_ids,
        columns=truth.gene_names,
    )
    expected_table.index.name = "cell_id"

    sizes = obs.groupby("tissue_id", sort=False).size().to_numpy()
    if not np.all(sizes == config.n_cells_per_tissue):
        raise RuntimeError("At least one tissue has the wrong number of cells.")
    if np.any(counts < 0):
        raise RuntimeError("Poisson counts must be nonnegative.")

    return AtlasSample(
        obs=obs,
        counts=count_table,
        expected_mean=expected_table,
    )
