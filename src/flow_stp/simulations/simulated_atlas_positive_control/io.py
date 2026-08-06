"""Persistence helpers for positive-control simulation outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .simulation import SimulationResult


def save_simulation(
    result: SimulationResult,
    output_dir: str | Path,
    *,
    save_expected_mean: bool = True,
) -> Path:
    """Save observed data and compact latent truth to ``output_dir``.

    CSV files are gzip-compressed and require no optional parquet dependency.
    Dense latent arrays are saved in a single compressed NumPy archive.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    result.obs.to_csv(output / "cell_metadata.csv.gz", index=False, compression="gzip")
    result.counts.to_csv(output / "counts.csv.gz", compression="gzip")
    if save_expected_mean:
        result.expected_mean.to_csv(
            output / "expected_mean.csv.gz",
            compression="gzip",
        )
    result.cohort_table.to_csv(output / "cohorts.csv", index=False)
    result.decoder.gene_metadata.to_csv(output / "gene_metadata.csv", index=False)

    truth_arrays: dict[str, np.ndarray] = {
        "z": result.pde.z,
        "time": result.pde.time,
        # FLOW_STP_ANALYTIC_FIELDS_IO_BEGIN
        "flow_over_time": result.pde.flow_over_time,
        "rho_over_time": result.pde.rho_over_time,
        # FLOW_STP_ANALYTIC_FIELDS_IO_END
        "q_over_time": result.pde.q_over_time,
        "bend_angles_deg": result.config.bend_angles_deg,
        "stopping_points": result.config.stopping_points,
        "velocity_over_time": result.pde.velocity_over_time,
        "velocity_stopping_points_over_time": (
            result.pde.velocity_stopping_points_over_time
        ),
        "density_total": result.pde.density_total,
        "rho_observed": result.pde.rho_observed,
        "rho_support_mask": result.pde.rho_support_mask,
        "rho_absolute": result.decoder.rho_absolute,
        "rho_decoder": result.decoder.rho_decoder,
        "baseline_mean": result.decoder.baseline_mean,
        "baseline_log_mean": result.decoder.baseline_log_mean,
        "developmental_log_effect": result.decoder.developmental_log_effect,
        "log_mean_expression": result.decoder.log_mean,
        "mean_expression": result.decoder.mean,
    }
    for name, values in result.pde.velocity_by_cohort.items():
        truth_arrays[f"velocity__{name}"] = values
    for name, values in result.pde.density_by_cohort.items():
        truth_arrays[f"density__{name}"] = values
    for name, values in result.pde.rho_by_cohort.items():
        truth_arrays[f"rho__{name}"] = values

    np.savez_compressed(output / "truth_arrays.npz", **truth_arrays)

    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(result.config.to_dict(), handle, indent=2, sort_keys=True)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result.summary(), handle, indent=2, sort_keys=True)

    return output

# FLOW_STP_TIME_INDEXED_VELOCITY_V2
