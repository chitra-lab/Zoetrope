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
    """Save observed data and compact latent truth to output_dir."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    result.obs.to_csv(
        output / "cell_metadata.csv.gz",
        index=False,
        compression="gzip",
    )
    result.counts.to_csv(output / "counts.csv.gz", compression="gzip")
    if save_expected_mean:
        result.expected_mean.to_csv(
            output / "expected_mean.csv.gz",
            compression="gzip",
        )
    result.truth.gene_metadata.to_csv(
        output / "gene_metadata.csv",
        index=False,
    )

    np.savez_compressed(
        output / "truth_arrays.npz",
        z_grid=result.truth.z,
        time_grid=result.truth.time,
        initial_position_grid=result.truth.initial_position,
        psi_true=result.truth.psi,
        E_true=result.truth.mean,
        psi_z_true=result.truth.psi_z,
        psi_t_true=result.truth.psi_t,
        V_true=result.truth.velocity,
        observability_true=result.truth.observability,
        residual_speed_true=result.truth.residual_speed,
        F_true=result.truth.flow,
        rho_true=result.truth.rho,
        bend_angles_deg=result.config.bend_angles_deg,
    )

    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(result.config.to_dict(), handle, indent=2, sort_keys=True)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result.summary(), handle, indent=2, sort_keys=True)

    return output
