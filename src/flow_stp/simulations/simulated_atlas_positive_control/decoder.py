"""Bare-bones centered-rho gene-expression truth generator."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .dynamics import smootherstep
from .pde import PDEResult


GENE_PROGRAMS = (
    "radial_glia_down",
    "migratory_transient",
    "mature_neuron_up",
    "deep_layer_maturation",
    "upper_layer_maturation",
    "housekeeping_stable",
)


@dataclass(frozen=True)
class DecoderResult:
    """Ground-truth baseline, developmental coordinate, and expression surfaces."""

    gene_names: tuple[str, ...]
    gene_metadata: pd.DataFrame
    rho_absolute: np.ndarray
    rho_decoder: np.ndarray
    rho_unit: np.ndarray
    baseline_mean: np.ndarray
    baseline_log_mean: np.ndarray
    raw_decoder: np.ndarray
    developmental_log_effect: np.ndarray
    log_mean: np.ndarray
    mean: np.ndarray
    baseline_anchor_error: float


def _gaussian(x: np.ndarray, *, center: float, width: float) -> np.ndarray:
    if width <= 0.0:
        raise ValueError("width must be positive.")
    return np.exp(-0.5 * ((x - center) / width) ** 2)


def _smooth_activation(
    rho_unit: np.ndarray,
    *,
    start: float,
    duration: float,
) -> np.ndarray:
    if duration <= 0.0:
        raise ValueError("duration must be positive.")
    return smootherstep((rho_unit - start) / duration)


def continue_rho_across_empty_space(
    rho_grid: np.ndarray,
    support_mask: np.ndarray,
    z_grid: np.ndarray,
) -> np.ndarray:
    """Continue ``rho`` over empty locations without changing supported values."""
    rho_grid = np.asarray(rho_grid, dtype=float)
    support_mask = np.asarray(support_mask, dtype=bool)
    z_grid = np.asarray(z_grid, dtype=float)
    if rho_grid.shape != support_mask.shape:
        raise ValueError("rho_grid and support_mask must have equal shape.")
    if rho_grid.shape[1] != len(z_grid):
        raise ValueError("z_grid length must match the second rho dimension.")

    filled = np.empty_like(rho_grid, dtype=float)
    for time_index in range(rho_grid.shape[0]):
        valid = support_mask[time_index] & np.isfinite(rho_grid[time_index])
        if np.any(valid):
            filled[time_index] = np.interp(
                z_grid,
                z_grid[valid],
                rho_grid[time_index, valid],
            )
            filled[time_index, valid] = rho_grid[time_index, valid]
        elif time_index > 0:
            filled[time_index] = filled[time_index - 1]
        else:
            raise RuntimeError("The first PDE time slice has no supported rho values.")

    if not np.all(np.isfinite(filled)):
        raise RuntimeError("Could not construct a finite rho continuation.")
    return filled


def _build_gene_truth(
    z: np.ndarray,
    config: SimulationConfig,
) -> tuple[tuple[str, ...], pd.DataFrame, np.ndarray, list[dict[str, float | str]]]:
    rng = np.random.default_rng(config.gene_generator_seed)
    prefix = {
        "radial_glia_down": "RG",
        "migratory_transient": "MIG",
        "mature_neuron_up": "MAT",
        "deep_layer_maturation": "DEEP",
        "upper_layer_maturation": "UPPER",
        "housekeeping_stable": "HOUSE",
    }

    metadata_rows: list[dict[str, float | str]] = []
    parameters: list[dict[str, float | str]] = []
    baseline_columns: list[np.ndarray] = []
    gene_names: list[str] = []

    for program in GENE_PROGRAMS:
        for within_program_index in range(config.genes_per_program):
            gene_name = f"{prefix[program]}_{within_program_index + 1:02d}"

            if program == "radial_glia_down":
                baseline_center = rng.normal(0.04, 0.015)
                baseline_width = rng.uniform(0.05, 0.10)
                baseline_floor = rng.uniform(0.15, 0.40)
                baseline_peak = rng.uniform(4.0, 8.0)
            elif program == "migratory_transient":
                baseline_center = rng.uniform(0.15, 0.45)
                baseline_width = rng.uniform(0.15, 0.28)
                baseline_floor = rng.uniform(0.15, 0.45)
                baseline_peak = rng.uniform(0.5, 1.5)
            elif program == "mature_neuron_up":
                baseline_center = rng.uniform(0.35, 0.75)
                baseline_width = rng.uniform(0.22, 0.40)
                baseline_floor = rng.uniform(0.10, 0.35)
                baseline_peak = rng.uniform(0.3, 1.0)
            elif program == "deep_layer_maturation":
                baseline_center = float(rng.choice([0.34, 0.50])) + rng.normal(0.0, 0.02)
                baseline_width = rng.uniform(0.055, 0.10)
                baseline_floor = rng.uniform(0.10, 0.30)
                baseline_peak = rng.uniform(1.0, 2.5)
            elif program == "upper_layer_maturation":
                baseline_center = float(rng.choice([0.69, 0.87])) + rng.normal(0.0, 0.02)
                baseline_width = rng.uniform(0.055, 0.10)
                baseline_floor = rng.uniform(0.10, 0.30)
                baseline_peak = rng.uniform(0.8, 2.0)
            else:
                baseline_center = rng.uniform(0.15, 0.85)
                baseline_width = rng.uniform(0.25, 0.50)
                baseline_floor = rng.uniform(2.0, 4.0)
                baseline_peak = rng.uniform(0.2, 0.8)

            baseline_mean = baseline_floor + baseline_peak * _gaussian(
                z,
                center=float(baseline_center),
                width=float(baseline_width),
            )
            baseline_mean += rng.uniform(0.0, 0.35) * _gaussian(
                z,
                center=float(rng.uniform(0.0, 1.0)),
                width=float(rng.uniform(0.20, 0.45)),
            )

            if program == "radial_glia_down":
                amplitude = rng.uniform(1.0, 1.8)
                spatial_center = rng.normal(0.05, 0.02)
                spatial_width = rng.uniform(0.08, 0.16)
                rho_center = 0.35
                rho_width = 0.18
                activation_start = 0.05
                activation_duration = 0.55
            elif program == "migratory_transient":
                amplitude = rng.uniform(1.2, 2.2)
                spatial_center = rng.uniform(0.25, 0.70)
                spatial_width = rng.uniform(0.25, 0.45)
                rho_center = rng.uniform(0.30, 0.48)
                rho_width = rng.uniform(0.10, 0.17)
                activation_start = 0.05
                activation_duration = 0.70
            elif program == "mature_neuron_up":
                amplitude = rng.uniform(1.3, 2.3)
                spatial_center = rng.uniform(0.35, 0.75)
                spatial_width = rng.uniform(0.35, 0.60)
                rho_center = 0.70
                rho_width = 0.20
                activation_start = rng.uniform(0.12, 0.22)
                activation_duration = rng.uniform(0.55, 0.70)
            elif program == "deep_layer_maturation":
                amplitude = rng.uniform(1.2, 2.0)
                spatial_center = baseline_center
                spatial_width = rng.uniform(0.08, 0.14)
                rho_center = 0.70
                rho_width = 0.18
                activation_start = rng.uniform(0.12, 0.22)
                activation_duration = rng.uniform(0.55, 0.70)
            elif program == "upper_layer_maturation":
                amplitude = rng.uniform(1.4, 2.4)
                spatial_center = baseline_center
                spatial_width = rng.uniform(0.08, 0.14)
                rho_center = 0.78
                rho_width = 0.16
                activation_start = rng.uniform(0.22, 0.34)
                activation_duration = rng.uniform(0.50, 0.65)
            else:
                amplitude = rng.uniform(-0.15, 0.15)
                spatial_center = rng.uniform(0.0, 1.0)
                spatial_width = rng.uniform(0.30, 0.60)
                rho_center = 0.50
                rho_width = 0.25
                activation_start = 0.10
                activation_duration = 0.75

            interaction_strength = rng.uniform(-0.25, 0.25)
            row = {
                "gene": gene_name,
                "program": program,
                "baseline_center": float(baseline_center),
                "baseline_width": float(baseline_width),
                "baseline_floor": float(baseline_floor),
                "baseline_peak": float(baseline_peak),
                "decoder_amplitude": float(amplitude),
                "decoder_spatial_center": float(spatial_center),
                "decoder_spatial_width": float(spatial_width),
                "decoder_rho_center": float(rho_center),
                "decoder_rho_width": float(rho_width),
                "decoder_activation_start": float(activation_start),
                "decoder_activation_duration": float(activation_duration),
                "decoder_interaction_strength": float(interaction_strength),
            }
            metadata_rows.append(row)
            parameters.append(
                {
                    "program": program,
                    "amplitude": float(amplitude),
                    "spatial_center": float(spatial_center),
                    "spatial_width": float(spatial_width),
                    "rho_center": float(rho_center),
                    "rho_width": float(rho_width),
                    "activation_start": float(activation_start),
                    "activation_duration": float(activation_duration),
                    "interaction_strength": float(interaction_strength),
                }
            )
            baseline_columns.append(baseline_mean)
            gene_names.append(gene_name)

    return (
        tuple(gene_names),
        pd.DataFrame(metadata_rows),
        np.stack(baseline_columns, axis=1),
        parameters,
    )


def evaluate_raw_decoder(
    z: np.ndarray,
    rho_unit: np.ndarray,
    parameters: list[dict[str, float | str]],
) -> np.ndarray:
    """Evaluate the general unanchored truth decoder ``D_raw(z,rho)``."""
    z = np.asarray(z, dtype=float)
    rho_unit = np.asarray(rho_unit, dtype=float)
    z_broadcast, rho_broadcast = np.broadcast_arrays(z, rho_unit)
    output = np.empty(z_broadcast.shape + (len(parameters),), dtype=float)

    for gene_index, p in enumerate(parameters):
        program = str(p["program"])
        spatial_gate = _gaussian(
            z_broadcast,
            center=float(p["spatial_center"]),
            width=float(p["spatial_width"]),
        )
        activation = _smooth_activation(
            rho_broadcast,
            start=float(p["activation_start"]),
            duration=float(p["activation_duration"]),
        )
        transient = _gaussian(
            rho_broadcast,
            center=float(p["rho_center"]),
            width=float(p["rho_width"]),
        )
        amplitude = float(p["amplitude"])
        interaction = float(p["interaction_strength"])

        if program == "radial_glia_down":
            raw = -amplitude * spatial_gate * activation + (
                interaction * rho_broadcast * (z_broadcast - 0.5)
            )
        elif program == "migratory_transient":
            raw = amplitude * spatial_gate * transient + (
                interaction * rho_broadcast * (z_broadcast - 0.5)
            )
        elif program == "mature_neuron_up":
            raw = amplitude * (0.65 + 0.35 * spatial_gate) * activation + (
                interaction * rho_broadcast * (z_broadcast - 0.5)
            )
        elif program in {"deep_layer_maturation", "upper_layer_maturation"}:
            raw = amplitude * spatial_gate * activation + (
                interaction * spatial_gate * rho_broadcast**2
            )
        else:
            raw = amplitude * rho_broadcast + (
                interaction * np.sin(math.pi * z_broadcast) * rho_broadcast
            )
        output[..., gene_index] = raw

    return output


def generate_expression_surfaces(
    config: SimulationConfig,
    pde: PDEResult,
) -> DecoderResult:
    """Generate ``B(z)+D_raw(z,rho)-D_raw(z,0)`` truth surfaces."""
    rho_absolute = continue_rho_across_empty_space(
        pde.rho_observed,
        pde.rho_support_mask,
        pde.z,
    )
    rho_decoder = np.maximum(rho_absolute - rho_absolute[0:1, :], 0.0)
    rho_decoder[0, :] = 0.0
    rho_scale = float(np.max(rho_decoder))
    if rho_scale <= 0.0:
        raise RuntimeError("The centered PDE rho field is identically zero.")
    rho_unit = rho_decoder / rho_scale

    gene_names, gene_metadata, baseline_mean, parameters = _build_gene_truth(
        pde.z,
        config,
    )
    baseline_log_mean = np.log(baseline_mean)
    raw_decoder = evaluate_raw_decoder(pde.z[None, :], rho_unit, parameters)
    raw_decoder_zero = evaluate_raw_decoder(
        pde.z[None, :],
        np.zeros_like(rho_unit),
        parameters,
    )
    developmental_log_effect = raw_decoder - raw_decoder_zero
    log_mean = baseline_log_mean[None, :, :] + developmental_log_effect
    mean = np.exp(log_mean)
    baseline_anchor_error = float(
        np.max(np.abs(log_mean[0] - baseline_log_mean))
    )
    if baseline_anchor_error != 0.0:
        raise RuntimeError("The exact t=0 decoder anchor was not preserved.")

    return DecoderResult(
        gene_names=gene_names,
        gene_metadata=gene_metadata,
        rho_absolute=rho_absolute,
        rho_decoder=rho_decoder,
        rho_unit=rho_unit,
        baseline_mean=baseline_mean,
        baseline_log_mean=baseline_log_mean,
        raw_decoder=raw_decoder,
        developmental_log_effect=developmental_log_effect,
        log_mean=log_mean,
        mean=mean,
        baseline_anchor_error=baseline_anchor_error,
    )
