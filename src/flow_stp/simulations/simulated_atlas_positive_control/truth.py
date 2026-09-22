"""Analytic truth for the current Fisher-Lagrangian positive control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SimulationConfig


@dataclass(frozen=True)
class TruthResult:
    """Ground-truth expression geometry and derived trajectories."""

    z: np.ndarray
    time: np.ndarray
    initial_position: np.ndarray
    gene_names: tuple[str, ...]
    gene_metadata: pd.DataFrame
    psi: np.ndarray
    mean: np.ndarray
    psi_z: np.ndarray
    psi_t: np.ndarray
    velocity: np.ndarray
    observability: np.ndarray
    residual_speed: np.ndarray
    flow: np.ndarray
    rho: np.ndarray
    developmental_speed: float


def _gaussian(
    x: np.ndarray,
    *,
    center: float,
    width: float,
) -> tuple[np.ndarray, np.ndarray]:
    value = np.exp(-0.5 * ((x - center) / width) ** 2)
    derivative = -(x - center) / (width**2) * value
    return value, derivative


def generate_truth(config: SimulationConfig) -> TruthResult:
    """Generate E and downstream truth from current-model definitions.

    Spatial genes are transported rigidly with speed U in Fisher-flat
    coordinates. Developmental genes vary only with time. Therefore the
    Fisher-Lagrangian velocity is exactly U, the flow is F(a,t)=a+Ut, and
    trajectory progression is the Fisher arclength implied by E.
    """
    config.validate()
    rng = np.random.default_rng(config.gene_generator_seed)

    z = config.z_grid
    time = config.tissue_times
    zz = z[None, :]
    tt = time[:, None]
    u = zz - config.flow_speed * tt

    n_t = len(time)
    n_z = len(z)
    n_g = config.n_genes
    psi = np.empty((n_t, n_z, n_g), dtype=float)
    psi_z = np.zeros_like(psi)
    psi_t = np.zeros_like(psi)

    gene_names: list[str] = []
    rows: list[dict[str, float | str]] = []

    for g in range(config.n_spatial_genes):
        baseline = rng.uniform(*config.spatial_baseline_range)
        slope = rng.uniform(*config.spatial_linear_slope_range)
        if g % 2:
            slope = -slope
        amplitude = rng.uniform(*config.spatial_bump_amplitude_range)
        center = rng.uniform(config.z_min + 0.15, config.z_max - 0.15)
        width = rng.uniform(*config.spatial_bump_width_range)

        bump, bump_prime = _gaussian(u, center=center, width=width)
        value = (
            baseline
            + slope * (u - 0.5 * (config.z_min + config.z_max))
            + amplitude * bump
        )
        derivative_u = slope + amplitude * bump_prime

        psi[..., g] = value
        psi_z[..., g] = derivative_u
        psi_t[..., g] = -config.flow_speed * derivative_u

        name = f"SPATIAL_{g + 1:02d}"
        gene_names.append(name)
        rows.append(
            {
                "gene": name,
                "program": "transported_spatial",
                "psi_baseline": float(baseline),
                "psi_linear_slope": float(slope),
                "psi_bump_amplitude": float(amplitude),
                "psi_bump_center": float(center),
                "psi_bump_width": float(width),
                "psi_developmental_slope": 0.0,
            }
        )

    developmental_slopes = np.empty(config.n_developmental_genes, dtype=float)
    for j in range(config.n_developmental_genes):
        g = config.n_spatial_genes + j
        baseline = rng.uniform(*config.developmental_baseline_range)
        slope = rng.uniform(*config.developmental_slope_range)
        developmental_slopes[j] = slope

        psi[..., g] = baseline + slope * tt
        psi_z[..., g] = 0.0
        psi_t[..., g] = slope

        name = f"DEVELOPMENTAL_{j + 1:02d}"
        gene_names.append(name)
        rows.append(
            {
                "gene": name,
                "program": "developmental",
                "psi_baseline": float(baseline),
                "psi_linear_slope": 0.0,
                "psi_bump_amplitude": 0.0,
                "psi_bump_center": np.nan,
                "psi_bump_width": np.nan,
                "psi_developmental_slope": float(slope),
            }
        )

    if np.any(psi <= 0.0):
        raise RuntimeError(
            "Fisher-flat expression truth became nonpositive; increase baselines "
            "or reduce spatial amplitudes."
        )

    mean = 0.25 * psi**2
    numerator = np.sum(psi_z * psi_t, axis=2)
    observability = np.sum(psi_z**2, axis=2)
    if np.any(observability <= 0.0):
        raise RuntimeError("Spatial observability must be positive everywhere.")
    velocity = -numerator / observability

    residual = psi_t + velocity[..., None] * psi_z
    residual_speed = np.linalg.norm(residual, axis=2)

    expected_velocity = np.full_like(velocity, config.flow_speed)
    if not np.allclose(velocity, expected_velocity, atol=1e-12, rtol=1e-12):
        raise RuntimeError("Derived velocity does not match the analytic flow speed.")

    developmental_speed = float(np.linalg.norm(developmental_slopes))
    if not np.allclose(
        residual_speed,
        developmental_speed,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise RuntimeError("Derived Fisher speed does not match analytic truth.")

    initial_position = config.initial_position_grid
    flow = initial_position[None, :] + config.flow_speed * time[:, None]
    rho = developmental_speed * np.broadcast_to(time[:, None], flow.shape).copy()

    return TruthResult(
        z=z,
        time=time,
        initial_position=initial_position,
        gene_names=tuple(gene_names),
        gene_metadata=pd.DataFrame(rows),
        psi=psi,
        mean=mean,
        psi_z=psi_z,
        psi_t=psi_t,
        velocity=velocity,
        observability=observability,
        residual_speed=residual_speed,
        flow=flow,
        rho=rho,
        developmental_speed=developmental_speed,
    )
