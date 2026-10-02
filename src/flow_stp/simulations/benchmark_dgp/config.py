"""Configuration for the method-independent developmental benchmark simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class BenchmarkSimulationConfig:
    """Parameters for the comparative spatial-TI benchmark.

    The default benchmark deliberately separates the latent biological process
    from any one estimator.  The exact structured regime is obtained with
    ``off_axis_strength=0`` and ``parallel_strength=0``.  Those two parameters
    independently violate the one-dimensional spatial-sufficiency assumption
    and the local molecular-persistence identification assumption.
    """

    tissue_times: Tuple[float, ...] = (0.0, 0.25, 0.55, 0.80, 1.0)
    replicates_per_time: int = 1
    cells_per_tissue: int = 3000
    latent_population_size: Optional[int] = None

    n_genes: int = 128
    n_a_grid: int = 161
    n_time_grid: int = 161
    n_spatial_programs: int = 6
    n_molecular_programs: int = 6
    n_off_axis_programs: int = 4

    # Physical flow.  The z(1-z) factor in the implementation keeps the flow
    # inside [0,1] without making the velocity field an estimator artifact.
    velocity_scale: float = 0.22
    velocity_temporal_modulation: float = 0.30
    velocity_spatial_modulation: float = 0.20

    # Fisher-flat expression-state generator Psi = 2 sqrt(E).
    psi_baseline: float = 4.0
    psi_baseline_jitter: float = 0.35
    psi_floor: float = 0.50
    spatial_program_scale: float = 0.45
    molecular_program_scale: float = 0.18
    off_axis_program_scale: float = 0.30

    # Controlled assumption violations.
    off_axis_strength: float = 0.0
    parallel_strength: float = 0.0

    # Observation model.  NB is the main benchmark; Poisson is a sanity check.
    observation_model: str = "negative_binomial"
    library_log_sd: float = 0.35
    nb_dispersion_log_mean: float = 3.4
    nb_dispersion_log_sd: float = 0.35

    # Curved-ribbon tissue geometry with independent section coordinate frames.
    tissue_half_width: float = 0.12
    width_modulation: float = 0.20
    curvature_base: float = 0.12
    curvature_time_trend: float = 0.20
    curvature_jitter: float = 0.05
    random_rotation: bool = True
    random_reflection: bool = True
    translation_sd: float = 0.35

    random_seed: int = 20261002

    @property
    def a_grid(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, int(self.n_a_grid), dtype=float)

    @property
    def time_grid(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, int(self.n_time_grid), dtype=float)

    @property
    def n_tissues(self) -> int:
        return len(self.tissue_times) * int(self.replicates_per_time)

    @property
    def n_observed_cells(self) -> int:
        return int(self.cells_per_tissue) * self.n_tissues

    @property
    def resolved_latent_population_size(self) -> int:
        if self.latent_population_size is None:
            return 2 * self.n_observed_cells
        return int(self.latent_population_size)

    def validate(self) -> None:
        if len(self.tissue_times) < 2:
            raise ValueError("Need at least two tissue times.")
        times = np.asarray(self.tissue_times, dtype=float)
        if np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError("tissue_times must be finite and strictly increasing.")
        if times[0] < 0.0 or times[-1] > 1.0:
            raise ValueError("tissue_times must lie in [0,1].")
        if self.replicates_per_time < 1 or self.cells_per_tissue < 1:
            raise ValueError("replicates_per_time and cells_per_tissue must be positive.")
        if self.resolved_latent_population_size < self.n_observed_cells:
            raise ValueError(
                "latent_population_size must be at least the total number of "
                "observed cells so destructive snapshots can use unique particles."
            )
        if self.n_genes < 4:
            raise ValueError("n_genes must be at least 4.")
        if self.n_a_grid < 21 or self.n_time_grid < 21:
            raise ValueError("n_a_grid and n_time_grid must both be at least 21.")
        if min(
            self.n_spatial_programs,
            self.n_molecular_programs,
            self.n_off_axis_programs,
        ) < 1:
            raise ValueError("All latent-program counts must be positive.")
        if self.velocity_scale < 0.0:
            raise ValueError("velocity_scale must be nonnegative.")
        if abs(self.velocity_temporal_modulation) >= 1.0:
            raise ValueError("velocity_temporal_modulation must have magnitude < 1.")
        if abs(self.velocity_spatial_modulation) >= 1.0:
            raise ValueError("velocity_spatial_modulation must have magnitude < 1.")
        if self.psi_floor <= 0.0 or self.psi_baseline <= self.psi_floor:
            raise ValueError("Require psi_baseline > psi_floor > 0.")
        if self.off_axis_strength < 0.0 or self.parallel_strength < 0.0:
            raise ValueError("Assumption-violation strengths must be nonnegative.")
        if self.observation_model not in {"negative_binomial", "poisson"}:
            raise ValueError("observation_model must be 'negative_binomial' or 'poisson'.")
        if self.library_log_sd < 0.0:
            raise ValueError("library_log_sd must be nonnegative.")
        if self.tissue_half_width <= 0.0:
            raise ValueError("tissue_half_width must be positive.")
