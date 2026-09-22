"""Configuration for the current-framework simulated cortical atlas."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters for the Fisher-Lagrangian positive-control atlas.

    The default experiment contains six progressively bent tissue slices with
    10,000 cells per tissue. Expression is generated directly from a smooth
    mean-expression state map E(z,t). Velocity, flow, and trajectory progression
    are derived from E by the same Fisher-Lagrangian definitions used by the
    current method.
    """

    n_timepoints: int = 6
    n_cells_per_tissue: int = 10_000
    random_seed: int = 101

    tissue_width: float = 4.0
    tissue_height: float = 1.0
    maximum_bend_deg: float = 120.0

    n_z: int = 401
    z_min: float = 0.0
    z_max: float = 1.0

    # Ground-truth anatomical flow F(a,t)=a+flow_speed*t.
    flow_speed: float = 0.2
    trajectory_n_initial: int = 101

    # Fisher-flat expression truth Psi=2*sqrt(E).
    n_spatial_genes: int = 16
    n_developmental_genes: int = 8
    gene_generator_seed: int = 31

    spatial_baseline_range: tuple[float, float] = (5.0, 7.0)
    spatial_linear_slope_range: tuple[float, float] = (0.35, 0.70)
    spatial_bump_amplitude_range: tuple[float, float] = (0.50, 1.50)
    spatial_bump_width_range: tuple[float, float] = (0.12, 0.25)

    developmental_baseline_range: tuple[float, float] = (5.0, 7.0)
    developmental_slope_range: tuple[float, float] = (0.30, 0.80)

    def validate(self) -> None:
        """Raise ValueError when parameters are internally inconsistent."""
        if self.n_timepoints < 2:
            raise ValueError("n_timepoints must be at least 2.")
        if self.n_cells_per_tissue <= 0:
            raise ValueError("n_cells_per_tissue must be positive.")
        if self.tissue_width <= 0.0 or self.tissue_height <= 0.0:
            raise ValueError("Tissue width and height must be positive.")
        if self.maximum_bend_deg < 0.0:
            raise ValueError("maximum_bend_deg must be nonnegative.")
        if self.n_z < 3:
            raise ValueError("n_z must be at least 3.")
        if not self.z_min < self.z_max:
            raise ValueError("z_min must be less than z_max.")
        if not np.isfinite(self.flow_speed) or self.flow_speed <= 0.0:
            raise ValueError("flow_speed must be finite and positive.")
        if self.flow_speed >= self.z_max - self.z_min:
            raise ValueError(
                "flow_speed must be smaller than the observed z span so that "
                "some full trajectories remain inside the observed domain."
            )
        if self.trajectory_n_initial < 2:
            raise ValueError("trajectory_n_initial must be at least 2.")
        if self.n_spatial_genes <= 0:
            raise ValueError("n_spatial_genes must be positive.")
        if self.n_developmental_genes <= 0:
            raise ValueError("n_developmental_genes must be positive.")

        for name, bounds in (
            ("spatial_baseline_range", self.spatial_baseline_range),
            ("spatial_linear_slope_range", self.spatial_linear_slope_range),
            ("spatial_bump_amplitude_range", self.spatial_bump_amplitude_range),
            ("spatial_bump_width_range", self.spatial_bump_width_range),
            ("developmental_baseline_range", self.developmental_baseline_range),
            ("developmental_slope_range", self.developmental_slope_range),
        ):
            low, high = bounds
            if (
                not np.isfinite(low)
                or not np.isfinite(high)
                or low <= 0.0
                or high < low
            ):
                raise ValueError(
                    f"{name} must contain finite positive ordered bounds."
                )

        theta = np.deg2rad(self.maximum_bend_deg)
        if theta > 0.0:
            radius_midline = self.tissue_width / theta
            if radius_midline <= 0.5 * self.tissue_height:
                raise ValueError(
                    "maximum_bend_deg is too tight for the tissue width and height."
                )

    @property
    def tissue_times(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.n_timepoints)

    @property
    def bend_angles_deg(self) -> np.ndarray:
        return np.linspace(0.0, self.maximum_bend_deg, self.n_timepoints)

    @property
    def z_grid(self) -> np.ndarray:
        return np.linspace(self.z_min, self.z_max, self.n_z)

    @property
    def initial_position_grid(self) -> np.ndarray:
        """Initial positions whose full trajectories stay inside observed support."""
        return np.linspace(
            self.z_min,
            self.z_max - self.flow_speed,
            self.trajectory_n_initial,
        )

    @property
    def total_cells(self) -> int:
        return self.n_timepoints * self.n_cells_per_tissue

    @property
    def n_genes(self) -> int:
        return self.n_spatial_genes + self.n_developmental_genes

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
