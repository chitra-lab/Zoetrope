"""Configuration for the simulated spatiotemporal cortical atlas."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters for the complete positive-control atlas simulation.

    The defaults reproduce the six-tissue experiment developed in the notebook:
    10,000 cells per tissue, six evenly spaced developmental times, progressive
    bending from 0 to 120 degrees, and six neuronal stopping positions at the
    centers of equal-width layers spanning the upper half of intrinsic ``z``.
    """

    n_timepoints: int = 6

    # By default there is one neuronal cohort for every post-zero tissue time.
    n_neuron_cohorts: int | None = None

    n_cells_per_tissue: int = 10_000
    random_seed: int = 101

    tissue_width: float = 4.0
    tissue_height: float = 1.0
    maximum_bend_deg: float = 120.0

    stopping_region_start: float = 0.5
    stopping_region_end: float = 1.0
    birth_time_start: float = 0.05
    birth_time_end: float = 0.55

    # FLOW_STP_ANALYTIC_FIELDS_CONFIG_BEGIN
    # F(a,t) = a + flow_speed*t
    flow_speed: float = 0.2

    # rho(z,t) = A*t + B*z*t + C*t^2
    rho_time_linear: float = 0.3
    rho_space_time: float = 0.35
    rho_time_quadratic: float = 0.2
    # FLOW_STP_ANALYTIC_FIELDS_CONFIG_END

    peak_velocity: float = 2.8
    deceleration_width: float = 0.10
    cfl: float = 0.45

    n_z: int = 401
    z_min: float = 0.0
    z_max: float = 1.0

    birth_z_center: float = 0.025
    birth_z_sigma: float = 0.018
    birth_time_sigma: float = 0.018
    cohort_total_mass: float = 1.0

    rg_z_center: float = 0.025
    rg_z_sigma: float = 0.025
    rg_total_mass: float = 1.0

    rg_rho_start: float = 0.02
    rg_rho_ceiling: float = 0.25
    rg_rho_rise_start: float = 0.05
    rg_rho_rise_duration: float = 0.80

    mature_neuron_rho: float = 1.0
    constant_q: float = 0.98
    neuron_maturation_delay: float = 0.03
    neuron_maturation_duration: float = 0.24

    density_support_fraction: float = 1e-10

    gene_generator_seed: int = 31
    genes_per_program: int = 4

    def validate(self) -> None:
        """Raise ``ValueError`` when parameters are internally inconsistent."""
        if self.n_timepoints < 2:
            raise ValueError("n_timepoints must be at least 2.")

        if self.cohort_count <= 0:
            raise ValueError(
                "n_neuron_cohorts must be positive."
            )

        if self.cohort_count != self.n_timepoints - 1:
            raise ValueError(
                "The time-indexed velocity design requires "
                "n_neuron_cohorts == n_timepoints - 1."
            )
        if self.n_cells_per_tissue <= 0:
            raise ValueError("n_cells_per_tissue must be positive.")
        if self.tissue_width <= 0.0 or self.tissue_height <= 0.0:
            raise ValueError("Tissue width and height must be positive.")
        if self.maximum_bend_deg < 0.0:
            raise ValueError("maximum_bend_deg must be nonnegative.")
        if not 0.0 <= self.stopping_region_start < self.stopping_region_end <= 1.0:
            raise ValueError("Stopping region must satisfy 0 <= start < end <= 1.")
        if not 0.0 <= self.birth_time_start <= self.birth_time_end <= 1.0:
            raise ValueError("Birth times must lie in [0,1] and be ordered.")
        if self.peak_velocity <= 0.0:
            raise ValueError("peak_velocity must be positive.")
        if self.deceleration_width <= 0.0:
            raise ValueError("deceleration_width must be positive.")
        if self.cfl <= 0.0 or self.cfl > 1.0:
            raise ValueError("cfl must lie in (0,1].")
        if self.n_z < 3:
            raise ValueError("n_z must be at least 3.")
        if not self.z_min < self.z_max:
            raise ValueError("z_min must be less than z_max.")
        if self.birth_z_sigma <= 0.0 or self.birth_time_sigma <= 0.0:
            raise ValueError("Birth-source widths must be positive.")
        if self.rg_z_sigma <= 0.0:
            raise ValueError("rg_z_sigma must be positive.")
        if self.cohort_total_mass <= 0.0 or self.rg_total_mass <= 0.0:
            raise ValueError("Population masses must be positive.")
        if not 0.0 <= self.rg_rho_start < self.rg_rho_ceiling < self.mature_neuron_rho:
            raise ValueError(
                "Require 0 <= rg_rho_start < rg_rho_ceiling < mature_neuron_rho."
            )
        if self.rg_rho_rise_duration <= 0.0:
            raise ValueError("rg_rho_rise_duration must be positive.")
        if self.neuron_maturation_delay < 0.0:
            raise ValueError("neuron_maturation_delay must be nonnegative.")
        if self.neuron_maturation_duration <= 0.0:
            raise ValueError("neuron_maturation_duration must be positive.")
        if not np.isfinite(self.constant_q) or self.constant_q <= 0.0:
            raise ValueError("constant_q must be positive.")

        final_constant_q_rho = (
            self.rg_rho_start
            + self.constant_q * float(self.tissue_times[-1])
        )
        if final_constant_q_rho > self.mature_neuron_rho + 1e-12:
            raise ValueError(
                "constant_q is too large: the linear rho trajectory "
                "would exceed mature_neuron_rho by the final time."
            )

        if self.density_support_fraction < 0.0:
            raise ValueError("density_support_fraction must be nonnegative.")
        if self.genes_per_program <= 0:
            raise ValueError("genes_per_program must be positive.")
        if np.any(self.stopping_points <= self.deceleration_width):
            raise ValueError(
                "Every stopping point must exceed the deceleration width."
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
    def cohort_count(self) -> int:
        """Resolved number of neuronal cohorts."""
        return (
            self.n_timepoints - 1
            if self.n_neuron_cohorts is None
            else int(self.n_neuron_cohorts)
        )

    @property
    def stopping_layer_edges(self) -> np.ndarray:
        return np.linspace(
            self.stopping_region_start,
            self.stopping_region_end,
            self.cohort_count + 1,
        )

    @property
    def stopping_points(self) -> np.ndarray:
        edges = self.stopping_layer_edges
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def birth_times(self) -> np.ndarray:
        return np.linspace(
            self.birth_time_start,
            self.birth_time_end,
            self.cohort_count,
        )

    @property
    def total_cells(self) -> int:
        return self.n_timepoints * self.n_cells_per_tissue

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

# FLOW_STP_TIME_INDEXED_VELOCITY_V2
