"""Top-level orchestration for the complete positive-control atlas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import SimulationConfig
from .decoder import DecoderResult, generate_expression_surfaces
from .pde import PDEResult, solve_cortical_pde
from .sampling import AtlasSample, sample_atlas


@dataclass(frozen=True)
class SimulationResult:
    """All latent truth and observed data produced by one simulation run."""

    config: SimulationConfig
    pde: PDEResult
    decoder: DecoderResult
    atlas: AtlasSample

    @property
    def obs(self):
        return self.atlas.obs

    @property
    def counts(self):
        return self.atlas.counts

    @property
    def expected_mean(self):
        return self.atlas.expected_mean

    @property
    def cohort_table(self):
        return self.pde.cohort_table

    def summary(self) -> dict[str, Any]:
        return {
            "n_timepoints": self.config.n_timepoints,
            "n_cells_per_tissue": self.config.n_cells_per_tissue,
            "total_cells": len(self.atlas.obs),
            "n_genes": len(self.decoder.gene_names),
            "count_matrix_shape": list(self.atlas.counts.shape),
            "tissue_times": self.pde.time.tolist(),
            "bend_angles_deg": self.config.bend_angles_deg.tolist(),
            "stopping_points": self.config.stopping_points.tolist(),
            "pde_explicit_steps": self.pde.explicit_steps,
            "pde_cfl_dt": self.pde.dt_cfl,
            "pde_dz": self.pde.dz,
            "baseline_anchor_error": self.decoder.baseline_anchor_error,
            "observed_count_mean": float(self.atlas.counts.to_numpy().mean()),
            "generating_mean": float(self.atlas.expected_mean.to_numpy().mean()),
        }


def run_simulation(config: SimulationConfig | None = None) -> SimulationResult:
    """Run geometry, dynamics, PDE, decoder, intrinsic sampling, and Poisson draws."""
    config = SimulationConfig() if config is None else config
    config.validate()
    pde = solve_cortical_pde(config)
    decoder = generate_expression_surfaces(config, pde)
    atlas = sample_atlas(config, pde, decoder)
    return SimulationResult(config=config, pde=pde, decoder=decoder, atlas=atlas)
