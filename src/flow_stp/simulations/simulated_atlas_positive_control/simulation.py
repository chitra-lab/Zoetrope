"""Top-level orchestration for the current positive-control atlas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import SimulationConfig
from .sampling import AtlasSample, sample_atlas
from .truth import TruthResult, generate_truth


@dataclass(frozen=True)
class SimulationResult:
    """All latent truth and observed data produced by one simulation run."""

    config: SimulationConfig
    truth: TruthResult
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

    def summary(self) -> dict[str, Any]:
        return {
            "n_timepoints": self.config.n_timepoints,
            "n_cells_per_tissue": self.config.n_cells_per_tissue,
            "total_cells": len(self.atlas.obs),
            "n_genes": len(self.truth.gene_names),
            "n_spatial_genes": self.config.n_spatial_genes,
            "n_developmental_genes": self.config.n_developmental_genes,
            "count_matrix_shape": list(self.atlas.counts.shape),
            "tissue_times": self.truth.time.tolist(),
            "bend_angles_deg": self.config.bend_angles_deg.tolist(),
            "flow_speed": self.config.flow_speed,
            "developmental_fisher_speed": self.truth.developmental_speed,
            "observed_count_mean": float(self.atlas.counts.to_numpy().mean()),
            "generating_mean": float(self.atlas.expected_mean.to_numpy().mean()),
        }


def run_simulation(config: SimulationConfig | None = None) -> SimulationResult:
    """Generate current-model truth, bent tissues, and Poisson observations."""
    config = SimulationConfig() if config is None else config
    config.validate()
    truth = generate_truth(config)
    atlas = sample_atlas(config, truth)
    return SimulationResult(config=config, truth=truth, atlas=atlas)
