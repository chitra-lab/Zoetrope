"""Spatiotemporal cortical-atlas positive-control simulation.

The package combines intrinsic tissue geometry, progressive block-arc bending,
smoothly stopping neuronal migration, radial-glia and neuronal developmental
sources, conservative density/rho-moment transport, the centered-rho bare-bones
decoder truth, uniform intrinsic cell sampling, and Poisson observations.
"""

from .config import SimulationConfig
from .decoder import DecoderResult, GENE_PROGRAMS, generate_expression_surfaces
from .dynamics import (
    neuron_q,
    neuron_rho,
    radial_glia_q,
    radial_glia_rho,
    smootherstep,
    stopping_velocity,
)
from .geometry import bend_tissue_coordinates, make_tissue_grid
from .io import save_simulation
from .pde import PDEResult, solve_cortical_pde
from .plotting import save_diagnostic_plots
from .sampling import AtlasSample, sample_atlas
from .simulation import SimulationResult, run_simulation

__all__ = [
    "AtlasSample",
    "DecoderResult",
    "GENE_PROGRAMS",
    "PDEResult",
    "SimulationConfig",
    "SimulationResult",
    "bend_tissue_coordinates",
    "generate_expression_surfaces",
    "make_tissue_grid",
    "neuron_q",
    "neuron_rho",
    "radial_glia_q",
    "radial_glia_rho",
    "run_simulation",
    "sample_atlas",
    "save_diagnostic_plots",
    "save_simulation",
    "smootherstep",
    "solve_cortical_pde",
    "stopping_velocity",
]
