"""Current-framework spatiotemporal cortical-atlas positive control.

The simulator keeps realistic bent tissue geometry and Poisson sampling while
generating expression directly from a smooth mean-expression state map E(z,t).
Velocity, flow, and trajectory progression are derived from E by the current
Fisher-Lagrangian definitions rather than generated as independent latent
objects.
"""

from .comparison_plotting import (
    plot_bent_tissue_field_comparison,
    plot_field_comparison,
    plot_flow_trajectories,
    plot_metric_by_time_heatmap,
    plot_metric_grid,
    save_figure,
)
from .config import SimulationConfig
from .geometry import (
    bend_tissue_coordinates,
    make_tissue_grid,
    true_common_coordinate,
)
from .io import save_simulation
from .plotting import save_diagnostic_plots
from .sampling import AtlasSample, sample_atlas
from .simulation import SimulationResult, run_simulation
from .truth import TruthResult, generate_truth

__all__ = [
    "AtlasSample",
    "SimulationConfig",
    "SimulationResult",
    "TruthResult",
    "bend_tissue_coordinates",
    "generate_truth",
    "make_tissue_grid",
    "plot_bent_tissue_field_comparison",
    "plot_field_comparison",
    "plot_flow_trajectories",
    "plot_metric_by_time_heatmap",
    "plot_metric_grid",
    "run_simulation",
    "sample_atlas",
    "save_diagnostic_plots",
    "save_figure",
    "save_simulation",
    "true_common_coordinate",
]
