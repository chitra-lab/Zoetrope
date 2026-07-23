"""Public API for the frozen-axis Stage 2 dynamics model."""

from .data import Stage2Dataset, normalize_times
from .diagnostics import (
    evaluate_fields_on_grid,
    flow_health_diagnostics,
    initial_condition_diagnostics,
)
from .model import Stage2Config, Stage2DynamicsModel
from .training import evaluate_poisson_nll, train_one_epoch

__all__ = [
    "Stage2Config",
    "Stage2Dataset",
    "Stage2DynamicsModel",
    "evaluate_fields_on_grid",
    "evaluate_poisson_nll",
    "flow_health_diagnostics",
    "initial_condition_diagnostics",
    "normalize_times",
    "train_one_epoch",
]
