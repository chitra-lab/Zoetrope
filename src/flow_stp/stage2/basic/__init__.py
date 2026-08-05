\
"""Public API for the bare-bones Stage 2 implementation."""

from .diagnostics import (
    basic_initial_condition_diagnostics,
    evaluate_basic_decoder_derivative,
    evaluate_basic_fields_on_grid,
)
from .model import BasicStage2Config, BasicStage2Model
from .training import (
    BasicStage2TrainingConfig,
    BasicStage2TrainingResult,
    evaluate_basic_poisson_nll,
    train_basic_baseline_one_epoch,
    train_basic_model,
    train_basic_one_epoch,
)

__all__ = [
    "BasicStage2Config",
    "BasicStage2Model",
    "BasicStage2TrainingConfig",
    "BasicStage2TrainingResult",
    "basic_initial_condition_diagnostics",
    "evaluate_basic_decoder_derivative",
    "evaluate_basic_fields_on_grid",
    "evaluate_basic_poisson_nll",
    "train_basic_baseline_one_epoch",
    "train_basic_model",
    "train_basic_one_epoch",
]
