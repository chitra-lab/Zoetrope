\
"""Flow-STP Stage 2: centered neural rho with a general expression decoder.

Stage 1 supplies a frozen anatomical coordinate z. Stage 2 fits a t=0 neural
anatomical baseline, a centered scalar neural developmental coordinate, and a
general decoder D(z,rho). The former constrained/PDE implementation has been
removed from the package.
"""

from .data import Stage2Dataset, normalize_times
from .basic import (
    BasicStage2Config,
    BasicStage2Model,
    BasicStage2TrainingConfig,
    BasicStage2TrainingResult,
    basic_initial_condition_diagnostics,
    evaluate_basic_decoder_derivative,
    evaluate_basic_fields_on_grid,
    evaluate_basic_poisson_nll,
    train_basic_baseline_one_epoch,
    train_basic_model,
    train_basic_one_epoch,
)

__all__ = [
    "Stage2Dataset",
    "normalize_times",
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
