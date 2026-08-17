"""Flow-STP Stage 2 models on a frozen Stage-1 anatomical coordinate.

Stage 1 supplies the anatomical coordinate z. Stage 2 implementations are kept
as separate models so their dynamical assumptions can be tested independently:
the basic centered-rho model, the monotone flow-aware model, and the Eulerian
neural-PDE model.
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
from .neural_pde import (
    NEURAL_PDE_ARCHITECTURE_VERSION,
    NeuralPDEStage2Config,
    NeuralPDEStage2Model,
    NeuralPDETrainingConfig,
    evaluate_neural_pde_poisson_nll,
    fit_neural_pde,
    make_neural_pde_baseline_optimizer,
    make_neural_pde_optimizer,
    train_neural_pde_baseline_one_epoch,
    train_neural_pde_one_epoch,
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
    "NEURAL_PDE_ARCHITECTURE_VERSION",
    "NeuralPDEStage2Config",
    "NeuralPDEStage2Model",
    "NeuralPDETrainingConfig",
    "evaluate_neural_pde_poisson_nll",
    "fit_neural_pde",
    "make_neural_pde_baseline_optimizer",
    "make_neural_pde_optimizer",
    "train_neural_pde_baseline_one_epoch",
    "train_neural_pde_one_epoch",
]
