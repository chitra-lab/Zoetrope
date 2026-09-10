"""Backward-compatible import for the multiscale Fisher-score variant."""

from .competing_models.fisher_score import (
    MultiscaleScoreRayleighConfig as Config,
    fit_multiscale_score_rayleigh as fit,
)

__all__ = ["Config", "fit"]
