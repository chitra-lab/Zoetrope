"""Backward-compatible import for the Fisher-score competing family."""

from .competing_models.fisher_score import (
    ScoreRayleighConfig as Config,
    fit_score_rayleigh as fit,
)

__all__ = ["Config", "fit"]
