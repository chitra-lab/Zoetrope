"""Surviving Stage-1 modeling families after the Qian PFC model-selection sweeps.

The three families are:

1. Stage1a: joint spatial-coordinate / expression model (production implementation).
2. Fisher-score: direct local Fisher geometry with single- and multiscale variants.
3. Curve registration: expression-first Fisher principal-curve registration.
"""

from .fisher_score import (
    ScoreRayleighConfig,
    MultiscaleScoreRayleighConfig,
    fit_score_rayleigh,
    fit_multiscale_score_rayleigh,
)
from .curve_registration import Config as CurveRegistrationConfig
from .curve_registration import fit as fit_curve_registration
from . import stage1a

COMPETING_FAMILIES = {
    "stage1a": stage1a,
    "fisher_score": {
        "single_scale": fit_score_rayleigh,
        "multiscale": fit_multiscale_score_rayleigh,
    },
    "curve_registration": fit_curve_registration,
}

__all__ = [
    "ScoreRayleighConfig",
    "MultiscaleScoreRayleighConfig",
    "CurveRegistrationConfig",
    "fit_score_rayleigh",
    "fit_multiscale_score_rayleigh",
    "fit_curve_registration",
    "stage1a",
    "COMPETING_FAMILIES",
]
