"""Active Stage-1 competitors for simulation and real-data model selection.

The zoo contains three conceptually distinct coordinate families plus two
Stage1a spline-expression variants that isolate expression representation and
optimization strategy without changing the scientific estimands.
"""

from .fisher_score import (
    ScoreRayleighConfig,
    MultiscaleScoreRayleighConfig,
    fit_score_rayleigh,
    fit_multiscale_score_rayleigh,
)
from .curve_registration import Config as CurveRegistrationConfig
from .curve_registration import fit as fit_curve_registration
from .stage1a_spline import (
    Stage1aSplineModelConfig,
    Stage1aSplineJointConfig,
    Stage1aSplineProfiledConfig,
    Stage1aSplineModel,
    fit_stage1a_spline_joint,
    fit_stage1a_spline_profiled,
)
from . import stage1a

COMPETING_FAMILIES = {
    "stage1a_nn_nn": stage1a,
    "stage1a_nn_spline_joint": fit_stage1a_spline_joint,
    "stage1a_nn_spline_profiled": fit_stage1a_spline_profiled,
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
    "Stage1aSplineModelConfig",
    "Stage1aSplineJointConfig",
    "Stage1aSplineProfiledConfig",
    "Stage1aSplineModel",
    "fit_score_rayleigh",
    "fit_multiscale_score_rayleigh",
    "fit_curve_registration",
    "fit_stage1a_spline_joint",
    "fit_stage1a_spline_profiled",
    "stage1a",
    "COMPETING_FAMILIES",
]
