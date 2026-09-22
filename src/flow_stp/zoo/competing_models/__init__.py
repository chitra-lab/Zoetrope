"""Active Stage-1 competitors for simulation and real-data model selection.

The zoo contains the retained coordinate families, controlled Stage1a
expression variants, and a discrete geometric-atlas formulation that estimates
adjacent-time transport directly modulo tissue-chart gauge.
"""

from .fisher_score import (
    ScoreRayleighConfig,
    MultiscaleScoreRayleighConfig,
    fit_score_rayleigh,
    fit_multiscale_score_rayleigh,
)
from .curve_registration import Config as CurveRegistrationConfig
from .curve_registration import fit as fit_curve_registration
from .discrete_geometric_atlas import (
    Config as DiscreteGeometricAtlasConfig,
    AdjacentTransportMap,
    DiscreteGeometricAtlas,
    fit as fit_discrete_geometric_atlas,
    fit_adjacent_transport,
)
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
    "discrete_geometric_atlas": fit_discrete_geometric_atlas,
}

__all__ = [
    "ScoreRayleighConfig",
    "MultiscaleScoreRayleighConfig",
    "CurveRegistrationConfig",
    "DiscreteGeometricAtlasConfig",
    "AdjacentTransportMap",
    "DiscreteGeometricAtlas",
    "Stage1aSplineModelConfig",
    "Stage1aSplineJointConfig",
    "Stage1aSplineProfiledConfig",
    "Stage1aSplineModel",
    "fit_score_rayleigh",
    "fit_multiscale_score_rayleigh",
    "fit_curve_registration",
    "fit_discrete_geometric_atlas",
    "fit_adjacent_transport",
    "fit_stage1a_spline_joint",
    "fit_stage1a_spline_profiled",
    "stage1a",
    "COMPETING_FAMILIES",
]
