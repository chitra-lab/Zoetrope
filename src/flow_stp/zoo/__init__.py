"""Stage-1 estimator comparison package.

The active simulation comparison contains the retained coordinate estimators,
controlled Stage1a expression variants, and the discrete geometric-atlas
formulation for gauge-equivariant adjacent-time transport.
"""

from .common import AtlasData, TissueData, ZooResult, prepare_atlas
from .competing_models import (
    COMPETING_FAMILIES,
    CurveRegistrationConfig,
    DiscreteGeometricAtlasConfig,
    MultiscaleScoreRayleighConfig,
    ScoreRayleighConfig,
    fit_curve_registration,
    fit_discrete_geometric_atlas,
    fit_multiscale_score_rayleigh,
    fit_score_rayleigh,
    stage1a,
)

# Backward-compatible names used by existing benchmark scripts.
ScoreRayleighWarpConfig = ScoreRayleighConfig

METHODS = {
    "score_rayleigh_warp": fit_score_rayleigh,
    "multiscale_score_rayleigh": fit_multiscale_score_rayleigh,
    "curve_registered_principal": fit_curve_registration,
    "discrete_geometric_atlas": fit_discrete_geometric_atlas,
}

__all__ = [
    "AtlasData",
    "TissueData",
    "ZooResult",
    "prepare_atlas",
    "ScoreRayleighConfig",
    "ScoreRayleighWarpConfig",
    "MultiscaleScoreRayleighConfig",
    "CurveRegistrationConfig",
    "DiscreteGeometricAtlasConfig",
    "fit_score_rayleigh",
    "fit_multiscale_score_rayleigh",
    "fit_curve_registration",
    "fit_discrete_geometric_atlas",
    "stage1a",
    "COMPETING_FAMILIES",
    "METHODS",
]
