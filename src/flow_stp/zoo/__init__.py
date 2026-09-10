"""Stage-1 estimator comparison package.

After the frozen Qian PFC model-selection sweeps, the active comparison is
organized around three modeling families in ``zoo.competing_models``:

1. production Stage1a;
2. the Fisher-score/Rayleigh family, including the multiscale robustness variant;
3. expression-first Fisher principal-curve registration.

Historical unsuccessful experimental estimators have been removed from the
active package.
"""

from .common import AtlasData, TissueData, ZooResult, prepare_atlas
from .competing_models import (
    COMPETING_FAMILIES,
    CurveRegistrationConfig,
    MultiscaleScoreRayleighConfig,
    ScoreRayleighConfig,
    fit_curve_registration,
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
    "fit_score_rayleigh",
    "fit_multiscale_score_rayleigh",
    "fit_curve_registration",
    "stage1a",
    "COMPETING_FAMILIES",
    "METHODS",
]
