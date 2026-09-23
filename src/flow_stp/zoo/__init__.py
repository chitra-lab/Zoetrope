"""Frozen Stage-1 model-selection interface.

The active package intentionally exposes only the selected primary neural model
and the secondary Fisher principal-curve model.
"""

from .common import AtlasData, TissueData, ZooResult, prepare_atlas
from .competing_models import (
    MODELS,
    PRIMARY_DEFAULT_CONFIG,
    SECONDARY_DEFAULT_CONFIG,
    SecondaryModelConfig,
    Stage1ARunConfig,
    fit_primary,
    fit_secondary,
)

METHODS = MODELS

__all__ = [
    "AtlasData",
    "TissueData",
    "ZooResult",
    "prepare_atlas",
    "PRIMARY_DEFAULT_CONFIG",
    "SECONDARY_DEFAULT_CONFIG",
    "Stage1ARunConfig",
    "SecondaryModelConfig",
    "fit_primary",
    "fit_secondary",
    "MODELS",
    "METHODS",
]
