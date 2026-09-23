"""Frozen model-selection interface.

Only two coordinate estimators are retained after the positive-control model
selection study:

* primary: flexible Stage1a neural coordinate / neural expression model.
* secondary: constrained Fisher principal-curve registration model.

The removed zoo variants remain available in git history but are intentionally
absent from the active package API.
"""

from .primary import (
    PRIMARY_BELLS_AND_WHISTLES_CONFIG,
    PRIMARY_DEFAULT_CONFIG,
    Stage1ARunConfig,
    fit_primary,
)
from .secondary import (
    SECONDARY_DEFAULT_CONFIG,
    SecondaryModelConfig,
    fit_secondary,
)

MODELS = {
    "primary": fit_primary,
    "secondary": fit_secondary,
}

__all__ = [
    "PRIMARY_DEFAULT_CONFIG",
    "PRIMARY_BELLS_AND_WHISTLES_CONFIG",
    "SECONDARY_DEFAULT_CONFIG",
    "Stage1ARunConfig",
    "SecondaryModelConfig",
    "fit_primary",
    "fit_secondary",
    "MODELS",
]
