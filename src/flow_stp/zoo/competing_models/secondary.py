"""Secondary trajectory-inference coordinate model.

The secondary model is the constrained Fisher principal-curve registration
estimator.  It is retained as the deliberately low-flexibility counterpart to
the primary neural model for real-data robustness checks.
"""

from .curve_registration import Config as SecondaryModelConfig
from .curve_registration import fit as fit_secondary

SECONDARY_DEFAULT_CONFIG = SecondaryModelConfig()

__all__ = [
    "SecondaryModelConfig",
    "SECONDARY_DEFAULT_CONFIG",
    "fit_secondary",
]
