"""Primary trajectory-inference coordinate model.

The primary model is the production Stage1a neural coordinate / neural
expression estimator.  Model selection is frozen: this wrapper intentionally
exposes only the selected production implementation and its fixed defaults.
"""

from ...stage1.stage1a import (
    FlowSTPConfig,
    FlowSTPModel,
    SliceCoordinateMaps,
    Stage1ARunConfig,
    Stage1AResult,
    run_stage1a,
)

# Frozen model-selection default.  Dataset-specific output paths may still be
# supplied with dataclasses.replace without changing the scientific defaults.
PRIMARY_DEFAULT_CONFIG = Stage1ARunConfig()

fit_primary = run_stage1a

__all__ = [
    "FlowSTPConfig",
    "FlowSTPModel",
    "SliceCoordinateMaps",
    "Stage1ARunConfig",
    "Stage1AResult",
    "PRIMARY_DEFAULT_CONFIG",
    "fit_primary",
]
