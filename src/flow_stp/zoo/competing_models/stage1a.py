"""Stage1a competing-model family entry point.

The canonical Stage1a implementation remains in ``flow_stp.stage1.stage1a``
because downstream Stage1b code depends on that production API.  This module
re-exports the implementation from the competing-models namespace so the three
surviving modeling families can be browsed together without duplicating code.
"""

from ...stage1.stage1a import (
    FlowSTPConfig,
    FlowSTPModel,
    SliceCoordinateMaps,
    Stage1ARunConfig,
    Stage1AResult,
)

# Re-export the public runner when present in the production module.
try:
    from ...stage1.stage1a import run_stage1a
except ImportError:  # pragma: no cover - compatibility with older revisions
    run_stage1a = None

__all__ = [
    "FlowSTPConfig",
    "FlowSTPModel",
    "SliceCoordinateMaps",
    "Stage1ARunConfig",
    "Stage1AResult",
    "run_stage1a",
]
