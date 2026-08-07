"""Backward-compatible import shim for the Stage 1a model.

The selected Stage 1a implementation now lives directly in ``stage1a.py``.
This module remains so older internal imports continue to work.
"""

from .stage1a import (
    FlowSTPConfig,
    FlowSTPModel,
    MLP,
    SliceCoordinateMaps,
)

__all__ = [
    "FlowSTPConfig",
    "FlowSTPModel",
    "MLP",
    "SliceCoordinateMaps",
]
