"""Frozen direct flow-aware Stage-2 package.

The original ``flow_stp.stage2.basic`` package remains separate and unchanged.
"""

from .model import (
    DirectFlowStage2,
    FlowAwareStage2Config,
    FlowAwareStage2Model,
    FROZEN_ARCHITECTURE_VERSION,
    MonotoneInverseFlow,
)
from .training import (
    FlowAwareTrainingConfig,
    fit_flow_aware,
    make_flow_aware_baseline_optimizer,
    make_flow_aware_optimizer,
    train_flow_aware_baseline_one_epoch,
    train_flow_aware_one_epoch,
)

__all__ = [
    "DirectFlowStage2",
    "FlowAwareStage2Config",
    "FlowAwareStage2Model",
    "FROZEN_ARCHITECTURE_VERSION",
    "MonotoneInverseFlow",
    "FlowAwareTrainingConfig",
    "fit_flow_aware",
    "make_flow_aware_baseline_optimizer",
    "make_flow_aware_optimizer",
    "train_flow_aware_baseline_one_epoch",
    "train_flow_aware_one_epoch",
]
