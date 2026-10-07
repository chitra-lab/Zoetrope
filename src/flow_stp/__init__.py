"""Flow-STP: anatomical-axis learning followed by derived Fisher dynamics."""

__version__ = "0.0.0"

from .stage1 import (
    METHOD_VERSION,
    OPTIMIZATION_PRESET_VERSION,
    FlowSTPModel,
    Stage1ARunConfig,
    Stage1AOptimizedRunConfig,
    Stage1AResult,
    run_stage1a,
    run_stage1a_optimized,
)
from .stage2 import (
    FlowTrajectory,
    LocalDynamics,
    derived_velocity,
    estimate_local_dynamics,
    field_derivatives,
    integrate_flow_rk4,
    velocity_field,
)

__all__ = [
    "METHOD_VERSION",
    "OPTIMIZATION_PRESET_VERSION",
    "FlowSTPModel",
    "Stage1ARunConfig",
    "Stage1AOptimizedRunConfig",
    "Stage1AResult",
    "run_stage1a",
    "run_stage1a_optimized",
    "FlowTrajectory",
    "LocalDynamics",
    "field_derivatives",
    "derived_velocity",
    "estimate_local_dynamics",
    "velocity_field",
    "integrate_flow_rk4",
]
