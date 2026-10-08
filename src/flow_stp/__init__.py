"""Flow-STP: shared anatomical axis and continuously derived Fisher dynamics."""

__version__ = "0.0.0"

from .stage1 import (
    METHOD_VERSION,
    FlowSTPConfig,
    FlowSTPModel,
    Stage1ARunConfig,
    Stage1AResult,
    run_stage1a,
)
from .stage2 import (
    FlowTrajectory,
    LocalDynamics,
    derived_velocity,
    estimate_local_dynamics,
    field_derivatives,
    integrate_flow_rk4,
    velocity_field,
    RefinementConfig,
    RefinedVelocityField,
    RefinementResult,
    fit_trajectory_velocity,
    physical_velocity,
)

__all__ = [
    "METHOD_VERSION", "FlowSTPConfig", "FlowSTPModel",
    "Stage1ARunConfig", "Stage1AResult", "run_stage1a",
    "FlowTrajectory", "LocalDynamics", "field_derivatives",
    "derived_velocity", "estimate_local_dynamics", "velocity_field",
    "integrate_flow_rk4",
    "RefinementConfig", "RefinedVelocityField", "RefinementResult",
    "fit_trajectory_velocity", "physical_velocity",
]
