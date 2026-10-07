"""Stage 2: derivative-based Fisher dynamics derived from Stage-1 E(z,t)."""

from .dynamics import (
    ExpressionField,
    FlowTrajectory,
    LocalDynamics,
    derived_velocity,
    estimate_local_dynamics,
    field_derivatives,
    integrate_flow_rk4,
    velocity_field,
)

__all__ = [
    "ExpressionField",
    "LocalDynamics",
    "FlowTrajectory",
    "field_derivatives",
    "derived_velocity",
    "estimate_local_dynamics",
    "velocity_field",
    "integrate_flow_rk4",
]
