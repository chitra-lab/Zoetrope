"""Independent registered transcriptomic arclength model."""

from .coordinate_expression import (
    ArclengthConfig,
    ArclengthModel,
    SliceCoordinateMaps,
    TemporalBasisExpressionField,
    TruncatedCubicTimeBasis,
    normalize_time,
)
from .training import FitHistory, fit_arclength_model
from .derivatives import ExpressionGrid, evaluate_expression_grid, expression_jvp
from .registration import RegistrationResult, solve_smooth_registration
from .trajectory import FlowResult, assign_rho_to_observations, integrate_flow_and_arclength, no_flow_arclength
from .diagnostics import coordinate_gradient_summary, flow_summary, registration_summary

ARCLENGTH_MODEL_VERSION = "arclength-temporal-basis-flow-v1-2026-08-18"

__all__ = [
    "ARCLENGTH_MODEL_VERSION",
    "ArclengthConfig",
    "ArclengthModel",
    "SliceCoordinateMaps",
    "TemporalBasisExpressionField",
    "TruncatedCubicTimeBasis",
    "normalize_time",
    "FitHistory",
    "fit_arclength_model",
    "ExpressionGrid",
    "expression_jvp",
    "evaluate_expression_grid",
    "RegistrationResult",
    "solve_smooth_registration",
    "FlowResult",
    "integrate_flow_and_arclength",
    "no_flow_arclength",
    "assign_rho_to_observations",
    "coordinate_gradient_summary",
    "registration_summary",
    "flow_summary",
]
