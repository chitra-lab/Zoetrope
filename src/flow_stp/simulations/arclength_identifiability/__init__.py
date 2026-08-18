"""Synthetic positive control for the registered transcriptomic arclength model.

This simulator reuses only the intrinsic sampling / bending geometry from the
packaged atlas positive control.  Its expression truth is specific to the new
arclength model and does not use the historical latent-rho decoder or PDE.
"""

from .simulation import (
    ArclengthSyntheticAtlas,
    ArclengthSyntheticConfig,
    developmental_derivative,
    generate_arclength_synthetic_atlas,
    inverse_logistic_flow,
    logistic_flow,
    logistic_velocity,
    true_log_mean,
    true_q,
    true_rho,
)

__all__ = [
    "ArclengthSyntheticAtlas",
    "ArclengthSyntheticConfig",
    "developmental_derivative",
    "generate_arclength_synthetic_atlas",
    "inverse_logistic_flow",
    "logistic_flow",
    "logistic_velocity",
    "true_log_mean",
    "true_q",
    "true_rho",
]
