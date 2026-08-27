"""Fisher-Lagrangian trajectory inference.

Primary learned objects: anatomical coordinate maps and a shared mean-expression
field. Flow and developmental progress are derived in Poisson Fisher geometry.
"""

from .geometry import (
    FisherGeometryResult,
    derived_velocity,
    field_derivatives,
    fisher_embedding_from_log_mean,
    fisher_geometry,
    orthogonality_error,
)
from .model import (
    FisherLagrangianConfig,
    FisherLagrangianModel,
    LogMeanExpressionField,
    TissueCoordinateMap,
)
from .training import TrainConfig, fit, train_step
from .trajectory import (
    LagrangianTrajectory,
    integrate_flow_and_rho,
    local_dynamics,
    rho_at,
)

__all__ = [
    "FisherGeometryResult",
    "FisherLagrangianConfig",
    "FisherLagrangianModel",
    "LagrangianTrajectory",
    "LogMeanExpressionField",
    "TissueCoordinateMap",
    "TrainConfig",
    "derived_velocity",
    "field_derivatives",
    "fisher_embedding_from_log_mean",
    "fisher_geometry",
    "fit",
    "integrate_flow_and_rho",
    "local_dynamics",
    "orthogonality_error",
    "rho_at",
    "train_step",
]
