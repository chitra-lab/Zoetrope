"""Fisher-Lagrangian spatial trajectory inference."""
from .geometry import (
    FisherGeometryResult,
    derived_velocity,
    field_derivatives,
    fisher_embedding_from_log_mean,
    fisher_geometry,
    orthogonality_error,
    spatial_progression,
)
from .model import (
    FisherLagrangianConfig,
    FisherLagrangianModel,
    LogMeanExpressionField,
    SplineLogMeanExpressionField,
    TissueCoordinateMap,
)
from .spline import AnchoredTemporalSplineBasis, FixedBSplineBasis
from .training import (
    BlockFitConfig,
    CoordinateInitConfig,
    TrainConfig,
    fit,
    fit_block_coordinate,
    fit_expression_irls,
    initialize_coordinate_maps_from_pca,
    initialize_expression_from_global_means,
    train_step,
)
from .trajectory import (
    LagrangianTrajectory,
    integrate_flow_and_rho,
    local_dynamics,
    rho_at,
)

__all__ = [
    "AnchoredTemporalSplineBasis",
    "BlockFitConfig",
    "CoordinateInitConfig",
    "FisherGeometryResult",
    "FisherLagrangianConfig",
    "FisherLagrangianModel",
    "FixedBSplineBasis",
    "LagrangianTrajectory",
    "LogMeanExpressionField",
    "SplineLogMeanExpressionField",
    "TissueCoordinateMap",
    "TrainConfig",
    "derived_velocity",
    "field_derivatives",
    "fisher_embedding_from_log_mean",
    "fisher_geometry",
    "fit",
    "fit_block_coordinate",
    "fit_expression_irls",
    "initialize_coordinate_maps_from_pca",
    "initialize_expression_from_global_means",
    "integrate_flow_and_rho",
    "local_dynamics",
    "orthogonality_error",
    "spatial_progression",
    "rho_at",
    "train_step",
]
