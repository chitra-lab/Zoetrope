"""Experimental Stage-1 estimator zoo.

The frozen geometric estimand is shared across methods; the modules differ only
in how they estimate the local Fisher geometry, process coordinate, and
anatomical distortion nuisance.

Ranked methods implemented here (the existing ``flow_stp.stage1.stage1a`` is
rank 4 and is intentionally not duplicated):

1. joint_likelihood_fisher_warp
2. meanfield_rayleigh_warp
3. score_rayleigh_warp
5. independent_register
6. eigenfield_warp
7. principal_arclength_warp
"""

from .common import AtlasData, TissueData, ZooResult, prepare_atlas
from .joint_likelihood_fisher_warp import Config as JointLikelihoodFisherWarpConfig
from .joint_likelihood_fisher_warp import fit as fit_joint_likelihood_fisher_warp
from .meanfield_rayleigh_warp import Config as MeanFieldRayleighWarpConfig
from .meanfield_rayleigh_warp import fit as fit_meanfield_rayleigh_warp
from .score_rayleigh_warp import Config as ScoreRayleighWarpConfig
from .score_rayleigh_warp import fit as fit_score_rayleigh_warp
from .independent_register import Config as IndependentRegisterConfig
from .independent_register import fit as fit_independent_register
from .eigenfield_warp import Config as EigenfieldWarpConfig
from .eigenfield_warp import fit as fit_eigenfield_warp
from .principal_arclength_warp import Config as PrincipalArclengthWarpConfig
from .principal_arclength_warp import fit as fit_principal_arclength_warp

METHODS = {
    "joint_likelihood_fisher_warp": fit_joint_likelihood_fisher_warp,
    "meanfield_rayleigh_warp": fit_meanfield_rayleigh_warp,
    "score_rayleigh_warp": fit_score_rayleigh_warp,
    "independent_register": fit_independent_register,
    "eigenfield_warp": fit_eigenfield_warp,
    "principal_arclength_warp": fit_principal_arclength_warp,
}

__all__ = [
    "AtlasData",
    "TissueData",
    "ZooResult",
    "prepare_atlas",
    "JointLikelihoodFisherWarpConfig",
    "MeanFieldRayleighWarpConfig",
    "ScoreRayleighWarpConfig",
    "IndependentRegisterConfig",
    "EigenfieldWarpConfig",
    "PrincipalArclengthWarpConfig",
    "fit_joint_likelihood_fisher_warp",
    "fit_meanfield_rayleigh_warp",
    "fit_score_rayleigh_warp",
    "fit_independent_register",
    "fit_eigenfield_warp",
    "fit_principal_arclength_warp",
    "METHODS",
]
