"""Experimental Stage-1 estimator zoo.

The zoo now contains only the surviving direct-score estimator plus targeted
successors motivated by the first Qian PFC benchmark.  The production
``flow_stp.stage1.stage1a`` estimator remains outside this package and is not
duplicated here.

Current zoo estimators:

- score_rayleigh_warp
- multiscale_score_rayleigh
- rkhs_score_rayleigh
- fisher_diffusion
- geometry_anchored_lowrank
- multiblock_kernel_pls
"""

from .common import AtlasData, TissueData, ZooResult, prepare_atlas
from .score_rayleigh_warp import Config as ScoreRayleighWarpConfig
from .score_rayleigh_warp import fit as fit_score_rayleigh_warp
from .multiscale_score_rayleigh import Config as MultiscaleScoreRayleighConfig
from .multiscale_score_rayleigh import fit as fit_multiscale_score_rayleigh
from .rkhs_score_rayleigh import Config as RKHSScoreRayleighConfig
from .rkhs_score_rayleigh import fit as fit_rkhs_score_rayleigh
from .fisher_diffusion import Config as FisherDiffusionConfig
from .fisher_diffusion import fit as fit_fisher_diffusion
from .geometry_anchored_lowrank import Config as GeometryAnchoredLowRankConfig
from .geometry_anchored_lowrank import fit as fit_geometry_anchored_lowrank
from .multiblock_kernel_pls import Config as MultiblockKernelPLSConfig
from .multiblock_kernel_pls import fit as fit_multiblock_kernel_pls


METHODS = {
    "score_rayleigh_warp": fit_score_rayleigh_warp,
    "multiscale_score_rayleigh": fit_multiscale_score_rayleigh,
    "rkhs_score_rayleigh": fit_rkhs_score_rayleigh,
    "fisher_diffusion": fit_fisher_diffusion,
    "geometry_anchored_lowrank": fit_geometry_anchored_lowrank,
    "multiblock_kernel_pls": fit_multiblock_kernel_pls,
}


__all__ = [
    "AtlasData",
    "TissueData",
    "ZooResult",
    "prepare_atlas",
    "ScoreRayleighWarpConfig",
    "MultiscaleScoreRayleighConfig",
    "RKHSScoreRayleighConfig",
    "FisherDiffusionConfig",
    "GeometryAnchoredLowRankConfig",
    "MultiblockKernelPLSConfig",
    "fit_score_rayleigh_warp",
    "fit_multiscale_score_rayleigh",
    "fit_rkhs_score_rayleigh",
    "fit_fisher_diffusion",
    "fit_geometry_anchored_lowrank",
    "fit_multiblock_kernel_pls",
    "METHODS",
]
