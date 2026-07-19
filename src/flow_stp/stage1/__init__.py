"""Stage 1 anatomical-axis learning interfaces."""

from .stage1a import (
    FlowSTPConfig,
    FlowSTPModel,
    Stage1ARunConfig,
    Stage1AResult,
    run_stage1a,
)

__all__ = [
    "FlowSTPConfig",
    "FlowSTPModel",
    "Stage1ARunConfig",
    "Stage1AResult",
    "run_stage1a",
]

# BEGIN FLOW_STP_STAGE1B_PUBLIC_API
from .stage1b import (
    Stage1BEllConfig,
    Stage1BRConfig,
    Stage1BEllRun,
    Stage1BRRun,
    Stage1BRun,
    run_stage1b_ell,
    run_stage1b_r,
    run_stage1b,
)
from .stage1b_plotting import (
    Stage1BPlotConfig,
    Stage1BPlotArtifacts,
    plot_stage1b_ell_slice,
    plot_stage1b_r_slice,
    plot_stage1b_ell,
    plot_stage1b_r,
    plot_stage1b,
)

try:
    __all__
except NameError:
    __all__ = []

for _name in (
    "Stage1BEllConfig", "Stage1BRConfig", "Stage1BEllRun",
    "Stage1BRRun", "Stage1BRun", "run_stage1b_ell",
    "run_stage1b_r", "run_stage1b", "Stage1BPlotConfig",
    "Stage1BPlotArtifacts", "plot_stage1b_ell_slice",
    "plot_stage1b_r_slice", "plot_stage1b_ell",
    "plot_stage1b_r", "plot_stage1b",
):
    if _name not in __all__:
        __all__.append(_name)
del _name
# END FLOW_STP_STAGE1B_PUBLIC_API
