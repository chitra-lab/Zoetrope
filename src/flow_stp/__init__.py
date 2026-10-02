"""Flow-STP package."""

__version__ = "0.0.0"

from .model import FlowSTPConfig, FlowSTPModel

__all__ = [
    "FlowSTPConfig",
    "FlowSTPModel",
]

# BEGIN FLOW_STP_STAGE1_PUBLIC_API
from .stage1 import (
    OPTIMIZATION_PRESET_VERSION,
    Stage1ARunConfig,
    Stage1AOptimizedRunConfig,
    Stage1AResult,
    run_stage1a,
    run_stage1a_optimized,
)

try:
    __all__
except NameError:
    __all__ = []

for _stage1_public_name in (
    "OPTIMIZATION_PRESET_VERSION",
    "Stage1ARunConfig",
    "Stage1AOptimizedRunConfig",
    "Stage1AResult",
    "run_stage1a",
    "run_stage1a_optimized",
):
    if _stage1_public_name not in __all__:
        __all__.append(_stage1_public_name)

del _stage1_public_name
# END FLOW_STP_STAGE1_PUBLIC_API

# BEGIN FLOW_STP_STAGE1B_PUBLIC_API
from .stage1 import (
    Stage1BEllConfig,
    Stage1BRConfig,
    Stage1BRun,
    Stage1BPlotConfig,
    run_stage1b_ell,
    run_stage1b_r,
    run_stage1b,
    plot_stage1b_ell,
    plot_stage1b_r,
    plot_stage1b,
)

try:
    __all__
except NameError:
    __all__ = []

for _name in (
    "Stage1BEllConfig", "Stage1BRConfig", "Stage1BRun",
    "Stage1BPlotConfig", "run_stage1b_ell", "run_stage1b_r",
    "run_stage1b", "plot_stage1b_ell", "plot_stage1b_r",
    "plot_stage1b",
):
    if _name not in __all__:
        __all__.append(_name)
del _name
# END FLOW_STP_STAGE1B_PUBLIC_API

# BEGIN FLOW_STP_STAGE1B_ROOT_API
from .stage1 import (
    Stage1BEllConfig,
    Stage1BRConfig,
    Stage1BEllRun,
    Stage1BRRun,
    Stage1BRun,
    run_stage1b_ell,
    run_stage1b_r,
    run_stage1b,
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

for _stage1b_root_name in (
    "Stage1BEllConfig",
    "Stage1BRConfig",
    "Stage1BEllRun",
    "Stage1BRRun",
    "Stage1BRun",
    "run_stage1b_ell",
    "run_stage1b_r",
    "run_stage1b",
    "Stage1BPlotConfig",
    "Stage1BPlotArtifacts",
    "plot_stage1b_ell_slice",
    "plot_stage1b_r_slice",
    "plot_stage1b_ell",
    "plot_stage1b_r",
    "plot_stage1b",
):
    if _stage1b_root_name not in __all__:
        __all__.append(_stage1b_root_name)

del _stage1b_root_name
# END FLOW_STP_STAGE1B_ROOT_API

# BEGIN FLOW_STP_QIAN_PFC_PUBLIC_API
from .data import (
    DEFAULT_DATA_ROOT,
    QianPFCConfig,
    QianPFCData,
    build_qian_pfc_panel,
    load_qian_pfc,
    load_qian_pfc_for_stage1,
    merge_shared_metadata,
    read_h5ad_rows_direct,
    read_shared_metadata,
    validate_qian_pfc_data,
)

try:
    __all__
except NameError:
    __all__ = []

for _qian_pfc_public_name in (
    "DEFAULT_DATA_ROOT",
    "QianPFCConfig",
    "QianPFCData",
    "build_qian_pfc_panel",
    "load_qian_pfc",
    "load_qian_pfc_for_stage1",
    "merge_shared_metadata",
    "read_h5ad_rows_direct",
    "read_shared_metadata",
    "validate_qian_pfc_data",
):
    if _qian_pfc_public_name not in __all__:
        __all__.append(_qian_pfc_public_name)

del _qian_pfc_public_name
# END FLOW_STP_QIAN_PFC_PUBLIC_API
