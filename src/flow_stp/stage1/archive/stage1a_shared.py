"""Named presets for the two shared-time Stage-1 coordinate decoders.

All scientific components besides the coordinate decoder/gauge choice are
identical to the selected library-offset Fisher-flat Stage-1 model.

Variants:
    shared_time + per_tissue gauge
    shared_time + global gauge
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from .stage1a import (
    Stage1AResult,
    Stage1ARunConfig,
    run_stage1a,
)


SHARED_TIME_PER_TISSUE_VERSION = (
    "stage1a_library_offset_fisher_flat_shared_time_per_tissue_v1"
)
SHARED_TIME_GLOBAL_VERSION = (
    "stage1a_library_offset_fisher_flat_shared_time_global_v1"
)


@dataclass(frozen=True)
class Stage1ASharedTimePerTissueConfig(Stage1ARunConfig):
    """Shared alpha(x,y,t) with the original per-tissue robust affine gauge."""

    output_dir: Path = Path(
        "flow_stp_stage1a_shared_time_per_tissue_v1"
    )
    coordinate_decoder: str = "shared_time"
    gauge_mode: str = "per_tissue"
    learning_rate: float = 2e-3


@dataclass(frozen=True)
class Stage1ASharedTimeGlobalConfig(Stage1ARunConfig):
    """Shared alpha(x,y,t) with one robust affine gauge across all tissues."""

    output_dir: Path = Path(
        "flow_stp_stage1a_shared_time_global_v1"
    )
    coordinate_decoder: str = "shared_time"
    gauge_mode: str = "global"
    learning_rate: float = 2e-3


def _validate_shared_config(
    config: Stage1ARunConfig,
    *,
    gauge_mode: str,
) -> None:
    expected = {
        "coordinate_decoder": "shared_time",
        "gauge_mode": gauge_mode,
        "use_library_size_offset": True,
        "submersion_weight": 0.0,
        "scale_variation_weight": 0.0,
        "coverage_prior_weight": 0.0,
        "nuisance_weight": 0.0,
        "slice_effect_dim": 0,
        "max_missing_each_end": 0.0,
        "initial_missing": 0.0,
    }
    mismatches = {
        name: (getattr(config, name), value)
        for name, value in expected.items()
        if getattr(config, name) != value
    }
    if mismatches:
        details = ", ".join(
            f"{name}={actual!r} (expected {expected_value!r})"
            for name, (actual, expected_value) in mismatches.items()
        )
        raise ValueError(
            "Shared-time Stage 1 wrappers only change the coordinate "
            f"decoder/gauge structure; got {details}."
        )


def run_stage1a_shared_time_per_tissue(
    *,
    counts_by_slice: Sequence[Any],
    spatial_by_slice: Sequence[Any],
    panel_df: pd.DataFrame,
    gene_names: Sequence[str],
    obs_df: Optional[pd.DataFrame] = None,
    config: Optional[Stage1ASharedTimePerTissueConfig] = None,
) -> Stage1AResult:
    """Run shared alpha(x,y,t) with a per-tissue robust affine gauge."""
    config = (
        Stage1ASharedTimePerTissueConfig()
        if config is None
        else config
    )
    _validate_shared_config(
        config,
        gauge_mode="per_tissue",
    )
    return run_stage1a(
        counts_by_slice=counts_by_slice,
        spatial_by_slice=spatial_by_slice,
        panel_df=panel_df,
        gene_names=gene_names,
        obs_df=obs_df,
        config=config,
    )


def run_stage1a_shared_time_global(
    *,
    counts_by_slice: Sequence[Any],
    spatial_by_slice: Sequence[Any],
    panel_df: pd.DataFrame,
    gene_names: Sequence[str],
    obs_df: Optional[pd.DataFrame] = None,
    config: Optional[Stage1ASharedTimeGlobalConfig] = None,
) -> Stage1AResult:
    """Run shared alpha(x,y,t) with one robust affine gauge across time."""
    config = (
        Stage1ASharedTimeGlobalConfig()
        if config is None
        else config
    )
    _validate_shared_config(
        config,
        gauge_mode="global",
    )
    return run_stage1a(
        counts_by_slice=counts_by_slice,
        spatial_by_slice=spatial_by_slice,
        panel_df=panel_df,
        gene_names=gene_names,
        obs_df=obs_df,
        config=config,
    )


__all__ = [
    "SHARED_TIME_PER_TISSUE_VERSION",
    "SHARED_TIME_GLOBAL_VERSION",
    "Stage1ASharedTimePerTissueConfig",
    "Stage1ASharedTimeGlobalConfig",
    "run_stage1a_shared_time_per_tissue",
    "run_stage1a_shared_time_global",
]
