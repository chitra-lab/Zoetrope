"""High-mobility optimization preset for the selected Stage 1a model.

This module complements :mod:`flow_stp.stage1.stage1a` without introducing a
new scientific model.  It uses the same library-offset-only likelihood,
coordinate maps, neural expression decoder, gauges, and regularizer defaults,
but raises the AdamW learning rate from 1e-3 to 2e-3.

The preset was introduced after restart diagnostics showed that the selected
Stage 1a objective can admit competing coordinate basins.  Keeping this as a
separate preset makes the optimization change explicit while preserving the
original library-offset-only defaults for reproducibility.
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


OPTIMIZATION_PRESET_VERSION = "stage1a_library_offset_high_lr_v1"


@dataclass(frozen=True)
class Stage1AOptimizedRunConfig(Stage1ARunConfig):
    """Library-offset-only Stage 1a with the high-mobility optimizer preset."""

    output_dir: Path = Path(
        "flow_stp_stage1a_library_offset_high_lr_v1"
    )

    # The scientific model is unchanged; this is the selected optimizer change.
    learning_rate: float = 2e-3


def _validate_selected_model(config: Stage1ARunConfig) -> None:
    """Prevent the optimization preset from silently changing the model."""

    expected = {
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
            "run_stage1a_optimized is reserved for the selected "
            f"library-offset-only scientific model; got {details}."
        )


def run_stage1a_optimized(
    *,
    counts_by_slice: Sequence[Any],
    spatial_by_slice: Sequence[Any],
    panel_df: pd.DataFrame,
    gene_names: Sequence[str],
    obs_df: Optional[pd.DataFrame] = None,
    config: Optional[Stage1AOptimizedRunConfig] = None,
) -> Stage1AResult:
    """Run the selected Stage 1a model with the 2e-3 learning-rate preset.

    The underlying model and runner are exactly those in :func:`run_stage1a`.
    This wrapper exists only so the empirically selected optimization setting
    is explicit, named, and reproducible alongside the original defaults.
    """

    config = (
        Stage1AOptimizedRunConfig()
        if config is None
        else config
    )

    _validate_selected_model(config)

    return run_stage1a(
        counts_by_slice=counts_by_slice,
        spatial_by_slice=spatial_by_slice,
        panel_df=panel_df,
        gene_names=gene_names,
        obs_df=obs_df,
        config=config,
    )


__all__ = [
    "OPTIMIZATION_PRESET_VERSION",
    "Stage1AOptimizedRunConfig",
    "run_stage1a_optimized",
]
