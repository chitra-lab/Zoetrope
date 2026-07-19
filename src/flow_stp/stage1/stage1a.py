"""Preservation-first callable interface for Flow-STP Stage 1a.

The mathematical model is kept in ``_stage1a_model_impl.py``.

The original notebook training cell is kept in
``_stage1a_training_source.py`` and executed in a controlled namespace.
This preserves the validated notebook behavior while providing a clean,
reusable package call.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ._stage1a_model_impl import FlowSTPConfig, FlowSTPModel


@dataclass(frozen=True)
class Stage1ARunConfig:
    """Configuration corresponding to the original Stage 1a training cell."""

    output_dir: Path = Path(
        "flow_stp_stage1a_lambda_0p11364637"
    )

    submersion_weight: float = 0.11364637

    epochs: int = 1500
    batch_size_per_slice: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: Optional[float] = 5.0

    print_every: int = 25
    checkpoint_every: int = 100
    validation_fraction: float = 0.10
    eval_batch_size: int = 8192
    diagnostic_cells_per_slice: int = 4096

    seed: int = 20260622
    resume: bool = True

    coord_grad_min: float = 0.05
    coord_grad_samples_per_slice: Optional[int] = 256
    coord_grad_eps: float = 1e-12

    coord_hidden: tuple[int, ...] = (64, 64)
    field_hidden: tuple[int, ...] = (128, 128)
    decoder_hidden: tuple[int, ...] = (128, 128)

    n_flow_steps: int = 16
    coord_min_width: float = 0.05
    slice_effect_dim: int = 0
    use_library_size_offset: bool = True


@dataclass
class Stage1AResult:
    """Objects produced by a completed or resumed Stage 1a run."""

    model: FlowSTPModel
    summary: dict[str, Any]

    history: pd.DataFrame
    coordinate_diagnostics: pd.DataFrame
    cells: pd.DataFrame
    obs: Optional[pd.DataFrame]

    xy_norm_params: pd.DataFrame
    panel_df: pd.DataFrame
    normalized_times: np.ndarray

    output_dir: Path


_CONFIG_TO_SOURCE_NAME = {
    "submersion_weight": "SUBMERSION_WEIGHT",
    "epochs": "STAGE1A_EPOCHS",
    "batch_size_per_slice": "STAGE1A_BATCH_SIZE_PER_SLICE",
    "learning_rate": "STAGE1A_LR",
    "weight_decay": "STAGE1A_WEIGHT_DECAY",
    "grad_clip_norm": "STAGE1A_GRAD_CLIP",
    "print_every": "STAGE1A_PRINT_EVERY",
    "checkpoint_every": "STAGE1A_CHECKPOINT_EVERY",
    "validation_fraction": "STAGE1A_VALIDATION_FRACTION",
    "eval_batch_size": "STAGE1A_EVAL_BATCH_SIZE",
    "diagnostic_cells_per_slice": (
        "STAGE1A_DIAGNOSTIC_CELLS_PER_SLICE"
    ),
    "seed": "STAGE1A_SEED",
    "resume": "RESUME_EXISTING_RUN",
    "coord_grad_min": "COORD_GRAD_MIN",
    "coord_grad_samples_per_slice": (
        "COORD_GRAD_SAMPLES_PER_SLICE"
    ),
    "coord_grad_eps": "COORD_GRAD_EPS",
    "coord_hidden": "COORD_HIDDEN",
    "field_hidden": "FIELD_HIDDEN",
    "decoder_hidden": "DECODER_HIDDEN",
    "n_flow_steps": "N_FLOW_STEPS",
    "coord_min_width": "COORD_MIN_WIDTH",
    "slice_effect_dim": "SLICE_EFFECT_DIM",
    "use_library_size_offset": "USE_LIBRARY_SIZE_OFFSET",
}


def _replace_simple_assignment(
    source: str,
    variable_name: str,
    replacement_expression: str,
) -> str:
    """Replace one single-line top-level constant assignment."""
    import re

    pattern = re.compile(
        rf"(?m)^{re.escape(variable_name)}\s*=.*$"
    )

    matches = list(pattern.finditer(source))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one assignment for "
            f"{variable_name!r}, found {len(matches)}."
        )

    return pattern.sub(
        f"{variable_name} = {replacement_expression}",
        source,
        count=1,
    )


def _training_source_with_config(
    config: Stage1ARunConfig,
) -> str:
    source_resource = resources.files(__package__).joinpath(
        "_stage1a_training_source.py"
    )

    source = source_resource.read_text(encoding="utf-8")

    output_dir = Path(config.output_dir).expanduser().resolve()

    source = _replace_simple_assignment(
        source,
        "STAGE1A_ROOT",
        f"Path({str(output_dir)!r})",
    )

    for config_name, source_name in _CONFIG_TO_SOURCE_NAME.items():
        value = getattr(config, config_name)
        source = _replace_simple_assignment(
            source,
            source_name,
            repr(value),
        )

    return source


def run_stage1a(
    *,
    counts_by_slice: Sequence[Any],
    spatial_by_slice: Sequence[Any],
    panel_df: pd.DataFrame,
    gene_names: Sequence[str],
    obs_df: Optional[pd.DataFrame] = None,
    config: Optional[Stage1ARunConfig] = None,
) -> Stage1AResult:
    """Train, resume, or reload the preserved Stage 1a implementation.

    Parameters
    ----------
    counts_by_slice:
        Sequence of cell-by-gene count matrices, one per tissue slice.
    spatial_by_slice:
        Sequence of cell-by-two-coordinate arrays, one per slice.
    panel_df:
        Slice-level metadata containing ``slice_id`` and time information.
    gene_names:
        Gene names in the same order as the columns of the count matrices.
    obs_df:
        Optional cell-level metadata. When supplied, Stage 1a coordinates
        are merged into it using ``slice_id`` and within-slice row order.
    config:
        Run configuration. Defaults reproduce the original selected
        moderate-submersion notebook run.

    Returns
    -------
    Stage1AResult
        Model, summaries, diagnostics, coordinates, metadata, and paths.
    """
    if config is None:
        config = Stage1ARunConfig()

    if len(counts_by_slice) == 0:
        raise ValueError("counts_by_slice cannot be empty.")

    if len(counts_by_slice) != len(spatial_by_slice):
        raise ValueError(
            "counts_by_slice and spatial_by_slice must have "
            "the same number of slices."
        )

    runtime_namespace: dict[str, Any] = {
        "__name__": f"{__package__}._stage1a_runtime",
        "__package__": __package__,
        "FlowSTPConfig": FlowSTPConfig,
        "FlowSTPModel": FlowSTPModel,
        "counts_by_slice": counts_by_slice,
        "spatial_by_slice": spatial_by_slice,
        "panel_df": panel_df,
        "gene_names": list(gene_names),
    }

    if obs_df is not None:
        runtime_namespace["obs_df"] = obs_df

    configured_source = _training_source_with_config(config)

    source_filename = str(
        resources.files(__package__).joinpath(
            "_stage1a_training_source.py"
        )
    )

    exec(
        compile(
            configured_source,
            source_filename,
            "exec",
        ),
        runtime_namespace,
        runtime_namespace,
    )

    required_outputs = [
        "model_flow_stp",
        "stage1a_training_summary",
        "stage1a_training_history_df",
        "stage1a_coordinate_diagnostics_df",
        "stage1a_cells",
        "xy_norm_params",
        "panel_df_flow_stp",
        "times_norm_flow_stp",
    ]

    missing = [
        name
        for name in required_outputs
        if name not in runtime_namespace
    ]

    if missing:
        raise RuntimeError(
            "Stage 1a training completed without producing the "
            "expected output objects: "
            + ", ".join(missing)
        )

    return Stage1AResult(
        model=runtime_namespace["model_flow_stp"],
        summary=runtime_namespace[
            "stage1a_training_summary"
        ],
        history=runtime_namespace[
            "stage1a_training_history_df"
        ],
        coordinate_diagnostics=runtime_namespace[
            "stage1a_coordinate_diagnostics_df"
        ],
        cells=runtime_namespace["stage1a_cells"],
        obs=runtime_namespace.get("obs_flow_stp"),
        xy_norm_params=runtime_namespace["xy_norm_params"],
        panel_df=runtime_namespace["panel_df_flow_stp"],
        normalized_times=np.asarray(
            runtime_namespace["times_norm_flow_stp"],
            dtype=np.float32,
        ),
        output_dir=Path(config.output_dir).expanduser().resolve(),
    )


__all__ = [
    "FlowSTPConfig",
    "FlowSTPModel",
    "Stage1ARunConfig",
    "Stage1AResult",
    "run_stage1a",
]
