"""Callable, preservation-first Stage 1b computation API.

The validated notebook algorithms are retained verbatim in
``_stage1b_ell_source.py`` and ``_stage1b_r_source.py``. This module executes
only their definition sections, then orchestrates slice computation and table
writing without generating figures. Plotting lives in ``stage1b_plotting``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import gc
import json
import math
import re
import sys
import types
import warnings

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Stage1BEllConfig:
    output_dir: Path = Path("stage1b_master_contour_aligned_circular_ell")
    slice_ids: Optional[tuple[int, ...]] = None

    grid_max_dim: int = 360
    ambient_grid_padding_fraction: float = 0.20
    grid_eval_batch_size: int = 65536
    cell_z_eval_batch_size: int = 65536

    tissue_mask_spacing_quantile: float = 0.95
    tissue_mask_radius_factor: float = 2.75
    tissue_mask_closing_iterations: int = 2

    n_reference_levels: int = 100
    level_quantile_range: tuple[float, float] = (0.01, 0.99)
    master_level_quantile: float = 0.50
    min_contour_points: int = 25
    contour_resample_n: int = 2048
    contour_orient_by: str = "x"
    keep_all_ambient_components: bool = False

    contour_score_near_cell_grid_steps: float = 5.0
    contour_score_max_points: int = 800

    backflow_n_steps: int = 64
    backflow_method: str = "rk2"
    flow_eps: float = 1e-10
    gradient_floor_quantile: float = 0.001
    absolute_gradient_floor: float = 0.0
    max_vector_norm_grid_steps: float = 10.0
    backflow_chunk_size: int = 60000
    backflow_z_tol_fraction: float = 0.05
    contour_alignment_z_tol_fraction: float = 0.08

    n_nearest_levels_to_try: int = 5
    projection_k_segments: int = 16
    max_projection_distance_grid_steps: float = 8.0
    score_z_error_weight: float = 2.0
    score_travel_weight: float = 0.02
    score_alignment_confidence_weight: float = 0.50

    use_circular_ell_embedding: bool = True
    circular_average_top_k: int = 8
    circular_average_temperature: float = 2.0
    circular_average_max_score_gap: Optional[float] = 8.0

    save_per_slice_tables: bool = True
    save_combined_table: bool = True
    save_contour_tables: bool = True


@dataclass(frozen=True)
class Stage1BRConfig:
    output_dir: Path = Path("stage1b_deterministic_r_from_master_ell")
    slice_ids: Optional[tuple[int, ...]] = None

    n_ell_trajectories: int = 256
    n_z_samples: int = 181
    z_quantile_range: tuple[float, float] = (0.005, 0.995)
    flow_substeps_per_z_interval: int = 2
    flow_method: str = "rk2"
    flow_eps: float = 1e-10
    gradient_floor_scale: float = 1.0
    max_spatial_step_grid_steps: float = 3.0
    require_master_connected_block: bool = True

    assign_k_nearest: int = 8
    assign_idw_power: float = 2.0
    assign_z_feature_weight: float = 2.0
    clip_cell_z_to_trajectory_range: bool = True
    max_assign_feature_distance: float = math.inf

    save_per_slice_tables: bool = True
    save_combined_tables: bool = True
    save_trajectory_tables: bool = True


@dataclass
class Stage1BEllRun:
    results: dict[int, Any]
    cells: pd.DataFrame
    summary: pd.DataFrame
    failures: pd.DataFrame
    output_dir: Path
    _runtime: dict[str, Any] = field(repr=False)


@dataclass
class Stage1BRRun:
    results: dict[int, Any]
    cells: pd.DataFrame
    summary: pd.DataFrame
    trajectories: pd.DataFrame
    failures: pd.DataFrame
    output_dir: Path
    _runtime: dict[str, Any] = field(repr=False)


@dataclass
class Stage1BRun:
    ell: Stage1BEllRun
    r: Stage1BRRun


_ELL_CONFIG_GLOBALS = {
    "grid_max_dim": "GRID_MAX_DIM",
    "ambient_grid_padding_fraction": "AMBIENT_GRID_PADDING_FRACTION",
    "grid_eval_batch_size": "GRID_EVAL_BATCH_SIZE",
    "cell_z_eval_batch_size": "CELL_Z_EVAL_BATCH_SIZE",
    "tissue_mask_spacing_quantile": "TISSUE_MASK_SPACING_QUANTILE",
    "tissue_mask_radius_factor": "TISSUE_MASK_RADIUS_FACTOR",
    "tissue_mask_closing_iterations": "TISSUE_MASK_CLOSING_ITERATIONS",
    "n_reference_levels": "N_REFERENCE_LEVELS",
    "level_quantile_range": "LEVEL_QUANTILE_RANGE",
    "master_level_quantile": "MASTER_LEVEL_QUANTILE",
    "min_contour_points": "MIN_CONTOUR_POINTS",
    "contour_resample_n": "CONTOUR_RESAMPLE_N",
    "contour_orient_by": "CONTOUR_ORIENT_BY",
    "keep_all_ambient_components": "KEEP_ALL_AMBIENT_COMPONENTS",
    "contour_score_near_cell_grid_steps": "CONTOUR_SCORE_NEAR_CELL_GRID_STEPS",
    "contour_score_max_points": "CONTOUR_SCORE_MAX_POINTS",
    "backflow_n_steps": "BACKFLOW_N_STEPS",
    "backflow_method": "BACKFLOW_METHOD",
    "flow_eps": "FLOW_EPS",
    "gradient_floor_quantile": "GRADIENT_FLOOR_QUANTILE",
    "absolute_gradient_floor": "ABSOLUTE_GRADIENT_FLOOR",
    "max_vector_norm_grid_steps": "MAX_VECTOR_NORM_GRID_STEPS",
    "backflow_chunk_size": "BACKFLOW_CHUNK_SIZE",
    "backflow_z_tol_fraction": "BACKFLOW_Z_TOL_FRACTION",
    "contour_alignment_z_tol_fraction": "CONTOUR_ALIGNMENT_Z_TOL_FRACTION",
    "n_nearest_levels_to_try": "N_NEAREST_LEVELS_TO_TRY",
    "projection_k_segments": "PROJECTION_K_SEGMENTS",
    "max_projection_distance_grid_steps": "MAX_PROJECTION_DISTANCE_GRID_STEPS",
    "score_z_error_weight": "SCORE_Z_ERROR_WEIGHT",
    "score_travel_weight": "SCORE_TRAVEL_WEIGHT",
    "score_alignment_confidence_weight": "SCORE_ALIGNMENT_CONFIDENCE_WEIGHT",
    "use_circular_ell_embedding": "USE_CIRCULAR_ELL_EMBEDDING",
    "circular_average_top_k": "CIRCULAR_AVERAGE_TOP_K",
    "circular_average_temperature": "CIRCULAR_AVERAGE_TEMPERATURE",
    "circular_average_max_score_gap": "CIRCULAR_AVERAGE_MAX_SCORE_GAP",
}

_R_CONFIG_GLOBALS = {
    "n_ell_trajectories": "R_N_ELL_TRAJECTORIES",
    "n_z_samples": "R_N_Z_SAMPLES",
    "z_quantile_range": "R_Z_QUANTILE_RANGE",
    "flow_substeps_per_z_interval": "R_FLOW_SUBSTEPS_PER_Z_INTERVAL",
    "flow_method": "R_FLOW_METHOD",
    "flow_eps": "R_FLOW_EPS",
    "gradient_floor_scale": "R_GRADIENT_FLOOR_SCALE",
    "max_spatial_step_grid_steps": "R_MAX_SPATIAL_STEP_GRID_STEPS",
    "require_master_connected_block": "R_REQUIRE_MASTER_CONNECTED_BLOCK",
    "assign_k_nearest": "R_ASSIGN_K_NEAREST",
    "assign_idw_power": "R_ASSIGN_IDW_POWER",
    "assign_z_feature_weight": "R_ASSIGN_Z_FEATURE_WEIGHT",
    "clip_cell_z_to_trajectory_range": "R_CLIP_CELL_Z_TO_TRAJECTORY_RANGE",
    "max_assign_feature_distance": "R_MAX_ASSIGN_FEATURE_DISTANCE",
}


def _replace_assignment(source: str, name: str, expression: str) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(name)}\s*=.*$")
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one assignment for {name!r}; found {len(matches)}."
        )
    return pattern.sub(f"{name} = {expression}", source, count=1)


def _definition_prefix(source_name: str, output_dir: Path) -> str:
    source = resources.files(__package__).joinpath(source_name).read_text(
        encoding="utf-8"
    )
    marker = "# Run all requested slices"
    if marker not in source:
        raise RuntimeError(f"Execution marker missing from {source_name}.")
    source = source.split(marker, 1)[0]
    source = _replace_assignment(
        source,
        "OUT_DIR",
        f"Path({str(Path(output_dir).expanduser().resolve())!r})",
    )
    return source



def _exec_runtime_source(
    *,
    source: str,
    filename: str,
    namespace: dict[str, Any],
) -> dict[str, Any]:
    """Execute preserved notebook definitions in a registered module.

    Dataclass processing and postponed annotations inspect
    ``sys.modules[cls.__module__]``. A plain dictionary passed to exec()
    does not create that module entry, so the preserved Stage 1b source
    must be executed inside a real ModuleType.
    """
    module_name = str(namespace.get("__name__") or "")

    if not module_name:
        raise ValueError(
            "Runtime namespace must define a non-empty __name__."
        )

    runtime_module = types.ModuleType(module_name)
    runtime_module.__file__ = filename
    runtime_module.__package__ = namespace.get("__package__")
    runtime_module.__dict__.update(namespace)

    # Keep the module registered because returned dataclass instances retain
    # classes whose __module__ points to this runtime module.
    sys.modules[module_name] = runtime_module

    exec(
        compile(source, filename, "exec"),
        runtime_module.__dict__,
        runtime_module.__dict__,
    )

    return runtime_module.__dict__


def _json_safe(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _write_json(payload: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2)


def run_stage1b_ell(
    *,
    model: Any,
    spatial_by_slice: Sequence[Any] | Mapping[int, Any],
    xy_norm_params: Optional[Any] = None,
    panel_df: Optional[pd.DataFrame] = None,
    config: Optional[Stage1BEllConfig] = None,
) -> Stage1BEllRun:
    """Compute circular master-contour-aligned ell without plotting."""
    config = config or Stage1BEllConfig()
    output_dir = Path(config.output_dir).expanduser().resolve()

    namespace: dict[str, Any] = {
        "__name__": f"{__package__}._stage1b_ell_runtime",
        "__package__": __package__,
        "model_flow_stp": model,
        "spatial_by_slice": spatial_by_slice,
    }
    if xy_norm_params is not None:
        namespace["xy_norm_params"] = xy_norm_params
    if panel_df is not None:
        namespace["panel_df"] = panel_df

    source = _definition_prefix("_stage1b_ell_source.py", output_dir)
    namespace = _exec_runtime_source(
        source=source,
        filename="_stage1b_ell_source.py",
        namespace=namespace,
    )

    for config_name, global_name in _ELL_CONFIG_GLOBALS.items():
        namespace[global_name] = getattr(config, config_name)

    slice_ids = (
        list(config.slice_ids)
        if config.slice_ids is not None
        else list(range(len(spatial_by_slice)))
    )
    namespace["SLICE_IDS"] = [int(x) for x in slice_ids]

    tables_dir = output_dir / "tables"
    contours_dir = output_dir / "contours"
    tables_dir.mkdir(parents=True, exist_ok=True)
    contours_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    cell_frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []

    compute = namespace["compute_master_contour_aligned_ell_for_slice"]
    contours_dataframe = namespace["ambient_contours_dataframe"]

    for slice_id in slice_ids:
        slice_id = int(slice_id)
        print(f"Stage 1b ell: slice {slice_id}")
        try:
            result = compute(slice_id)
            results[slice_id] = result
            summary_rows.append(result.summary)
            cell_frames.append(result.cell_table)

            if config.save_per_slice_tables:
                result.cell_table.to_csv(
                    tables_dir
                    / f"slice_{slice_id:02d}_master_contour_aligned_circular_ell_cells.csv.gz",
                    index=False,
                    compression="gzip",
                )
            if config.save_contour_tables:
                contours_dataframe(result.contours_by_level).to_csv(
                    contours_dir
                    / f"slice_{slice_id:02d}_master_aligned_contours.csv.gz",
                    index=False,
                    compression="gzip",
                )
        except Exception as exc:
            failures.append(
                {
                    "slice_id": slice_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            warnings.warn(
                f"Stage 1b ell slice {slice_id} failed: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            gc.collect()
            torch_module = namespace.get("torch")
            if torch_module is not None and torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()

    if not results:
        raise RuntimeError(f"No Stage 1b ell slices succeeded: {failures}")

    summary = pd.DataFrame(summary_rows).sort_values("slice_id").reset_index(drop=True)
    cells = (
        pd.concat(cell_frames, ignore_index=True)
        .sort_values(["slice_id", "cell_index"])
        .reset_index(drop=True)
    )
    failures_df = pd.DataFrame(failures)

    summary.to_csv(
        tables_dir / "master_contour_aligned_circular_ell_summary_by_slice.csv",
        index=False,
    )
    if config.save_combined_table:
        cells.to_csv(
            tables_dir / "master_contour_aligned_circular_ell_all_cells.csv.gz",
            index=False,
            compression="gzip",
        )

    _write_json(
        {"config": asdict(config), "failures": failures},
        output_dir / "master_contour_aligned_circular_ell_config.json",
    )

    return Stage1BEllRun(
        results=results,
        cells=cells,
        summary=summary,
        failures=failures_df,
        output_dir=output_dir,
        _runtime=namespace,
    )


def run_stage1b_r(
    *,
    ell: Stage1BEllRun,
    model: Optional[Any] = None,
    config: Optional[Stage1BRConfig] = None,
) -> Stage1BRRun:
    """Compute deterministic constant-speed r from an ell run, without plots."""
    config = config or Stage1BRConfig()
    output_dir = Path(config.output_dir).expanduser().resolve()

    namespace: dict[str, Any] = {
        "__name__": f"{__package__}._stage1b_r_runtime",
        "__package__": __package__,
        "master_contour_aligned_circular_ell_results": ell.results,
    }
    if model is not None:
        namespace["model_flow_stp"] = model

    source = _definition_prefix("_stage1b_r_source.py", output_dir)
    namespace = _exec_runtime_source(
        source=source,
        filename="_stage1b_r_source.py",
        namespace=namespace,
    )

    for config_name, global_name in _R_CONFIG_GLOBALS.items():
        namespace[global_name] = getattr(config, config_name)

    slice_ids = (
        list(config.slice_ids)
        if config.slice_ids is not None
        else sorted(int(k) for k in ell.results)
    )
    namespace["SLICE_IDS"] = [int(x) for x in slice_ids]

    tables_dir = output_dir / "tables"
    trajectory_dir = output_dir / "trajectories"
    tables_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    cell_frames: list[pd.DataFrame] = []
    trajectory_frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []

    compute = namespace["compute_deterministic_r_for_slice"]

    for slice_id in slice_ids:
        slice_id = int(slice_id)
        print(f"Stage 1b r: slice {slice_id}")
        try:
            result = compute(slice_id)
            results[slice_id] = result
            summary_rows.append(result.summary)
            cell_frames.append(result.cell_table)
            trajectory_frames.append(result.trajectory_table)

            if config.save_per_slice_tables:
                result.cell_table.to_csv(
                    tables_dir / f"slice_{slice_id:02d}_deterministic_r_cells.csv.gz",
                    index=False,
                    compression="gzip",
                )
            if config.save_trajectory_tables:
                result.trajectory_table.to_csv(
                    trajectory_dir
                    / f"slice_{slice_id:02d}_deterministic_r_trajectories.csv.gz",
                    index=False,
                    compression="gzip",
                )
        except Exception as exc:
            failures.append(
                {
                    "slice_id": slice_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            warnings.warn(
                f"Stage 1b r slice {slice_id} failed: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            gc.collect()

    if not results:
        raise RuntimeError(f"No Stage 1b r slices succeeded: {failures}")

    summary = pd.DataFrame(summary_rows).sort_values("slice_id").reset_index(drop=True)
    cells = (
        pd.concat(cell_frames, ignore_index=True)
        .sort_values(["slice_id", "cell_index"])
        .reset_index(drop=True)
    )
    trajectories = (
        pd.concat(trajectory_frames, ignore_index=True)
        .sort_values(["slice_id", "trajectory_id", "z_target"])
        .reset_index(drop=True)
    )
    failures_df = pd.DataFrame(failures)

    summary.to_csv(
        tables_dir / "deterministic_r_summary_by_slice.csv",
        index=False,
    )
    if config.save_combined_tables:
        cells.to_csv(
            tables_dir / "deterministic_r_all_cells.csv.gz",
            index=False,
            compression="gzip",
        )
        trajectories.to_csv(
            trajectory_dir / "deterministic_r_all_trajectories.csv.gz",
            index=False,
            compression="gzip",
        )

    _write_json(
        {"config": asdict(config), "failures": failures},
        output_dir / "deterministic_r_config.json",
    )

    return Stage1BRRun(
        results=results,
        cells=cells,
        summary=summary,
        trajectories=trajectories,
        failures=failures_df,
        output_dir=output_dir,
        _runtime=namespace,
    )


def run_stage1b(
    *,
    model: Any,
    spatial_by_slice: Sequence[Any] | Mapping[int, Any],
    xy_norm_params: Optional[Any] = None,
    panel_df: Optional[pd.DataFrame] = None,
    ell_config: Optional[Stage1BEllConfig] = None,
    r_config: Optional[Stage1BRConfig] = None,
) -> Stage1BRun:
    ell = run_stage1b_ell(
        model=model,
        spatial_by_slice=spatial_by_slice,
        xy_norm_params=xy_norm_params,
        panel_df=panel_df,
        config=ell_config,
    )
    r = run_stage1b_r(ell=ell, model=model, config=r_config)
    return Stage1BRun(ell=ell, r=r)


__all__ = [
    "Stage1BEllConfig",
    "Stage1BRConfig",
    "Stage1BEllRun",
    "Stage1BRRun",
    "Stage1BRun",
    "run_stage1b_ell",
    "run_stage1b_r",
    "run_stage1b",
]
