"""Paired held-out displacement comparison: arclength versus oracle contour.

The canonical predictor only sees the starting XY and predicted anatomical
depth. Ground-truth endpoint is used *after prediction* for the oracle
projection and error calculation. Failed predictions are never silently dropped.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from flow_stp.tissue_coordinates import (
    AnatomicalContourField, reconstruct_by_arclength,
    oracle_nearest_contour_point,
)


@dataclass(frozen=True)
class ContourDisplacementEvaluation:
    summary: dict[str, float | int]
    per_cell: pd.DataFrame


def evaluate_arclength_vs_oracle(
    source_xy: np.ndarray,
    true_target_xy: np.ndarray,
    source_depth: np.ndarray,
    predicted_target_depth: np.ndarray,
    source_field: AnatomicalContourField,
    target_field: AnatomicalContourField,
    *,
    cell_ids: np.ndarray | None = None,
    on_error: str = "raise",
    max_source_distance: float | None = None,
) -> ContourDisplacementEvaluation:
    """Compare both methods on identical predicted contours and held-out cells.

    on_error='record' tracks reconstruction failures and reports coverage,
    measuring both NRMSE values on exactly the same successful subset.
    """
    source = np.asarray(source_xy, float)
    truth = np.asarray(true_target_xy, float)
    initial_z = np.asarray(source_depth, float).reshape(-1)
    target_z = np.asarray(predicted_target_depth, float).reshape(-1)
    n = len(source)
    if (source.shape != (n, 2) or truth.shape != (n, 2) or
        len(initial_z) != n or len(target_z) != n or n == 0):
        raise ValueError("Expected nonempty matching [N,2] XY and [N] depth arrays.")
    if not all(np.isfinite(a).all() for a in (source, truth, initial_z, target_z)):
        raise ValueError("All evaluation inputs must be finite.")
    if on_error not in ("raise", "record"):
        raise ValueError("on_error must be 'raise' or 'record'.")
    ids = np.arange(n) if cell_ids is None else np.asarray(cell_ids)
    if len(ids) != n or len(np.unique(ids)) != n:
        raise ValueError("cell_ids must be unique and match number of cells.")
    rows = []
    for k in range(n):
        row = {"cell_id": ids[k], "source_depth": float(initial_z[k]),
               "target_depth": float(target_z[k]), "success": False, "failure": ""}
        try:
            prediction = reconstruct_by_arclength(
                source[k], initial_z[k], target_z[k], source_field,
                target_field, max_source_distance=max_source_distance,
            )
            oracle_xy, _ = oracle_nearest_contour_point(
                truth[k], target_z[k], target_field,
            )
        except (ValueError, IndexError) as exc:
            if on_error == "raise":
                raise ValueError(f"Cell {ids[k]} reconstruction failed: {exc}") from exc
            row["failure"] = str(exc)
            rows.append(row)
            continue
        row.update(
            success=True,
            source_fraction=float(prediction.source_fraction),
            source_contour_distance=float(prediction.source_contour_distance),
            predicted_x=float(prediction.xy[0]),
            predicted_y=float(prediction.xy[1]),
            oracle_x=float(oracle_xy[0]),
            oracle_y=float(oracle_xy[1]),
            motion_squared=float(np.sum((truth[k] - source[k]) ** 2)),
            arclength_error_squared=float(np.sum((prediction.xy - truth[k]) ** 2)),
            oracle_error_squared=float(np.sum((oracle_xy - truth[k]) ** 2)),
        )
        rows.append(row)
    frame = pd.DataFrame.from_records(rows)
    valid = frame.loc[frame["success"]]
    if len(valid):
        motion_sum = float(valid["motion_squared"].sum())
        arc_sum = float(valid["arclength_error_squared"].sum())
        oracle_sum = float(valid["oracle_error_squared"].sum())
        scale = float(np.sqrt(motion_sum))
        arc_nrmse = float(np.sqrt(arc_sum) / scale) if scale > 0 else float("nan")
        oracle_nrmse = float(np.sqrt(oracle_sum) / scale) if scale > 0 else float("nan")
        if oracle_sum > arc_sum + 1e-6 * max(1., arc_sum):
            raise RuntimeError("Oracle error exceeds contour prediction error; check contour consistency.")
    else:
        arc_nrmse = oracle_nrmse = float("nan")
    return ContourDisplacementEvaluation(
        summary={
            "n_requested": int(n),
            "n_success": int(len(valid)),
            "n_failure": int(n - len(valid)),
            "success_fraction": float(len(valid)/n),
            "arclength_displacement_nrmse": arc_nrmse,
            "oracle_contour_displacement_nrmse": oracle_nrmse,
            "arclength_minus_oracle_nrmse": float(arc_nrmse - oracle_nrmse),
        },
        per_cell=frame,
    )


__all__ = ["ContourDisplacementEvaluation", "evaluate_arclength_vs_oracle"]
