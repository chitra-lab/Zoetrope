"""Diagnostics for coordinate geometry, registration, and flow."""
from __future__ import annotations

from typing import Mapping
import numpy as np


def registration_summary(result) -> Mapping[str, float]:
    interior = result.V[:, 1:-1]
    explained = result.explained_fraction[:, 1:-1]
    return {
        "max_abs_V": float(np.max(np.abs(interior))) if interior.size else 0.0,
        "median_abs_V": float(np.median(np.abs(interior))) if interior.size else 0.0,
        "median_explained_fraction": float(np.nanmedian(explained)) if explained.size else 0.0,
        "mean_q0": float(np.mean(result.q0)),
        "mean_q_registered": float(np.mean(result.q)),
        "boundary_max_abs_V": float(max(np.max(np.abs(result.V[:, 0])), np.max(np.abs(result.V[:, -1])))),
    }


def flow_summary(flow) -> Mapping[str, float | bool]:
    diffs = np.diff(flow.F, axis=1)
    return {
        "strictly_monotone": bool(np.all(diffs > 0)),
        "min_flow_spacing": float(np.min(diffs)) if diffs.size else float("nan"),
        "rho_nonnegative": bool(np.all(flow.rho >= -1e-10)),
        "rho_nondecreasing": bool(np.all(np.diff(flow.rho, axis=0) >= -1e-8)),
        "max_rho": float(np.max(flow.rho)),
    }


def coordinate_gradient_summary(grad_norm) -> Mapping[str, float]:
    x = np.asarray(grad_norm, dtype=float).reshape(-1)
    return {
        "min": float(np.min(x)),
        "p01": float(np.quantile(x, 0.01)),
        "median": float(np.median(x)),
        "p99": float(np.quantile(x, 0.99)),
    }
