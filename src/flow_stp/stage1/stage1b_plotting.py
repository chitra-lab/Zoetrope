"""Plotting-only API for computed Stage 1b results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import math

import matplotlib.pyplot as plt
import numpy as np

from .stage1b import Stage1BEllRun, Stage1BRRun, Stage1BRun


@dataclass(frozen=True)
class Stage1BPlotConfig:
    display: bool = True
    save_png: bool = True
    save_pdf: bool = False
    dpi: int = 300
    max_points_per_plot: int = 140000
    point_size: float = 0.7
    point_alpha: float = 0.86

    ell_cmap: str = "turbo"
    r_cmap: str = "viridis"
    invalid_color: str = "0.80"

    contour_color: str = "black"
    contour_alpha: float = 0.18
    contour_linewidth: float = 0.50
    master_contour_color: str = "black"
    master_contour_linewidth: float = 1.50
    contour_plot_stride: int = 5

    trajectory_color: str = "black"
    trajectory_alpha: float = 0.18
    trajectory_linewidth: float = 0.45
    trajectory_plot_stride: int = 10

    figsize_per_slice: tuple[float, float] = (6.4, 5.6)
    make_overview: bool = True
    overview_ncols: int = 5
    make_ambient_z_plots: bool = True


@dataclass
class Stage1BPlotArtifacts:
    files: list[Path] = field(default_factory=list)
    figures: list[Any] = field(default_factory=list, repr=False)


def _save_figure(fig, base_path: Path, config: Stage1BPlotConfig, files: list[Path]):
    if config.save_png:
        png = base_path.with_suffix(".png")
        fig.savefig(png, dpi=config.dpi, bbox_inches="tight")
        files.append(png)
    if config.save_pdf:
        pdf = base_path.with_suffix(".pdf")
        fig.savefig(pdf, bbox_inches="tight")
        files.append(pdf)


def _finish_figure(fig, config: Stage1BPlotConfig, figures: list[Any]):
    if config.display:
        plt.show()
        figures.append(fig)
    else:
        plt.close(fig)


def _apply_ell_style(runtime: dict[str, Any], config: Stage1BPlotConfig):
    runtime.update(
        {
            "MAX_POINTS_PER_PLOT": config.max_points_per_plot,
            "POINT_SIZE": config.point_size,
            "POINT_ALPHA": config.point_alpha,
            "ELL_CMAP": config.ell_cmap,
            "INVALID_COLOR": config.invalid_color,
            "CONTOUR_COLOR": config.contour_color,
            "CONTOUR_ALPHA": config.contour_alpha,
            "CONTOUR_LINEWIDTH": config.contour_linewidth,
            "MASTER_CONTOUR_COLOR": config.master_contour_color,
            "MASTER_CONTOUR_LINEWIDTH": config.master_contour_linewidth,
            "CONTOUR_PLOT_STRIDE": config.contour_plot_stride,
            "FIGSIZE_PER_SLICE": config.figsize_per_slice,
        }
    )


def _apply_r_style(runtime: dict[str, Any], config: Stage1BPlotConfig):
    runtime.update(
        {
            "MAX_POINTS_PER_PLOT": config.max_points_per_plot,
            "POINT_SIZE": config.point_size,
            "POINT_ALPHA": config.point_alpha,
            "R_CMAP": config.r_cmap,
            "INVALID_COLOR": config.invalid_color,
            "TRAJECTORY_COLOR": config.trajectory_color,
            "TRAJECTORY_ALPHA": config.trajectory_alpha,
            "TRAJECTORY_LINEWIDTH": config.trajectory_linewidth,
            "TRAJECTORY_PLOT_STRIDE": config.trajectory_plot_stride,
            "FIGSIZE_PER_SLICE": config.figsize_per_slice,
        }
    )


def plot_stage1b_ell_slice(
    ell: Stage1BEllRun,
    slice_id: int,
    *,
    ambient_z: bool = False,
    config: Optional[Stage1BPlotConfig] = None,
):
    config = config or Stage1BPlotConfig()
    _apply_ell_style(ell._runtime, config)
    result = ell.results[int(slice_id)]
    function_name = (
        "plot_ambient_z_and_master_aligned_contours"
        if ambient_z
        else "plot_master_contour_aligned_ell_heatmap"
    )
    return ell._runtime[function_name](result)


def plot_stage1b_r_slice(
    r: Stage1BRRun,
    slice_id: int,
    *,
    config: Optional[Stage1BPlotConfig] = None,
):
    config = config or Stage1BPlotConfig()
    _apply_r_style(r._runtime, config)
    return r._runtime["plot_deterministic_r_heatmap"](
        r.results[int(slice_id)]
    )


def plot_stage1b_ell(
    ell: Stage1BEllRun,
    *,
    config: Optional[Stage1BPlotConfig] = None,
) -> Stage1BPlotArtifacts:
    config = config or Stage1BPlotConfig()
    _apply_ell_style(ell._runtime, config)
    plot_dir = ell.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    figures: list[Any] = []

    heatmap = ell._runtime["plot_master_contour_aligned_ell_heatmap"]
    ambient = ell._runtime["plot_ambient_z_and_master_aligned_contours"]

    for slice_id in sorted(ell.results):
        result = ell.results[slice_id]
        fig, _ = heatmap(result)
        _save_figure(
            fig,
            plot_dir
            / f"slice_{slice_id:02d}_master_contour_aligned_circular_ell_heatmap",
            config,
            files,
        )
        _finish_figure(fig, config, figures)

        if config.make_ambient_z_plots:
            fig, _ = ambient(result)
            _save_figure(
                fig,
                plot_dir / f"slice_{slice_id:02d}_ambient_z_master_aligned_contours",
                config,
                files,
            )
            _finish_figure(fig, config, figures)

    if config.make_overview:
        n = len(ell.results)
        ncols = max(1, min(config.overview_ncols, n))
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.8 * ncols, 3.4 * nrows),
            constrained_layout=True,
        )
        axes = np.asarray(axes).reshape(-1)
        for ax, slice_id in zip(axes, sorted(ell.results)):
            heatmap(ell.results[slice_id], ax=ax, show_colorbar=False)
        for ax in axes[len(ell.results):]:
            ax.axis("off")
        fig.suptitle("Circular master-contour-aligned ell", fontsize=14)
        _save_figure(
            fig,
            plot_dir / "all_slices_master_contour_aligned_circular_ell_heatmaps",
            config,
            files,
        )
        _finish_figure(fig, config, figures)

    return Stage1BPlotArtifacts(files=files, figures=figures)


def plot_stage1b_r(
    r: Stage1BRRun,
    *,
    config: Optional[Stage1BPlotConfig] = None,
) -> Stage1BPlotArtifacts:
    config = config or Stage1BPlotConfig()
    _apply_r_style(r._runtime, config)
    plot_dir = r.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    figures: list[Any] = []
    heatmap = r._runtime["plot_deterministic_r_heatmap"]

    for slice_id in sorted(r.results):
        fig, _ = heatmap(r.results[slice_id])
        _save_figure(
            fig,
            plot_dir / f"slice_{slice_id:02d}_deterministic_r_heatmap",
            config,
            files,
        )
        _finish_figure(fig, config, figures)

    if config.make_overview:
        n = len(r.results)
        ncols = max(1, min(config.overview_ncols, n))
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.8 * ncols, 3.4 * nrows),
            constrained_layout=True,
        )
        axes = np.asarray(axes).reshape(-1)
        for ax, slice_id in zip(axes, sorted(r.results)):
            heatmap(r.results[slice_id], ax=ax, show_colorbar=False)
        for ax in axes[len(r.results):]:
            ax.axis("off")
        fig.suptitle("Deterministic Stage 1b constant-speed r", fontsize=14)
        _save_figure(
            fig,
            plot_dir / "all_slices_deterministic_r_heatmaps",
            config,
            files,
        )
        _finish_figure(fig, config, figures)

    return Stage1BPlotArtifacts(files=files, figures=figures)


def plot_stage1b(
    stage1b: Stage1BRun,
    *,
    config: Optional[Stage1BPlotConfig] = None,
) -> dict[str, Stage1BPlotArtifacts]:
    config = config or Stage1BPlotConfig()
    return {
        "ell": plot_stage1b_ell(stage1b.ell, config=config),
        "r": plot_stage1b_r(stage1b.r, config=config),
    }


__all__ = [
    "Stage1BPlotConfig",
    "Stage1BPlotArtifacts",
    "plot_stage1b_ell_slice",
    "plot_stage1b_r_slice",
    "plot_stage1b_ell",
    "plot_stage1b_r",
    "plot_stage1b",
]
