"""Reusable model-comparison plots for the simulated-atlas positive control.

These helpers package the PI-facing plots used in confirmatory synthetic
experiments so new models can be compared without copying notebook-specific
plotting code.  All functions are model-agnostic: callers provide metric
tables or truth/model fields on the simulator's common ``(t, z)`` grids.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# =============================================================================
# Reusable positive-control comparison plots
# =============================================================================

def _display_label(
    name: str,
    labels: dict[str, str] | None,
) -> str:
    if labels is None:
        return str(name)
    return str(labels.get(name, name))


def plot_metric_grid(
    results,
    metric_specs,
    *,
    model_order,
    model_labels: dict[str, str] | None = None,
    model_column: str = "model",
    title: str | None = None,
    ncols: int = 2,
):
    """Plot a reusable PI-facing grid of model-recovery metrics.

    Parameters
    ----------
    results
        Table-like object (typically a pandas DataFrame) with one or more rows
        per model. Repeated rows are shown as boxplots; a single row is shown
        as a bar.
    metric_specs
        Sequence of dictionaries. Each dictionary requires ``column`` and may
        provide ``title``, ``ylabel``, ``ylim``, and ``reference``.
    model_order
        Model names in display order.
    model_labels
        Optional mapping from model names to human-readable labels.

    Returns
    -------
    (fig, axes)
        Matplotlib figure and 2-D axes array.

    Notes
    -----
    This generalizes the confirmatory notebook's 2x2 metric panel so future
    positive-control experiments can reuse the same presentation without
    copying notebook plotting code.
    """
    import matplotlib.pyplot as plt

    specs = list(metric_specs)
    if not specs:
        raise ValueError("metric_specs must contain at least one metric.")

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(specs) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5 * ncols, 4.0 * nrows),
        squeeze=False,
    )

    for ax, spec in zip(axes.ravel(), specs):
        column = str(spec["column"])
        groups = [
            np.asarray(
                results.loc[
                    results[model_column] == model,
                    column,
                ].dropna(),
                dtype=float,
            )
            for model in model_order
        ]
        labels = [
            _display_label(model, model_labels)
            for model in model_order
        ]

        if all(len(group) <= 1 for group in groups):
            values = [
                float(group[0]) if len(group) == 1 else np.nan
                for group in groups
            ]
            bars = ax.bar(labels, values)
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    ax.text(
                        bar.get_x() + 0.5 * bar.get_width(),
                        bar.get_height(),
                        f"{value:.4f}",
                        ha="center",
                        va="bottom",
                    )
        else:
            ax.boxplot(groups, tick_labels=labels)

        if spec.get("reference") is not None:
            ax.axhline(
                float(spec["reference"]),
                linestyle="--",
                linewidth=1.0,
            )

        if spec.get("ylim") is not None:
            lo, hi = spec["ylim"]
            ax.set_ylim(float(lo), float(hi))

        ax.set_ylabel(
            str(spec.get("ylabel", column))
        )
        ax.set_title(
            str(spec.get("title", column))
        )

    for ax in axes.ravel()[len(specs):]:
        ax.axis("off")

    if title is not None:
        fig.suptitle(str(title))

    fig.tight_layout()
    return fig, axes


def plot_metric_by_time_heatmap(
    summary,
    *,
    value_column: str,
    model_order,
    times,
    model_labels: dict[str, str] | None = None,
    model_column: str = "model",
    time_column: str = "time",
    title: str | None = None,
    colorbar_label: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    annotation_format: str = ".3f",
):
    """Plot an annotated model-by-time recovery heatmap.

    This reproduces the confirmatory notebook's Spearman-by-time presentation
    but works for any scalar metric, including RMSE.
    """
    import matplotlib.pyplot as plt

    pivot = (
        summary
        .pivot(
            index=model_column,
            columns=time_column,
            values=value_column,
        )
        .reindex(model_order)
        .reindex(columns=np.asarray(times, dtype=float))
    )

    fig, ax = plt.subplots(figsize=(9, 3.6))

    values = pivot.values.astype(float)
    masked = np.ma.masked_invalid(values)

    im = ax.imshow(
        masked,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [f"{float(t):.1f}" for t in pivot.columns]
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(
        [
            _display_label(model, model_labels)
            for model in pivot.index
        ]
    )
    ax.set_xlabel("developmental time")

    if title is not None:
        ax.set_title(str(title))

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            label = (
                "NA"
                if not np.isfinite(value)
                else format(float(value), annotation_format)
            )
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
            )

    fig.colorbar(
        im,
        ax=ax,
        label=(
            str(colorbar_label)
            if colorbar_label is not None
            else value_column
        ),
    )

    fig.tight_layout()
    return fig, ax


def plot_bent_tissue_field_comparison(
    *,
    config,
    z_axis,
    fields,
    times=None,
    bends=None,
    title: str | None = None,
    colorbar_label: str = "field value",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    n_tangential: int = 161,
    n_depth: int = 61,
    figsize=None,
):
    """Plot truth/model scalar fields over the bent positive-control tissues.

    Parameters
    ----------
    config
        ``SimulationConfig`` or compatible object with tissue dimensions.
    z_axis
        1-D intrinsic z grid on which each field is evaluated.
    fields
        Ordered mapping ``row_label -> array``. Each array must have shape
        ``[n_timepoints, len(z_axis)]``.
    times, bends
        Optional time and bend arrays. Defaults to the values in ``config``.
    title
        Optional figure title.
    colorbar_label
        Shared colorbar label.

    Returns
    -------
    (fig, axes)
        Matplotlib figure and axes.

    Notes
    -----
    This is the reusable version of the confirmatory notebook's bent-tissue
    rho heatmap. It can be used for rho, z, q, or any other scalar field
    defined on the common intrinsic z grid.
    """
    import matplotlib.pyplot as plt

    from .geometry import make_tissue_grid

    z_axis = np.asarray(z_axis, dtype=float).reshape(-1)
    if z_axis.ndim != 1 or len(z_axis) < 2:
        raise ValueError("z_axis must be a one-dimensional grid with >=2 points.")

    if times is None:
        times = np.asarray(config.tissue_times, dtype=float)
    else:
        times = np.asarray(times, dtype=float)

    if bends is None:
        bends = np.asarray(config.bend_angles_deg, dtype=float)
    else:
        bends = np.asarray(bends, dtype=float)

    if len(times) != len(bends):
        raise ValueError("times and bends must have the same length.")

    field_items = list(fields.items())
    if not field_items:
        raise ValueError("fields must contain at least one row.")

    normalized_fields = []
    for label, field in field_items:
        arr = np.asarray(field, dtype=float)
        expected = (len(times), len(z_axis))
        if arr.shape != expected:
            raise ValueError(
                f"Field {label!r} has shape {arr.shape}; expected {expected}."
            )
        normalized_fields.append((str(label), arr))

    if vmin is None or vmax is None:
        all_values = np.concatenate(
            [arr.reshape(-1) for _, arr in normalized_fields]
        )
        finite = all_values[np.isfinite(all_values)]
        if finite.size == 0:
            raise ValueError("fields contain no finite values.")
        if vmin is None:
            vmin = float(np.min(finite))
        if vmax is None:
            vmax = float(np.max(finite))

    if abs(float(vmax) - float(vmin)) < 1e-12:
        vmin = float(vmin) - 0.01
        vmax = float(vmax) + 0.01

    nrows = len(normalized_fields)
    ncols = len(times)

    if figsize is None:
        figsize = (3.0 * ncols, 2.25 * nrows + 0.8)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
    )

    mappable = None

    for col, (t, bend) in enumerate(zip(times, bends)):
        tissue = make_tissue_grid(
            width=float(config.tissue_width),
            height=float(config.tissue_height),
            bend_angle_deg=float(bend),
            n_tangential=int(n_tangential),
            n_depth=int(n_depth),
        )

        z_grid = np.asarray(tissue["z"], dtype=float)
        depth_axis = z_grid[:, 0]

        for row_idx, (row_label, field) in enumerate(normalized_fields):
            values_1d = np.interp(
                depth_axis,
                z_axis,
                field[col],
            )
            values = np.broadcast_to(
                values_1d[:, None],
                z_grid.shape,
            ).copy()

            ax = axes[row_idx, col]
            mappable = ax.pcolormesh(
                tissue["x"],
                tissue["y"],
                values,
                shading="auto",
                cmap=cmap,
                vmin=float(vmin),
                vmax=float(vmax),
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(
                    f"t={float(t):.1f}\n"
                    f"bend={float(bend):.0f}°"
                )

            if col == 0:
                ax.set_ylabel(row_label)

    if title is not None:
        fig.suptitle(str(title))

    fig.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        label=str(colorbar_label),
        shrink=0.75,
    )

    return fig, axes


def plot_field_comparison(
    *,
    z_axis,
    times,
    fields,
    title: str | None = None,
    x_label: str = "z",
    y_label: str = "time",
    colorbar_label: str = "field value",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize=None,
):
    """Plot truth/model time-by-z fields side by side.

    This is the reusable version of the confirmatory notebook's V(z,t)
    visualization.
    """
    import matplotlib.pyplot as plt

    z_axis = np.asarray(z_axis, dtype=float).reshape(-1)
    times = np.asarray(times, dtype=float).reshape(-1)

    field_items = []
    for label, field in fields.items():
        arr = np.asarray(field, dtype=float)
        expected = (len(times), len(z_axis))
        if arr.shape != expected:
            raise ValueError(
                f"Field {label!r} has shape {arr.shape}; expected {expected}."
            )
        field_items.append((str(label), arr))

    if not field_items:
        raise ValueError("fields must contain at least one panel.")

    if vmin is None or vmax is None:
        all_values = np.concatenate(
            [field.reshape(-1) for _, field in field_items]
        )
        finite = all_values[np.isfinite(all_values)]
        if finite.size == 0:
            raise ValueError("fields contain no finite values.")
        if vmin is None:
            vmin = float(np.min(finite))
        if vmax is None:
            vmax = float(np.max(finite))

    if abs(float(vmax) - float(vmin)) < 1e-12:
        vmin = float(vmin) - 0.01
        vmax = float(vmax) + 0.01

    if figsize is None:
        figsize = (4.3 * len(field_items), 3.8)

    fig, axes = plt.subplots(
        1,
        len(field_items),
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()

    mappable = None
    for ax, (label, field) in zip(axes, field_items):
        mappable = ax.imshow(
            field,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=[
                float(z_axis[0]),
                float(z_axis[-1]),
                float(times[0]),
                float(times[-1]),
            ],
            vmin=float(vmin),
            vmax=float(vmax),
        )
        ax.set_title(label)
        ax.set_xlabel(str(x_label))

    axes[0].set_ylabel(str(y_label))

    if title is not None:
        fig.suptitle(str(title))

    fig.colorbar(
        mappable,
        ax=axes.tolist(),
        label=str(colorbar_label),
        shrink=0.85,
    )

    return fig, axes


def plot_flow_trajectories(
    *,
    a_axis,
    time_axis,
    flows,
    model_labels: dict[str, str] | None = None,
    truth_name: str = "truth",
    n_trajectories: int = 7,
    title: str | None = None,
    x_label: str = "time",
    y_label: str = "F(a,t)",
    linestyles: dict[str, str] | None = None,
):
    """Plot selected flow trajectories for truth and fitted models.

    Parameters
    ----------
    a_axis
        Material-origin grid.
    time_axis
        Integration/evaluation time grid.
    flows
        Ordered mapping ``name -> array`` with shape
        ``[len(time_axis), len(a_axis)]``.
    truth_name
        Mapping key to draw first with a thicker solid line.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and axis.
    """
    import matplotlib.pyplot as plt

    a_axis = np.asarray(a_axis, dtype=float).reshape(-1)
    time_axis = np.asarray(time_axis, dtype=float).reshape(-1)

    normalized = {}
    expected = (len(time_axis), len(a_axis))
    for name, field in flows.items():
        arr = np.asarray(field, dtype=float)
        if arr.shape != expected:
            raise ValueError(
                f"Flow {name!r} has shape {arr.shape}; expected {expected}."
            )
        normalized[str(name)] = arr

    if truth_name not in normalized:
        raise KeyError(f"truth_name {truth_name!r} is not present in flows.")

    n_trajectories = max(
        1,
        min(
            int(n_trajectories),
            len(a_axis),
        ),
    )
    trajectory_indices = np.unique(
        np.linspace(
            0,
            len(a_axis) - 1,
            n_trajectories,
        ).astype(int)
    )

    default_styles = ["--", ":", "-.", (0, (3, 1, 1, 1))]
    style_map = dict(linestyles or {})
    nontruth_names = [
        name
        for name in normalized
        if name != truth_name
    ]
    for i, name in enumerate(nontruth_names):
        style_map.setdefault(
            name,
            default_styles[i % len(default_styles)],
        )

    fig, ax = plt.subplots(figsize=(8, 5.5))

    truth_flow = normalized[truth_name]
    for j, idx in enumerate(trajectory_indices):
        ax.plot(
            time_axis,
            truth_flow[:, idx],
            linewidth=2.5,
            label=(
                _display_label(truth_name, model_labels)
                if j == 0
                else None
            ),
        )

    for name in nontruth_names:
        flow = normalized[name]
        for j, idx in enumerate(trajectory_indices):
            ax.plot(
                time_axis,
                flow[:, idx],
                linestyle=style_map[name],
                linewidth=1.5,
                label=(
                    _display_label(name, model_labels)
                    if j == 0
                    else None
                ),
            )

    ax.set_xlabel(str(x_label))
    ax.set_ylabel(str(y_label))
    if title is not None:
        ax.set_title(str(title))
    ax.legend()
    fig.tight_layout()
    return fig, ax


def save_figure(
    fig,
    path: str | Path,
    *,
    dpi: int = 180,
) -> Path:
    """Save one comparison figure, creating parent directories as needed."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=int(dpi),
        bbox_inches="tight",
    )
    return output
