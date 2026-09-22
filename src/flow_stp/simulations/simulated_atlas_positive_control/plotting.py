"""Optional diagnostic plots for the current positive-control atlas."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .simulation import SimulationResult


def save_diagnostic_plots(
    result: SimulationResult,
    output_dir: str | Path,
    *,
    gene: str | None = None,
) -> list[Path]:
    """Write geometry, velocity, progression, and Poisson diagnostics."""
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for index, tissue_time in enumerate(result.truth.time):
        ax.plot(
            result.truth.z,
            result.truth.velocity[index],
            linewidth=2.0,
            label=f"t={tissue_time:.2f}",
        )
    ax.axhline(
        result.config.flow_speed,
        linestyle="--",
        linewidth=1.5,
        label="analytic truth",
    )
    ax.set_xlabel("True common coordinate z")
    ax.set_ylabel("Velocity V(z,t)")
    ax.set_title("Fisher-Lagrangian velocity truth")
    ax.legend()
    fig.tight_layout()
    path = output / "velocity_truth.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for index in np.linspace(
        0,
        len(result.truth.initial_position) - 1,
        8,
        dtype=int,
    ):
        ax.plot(
            result.truth.time,
            result.truth.flow[:, index],
            linewidth=2.0,
        )
    ax.set_xlabel("Developmental time")
    ax.set_ylabel("Common coordinate")
    ax.set_title("Ground-truth spatial trajectories F(a,t)")
    fig.tight_layout()
    path = output / "flow_truth.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for index in np.linspace(
        0,
        len(result.truth.initial_position) - 1,
        8,
        dtype=int,
    ):
        ax.plot(
            result.truth.time,
            result.truth.rho[:, index],
            linewidth=2.0,
        )
    ax.set_xlabel("Developmental time")
    ax.set_ylabel("Fisher arclength")
    ax.set_title("Ground-truth trajectory progression")
    fig.tight_layout()
    path = output / "rho_truth.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    all_x = result.obs["x_extrinsic"].to_numpy()
    all_y = result.obs["y_extrinsic"].to_numpy()
    x_limits = (float(all_x.min() - 0.5), float(all_x.max() + 0.5))
    y_limits = (float(all_y.min() - 0.5), float(all_y.max() + 0.5))
    for tissue_index in range(result.config.n_timepoints):
        tissue_id = f"tissue_{tissue_index:02d}"
        tissue_obs = result.obs[result.obs["tissue_id"] == tissue_id]
        plot_obs = tissue_obs.iloc[:: max(1, len(tissue_obs) // 3500)]
        fig, ax = plt.subplots(figsize=(7.5, 5.7))
        scatter = ax.scatter(
            plot_obs["x_extrinsic"],
            plot_obs["y_extrinsic"],
            c=plot_obs["z_true"],
            s=7,
            alpha=0.65,
        )
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_aspect("equal")
        ax.set_xlabel("Observed extrinsic x")
        ax.set_ylabel("Observed extrinsic y")
        ax.set_title(
            f"{tissue_id}: t={result.truth.time[tissue_index]:.2f}, "
            f"bend={result.config.bend_angles_deg[tissue_index]:.1f}°"
        )
        fig.colorbar(scatter, ax=ax, label="True common coordinate z")
        fig.tight_layout()
        path = output / f"geometry_{tissue_id}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    if gene is None:
        gene = result.truth.gene_names[0]
    if gene not in result.truth.gene_names:
        raise KeyError(f"Unknown gene {gene!r}.")

    final_tissue_id = f"tissue_{result.config.n_timepoints - 1:02d}"
    final_obs = result.obs[result.obs["tissue_id"] == final_tissue_id]
    counts = result.counts.loc[final_obs.index, gene].to_numpy()
    means = result.expected_mean.loc[final_obs.index, gene].to_numpy()
    z = final_obs["z_true"].to_numpy()
    edges = np.linspace(result.config.z_min, result.config.z_max, 31)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bins = np.clip(np.digitize(z, edges) - 1, 0, len(centers) - 1)
    count_mean = np.full(len(centers), np.nan)
    expected_mean = np.full(len(centers), np.nan)
    for index in range(len(centers)):
        mask = bins == index
        if np.any(mask):
            count_mean[index] = float(np.mean(counts[mask]))
            expected_mean[index] = float(np.mean(means[mask]))

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(centers, expected_mean, linewidth=2.5, label="Generating mean")
    ax.scatter(centers, count_mean, s=38, label="Binned Poisson counts")
    ax.set_xlabel("True common coordinate z")
    ax.set_ylabel(f"{gene} expression")
    ax.set_title(f"{gene}: final-tissue Poisson diagnostic")
    ax.legend()
    fig.tight_layout()
    path = output / f"poisson_diagnostic_{gene}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    return paths
