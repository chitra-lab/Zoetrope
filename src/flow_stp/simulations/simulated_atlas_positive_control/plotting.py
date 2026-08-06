"""Optional diagnostic plots for the simulated atlas."""

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
    """Write velocity, geometry, rho, and Poisson diagnostics as PNG files."""
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(8.5, 5.3))

    for index, tissue_time in enumerate(result.pde.time):
        stopping_point = (
            result.pde
            .velocity_stopping_points_over_time[index]
        )

        label = (
            f"t={tissue_time:.2f}: constant"
            if np.isnan(stopping_point)
            else (
                f"t={tissue_time:.2f}: "
                f"stop={stopping_point:.3f}"
            )
        )

        ax.plot(
            result.pde.z,
            result.pde.velocity_over_time[index],
            linewidth=2.0,
            label=label,
        )

    ax.set_xlabel("Intrinsic cortical coordinate z")
    ax.set_ylabel("Migration velocity")
    ax.set_title("Time-indexed stopping velocity programs")
    ax.grid(alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    path = output / "velocity_profiles.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    heatmap = ax.pcolormesh(
        result.pde.z,
        result.pde.time,
        result.decoder.rho_decoder,
        shading="auto",
    )
    ax.set_xlabel("Intrinsic cortical coordinate z")
    ax.set_ylabel("Developmental time")
    ax.set_title("Centered developmental coordinate rho(z,t)")
    fig.colorbar(heatmap, ax=ax, label="True centered rho")
    fig.tight_layout()
    path = output / "rho_heatmap.png"
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
            c=plot_obs["z_intrinsic"],
            s=7,
            alpha=0.65,
        )
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_aspect("equal")
        ax.set_xlabel("Observed extrinsic x")
        ax.set_ylabel("Observed extrinsic y")
        ax.set_title(
            f"{tissue_id}: t={result.pde.time[tissue_index]:.2f}, "
            f"bend={result.config.bend_angles_deg[tissue_index]:.1f}°"
        )
        ax.grid(alpha=0.20)
        fig.colorbar(scatter, ax=ax, label="True intrinsic z")
        fig.tight_layout()
        path = output / f"geometry_{tissue_id}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    if gene is None:
        gene = "MIG_01" if "MIG_01" in result.decoder.gene_names else result.decoder.gene_names[0]
    if gene not in result.decoder.gene_names:
        raise KeyError(f"Unknown gene {gene!r}.")

    final_tissue_id = f"tissue_{result.config.n_timepoints - 1:02d}"
    final_obs = result.obs[result.obs["tissue_id"] == final_tissue_id]
    counts = result.counts.loc[final_obs.index, gene].to_numpy()
    means = result.expected_mean.loc[final_obs.index, gene].to_numpy()
    z = final_obs["z_intrinsic"].to_numpy()
    edges = np.linspace(0.0, 1.0, 31)
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
    ax.set_xlabel("Intrinsic cortical coordinate z")
    ax.set_ylabel(f"{gene} expression")
    ax.set_title(f"{gene}: final-tissue Poisson diagnostic")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output / f"poisson_diagnostic_{gene}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    return paths

# FLOW_STP_TIME_INDEXED_VELOCITY_V2
