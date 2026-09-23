"""One-seed positive-control ablation: packed primary Stage1a vs bare Stage1a.

The bare model keeps only:
  * one tissue-specific neural coordinate map per tissue,
  * the detached robust affine gauge needed to choose a numerical chart,
  * the shared neural expression field,
  * the Poisson likelihood.

For the clean simulator both models disable library-size offsets and slice
nuisance effects.  The bare model additionally fixes coverage to [0, 1] and
sets the submersion, relative-scale, and coverage penalties to zero.

The downstream Fisher-Lagrangian construction is identical for both models.
Finite-flow accuracy is evaluated after endpoint chart calibration, consistent
with gauge conjugacy.
"""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

from flow_stp.stage1.stage1a import Stage1ARunConfig, run_stage1a
from flow_stp.fisher_lagrangian.trajectory import integrate_flow_and_rho, local_dynamics
from flow_stp.simulations.simulated_atlas_positive_control import (
    SimulationConfig,
    run_simulation,
)


class Stage1aExpressionField(torch.nn.Module):
    """Scientific E(z,t): shared beta + decoder, excluding observation nuisances."""

    def __init__(self, stage1a_model):
        super().__init__()
        self.stage1a_model = stage1a_model

    def forward(self, z: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        z = z.reshape(-1, 1)
        time = time.reshape(-1, 1).to(device=z.device, dtype=z.dtype)
        x = torch.cat([z, time], dim=1)
        return self.stage1a_model.beta[None, :] + self.stage1a_model.decoder(x)


class EmpiricalChart:
    """Monotone empirical true-z -> learned-z chart with an inverse on support."""

    def __init__(self, true_z, learned_z, *, n_bins=64):
        x = np.asarray(true_z, dtype=float).reshape(-1)
        y = np.asarray(learned_z, dtype=float).reshape(-1)
        good = np.isfinite(x) & np.isfinite(y)
        x, y = x[good], y[good]
        if len(x) < 20:
            raise RuntimeError("Too few finite cells for chart calibration.")

        rho = spearmanr(x, y).statistic
        if not np.isfinite(rho) or abs(rho) < 1e-3:
            raise RuntimeError("Learned coordinate has essentially no monotone relation to truth.")
        self.spearman = float(rho)
        increasing = bool(rho > 0)

        # Equal-count bins make the calibration robust to cell-level Poisson noise
        # and reproduce the binned-median + PAVA logic used in prior benchmarks.
        order = np.argsort(x)
        x, y = x[order], y[order]
        bins = np.array_split(np.arange(len(x)), min(int(n_bins), len(x)))
        xb = np.asarray([np.median(x[idx]) for idx in bins if len(idx)], dtype=float)
        yb = np.asarray([np.median(y[idx]) for idx in bins if len(idx)], dtype=float)

        iso = IsotonicRegression(increasing=increasing, out_of_bounds="nan")
        yfit = np.asarray(iso.fit_transform(xb, yb), dtype=float)

        self.x_min = float(xb.min())
        self.x_max = float(xb.max())
        self._iso = iso

        # Build a monotone inverse. PAVA can create plateaus, so collapse each
        # fitted learned-z value to the median corresponding true-z location.
        frame = pd.DataFrame({"y": yfit, "x": xb}).groupby("y", as_index=False)["x"].median()
        frame = frame.sort_values("y")
        self._inv_y = frame["y"].to_numpy(dtype=float)
        self._inv_x = frame["x"].to_numpy(dtype=float)
        self.y_min = float(self._inv_y.min())
        self.y_max = float(self._inv_y.max())

    def forward(self, true_z):
        x = np.asarray(true_z, dtype=float)
        return np.asarray(self._iso.predict(x), dtype=float)

    def inverse(self, learned_z):
        y = np.asarray(learned_z, dtype=float)
        out = np.full(y.shape, np.nan, dtype=float)
        good = np.isfinite(y) & (y >= self.y_min) & (y <= self.y_max)
        if np.any(good):
            out[good] = np.interp(y[good], self._inv_y, self._inv_x)
        return out


def split_simulation(sim):
    obs = sim.obs.reset_index(drop=True)
    counts_by_slice = []
    spatial_by_slice = []
    true_z_by_slice = []

    for tissue_index in range(sim.config.n_timepoints):
        mask = obs["tissue_index"].to_numpy() == tissue_index
        counts_by_slice.append(sim.counts.iloc[np.flatnonzero(mask)].to_numpy(np.float32))
        spatial_by_slice.append(
            obs.loc[mask, ["x_extrinsic", "y_extrinsic"]].to_numpy(np.float32)
        )
        true_z_by_slice.append(obs.loc[mask, "z_true"].to_numpy(float))

    panel = pd.DataFrame(
        {
            "slice_id": np.arange(sim.config.n_timepoints, dtype=int),
            "time": sim.config.tissue_times,
        }
    )
    return counts_by_slice, spatial_by_slice, true_z_by_slice, panel


def calibrate_charts(result, true_z_by_slice):
    cells = result.cells
    charts = []
    zhat_by_slice = []
    for s, ztrue in enumerate(true_z_by_slice):
        zhat = (
            cells.loc[cells["slice_id"].to_numpy() == s, "z_stage1a"]
            .to_numpy(float)
        )
        if len(zhat) != len(ztrue):
            raise RuntimeError(f"Slice {s}: inferred/true cell count mismatch.")
        charts.append(EmpiricalChart(ztrue, zhat))
        zhat_by_slice.append(zhat)
    return charts, zhat_by_slice


def evaluate(result, sim, true_z_by_slice):
    model = result.model
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    field = Stage1aExpressionField(model).to(device)
    field.eval()

    charts, zhat_by_slice = calibrate_charts(result, true_z_by_slice)

    # Coordinate recovery after monotone chart calibration.
    spearman_abs = []
    z_sqerr = []
    z_valid = []
    for chart, ztrue, zhat in zip(charts, true_z_by_slice, zhat_by_slice):
        spearman_abs.append(abs(chart.spearman))
        z_back = chart.inverse(zhat)
        good = np.isfinite(z_back)
        z_valid.append(float(good.mean()))
        if np.any(good):
            z_sqerr.extend((z_back[good] - ztrue[good]) ** 2)

    # q is intrinsic, but evaluate it at matched physical locations using the
    # empirical tissue chart at each observed time.
    q_err = []
    q_true = float(sim.truth.developmental_speed)
    z_eval = np.linspace(0.05, 0.95, 181)
    with torch.enable_grad():
        for t, chart in zip(sim.config.tissue_times, charts):
            z_learned = chart.forward(z_eval)
            good = np.isfinite(z_learned)
            if not np.any(good):
                continue
            zt = torch.as_tensor(z_learned[good], dtype=dtype, device=device)[:, None]
            tt = torch.full_like(zt, float(t))
            _, qhat = local_dynamics(field, zt, tt)
            qnp = qhat.detach().cpu().numpy().reshape(-1)
            q_err.extend((qnp - q_true) ** 2)

    # Final-time trajectories.  Endpoint chart conjugacy is sufficient:
    # learned start = h_0(a); true final = h_1^{-1}(F_learned(h_0(a))).
    a_true = np.linspace(0.05, 0.75, 101)
    z0_learned = charts[0].forward(a_true)
    start_good = np.isfinite(z0_learned)

    f_sqerr = []
    rho_sqerr = []
    rho_bias = []
    final_valid_fraction = 0.0

    if np.any(start_good):
        a0 = a_true[start_good]
        z0 = torch.as_tensor(z0_learned[start_good], dtype=dtype, device=device)[:, None]
        with torch.enable_grad():
            traj = integrate_flow_and_rho(
                field,
                z0,
                t0=0.0,
                t1=1.0,
                n_steps=128,
                eps=1e-8,
            )

        z1_learned = traj.position[-1].detach().cpu().numpy().reshape(-1)
        rho1 = traj.rho[-1].detach().cpu().numpy().reshape(-1)
        z1_true_hat = charts[-1].inverse(z1_learned)

        good = np.isfinite(z1_true_hat) & np.isfinite(rho1)
        final_valid_fraction = float(good.sum() / len(a_true))

        if np.any(good):
            f_displacement_hat = z1_true_hat[good] - a0[good]
            f_sqerr = (f_displacement_hat - float(sim.config.flow_speed)) ** 2

            rho_truth_final = q_true
            rho_sqerr = (rho1[good] - rho_truth_final) ** 2
            rho_bias = rho1[good] - rho_truth_final

    return {
        "validation_nll": float(result.summary["validation_poisson_nll"]),
        "median_abs_spearman_z": float(np.median(spearman_abs)),
        "mean_abs_spearman_z": float(np.mean(spearman_abs)),
        "calibrated_z_rmse": float(np.sqrt(np.mean(z_sqerr))),
        "coordinate_valid_fraction": float(np.mean(z_valid)),
        "q_relative_rmse": float(np.sqrt(np.mean(q_err)) / q_true),
        "rho_final_relative_rmse": (
            float(np.sqrt(np.mean(rho_sqerr)) / q_true) if len(rho_sqerr) else np.nan
        ),
        "rho_final_relative_bias": (
            float(np.mean(rho_bias) / q_true) if len(rho_bias) else np.nan
        ),
        "F_displacement_rmse": (
            float(np.sqrt(np.mean(f_sqerr))) if len(f_sqerr) else np.nan
        ),
        "trajectory_valid_fraction": final_valid_fraction,
    }


def make_configs(output_root: Path, *, seed: int, epochs: int):
    common = dict(
        epochs=int(epochs),
        seed=int(seed),
        deterministic=True,
        resume=False,
        # Clean positive control: no exposure/nuisance structure in the DGP.
        slice_effect_dim=0,
        use_library_size_offset=False,
    )

    packed = replace(
        Stage1ARunConfig(),
        output_dir=output_root / "packed_primary",
        **common,
    )

    bare = replace(
        Stage1ARunConfig(),
        output_dir=output_root / "bare_primary",
        submersion_weight=0.0,
        scale_variation_weight=0.0,
        coverage_prior_weight=0.0,
        nuisance_weight=0.0,
        max_missing_each_end=0.0,
        initial_missing=0.0,
        **common,
    )
    return packed, bare


def run_one(name, cfg, *, counts_by_slice, spatial_by_slice, panel, genes, sim, true_z_by_slice):
    print("\n" + "=" * 88)
    print(name)
    print("=" * 88)
    try:
        result = run_stage1a(
            counts_by_slice=counts_by_slice,
            spatial_by_slice=spatial_by_slice,
            panel_df=panel,
            gene_names=genes,
            config=cfg,
        )
        metrics = evaluate(result, sim, true_z_by_slice)
        metrics.update({"model": name, "status": "ok", "error": ""})
        return metrics
    except Exception as exc:
        traceback.print_exc()
        return {
            "model": name,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/bare_primary_ablation")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--cells-per-tissue", type=int, default=5000)
    args = parser.parse_args()

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    sim_cfg = SimulationConfig(
        n_timepoints=8,
        n_cells_per_tissue=int(args.cells_per_tissue),
        random_seed=int(args.seed),
        maximum_bend_deg=120.0,
        n_z=401,
        flow_speed=0.2,
        n_spatial_genes=16,
        n_developmental_genes=8,
        gene_generator_seed=31,
    )
    sim = run_simulation(sim_cfg)
    counts_by_slice, spatial_by_slice, true_z_by_slice, panel = split_simulation(sim)

    print("Simulation:", json.dumps(sim.summary(), indent=2))
    print("Truth developmental Fisher speed:", sim.truth.developmental_speed)

    packed_cfg, bare_cfg = make_configs(
        output_root,
        seed=int(args.seed),
        epochs=int(args.epochs),
    )

    rows = [
        run_one(
            "packed_primary",
            packed_cfg,
            counts_by_slice=counts_by_slice,
            spatial_by_slice=spatial_by_slice,
            panel=panel,
            genes=list(sim.truth.gene_names),
            sim=sim,
            true_z_by_slice=true_z_by_slice,
        ),
        run_one(
            "bare_primary",
            bare_cfg,
            counts_by_slice=counts_by_slice,
            spatial_by_slice=spatial_by_slice,
            panel=panel,
            genes=list(sim.truth.gene_names),
            sim=sim,
            true_z_by_slice=true_z_by_slice,
        ),
    ]

    table = pd.DataFrame(rows)
    table.to_csv(output_root / "comparison.csv", index=False)

    print("\nFINAL COMPARISON")
    print(table.to_string(index=False))

    if set(table["status"]) == {"ok"}:
        packed = table.loc[table["model"] == "packed_primary"].iloc[0]
        bare = table.loc[table["model"] == "bare_primary"].iloc[0]
        print("\nBare / packed ratios (lower is better for error metrics)")
        for key in (
            "calibrated_z_rmse",
            "q_relative_rmse",
            "rho_final_relative_rmse",
            "F_displacement_rmse",
        ):
            denom = float(packed[key])
            ratio = float(bare[key]) / denom if np.isfinite(denom) and denom > 0 else np.nan
            print(f"{key:30s}: {ratio:.4f}")


if __name__ == "__main__":
    main()
