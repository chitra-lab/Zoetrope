"""Logistic material-flow tissue benchmark used in the displacement tournament.

Training receives counts, XY and times only. Hidden material labels are used
solely for evaluation. This module imports the selected repo model at run time;
it does not duplicate or refit its expression decoder.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from scipy.linalg import hadamard
from scipy.stats import spearmanr

from flow_stp.stage2 import integrate_flow_rk4


@dataclass(frozen=True)
class LogisticDisplacementConfig:
    n_times: int = 50
    cells_per_tissue: int = 100_000
    seed: int = 20261006
    evaluation_seed: int = 20261010
    beta: float = 0.8
    n_source: int = 2500
    contour_k: int = 316
    max_dt: float = 0.0025

    def validate(self):
        if self.n_times < 3 or self.cells_per_tissue < 20:
            raise ValueError("Need at least three times and twenty cells per tissue.")
        if self.n_source < 1 or self.contour_k < 1 or self.max_dt <= 0:
            raise ValueError("Source count, contour K and max_dt must be positive.")
        if not np.isfinite(self.beta):
            raise ValueError("beta must be finite.")


@dataclass
class LogisticTissues:
    config: LogisticDisplacementConfig
    times: np.ndarray
    counts: list[np.ndarray]
    xy: list[np.ndarray]
    material_labels: list[np.ndarray]

    def stage1_kwargs(self):
        return dict(
            counts_by_slice=self.counts, spatial_by_slice=self.xy,
            panel_df=pd.DataFrame({"slice_id": np.arange(len(self.times)),
                                   "time": self.times}),
            gene_names=[f"gene_{g}" for g in range(8)],
        )


def true_flow(a, t, beta=0.8):
    a = np.asarray(a, dtype=np.float64)
    e = np.exp(beta * np.asarray(t, dtype=np.float64))
    return a * e / (1.0 - a + a * e)


def generate_logistic_tissues(config=LogisticDisplacementConfig()):
    config.validate()
    rng = np.random.default_rng(config.seed)
    times = np.linspace(0, 1, config.n_times)
    h = hadamard(8).astype(float) / np.sqrt(8)
    counts, xy, labels = [], [], []
    for t in times:
        a = rng.uniform(0, 1, config.cells_per_tissue)
        u = rng.uniform(0, 1, config.cells_per_tissue)
        psi = (10 + 3 * (a[:, None] - 0.5) * h[:, 1]
               + np.cos(np.pi*t/2) * h[:, 2]
               + np.sin(np.pi*t/2) * h[:, 3])
        counts.append(rng.poisson(psi**2 / 4).astype(np.uint16))
        xy.append(np.column_stack([true_flow(a, t, config.beta), 0.15*u]).astype(np.float32))
        labels.append(a.astype(np.float32))
    return LogisticTissues(config, times, counts, xy, labels)


def contour_lift(predicted_z, source_y, target_z, target_xy, k=316):
    """K nearest z contours, then nearest source transverse y (original lift)."""
    if k < 1:
        raise ValueError("k must be positive.")
    predicted_z = np.asarray(predicted_z).reshape(-1)
    target_z = np.asarray(target_z).reshape(-1)
    xy = np.asarray(target_xy)
    if not len(target_z) or not np.isfinite(target_z).all() or not np.isfinite(predicted_z).all():
        raise ValueError("Contour coordinates must be finite and target nonempty.")
    if len(source_y) != len(predicted_z) or xy.shape != (len(target_z), 2):
        raise ValueError("Mismatched source or target shapes.")
    order = np.argsort(target_z)
    zs, points = target_z[order], xy[order]
    out = np.empty((len(predicted_z), 2))
    half_window = max(2*k, 32)
    for i, (z, y) in enumerate(zip(predicted_z, source_y)):
        pos = int(np.searchsorted(zs, z))
        lo, hi = max(0, pos-half_window), min(len(zs), pos+half_window)
        dz = np.abs(zs[lo:hi]-z)
        kk = min(k, len(dz))
        near = np.argpartition(dz, kk-1)[:kk] if kk < len(dz) else np.arange(len(dz))
        out[i] = points[lo+near[np.argmin(np.abs(points[lo+near, 1]-y))]]
    return out


def displacement_metrics(predicted_xy, true_xy, source_xy):
    """Endpoint RMSE equals displacement RMSE for the same source points."""
    pred, truth, source = map(lambda x: np.asarray(x, float), (predicted_xy, true_xy, source_xy))
    if pred.shape != truth.shape or pred.shape != source.shape or pred.ndim != 2 or pred.shape[1] != 2:
        raise ValueError("Expected matching [N,2] arrays.")
    if not all(np.isfinite(x).all() for x in (pred, truth, source)) or not len(pred):
        raise ValueError("Expected nonempty finite endpoints.")
    err2 = np.sum((pred-truth)**2, axis=1)
    motion2 = np.sum((truth-source)**2, axis=1)
    denom = motion2.sum()
    return dict(
        physical_displacement_rmse=float(np.sqrt(err2.mean())),
        true_displacement_rms=float(np.sqrt(motion2.mean())),
        pooled_displacement_nrmse=float(np.sqrt(err2.sum()/denom)) if denom > 0 else float('nan'),
        pooled_motion_skill=float(1-err2.sum()/denom) if denom > 0 else float('nan'),
    )


def _source_rows(data, source_id):
    """Original physical source selection, excluding fixed legacy fit/calibration sets."""
    cfg, n = data.config, data.config.cells_per_tissue
    rng = np.random.default_rng(cfg.evaluation_seed)
    sample = None
    for _ in range(source_id+1):
        sample = np.sort(rng.choice(n, min(10_000, n//10), replace=False))
    cal_rng = np.random.default_rng(cfg.evaluation_seed+101)
    # Reproduce the per-time calibration stream, including preceding tissues.
    sample_rng = np.random.default_rng(cfg.evaluation_seed)
    for i in range(source_id+1):
        sampled = sample_rng.choice(n, min(10_000, n//10), replace=False)
        mask = np.ones(n, bool); mask[sampled] = False
        cal = cal_rng.choice(np.flatnonzero(mask), min(5000, n//20), replace=False)
    mask = np.ones(n, bool); mask[sample] = False; mask[cal] = False
    x = data.xy[source_id][:, 0]
    lo, hi = np.quantile(x[mask], [0.005, 0.995])
    eligible = np.flatnonzero(mask & (x >= lo) & (x <= hi))
    return np.sort(np.random.default_rng(cfg.evaluation_seed+707).choice(
        eligible, min(cfg.n_source, len(eligible)), replace=False))


def evaluate_displacement(data, z_by_tissue, field, *, device='cpu', output_dir=None):
    """Evaluate T25 to the original twelve targets; reduced runs scale the indices.

    No truth-based gauge correction is applied. A single joint reflection of z
    and its fitted field would leave these physical predictions unchanged.
    """
    cfg = data.config
    source_id = int(round(25*(cfg.n_times-1)/49))
    targets = sorted(set(int(round(i*(cfg.n_times-1)/49))
                         for i in [0,5,10,15,20,24,26,30,35,40,45,49]) - {source_id})
    rows = _source_rows(data, source_id)
    source = data.xy[source_id][rows].astype(float)
    source_a = data.material_labels[source_id][rows]
    z0 = torch.as_tensor(np.asarray(z_by_tissue[source_id])[rows], dtype=torch.float32, device=device)
    t0 = float(data.times[source_id])
    pred_all, truth_all, source_all, per_target = [], [], [], []
    lift_all = []
    for target in targets:
        t1 = float(data.times[target]); direction = 1 if t1 > t0 else -1
        # Reparameterizing time permits backward integration through the existing
        # forward RK4 API. Autodiff supplies the correct velocity sign.
        def directed_field(z, elapsed):
            return field(z, t0 + direction*elapsed)
        trajectory = integrate_flow_rk4(directed_field, z0, t0=0, t1=abs(t1-t0),
            n_steps=max(1, int(np.ceil(abs(t1-t0)/cfg.max_dt))))
        pred_z = trajectory.final_position.detach().cpu().numpy().reshape(-1)
        pred = contour_lift(pred_z, source[:,1], z_by_tissue[target], data.xy[target], cfg.contour_k)
        truth = np.column_stack([true_flow(source_a, t1, cfg.beta), source[:,1]])
        floor = contour_lift(truth[:,0], source[:,1], data.xy[target][:,0], data.xy[target], cfg.contour_k)
        metrics = displacement_metrics(pred, truth, source)
        dx_pred, dx_true = pred[:,0]-source[:,0], truth[:,0]-source[:,0]
        metrics.update(target_id=target, time=t1,
            dx_spearman=float(spearmanr(dx_pred, dx_true).statistic),
            oracle_lift_nrmse=displacement_metrics(floor, truth, source)['pooled_displacement_nrmse'])
        per_target.append(metrics)
        pred_all.append(pred); truth_all.append(truth); source_all.append(source); lift_all.append(floor)
    pred, truth, sources = map(np.concatenate, (pred_all, truth_all, source_all))
    summary = displacement_metrics(pred, truth, sources)
    summary.update(source_id=source_id, source_cells=len(rows),
        oracle_lift_nrmse=displacement_metrics(np.concatenate(lift_all), truth, sources)['pooled_displacement_nrmse'],
        no_motion_nrmse=1.0,
        pooled_dx_spearman=float(spearmanr(pred[:,0]-sources[:,0], truth[:,0]-sources[:,0]).statistic))
    by = pd.DataFrame(per_target)
    for label, lo, hi in [('tiny',0,0.01), ('mid',0.01,0.03), ('large',0.03,np.inf)]:
        ids = np.flatnonzero((by.true_displacement_rms >= lo) & (by.true_displacement_rms < hi))
        summary[f'{label}_motion_pooled_nrmse'] = (
            displacement_metrics(np.concatenate([pred_all[i] for i in ids]),
                np.concatenate([truth_all[i] for i in ids]),
                np.concatenate([source_all[i] for i in ids]))['pooled_displacement_nrmse']
            if len(ids) else None)
    if output_dir is not None:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        by.to_csv(out/'displacement_by_target.csv', index=False)
        (out/'displacement_summary.json').write_text(json.dumps(summary, indent=2))
        (out/'benchmark_manifest.json').write_text(json.dumps({
            'dgp': 'logistic_material_flow_fisher_orthogonal_v1',
            'config': asdict(cfg), 'targets': targets,
            'stage2': 'differentiate fitted Stage1 logE; no decoder refit',
            'integration': 'direct autodiff RK4; no velocity-grid interpolation',
            'truth_usage': 'evaluation only; no gauge calibration',
        }, indent=2))
        np.savez_compressed(out/'displacement_endpoints.npz',
            source_rows=rows, predicted=pred, truth=truth, source=sources)
    return summary, by


def coordinates_for_tissues(result, data):
    """Evaluate the fitted chart using the saved per-tissue XY normalization."""
    norm = result.xy_norm_params.sort_values('slice_id')
    device = next(result.model.parameters()).device
    z_values = []
    with torch.no_grad():
        for sid, xy in enumerate(data.xy):
            row = norm.iloc[sid]
            standardized = (xy - np.array([row.center_x,row.center_y])) / np.array([row.scale_x,row.scale_y])
            pieces = []
            for start in range(0, len(xy), 8192):
                x = torch.as_tensor(standardized[start:start+8192], dtype=torch.float32, device=device)
                pieces.append(result.model.coordinate_maps.one_slice(x,sid).cpu().numpy())
            z_values.append(np.concatenate(pieces))
    return z_values
