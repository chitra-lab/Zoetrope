"""Anisotropic diffusion-map coordinate guided by local Fisher principal lines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .common import (
    AtlasData,
    ZooResult,
    set_seed,
    sqrt_fisher_coordinate,
)
from .fisher import _knn_indices, score_first_fisher_geometry

Tensor = torch.Tensor


@dataclass
class Config:
    score_k_neighbors: int = 16
    graph_k_neighbors: int = 12
    ridge: float = 1e-3
    pseudocount: float = 0.5
    transverse_penalty: float = 8.0
    density_alpha: float = 1.0
    orientation_cells_per_tissue: int = 4096
    lower_quantile: float = 0.005
    upper_quantile: float = 0.995
    seed: int = 20260908


def _global_expression_loading(
    geometry,
    *,
    cells_per_tissue: int,
    seed: int,
) -> Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    samples = []
    for rate in geometry.mean_rates:
        y = sqrt_fisher_coordinate(rate.detach().cpu())
        if cells_per_tissue > 0 and len(y) > int(cells_per_tissue):
            idx = torch.randperm(len(y), generator=gen)[: int(cells_per_tissue)]
            y = y[idx]
        samples.append(y)
    Y = torch.cat(samples, dim=0)
    Y = Y - Y.mean(dim=0)
    _, _, V = torch.pca_lowrank(Y, q=1, center=False)
    return V[:, 0]


def _anisotropic_affinity(
    xy: np.ndarray,
    lines: np.ndarray,
    *,
    k_neighbors: int,
    transverse_penalty: float,
):
    idx, dist = _knn_indices(xy, k_neighbors)
    n, k = idx.shape
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = idx.reshape(-1)

    dx = xy[idx] - xy[:, None, :]
    u = lines[:, None, :]
    parallel2 = np.sum(dx * u, axis=2) ** 2
    total2 = np.sum(dx * dx, axis=2)
    perp2 = np.maximum(total2 - parallel2, 0.0)

    dpos = np.where(dist > 0, dist, np.nan)
    bw = np.nanmedian(dpos, axis=1)
    bw = np.maximum(np.where(np.isfinite(bw), bw, 1.0), 1e-3)
    cost = (
        parallel2 + float(transverse_penalty) * perp2
    ) / (2.0 * bw[:, None] ** 2)
    vals = np.exp(-cost).astype(np.float64).reshape(-1)

    mask = rows != cols
    return rows[mask], cols[mask], vals[mask], n


def _diffusion_coordinate(
    xy: np.ndarray,
    lines: np.ndarray,
    *,
    k_neighbors: int,
    transverse_penalty: float,
    density_alpha: float,
) -> np.ndarray:
    rows, cols, vals, n = _anisotropic_affinity(
        xy,
        lines,
        k_neighbors=k_neighbors,
        transverse_penalty=transverse_penalty,
    )

    try:
        from scipy import sparse
        from scipy.sparse.linalg import eigsh

        W = sparse.coo_matrix(
            (vals, (rows, cols)), shape=(n, n)
        ).tocsr()
        W = 0.5 * (W + W.T)
        q = np.asarray(W.sum(axis=1)).reshape(-1)
        q = np.maximum(q, 1e-12)
        if float(density_alpha) != 0.0:
            scale = np.power(q, -float(density_alpha))
            W = sparse.diags(scale) @ W @ sparse.diags(scale)
        d = np.asarray(W.sum(axis=1)).reshape(-1)
        d = np.maximum(d, 1e-12)
        inv_sqrt_d = 1.0 / np.sqrt(d)
        S = sparse.diags(inv_sqrt_d) @ W @ sparse.diags(inv_sqrt_d)
        k_eig = min(3, max(n - 1, 1))
        if n <= 3:
            dense = S.toarray()
            evals, evecs = np.linalg.eigh(dense)
            vec = evecs[:, np.argsort(evals)[-2]]
        else:
            evals, evecs = eigsh(S, k=k_eig, which="LA")
            order = np.argsort(evals)
            vec = evecs[:, order[-2]]
        return vec * inv_sqrt_d
    except Exception:
        if n > 5000:
            raise RuntimeError(
                "scipy sparse eigensolver is required for Fisher diffusion "
                "with more than 5000 cells"
            )
        W = np.zeros((n, n), dtype=np.float64)
        W[rows, cols] = vals
        W = 0.5 * (W + W.T)
        q = np.maximum(W.sum(axis=1), 1e-12)
        if float(density_alpha) != 0.0:
            scale = np.power(q, -float(density_alpha))
            W = scale[:, None] * W * scale[None, :]
        d = np.maximum(W.sum(axis=1), 1e-12)
        S = W / np.sqrt(d[:, None] * d[None, :])
        evals, evecs = np.linalg.eigh(S)
        vec = evecs[:, np.argsort(evals)[-2]]
        return vec / np.sqrt(d)


def fit(
    atlas: AtlasData,
    config: Config = Config(),
    *,
    device: Optional[torch.device] = None,
) -> ZooResult:
    del device
    set_seed(config.seed)
    geometry = score_first_fisher_geometry(
        atlas,
        k_neighbors=config.score_k_neighbors,
        ridge=config.ridge,
        pseudocount=config.pseudocount,
    )
    loading = _global_expression_loading(
        geometry,
        cells_per_tissue=config.orientation_cells_per_tissue,
        seed=config.seed,
    )

    z_by_tissue: list[np.ndarray] = []
    raw_by_tissue: list[np.ndarray] = []
    for tid, tissue in enumerate(atlas.tissues):
        xy = tissue.coords.detach().cpu().numpy().astype(np.float32)
        lines = (
            geometry.principal_lines[tid]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        raw = _diffusion_coordinate(
            xy,
            lines,
            k_neighbors=config.graph_k_neighbors,
            transverse_penalty=config.transverse_penalty,
            density_alpha=config.density_alpha,
        )

        yscore = (
            sqrt_fisher_coordinate(geometry.mean_rates[tid])
            @ loading
        ).detach().cpu().numpy()
        if np.std(raw) > 1e-12 and np.std(yscore) > 1e-12:
            corr = np.corrcoef(raw, yscore)[0, 1]
            if np.isfinite(corr) and corr < 0:
                raw = -raw

        qlo, qhi = np.quantile(
            raw,
            [float(config.lower_quantile), float(config.upper_quantile)],
        )
        span = max(float(qhi - qlo), 1e-8)
        z = (raw - qlo) / span
        raw_by_tissue.append(raw.astype(np.float32))
        z_by_tissue.append(z.astype(np.float32))

    result = ZooResult(
        method="fisher_diffusion",
        z_by_tissue=z_by_tissue,
        raw_coordinate_by_tissue=raw_by_tissue,
        auxiliary={
            "geometry": geometry,
            "orientation_loading": loading,
        },
    )
    result.check()
    return result
