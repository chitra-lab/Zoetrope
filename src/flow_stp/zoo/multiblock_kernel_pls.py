"""Multiblock RBF-feature PLS baseline for a common process coordinate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .common import (
    AtlasData,
    ZooResult,
    normalized_rate,
    set_seed,
    sqrt_fisher_coordinate,
)

Tensor = torch.Tensor


@dataclass
class Config:
    n_centers: int = 48
    fit_cells_per_tissue: int = 20000
    bandwidth_scale: float = 1.0
    pseudocount: float = 0.5
    lower_quantile: float = 0.005
    upper_quantile: float = 0.995
    seed: int = 20260908


def _choose_centers(
    x: Tensor,
    n_centers: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    m = min(max(int(n_centers), 4), len(x))
    if len(x) <= m:
        centers = x.clone()
    else:
        idx = torch.randperm(len(x), generator=generator)[:m]
        centers = x[idx].clone()
    if len(centers) > 1:
        d = torch.pdist(centers)
        positive = d[d > 1e-6]
        bw = positive.median() if positive.numel() else torch.tensor(1.0)
    else:
        bw = torch.tensor(1.0)
    return centers, torch.clamp(bw, min=1e-2)


def _rbf_features(x: Tensor, centers: Tensor, bandwidth: Tensor) -> Tensor:
    sqdist = torch.cdist(x, centers).square()
    return torch.exp(-0.5 * sqdist / bandwidth.square())


def fit(
    atlas: AtlasData,
    config: Config = Config(),
    *,
    device: Optional[torch.device] = None,
) -> ZooResult:
    # This estimator is algebraic and deliberately stays on CPU.
    del device
    set_seed(config.seed)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(config.seed))

    centers_all: list[Tensor] = []
    bandwidths: list[Tensor] = []
    feature_means: list[Tensor] = []
    cross_covariances: list[Tensor] = []

    for tissue in atlas.tissues:
        x = tissue.coords.detach().cpu()
        y = sqrt_fisher_coordinate(
            normalized_rate(
                tissue, pseudocount=config.pseudocount
            ).detach().cpu()
        )
        centers, bw = _choose_centers(x, config.n_centers, gen)
        bw = bw * float(config.bandwidth_scale)

        if (
            config.fit_cells_per_tissue > 0
            and len(x) > int(config.fit_cells_per_tissue)
        ):
            idx = torch.randperm(len(x), generator=gen)[
                : int(config.fit_cells_per_tissue)
            ]
        else:
            idx = torch.arange(len(x))

        phi = _rbf_features(x[idx], centers, bw)
        phi_mean = phi.mean(dim=0)
        phi_c = phi - phi_mean
        y_c = y[idx] - y[idx].mean(dim=0)
        C = phi_c.T @ y_c / max(len(idx) - 1, 1)

        centers_all.append(centers)
        bandwidths.append(bw)
        feature_means.append(phi_mean)
        cross_covariances.append(C)

    response_gram = torch.zeros(
        atlas.n_genes, atlas.n_genes, dtype=torch.float32
    )
    for C in cross_covariances:
        response_gram += C.T @ C
    evals, evecs = torch.linalg.eigh(response_gram)
    response_loading = evecs[:, -1]

    z_by_tissue: list[np.ndarray] = []
    spatial_weights: list[Tensor] = []
    for tissue, centers, bw, phi_mean, C in zip(
        atlas.tissues,
        centers_all,
        bandwidths,
        feature_means,
        cross_covariances,
    ):
        w = C @ response_loading
        w = w / torch.clamp(torch.linalg.vector_norm(w), min=1e-8)
        spatial_weights.append(w)

        phi = _rbf_features(tissue.coords.detach().cpu(), centers, bw)
        score = (phi - phi_mean) @ w
        q = torch.quantile(
            score,
            torch.tensor(
                [float(config.lower_quantile), float(config.upper_quantile)]
            ),
        )
        span = torch.clamp(q[1] - q[0], min=1e-6)
        z = (score - q[0]) / span
        z_by_tissue.append(z.detach().cpu().numpy())

    result = ZooResult(
        method="multiblock_kernel_pls",
        z_by_tissue=z_by_tissue,
        auxiliary={
            "response_loading": response_loading,
            "spatial_weights": spatial_weights,
            "centers": centers_all,
            "bandwidths": bandwidths,
        },
    )
    result.check()
    return result
