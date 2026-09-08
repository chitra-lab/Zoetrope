"""Estimators of local Fisher spatial-expression geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .common import AtlasData, LocalMeanField, default_device, normalized_rate, sample_indices, set_seed

Tensor = torch.Tensor


@dataclass
class MeanFieldFitConfig:
    epochs: int = 500
    batch_size: int = 2048
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden: tuple[int, ...] = (96, 96)
    seed: int = 20260908


@dataclass
class FisherGeometry:
    tensors: list[Tensor]
    lambda1: list[Tensor]
    lambda2: list[Tensor]
    principal_lines: list[Tensor]
    mean_rates: list[Tensor]
    metadata: dict


def fit_local_mean_fields(atlas: AtlasData, config: MeanFieldFitConfig = MeanFieldFitConfig(), *, device: Optional[torch.device] = None) -> tuple[list[LocalMeanField], list[dict[str, float]]]:
    if device is None:
        device = default_device()
    set_seed(config.seed)
    models: list[LocalMeanField] = []
    summaries: list[dict[str, float]] = []
    for tid, tissue in enumerate(atlas.tissues):
        x = tissue.coords.to(device)
        counts = tissue.counts.to(device)
        log_lib = tissue.log_library_size.to(device)
        model = LocalMeanField(atlas.n_genes, hidden=config.hidden).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay))
        last = float("nan")
        for _ in range(int(config.epochs)):
            idx = sample_indices(len(x), config.batch_size, device=device)
            log_rate = model(x[idx])
            loss = F.poisson_nll_loss(log_rate + log_lib[idx, None], counts[idx], log_input=True, full=False, reduction="mean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last = float(loss.detach().cpu())
        model.eval()
        models.append(model)
        summaries.append({"tissue_id": float(tid), "final_nll": last})
    return models, summaries


def _jacobian_log_rate(model: nn.Module, coords: Tensor) -> tuple[Tensor, Tensor]:
    x = coords.detach().clone()
    if hasattr(torch, "func") and hasattr(torch.func, "jacrev") and hasattr(torch.func, "vmap"):
        def one(sample: Tensor) -> Tensor:
            return model(sample[None, :]).squeeze(0)
        log_rate = model(x)
        jac = torch.func.vmap(torch.func.jacrev(one))(x)
        return log_rate.detach(), jac.detach()
    x = x.requires_grad_(True)
    log_rate = model(x)
    grads = []
    for g in range(log_rate.shape[1]):
        grad_g = torch.autograd.grad(log_rate[:, g].sum(), x, retain_graph=True, create_graph=False, only_inputs=True)[0]
        grads.append(grad_g)
    return log_rate.detach(), torch.stack(grads, dim=1).detach()


def fisher_geometry_from_mean_fields(atlas: AtlasData, models: Sequence[nn.Module], *, batch_size: int = 2048, device: Optional[torch.device] = None) -> FisherGeometry:
    if len(models) != atlas.n_tissues:
        raise ValueError("one local mean-field model is required per tissue")
    if device is None:
        device = default_device()
    tensors, l1s, l2s, lines, rates_all = [], [], [], [], []
    for tissue, model in zip(atlas.tissues, models):
        model = model.to(device).eval()
        x = tissue.coords.to(device)
        gs, rs = [], []
        for start in range(0, len(x), int(batch_size)):
            xb = x[start:start + int(batch_size)]
            log_rate, jac_log = _jacobian_log_rate(model, xb)
            rate = torch.exp(torch.clamp(log_rate, min=-15.0, max=15.0))
            G = torch.einsum("bg,bgd,bge->bde", rate, jac_log, jac_log)
            gs.append(G.cpu())
            rs.append(rate.cpu())
        G = torch.cat(gs, dim=0)
        rate = torch.cat(rs, dim=0)
        evals, evecs = torch.linalg.eigh(G)
        tensors.append(G)
        l2s.append(evals[:, 0])
        l1s.append(evals[:, 1])
        lines.append(evecs[:, :, 1])
        rates_all.append(rate)
    return FisherGeometry(tensors=tensors, lambda1=l1s, lambda2=l2s, principal_lines=lines, mean_rates=rates_all, metadata={"source": "local_poisson_mean_fields"})


def _knn_indices(coords: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(coords)
    k_eff = min(max(int(k), 2), n)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        dist, idx = tree.query(coords, k=k_eff)
        return np.asarray(idx, dtype=np.int64), np.asarray(dist, dtype=np.float32)
    except Exception:
        x = torch.as_tensor(coords, dtype=torch.float32)
        all_idx, all_dist = [], []
        for start in range(0, n, 512):
            q = x[start:start + 512]
            d = torch.cdist(q, x)
            vals, inds = torch.topk(d, k=k_eff, largest=False, dim=1)
            all_idx.append(inds.cpu().numpy())
            all_dist.append(vals.cpu().numpy())
        return np.concatenate(all_idx), np.concatenate(all_dist)


def score_first_fisher_geometry(atlas: AtlasData, *, k_neighbors: int = 32, ridge: float = 1e-3, pseudocount: float = 0.5) -> FisherGeometry:
    tensors, l1s, l2s, lines, rates_all = [], [], [], [], []
    for tissue in atlas.tissues:
        xy = tissue.coords.cpu().numpy().astype(np.float32)
        idx, dist = _knn_indices(xy, k_neighbors)
        raw_rate = normalized_rate(tissue).cpu().numpy()
        log_rate = np.log(np.maximum(raw_rate + float(pseudocount), 1e-8))
        n, g = raw_rate.shape
        G_out = np.zeros((n, 2, 2), dtype=np.float32)
        rate_out = np.zeros((n, g), dtype=np.float32)
        eye = np.eye(2, dtype=np.float32)[None, :, :]
        for start in range(0, n, 512):
            stop = min(start + 512, n)
            nbr = idx[start:stop]
            qxy = xy[start:stop, None, :]
            dx = xy[nbr] - qxy
            d = dist[start:stop]
            dpos = np.where(d > 0, d, np.nan)
            bw = np.nanmedian(dpos, axis=1)
            bw = np.maximum(np.where(np.isfinite(bw), bw, 1.0).astype(np.float32), 1e-3)
            w = np.exp(-0.5 * (d / bw[:, None]) ** 2).astype(np.float32)
            w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-8)
            y = log_rate[nbr]
            ybar = np.einsum("bk,bkg->bg", w, y)[:, None, :]
            yc = y - ybar
            A = np.einsum("bk,bkd,bke->bde", w, dx, dx) + float(ridge) * eye
            Bmat = np.einsum("bk,bkd,bkg->bdg", w, dx, yc)
            beta = np.linalg.solve(A, Bmat).transpose(0, 2, 1)
            m = np.maximum(np.einsum("bk,bkg->bg", w, raw_rate[nbr]), 1e-8).astype(np.float32)
            rate_out[start:stop] = m
            G_out[start:stop] = np.einsum("bg,bgd,bge->bde", m, beta, beta)
        G_t = torch.from_numpy(G_out)
        evals, evecs = torch.linalg.eigh(G_t)
        tensors.append(G_t)
        l2s.append(evals[:, 0])
        l1s.append(evals[:, 1])
        lines.append(evecs[:, :, 1])
        rates_all.append(torch.from_numpy(rate_out))
    return FisherGeometry(tensors=tensors, lambda1=l1s, lambda2=l2s, principal_lines=lines, mean_rates=rates_all, metadata={"source": "local_log_rate_scores", "k_neighbors": int(k_neighbors), "ridge": float(ridge), "pseudocount": float(pseudocount)})


def eigengap_weight(lambda1: Tensor, lambda2: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.clamp((lambda1 - lambda2) / (lambda1 + lambda2 + float(eps)), min=0.0, max=1.0)
