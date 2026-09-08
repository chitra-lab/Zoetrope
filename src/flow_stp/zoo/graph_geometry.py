"""Discrete principal-line integration utilities for graph-based zoo methods."""
from __future__ import annotations
from typing import Sequence
import heapq
import numpy as np
import torch
from .fisher import _knn_indices

Tensor = torch.Tensor

def orient_line_field(coords: Tensor, lines: Tensor, *, k_neighbors: int = 10) -> Tensor:
    xy = coords.detach().cpu().numpy().astype(np.float32)
    u = lines.detach().cpu().numpy().astype(np.float32).copy()
    idx, _ = _knn_indices(xy, k_neighbors)
    n = len(xy); visited = np.zeros(n, dtype=bool); out = np.zeros_like(u)
    root = int(np.argmax(np.linalg.norm(u, axis=1))); out[root] = u[root]; visited[root] = True; queue = [root]
    while queue:
        i = queue.pop(0)
        for j0 in idx[i]:
            j = int(j0)
            if j == i or visited[j]: continue
            cand = u[j]
            if float(np.dot(out[i], cand)) < 0: cand = -cand
            out[j] = cand; visited[j] = True; queue.append(j)
    for root in np.flatnonzero(~visited):
        out[root] = u[root]; visited[root] = True; queue = [int(root)]
        while queue:
            i = queue.pop(0)
            for j0 in idx[i]:
                j = int(j0)
                if j == i or visited[j]: continue
                cand = u[j]
                if float(np.dot(out[i], cand)) < 0: cand = -cand
                out[j] = cand; visited[j] = True; queue.append(j)
    mean = out.mean(axis=0); axis = int(np.argmax(np.abs(mean)))
    if mean[axis] < 0: out *= -1.0
    out = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)
    return torch.from_numpy(out)

def integrate_eigenfield_nodes(coords: Tensor, lines: Tensor, *, k_neighbors: int = 10, epochs: int = 800, learning_rate: float = 5e-2, alignment_power: float = 2.0, seed: int = 20260908) -> Tensor:
    torch.manual_seed(int(seed))
    xy = coords.detach().cpu().float(); u = orient_line_field(xy, lines, k_neighbors=k_neighbors).float(); idx, _ = _knn_indices(xy.numpy(), k_neighbors)
    src, dst, target, weight = [], [], [], []
    for i in range(len(xy)):
        for j0 in idx[i]:
            j = int(j0)
            if j <= i: continue
            dx = xy[j] - xy[i]; dist = torch.linalg.norm(dx)
            if float(dist) <= 1e-8: continue
            uavg = u[i] + u[j]
            if float(torch.linalg.norm(uavg)) <= 1e-8: uavg = u[i]
            uavg = uavg / torch.clamp(torch.linalg.norm(uavg), min=1e-8)
            t = torch.dot(uavg, dx); align = torch.abs(t) / torch.clamp(dist, min=1e-8)
            src.append(i); dst.append(j); target.append(float(t)); weight.append(float(torch.clamp(align, 0.0, 1.0).pow(alignment_power)))
    if not src: raise RuntimeError("principal graph has no usable edges")
    src_t = torch.tensor(src, dtype=torch.long); dst_t = torch.tensor(dst, dtype=torch.long); target_t = torch.tensor(target, dtype=torch.float32); weight_t = torch.tensor(weight, dtype=torch.float32)
    z = torch.nn.Parameter(torch.zeros(len(xy), dtype=torch.float32)); opt = torch.optim.Adam([z], lr=float(learning_rate))
    for _ in range(int(epochs)):
        residual = z[dst_t] - z[src_t] - target_t
        loss = (weight_t * residual.square()).sum() / torch.clamp(weight_t.sum(), min=1.0) + 1e-4 * z.mean().square()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    z_out = z.detach(); lo = torch.quantile(z_out, 0.01); hi = torch.quantile(z_out, 0.99)
    return torch.clamp((z_out - lo) / torch.clamp(hi - lo, min=1e-6), 0.0, 1.0)

def _dijkstra(adjacency: Sequence[list[tuple[int, float]]], source: int) -> np.ndarray:
    n = len(adjacency); dist = np.full(n, np.inf, dtype=np.float64); dist[int(source)] = 0.0; heap = [(0.0, int(source))]
    while heap:
        d, i = heapq.heappop(heap)
        if d != dist[i]: continue
        for j, w in adjacency[i]:
            nd = d + float(w)
            if nd < dist[j]: dist[j] = nd; heapq.heappush(heap, (nd, j))
    return dist

def principal_graph_arclength(coords: Tensor, lines: Tensor, *, k_neighbors: int = 12, alignment_penalty: float = 4.0) -> Tensor:
    xy = coords.detach().cpu().numpy().astype(np.float32); u = lines.detach().cpu().numpy().astype(np.float32); u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-8)
    idx, dist_knn = _knn_indices(xy, k_neighbors); adjacency = [[] for _ in range(len(xy))]
    for i in range(len(xy)):
        for pos, j0 in enumerate(idx[i]):
            j = int(j0)
            if j == i: continue
            d = float(dist_knn[i, pos])
            if d <= 1e-8: continue
            e = (xy[j] - xy[i]) / d; uavg = u[i] + np.sign(np.dot(u[i], u[j]) + 1e-12) * u[j]; un = np.linalg.norm(uavg); uavg = u[i] if un <= 1e-8 else uavg / un
            alignment = abs(float(np.dot(e, uavg))); weight = d * (1.0 + float(alignment_penalty) * (1.0 - alignment) ** 2)
            adjacency[i].append((j, weight)); adjacency[j].append((i, weight))
    d0 = _dijkstra(adjacency, 0); finite0 = np.isfinite(d0)
    if not finite0.any(): raise RuntimeError("principal KNN graph is disconnected from every point")
    a = int(np.nanargmax(np.where(finite0, d0, -np.inf))); da = _dijkstra(adjacency, a); finite = np.isfinite(da)
    if not finite.all():
        fallback = np.linalg.norm(xy - xy[a], axis=1); max_finite = float(np.max(da[finite])) if finite.any() else 1.0; da[~finite] = max_finite + fallback[~finite]
    b = int(np.argmax(da)); denom = max(float(da[b]), 1e-8)
    return torch.from_numpy(np.clip(da / denom, 0.0, 1.0).astype(np.float32))
