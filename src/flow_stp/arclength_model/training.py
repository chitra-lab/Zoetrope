"""Minimal joint optimizer for alpha_i and M.

This is deliberately tensor-first and dataset-agnostic.  Qian/AnnData loading
belongs outside the arclength model so the mathematical implementation remains
independent of the staged pipelines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from .coordinate_expression import ArclengthModel


@dataclass
class FitHistory:
    epoch: list[int]
    loss: list[float]
    nll: list[float]
    submersion: list[float]
    scale_variation: list[float]
    coverage: list[float]


def fit_arclength_model(
    model: ArclengthModel,
    *,
    coords: Tensor,
    slice_idx: Tensor,
    time: Tensor,
    counts: Tensor,
    log_library_size: Optional[Tensor] = None,
    epochs: int = 1000,
    batch_size: int = 2048,
    lr: float = 1e-3,
    gauge_every: int = 25,
    gauge_reference_size: int = 50000,
    submersion_samples_per_slice: int = 128,
    scale_samples_per_slice: int = 64,
    seed: int = 0,
    verbose_every: int = 50,
) -> FitHistory:
    """Jointly optimize coordinate maps and temporal-basis expression field."""
    if len(coords) != len(slice_idx) or len(coords) != len(time) or len(coords) != len(counts):
        raise ValueError("all cell-level tensors must have the same first dimension")
    device = next(model.parameters()).device
    coords = coords.to(device)
    slice_idx = slice_idx.to(device)
    time = time.to(device)
    counts = counts.to(device)
    if log_library_size is not None:
        log_library_size = log_library_size.to(device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = FitHistory([], [], [], [], [], [])
    n = len(coords)

    def refresh_gauge():
        if gauge_reference_size and n > gauge_reference_size:
            ref = torch.randperm(n, generator=generator, device=device)[:gauge_reference_size]
        else:
            ref = torch.arange(n, device=device)
        model.coordinate_maps.refresh_gauge_(coords[ref], slice_idx[ref])

    refresh_gauge()
    model.train()
    for epoch in range(1, epochs + 1):
        if gauge_every and epoch > 1 and (epoch - 1) % gauge_every == 0:
            refresh_gauge()

        order = torch.randperm(n, generator=generator, device=device)
        sums = {k: 0.0 for k in ["loss", "nll", "submersion", "scale_variation", "coverage"]}
        seen = 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            opt.zero_grad(set_to_none=True)
            terms = model.loss(
                coords=coords[idx],
                slice_idx=slice_idx[idx],
                time=time[idx],
                counts=counts[idx],
                log_library_size=None if log_library_size is None else log_library_size[idx],
                submersion_samples_per_slice=submersion_samples_per_slice,
                scale_samples_per_slice=scale_samples_per_slice,
            )
            terms["loss"].backward()
            opt.step()
            model.coordinate_maps.project_endpoints_()
            b = len(idx)
            seen += b
            for key in sums:
                sums[key] += float(terms[key].detach().cpu()) * b

        history.epoch.append(epoch)
        history.loss.append(sums["loss"] / seen)
        history.nll.append(sums["nll"] / seen)
        history.submersion.append(sums["submersion"] / seen)
        history.scale_variation.append(sums["scale_variation"] / seen)
        history.coverage.append(sums["coverage"] / seen)

        if verbose_every and (epoch == 1 or epoch % verbose_every == 0 or epoch == epochs):
            print(
                f"epoch={epoch:4d} loss={history.loss[-1]:.6f} "
                f"nll={history.nll[-1]:.6f} sub={history.submersion[-1]:.6f} "
                f"scale={history.scale_variation[-1]:.6f} cov={history.coverage[-1]:.6f}"
            )

    refresh_gauge()
    return history
