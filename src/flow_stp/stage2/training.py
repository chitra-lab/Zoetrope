"""Training helpers for Stage 2."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import torch
from torch import nn

from .model import Stage2DynamicsModel, Tensor


def _move_batch(
    batch: Mapping[str, Tensor],
    device: torch.device,
) -> Dict[str, Tensor]:
    return {
        key: value.to(device, non_blocking=(device.type == "cuda"))
        for key, value in batch.items()
    }


def train_one_epoch(
    model: Stage2DynamicsModel,
    dataloader: Iterable[Mapping[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: Optional[float] = 5.0,
) -> Dict[str, float]:
    """Train the Stage 2 model for one epoch."""
    model.train()

    totals: Dict[str, float] = {}
    n_batches = 0

    for raw_batch in dataloader:
        batch = _move_batch(raw_batch, device)

        optimizer.zero_grad(set_to_none=True)
        loss, parts = model.loss(batch)
        loss.backward()

        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        n_batches += 1

    return {
        key: value / max(n_batches, 1)
        for key, value in totals.items()
    }


@torch.no_grad()
def evaluate_poisson_nll(
    model: Stage2DynamicsModel,
    dataloader: Iterable[Mapping[str, Tensor]],
    device: torch.device,
) -> float:
    """Evaluate held-out mean Poisson NLL."""
    model.eval()

    weighted_total = 0.0
    total_elements = 0

    for raw_batch in dataloader:
        batch = _move_batch(raw_batch, device)
        output = model(
            z=batch["z"],
            time=batch["time"],
            log_library_size=batch.get("log_library_size"),
        )

        nll_sum = torch.nn.functional.poisson_nll_loss(
            input=output["log_mu"],
            target=batch["counts"].float(),
            log_input=True,
            full=True,
            reduction="sum",
        )
        weighted_total += float(nll_sum)
        total_elements += int(batch["counts"].numel())

    return weighted_total / max(total_elements, 1)
