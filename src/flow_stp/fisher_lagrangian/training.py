"""Minimal training helpers for the primary Fisher-Lagrangian fit.

The optimization fits only coordinate maps and the shared expression field.
Derived dynamics are intentionally excluded from this objective in prototype 1
to avoid allowing the coordinate, expression field, and flow to co-adapt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch

from .model import FisherLagrangianModel, Tensor


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    submersion_weight: float = 1.0
    grad_clip_norm: float | None = 5.0


def train_step(
    model: FisherLagrangianModel,
    batch: Mapping[str, Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    submersion_weight: float,
    grad_clip_norm: float | None = None,
) -> dict[str, float]:
    """One joint coordinate/expression optimization step.

    Required batch keys are ``xy``, ``tissue_index``, ``time``, and ``counts``.
    ``log_exposure`` is optional.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, diag = model.loss(
        xy=batch["xy"],
        tissue_index=batch["tissue_index"],
        time=batch["time"],
        counts=batch["counts"],
        log_exposure=batch.get("log_exposure"),
        submersion_weight=submersion_weight,
    )
    loss.backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        **{k: float(v.cpu()) for k, v in diag.items()},
    }


def fit(
    model: FisherLagrangianModel,
    batches: Iterable[Mapping[str, Tensor]],
    *,
    epochs: int,
    config: TrainConfig = TrainConfig(),
) -> list[dict[str, float]]:
    """Fit the primary learned objects and return epoch-mean diagnostics."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        totals: dict[str, float] = {}
        count = 0
        for batch in batches:
            stats = train_step(
                model,
                batch,
                optimizer,
                submersion_weight=config.submersion_weight,
                grad_clip_norm=config.grad_clip_norm,
            )
            count += 1
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + value
        if count == 0:
            raise ValueError("batches yielded no batches")
        history.append(
            {"epoch": float(epoch), **{k: v / count for k, v in totals.items()}}
        )
    return history
