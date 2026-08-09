"""Training helpers for the frozen direct flow-aware Stage-2 model."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import nn

from .model import FlowAwareStage2Model


@dataclass(frozen=True)
class FlowAwareTrainingConfig:
    baseline_lr: float = 2e-3
    rate_lr: float = 1e-3
    decoder_lr: float = 2e-3
    flow_lr: float = 1.5e-3
    grad_clip: float = 5.0

    def to_dict(self) -> dict:
        return asdict(self)


def make_flow_aware_optimizer(
    model: FlowAwareStage2Model,
    config: FlowAwareTrainingConfig | None = None,
) -> torch.optim.Optimizer:
    config = config or FlowAwareTrainingConfig()
    return torch.optim.Adam(
        [
            {"params": model.rate_net.parameters(), "lr": config.rate_lr},
            {"params": model.decoder_net.parameters(), "lr": config.decoder_lr},
            {"params": model.inverse_flow.parameters(), "lr": config.flow_lr},
        ]
    )


def make_flow_aware_baseline_optimizer(
    model: FlowAwareStage2Model,
    config: FlowAwareTrainingConfig | None = None,
) -> torch.optim.Optimizer:
    config = config or FlowAwareTrainingConfig()
    return torch.optim.Adam(
        model.baseline_net.parameters(),
        lr=config.baseline_lr,
    )


def train_flow_aware_baseline_one_epoch(
    model: FlowAwareStage2Model,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    grad_clip: float = 5.0,
) -> dict[str, float]:
    """Train B(a) on t=0 cells. Batch keys: z, counts."""
    model.train()
    model.unfreeze_baseline()

    weighted_nll = 0.0
    total_cells = 0

    for batch in loader:
        z = batch["z"].to(device)
        counts = batch["counts"].to(device)

        optimizer.zero_grad(set_to_none=True)
        log_mu = model.baseline(z)
        nll = model.data_nll(counts, log_mu)

        if not bool(torch.isfinite(nll)):
            raise FloatingPointError("Non-finite flow-aware baseline NLL.")

        nll.backward()
        nn.utils.clip_grad_norm_(
            model.baseline_net.parameters(),
            grad_clip,
        )
        optimizer.step()

        n = len(z)
        weighted_nll += float(nll.detach()) * n
        total_cells += n

    return {
        "nll": weighted_nll / max(total_cells, 1),
        "cells": float(total_cells),
    }


def train_flow_aware_one_epoch(
    model: FlowAwareStage2Model,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    grad_clip: float = 5.0,
) -> dict[str, float]:
    """Train rate, decoder and inverse flow. Batch keys: z, time, counts."""
    model.train()
    model.freeze_baseline()

    trainable = (
        list(model.rate_net.parameters())
        + list(model.decoder_net.parameters())
        + list(model.inverse_flow.parameters())
    )

    weighted_nll = 0.0
    weighted_reg = 0.0
    total_cells = 0

    for batch in loader:
        z = batch["z"].to(device)
        time = batch["time"].to(device)
        counts = batch["counts"].to(device)

        optimizer.zero_grad(set_to_none=True)

        output = model(z, time)
        nll = model.data_nll(counts, output["log_mu"])
        flow_reg = model.flow_penalty()
        loss = nll + flow_reg

        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite flow-aware Stage-2 objective.")

        loss.backward()
        nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()

        n = len(z)
        weighted_nll += float(nll.detach()) * n
        weighted_reg += float(flow_reg.detach()) * n
        total_cells += n

    return {
        "nll": weighted_nll / max(total_cells, 1),
        "flow_regularizer": weighted_reg / max(total_cells, 1),
        "objective": (
            weighted_nll + weighted_reg
        ) / max(total_cells, 1),
        "cells": float(total_cells),
    }


def fit_flow_aware(
    model: FlowAwareStage2Model,
    loader: Iterable,
    *,
    epochs: int,
    device: torch.device | str,
    config: FlowAwareTrainingConfig | None = None,
    print_every: int = 50,
    restore_best: bool = True,
) -> dict:
    """Full Stage-2 trainer after baseline fitting/freezing."""
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    config = config or FlowAwareTrainingConfig()
    optimizer = make_flow_aware_optimizer(model, config)

    best_objective = float("inf")
    best_epoch = -1
    best_state = None
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        metrics = train_flow_aware_one_epoch(
            model,
            loader,
            optimizer,
            device,
            grad_clip=config.grad_clip,
        )
        history.append(metrics)

        if metrics["objective"] < best_objective:
            best_objective = metrics["objective"]
            best_epoch = epoch
            best_state = copy.deepcopy(
                {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                }
            )

        if (
            epoch == 1
            or epoch == epochs
            or (print_every > 0 and epoch % print_every == 0)
        ):
            print(
                f"flow-aware epoch={epoch:4d} "
                f"nll={metrics['nll']:.6f} "
                f"flow_reg={metrics['flow_regularizer']:.6g}"
            )

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_objective": best_objective,
    }
