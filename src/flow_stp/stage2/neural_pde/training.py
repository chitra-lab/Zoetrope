"""Training helpers for the Eulerian neural-PDE Stage-2 model."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import torch
from torch import nn

from .model import NeuralPDEStage2Model, Tensor


@dataclass(frozen=True)
class NeuralPDETrainingConfig:
    """Optimizer settings matching the confirmatory neural-PDE experiment."""

    baseline_lr: float = 2e-3
    dynamic_lr: float = 1e-3
    decoder_lr: float = 1.5e-3
    grad_clip: float = 5.0

    def to_dict(self) -> dict:
        return asdict(self)


def make_neural_pde_optimizer(
    model: NeuralPDEStage2Model,
    config: NeuralPDETrainingConfig | None = None,
) -> torch.optim.Optimizer:
    config = config or NeuralPDETrainingConfig()
    return torch.optim.Adam(
        [
            {"params": model.v_net.parameters(), "lr": config.dynamic_lr},
            {"params": model.q_net.parameters(), "lr": config.dynamic_lr},
            {"params": model.decoder_net.parameters(), "lr": config.decoder_lr},
        ]
    )


def make_neural_pde_baseline_optimizer(
    model: NeuralPDEStage2Model,
    config: NeuralPDETrainingConfig | None = None,
) -> torch.optim.Optimizer:
    config = config or NeuralPDETrainingConfig()
    return torch.optim.Adam(model.baseline_net.parameters(), lr=config.baseline_lr)


def _move_batch(batch: Mapping[str, Tensor], device: torch.device | str) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_neural_pde_baseline_one_epoch(
    model: NeuralPDEStage2Model,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    grad_clip: float = 5.0,
) -> dict[str, float]:
    """Train B(z) on t=0 cells only. Batch keys: z, counts."""
    model.train()
    model.unfreeze_baseline()
    weighted_nll = 0.0
    total_cells = 0

    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        z = batch["z"]
        counts = batch["counts"]

        optimizer.zero_grad(set_to_none=True)
        log_mu = model.baseline(z)
        nll = model.poisson_nll(counts, log_mu)
        if not bool(torch.isfinite(nll)):
            raise FloatingPointError("Non-finite neural-PDE baseline NLL.")
        nll.backward()
        nn.utils.clip_grad_norm_(model.baseline_net.parameters(), grad_clip)
        optimizer.step()

        n = int(counts.shape[0])
        weighted_nll += float(nll.detach()) * n
        total_cells += n

    return {"nll": weighted_nll / max(total_cells, 1), "cells": float(total_cells)}


def train_neural_pde_one_epoch(
    model: NeuralPDEStage2Model,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    grad_clip: float = 5.0,
) -> dict[str, float]:
    """Train V, q and D with B(z) frozen."""
    model.train()
    model.freeze_baseline()
    trainable = model.dynamic_parameters()

    totals: dict[str, float] = {}
    total_cells = 0

    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = model.loss(batch)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite neural-PDE Stage-2 NLL.")
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()

        n = int(batch["counts"].shape[0])
        totals["nll"] = totals.get("nll", 0.0) + float(loss.detach()) * n
        for key, value in diagnostics.items():
            if key == "data":
                continue
            totals[key] = totals.get(key, 0.0) + float(value) * n
        total_cells += n

    return {key: value / max(total_cells, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_neural_pde_poisson_nll(
    model: NeuralPDEStage2Model,
    loader: Iterable,
    device: torch.device | str,
) -> float:
    """Mean full Poisson NLL per count-matrix element."""
    model.eval()
    total_loss = 0.0
    total_elements = 0

    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        output = model(batch["z"], batch["time"])
        loss_sum = model.poisson_nll(
            batch["counts"],
            output["log_mu"],
            reduction="sum",
        )
        total_loss += float(loss_sum)
        total_elements += int(batch["counts"].numel())

    return total_loss / max(total_elements, 1)


def fit_neural_pde(
    model: NeuralPDEStage2Model,
    loader: Iterable,
    *,
    epochs: int,
    device: torch.device | str,
    config: NeuralPDETrainingConfig | None = None,
    validation_loader: Iterable | None = None,
    print_every: int = 25,
    restore_best: bool = True,
    early_stop_patience: int | None = None,
    min_epochs_before_stop: int = 0,
) -> dict:
    """Fit V, q and D after the t=0 baseline has been fitted and frozen.

    If ``validation_loader`` is supplied, checkpoint selection uses validation
    Poisson NLL; otherwise it uses training NLL. The returned best state is
    restored by default.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if early_stop_patience is not None and early_stop_patience <= 0:
        raise ValueError("early_stop_patience must be positive when supplied.")

    config = config or NeuralPDETrainingConfig()
    optimizer = make_neural_pde_optimizer(model, config)

    best_metric = float("inf")
    best_epoch = -1
    best_state = None
    history: list[dict[str, float]] = []
    epochs_since_best = 0

    for epoch in range(1, epochs + 1):
        metrics = train_neural_pde_one_epoch(
            model,
            loader,
            optimizer,
            device,
            grad_clip=config.grad_clip,
        )

        if validation_loader is not None:
            selection_nll = evaluate_neural_pde_poisson_nll(
                model,
                validation_loader,
                device,
            )
            selection_name = "validation_nll"
        else:
            selection_nll = float(metrics["nll"])
            selection_name = "training_nll"

        row = {"epoch": float(epoch), **metrics, selection_name: float(selection_nll)}
        history.append(row)

        if selection_nll < best_metric - 1e-7:
            best_metric = float(selection_nll)
            best_epoch = epoch
            epochs_since_best = 0
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
        else:
            epochs_since_best += 1

        if epoch == 1 or epoch == epochs or (print_every > 0 and epoch % print_every == 0):
            print(
                f"neural-pde epoch={epoch:4d} "
                f"train_nll={metrics['nll']:.6f} "
                f"select_nll={selection_nll:.6f} "
                f"rho=[{metrics['rho_min']:.4f},{metrics['rho_max']:.4f}] "
                f"a=[{metrics['a_min']:.4f},{metrics['a_max']:.4f}]"
            )

        if (
            early_stop_patience is not None
            and epoch >= int(min_epochs_before_stop)
            and epochs_since_best >= int(early_stop_patience)
        ):
            break

    if best_state is None:
        raise RuntimeError("No finite neural-PDE checkpoint was created.")

    if restore_best:
        model.load_state_dict(best_state)
        model.to(device)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_selection_nll": best_metric,
        "selection_metric": (
            "validation_nll" if validation_loader is not None else "training_nll"
        ),
    }
