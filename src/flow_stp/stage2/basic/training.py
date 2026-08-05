\
"""Two-phase training for the bare-bones Stage 2 model."""

from __future__ import annotations

import copy
import random
import time as wall_time
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from ..data import Stage2Dataset
from .model import BasicStage2Model, Tensor


@dataclass(frozen=True)
class BasicStage2TrainingConfig:
    """Training configuration matching the validated positive-control cell."""

    baseline_epochs: int = 200
    stage2_epochs: int = 400
    batch_size: int = 8192
    baseline_learning_rate: float = 2e-3
    rho_learning_rate: float = 1e-3
    decoder_learning_rate: float = 2e-3
    gradient_clip_norm: float = 5.0
    model_seed: int = 23
    loader_seed: int = 24
    num_workers: int = 0
    print_every: int = 50


@dataclass(frozen=True)
class BasicStage2TrainingResult:
    """Histories and checkpoint information returned by ``train_basic_model``."""

    baseline_history: tuple[Dict[str, float], ...]
    stage2_history: tuple[Dict[str, float], ...]
    best_epoch: int
    best_training_nll: float
    baseline_elapsed_seconds: float
    stage2_elapsed_seconds: float


def _move_batch(
    batch: Mapping[str, Tensor],
    device: torch.device,
) -> Dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _reset_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _validate_training_config(config: BasicStage2TrainingConfig) -> None:
    if config.baseline_epochs <= 0 or config.stage2_epochs <= 0:
        raise ValueError("baseline_epochs and stage2_epochs must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if min(
        config.baseline_learning_rate,
        config.rho_learning_rate,
        config.decoder_learning_rate,
    ) <= 0.0:
        raise ValueError("All learning rates must be positive.")
    if config.gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be positive.")
    if config.num_workers < 0:
        raise ValueError("num_workers must be nonnegative.")


def train_basic_baseline_one_epoch(
    model: BasicStage2Model,
    dataloader: Iterable[Mapping[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    gradient_clip_norm: Optional[float] = 5.0,
) -> Dict[str, float]:
    """Train ``B(z)`` for one epoch using t=0 cells only."""
    model.train()
    weighted_loss = 0.0
    total_elements = 0

    for raw_batch in dataloader:
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        log_mu = model.baseline(batch["z"])
        loss = model.poisson_nll(batch["counts"], log_mu)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite baseline Poisson loss.")
        loss.backward()
        if gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                model.baseline_net.parameters(),
                gradient_clip_norm,
            )
        optimizer.step()

        elements = int(batch["counts"].numel())
        weighted_loss += float(loss.detach()) * elements
        total_elements += elements

    return {"nll": weighted_loss / max(total_elements, 1)}


def train_basic_one_epoch(
    model: BasicStage2Model,
    dataloader: Iterable[Mapping[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    gradient_clip_norm: Optional[float] = 5.0,
) -> Dict[str, float]:
    """Jointly train centered rho and the general decoder for one epoch."""
    model.train()
    totals: Dict[str, float] = {}
    total_cells = 0

    for raw_batch in dataloader:
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = model.loss(batch)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite Stage 2 Poisson loss.")
        loss.backward()
        if gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                model.developmental_parameters(),
                gradient_clip_norm,
            )
        optimizer.step()

        n_cells = int(batch["counts"].shape[0])
        totals["nll"] = totals.get("nll", 0.0) + float(loss.detach()) * n_cells
        for key, value in diagnostics.items():
            if key == "data":
                continue
            totals[key] = totals.get(key, 0.0) + float(value) * n_cells
        total_cells += n_cells

    return {key: value / max(total_cells, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_basic_poisson_nll(
    model: BasicStage2Model,
    dataloader: Iterable[Mapping[str, Tensor]],
    device: torch.device,
) -> float:
    """Evaluate mean full Poisson NLL per count-matrix element."""
    model.eval()
    total_loss = 0.0
    total_elements = 0

    for raw_batch in dataloader:
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


def train_basic_model(
    model: BasicStage2Model,
    dataset: Stage2Dataset,
    *,
    config: Optional[BasicStage2TrainingConfig] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> BasicStage2TrainingResult:
    """Fit the t=0 baseline, freeze it, then fit centered rho and D.

    The model is modified in place and restored to the lowest Stage 2 training
    NLL checkpoint. Synthetic truth is never used.
    """
    config = config or BasicStage2TrainingConfig()
    _validate_training_config(config)
    if dataset.counts.shape[1] != model.config.num_genes:
        raise ValueError(
            "Dataset gene count does not match BasicStage2Config.num_genes."
        )

    if device is None:
        device = next(model.parameters()).device
    model.to(device)
    _reset_seeds(config.model_seed)

    t0_indices = torch.nonzero(
        torch.isclose(dataset.time, torch.zeros_like(dataset.time), atol=1e-8),
        as_tuple=False,
    ).reshape(-1)
    if t0_indices.numel() == 0:
        raise ValueError("Stage 2 baseline training requires at least one t=0 cell.")

    baseline_generator = torch.Generator(device="cpu")
    baseline_generator.manual_seed(config.loader_seed)
    stage2_generator = torch.Generator(device="cpu")
    stage2_generator.manual_seed(config.loader_seed + 1)

    baseline_loader = DataLoader(
        Subset(dataset, t0_indices.tolist()),
        batch_size=min(config.batch_size, int(t0_indices.numel())),
        shuffle=True,
        generator=baseline_generator,
        num_workers=config.num_workers,
    )
    stage2_loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        generator=stage2_generator,
        num_workers=config.num_workers,
    )

    model.unfreeze_baseline()
    baseline_optimizer = torch.optim.Adam(
        model.baseline_net.parameters(),
        lr=config.baseline_learning_rate,
    )
    baseline_history: list[Dict[str, float]] = []
    baseline_started = wall_time.time()
    for epoch in range(1, config.baseline_epochs + 1):
        metrics = train_basic_baseline_one_epoch(
            model,
            baseline_loader,
            baseline_optimizer,
            device,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        row = {"epoch": float(epoch), **metrics}
        baseline_history.append(row)
        if verbose and (
            epoch == 1
            or epoch == config.baseline_epochs
            or epoch % config.print_every == 0
        ):
            print(f"baseline epoch={epoch:4d} nll={metrics['nll']:.6f}")
    baseline_elapsed = wall_time.time() - baseline_started
    model.freeze_baseline()

    stage2_optimizer = torch.optim.Adam(
        [
            {"params": model.rho_net.parameters(), "lr": config.rho_learning_rate},
            {
                "params": model.decoder_net.parameters(),
                "lr": config.decoder_learning_rate,
            },
        ]
    )

    stage2_history: list[Dict[str, float]] = []
    best_epoch = 0
    best_nll = float("inf")
    best_state: dict[str, Tensor] | None = None
    stage2_started = wall_time.time()

    for epoch in range(1, config.stage2_epochs + 1):
        metrics = train_basic_one_epoch(
            model,
            stage2_loader,
            stage2_optimizer,
            device,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        row = {"epoch": float(epoch), **metrics}
        stage2_history.append(row)
        epoch_nll = float(metrics["nll"])

        if epoch_nll < best_nll:
            best_nll = epoch_nll
            best_epoch = epoch
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )

        if verbose and (
            epoch == 1
            or epoch == config.stage2_epochs
            or epoch % config.print_every == 0
        ):
            print(
                f"stage2 epoch={epoch:4d} nll={epoch_nll:.6f} "
                f"rho=[{metrics['rho_min']:.4f},{metrics['rho_max']:.4f}] "
                f"D_rms={metrics['developmental_effect_rms']:.4f}"
            )

    if best_state is None:
        raise RuntimeError("No finite Stage 2 checkpoint was created.")
    model.load_state_dict(best_state)
    model.to(device)
    stage2_elapsed = wall_time.time() - stage2_started

    return BasicStage2TrainingResult(
        baseline_history=tuple(baseline_history),
        stage2_history=tuple(stage2_history),
        best_epoch=best_epoch,
        best_training_nll=best_nll,
        baseline_elapsed_seconds=baseline_elapsed,
        stage2_elapsed_seconds=stage2_elapsed,
    )
