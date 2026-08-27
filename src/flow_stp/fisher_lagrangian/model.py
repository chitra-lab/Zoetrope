"""Primary learned model for the Fisher-Lagrangian formulation.

Only two objects are learned directly:

    z = alpha_i(x, y)
    log lambda(z, t)

The flow and developmental progress are derived later from the Fisher geometry
of ``lambda``. This module deliberately contains no learned velocity, q, rho,
or Stage-2 decoder network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

Tensor = torch.Tensor


def _as_col(x: Tensor) -> Tensor:
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2 and x.shape[1] == 1:
        return x
    raise ValueError(f"Expected [N] or [N,1], got {tuple(x.shape)}")


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int],
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation {activation!r}")
        act = activations[activation]
        dims = [in_dim, *hidden, out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TissueCoordinateMap(nn.Module):
    """Smooth tissue-specific map ``(x,y) -> z in (0,1)``.

    The sigmoid fixes the numerical range. Orientation/end-point anchoring and
    the submersion condition are imposed through the training objective rather
    than by adding another latent coordinate transformation.
    """

    def __init__(
        self,
        spatial_dim: int = 2,
        hidden: Sequence[int] = (64, 64),
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.raw = MLP(spatial_dim, 1, hidden, activation)

    def forward(self, xy: Tensor) -> Tensor:
        if xy.ndim != 2:
            raise ValueError(f"Expected [N,D] coordinates, got {tuple(xy.shape)}")
        return torch.sigmoid(self.raw(xy))


class LogMeanExpressionField(nn.Module):
    """Shared atlas-wide smooth log-mean field ``eta(z,t)=log lambda(z,t)``."""

    def __init__(
        self,
        num_genes: int,
        hidden: Sequence[int] = (128, 128, 128),
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if num_genes <= 0:
            raise ValueError("num_genes must be positive")
        self.num_genes = int(num_genes)
        self.net = MLP(2, self.num_genes, hidden, activation)

    def forward(self, z: Tensor, time: Tensor) -> Tensor:
        zc = _as_col(z)
        tc = _as_col(time)
        if zc.shape[0] != tc.shape[0]:
            raise ValueError("z and time must have matching rows")
        return self.net(torch.cat((zc, tc), dim=1))

    def mean(self, z: Tensor, time: Tensor) -> Tensor:
        return torch.exp(self.forward(z, time))


@dataclass(frozen=True)
class FisherLagrangianConfig:
    num_tissues: int
    num_genes: int
    spatial_dim: int = 2
    coordinate_hidden: tuple[int, ...] = (64, 64)
    expression_hidden: tuple[int, ...] = (128, 128, 128)
    activation: str = "gelu"
    submersion_floor: float = 1e-3


class FisherLagrangianModel(nn.Module):
    """Joint anatomical-coordinate and mean-expression model.

    Parameters are tissue-specific coordinate maps and one shared expression
    field. Dynamics are derived from the fitted field in ``geometry.py``.
    """

    def __init__(self, config: FisherLagrangianConfig) -> None:
        super().__init__()
        if config.num_tissues <= 0:
            raise ValueError("num_tissues must be positive")
        self.config = config
        self.coordinate_maps = nn.ModuleList(
            [
                TissueCoordinateMap(
                    spatial_dim=config.spatial_dim,
                    hidden=config.coordinate_hidden,
                    activation=config.activation,
                )
                for _ in range(config.num_tissues)
            ]
        )
        self.expression_field = LogMeanExpressionField(
            num_genes=config.num_genes,
            hidden=config.expression_hidden,
            activation=config.activation,
        )

    def coordinate(self, xy: Tensor, tissue_index: Tensor) -> Tensor:
        """Evaluate the appropriate tissue-specific anatomical map per row."""
        if xy.ndim != 2:
            raise ValueError("xy must be [N,D]")
        if tissue_index.ndim != 1 or tissue_index.shape[0] != xy.shape[0]:
            raise ValueError("tissue_index must be [N]")
        out = torch.empty((xy.shape[0], 1), dtype=xy.dtype, device=xy.device)
        for idx, mapping in enumerate(self.coordinate_maps):
            mask = tissue_index == idx
            if mask.any():
                out[mask] = mapping(xy[mask])
        valid = (tissue_index >= 0) & (tissue_index < len(self.coordinate_maps))
        if not bool(valid.all()):
            raise ValueError("tissue_index contains an out-of-range value")
        return out

    def log_mean(self, z: Tensor, time: Tensor) -> Tensor:
        return self.expression_field(z, time)

    def mean(self, z: Tensor, time: Tensor) -> Tensor:
        return self.expression_field.mean(z, time)

    def forward(
        self,
        xy: Tensor,
        tissue_index: Tensor,
        time: Tensor,
        log_exposure: Tensor | None = None,
    ) -> dict[str, Tensor]:
        z = self.coordinate(xy, tissue_index)
        eta = self.log_mean(z, time)
        if log_exposure is None:
            log_mu = eta
        else:
            le = _as_col(log_exposure)
            if le.shape[0] != eta.shape[0]:
                raise ValueError("log_exposure must have one row per cell")
            log_mu = eta + le
        return {
            "z": z,
            "log_lambda": eta,
            "lambda": torch.exp(eta),
            "log_mu": log_mu,
            "mu": torch.exp(log_mu),
        }

    @staticmethod
    def poisson_nll(counts: Tensor, log_mu: Tensor, reduction: str = "mean") -> Tensor:
        return F.poisson_nll_loss(
            log_mu,
            counts.float(),
            log_input=True,
            full=True,
            reduction=reduction,
        )

    def submersion_penalty(self, xy: Tensor, tissue_index: Tensor) -> Tensor:
        """Penalize coordinate-gradient norms below a small positive floor."""
        xy_req = xy.detach().clone().requires_grad_(True)
        z = self.coordinate(xy_req, tissue_index)
        grad = torch.autograd.grad(z.sum(), xy_req, create_graph=True)[0]
        norm = torch.linalg.vector_norm(grad, dim=1)
        return F.relu(self.config.submersion_floor - norm).square().mean()

    def loss(
        self,
        *,
        xy: Tensor,
        tissue_index: Tensor,
        time: Tensor,
        counts: Tensor,
        log_exposure: Tensor | None = None,
        submersion_weight: float = 0.0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        out = self.forward(xy, tissue_index, time, log_exposure)
        data = self.poisson_nll(counts, out["log_mu"])
        sub = (
            self.submersion_penalty(xy, tissue_index)
            if submersion_weight > 0
            else data.new_zeros(())
        )
        total = data + float(submersion_weight) * sub
        return total, {
            "data_nll": data.detach(),
            "submersion": sub.detach(),
            "z_min": out["z"].min().detach(),
            "z_max": out["z"].max().detach(),
        }
