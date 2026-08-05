\
"""Bare-bones Stage 2 model on a frozen Stage 1 anatomical coordinate.

The model contains three neural functions:

    B_phi(z)                         static t=0 anatomical baseline
    r_theta(z,t)                     unconstrained scalar neural field
    D_psi(z,rho)                     general gene-expression decoder

The centered spatial developmental coordinate is

    rho(z,t) = r_theta(z,t) - r_theta(z,0),

which imposes rho(z,0)=0 exactly without multiplying by chronological time and
without constraining rho to be positive. The expression model is

    log mu_g(z,t)
        = B_g(z)
        + D_raw,g(z,rho(z,t))
        - D_raw,g(z,0).

The decoder subtraction imposes an exact zero developmental effect at rho=0,
so the separately fitted t=0 baseline retains its meaning after it is frozen.
There is no library-size offset, PDE, flow, velocity, source, loading vector,
or explicit smoothness or monotonicity penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

Tensor = torch.Tensor


def as_col(value: Tensor) -> Tensor:
    """Return a tensor with shape ``[batch, 1]``."""
    if value.ndim == 1:
        return value[:, None]
    if value.ndim == 2 and value.shape[1] == 1:
        return value
    raise ValueError(
        f"Expected a tensor with shape [N] or [N,1], received {tuple(value.shape)}."
    )


class MLP(nn.Module):
    """Small fully connected network shared by the Stage 2 components."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        activation_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
        }
        if activation not in activation_map:
            raise ValueError(f"Unknown activation {activation!r}.")
        act = activation_map[activation]

        layers: list[nn.Module] = []
        previous = int(in_dim)
        for width in hidden:
            if int(width) <= 0:
                raise ValueError("All hidden-layer widths must be positive.")
            layers.extend((nn.Linear(previous, int(width)), act()))
            previous = int(width)
        layers.append(nn.Linear(previous, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


def final_linear_layer(network: nn.Module) -> nn.Linear:
    """Return the final linear layer in ``network``."""
    layers = [module for module in network.modules() if isinstance(module, nn.Linear)]
    if not layers:
        raise RuntimeError("Network contains no nn.Linear layer.")
    return layers[-1]


@dataclass(frozen=True)
class BasicStage2Config:
    """Architecture for :class:`BasicStage2Model`."""

    num_genes: int
    baseline_hidden: Tuple[int, ...] = (64, 64)
    rho_hidden: Tuple[int, ...] = (64, 64)
    decoder_hidden: Tuple[int, ...] = (64, 64)
    baseline_activation: str = "gelu"
    rho_activation: str = "relu"
    decoder_activation: str = "gelu"
    decoder_output_initial_std: float = 1e-3


class BasicStage2Model(nn.Module):
    """Neural baseline, centered developmental coordinate, and general decoder."""

    def __init__(self, config: BasicStage2Config) -> None:
        super().__init__()
        self.config = config
        if config.num_genes <= 0:
            raise ValueError("num_genes must be positive.")
        if config.decoder_output_initial_std < 0.0:
            raise ValueError("decoder_output_initial_std must be nonnegative.")

        self.baseline_net = MLP(
            in_dim=1,
            out_dim=config.num_genes,
            hidden=config.baseline_hidden,
            activation=config.baseline_activation,
        )
        self.rho_net = MLP(
            in_dim=2,
            out_dim=1,
            hidden=config.rho_hidden,
            activation=config.rho_activation,
        )
        self.decoder_net = MLP(
            in_dim=2,
            out_dim=config.num_genes,
            hidden=config.decoder_hidden,
            activation=config.decoder_activation,
        )

        # Joint training begins close to the separately fitted t=0 baseline.
        decoder_final = final_linear_layer(self.decoder_net)
        nn.init.normal_(
            decoder_final.weight,
            mean=0.0,
            std=config.decoder_output_initial_std,
        )
        if decoder_final.bias is not None:
            nn.init.zeros_(decoder_final.bias)

    @staticmethod
    def _matching_rows(first: Tensor, second: Tensor, names: str) -> None:
        if first.shape[0] != second.shape[0]:
            raise ValueError(
                f"Row mismatch for {names}: {first.shape[0]} versus {second.shape[0]}."
            )

    def baseline(self, z: Tensor) -> Tensor:
        """Return the static t=0 anatomical log-expression baseline ``B(z)``."""
        return self.baseline_net(as_col(z).clamp(0.0, 1.0))

    def raw_rho(self, z: Tensor, time: Tensor) -> Tensor:
        """Evaluate the unconstrained scalar neural field ``r(z,t)``."""
        z_col = as_col(z).clamp(0.0, 1.0)
        time_col = as_col(time).clamp(0.0, 1.0)
        self._matching_rows(z_col, time_col, "z and time")
        return self.rho_net(torch.cat((z_col, time_col), dim=1))

    def rho(self, z: Tensor, time: Tensor) -> Tensor:
        """Return ``rho(z,t)=r(z,t)-r(z,0)`` with exact ``rho(z,0)=0``."""
        z_col = as_col(z).clamp(0.0, 1.0)
        time_col = as_col(time).clamp(0.0, 1.0)
        self._matching_rows(z_col, time_col, "z and time")
        raw_at_time = self.raw_rho(z_col, time_col)
        raw_at_origin = self.raw_rho(z_col, torch.zeros_like(time_col))
        return raw_at_time - raw_at_origin

    def raw_decoder(self, z: Tensor, rho: Tensor) -> Tensor:
        """Evaluate the unanchored general decoder ``D_raw(z,rho)``."""
        z_col = as_col(z).clamp(0.0, 1.0)
        rho_col = as_col(rho)
        self._matching_rows(z_col, rho_col, "z and rho")
        return self.decoder_net(torch.cat((z_col, rho_col), dim=1))

    def developmental_residual(self, z: Tensor, rho: Tensor) -> Tensor:
        """Return the exactly anchored developmental log-expression effect."""
        rho_col = as_col(rho)
        return self.raw_decoder(z, rho_col) - self.raw_decoder(
            z,
            torch.zeros_like(rho_col),
        )

    def decode_log_mean(self, z: Tensor, rho: Tensor) -> Tensor:
        """Decode log expected counts from anatomical position and rho."""
        return self.baseline(z) + self.developmental_residual(z, rho)

    def forward(self, z: Tensor, time: Tensor) -> dict[str, Tensor]:
        rho = self.rho(z, time)
        baseline = self.baseline(z)
        developmental = self.developmental_residual(z, rho)
        log_mu = baseline + developmental
        return {
            "rho": rho,
            "baseline_log_mean": baseline,
            "developmental_log_effect": developmental,
            "log_mu": log_mu,
            "mu": torch.exp(log_mu),
        }

    @staticmethod
    def poisson_nll(
        counts: Tensor,
        log_mu: Tensor,
        *,
        reduction: str = "mean",
    ) -> Tensor:
        """Full Poisson negative log-likelihood."""
        return F.poisson_nll_loss(
            input=log_mu,
            target=counts.float(),
            log_input=True,
            full=True,
            reduction=reduction,
        )

    def loss(
        self,
        batch: Mapping[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return the unregularized Poisson loss and detached diagnostics."""
        output = self.forward(batch["z"], batch["time"])
        data_loss = self.poisson_nll(batch["counts"], output["log_mu"])
        diagnostics = {
            "data": data_loss.detach(),
            "rho_mean": output["rho"].mean().detach(),
            "rho_min": output["rho"].min().detach(),
            "rho_max": output["rho"].max().detach(),
            "developmental_effect_rms": (
                output["developmental_log_effect"].square().mean().sqrt().detach()
            ),
        }
        return data_loss, diagnostics

    def freeze_baseline(self) -> None:
        """Freeze all parameters of ``B(z)``."""
        for parameter in self.baseline_net.parameters():
            parameter.requires_grad_(False)

    def unfreeze_baseline(self) -> None:
        """Unfreeze all parameters of ``B(z)``."""
        for parameter in self.baseline_net.parameters():
            parameter.requires_grad_(True)

    def baseline_is_frozen(self) -> bool:
        return all(not parameter.requires_grad for parameter in self.baseline_net.parameters())

    def developmental_parameters(self) -> list[nn.Parameter]:
        """Return the trainable rho-network and decoder parameters."""
        return list(self.rho_net.parameters()) + list(self.decoder_net.parameters())
