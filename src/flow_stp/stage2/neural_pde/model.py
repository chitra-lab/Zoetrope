"""Eulerian neural-PDE Stage-2 model on a frozen Stage-1 anatomical coordinate.

Model
-----
    rho_t + V(z,t) rho_z = q(z,t),      q(z,t) >= 0

rho is evaluated by reverse-characteristic RK4 integration from the observed
(z,t) back to t=0. The same integration returns the material origin a, but
expression is decoded at the *current* anatomical coordinate z:

    log mu = B(z) + D(z,rho) - D(z,0)

This is the learned architecture used in the confirmatory neural-PDE experiment:
- B: 1 -> 64 -> 64 -> genes, GELU
- D: 2 -> 64 -> 64 -> genes, GELU
- V: 2 -> 96 -> 96 -> 96 -> 1, tanh
- q: 2 -> 96 -> 96 -> 96 -> 1, SiLU then ReLU output
- V final layer initialized to zero
- q final layer initialized near zero with positive bias
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

Tensor = torch.Tensor

NEURAL_PDE_ARCHITECTURE_VERSION = "eulerian-neural-pde-v1-2026-08-17"


def _as_col(x: Tensor) -> Tensor:
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2 and x.shape[1] == 1:
        return x
    raise ValueError(f"Expected [N] or [N,1], got {tuple(x.shape)}")


class _MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation!r}")
        act = activations[activation]
        dims = [int(in_dim), *(int(x) for x in hidden), int(out_dim)]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _final_linear(module: nn.Module) -> nn.Linear:
    layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if not layers:
        raise RuntimeError("Network contains no nn.Linear layer.")
    return layers[-1]


@dataclass(frozen=True)
class NeuralPDEStage2Config:
    """Architecture for :class:`NeuralPDEStage2Model`."""

    num_genes: int
    baseline_hidden: Tuple[int, ...] = (64, 64)
    decoder_hidden: Tuple[int, ...] = (64, 64)
    dynamic_hidden: Tuple[int, ...] = (96, 96, 96)
    baseline_activation: str = "gelu"
    decoder_activation: str = "gelu"
    velocity_activation: str = "tanh"
    source_activation: str = "silu"
    train_rk4_steps: int = 16
    decoder_output_initial_std: float = 1e-3
    source_output_initial_std: float = 1e-3
    source_initial_bias: float = 0.40


class NeuralPDEStage2Model(nn.Module):
    """Frozen-z neural PDE with an Eulerian anchored expression decoder."""

    def __init__(
        self,
        num_genes: int | None = None,
        *,
        config: NeuralPDEStage2Config | None = None,
        baseline_hidden: Sequence[int] = (64, 64),
        decoder_hidden: Sequence[int] = (64, 64),
        dynamic_hidden: Sequence[int] = (96, 96, 96),
        baseline_activation: str = "gelu",
        decoder_activation: str = "gelu",
        velocity_activation: str = "tanh",
        source_activation: str = "silu",
        train_rk4_steps: int = 16,
        decoder_output_initial_std: float = 1e-3,
        source_output_initial_std: float = 1e-3,
        source_initial_bias: float = 0.40,
    ) -> None:
        super().__init__()

        if config is not None:
            if num_genes is not None:
                raise ValueError("Pass either config or num_genes/kwargs, not both.")
            cfg = config
        else:
            if num_genes is None:
                raise ValueError("num_genes is required when config is not supplied.")
            cfg = NeuralPDEStage2Config(
                num_genes=int(num_genes),
                baseline_hidden=tuple(int(x) for x in baseline_hidden),
                decoder_hidden=tuple(int(x) for x in decoder_hidden),
                dynamic_hidden=tuple(int(x) for x in dynamic_hidden),
                baseline_activation=str(baseline_activation),
                decoder_activation=str(decoder_activation),
                velocity_activation=str(velocity_activation),
                source_activation=str(source_activation),
                train_rk4_steps=int(train_rk4_steps),
                decoder_output_initial_std=float(decoder_output_initial_std),
                source_output_initial_std=float(source_output_initial_std),
                source_initial_bias=float(source_initial_bias),
            )

        if cfg.num_genes <= 0:
            raise ValueError("num_genes must be positive.")
        if cfg.train_rk4_steps <= 0:
            raise ValueError("train_rk4_steps must be positive.")
        if cfg.decoder_output_initial_std < 0 or cfg.source_output_initial_std < 0:
            raise ValueError("Initialization standard deviations must be nonnegative.")

        self.config = cfg
        self.num_genes = int(cfg.num_genes)

        self.baseline_net = _MLP(
            1,
            self.num_genes,
            cfg.baseline_hidden,
            cfg.baseline_activation,
        )
        self.decoder_net = _MLP(
            2,
            self.num_genes,
            cfg.decoder_hidden,
            cfg.decoder_activation,
        )
        self.v_net = _MLP(
            2,
            1,
            cfg.dynamic_hidden,
            cfg.velocity_activation,
        )
        self.q_net = _MLP(
            2,
            1,
            cfg.dynamic_hidden,
            cfg.source_activation,
        )

        # Match the confirmatory model initialization.
        v_final = _final_linear(self.v_net)
        nn.init.zeros_(v_final.weight)
        if v_final.bias is not None:
            nn.init.zeros_(v_final.bias)

        q_final = _final_linear(self.q_net)
        nn.init.normal_(q_final.weight, mean=0.0, std=cfg.source_output_initial_std)
        if q_final.bias is not None:
            nn.init.constant_(q_final.bias, cfg.source_initial_bias)

        decoder_final = _final_linear(self.decoder_net)
        nn.init.normal_(
            decoder_final.weight,
            mean=0.0,
            std=cfg.decoder_output_initial_std,
        )
        if decoder_final.bias is not None:
            nn.init.zeros_(decoder_final.bias)

    def config_dict(self) -> dict:
        return asdict(self.config)

    @classmethod
    def from_config(cls, config: dict) -> "NeuralPDEStage2Model":
        return cls(config=NeuralPDEStage2Config(**config))

    @staticmethod
    def _matching_rows(first: Tensor, second: Tensor, names: str) -> None:
        if first.shape[0] != second.shape[0]:
            raise ValueError(
                f"Row mismatch for {names}: {first.shape[0]} versus {second.shape[0]}."
            )

    def baseline(self, z: Tensor) -> Tensor:
        """Static t=0 anatomical log-expression baseline B(z)."""
        return self.baseline_net(_as_col(z).clamp(0.0, 1.0))

    def raw_decoder(self, z: Tensor, rho: Tensor) -> Tensor:
        """Unanchored developmental decoder D_raw(z,rho)."""
        zc = _as_col(z).clamp(0.0, 1.0)
        rc = _as_col(rho)
        self._matching_rows(zc, rc, "z and rho")
        return self.decoder_net(torch.cat((zc, rc), dim=1))

    def developmental_residual(self, z: Tensor, rho: Tensor) -> Tensor:
        """Anchored developmental effect D(z,rho)-D(z,0)."""
        rc = _as_col(rho)
        return self.raw_decoder(z, rc) - self.raw_decoder(z, torch.zeros_like(rc))

    # Alias matching the confirmatory experiment terminology.
    developmental = developmental_residual

    def decode_log_mean(self, z: Tensor, rho: Tensor) -> Tensor:
        """Eulerian expression decoder at current z."""
        return self.baseline(z) + self.developmental_residual(z, rho)

    # Short alias for parity with the confirmatory cell.
    decode = decode_log_mean

    def velocity(
        self,
        z: Tensor,
        time: Tensor,
        *,
        create_graph: bool = False,
    ) -> Tensor:
        """Evaluate the unconstrained learned velocity field V(z,t)."""
        del create_graph  # kept for API compatibility with the other dynamic model
        zc = _as_col(z)
        tc = _as_col(time)
        self._matching_rows(zc, tc, "z and time")
        return self.v_net(torch.cat((zc, tc), dim=1))

    def q(self, z: Tensor, time: Tensor) -> Tensor:
        """Evaluate the nonnegative material source q(z,t)=ReLU(h_q(z,t))."""
        zc = _as_col(z)
        tc = _as_col(time)
        self._matching_rows(zc, tc, "z and time")
        raw = self.q_net(torch.cat((zc, tc), dim=1))
        return F.relu(raw)

    # More explicit alias.
    source = q

    def inverse_and_rho(
        self,
        z: Tensor,
        time: Tensor,
        *,
        n_steps: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Reverse-characteristic RK4 integration returning (a, rho)."""
        steps = int(self.config.train_rk4_steps if n_steps is None else n_steps)
        if steps <= 0:
            raise ValueError("n_steps must be positive.")

        x = _as_col(z)
        tc = _as_col(time)
        self._matching_rows(x, tc, "z and time")
        rho = torch.zeros_like(x)
        h = tc / float(steps)

        for k in range(steps):
            u0 = float(k) * h
            s1 = tc - u0
            k1x = -self.velocity(x, s1)
            k1r = self.q(x, s1)

            s2 = tc - (u0 + 0.5 * h)
            x2 = x + 0.5 * h * k1x
            k2x = -self.velocity(x2, s2)
            k2r = self.q(x2, s2)

            x3 = x + 0.5 * h * k2x
            k3x = -self.velocity(x3, s2)
            k3r = self.q(x3, s2)

            s4 = tc - (u0 + h)
            x4 = x + h * k3x
            k4x = -self.velocity(x4, s4)
            k4r = self.q(x4, s4)

            x = x + (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
            rho = rho + (h / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)

        return x, rho

    def inverse_coordinate(self, z: Tensor, time: Tensor) -> Tensor:
        return self.inverse_and_rho(z, time)[0]

    def rho(self, z: Tensor, time: Tensor) -> Tensor:
        return self.inverse_and_rho(z, time)[1]

    def forward(self, z: Tensor, time: Tensor) -> dict[str, Tensor]:
        a, rho = self.inverse_and_rho(z, time)
        baseline = self.baseline(z)
        developmental = self.developmental_residual(z, rho)
        log_mu = baseline + developmental
        return {
            "a": a,
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
        return F.poisson_nll_loss(
            input=log_mu,
            target=counts.float(),
            log_input=True,
            full=True,
            reduction=reduction,
        )

    # Alias matching flow-aware naming.
    data_nll = poisson_nll

    def loss(
        self,
        batch: Mapping[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        output = self.forward(batch["z"], batch["time"])
        data_loss = self.poisson_nll(batch["counts"], output["log_mu"])
        diagnostics = {
            "data": data_loss.detach(),
            "rho_mean": output["rho"].mean().detach(),
            "rho_min": output["rho"].min().detach(),
            "rho_max": output["rho"].max().detach(),
            "a_min": output["a"].min().detach(),
            "a_max": output["a"].max().detach(),
            "developmental_effect_rms": (
                output["developmental_log_effect"].square().mean().sqrt().detach()
            ),
        }
        return data_loss, diagnostics

    def freeze_baseline(self) -> None:
        for parameter in self.baseline_net.parameters():
            parameter.requires_grad_(False)

    def unfreeze_baseline(self) -> None:
        for parameter in self.baseline_net.parameters():
            parameter.requires_grad_(True)

    def baseline_is_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.baseline_net.parameters())

    def dynamic_parameters(self) -> list[nn.Parameter]:
        return (
            list(self.v_net.parameters())
            + list(self.q_net.parameters())
            + list(self.decoder_net.parameters())
        )
