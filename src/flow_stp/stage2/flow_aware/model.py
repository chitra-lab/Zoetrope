"""Frozen direct flow-aware Stage-2 model.

This module lives beside, rather than replacing, ``flow_stp.stage2.basic``.

Model
-----
    a = A(z,t) = F_t^{-1}(z)
    R(a,t) = int_0^t ReLU(h(a,tau)) d tau
    rho(z,t) = R(A(z,t),t)
    V(z,t) = -A_t / A_z
    q(z,t) = R_t(A(z,t),t)

Expression decoder:
    log mu = B(a) + D(a,R) - D(a,0)

The inverse flow is a time-conditioned monotone piecewise-linear map.
A(z,0)=z exactly and A_z>0 prevents trajectory crossing. Velocity may have
either sign.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


FROZEN_ARCHITECTURE_VERSION = "direct-flow-confirmatory-v2-2026-08-12"


def _as_col(x: torch.Tensor) -> torch.Tensor:
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
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")

        dims = [in_dim, *hidden, out_dim]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(activations[activation]())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _final_linear(module: nn.Module) -> nn.Linear:
    layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if not layers:
        raise RuntimeError("Expected at least one Linear layer.")
    return layers[-1]


@dataclass(frozen=True)
class FlowAwareStage2Config:
    num_genes: int
    flow_segments: int = 24
    flow_hidden: tuple[int, ...] = (64, 64)
    baseline_hidden: tuple[int, ...] = (64, 64)
    rate_hidden: tuple[int, ...] = (96, 96, 96)
    decoder_hidden: tuple[int, ...] = (64, 64)
    r_quadrature: int = 12
    native_rate_init_bias: float = 0.40
    flow_spatial_smooth: float = 0.0
    flow_temporal_smooth: float = 0.0
    flow_smooth_time_points: int = 9
    rate_floor: float = 0.0

    def __post_init__(self) -> None:
        if self.num_genes <= 0:
            raise ValueError("num_genes must be positive.")
        if self.flow_segments < 2:
            raise ValueError("flow_segments must be >= 2.")
        if self.r_quadrature < 2:
            raise ValueError("r_quadrature must be >= 2.")
        if self.flow_smooth_time_points < 3:
            raise ValueError("flow_smooth_time_points must be >= 3.")
        if self.native_rate_init_bias < 0:
            raise ValueError("native_rate_init_bias must be nonnegative.")
        if self.rate_floor < 0:
            raise ValueError("rate_floor must be nonnegative.")


class MonotoneInverseFlow(nn.Module):
    """Time-conditioned monotone piecewise-linear inverse flow A(z,t)."""

    def __init__(
        self,
        n_segments: int = 24,
        hidden: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()
        self.n_segments = int(n_segments)
        self.time_net = _MLP(
            1,
            self.n_segments,
            tuple(hidden),
            "tanh",
        )
        last = _final_linear(self.time_net)
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def widths(self, time: torch.Tensor) -> torch.Tensor:
        t = _as_col(time).clamp(0.0, 1.0)
        logits = t * self.time_net(t)
        return torch.softmax(logits, dim=1)

    def forward(
        self,
        z: torch.Tensor,
        time: torch.Tensor,
        *,
        return_dz: bool = False,
    ):
        zc = _as_col(z).clamp(0.0, 1.0)
        tc = _as_col(time).clamp(0.0, 1.0)
        widths = self.widths(tc)
        k = self.n_segments

        scaled = (zc[:, 0] * k).clamp(0.0, k - 1e-7)
        idx = torch.floor(scaled).long().clamp(0, k - 1)
        frac = scaled - idx.to(scaled.dtype)

        cumulative = torch.cumsum(widths, dim=1)
        cumulative0 = torch.cat(
            (torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]),
            dim=1,
        )

        row = torch.arange(zc.shape[0], device=zc.device)
        width = widths[row, idx]
        a0 = cumulative0[row, idx]
        a = (a0 + frac * width)[:, None]

        if return_dz:
            a_z = (k * width)[:, None]
            return a, a_z
        return a

    @torch.no_grad()
    def forward_map(
        self,
        a: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Invert A(z,t) to obtain F(a,t)=z for inference."""
        ac = _as_col(a).clamp(0.0, 1.0)
        tc = _as_col(time).clamp(0.0, 1.0)
        widths = self.widths(tc)
        k = self.n_segments

        cumulative = torch.cumsum(widths, dim=1)
        idx = (ac > cumulative[:, :-1]).sum(dim=1).long().clamp(0, k - 1)
        cumulative0 = torch.cat(
            (torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]),
            dim=1,
        )

        row = torch.arange(ac.shape[0], device=ac.device)
        width = widths[row, idx].clamp_min(1e-8)
        a0 = cumulative0[row, idx]
        frac = ((ac[:, 0] - a0) / width).clamp(0.0, 1.0)

        return ((idx.to(ac.dtype) + frac) / k)[:, None]

    def smoothness_penalty(
        self,
        time_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        widths = self.widths(time_grid)
        log_widths = torch.log(widths.clamp_min(1e-8))

        spatial_second = (
            log_widths[:, 2:]
            - 2.0 * log_widths[:, 1:-1]
            + log_widths[:, :-2]
        )
        temporal_second = (
            log_widths[2:]
            - 2.0 * log_widths[1:-1]
            + log_widths[:-2]
        )

        return (
            spatial_second.square().mean(),
            temporal_second.square().mean(),
        )


class FlowAwareStage2Model(nn.Module):
    """Frozen direct flow-aware Stage-2 model."""

    architecture_version = FROZEN_ARCHITECTURE_VERSION

    def __init__(
        self,
        num_genes: int,
        *,
        flow_segments: int = 24,
        flow_hidden: Sequence[int] = (64, 64),
        baseline_hidden: Sequence[int] = (64, 64),
        rate_hidden: Sequence[int] = (96, 96, 96),
        decoder_hidden: Sequence[int] = (64, 64),
        r_quadrature: int = 12,
        native_rate_init_bias: float = 0.40,
        flow_spatial_smooth: float = 0.0,
        flow_temporal_smooth: float = 0.0,
        flow_smooth_time_points: int = 9,
        rate_floor: float = 0.0,
    ) -> None:
        super().__init__()

        self.config = FlowAwareStage2Config(
            num_genes=int(num_genes),
            flow_segments=int(flow_segments),
            flow_hidden=tuple(int(x) for x in flow_hidden),
            baseline_hidden=tuple(int(x) for x in baseline_hidden),
            rate_hidden=tuple(int(x) for x in rate_hidden),
            decoder_hidden=tuple(int(x) for x in decoder_hidden),
            r_quadrature=int(r_quadrature),
            native_rate_init_bias=float(native_rate_init_bias),
            flow_spatial_smooth=float(flow_spatial_smooth),
            flow_temporal_smooth=float(flow_temporal_smooth),
            flow_smooth_time_points=int(flow_smooth_time_points),
            rate_floor=float(rate_floor),
        )

        self.num_genes = self.config.num_genes

        self.baseline_net = _MLP(
            1,
            self.num_genes,
            self.config.baseline_hidden,
            "gelu",
        )
        self.rate_net = _MLP(
            2,
            1,
            self.config.rate_hidden,
            "silu",
        )
        self.decoder_net = _MLP(
            2,
            self.num_genes,
            self.config.decoder_hidden,
            "gelu",
        )
        self.inverse_flow = MonotoneInverseFlow(
            self.config.flow_segments,
            self.config.flow_hidden,
        )

        decoder_final = _final_linear(self.decoder_net)
        nn.init.normal_(decoder_final.weight, mean=0.0, std=1e-3)
        if decoder_final.bias is not None:
            nn.init.zeros_(decoder_final.bias)

        rate_final = _final_linear(self.rate_net)
        nn.init.normal_(rate_final.weight, mean=0.0, std=1e-3)
        if rate_final.bias is not None:
            nn.init.constant_(rate_final.bias, self.config.native_rate_init_bias)

        nodes_np, weights_np = np.polynomial.legendre.leggauss(
            self.config.r_quadrature
        )
        nodes_np = (0.5 * (nodes_np + 1.0)).astype(np.float32)
        weights_np = (0.5 * weights_np).astype(np.float32)
        self.register_buffer(
            "quadrature_nodes",
            torch.tensor(nodes_np, dtype=torch.float32),
        )
        self.register_buffer(
            "quadrature_weights",
            torch.tensor(weights_np, dtype=torch.float32),
        )

    def config_dict(self) -> dict:
        return asdict(self.config)

    @classmethod
    def from_config(cls, config: dict) -> "FlowAwareStage2Model":
        return cls(**config)

    def baseline(self, a: torch.Tensor) -> torch.Tensor:
        return self.baseline_net(_as_col(a).clamp(0.0, 1.0))

    def inverse_coordinate(
        self,
        z: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        return self.inverse_flow(z, time)

    def material_rate(
        self,
        a: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        ac = _as_col(a).clamp(0.0, 1.0)
        tc = _as_col(time).clamp(0.0, 1.0)
        raw = self.rate_net(torch.cat((ac, tc), dim=1))
        return F.relu(raw) + self.config.rate_floor

    def material_R(
        self,
        a: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        ac = _as_col(a).clamp(0.0, 1.0)
        tc = _as_col(time).clamp(0.0, 1.0)

        n = ac.shape[0]
        q = self.config.r_quadrature
        tau = tc * self.quadrature_nodes[None, :]
        a_rep = ac.expand(n, q)

        inp = torch.stack((a_rep, tau), dim=-1).reshape(-1, 2)
        rate = F.relu(self.rate_net(inp)).reshape(n, q)
        if self.config.rate_floor > 0.0:
            rate = rate + self.config.rate_floor

        return tc * (
            rate * self.quadrature_weights[None, :]
        ).sum(dim=1, keepdim=True)

    def raw_decoder(
        self,
        a: torch.Tensor,
        R: torch.Tensor,
    ) -> torch.Tensor:
        ac = _as_col(a).clamp(0.0, 1.0)
        rc = _as_col(R)
        return self.decoder_net(torch.cat((ac, rc), dim=1))

    def developmental(
        self,
        a: torch.Tensor,
        R: torch.Tensor,
    ) -> torch.Tensor:
        rc = _as_col(R)
        return (
            self.raw_decoder(a, rc)
            - self.raw_decoder(a, torch.zeros_like(rc))
        )

    def rho(
        self,
        z: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        a = self.inverse_flow(z, time)
        return self.material_R(a, time)

    def forward(
        self,
        z: torch.Tensor,
        time: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        a = self.inverse_flow(z, time)
        R_value = self.material_R(a, time)
        log_mu = self.baseline(a) + self.developmental(a, R_value)

        return {
            "a": a,
            "R": R_value,
            "rho": R_value,
            "log_mu": log_mu,
            "mu": torch.exp(log_mu),
        }

    def data_nll(
        self,
        counts: torch.Tensor,
        log_mu: torch.Tensor,
    ) -> torch.Tensor:
        return F.poisson_nll_loss(
            log_mu,
            counts.float(),
            log_input=True,
            full=True,
            reduction="mean",
        )

    def velocity(
        self,
        z: torch.Tensor,
        time: torch.Tensor,
        *,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """Eulerian V(z,t)=-A_t/A_z."""
        zc = _as_col(z).detach()
        tc = _as_col(time).detach().clone().requires_grad_(True)

        with torch.enable_grad():
            a, a_z = self.inverse_flow(zc, tc, return_dz=True)
            a_t = torch.autograd.grad(
                a.sum(),
                tc,
                create_graph=create_graph,
            )[0]
            velocity = -a_t / a_z.clamp_min(1e-8)

        return velocity if create_graph else velocity.detach()

    def material_R_t(
        self,
        a: torch.Tensor,
        time: torch.Tensor,
        *,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """Derivative of the actual quadrature-defined R at fixed material a."""
        ac = _as_col(a).detach()
        tc = _as_col(time).detach().clone().requires_grad_(True)

        with torch.enable_grad():
            R_value = self.material_R(ac, tc)
            derivative = torch.autograd.grad(
                R_value.sum(),
                tc,
                create_graph=create_graph,
            )[0]

        return derivative if create_graph else derivative.detach()

    def eulerian_q(
        self,
        z: torch.Tensor,
        time: torch.Tensor,
        *,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """q(z,t)=R_t(A(z,t),t)."""
        with torch.no_grad():
            a = self.inverse_flow(z, time).detach()
        return self.material_R_t(a, time, create_graph=create_graph)

    @torch.no_grad()
    def forward_map(
        self,
        a: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        return self.inverse_flow.forward_map(a, time)

    def flow_penalty(self) -> torch.Tensor:
        if (
            self.config.flow_spatial_smooth == 0.0
            and self.config.flow_temporal_smooth == 0.0
        ):
            parameter = next(self.parameters())
            return torch.zeros((), device=parameter.device, dtype=parameter.dtype)

        device = next(self.parameters()).device
        time_grid = torch.linspace(
            0.0,
            1.0,
            self.config.flow_smooth_time_points,
            device=device,
        )[:, None]
        spatial, temporal = self.inverse_flow.smoothness_penalty(time_grid)
        return (
            self.config.flow_spatial_smooth * spatial
            + self.config.flow_temporal_smooth * temporal
        )

    def freeze_baseline(self) -> None:
        for parameter in self.baseline_net.parameters():
            parameter.requires_grad_(False)

    def unfreeze_baseline(self) -> None:
        for parameter in self.baseline_net.parameters():
            parameter.requires_grad_(True)


DirectFlowStage2 = FlowAwareStage2Model
