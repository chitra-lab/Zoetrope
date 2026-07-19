"""Flow-STP model source.

Extracted from the prototype notebook.

Important convention:
- rho_total = accumulated q
- rho_static = 0
- rho_dev = rho_total

The q implementation removes the learned spatially constant q baseline at
each time by subtracting min_z q_pos(z,t).
"""

# ============================================================
# Flow-aware STP prototype: unregularized non-OT version
# with NO learned constant baseline in q
#
# Full source-code replacement cell.
#
# Original q:
# q(z,t) = softplus(q_net(z,t)) + epsilon
#
# New q:
# q_pos(z,t) = softplus(q_net(z,t))
# q_floor(t) = min_z q_pos(z,t)
# q(z,t) = q_pos(z,t) - q_floor(t)
#
# This removes:
# 1. explicit q_epsilon floor
# 2. learned spatially constant q baseline at each t
#
# But q is NOT forced to be constant.
# It can still vary over z and t.
#
# Objective:
# total loss = Poisson NLL only
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


def as_col(x: Tensor) -> Tensor:
    """Ensure tensor has shape [B, 1]."""
    if x.ndim == 1:
        return x[:, None]

    if x.ndim == 2 and x.shape[1] == 1:
        return x

    raise ValueError(f"Expected shape [B] or [B,1], got {tuple(x.shape)}")


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int] = (128, 128),
        activation: str = "tanh",
    ):
        super().__init__()

        if activation == "tanh":
            act = nn.Tanh
        elif activation == "gelu":
            act = nn.GELU
        elif activation == "relu":
            act = nn.ReLU
        elif activation == "silu":
            act = nn.SiLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        prev = in_dim

        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(act())
            prev = h

        layers.append(nn.Linear(prev, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SliceCoordinateMaps(nn.Module):
    """
    One learned coordinate map alpha_i per tissue slice.

        alpha_i(s) = a_i + (b_i - a_i) sigmoid(f_i(s))

    where s is a native 2D tissue coordinate and alpha_i(s) lies in [0,1].
    """

    def __init__(
        self,
        num_slices: int,
        hidden: Sequence[int] = (64, 64),
        min_width: float = 0.05,
    ):
        super().__init__()

        self.num_slices = int(num_slices)
        self.min_width = float(min_width)

        self.maps = nn.ModuleList(
            [
                MLP(
                    in_dim=2,
                    out_dim=1,
                    hidden=hidden,
                    activation="tanh",
                )
                for _ in range(num_slices)
            ]
        )

        self.raw_a = nn.Parameter(torch.zeros(num_slices))
        self.raw_width = nn.Parameter(torch.zeros(num_slices))

    def intervals(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns slice-specific interval endpoints a_i, b_i and width b_i-a_i.
        """
        width = self.min_width + (1.0 - self.min_width) * torch.sigmoid(
            self.raw_width
        )

        a = (1.0 - width) * torch.sigmoid(self.raw_a)
        b = a + width

        return a, b, width

    def forward(self, coords: Tensor, slice_idx: Tensor) -> Tensor:
        """
        Args:
            coords: [B, 2] native tissue coordinates
            slice_idx: [B] integer slice IDs

        Returns:
            z: [B, 1] aligned atlas coordinate in [0,1]
        """
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"coords must have shape [B,2], got {tuple(coords.shape)}"
            )

        slice_idx = slice_idx.long().view(-1)

        if slice_idx.shape[0] != coords.shape[0]:
            raise ValueError("slice_idx must have one entry per coordinate.")

        a, b, width = self.intervals()

        z = torch.empty(
            coords.shape[0],
            1,
            device=coords.device,
            dtype=coords.dtype,
        )

        for i in torch.unique(slice_idx).tolist():
            mask = slice_idx == int(i)
            raw = torch.sigmoid(self.maps[int(i)](coords[mask]))
            z[mask] = a[int(i)] + width[int(i)] * raw

        return z


@dataclass
class FlowSTPConfig:
    num_slices: int
    num_genes: int

    coord_hidden: Tuple[int, ...] = (64, 64)
    field_hidden: Tuple[int, ...] = (128, 128)
    decoder_hidden: Tuple[int, ...] = (128, 128)

    n_flow_steps: int = 24
    coord_min_width: float = 0.05

    # Kept for compatibility with older config objects.
    # This new source code does NOT add q_epsilon to q.
    q_epsilon: float = 0.0

    clamp_eps: float = 1e-5

    # Keep 0 for the simplest prototype.
    # Set >0 later for low-rank slice effects.
    slice_effect_dim: int = 0


class FlowSTPModel(nn.Module):
    """
    Unregularized non-OT flow-aware STP model.

    Model:

        z_c = alpha_i(x_c)

        dF(a,t)/dt = V(F(a,t),t), F(a,0)=a

        V(z,t) = z(1-z) V_tilde(z,t)

        q_pos(z,t) = softplus(q_net(z,t))
        q_floor(t) = min_z q_pos(z,t)
        q(z,t) = q_pos(z,t) - q_floor(t)

        R(a,t) = int_0^t q(F(a,tau), tau) d tau

        rho(z_c,t_i) = R(F^{-1}(z_c,t_i), t_i)

        log mu_gc =
            beta_g
            + A_g(z_c)
            + B_g(a_c)
            + C_g(rho_c)
            + D_g(a_c,rho_c)
            + optional log_library_size_c

        y_gc ~ Poisson(mu_gc)

    Training objective:

        total loss = Poisson NLL only
    """

    # Fixed internal grid used only to remove the spatially constant q mode.
    # This is intentionally not exposed as a tuning hyperparameter.
    Q_FLOOR_GRID_N = 128

    def __init__(self, config: FlowSTPConfig):
        super().__init__()

        self.config = config

        self.coord_maps = SliceCoordinateMaps(
            num_slices=config.num_slices,
            hidden=config.coord_hidden,
            min_width=config.coord_min_width,
        )

        self.v_tilde_net = MLP(
            in_dim=2,
            out_dim=1,
            hidden=config.field_hidden,
            activation="tanh",
        )

        self.q_net = MLP(
            in_dim=2,
            out_dim=1,
            hidden=config.field_hidden,
            activation="tanh",
        )

        # No learned R0_net in this variant.
        # The static/reference potential is fixed to zero.

        G = int(config.num_genes)

        self.beta = nn.Parameter(torch.zeros(G))

        self.A = MLP(
            1,
            G,
            hidden=config.decoder_hidden,
            activation="gelu",
        )

        self.B = MLP(
            1,
            G,
            hidden=config.decoder_hidden,
            activation="gelu",
        )

        self.C = MLP(
            1,
            G,
            hidden=config.decoder_hidden,
            activation="gelu",
        )

        self.D = MLP(
            2,
            G,
            hidden=config.decoder_hidden,
            activation="gelu",
        )

        if config.slice_effect_dim > 0:
            self.slice_embedding = nn.Embedding(
                config.num_slices,
                config.slice_effect_dim,
            )

            self.slice_gene_loadings = nn.Parameter(
                0.01 * torch.randn(G, config.slice_effect_dim)
            )
        else:
            self.slice_embedding = None
            self.slice_gene_loadings = None

    # ------------------------------------------------------------------
    # Core fields
    # ------------------------------------------------------------------

    def velocity(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Boundary-preserving velocity field:

            V(z,t) = z(1-z) V_tilde(z,t)

        This guarantees V(0,t)=V(1,t)=0.
        """
        z = as_col(z)
        t = as_col(t)

        inp = torch.cat([z, t], dim=1)

        return z * (1.0 - z) * self.v_tilde_net(inp)

    def q_positive(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Raw nonnegative q before removing the spatially constant floor:

            q_pos(z,t) = softplus(q_net(z,t))
        """
        z = as_col(z)
        t = as_col(t)

        inp = torch.cat([z, t], dim=1)

        return F.softplus(self.q_net(inp))

    def q_floor_for_t(self, t: Tensor) -> Tensor:
        """
        Approximate:

            q_floor(t) = min_z q_pos(z,t)

        on a fixed z grid.

        This removes the learned spatially constant q baseline at each t.
        """
        t = as_col(t)

        device = t.device
        dtype = t.dtype

        t_flat = t.reshape(-1)

        # t is not learned, so detaching t for grouping is safe.
        t_unique, inverse = torch.unique(
            t_flat.detach(),
            sorted=True,
            return_inverse=True,
        )

        t_unique = t_unique.to(
            device=device,
            dtype=dtype,
        ).reshape(-1, 1)

        z_grid = torch.linspace(
            0.0,
            1.0,
            int(self.Q_FLOOR_GRID_N),
            device=device,
            dtype=dtype,
        ).reshape(1, -1, 1)

        n_t = t_unique.shape[0]
        n_z = z_grid.shape[1]

        z_eval = z_grid.expand(n_t, n_z, 1).reshape(-1, 1)
        t_eval = t_unique[:, None, :].expand(n_t, n_z, 1).reshape(-1, 1)

        q_eval = self.q_positive(z_eval, t_eval).reshape(n_t, n_z)

        q_floor_unique = q_eval.min(dim=1, keepdim=False).values.reshape(-1, 1)
        q_floor = q_floor_unique[inverse].reshape(-1, 1)

        return q_floor

    def q(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Nonnegative local developmental rate with no spatially constant
        learned baseline:

            q_pos(z,t) = softplus(q_net(z,t))
            q_floor(t) = min_z q_pos(z,t)
            q(z,t) = q_pos(z,t) - q_floor(t)

        If q_pos is spatially constant at a fixed time t, q becomes zero
        at that time. If q_pos varies over z, q keeps the variation.
        """
        z = as_col(z)
        t = as_col(t)

        q_pos = self.q_positive(z, t)
        q_floor = self.q_floor_for_t(t)
        q_centered = q_pos - q_floor

        # Numerical safety only. This is not an additive epsilon floor.
        return torch.clamp(q_centered, min=0.0)

    def R0(self, a: Tensor) -> Tensor:
        """Compatibility method for the no-R0 variant.

        This model version removes the learned static/reference potential.
        Therefore R0(a) is identically zero and rho_total is purely the
        accumulated q integral.
        """
        return torch.zeros_like(as_col(a))

    # ------------------------------------------------------------------
    # Flow and STP evaluation
    # ------------------------------------------------------------------

    def inverse_reference_and_rho(self, z: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute:

            a = F^{-1}(z,t)

        and:

            rho(z,t) = R(a,t)

        This integrates the flow backward from current coordinate z at time t
        to time 0. Along that path, it accumulates:

            int_0^t q(F(a,tau), tau) d tau.
        """
        z = as_col(z)
        t = as_col(t)

        x = z
        integral = torch.zeros_like(z)

        n_steps = int(self.config.n_flow_steps)
        h = t / n_steps

        for k in range(n_steps):
            # Reverse-time point s: t -> 0.
            s = t - k * h

            # Midpoint step.
            v1 = self.velocity(x, s)
            x_mid = x - 0.5 * h * v1
            s_mid = s - 0.5 * h

            x_mid = x_mid.clamp(
                self.config.clamp_eps,
                1.0 - self.config.clamp_eps,
            )
            s_mid = s_mid.clamp(0.0, 1.0)

            integral = integral + h * self.q(x_mid, s_mid)

            v_mid = self.velocity(x_mid, s_mid)
            x = x - h * v_mid

            x = x.clamp(
                self.config.clamp_eps,
                1.0 - self.config.clamp_eps,
            )

        a = x
        rho = integral

        return a, rho

    def flow_forward(self, a: Tensor, t: Tensor) -> Tensor:
        """
        Compute F(a,t) by integrating the ODE forward.
        Mostly useful for diagnostics.
        """
        a = as_col(a)
        t = as_col(t)

        x = a

        n_steps = int(self.config.n_flow_steps)
        h = t / n_steps

        for k in range(n_steps):
            s = k * h

            v1 = self.velocity(x, s)
            x_mid = x + 0.5 * h * v1
            s_mid = s + 0.5 * h

            x_mid = x_mid.clamp(
                self.config.clamp_eps,
                1.0 - self.config.clamp_eps,
            )
            s_mid = s_mid.clamp(0.0, 1.0)

            v_mid = self.velocity(x_mid, s_mid)
            x = x + h * v_mid

            x = x.clamp(
                self.config.clamp_eps,
                1.0 - self.config.clamp_eps,
            )

        return x

    def R_from_reference(self, a: Tensor, t: Tensor) -> Tensor:
        """
        Compute R(a,t) by integrating forward from reference coordinate a.

        This is retained for diagnostics, but is not used in the loss.
        """
        a = as_col(a)
        t = as_col(t)

        x = a
        integral = torch.zeros_like(a)

        n_steps = int(self.config.n_flow_steps)
        h = t / n_steps

        for k in range(n_steps):
            s = k * h

            v1 = self.velocity(x, s)
            x_mid = x + 0.5 * h * v1
            s_mid = s + 0.5 * h

            x_mid = x_mid.clamp(
                self.config.clamp_eps,
                1.0 - self.config.clamp_eps,
            )
            s_mid = s_mid.clamp(0.0, 1.0)

            integral = integral + h * self.q(x_mid, s_mid)

            v_mid = self.velocity(x_mid, s_mid)
            x = x + h * v_mid

            x = x.clamp(
                self.config.clamp_eps,
                1.0 - self.config.clamp_eps,
            )

        return integral

    # ------------------------------------------------------------------
    # Gene-expression decoder
    # ------------------------------------------------------------------

    def decode_log_mean(
        self,
        z: Tensor,
        a: Tensor,
        rho: Tensor,
        slice_idx: Tensor,
        log_library_size: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Decode log mean expression:

            log mu_gc =
                beta_g
                + A_g(z_c)
                + B_g(a_c)
                + C_g(rho_c)
                + D_g(a_c,rho_c)

        Optional library-size normalization adds log ell_c.
        """
        z = as_col(z)
        a = as_col(a)
        rho = as_col(rho)

        eta = (
            self.beta[None, :]
            + self.A(z)
            + self.B(a)
            + self.C(rho)
            + self.D(torch.cat([a, rho], dim=1))
        )

        if self.slice_embedding is not None:
            emb = self.slice_embedding(slice_idx.long().view(-1))
            slice_effect = emb @ self.slice_gene_loadings.T
            eta = eta + slice_effect

        if log_library_size is not None:
            eta = eta + as_col(log_library_size)

        return eta

    def forward(
        self,
        coords: Tensor,
        slice_idx: Tensor,
        time: Tensor,
        log_library_size: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            coords: [B,2] native spatial coordinates
            slice_idx: [B] integer slice IDs
            time: [B] or [B,1] observation times in [0,1]
            log_library_size: optional [B] or [B,1]

        Returns:
            Dictionary with z, a, rho/rho_total, rho_static, rho_dev,
            log_mu, and mu.

        No-R0 variant convention:
            rho_total = accumulated q integral
            rho_static = 0
            rho_dev = rho_total
        """
        z = self.coord_maps(coords, slice_idx)
        a, rho_total = self.inverse_reference_and_rho(z, time)

        rho_static = self.R0(a)
        rho_dev = rho_total - rho_static

        log_mu = self.decode_log_mean(
            z=z,
            a=a,
            rho=rho_total,
            slice_idx=slice_idx,
            log_library_size=log_library_size,
        )

        mu = torch.exp(log_mu)

        return {
            "z": z,
            "a": a,

            # Backwards-compatible alias.
            "rho": rho_total,

            # Explicit decomposition.
            "rho_total": rho_total,
            "rho_static": rho_static,
            "rho_dev": rho_dev,

            "log_mu": log_mu,
            "mu": mu,
        }

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def poisson_nll(
        self,
        counts: Tensor,
        log_mu: Tensor,
        full: bool = True,
    ) -> Tensor:
        """Poisson negative log likelihood using log_mu as the log-rate."""
        return F.poisson_nll_loss(
            input=log_mu,
            target=counts.float(),
            log_input=True,
            full=full,
            reduction="mean",
        )

    def loss(
        self,
        batch: Mapping[str, Tensor],
        weights: Optional[Mapping[str, float]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Expected batch keys:
            coords: [B,2]
            slice_idx: [B]
            time: [B] or [B,1]
            counts: [B,G]
            log_library_size: optional [B] or [B,1]

        Explicit regularization terms are intentionally removed.
        The weights argument is accepted for compatibility but ignored.
        """
        out = self.forward(
            coords=batch["coords"],
            slice_idx=batch["slice_idx"],
            time=batch["time"],
            log_library_size=batch.get("log_library_size", None),
        )

        data_loss = self.poisson_nll(batch["counts"], out["log_mu"])
        total = data_loss

        zero = torch.zeros(
            (),
            device=out["log_mu"].device,
            dtype=out["log_mu"].dtype,
        )

        parts = {
            "total": total.detach(),
            "data": data_loss.detach(),

            # Compatibility keys for existing training loops/logging.
            # These are not computed and not used in the objective.
            "velocity": zero.detach(),
            "q": zero.detach(),
            "scale": zero.detach(),
            "coord_smooth": zero.detach(),
            "coord_width": zero.detach(),
            "coord_var_reward": zero.detach(),
            "slice": zero.detach(),

            # Useful diagnostics.
            "z_mean": out["z"].mean().detach(),
            "a_mean": out["a"].mean().detach(),
            "rho_mean": out["rho"].mean().detach(),
        }

        return total, parts


def train_one_epoch(
    model: FlowSTPModel,
    dataloader: Iterable[Mapping[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weights: Optional[Mapping[str, float]] = None,
    grad_clip_norm: Optional[float] = 5.0,
) -> Dict[str, float]:
    """
    Basic one-epoch training loop.

    The dataloader should yield dictionaries with keys:
        coords
        slice_idx
        time
        counts

    and optionally:
        log_library_size

    The weights argument is accepted for compatibility but ignored by the
    unregularized model loss.
    """
    model.train()

    totals: Dict[str, float] = {}
    n_batches = 0

    for batch in dataloader:
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        optimizer.zero_grad(set_to_none=True)

        loss, parts = model.loss(batch, weights=weights)
        loss.backward()

        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + float(v)

        n_batches += 1

    return {
        k: v / max(n_batches, 1)
        for k, v in totals.items()
    }


@torch.no_grad()
def evaluate_poisson_nll(
    model: FlowSTPModel,
    dataloader: Iterable[Mapping[str, Tensor]],
    device: torch.device,
) -> float:
    """Compute held-out Poisson NLL."""
    model.eval()

    total = 0.0
    n_batches = 0

    for batch in dataloader:
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        out = model(
            coords=batch["coords"],
            slice_idx=batch["slice_idx"],
            time=batch["time"],
            log_library_size=batch.get("log_library_size", None),
        )

        nll = model.poisson_nll(batch["counts"], out["log_mu"])

        total += float(nll)
        n_batches += 1

    return total / max(n_batches, 1)
