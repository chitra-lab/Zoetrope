"""Stage 2: derivatives of E(z,t), closed-form Fisher velocity, and RK4 flow.

Stage 2 has no separately learned neural model. It differentiates the
Stage-1 biological expression field E(z,t), computes

    V(z,t) = - <Psi_z, Psi_t> / <Psi_z, Psi_z>,
    Psi = 2 sqrt(E),

and integrates dF/dt = V(F,t) with fourth-order Runge-Kutta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch

Tensor = torch.Tensor


class ExpressionField(Protocol):
    def __call__(self, z: Tensor, time: Tensor) -> Tensor:
        """Return log E(z,t) with shape [n_cells, n_genes]."""


def _as_col(x: Tensor) -> Tensor:
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2 and x.shape[1] == 1:
        return x
    raise ValueError(f"Expected [N] or [N,1], got {tuple(x.shape)}")


def _component_derivative(
    values: Tensor,
    inputs: Tensor,
    *,
    create_graph: bool,
) -> Tensor:
    columns = []
    for gene in range(values.shape[1]):
        grad = torch.autograd.grad(
            values[:, gene].sum(),
            inputs,
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad is None:
            grad = torch.zeros_like(inputs)
        columns.append(grad)
    return torch.cat(columns, dim=1)


def field_derivatives(
    field: ExpressionField | Callable[[Tensor, Tensor], Tensor],
    z: Tensor,
    time: Tensor,
    *,
    create_graph: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (log_E, Psi, Psi_z, Psi_t) in Poisson Fisher coordinates.

    If the field provides an analytic log_mean_derivatives method it is used
    directly. Otherwise derivatives are obtained by autograd, which is the
    path used by the neural Stage-1 expression decoder.
    """
    analytic = getattr(field, "log_mean_derivatives", None)
    if analytic is not None:
        zc = _as_col(z)
        tc = _as_col(time)
        log_e, eta_z, eta_t = analytic(zc, tc)
        sqrt_e = torch.exp(0.5 * log_e)
        psi = 2.0 * sqrt_e
        return log_e, psi, sqrt_e * eta_z, sqrt_e * eta_t

    zc = _as_col(z).detach().clone().requires_grad_(True)
    tc = _as_col(time).detach().clone().requires_grad_(True)
    if zc.shape[0] != tc.shape[0]:
        raise ValueError("z and time must have matching rows.")

    log_e = field(zc, tc)
    if log_e.ndim != 2 or log_e.shape[0] != zc.shape[0]:
        raise ValueError("Expression field must return shape [N, genes].")

    psi = 2.0 * torch.exp(0.5 * log_e)
    psi_z = _component_derivative(
        psi,
        zc,
        create_graph=create_graph,
    )
    psi_t = _component_derivative(
        psi,
        tc,
        create_graph=create_graph,
    )
    return log_e, psi, psi_z, psi_t


def derived_velocity(
    psi_z: Tensor,
    psi_t: Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Closed-form minimum-expression-change velocity and observability.

    Returns (V, <Psi_z,Psi_z>).
    """
    if psi_z.shape != psi_t.shape:
        raise ValueError("psi_z and psi_t must have matching shapes.")
    numerator = (psi_z * psi_t).sum(dim=1, keepdim=True)
    tangent_energy = psi_z.square().sum(dim=1, keepdim=True)
    velocity = -numerator / (tangent_energy + float(eps))
    return velocity, tangent_energy


@dataclass(frozen=True)
class LocalDynamics:
    z: Tensor
    time: Tensor
    log_expression: Tensor
    expression: Tensor
    fisher_embedding: Tensor
    dpsi_dz: Tensor
    dpsi_dt: Tensor
    velocity: Tensor
    residual: Tensor
    residual_speed: Tensor
    tangent_energy: Tensor

    @property
    def observability(self) -> Tensor:
        return self.tangent_energy


def estimate_local_dynamics(
    field: ExpressionField | Callable[[Tensor, Tensor], Tensor],
    z: Tensor,
    time: Tensor,
    *,
    eps: float = 1e-8,
    create_graph: bool = False,
) -> LocalDynamics:
    """Differentiate E(z,t) and compute the closed-form Fisher velocity."""
    log_e, psi, psi_z, psi_t = field_derivatives(
        field,
        z,
        time,
        create_graph=create_graph,
    )
    velocity, tangent_energy = derived_velocity(
        psi_z,
        psi_t,
        eps=eps,
    )
    residual = psi_t + velocity * psi_z
    residual_speed = torch.linalg.vector_norm(
        residual,
        dim=1,
        keepdim=True,
    )
    return LocalDynamics(
        z=_as_col(z),
        time=_as_col(time),
        log_expression=log_e,
        expression=torch.exp(log_e),
        fisher_embedding=psi,
        dpsi_dz=psi_z,
        dpsi_dt=psi_t,
        velocity=velocity,
        residual=residual,
        residual_speed=residual_speed,
        tangent_energy=tangent_energy,
    )


def velocity_field(
    field: ExpressionField | Callable[[Tensor, Tensor], Tensor],
    z: Tensor,
    time: Tensor,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Evaluate V(z,t) from Stage-1 expression derivatives."""
    return estimate_local_dynamics(
        field,
        z,
        time,
        eps=eps,
        create_graph=False,
    ).velocity.detach()


@dataclass(frozen=True)
class FlowTrajectory:
    """RK4 solution of dF/dt = V(F,t).

    position[k, j] is F(a_j, time[k]).
    """

    time: Tensor
    position: Tensor
    velocity: Tensor

    @property
    def final_position(self) -> Tensor:
        return self.position[-1]


def integrate_flow_rk4(
    field: ExpressionField | Callable[[Tensor, Tensor], Tensor],
    a: Tensor,
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    n_steps: int = 128,
    eps: float = 1e-8,
) -> FlowTrajectory:
    """Integrate F(a,t) with classical fourth-order Runge-Kutta."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if t1 <= t0:
        raise ValueError("Require t1 > t0.")

    z = _as_col(a).detach().clone()
    dt = float(t1 - t0) / float(n_steps)

    times = [torch.full_like(z, float(t0))]
    positions = [z.clone()]
    velocities = [
        velocity_field(field, z, times[-1], eps=eps)
    ]

    def v(z_state: Tensor, t_scalar: float) -> Tensor:
        tt = torch.full_like(z_state, float(t_scalar))
        return velocity_field(
            field,
            z_state,
            tt,
            eps=eps,
        )

    for step in range(n_steps):
        t = float(t0) + step * dt
        k1 = v(z, t)
        k2 = v(z + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = v(z + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = v(z + dt * k3, t + dt)

        z = z + (dt / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )

        tt = torch.full_like(z, t + dt)
        times.append(tt)
        positions.append(z.clone())
        velocities.append(
            velocity_field(field, z, tt, eps=eps)
        )

    return FlowTrajectory(
        time=torch.stack([x[:, 0] for x in times], dim=0),
        position=torch.stack([x[:, 0] for x in positions], dim=0),
        velocity=torch.stack([x[:, 0] for x in velocities], dim=0),
    )


__all__ = [
    "ExpressionField",
    "LocalDynamics",
    "FlowTrajectory",
    "field_derivatives",
    "derived_velocity",
    "estimate_local_dynamics",
    "velocity_field",
    "integrate_flow_rk4",
]
