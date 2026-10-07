"""Stage 2: derivatives of E(z,t), closed-form Fisher velocity, and RK4 flow.

Stage 2 has no separately learned trajectory model. It differentiates the
Stage-1 biological mean-expression field E(z,t), computes the Fisher-projection
velocity in closed form, and integrates dF/dt = V(F,t) with RK4.
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
    """Return (E, E_z, E_t, log_E).

    If the field supplies analytic log-mean derivatives, they are converted to
    mean-scale derivatives. Otherwise E is differentiated directly by autograd.
    """
    analytic = getattr(field, "log_mean_derivatives", None)
    if analytic is not None:
        zc = _as_col(z)
        tc = _as_col(time)
        log_e, eta_z, eta_t = analytic(zc, tc)
        expression = torch.exp(log_e)
        return (
            expression,
            expression * eta_z,
            expression * eta_t,
            log_e,
        )

    zc = _as_col(z).detach().clone().requires_grad_(True)
    tc = _as_col(time).detach().clone().requires_grad_(True)
    if zc.shape[0] != tc.shape[0]:
        raise ValueError("z and time must have matching rows.")

    log_e = field(zc, tc)
    if log_e.ndim != 2 or log_e.shape[0] != zc.shape[0]:
        raise ValueError("Expression field must return shape [N, genes].")

    expression = torch.exp(log_e)
    dE_dz = _component_derivative(
        expression,
        zc,
        create_graph=create_graph,
    )
    dE_dt = _component_derivative(
        expression,
        tc,
        create_graph=create_graph,
    )
    return expression, dE_dz, dE_dt, log_e


def derived_velocity(
    expression: Tensor,
    dE_dz: Tensor,
    dE_dt: Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Return the Fisher closed-form velocity and tangent information.

    V = - <E_z,E_t>_I / <E_z,E_z>_I, with
    <u,v>_I = sum_g u_g v_g / E_g.
    """
    if expression.shape != dE_dz.shape or expression.shape != dE_dt.shape:
        raise ValueError("expression, dE_dz, and dE_dt must match in shape.")

    weight = expression.clamp_min(float(eps)).reciprocal()
    numerator = (dE_dz * dE_dt * weight).sum(dim=1, keepdim=True)
    tangent_energy = (dE_dz.square() * weight).sum(dim=1, keepdim=True)
    velocity = -numerator / (tangent_energy + float(eps))
    return velocity, tangent_energy


@dataclass(frozen=True)
class LocalDynamics:
    z: Tensor
    time: Tensor
    log_expression: Tensor
    expression: Tensor
    dE_dz: Tensor
    dE_dt: Tensor
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
    """Estimate E_z and E_t, then compute the closed-form Fisher velocity."""
    expression, dE_dz, dE_dt, log_e = field_derivatives(
        field,
        z,
        time,
        create_graph=create_graph,
    )
    velocity, tangent_energy = derived_velocity(
        expression,
        dE_dz,
        dE_dt,
        eps=eps,
    )
    residual = dE_dt + velocity * dE_dz
    weight = expression.clamp_min(float(eps)).reciprocal()
    residual_speed = torch.sqrt(
        (residual.square() * weight).sum(dim=1, keepdim=True)
    )
    return LocalDynamics(
        z=_as_col(z),
        time=_as_col(time),
        log_expression=log_e,
        expression=expression,
        dE_dz=dE_dz,
        dE_dt=dE_dt,
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
    """Evaluate V(z,t) from the Stage-1 expression field."""
    return estimate_local_dynamics(
        field,
        z,
        time,
        eps=eps,
        create_graph=False,
    ).velocity.detach()


@dataclass(frozen=True)
class FlowTrajectory:
    """RK4 solution of dF/dt = V(F,t)."""

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
    velocities = [velocity_field(field, z, times[-1], eps=eps)]

    def v(z_state: Tensor, t_scalar: float) -> Tensor:
        tt = torch.full_like(z_state, float(t_scalar))
        return velocity_field(field, z_state, tt, eps=eps)

    for step in range(n_steps):
        t = float(t0) + step * dt
        k1 = v(z, t)
        k2 = v(z + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = v(z + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = v(z + dt * k3, t + dt)

        z = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        tt = torch.full_like(z, t + dt)
        times.append(tt)
        positions.append(z.clone())
        velocities.append(velocity_field(field, z, tt, eps=eps))

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
