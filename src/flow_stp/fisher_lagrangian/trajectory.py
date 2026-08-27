"""Lagrangian integration for the Fisher-Lagrangian model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry import fisher_geometry
from .model import LogMeanExpressionField, Tensor, _as_col


@dataclass(frozen=True)
class LagrangianTrajectory:
    time: Tensor
    position: Tensor
    rho: Tensor
    speed: Tensor
    velocity: Tensor


def local_dynamics(
    field: LogMeanExpressionField,
    z: Tensor,
    time: Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Return ``(V(z,t), q(z,t))`` derived from Fisher geometry."""
    geom = fisher_geometry(field, z, time, eps=eps, create_graph=False)
    return geom.velocity.detach(), geom.speed.detach()


def integrate_flow_and_rho(
    field: LogMeanExpressionField,
    a: Tensor,
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    n_steps: int = 128,
    eps: float = 1e-8,
    clamp_coordinate: bool = True,
) -> LagrangianTrajectory:
    """Integrate material position and cumulative Fisher arclength with RK4.

    The state equations are

        dZ/dt   = V(Z,t)
        drho/dt = q(Z,t)

    with ``Z(a,t0)=a`` and ``rho(a,t0)=0``.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if t1 <= t0:
        raise ValueError("Require t1 > t0")

    z = _as_col(a).detach().clone()
    rho = torch.zeros_like(z)
    dt = float(t1 - t0) / float(n_steps)

    times = [torch.full_like(z, float(t0))]
    positions = [z.clone()]
    rhos = [rho.clone()]
    v0, q0 = local_dynamics(field, z, times[-1], eps=eps)
    velocities = [v0]
    speeds = [q0]

    def dyn(z_state: Tensor, t_scalar: float) -> tuple[Tensor, Tensor]:
        tt = torch.full_like(z_state, float(t_scalar))
        return local_dynamics(field, z_state, tt, eps=eps)

    for k in range(n_steps):
        tk = float(t0) + k * dt

        k1z, k1r = dyn(z, tk)
        k2z, k2r = dyn(z + 0.5 * dt * k1z, tk + 0.5 * dt)
        k3z, k3r = dyn(z + 0.5 * dt * k2z, tk + 0.5 * dt)
        k4z, k4r = dyn(z + dt * k3z, tk + dt)

        z = z + (dt / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
        rho = rho + (dt / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)

        if clamp_coordinate:
            z = z.clamp(0.0, 1.0)

        tnext = tk + dt
        tt = torch.full_like(z, tnext)
        vv, qq = local_dynamics(field, z, tt, eps=eps)

        times.append(tt)
        positions.append(z.clone())
        rhos.append(rho.clone())
        velocities.append(vv)
        speeds.append(qq)

    # [steps+1, trajectories]
    return LagrangianTrajectory(
        time=torch.cat(times, dim=1).T,
        position=torch.cat(positions, dim=1).T,
        rho=torch.cat(rhos, dim=1).T,
        speed=torch.cat(speeds, dim=1).T,
        velocity=torch.cat(velocities, dim=1).T,
    )


def rho_at(
    field: LogMeanExpressionField,
    a: Tensor,
    time: float,
    *,
    n_steps: int = 128,
    eps: float = 1e-8,
) -> Tensor:
    """Convenience function returning accumulated development at one time."""
    if time == 0:
        return torch.zeros_like(_as_col(a))
    traj = integrate_flow_and_rho(
        field,
        a,
        t0=0.0,
        t1=float(time),
        n_steps=n_steps,
        eps=eps,
    )
    return traj.rho[-1][:, None]
