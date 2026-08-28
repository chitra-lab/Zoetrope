"""Lagrangian integration of anatomical flow and Fisher-arclength pseudotime."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .geometry import fisher_geometry
from .model import Tensor, _as_col


class ExpressionField(Protocol):
    def __call__(self, z: Tensor, time: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class LagrangianTrajectory:
    time: Tensor
    position: Tensor
    rho: Tensor
    speed: Tensor
    velocity: Tensor


def local_dynamics(
    field: ExpressionField,
    z: Tensor,
    time: Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Return anatomical velocity and Fisher speed along the expression trajectory."""
    geom = fisher_geometry(field, z, time, eps=eps, create_graph=False)
    return geom.velocity.detach(), geom.speed.detach()


def integrate_flow_and_rho(
    field: ExpressionField,
    a: Tensor,
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    n_steps: int = 128,
    eps: float = 1e-8,
) -> LagrangianTrajectory:
    r"""Integrate ``dF/dt=V(F,t)`` and Fisher-arclength ``d rho/dt`` with RK4.

    The anatomical coordinate is not intrinsically bounded, so trajectories are
    never clamped to ``[0,1]``.  The fitted expression field must have spline
    support over the part of ``z`` visited by trajectories used for inference.
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
        tt = torch.full_like(z, tk + dt)
        vv, qq = local_dynamics(field, z, tt, eps=eps)
        times.append(tt)
        positions.append(z.clone())
        rhos.append(rho.clone())
        velocities.append(vv)
        speeds.append(qq)

    return LagrangianTrajectory(
        time=torch.cat(times, dim=1).T,
        position=torch.cat(positions, dim=1).T,
        rho=torch.cat(rhos, dim=1).T,
        speed=torch.cat(speeds, dim=1).T,
        velocity=torch.cat(velocities, dim=1).T,
    )


def rho_at(
    field: ExpressionField,
    a: Tensor,
    time: float,
    *,
    n_steps: int = 128,
    eps: float = 1e-8,
) -> Tensor:
    if time == 0:
        return torch.zeros_like(_as_col(a))
    traj = integrate_flow_and_rho(
        field, a, t0=0.0, t1=float(time), n_steps=n_steps, eps=eps
    )
    return traj.rho[-1][:, None]
