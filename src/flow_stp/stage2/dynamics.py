"""Stage 2: direct Fisher-flat derivatives, closed-form velocity, and RK4.

Stage 2 has no separately learned trajectory model. Stage 1 fits log E and
derives the independent-Poisson Fisher-flat field

    Psi(z,t) = 2 sqrt(E(z,t))

from that fit. Stage 2 differentiates the derived field, computes the Euclidean
projection velocity

    V(z,t) = - <Psi_z, Psi_t> / <Psi_z, Psi_z>,

and integrates dF/dt = V(F,t) with classical fourth-order Runge-Kutta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch

Tensor = torch.Tensor


class FisherFlatField(Protocol):
    def __call__(self, z: Tensor, time: Tensor) -> Tensor:
        """Return Psi(z,t) with shape [n_cells, n_genes]."""


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
    field: FisherFlatField | Callable[[Tensor, Tensor], Tensor],
    z: Tensor,
    time: Tensor,
    *,
    create_graph: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (Psi, Psi_z, Psi_t) by differentiating the fitted flat field."""
    zc = _as_col(z).detach().clone().requires_grad_(True)
    tc = _as_col(time).detach().clone().requires_grad_(True)

    if zc.shape[0] != tc.shape[0]:
        raise ValueError("z and time must have matching rows.")

    psi = field(zc, tc)

    if psi.ndim != 2 or psi.shape[0] != zc.shape[0]:
        raise ValueError(
            "Fisher-flat field must return shape [N, genes]."
        )

    dpsi_dz = _component_derivative(
        psi,
        zc,
        create_graph=create_graph,
    )
    dpsi_dt = _component_derivative(
        psi,
        tc,
        create_graph=create_graph,
    )

    return psi, dpsi_dz, dpsi_dt


def derived_velocity(
    dpsi_dz: Tensor,
    dpsi_dt: Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Return closed-form velocity and Fisher tangent energy.

    In flat coordinates,

        V = - <Psi_z,Psi_t> / <Psi_z,Psi_z>.

    The denominator ||Psi_z||^2 is the local observability/tangent-energy
    diagnostic.
    """
    if dpsi_dz.shape != dpsi_dt.shape:
        raise ValueError(
            "dpsi_dz and dpsi_dt must have matching shapes."
        )

    numerator = (
        dpsi_dz
        * dpsi_dt
    ).sum(
        dim=1,
        keepdim=True,
    )

    tangent_energy = (
        dpsi_dz.square()
    ).sum(
        dim=1,
        keepdim=True,
    )

    velocity = (
        -numerator
        / (
            tangent_energy
            + float(eps)
        )
    )

    return velocity, tangent_energy


@dataclass(frozen=True)
class LocalDynamics:
    z: Tensor
    time: Tensor
    fisher_flat: Tensor
    dpsi_dz: Tensor
    dpsi_dt: Tensor
    velocity: Tensor
    residual: Tensor
    residual_speed: Tensor
    tangent_energy: Tensor

    @property
    def observability(self) -> Tensor:
        return self.tangent_energy

    @property
    def expression(self) -> Tensor:
        """Poisson mean field E = Psi^2 / 4, excluding library exposure."""
        return 0.25 * self.fisher_flat.square()


def estimate_local_dynamics(
    field: FisherFlatField | Callable[[Tensor, Tensor], Tensor],
    z: Tensor,
    time: Tensor,
    *,
    eps: float = 1e-8,
    create_graph: bool = False,
) -> LocalDynamics:
    """Differentiate Psi(z,t) and compute the closed-form velocity."""
    psi, dpsi_dz, dpsi_dt = field_derivatives(
        field,
        z,
        time,
        create_graph=create_graph,
    )

    velocity, tangent_energy = derived_velocity(
        dpsi_dz,
        dpsi_dt,
        eps=eps,
    )

    residual = (
        dpsi_dt
        + velocity
        * dpsi_dz
    )

    residual_speed = torch.linalg.vector_norm(
        residual,
        dim=1,
        keepdim=True,
    )

    return LocalDynamics(
        z=_as_col(z),
        time=_as_col(time),
        fisher_flat=psi,
        dpsi_dz=dpsi_dz,
        dpsi_dt=dpsi_dt,
        velocity=velocity,
        residual=residual,
        residual_speed=residual_speed,
        tangent_energy=tangent_energy,
    )


def velocity_field(
    field: FisherFlatField | Callable[[Tensor, Tensor], Tensor],
    z: Tensor,
    time: Tensor,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Evaluate V(z,t) from the fitted Stage-1 Fisher-flat field."""
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
    field: FisherFlatField | Callable[[Tensor, Tensor], Tensor],
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

    times = [
        torch.full_like(
            z,
            float(t0),
        )
    ]
    positions = [
        z.clone()
    ]
    velocities = [
        velocity_field(
            field,
            z,
            times[-1],
            eps=eps,
        )
    ]

    def v(
        z_state: Tensor,
        t_scalar: float,
    ) -> Tensor:
        tt = torch.full_like(
            z_state,
            float(t_scalar),
        )
        return velocity_field(
            field,
            z_state,
            tt,
            eps=eps,
        )

    for step in range(n_steps):
        t = float(t0) + step * dt

        k1 = v(z, t)
        k2 = v(
            z + 0.5 * dt * k1,
            t + 0.5 * dt,
        )
        k3 = v(
            z + 0.5 * dt * k2,
            t + 0.5 * dt,
        )
        k4 = v(
            z + dt * k3,
            t + dt,
        )

        z = (
            z
            + (dt / 6.0)
            * (
                k1
                + 2.0 * k2
                + 2.0 * k3
                + k4
            )
        )

        tt = torch.full_like(
            z,
            t + dt,
        )

        times.append(tt)
        positions.append(z.clone())
        velocities.append(
            velocity_field(
                field,
                z,
                tt,
                eps=eps,
            )
        )

    return FlowTrajectory(
        time=torch.stack(
            [x[:, 0] for x in times],
            dim=0,
        ),
        position=torch.stack(
            [x[:, 0] for x in positions],
            dim=0,
        ),
        velocity=torch.stack(
            [x[:, 0] for x in velocities],
            dim=0,
        ),
    )


__all__ = [
    "FisherFlatField",
    "LocalDynamics",
    "FlowTrajectory",
    "field_derivatives",
    "derived_velocity",
    "estimate_local_dynamics",
    "velocity_field",
    "integrate_flow_rk4",
]

