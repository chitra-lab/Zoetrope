"""Fisher geometry and the derived local expression-persistence flow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .model import Tensor, _as_col


class ExpressionField(Protocol):
    def __call__(self, z: Tensor, time: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class FisherGeometryResult:
    z: Tensor
    time: Tensor
    log_lambda: Tensor
    lambda_: Tensor
    fisher_embedding: Tensor
    dz: Tensor
    dt: Tensor
    velocity: Tensor
    residual: Tensor
    speed: Tensor
    tangent_energy: Tensor

    @property
    def observability(self) -> Tensor:
        """Local Fisher information for displacement along the anatomical axis."""
        return self.tangent_energy

    @property
    def spatial_speed(self) -> Tensor:
        r"""Fisher arclength rate per unit anatomical displacement.

        This is ``||Psi_z||``. Integrating it over ``z`` gives the signed
        spatial-progression coordinate along the expression manifold.
        """
        return torch.sqrt(self.tangent_energy.clamp_min(0.0))

    @property
    def transport_speed(self) -> Tensor:
        r"""Instantaneous Fisher speed attributable to axial transport.

        This is ``|V| ||Psi_z||`` and is Fisher-orthogonal to the residual
        molecular speed ``speed = ||Psi_t + V Psi_z||`` under the selected
        minimum-residual velocity.
        """
        return self.velocity.abs() * self.spatial_speed


def fisher_embedding_from_log_mean(log_lambda: Tensor) -> Tensor:
    return 2.0 * torch.exp(0.5 * log_lambda)


def _component_derivative(values: Tensor, inputs: Tensor, *, create_graph: bool) -> Tensor:
    cols = []
    for g in range(values.shape[1]):
        grad = torch.autograd.grad(
            values[:, g].sum(),
            inputs,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        cols.append(grad)
    return torch.cat(cols, dim=1)


def field_derivatives(
    field: ExpressionField,
    z: Tensor,
    time: Tensor,
    *,
    create_graph: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return ``(log_lambda, Psi, Psi_z, Psi_t)`` in Poisson Fisher coordinates."""
    derivative_method = getattr(field, "log_mean_derivatives", None)
    if derivative_method is not None:
        zc = _as_col(z)
        tc = _as_col(time)
        log_lambda, eta_z, eta_t = derivative_method(zc, tc)
        sqrt_lambda = torch.exp(0.5 * log_lambda)
        psi = 2.0 * sqrt_lambda
        psi_z = sqrt_lambda * eta_z
        psi_t = sqrt_lambda * eta_t
        return log_lambda, psi, psi_z, psi_t

    zc = _as_col(z).detach().clone().requires_grad_(True)
    tc = _as_col(time).detach().clone().requires_grad_(True)
    if zc.shape[0] != tc.shape[0]:
        raise ValueError("z and time must have matching rows")
    log_lambda = field(zc, tc)
    psi = fisher_embedding_from_log_mean(log_lambda)
    psi_z = _component_derivative(psi, zc, create_graph=create_graph)
    psi_t = _component_derivative(psi, tc, create_graph=create_graph)
    return log_lambda, psi, psi_z, psi_t


def derived_velocity(
    psi_z: Tensor,
    psi_t: Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    r"""Return the local minimum-expression-change velocity.

    ``V = argmin_v ||Psi_t + v Psi_z||^2`` and therefore
    ``V = -(Psi_z^T Psi_t)/(Psi_z^T Psi_z)`` where identifiable. ``eps`` is
    numerical stabilization only; ``Psi_z^T Psi_z`` is returned so weakly
    identifiable regions remain explicit.
    """
    if psi_z.shape != psi_t.shape:
        raise ValueError("psi_z and psi_t must have matching shapes")
    numerator = (psi_z * psi_t).sum(dim=1, keepdim=True)
    denominator = psi_z.square().sum(dim=1, keepdim=True)
    velocity = -numerator / (denominator + float(eps))
    return velocity, denominator


def fisher_geometry(
    field: ExpressionField,
    z: Tensor,
    time: Tensor,
    *,
    eps: float = 1e-8,
    create_graph: bool = False,
) -> FisherGeometryResult:
    log_lambda, psi, psi_z, psi_t = field_derivatives(
        field, z, time, create_graph=create_graph
    )
    velocity, tangent_energy = derived_velocity(psi_z, psi_t, eps=eps)
    residual = psi_t + velocity * psi_z
    speed = torch.linalg.vector_norm(residual, dim=1, keepdim=True)
    return FisherGeometryResult(
        z=_as_col(z),
        time=_as_col(time),
        log_lambda=log_lambda,
        lambda_=torch.exp(log_lambda),
        fisher_embedding=psi,
        dz=psi_z,
        dt=psi_t,
        velocity=velocity,
        residual=residual,
        speed=speed,
        tangent_energy=tangent_energy,
    )


def orthogonality_error(result: FisherGeometryResult) -> Tensor:
    inner = (result.residual * result.dz).sum(dim=1)
    denom = (
        torch.linalg.vector_norm(result.residual, dim=1)
        * torch.linalg.vector_norm(result.dz, dim=1)
    ).clamp_min(1e-12)
    return (inner.abs() / denom).mean()
