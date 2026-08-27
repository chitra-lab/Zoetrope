"""Fisher geometry and derived registration flow.

For independent Poisson means lambda(z,t), the square-root embedding

    Y(z,t) = 2 * sqrt(lambda(z,t))

turns the Fisher metric into the Euclidean metric. We use automatic
differentiation to obtain Y_z and Y_t and derive the least-squares anatomical
registration velocity

    V = -(Y_z^T Y_t) / (Y_z^T Y_z + eps).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import LogMeanExpressionField, Tensor, _as_col


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


def fisher_embedding_from_log_mean(log_lambda: Tensor) -> Tensor:
    """Return the exact Poisson Fisher square-root coordinates ``2 sqrt(lambda)``."""
    return 2.0 * torch.exp(0.5 * log_lambda)


def _component_derivative(values: Tensor, inputs: Tensor, *, create_graph: bool) -> Tensor:
    """Differentiate each output feature with respect to one scalar input.

    ``values`` is [N,G] and ``inputs`` is [N,1]. The result is [N,G].
    Samples are evaluated row-wise by the expression network, so the gradient of
    the summed g-th output recovers the per-row derivative for feature g.
    """
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
    field: LogMeanExpressionField,
    z: Tensor,
    time: Tensor,
    *,
    create_graph: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Evaluate Fisher embedding and its z/time derivatives.

    Returns ``(log_lambda, Y, Y_z, Y_t)``.
    """
    zc = _as_col(z).detach().clone().requires_grad_(True)
    tc = _as_col(time).detach().clone().requires_grad_(True)
    if zc.shape[0] != tc.shape[0]:
        raise ValueError("z and time must have matching rows")

    log_lambda = field(zc, tc)
    Y = fisher_embedding_from_log_mean(log_lambda)
    Y_z = _component_derivative(Y, zc, create_graph=create_graph)
    Y_t = _component_derivative(Y, tc, create_graph=create_graph)
    return log_lambda, Y, Y_z, Y_t


def derived_velocity(Y_z: Tensor, Y_t: Tensor, *, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
    """Return Fisher-optimal scalar registration velocity and tangent energy."""
    if Y_z.shape != Y_t.shape:
        raise ValueError("Y_z and Y_t must have matching shapes")
    numerator = (Y_z * Y_t).sum(dim=1, keepdim=True)
    denominator = Y_z.square().sum(dim=1, keepdim=True)
    velocity = -numerator / (denominator + float(eps))
    return velocity, denominator


def fisher_geometry(
    field: LogMeanExpressionField,
    z: Tensor,
    time: Tensor,
    *,
    eps: float = 1e-8,
    create_graph: bool = False,
) -> FisherGeometryResult:
    """Evaluate all local Fisher-Lagrangian geometric quantities."""
    log_lambda, Y, Y_z, Y_t = field_derivatives(
        field, z, time, create_graph=create_graph
    )
    velocity, tangent_energy = derived_velocity(Y_z, Y_t, eps=eps)
    residual = Y_t + velocity * Y_z
    speed = torch.linalg.vector_norm(residual, dim=1, keepdim=True)
    return FisherGeometryResult(
        z=_as_col(z),
        time=_as_col(time),
        log_lambda=log_lambda,
        lambda_=torch.exp(log_lambda),
        fisher_embedding=Y,
        dz=Y_z,
        dt=Y_t,
        velocity=velocity,
        residual=residual,
        speed=speed,
        tangent_energy=tangent_energy,
    )


def orthogonality_error(result: FisherGeometryResult) -> Tensor:
    """Mean absolute cosine-like residual/tangent inner product diagnostic."""
    inner = (result.residual * result.dz).sum(dim=1)
    denom = (
        torch.linalg.vector_norm(result.residual, dim=1)
        * torch.linalg.vector_norm(result.dz, dim=1)
    ).clamp_min(1e-12)
    return (inner.abs() / denom).mean()
