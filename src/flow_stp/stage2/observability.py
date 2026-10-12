"""Post-hoc numerical observability audit for Fisher-projected trajectories.

Does not alter the fitted expression field, Fisher velocity, or RK4 integration.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import torch


@dataclass(frozen=True)
class FlowObservabilityReport:
    n_times: int
    n_trajectories: int
    threshold: float
    eps: float
    weak_node_mask: np.ndarray
    per_time_weak_fraction: np.ndarray
    weak_node_fraction: float
    weak_trajectory_fraction: float
    tangent_energy_min: float
    tangent_energy_median: float
    regularizer_dominance_max: float

    def summary(self) -> dict[str, float | int]:
        """JSON-serializable summary; node-level masks are stored separately."""
        return {
            "n_times": self.n_times,
            "n_trajectories": self.n_trajectories,
            "threshold": self.threshold,
            "eps": self.eps,
            "weak_node_fraction": self.weak_node_fraction,
            "weak_trajectory_fraction": self.weak_trajectory_fraction,
            "tangent_energy_min": self.tangent_energy_min,
            "tangent_energy_median": self.tangent_energy_median,
            "regularizer_dominance_max": self.regularizer_dominance_max,
        }


def audit_flow_observability(
    field: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    trajectory,
    *,
    eps: float = 1e-8,
    min_tangent_energy: float | None = None,
    chunk_size: int = 1024,
) -> FlowObservabilityReport:
    """Audit Fisher tangent energy at stored RK4 nodes (not its substeps).

    Default threshold marks nodes at which the epsilon term contributes
    at least 1% of the denominator. A scientific observability requirement
    should supply its own, dataset-dependent minimum tangent energy.
    """
    from .dynamics import field_derivatives
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    threshold = max(99. * eps, 1e-12) if min_tangent_energy is None else float(min_tangent_energy)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("min_tangent_energy must be finite and positive.")
    pos, times = trajectory.position, trajectory.time
    if not torch.is_tensor(pos) or not torch.is_tensor(times):
        raise TypeError("trajectory.position/time must be torch tensors.")
    if pos.ndim != 2 or pos.shape != times.shape or pos.numel() < 1:
        raise ValueError("trajectory position and time must have equal nonempty [T,N] shapes.")
    flat_z, flat_time = pos.reshape(-1), times.reshape(-1)
    pieces = []
    for k in range(0, flat_z.numel(), chunk_size):
        _, z_deriv, _ = field_derivatives(
            field, flat_z[k:k+chunk_size], flat_time[k:k+chunk_size],
            create_graph=False,
        )
        pieces.append(z_deriv.square().sum(dim=1).detach().cpu().numpy())
    energy = np.concatenate(pieces).reshape(pos.shape)
    dominance = np.divide(
        eps, energy + eps, out=np.full_like(energy, np.nan),
        where=(energy+eps)>0,
    )
    weak = ~np.isfinite(energy) | (energy < threshold)
    return FlowObservabilityReport(
        n_times=int(pos.shape[0]),
        n_trajectories=int(pos.shape[1]),
        threshold=threshold,
        eps=float(eps),
        weak_node_mask=weak,
        per_time_weak_fraction=np.mean(weak, axis=1),
        weak_node_fraction=float(np.mean(weak)),
        weak_trajectory_fraction=float(np.mean(weak.any(axis=0))),
        tangent_energy_min=float(np.nanmin(energy)),
        tangent_energy_median=float(np.nanmedian(energy)),
        regularizer_dominance_max=float(np.nanmax(dominance)),
    )


__all__ = ["FlowObservabilityReport", "audit_flow_observability"]
