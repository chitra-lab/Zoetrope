"""Optional trajectory-constrained Stage-2 velocity refinement.

Fit a smooth shared ODE to trajectories already inferred from the Fisher flow.
This does NOT change the Stage-1 expression model or the Fisher projection.

No-flux is imposed on a *common bounded material coordinate* z, not on an
arbitrary time-dependent learned coordinate. The caller must first provide
trajectories in a consistent coordinate with physically meaningful endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import torch

from .dynamics import FlowTrajectory

Tensor = torch.Tensor


def _basis(x: Tensor, degree: int) -> Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.stack(
        [math.comb(degree, j) * x.pow(j) * (1 - x).pow(degree - j)
         for j in range(degree + 1)], dim=-1,
    )


def _second_difference_penalty(coefficients: Tensor) -> Tensor:
    loss = coefficients.new_zeros(())
    if coefficients.shape[0] >= 3:
        loss = loss + (
            coefficients[2:] - 2 * coefficients[1:-1] + coefficients[:-2]
        ).square().mean()
    if coefficients.shape[1] >= 3:
        loss = loss + (
            coefficients[:, 2:] - 2 * coefficients[:, 1:-1] + coefficients[:, :-2]
        ).square().mean()
    return loss


@dataclass(frozen=True)
class RefinementConfig:
    """Fixed-complexity fit; defaults match the no-flux synthetic experiment.

    The boundary values refer to the *material coordinate*, not Cartesian XY.
    smoothness penalizes second differences of Bernstein coefficients.
    """

    degree_z: int = 3
    degree_t: int = 3
    smoothness: float = 1e-5
    steps_per_interval: int = 4
    adam_steps: int = 240
    lbfgs_steps: int = 35
    learning_rate: float = 0.035
    no_flux: bool = True
    z_bounds: tuple[float, float] = (0.0, 1.0)
    ridge_init: float = 1e-3

    def __post_init__(self) -> None:
        if self.degree_z < 0 or self.degree_t < 0:
            raise ValueError("Polynomial degrees must be nonnegative")
        if self.smoothness < 0 or self.ridge_init <= 0:
            raise ValueError("smoothness must be >= 0 and ridge_init > 0")
        if self.steps_per_interval < 1 or self.adam_steps < 0 or self.lbfgs_steps < 0:
            raise ValueError("Invalid integration/optimizer steps")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not (math.isfinite(self.z_bounds[0]) and math.isfinite(self.z_bounds[1])
                and self.z_bounds[0] < self.z_bounds[1]):
            raise ValueError("z_bounds must be finite and increasing")


@dataclass(frozen=True)
class RefinedVelocityField:
    """Fitted velocity in material z-units per unit of the supplied time axis."""

    coefficients: Tensor  # [degree_z+1, degree_t+1]
    z_bounds: tuple[float, float]
    t_bounds: tuple[float, float]
    no_flux: bool

    def velocity(self, z: Tensor, time: Tensor | float) -> Tensor:
        """Evaluate dz/dt. Supports broadcastable z, time, returns z's shape."""
        zz = torch.as_tensor(z, dtype=self.coefficients.dtype,
                             device=self.coefficients.device)
        tt = torch.as_tensor(time, dtype=zz.dtype, device=zz.device)
        zz, tt = torch.broadcast_tensors(zz, tt)
        zl, zr = self.z_bounds
        tl, tr = self.t_bounds
        s = ((zz - zl) / (zr - zl)).clamp(0., 1.)
        tau = (tt - tl) / (tr - tl)
        bz = _basis(s, self.coefficients.shape[0] - 1)
        bt = _basis(tau, self.coefficients.shape[1] - 1)
        v = torch.einsum('...i,ij,...j->...', bz, self.coefficients, bt)
        if self.no_flux:
            v = v * s * (1.0 - s)
        return v

    def integrate(self, initial_z: Tensor, times: Tensor,
                  *, steps_per_interval: int = 4) -> FlowTrajectory:
        """RK4 flow, returning [T,N] time, position and material velocity."""
        if steps_per_interval < 1:
            raise ValueError("steps_per_interval must be positive")
        ts = torch.as_tensor(times, dtype=self.coefficients.dtype,
                             device=self.coefficients.device).reshape(-1)
        if len(ts) < 2 or not torch.isfinite(ts).all() or not torch.all(ts[1:] > ts[:-1]):
            raise ValueError("times must be finite, increasing, with at least 2 entries")
        if ts[0] < self.t_bounds[0] - 1e-9 or ts[-1] > self.t_bounds[1] + 1e-9:
            raise ValueError("Integration times outside fitted interval")
        z = torch.as_tensor(initial_z, dtype=ts.dtype, device=ts.device).reshape(-1)
        if not torch.isfinite(z).all():
            raise ValueError("initial_z must be finite")
        positions = [z]
        for left, right in zip(ts[:-1], ts[1:]):
            dt = (right - left) / steps_per_interval
            for j in range(steps_per_interval):
                t = left + dt * j
                k1 = self.velocity(z, t)
                k2 = self.velocity(z + 0.5 * dt * k1, t + 0.5 * dt)
                k3 = self.velocity(z + 0.5 * dt * k2, t + 0.5 * dt)
                k4 = self.velocity(z + dt * k3, t + dt)
                z = z + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
            positions.append(z)
        pos = torch.stack(positions)
        time_grid = ts[:, None].expand_as(pos)
        return FlowTrajectory(time=time_grid, position=pos,
                              velocity=self.velocity(pos, time_grid))


@dataclass(frozen=True)
class RefinementResult:
    field: RefinedVelocityField
    train_position_rmse: float
    roughness: float
    observed_count: int


def fit_trajectory_velocity(
    positions: Tensor | np.ndarray,
    times: Tensor | np.ndarray,
    *,
    mask: Tensor | np.ndarray | None = None,
    config: RefinementConfig | None = None,
) -> RefinementResult:
    """Fit an ODE to a [T,N] matrix of Fisher-derived material trajectories.

    Input positions and times must share ONE fixed anatomical/material gauge.
    Missing entries are indicated by mask=False and/or nonfinite values.
    Initial positions at the first time must be observed for every track.
    The fit uses positions only; synthetic true velocities are not needed.
    """
    cfg = config or RefinementConfig()
    ts_np = np.asarray(times, dtype=np.float64).reshape(-1)
    raw = np.asarray(positions, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != len(ts_np) or raw.shape[1] < 1:
        raise ValueError("positions must have shape [len(times), n_tracks]")
    if len(ts_np) < 2 or not np.all(np.isfinite(ts_np)) or not np.all(np.diff(ts_np) > 0):
        raise ValueError("times must be finite and strictly increasing")
    good = np.isfinite(raw)
    if mask is not None:
        if np.asarray(mask).shape != raw.shape:
            raise ValueError("mask must match positions shape")
        good &= np.asarray(mask, dtype=bool)
    if not np.all(good[0]):
        raise ValueError("Every track requires an observed initial position")
    if good.sum() <= raw.shape[1]:
        raise ValueError("At least some observations after the initial time are required")
    if not np.all((raw[good] >= cfg.z_bounds[0] - 1e-8) &
                  (raw[good] <= cfg.z_bounds[1] + 1e-8)):
        raise ValueError("observed material positions outside z_bounds")

    # Snapshot velocity is for initialization only; the loss fits integrated
    # positions, not the potentially noisy instantaneous derivatives.
    filled = np.column_stack([
        np.interp(ts_np, ts_np[good[:, k]], raw[good[:, k], k])
        for k in range(raw.shape[1])
    ])
    deriv = np.gradient(filled, ts_np, axis=0, edge_order=2 if len(ts_np) > 2 else 1)
    zscaled = np.clip((filled - cfg.z_bounds[0]) /
                      (cfg.z_bounds[1] - cfg.z_bounds[0]), 0.0, 1.0)
    tscaled = (ts_np - ts_np[0]) / (ts_np[-1] - ts_np[0])
    zb = np.stack([math.comb(cfg.degree_z, j) * zscaled**j *
                   (1-zscaled)**(cfg.degree_z-j)
                   for j in range(cfg.degree_z+1)], axis=-1)
    tb = np.stack([math.comb(cfg.degree_t, j) * tscaled[:,None]**j *
                   (1-tscaled[:,None])**(cfg.degree_t-j)
                   for j in range(cfg.degree_t+1)], axis=-1)
    design = (zb[..., :, None] * tb[..., None, :]).reshape(
        -1, (cfg.degree_z+1)*(cfg.degree_t+1))
    if cfg.no_flux:
        design *= (zscaled*(1-zscaled)).reshape(-1,1)
    sel = good.reshape(-1)
    x, y = design[sel], deriv.reshape(-1)[sel]
    theta0 = np.linalg.solve(x.T @ x + cfg.ridge_init*np.eye(x.shape[1]), x.T @ y)
    coeff = torch.nn.Parameter(torch.as_tensor(theta0.reshape(cfg.degree_z+1, cfg.degree_t+1),
                                                dtype=torch.float64).clone())
    obs = torch.as_tensor(np.where(good, raw, 0.0), dtype=torch.float64)
    obs_mask = torch.as_tensor(good, dtype=torch.bool)
    t = torch.as_tensor(ts_np, dtype=torch.float64)
    init = obs[0]

    def objective() -> tuple[Tensor, Tensor, Tensor]:
        field = RefinedVelocityField(coeff, cfg.z_bounds,
                                     (float(ts_np[0]), float(ts_np[-1])), cfg.no_flux)
        pred = field.integrate(init, t, steps_per_interval=cfg.steps_per_interval).position
        fit = (pred[obs_mask] - obs[obs_mask]).square().mean()
        penalty = _second_difference_penalty(coeff)
        return fit + cfg.smoothness * penalty, fit, penalty

    opt = torch.optim.Adam([coeff], lr=cfg.learning_rate)
    for _ in range(cfg.adam_steps):
        opt.zero_grad()
        loss, _, _ = objective()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite trajectory-refinement loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([coeff], 10.)
        opt.step()
    if cfg.lbfgs_steps:
        opt_lbfgs = torch.optim.LBFGS([coeff], lr=0.7,
                                       max_iter=cfg.lbfgs_steps,
                                       tolerance_grad=1e-10, line_search_fn='strong_wolfe')
        def closure() -> Tensor:
            opt_lbfgs.zero_grad()
            loss, _, _ = objective()
            loss.backward()
            return loss
        opt_lbfgs.step(closure)
    with torch.no_grad():
        _, fit, roughness = objective()
        final_coeff = coeff.detach().clone()
    return RefinementResult(
        field=RefinedVelocityField(final_coeff, cfg.z_bounds,
                                   (float(ts_np[0]), float(ts_np[-1])), cfg.no_flux),
        train_position_rmse=float(torch.sqrt(fit)),
        roughness=float(roughness), observed_count=int(good.sum()),
    )


def physical_velocity(
    field: RefinedVelocityField,
    u: Tensor,
    z: Tensor,
    time: Tensor | float,
    *,
    dG_dz: Callable[[Tensor, Tensor, Tensor], Tensor],
    dG_dt: Callable[[Tensor, Tensor, Tensor], Tensor] | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Push a material velocity into physical XY at G_t(u,z).

    Returns (migration velocity, total velocity). Total velocity requires dG_dt.
    The caller supplies an identified geometry map and its derivatives; they
    are *not* inferred from the anatomical-coordinate network by this module.
    Each derivative callback returns [...,2] in physical distance units.
    """
    v = field.velocity(z, time)
    tangent = dG_dz(u, z, time)
    if tangent.shape != v.shape + (2,):
        raise ValueError("dG_dz must return shape z.shape + (2,)")
    migration = tangent * v[..., None]
    if dG_dt is None:
        return migration, None
    deformation = dG_dt(u, z, time)
    if deformation.shape != migration.shape:
        raise ValueError("dG_dt must return shape z.shape + (2,)")
    return migration, migration + deformation


__all__ = ["RefinementConfig", "RefinedVelocityField", "RefinementResult",
           "fit_trajectory_velocity", "physical_velocity"]
