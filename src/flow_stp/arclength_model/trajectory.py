"""Integrate the registration flow F(a,t)=Z(a,t) and transcriptomic arclength."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass
class FlowResult:
    a: np.ndarray
    t: np.ndarray
    F: np.ndarray      # [nt,na]
    rho: np.ndarray    # [nt,na]


def _interp2(t_grid, z_grid, values):
    return RegularGridInterpolator(
        (np.asarray(t_grid, dtype=float), np.asarray(z_grid, dtype=float)),
        np.asarray(values, dtype=float),
        bounds_error=False,
        fill_value=None,
    )


def integrate_flow_and_arclength(
    V: np.ndarray,
    q: np.ndarray,
    z_grid,
    t_grid,
    *,
    a_grid: Optional[np.ndarray] = None,
    steps_per_interval: int = 8,
) -> FlowResult:
    """Integrate dZ/dt=V(Z,t), d rho/dt=q(Z,t) with vectorized RK4.

    Boundary conditions are already imposed in V.  Numerical states are clipped
    to the z-grid bounds after each RK4 step to protect against interpolation
    overshoot.
    """
    z = np.asarray(z_grid, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    V = np.asarray(V, dtype=float)
    q = np.asarray(q, dtype=float)
    if V.shape != (len(t), len(z)) or q.shape != V.shape:
        raise ValueError("V and q must both have shape [nt,nz]")
    if a_grid is None:
        a = z.copy()
    else:
        a = np.asarray(a_grid, dtype=float)
    if np.any(a < z[0]) or np.any(a > z[-1]):
        raise ValueError("a_grid must lie inside z_grid support")
    if steps_per_interval < 1:
        raise ValueError("steps_per_interval must be >= 1")

    vfun = _interp2(t, z, V)
    qfun = _interp2(t, z, q)
    Z = a.copy()
    R = np.zeros_like(a)
    F_out = [Z.copy()]
    R_out = [R.copy()]

    def rhs(tt, zz):
        pts = np.column_stack([np.full_like(zz, tt), zz])
        return vfun(pts), np.maximum(qfun(pts), 0.0)

    for k in range(len(t) - 1):
        h = (t[k + 1] - t[k]) / steps_per_interval
        current_t = t[k]
        for _ in range(steps_per_interval):
            k1z, k1r = rhs(current_t, Z)
            k2z, k2r = rhs(current_t + 0.5 * h, Z + 0.5 * h * k1z)
            k3z, k3r = rhs(current_t + 0.5 * h, Z + 0.5 * h * k2z)
            k4z, k4r = rhs(current_t + h, Z + h * k3z)
            Z = Z + (h / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)
            R = R + (h / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
            Z = np.clip(Z, z[0], z[-1])
            current_t += h
        F_out.append(Z.copy())
        R_out.append(R.copy())

    return FlowResult(a=a, t=t, F=np.stack(F_out), rho=np.stack(R_out))


def no_flow_arclength(q0: np.ndarray, t_grid) -> np.ndarray:
    """Cumulative trapezoidal integral of q0(z,t) at fixed z."""
    q0 = np.asarray(q0, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    if q0.shape[0] != len(t):
        raise ValueError("q0 first axis must match t_grid")
    rho = np.zeros_like(q0)
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        rho[k] = rho[k - 1] + 0.5 * dt * (q0[k - 1] + q0[k])
    return rho


def assign_rho_to_observations(
    flow: FlowResult,
    z_obs,
    t_obs,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert monotone F_t(a)=z for each observation and return (a,rho).

    Linear interpolation is used in t between stored flow slices and in a along
    each monotone flow map.  A ValueError is raised if a flow slice loses
    monotonicity, which is an important registration diagnostic.
    """
    z_obs = np.asarray(z_obs, dtype=float).reshape(-1)
    t_obs = np.asarray(t_obs, dtype=float).reshape(-1)
    if len(z_obs) != len(t_obs):
        raise ValueError("z_obs and t_obs lengths differ")

    a_out = np.empty_like(z_obs)
    rho_out = np.empty_like(z_obs)
    for n, (zv, tv) in enumerate(zip(z_obs, t_obs)):
        if tv <= flow.t[0]:
            lo = hi = 0
            wt = 0.0
        elif tv >= flow.t[-1]:
            lo = hi = len(flow.t) - 1
            wt = 0.0
        else:
            hi = int(np.searchsorted(flow.t, tv))
            lo = hi - 1
            wt = (tv - flow.t[lo]) / (flow.t[hi] - flow.t[lo])

        F_t = (1 - wt) * flow.F[lo] + wt * flow.F[hi]
        rho_t = (1 - wt) * flow.rho[lo] + wt * flow.rho[hi]
        if np.any(np.diff(F_t) <= 0):
            raise ValueError("flow map is not strictly monotone; inspect V/step size")
        zc = np.clip(zv, F_t[0], F_t[-1])
        a_val = np.interp(zc, F_t, flow.a)
        rho_val = np.interp(a_val, flow.a, rho_t)
        a_out[n] = a_val
        rho_out[n] = rho_val
    return a_out, rho_out
