"""Migration velocity and developmental source functions."""

from __future__ import annotations

import numpy as np

from .config import SimulationConfig


def smootherstep(u: np.ndarray | float) -> np.ndarray:
    """Quintic smootherstep with zero first and second endpoint derivatives."""
    u = np.asarray(u, dtype=float)
    x = np.clip(u, 0.0, 1.0)
    return 6.0 * x**5 - 15.0 * x**4 + 10.0 * x**3


def smootherstep_derivative(u: np.ndarray | float) -> np.ndarray:
    """Derivative of :func:`smootherstep` with respect to its argument."""
    u = np.asarray(u, dtype=float)
    x = np.clip(u, 0.0, 1.0)
    derivative = 30.0 * x**2 * (x - 1.0) ** 2
    return np.where((u > 0.0) & (u < 1.0), derivative, 0.0)


def stopping_velocity(
    z: np.ndarray,
    *,
    peak_velocity: float,
    stopping_point: float,
    deceleration_width: float,
) -> np.ndarray:
    """Constant migration followed by smooth deceleration to exact zero."""
    if peak_velocity < 0.0:
        raise ValueError("peak_velocity must be nonnegative.")
    if not 0.0 < stopping_point <= 1.0:
        raise ValueError("stopping_point must lie in (0,1].")
    if not 0.0 < deceleration_width < stopping_point:
        raise ValueError("deceleration_width must lie in (0, stopping_point).")

    z = np.asarray(z, dtype=float)
    if np.any((z < 0.0) | (z > 1.0)):
        raise ValueError("z must lie in [0,1].")

    deceleration_start = stopping_point - deceleration_width
    u = (z - deceleration_start) / deceleration_width
    remaining = 1.0 - smootherstep(u)
    return np.where(
        z < deceleration_start,
        peak_velocity,
        np.where(z < stopping_point, peak_velocity * remaining, 0.0),
    )


def radial_glia_rho(t: np.ndarray | float, config: SimulationConfig) -> np.ndarray:
    """Slowly increasing radial-glia developmental coordinate."""
    u = (np.asarray(t, dtype=float) - config.rg_rho_rise_start) / (
        config.rg_rho_rise_duration
    )
    return config.rg_rho_start + (
        config.rg_rho_ceiling - config.rg_rho_start
    ) * smootherstep(u)


def radial_glia_q(t: np.ndarray | float, config: SimulationConfig) -> np.ndarray:
    """Radial-glia developmental source ``q=d rho/dt``."""
    u = (np.asarray(t, dtype=float) - config.rg_rho_rise_start) / (
        config.rg_rho_rise_duration
    )
    return (
        (config.rg_rho_ceiling - config.rg_rho_start)
        * smootherstep_derivative(u)
        / config.rg_rho_rise_duration
    )


def neuron_birth_rho(birth_time: float, config: SimulationConfig) -> float:
    """Developmental state inherited from radial glia at neuronal birth."""
    return float(np.asarray(radial_glia_rho(birth_time, config)))


def neuron_rho(
    t: np.ndarray | float,
    *,
    birth_time: float,
    config: SimulationConfig,
    before_birth_value: float = np.nan,
) -> np.ndarray:
    """Cohort maturation trajectory toward the shared neuronal endpoint."""
    t = np.asarray(t, dtype=float)
    inherited = neuron_birth_rho(birth_time, config)
    start = birth_time + config.neuron_maturation_delay
    u = (t - start) / config.neuron_maturation_duration
    rho = inherited + (config.mature_neuron_rho - inherited) * smootherstep(u)
    return np.where(t >= birth_time, rho, before_birth_value)


def neuron_q(
    t: np.ndarray | float,
    *,
    birth_time: float,
    config: SimulationConfig,
) -> np.ndarray:
    """Neuronal developmental source independent of velocity and stop position."""
    t = np.asarray(t, dtype=float)
    inherited = neuron_birth_rho(birth_time, config)
    start = birth_time + config.neuron_maturation_delay
    u = (t - start) / config.neuron_maturation_duration
    q = (
        (config.mature_neuron_rho - inherited)
        * smootherstep_derivative(u)
        / config.neuron_maturation_duration
    )
    return np.where(t >= birth_time, q, 0.0)
