"""Intrinsic rectangular tissue geometry and block-arc bending."""

from __future__ import annotations

import numpy as np


def bend_tissue_coordinates(
    s: np.ndarray,
    z: np.ndarray,
    *,
    width: float,
    height: float = 1.0,
    bend_angle_deg: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map intrinsic material coordinates (s,z) to extrinsic (x,y)."""
    if width <= 0.0 or height <= 0.0:
        raise ValueError("width and height must be positive.")

    s = np.asarray(s, dtype=float)
    z = np.asarray(z, dtype=float)
    bend_angle_deg = np.asarray(bend_angle_deg, dtype=float)
    s, z, bend_angle_deg = np.broadcast_arrays(s, z, bend_angle_deg)

    if np.any((z < 0.0) | (z > 1.0)):
        raise ValueError("z must lie in [0,1].")

    theta = np.deg2rad(bend_angle_deg)
    straight = np.abs(theta) < 1e-12

    x = np.empty_like(s, dtype=float)
    y = np.empty_like(s, dtype=float)
    x[straight] = s[straight]
    y[straight] = z[straight] * height

    bent = ~straight
    if np.any(bent):
        radius_midline = width / theta[bent]
        if np.any(radius_midline <= 0.5 * height):
            raise ValueError("At least one bend is too tight for the tissue thickness.")

        phi = s[bent] / radius_midline
        radius = radius_midline + (z[bent] - 0.5) * height
        x[bent] = radius * np.sin(phi)
        y[bent] = radius * np.cos(phi) - radius_midline + 0.5 * height

    return x, y


def true_common_coordinate(
    x: np.ndarray,
    y: np.ndarray,
    *,
    width: float,
    height: float = 1.0,
    bend_angle_deg: np.ndarray | float,
) -> np.ndarray:
    """Recover the exact intrinsic z coordinate from bent tissue coordinates."""
    if width <= 0.0 or height <= 0.0:
        raise ValueError("width and height must be positive.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    bend_angle_deg = np.asarray(bend_angle_deg, dtype=float)
    x, y, bend_angle_deg = np.broadcast_arrays(x, y, bend_angle_deg)

    theta = np.deg2rad(bend_angle_deg)
    straight = np.abs(theta) < 1e-12
    z = np.empty_like(x, dtype=float)
    z[straight] = y[straight] / height

    bent = ~straight
    if np.any(bent):
        radius_midline = width / theta[bent]
        radial_distance = np.sqrt(
            x[bent] ** 2
            + (y[bent] + radius_midline - 0.5 * height) ** 2
        )
        z[bent] = 0.5 + (radial_distance - radius_midline) / height

    return z


def make_tissue_grid(
    *,
    width: float,
    height: float = 1.0,
    bend_angle_deg: float,
    n_tangential: int = 241,
    n_depth: int = 81,
) -> dict[str, np.ndarray]:
    """Create a regular intrinsic grid and its bent extrinsic coordinates."""
    if n_tangential < 2 or n_depth < 2:
        raise ValueError("Grid dimensions must be at least 2.")
    s_axis = np.linspace(-0.5 * width, 0.5 * width, n_tangential)
    z_axis = np.linspace(0.0, 1.0, n_depth)
    s_grid, z_grid = np.meshgrid(s_axis, z_axis)
    x_grid, y_grid = bend_tissue_coordinates(
        s_grid,
        z_grid,
        width=width,
        height=height,
        bend_angle_deg=bend_angle_deg,
    )
    return {
        "s": s_grid,
        "z": z_grid,
        "x": x_grid,
        "y": y_grid,
        "bend_angle_deg": np.asarray(float(bend_angle_deg)),
    }


def sample_intrinsic_coordinates(
    n_cells: int,
    *,
    width: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample cells uniformly in the intrinsic rectangular material domain."""
    if n_cells <= 0:
        raise ValueError("n_cells must be positive.")
    if width <= 0.0:
        raise ValueError("width must be positive.")
    s = rng.uniform(-0.5 * width, 0.5 * width, size=n_cells)
    z = rng.uniform(0.0, 1.0, size=n_cells)
    return s, z
