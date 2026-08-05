from __future__ import annotations

import numpy as np

from flow_stp.simulations.simulated_atlas_positive_control import (
    SimulationConfig,
    bend_tissue_coordinates,
    run_simulation,
    stopping_velocity,
)


def test_geometry_is_identity_at_zero_bend() -> None:
    s = np.array([-2.0, 0.0, 2.0])
    z = np.array([0.0, 0.5, 1.0])
    x, y = bend_tissue_coordinates(
        s,
        z,
        width=4.0,
        height=1.0,
        bend_angle_deg=0.0,
    )
    np.testing.assert_allclose(x, s)
    np.testing.assert_allclose(y, z)


def test_stopping_velocity_is_exactly_zero_after_stop() -> None:
    z = np.linspace(0.0, 1.0, 101)
    velocity = stopping_velocity(
        z,
        peak_velocity=2.8,
        stopping_point=0.75,
        deceleration_width=0.06,
    )
    assert np.all(velocity[z >= 0.75] == 0.0)
    assert np.allclose(velocity[z < 0.69], 2.8)


def test_small_end_to_end_simulation() -> None:
    config = SimulationConfig(
        n_timepoints=3,
        n_cells_per_tissue=40,
        n_z=81,
        maximum_bend_deg=60.0,
        random_seed=7,
        birth_time_end=0.45,
    )
    result = run_simulation(config)

    assert result.obs.shape[0] == 120
    assert result.counts.shape == (120, 6 * config.genes_per_program)
    assert result.expected_mean.shape == result.counts.shape
    assert result.decoder.baseline_anchor_error == 0.0
    assert np.all(result.counts.to_numpy() >= 0)
    assert np.all(np.isfinite(result.expected_mean.to_numpy()))
    assert np.all((result.obs["z_intrinsic"] >= 0.0) & (result.obs["z_intrinsic"] <= 1.0))

    edges = np.linspace(0.5, 1.0, config.n_timepoints + 1)
    expected_stops = 0.5 * (edges[:-1] + edges[1:])
    np.testing.assert_allclose(result.config.stopping_points, expected_stops)
    np.testing.assert_allclose(result.decoder.rho_decoder[0], 0.0)
