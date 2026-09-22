from __future__ import annotations

import numpy as np

from flow_stp.simulations.simulated_atlas_positive_control import (
    SimulationConfig,
    bend_tissue_coordinates,
    run_simulation,
    true_common_coordinate,
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


def test_true_common_coordinate_inverts_bending() -> None:
    s = np.array([-1.5, -0.25, 0.75, 1.5])
    z = np.array([0.1, 0.35, 0.65, 0.9])
    bend = np.array([0.0, 30.0, 75.0, 120.0])
    x, y = bend_tissue_coordinates(
        s,
        z,
        width=4.0,
        height=1.0,
        bend_angle_deg=bend,
    )
    recovered = true_common_coordinate(
        x,
        y,
        width=4.0,
        height=1.0,
        bend_angle_deg=bend,
    )
    np.testing.assert_allclose(recovered, z, atol=1e-12, rtol=1e-12)


def test_truth_satisfies_current_fisher_lagrangian_model() -> None:
    config = SimulationConfig(
        n_timepoints=4,
        n_cells_per_tissue=20,
        n_z=101,
        maximum_bend_deg=90.0,
        flow_speed=0.15,
        random_seed=7,
    )
    result = run_simulation(config)
    truth = result.truth

    np.testing.assert_allclose(
        truth.velocity,
        config.flow_speed,
        atol=1e-12,
        rtol=1e-12,
    )
    expected_flow = (
        truth.initial_position[None, :]
        + config.flow_speed * truth.time[:, None]
    )
    np.testing.assert_allclose(truth.flow, expected_flow)

    expected_rho = truth.developmental_speed * np.broadcast_to(
        truth.time[:, None],
        truth.rho.shape,
    )
    np.testing.assert_allclose(truth.rho, expected_rho)

    residual = truth.psi_t + truth.velocity[..., None] * truth.psi_z
    np.testing.assert_allclose(
        np.linalg.norm(residual, axis=2),
        truth.residual_speed,
    )
    assert np.all(truth.observability > 0.0)


def test_small_end_to_end_simulation() -> None:
    config = SimulationConfig(
        n_timepoints=3,
        n_cells_per_tissue=40,
        n_z=81,
        maximum_bend_deg=60.0,
        random_seed=7,
    )
    result = run_simulation(config)

    assert result.obs.shape[0] == 120
    assert result.counts.shape == (120, config.n_genes)
    assert result.expected_mean.shape == result.counts.shape
    assert np.all(result.counts.to_numpy() >= 0)
    assert np.all(np.isfinite(result.expected_mean.to_numpy()))
    assert np.all(
        (result.obs["z_true"] >= config.z_min)
        & (result.obs["z_true"] <= config.z_max)
    )

    recovered = true_common_coordinate(
        result.obs["x_extrinsic"].to_numpy(),
        result.obs["y_extrinsic"].to_numpy(),
        width=config.tissue_width,
        height=config.tissue_height,
        bend_angle_deg=result.obs["bend_angle_deg"].to_numpy(),
    )
    np.testing.assert_allclose(
        recovered,
        result.obs["z_true"].to_numpy(),
        atol=1e-12,
        rtol=1e-12,
    )
