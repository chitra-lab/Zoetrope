from __future__ import annotations

import numpy as np

from flow_stp.simulations.benchmark_dgp import (
    BenchmarkSimulationConfig,
    run_benchmark_simulation,
)


def _small_config(**kwargs):
    base = dict(
        tissue_times=(0.0, 0.5, 1.0),
        replicates_per_time=1,
        cells_per_tissue=30,
        latent_population_size=200,
        n_genes=12,
        n_a_grid=41,
        n_time_grid=41,
        n_spatial_programs=3,
        n_molecular_programs=3,
        n_off_axis_programs=2,
        random_seed=7,
    )
    base.update(kwargs)
    return BenchmarkSimulationConfig(**base)


def test_structured_regime_identifies_physical_velocity():
    result = run_benchmark_simulation(_small_config())
    np.testing.assert_allclose(
        result.truth.fisher_velocity,
        result.truth.physical_velocity,
        atol=5e-5,
        rtol=5e-5,
    )
    np.testing.assert_allclose(
        result.truth.fisher_speed,
        result.truth.material_speed,
        atol=5e-5,
        rtol=5e-5,
    )


def test_destructive_snapshots_never_reuse_particles():
    result = run_benchmark_simulation(_small_config())
    ids = result.cell_truth["particle_id"].to_numpy()
    assert len(ids) == len(np.unique(ids))
    assert "z_true" not in result.obs.columns
    assert "a_true" not in result.obs.columns
    assert "u_true" not in result.obs.columns


def test_parallel_violation_separates_physical_and_fisher_flow():
    result = run_benchmark_simulation(_small_config(parallel_strength=1.0))
    gap = np.max(
        np.abs(result.truth.fisher_velocity - result.truth.physical_velocity)
    )
    assert gap > 1e-4


def test_negative_binomial_observations_are_valid_counts():
    result = run_benchmark_simulation(_small_config(observation_model="negative_binomial"))
    matrix = result.counts.drop(columns=["cell_id"]).to_numpy()
    assert np.issubdtype(matrix.dtype, np.integer)
    assert np.all(matrix >= 0)
    assert np.all(np.isfinite(result.expected_mean.drop(columns=["cell_id"]).to_numpy()))


def test_simulation_is_reproducible():
    a = run_benchmark_simulation(_small_config())
    b = run_benchmark_simulation(_small_config())
    np.testing.assert_array_equal(
        a.counts.drop(columns=["cell_id"]).to_numpy(),
        b.counts.drop(columns=["cell_id"]).to_numpy(),
    )
    np.testing.assert_allclose(a.obs[["x", "y"]], b.obs[["x", "y"]])
