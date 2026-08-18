import numpy as np

from flow_stp.simulations.arclength_identifiability import (
    ArclengthSyntheticConfig,
    inverse_logistic_flow,
    logistic_flow,
    logistic_velocity,
    true_log_mean,
    true_q,
    true_rho,
)


def test_logistic_flow_inverse_and_boundary_velocity():
    cfg = ArclengthSyntheticConfig(n_cells_per_tissue=10)
    a = np.linspace(0.0, 1.0, 101)
    for t in (0.0, 0.25, 0.7, 1.0):
        z = logistic_flow(a, t, cfg.flow_strength)
        a_back = inverse_logistic_flow(z, t, cfg.flow_strength)
        np.testing.assert_allclose(a_back, a, atol=2e-7)
        assert logistic_velocity(np.asarray(0.0), t, cfg.flow_strength) == 0.0
        assert logistic_velocity(np.asarray(1.0), t, cfg.flow_strength) == 0.0
        assert np.all(np.diff(z) > 0)


def test_transport_genes_have_zero_material_derivative():
    cfg = ArclengthSyntheticConfig(n_cells_per_tissue=10)
    z = np.linspace(0.05, 0.95, 81)
    t = 0.43
    h = 1e-5
    Mzp = true_log_mean(z + h, np.full_like(z, t), cfg)
    Mzm = true_log_mean(z - h, np.full_like(z, t), cfg)
    Mtp = true_log_mean(z, np.full_like(z, t + h), cfg)
    Mtm = true_log_mean(z, np.full_like(z, t - h), cfg)
    Mz = (Mzp - Mzm) / (2 * h)
    Mt = (Mtp - Mtm) / (2 * h)
    V = logistic_velocity(z, t, cfg.flow_strength)
    material = Mt + V[:, None] * Mz
    transport = material[:, : cfg.n_transport_genes]
    assert np.max(np.abs(transport)) < 2e-5


def test_true_rho_is_integral_of_true_q():
    cfg = ArclengthSyntheticConfig(n_cells_per_tissue=10)
    t = np.linspace(0.0, 1.0, 101)
    rho = true_rho(t, cfg, n_quad=513)
    assert rho[0] == 0.0
    assert np.all(np.diff(rho) >= -1e-10)
    # Finite-difference rho' agrees with q away from the endpoints.
    drho = np.gradient(rho, t, edge_order=2)
    np.testing.assert_allclose(drho[2:-2], true_q(t, cfg)[2:-2], rtol=2e-3, atol=2e-3)
