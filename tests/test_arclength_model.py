import numpy as np
import torch

from flow_stp.arclength_model.coordinate_expression import ArclengthConfig, ArclengthModel
from flow_stp.arclength_model.derivatives import expression_jvp
from flow_stp.arclength_model.registration import solve_smooth_registration
from flow_stp.arclength_model.trajectory import integrate_flow_and_arclength


def test_temporal_basis_field_and_jvp_shapes():
    cfg = ArclengthConfig(num_slices=2, num_genes=3, num_interior_time_knots=2)
    model = ArclengthModel(cfg)
    z = torch.linspace(0.1, 0.9, 5)
    t = torch.linspace(0.0, 1.0, 5)
    M, Mz, Mt = expression_jvp(model.expression_field, z, t)
    assert M.shape == (5, 3)
    assert Mz.shape == M.shape
    assert Mt.shape == M.shape
    assert torch.isfinite(Mz).all()
    assert torch.isfinite(Mt).all()


def test_registration_recovers_translation_interior_and_zero_boundaries():
    # M(z,t)=z-t for one gene.  M_z=1, M_t=-1, so the pointwise optimum
    # is V=1 because M_t + V M_z = 0.
    nt, nz = 4, 9
    z = np.linspace(0.0, 1.0, nz)
    t = np.linspace(0.0, 1.0, nt)
    Mz = np.ones((nt, nz, 1))
    Mt = -np.ones((nt, nz, 1))
    result = solve_smooth_registration(
        Mz, Mt, z, t,
        lambda_v=0.0,
        lambda_z=0.0,
        lambda_t=0.0,
    )
    assert np.allclose(result.V[:, 0], 0.0)
    assert np.allclose(result.V[:, -1], 0.0)
    assert np.allclose(result.V[:, 1:-1], 1.0)
    assert np.allclose(result.q[:, 1:-1], 0.0)


def test_flow_is_monotone_and_rho_accumulates():
    z = np.linspace(0.0, 1.0, 21)
    t = np.linspace(0.0, 1.0, 6)
    # Smooth boundary-compatible rightward velocity and constant q.
    V = np.tile((0.1 * z * (1.0 - z))[None, :], (len(t), 1))
    q = np.ones_like(V) * 2.0
    flow = integrate_flow_and_arclength(V, q, z, t, a_grid=np.linspace(0.0, 1.0, 25))
    assert np.all(np.diff(flow.F, axis=1) > 0)
    assert np.all(np.diff(flow.rho, axis=0) >= -1e-10)
    assert np.allclose(flow.rho[-1], 2.0, atol=1e-5)
