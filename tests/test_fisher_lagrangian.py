import torch
from torch import nn

from flow_stp.fisher_lagrangian import (
    FisherLagrangianConfig,
    FisherLagrangianModel,
    SplineLogMeanExpressionField,
    fisher_geometry,
    fit_expression_irls,
    integrate_flow_and_rho,
)


class OrthogonalToyField(nn.Module):
    """Fisher coordinates Psi=(z+2,t+2): V=0 and Fisher speed=1."""
    def forward(self, z, time):
        if z.ndim == 1:
            z = z[:, None]
        if time.ndim == 1:
            time = time[:, None]
        psi = torch.cat((z + 2.0, time + 2.0), dim=1)
        return torch.log((psi / 2.0).square())


class PureTransportToyField(nn.Module):
    """Fisher coordinates Psi=(z+t+2,2): V=-1 and rho speed=0."""
    def forward(self, z, time):
        if z.ndim == 1:
            z = z[:, None]
        if time.ndim == 1:
            time = time[:, None]
        y1 = z + time + 2.0
        psi = torch.cat((y1, torch.full_like(y1, 2.0)), dim=1)
        return torch.log((psi / 2.0).square())


def test_orthogonal_change_becomes_expression_arclength():
    field = OrthogonalToyField()
    z = torch.tensor([0.2, 0.7])
    t = torch.tensor([0.3, 0.8])
    g = fisher_geometry(field, z, t)
    assert torch.allclose(g.velocity, torch.zeros_like(g.velocity), atol=1e-5)
    assert torch.allclose(g.speed, torch.ones_like(g.speed), atol=1e-5)
    assert torch.allclose((g.residual * g.dz).sum(1), torch.zeros(2), atol=1e-5)


def test_pure_tangent_change_becomes_transport():
    field = PureTransportToyField()
    z = torch.tensor([0.2, 0.7])
    t = torch.tensor([0.3, 0.8])
    g = fisher_geometry(field, z, t)
    assert torch.allclose(g.velocity, -torch.ones_like(g.velocity), atol=1e-5)
    assert torch.allclose(g.speed, torch.zeros_like(g.speed), atol=1e-5)


def test_lagrangian_rho_accumulates_fisher_speed():
    field = OrthogonalToyField()
    a = torch.tensor([0.2, 0.8])
    traj = integrate_flow_and_rho(field, a, t0=0.0, t1=1.0, n_steps=16)
    assert torch.allclose(traj.position[-1], a, atol=1e-5)
    assert torch.allclose(traj.rho[-1], torch.ones_like(a), atol=1e-4)


def test_temporal_spline_is_exactly_anchored_at_reference_time():
    torch.manual_seed(1)
    field = SplineLogMeanExpressionField(
        3, spatial_basis_dim=6, temporal_basis_dim=3, reference_time=0.0
    )
    with torch.no_grad():
        field.baseline_coeff.normal_()
        field.temporal_coeff.normal_()
    z = torch.linspace(0.05, 0.95, 7)
    t = torch.zeros_like(z)
    eta = field(z, t)
    bz = field.spatial_basis(z)
    baseline = torch.einsum("nj,gj->ng", bz, field.baseline_coeff)
    assert torch.allclose(eta, baseline, atol=1e-6)


def test_spline_analytic_derivatives_match_finite_difference():
    torch.manual_seed(2)
    field = SplineLogMeanExpressionField(
        2, spatial_basis_dim=6, temporal_basis_dim=3, reference_time=0.0
    )
    with torch.no_grad():
        field.baseline_coeff.normal_(std=0.2)
        field.temporal_coeff.normal_(std=0.2)
    z = torch.tensor([0.23, 0.61])
    t = torch.tensor([0.31, 0.73])
    _, dz, dt = field.log_mean_derivatives(z, t)
    h = 1e-4
    fd_z = (field(z + h, t) - field(z - h, t)) / (2 * h)
    fd_t = (field(z, t + h) - field(z, t - h)) / (2 * h)
    assert torch.allclose(dz, fd_z, atol=2e-3, rtol=2e-3)
    assert torch.allclose(dt, fd_t, atol=2e-3, rtol=2e-3)


def test_irls_recovers_simple_poisson_mean_surface():
    torch.manual_seed(3)
    n = 400
    z = torch.rand(n)
    t = torch.rand(n)
    true_field = SplineLogMeanExpressionField(
        1, spatial_basis_dim=4, temporal_basis_dim=3
    )
    with torch.no_grad():
        true_field.baseline_coeff.fill_(0.4)
        true_field.temporal_coeff.zero_()
    lam = true_field.mean(z, t)
    counts = torch.poisson(lam)

    model = FisherLagrangianModel(
        FisherLagrangianConfig(
            num_tissues=1,
            num_genes=1,
            spatial_basis_dim=4,
            temporal_basis_dim=3,
        )
    )
    fit_expression_irls(model, z=z, time=t, counts=counts, max_iter=20)
    pred = model.expression_field.mean(z, t).mean()
    empirical = counts.float().mean()
    assert torch.allclose(pred, empirical, atol=0.12, rtol=0.12)
