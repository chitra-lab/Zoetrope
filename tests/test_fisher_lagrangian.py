import torch
from torch import nn

from flow_stp.fisher_lagrangian import (
    FisherLagrangianConfig,
    FisherLagrangianModel,
    FixedBSplineBasis,
    SplineLogMeanExpressionField,
    TissueCoordinateMap,
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


def test_coordinate_map_is_unbounded_and_has_explicit_offset():
    torch.manual_seed(0)
    mapping = TissueCoordinateMap(spatial_dim=2, hidden=(8, 8), activation="silu")
    xy = torch.randn(16, 2)
    with torch.no_grad():
        mapping.offset.fill_(3.0)
    raw = mapping.raw_coordinate(xy)
    z = mapping(xy)
    assert torch.allclose(z - raw, torch.full_like(z, 3.0))
    assert any((z < 0.0).flatten()) or any((z > 1.0).flatten())


def test_unit_speed_linear_coordinate_has_zero_gauge_penalty():
    model = FisherLagrangianModel(
        FisherLagrangianConfig(
            num_tissues=1,
            num_genes=1,
            coordinate_hidden=(),
            coordinate_domain=(-3.0, 3.0),
        )
    )
    linear = model.coordinate_maps[0].raw.net[0]
    with torch.no_grad():
        linear.weight.zero_()
        linear.weight[0, 0] = 1.0
        model.coordinate_maps[0].offset.zero_()
    xy = torch.randn(32, 2)
    tissue = torch.zeros(32, dtype=torch.long)
    gauge, mean_norm = model.coordinate_gauge_penalty(xy, tissue)
    assert torch.allclose(gauge, torch.zeros_like(gauge), atol=1e-6)
    assert torch.allclose(mean_norm, torch.ones_like(mean_norm), atol=1e-6)


def test_temporal_spline_is_exactly_anchored_at_reference_time():
    torch.manual_seed(1)
    field = SplineLogMeanExpressionField(
        3,
        spatial_basis_dim=6,
        temporal_basis_dim=3,
        reference_time=0.0,
        coordinate_domain=(-2.0, 2.0),
    )
    with torch.no_grad():
        field.baseline_coeff.normal_()
        field.temporal_coeff.normal_()
    z = torch.linspace(-1.5, 1.5, 7)
    t = torch.zeros_like(z)
    eta = field(z, t)
    bz = field.spatial_basis(z)
    baseline = torch.einsum("nj,gj->ng", bz, field.baseline_coeff)
    assert torch.allclose(eta, baseline, atol=1e-6)


def test_open_bspline_includes_right_endpoint():
    basis = FixedBSplineBasis(6, degree=3, domain=(-2.0, 3.0))
    values = basis(torch.tensor([3.0]))
    expected = torch.zeros_like(values)
    expected[0, -1] = 1.0
    assert torch.allclose(values, expected, atol=1e-7)


def test_spline_analytic_derivatives_match_finite_difference():
    torch.manual_seed(2)
    field = SplineLogMeanExpressionField(
        2,
        spatial_basis_dim=6,
        temporal_basis_dim=3,
        reference_time=0.0,
        coordinate_domain=(-1.0, 1.0),
    )
    with torch.no_grad():
        field.baseline_coeff.normal_(std=0.2)
        field.temporal_coeff.normal_(std=0.2)
    z = torch.tensor([-0.23, 0.61])
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
    z = 2.0 * torch.rand(n) - 1.0
    t = torch.rand(n)
    true_field = SplineLogMeanExpressionField(
        1,
        spatial_basis_dim=4,
        temporal_basis_dim=3,
        coordinate_domain=(-1.0, 1.0),
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
            coordinate_domain=(-1.0, 1.0),
        )
    )
    fit_expression_irls(model, z=z, time=t, counts=counts, max_iter=20)
    pred = model.expression_field.mean(z, t).mean()
    empirical = counts.float().mean()
    assert torch.allclose(pred, empirical, atol=0.12, rtol=0.12)


class AffineFlowToyField(nn.Module):
    """Exact variable-flow field with V(z,t)=lambda*z+c and constant rho speed.

    Fisher coordinates are
        Psi_1 = B + exp(-lambda*t) * (z + c/lambda)
        Psi_2 = B + q*t.

    Psi_1 is constant along the exact affine flow, while Psi_2 changes
    orthogonally to the spatial tangent. Therefore the Fisher projection gives
    V=lambda*z+c exactly and the Fisher arclength speed is q.
    """

    def __init__(
        self,
        *,
        lam: float = 0.4,
        c: float = 0.1,
        q: float = 0.35,
        baseline: float = 4.0,
    ):
        super().__init__()
        self.lam = float(lam)
        self.c = float(c)
        self.q = float(q)
        self.baseline = float(baseline)

    def forward(self, z, time):
        if z.ndim == 1:
            z = z[:, None]
        if time.ndim == 1:
            time = time[:, None]
        lam = self.lam
        transported = (
            self.baseline
            + torch.exp(-lam * time) * (z + self.c / lam)
        )
        developmental = self.baseline + self.q * time
        psi = torch.cat((transported, developmental), dim=1)
        return torch.log((psi / 2.0).square())

    def exact_flow(self, a, time):
        lam = self.lam
        return (
            (a + self.c / lam) * torch.exp(lam * time)
            - self.c / lam
        )


def test_rk4_recovers_nonconstant_affine_flow_and_rho():
    """Validate the trajectory solver independently of E estimation."""
    field = AffineFlowToyField()
    a = torch.tensor([0.05, 0.20, 0.40, 0.60], dtype=torch.float64)

    traj = integrate_flow_and_rho(
        field,
        a,
        t0=0.0,
        t1=1.0,
        n_steps=32,
        eps=0.0,
    )

    times = traj.time[:, 0]
    exact_position = field.exact_flow(
        a[None, :],
        times[:, None],
    )
    exact_rho = field.q * times[:, None].expand_as(exact_position)

    assert torch.allclose(
        traj.position,
        exact_position,
        atol=2e-10,
        rtol=2e-10,
    )
    assert torch.allclose(
        traj.rho,
        exact_rho,
        atol=2e-12,
        rtol=2e-12,
    )


def test_rk4_has_expected_fourth_order_convergence():
    """Halving the step should reduce affine-flow error by about 16x."""
    field = AffineFlowToyField()
    a = torch.tensor([0.05, 0.20, 0.40, 0.60], dtype=torch.float64)
    exact_final = field.exact_flow(
        a,
        torch.tensor(1.0, dtype=a.dtype),
    )

    errors = []
    for n_steps in (1, 2, 4, 8):
        traj = integrate_flow_and_rho(
            field,
            a,
            t0=0.0,
            t1=1.0,
            n_steps=n_steps,
            eps=0.0,
        )
        errors.append(
            float((traj.position[-1] - exact_final).abs().max())
        )

    assert all(
        later < earlier
        for earlier, later in zip(errors[:-1], errors[1:])
    )
    ratios = [
        earlier / later
        for earlier, later in zip(errors[:-1], errors[1:])
    ]
    assert min(ratios[1:]) > 12.0
