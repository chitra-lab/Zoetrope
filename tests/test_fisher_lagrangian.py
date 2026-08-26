import torch
from torch import nn

from flow_stp.fisher_lagrangian import fisher_geometry, integrate_flow_and_rho


class OrthogonalToyField(nn.Module):
    """Construct Fisher coordinates Y=(z+2, t+2).

    Then Y_z=(1,0), Y_t=(0,1), so V=0, residual=Y_t, q=1,
    and rho(a,t)=t exactly.
    """

    def forward(self, z, time):
        if z.ndim == 1:
            z = z[:, None]
        if time.ndim == 1:
            time = time[:, None]
        y1 = z + 2.0
        y2 = time + 2.0
        Y = torch.cat((y1, y2), dim=1)
        lam = (Y / 2.0).square()
        return torch.log(lam)


class PureTransportToyField(nn.Module):
    """Construct Fisher coordinates with Y=(z+t+2, constant).

    Temporal change lies entirely in the anatomical tangent direction, so
    V=-1 and the developmental residual/speed vanish.
    """

    def forward(self, z, time):
        if z.ndim == 1:
            z = z[:, None]
        if time.ndim == 1:
            time = time[:, None]
        y1 = z + time + 2.0
        y2 = torch.full_like(y1, 2.0)
        Y = torch.cat((y1, y2), dim=1)
        lam = (Y / 2.0).square()
        return torch.log(lam)


def test_orthogonal_change_becomes_development():
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
