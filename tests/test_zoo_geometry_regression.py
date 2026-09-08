import numpy as np
import torch

from flow_stp.zoo.coordinate_maps import ReferenceNormalizedCoordinateMaps
from flow_stp.zoo.warps import MonotoneWarpFamily


def test_reference_gauge_matches_nominal_coverage():
    torch.manual_seed(7)
    maps = ReferenceNormalizedCoordinateMaps(
        2,
        hidden=(8,),
        initial_missing=0.01,
    )
    coords = [torch.randn(500, 2), torch.randn(600, 2)]
    maps.refresh_gauge_(
        coords,
        max_points=1000,
        lower_quantile=0.01,
        upper_quantile=0.99,
    )

    left, right = maps.coverage_intervals()
    for tid, x in enumerate(coords):
        z = maps.one(x, tid).detach()
        q = torch.quantile(z, torch.tensor([0.01, 0.99]))
        assert torch.allclose(q[0], left[tid], atol=2e-4, rtol=0.0)
        assert torch.allclose(q[1], right[tid], atol=2e-4, rtol=0.0)


def test_nontrivial_normalized_warp_is_monotone_and_invertible():
    warps = MonotoneWarpFamily(1, degree=4)
    with torch.no_grad():
        warps.coefficients[0].copy_(
            torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        )

    z = torch.linspace(0.0, 1.0, 101)
    y = warps.normalized_one(z, 0)
    z_hat = warps.inverse_normalized_one(y, 0)

    assert torch.all(y[1:] > y[:-1])
    assert torch.max(torch.abs(z_hat - z)) < 1e-5


def test_warp_bound_has_nonzero_parameter_gradient():
    warps = MonotoneWarpFamily(1, degree=2, clamp_log_derivative=5.0)
    with torch.no_grad():
        warps.coefficients.fill_(8.0)

    z = torch.tensor([0.2, 0.5, 0.8])
    loss = warps.log_derivative(z, 0).sum()
    loss.backward()

    grad = warps.coefficients.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


def test_identity_inverse_gauge_is_centered():
    warps = MonotoneWarpFamily(3, degree=3)
    left = torch.tensor([0.0, 0.1, 0.2])
    right = torch.tensor([1.0, 0.9, 0.8])
    penalty = warps.inverse_consensus_gauge_penalty(left, right)
    assert torch.allclose(penalty, torch.tensor(0.0), atol=1e-8)
