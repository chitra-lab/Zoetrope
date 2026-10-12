"""Audit weak observability without changing the Fisher velocity."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from flow_stp.stage2.observability import audit_flow_observability


def trajectory(z_values):
    z = torch.as_tensor(z_values, dtype=torch.float64)
    t = torch.linspace(0., 1., len(z), dtype=z.dtype)[:, None].expand_as(z)
    return SimpleNamespace(position=z, time=t)


def test_well_observed_constant_tangent():
    r = audit_flow_observability(
        lambda z, t: 4 + z - 2*t,
        trajectory([[0., .3], [.1, .4], [.2, .5]]),
    )
    assert r.weak_node_fraction == 0
    assert r.weak_trajectory_fraction == 0
    assert r.tangent_energy_median == pytest.approx(1.)
    assert len(r.per_time_weak_fraction) == 3
    assert r.regularizer_dominance_max < 1e-6
    assert r.summary()["n_trajectories"] == 2


def test_flags_flat_tangent():
    r = audit_flow_observability(
        lambda z, t: 4 + 1e-8*z - 2*t,
        trajectory([[0., .3], [.1, .4]]),
    )
    assert r.weak_node_fraction == 1
    assert r.weak_trajectory_fraction == 1
    assert r.regularizer_dominance_max > .99


def test_mixed_weak_and_strong_trajectories():
    r = audit_flow_observability(
        lambda z,t: 4 + z*z - t,
        trajectory([[0., .5], [0., .5]]),
    )
    assert r.weak_node_fraction == pytest.approx(.5)
    assert r.weak_trajectory_fraction == pytest.approx(.5)
    np.testing.assert_array_equal(r.weak_node_mask, [[True,False],[True,False]])


def test_invalid_parameters():
    with pytest.raises(ValueError, match="eps"):
        audit_flow_observability(lambda z,t: z+t, trajectory([[.1]]), eps=-1)
    with pytest.raises(ValueError, match="positive"):
        audit_flow_observability(
            lambda z,t: z+t, trajectory([[.1]]), min_tangent_energy=0.,
        )
