import numpy as np


def test_finite_flow_changes_by_endpoint_conjugacy():
    """A finite flow can be represented in any monotone chart by conjugacy."""
    # Reference-gauge flow F(z)=z+0.1.
    F = lambda z: z + 0.1

    # Different monotone endpoint gauges.
    h0 = lambda z: 2.0 * z + 1.0
    h1 = lambda z: 3.0 * z - 0.5
    h0_inv = lambda u: (u - 1.0) / 2.0

    u0 = np.linspace(1.0, 2.0, 11)
    transformed = h1(F(h0_inv(u0)))
    expected = 3.0 * ((u0 - 1.0) / 2.0 + 0.1) - 0.5

    assert np.allclose(transformed, expected)
