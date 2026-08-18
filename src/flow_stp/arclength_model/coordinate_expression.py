"""Joint anatomical-coordinate and mean-expression fitting for the arclength model.

Planned learned objects:

    z = h_i(x, y)
    M(z, t) = log mean expression

The coordinate maps retain the nonzero-gradient (submersion) and geometry/gauge
constraints.  No Stage-2 latent rho or decoder belongs in this module.
"""
