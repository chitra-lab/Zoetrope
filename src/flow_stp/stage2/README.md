# Stage 2: Fisher dynamics and optional trajectory-constrained refinement

Stage 1 fits log E(z,t) and exposes Fisher-flat Psi = 2 exp(log E/2). Stage 2
differentiates Psi to estimate the Fisher-projection velocity

    V(z,t) = - <Psi_z, Psi_t> / <Psi_z, Psi_z>
    q(z,t) = Psi_t + V(z,t) Psi_z

and integrates dF/dt = V(F,t) by RK4. These existing APIs remain unchanged.

## Optional no-flux refinement

A separate postprocessing step fits one low-dimensional velocity field to sampled
Fisher-derived trajectories (no synthetic true velocity or Stage-1 retraining):

    v_theta(z,t) = z(1-z) * sum_{i=0}^3 sum_{j=0}^3 theta_ij B_i^3(z) B_j^3(t)

The field is a tensor-product Bernstein polynomial; the endpoint factor
enforces zero **relative migration** at the two boundaries. The fitting loss is
mean squared error between *integrated* ODE positions and input trajectories,
plus a penalty on second differences of coefficients. Optimization uses Adam
followed by LBFGS. The refined velocity is not necessarily the pointwise Fisher
projection, so retain the original velocity for diagnostics.

**Critical prerequisite:** Trajectories must be provided in ONE common,
bounded material coordinate, with meaningful boundaries. Do NOT directly apply
this to independently normalized sections or assume learned z is automatically
a physical material coordinate. No-flux is optional and may be incorrect for
real tissues with boundary crossing. This step does not infer tissue geometry.

    import numpy as np
    from flow_stp.stage2 import (
        RefinementConfig, fit_trajectory_velocity, physical_velocity,
    )

    # positions: [n_times, n_anchors] in one fixed material z gauge
    # times: [n_times]; missing positions may be NaN or masked out
    result = fit_trajectory_velocity(
        positions, times, config=RefinementConfig(no_flux=True),
    )
    refined = result.field
    velocity_z = refined.velocity(z_tensor, t_tensor)
    refined_flow = refined.integrate(initial_z, times_tensor)

To represent migration in physical tissue coordinates, supply the independently
identified map G_t(u,z) and its derivatives:

    migration_xy, total_xy = physical_velocity(
        refined, u, z, time, dG_dz=geometry_z, dG_dt=geometry_t,
    )

where migration_xy = dG/dz * v_theta and
total_xy = dG/dt + migration_xy. Supplying only dG_dz returns
(migration_xy, None). Geometry estimation and XY inversion are the caller's
responsibility, not a capability of this postprocessor.

## Synthetic validation status

In the no-rotation cortical-crescent benchmark (using known geometry), the
optional no-flux refinement improved held-out migration velocity NRMSE from
0.950 (trajectory PCHIP) to 0.221, and migration displacement NRMSE from
0.234 (original Fisher contours) to 0.0678. These are experiment-specific
results, not guarantees for data with uncertain material registration.
