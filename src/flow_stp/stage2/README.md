# Stage 2: derivative-based Fisher dynamics

Stage 2 does not fit a second trajectory model. It consumes E(z,t) learned in
Stage 1, estimates E_z and E_t, and computes

V(z,t) = - <E_z,E_t>_I / <E_z,E_z>_I,

where the independent-Poisson Fisher inner product is

<u,v>_I = sum_g u_g v_g / E_g.

This is equivalent to the Euclidean projection formula in
Psi = 2 sqrt(E) coordinates.

The anatomical flow map then solves

dF(a,t)/dt = V(F(a,t),t), with F(a,0) = a,

using classical fourth-order Runge-Kutta (RK4).

Typical use:

    field = stage1_result.model.log_expression
    local = estimate_local_dynamics(field, z, t)
    trajectory = integrate_flow_rk4(field, anchors, t0=0.0, t1=1.0)

There is no finite-time flow fitting, learned inverse-flow model, neural PDE,
or separate Stage-2 decoder in the active repository.
