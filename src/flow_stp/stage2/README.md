# Stage 2: derived Fisher dynamics

Stage 2 does not fit a second neural trajectory model. It consumes the
biological expression field E(z,t) learned in Stage 1.

For independent Poisson means, define Fisher-flat coordinates

Psi(z,t) = 2 sqrt(E(z,t)).

Stage 2 differentiates the learned field to obtain Psi_z and Psi_t, then
computes the local anatomical velocity in closed form:

V(z,t) = - <Psi_z, Psi_t> / <Psi_z, Psi_z>.

Equivalently, this is the scalar v minimizing ||Psi_t + v Psi_z||^2. The
denominator is retained as an observability diagnostic.

The anatomical flow map solves

dF(a,t)/dt = V(F(a,t),t), with F(a,0) = a,

and is integrated numerically with classical fourth-order Runge-Kutta (RK4).

Typical use with the Stage-1 result:

    from flow_stp.stage2 import estimate_local_dynamics, integrate_flow_rk4

    field = stage1_result.model.log_expression
    local = estimate_local_dynamics(field, z, t)
    trajectory = integrate_flow_rk4(field, anchors, t0=0.0, t1=1.0)

There is no finite-time flow fitting, learned inverse-flow model, neural PDE,
or separate Stage-2 decoder in the active repository.
