# Stage 2: derivative-based Fisher dynamics

Stage 2 does not fit a second trajectory model. Stage 1 directly estimates the
independent-Poisson Fisher-flat expression coordinates

Psi(z,t) = 2 sqrt(E(z,t)).

Stage 2 differentiates that fitted field to obtain Psi_z and Psi_t and computes

V(z,t) = - <Psi_z, Psi_t> / <Psi_z, Psi_z>.

Because Psi is fitted directly, there is no fitted log-expression field and no
post-hoc exp(log E / 2) transformation before differentiation.

The anatomical flow map solves

dF(a,t)/dt = V(F(a,t),t), with F(a,0) = a,

using classical fourth-order Runge-Kutta (RK4).

Typical use:

    field = stage1_result.model.fisher_flat
    local = estimate_local_dynamics(field, z, t)
    trajectory = integrate_flow_rk4(field, anchors, t0=0.0, t1=1.0)

The corresponding biological Poisson mean is always recoverable as

E(z,t) = Psi(z,t)^2 / 4.

There is no finite-time flow fitting, learned inverse-flow model, neural PDE,
or separate Stage-2 decoder in the active repository.
