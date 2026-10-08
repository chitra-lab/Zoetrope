# Stage 2: continuous, derived Fisher dynamics

Stage 1 now **fits log E(z,t)**, rather than fitting Fisher-flat Psi directly.
Its method `model.fisher_flat(z,t)` returns

```text
Psi(z,t) = 2 exp(log E(z,t)/2).
```

This transformation is differentiable, so Stage 2 obtains Psi_z and Psi_t
through automatic differentiation and computes

```text
V(z,t) = - <Psi_z, Psi_t> / <Psi_z, Psi_z>.
q(z,t) = Psi_t + V(z,t) Psi_z.
```

The anatomical flow solves `dF/dt = V(F,t)` with RK4. No separately learned
trajectory model or additional fitted Stage-2 decoder is used.

```python
field = stage1_result.model.fisher_flat
local = estimate_local_dynamics(field, z, t)
trajectory = integrate_flow_rk4(field, anchors, t0=0., t1=1.)
```
