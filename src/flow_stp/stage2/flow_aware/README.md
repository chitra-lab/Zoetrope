# Flow-aware Stage 2 — frozen direct-flow v1

This directory preserves the direct flow-aware model used in the paired
identifiability benchmarks. It is a new sibling implementation and does **not**
replace `flow_stp.stage2.basic`.

The model learns

\[
a=A(z,t)=F_t^{-1}(z)
\]

and

\[
R(a,t)=\int_0^t \operatorname{softplus}(h_\theta(a,\tau))\,d\tau,
\]

then decodes expression as

\[
\log\mu_g(z,t)=B_g(a)+D_g(a,R(a,t))-D_g(a,0).
\]

Derived Eulerian quantities are

\[
\rho(z,t)=R(A(z,t),t),\qquad
V(z,t)=-A_t/A_z,\qquad
q(z,t)=R_t(A(z,t),t).
\]

Frozen benchmark defaults:
- 10 flow segments
- flow time MLP: 1 → 32 → 32 → 10, tanh
- baseline MLP: 1 → 64 → 64 → genes, GELU
- material-rate MLP: 2 → 64 → 64 → 1, SiLU + softplus
- decoder MLP: 2 → 64 → 64 → genes, GELU
- 12 midpoint quadrature nodes
- spatial/temporal flow smoothness weights: 2e-4 / 2e-4
- training LR: baseline 2e-3, rate 1e-3, decoder 2e-3, flow 1.5e-3
- gradient clipping: 5.0

Import:

```python
from flow_stp.stage2.flow_aware import FlowAwareStage2Model
model = FlowAwareStage2Model(num_genes=2000)
```
