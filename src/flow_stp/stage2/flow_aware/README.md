# Flow-aware Stage 2 — confirmatory direct-flow v2

The default implementation matches the recent confirmatory flow-aware model.

- decoder: `B(a) + D(a,R) - D(a,0)`
- inverse flow: 24-segment monotone PWL `A(z,t)` with 64x64 time MLP
- material rate: 96x96x96 SiLU MLP with ReLU output
- native-rate final bias: 0.40
- integration: 12-point Gauss-Legendre quadrature
- flow regularization: off by default
- learning rates: baseline 2e-3, rate 1e-3, decoder 1.5e-3, flow 7.5e-4

For the heavier final confirmatory setting use `flow_segments=32` and `r_quadrature=20`.
