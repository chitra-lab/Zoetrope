"""Differentiable fixed spline bases used by the Fisher-Lagrangian model.

The basis functions are fixed; only their coefficients are learned.  This keeps
``E(z,t)`` low-complexity while retaining differentiability with respect to the
learned anatomical coordinate ``z``.
"""
from __future__ import annotations

import torch
from torch import nn

Tensor = torch.Tensor


def _open_uniform_knots(
    n_basis: int,
    degree: int,
    domain: tuple[float, float],
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if n_basis < degree + 1:
        raise ValueError("n_basis must be at least degree + 1")
    lo, hi = map(float, domain)
    if not hi > lo:
        raise ValueError("spline domain must satisfy upper > lower")
    n_internal = n_basis - degree - 1
    left = torch.full((degree + 1,), lo, dtype=dtype)
    right = torch.full((degree + 1,), hi, dtype=dtype)
    if n_internal:
        internal = torch.linspace(lo, hi, n_internal + 2, dtype=dtype)[1:-1]
        return torch.cat((left, internal, right))
    return torch.cat((left, right))


def _bspline_basis(x: Tensor, knots: Tensor, degree: int) -> Tensor:
    """Evaluate all B-splines at scalar inputs ``x`` using Cox-de Boor recursion."""
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 1:
        raise ValueError("x must be [N] or [N,1]")
    t = knots.to(device=x.device, dtype=x.dtype)
    values = []
    for i in range(t.numel() - 1):
        mask = (x >= t[i]) & (x < t[i + 1])
        if i == t.numel() - 2:
            mask = mask | (x == t[-1])
        values.append(mask.to(x.dtype))
    basis = torch.stack(values, dim=1)

    for p in range(1, degree + 1):
        n = t.numel() - p - 1
        next_basis = []
        for i in range(n):
            left_denom = t[i + p] - t[i]
            right_denom = t[i + p + 1] - t[i + 1]
            left = torch.zeros_like(x)
            right = torch.zeros_like(x)
            if float(left_denom.abs()) > 0.0:
                left = (x - t[i]) / left_denom * basis[:, i]
            if float(right_denom.abs()) > 0.0:
                right = (t[i + p + 1] - x) / right_denom * basis[:, i + 1]
            next_basis.append(left + right)
        basis = torch.stack(next_basis, dim=1)
    return basis


def _bspline_derivative(x: Tensor, knots: Tensor, degree: int) -> Tensor:
    if degree == 0:
        n_basis = knots.numel() - 1
        n = x.shape[0]
        return torch.zeros((n, n_basis), dtype=x.dtype, device=x.device)
    t = knots.to(device=x.device, dtype=x.dtype)
    lower = _bspline_basis(x, t, degree - 1)
    n_basis = t.numel() - degree - 1
    cols = []
    for i in range(n_basis):
        left_denom = t[i + degree] - t[i]
        right_denom = t[i + degree + 1] - t[i + 1]
        left = torch.zeros_like(lower[:, 0])
        right = torch.zeros_like(lower[:, 0])
        if float(left_denom.abs()) > 0.0:
            left = degree / left_denom * lower[:, i]
        if float(right_denom.abs()) > 0.0:
            right = degree / right_denom * lower[:, i + 1]
        cols.append(left - right)
    return torch.stack(cols, dim=1)


class FixedBSplineBasis(nn.Module):
    """Fixed open-uniform B-spline basis on a finite interval."""

    def __init__(
        self,
        n_basis: int,
        *,
        degree: int = 3,
        domain: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__()
        self.n_basis = int(n_basis)
        self.degree = int(degree)
        self.domain = (float(domain[0]), float(domain[1]))
        self.register_buffer(
            "knots", _open_uniform_knots(self.n_basis, self.degree, self.domain)
        )

    def forward(self, x: Tensor) -> Tensor:
        return _bspline_basis(x, self.knots, self.degree)

    def derivative(self, x: Tensor) -> Tensor:
        return _bspline_derivative(x, self.knots, self.degree)


class AnchoredTemporalSplineBasis(nn.Module):
    """Linearly independent spline contrasts that are exactly zero at ``t0``.

    A raw B-spline basis partitions unity, so simply subtracting its value at
    ``t0`` leaves one redundant contrast.  We remove that gauge by contrasting
    each of the first raw basis functions against the final basis function.
    ``n_basis`` is the number of *active* temporal contrasts; internally one
    extra raw B-spline is used.
    """

    def __init__(
        self,
        n_basis: int,
        *,
        degree: int = 3,
        domain: tuple[float, float] = (0.0, 1.0),
        t0: float = 0.0,
    ) -> None:
        super().__init__()
        if n_basis < 1:
            raise ValueError("n_basis must be positive")
        self.n_basis = int(n_basis)
        self.t0 = float(t0)
        self.raw = FixedBSplineBasis(
            self.n_basis + 1, degree=degree, domain=domain
        )
        if not (domain[0] <= self.t0 <= domain[1]):
            raise ValueError("t0 must lie in the temporal spline domain")

    def _raw_reference(self, x: Tensor) -> Tensor:
        ref = torch.full((1,), self.t0, dtype=x.dtype, device=x.device)
        return self.raw(ref)

    def forward(self, time: Tensor) -> Tensor:
        raw = self.raw(time)
        ref = self._raw_reference(time)
        anchored = raw - ref
        return anchored[:, :-1] - anchored[:, -1:]

    def derivative(self, time: Tensor) -> Tensor:
        d_raw = self.raw.derivative(time)
        return d_raw[:, :-1] - d_raw[:, -1:]
