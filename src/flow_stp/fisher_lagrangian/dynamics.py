"""Temporal-resolution-aware Fisher-Lagrangian dynamics.

The default estimator fits finite-time monotone flow maps directly between
observed ages by molecular persistence in Fisher-flat expression coordinates.
The original continuous estimator remains available for densely sampled time
courses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

import torch

Tensor = torch.Tensor


class ExpressionField(Protocol):
    def __call__(self, z: Tensor, time: Tensor) -> Tensor:
        """Return log mean expression with shape [n, genes]."""


DynamicsMode = Literal["finite_time", "continuous"]


@dataclass(frozen=True)
class DynamicsConfig:
    mode: DynamicsMode = "finite_time"

    # Finite-time estimator.
    k_candidates: tuple[int, ...] = (4, 6, 8, 12, 16, 24, 32)
    validation_fraction: float = 0.25
    selection: Literal["one_se"] = "one_se"
    optimizer_steps: int = 1000
    learning_rate: float = 5e-2
    seed: int = 20261003

    # Continuous estimator.
    continuous_steps: int = 128
    fisher_eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.mode not in ("finite_time", "continuous"):
            raise ValueError("mode must be 'finite_time' or 'continuous'")
        if not self.k_candidates:
            raise ValueError("k_candidates cannot be empty")
        if min(self.k_candidates) < 2:
            raise ValueError("every finite-time K must be >= 2")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in (0,1)")
        if self.optimizer_steps <= 0:
            raise ValueError("optimizer_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.continuous_steps <= 0:
            raise ValueError("continuous_steps must be positive")


@dataclass(frozen=True)
class MonotonePiecewiseLinearFlow:
    source_knots: Tensor
    target_knots: Tensor
    source_time: float
    target_time: float

    def __post_init__(self) -> None:
        if self.source_knots.ndim != 1 or self.target_knots.ndim != 1:
            raise ValueError("flow knots must be one-dimensional")
        if self.source_knots.shape != self.target_knots.shape:
            raise ValueError("source and target knots must have matching shape")
        if len(self.source_knots) < 2:
            raise ValueError("at least two knots are required")
        if not torch.all(self.source_knots[1:] > self.source_knots[:-1]):
            raise ValueError("source knots must be strictly increasing")
        if not torch.all(self.target_knots[1:] > self.target_knots[:-1]):
            raise ValueError("target knots must be strictly increasing")
        if self.target_time <= self.source_time:
            raise ValueError("target_time must exceed source_time")

    @property
    def dt(self) -> float:
        return float(self.target_time - self.source_time)

    def __call__(self, z: Tensor) -> Tensor:
        original_shape = z.shape
        x = z.reshape(-1)
        sk = self.source_knots.to(device=x.device, dtype=x.dtype)
        tk = self.target_knots.to(device=x.device, dtype=x.dtype)

        if torch.any(x < sk[0]) or torch.any(x > sk[-1]):
            raise ValueError("query lies outside finite-time flow support")

        idx = torch.searchsorted(sk, x, right=True) - 1
        idx = idx.clamp(0, len(sk) - 2)
        x0 = sk[idx]
        x1 = sk[idx + 1]
        y0 = tk[idx]
        y1 = tk[idx + 1]
        weight = (x - x0) / (x1 - x0)
        return (y0 + weight * (y1 - y0)).reshape(original_shape)

    def jacobian(self, z: Tensor) -> Tensor:
        original_shape = z.shape
        x = z.reshape(-1)
        sk = self.source_knots.to(device=x.device, dtype=x.dtype)
        tk = self.target_knots.to(device=x.device, dtype=x.dtype)

        if torch.any(x < sk[0]) or torch.any(x > sk[-1]):
            raise ValueError("query lies outside finite-time flow support")

        idx = torch.searchsorted(sk, x, right=True) - 1
        idx = idx.clamp(0, len(sk) - 2)
        slope = (tk[idx + 1] - tk[idx]) / (sk[idx + 1] - sk[idx])
        return slope.reshape(original_shape)

    def interval_velocity(self, z: Tensor) -> Tensor:
        return (self(z) - z) / self.dt


@dataclass(frozen=True)
class FiniteTimeCVRow:
    k: int
    mean_validation_loss: float
    validation_se: float


@dataclass(frozen=True)
class FiniteTimeIntervalFit:
    flow: MonotonePiecewiseLinearFlow
    selected_k: int
    raw_best_k: int
    cv: tuple[FiniteTimeCVRow, ...]


@dataclass(frozen=True)
class FiniteTimeTrajectory:
    time: Tensor
    position: Tensor
    rho: Tensor
    interval_velocity: Tensor
    flows: tuple[MonotonePiecewiseLinearFlow, ...]
    interval_fits: tuple[FiniteTimeIntervalFit, ...]


@dataclass(frozen=True)
class DynamicsResult:
    mode: DynamicsMode
    finite_time: FiniteTimeTrajectory | None = None
    continuous: object | None = None

    @property
    def rho(self) -> Tensor:
        if self.mode == "finite_time":
            assert self.finite_time is not None
            return self.finite_time.rho
        assert self.continuous is not None
        return self.continuous.rho

    @property
    def position(self) -> Tensor:
        if self.mode == "finite_time":
            assert self.finite_time is not None
            return self.finite_time.position
        assert self.continuous is not None
        return self.continuous.position

    @property
    def flow(self):
        if self.mode == "finite_time":
            assert self.finite_time is not None
            return self.finite_time.flows
        return None

    @property
    def velocity(self) -> Tensor:
        """Velocity at the temporal resolution supported by the selected mode."""
        if self.mode == "finite_time":
            assert self.finite_time is not None
            return self.finite_time.interval_velocity
        assert self.continuous is not None
        return self.continuous.velocity

    @property
    def interval_velocity(self) -> Tensor | None:
        if self.mode == "finite_time":
            assert self.finite_time is not None
            return self.finite_time.interval_velocity
        return None


def _as_col(x: Tensor) -> Tensor:
    return x[:, None] if x.ndim == 1 else x


def _fisher_embedding(field: ExpressionField, z: Tensor, time: float) -> Tensor:
    zc = _as_col(z)
    tc = torch.full_like(zc, float(time))
    log_mean = field(zc, tc)
    return 2.0 * torch.exp(0.5 * log_mean)


def _target_knots_from_logits(
    logits: Tensor,
    *,
    lower: Tensor,
    upper: Tensor,
) -> Tensor:
    gaps = torch.softmax(logits, dim=0) * (upper - lower)
    tail = lower + torch.cumsum(gaps, dim=0)
    return torch.cat((lower.reshape(1), tail), dim=0)


def _piecewise_linear(
    x: Tensor,
    source_knots: Tensor,
    target_knots: Tensor,
) -> Tensor:
    original_shape = x.shape
    flat = x.reshape(-1)
    idx = torch.searchsorted(source_knots, flat, right=True) - 1
    idx = idx.clamp(0, len(source_knots) - 2)
    x0 = source_knots[idx]
    x1 = source_knots[idx + 1]
    y0 = target_knots[idx]
    y1 = target_knots[idx + 1]
    weight = (flat - x0) / (x1 - x0)
    return (y0 + weight * (y1 - y0)).reshape(original_shape)


def _fit_fixed_k(
    field: ExpressionField,
    *,
    source_time: float,
    target_time: float,
    z_grid: Tensor,
    genes: Tensor,
    k: int,
    steps: int,
    learning_rate: float,
) -> MonotonePiecewiseLinearFlow:
    z_grid = z_grid.detach()
    lower = z_grid.min()
    upper = z_grid.max()
    source_knots = torch.linspace(
        lower,
        upper,
        int(k),
        dtype=z_grid.dtype,
        device=z_grid.device,
    )

    logits = torch.zeros(
        int(k) - 1,
        dtype=z_grid.dtype,
        device=z_grid.device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([logits], lr=float(learning_rate))
    source_psi = _fisher_embedding(field, z_grid, source_time).detach()[:, genes]

    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        target_knots = _target_knots_from_logits(
            logits,
            lower=lower,
            upper=upper,
        )
        mapped = _piecewise_linear(z_grid, source_knots, target_knots)
        target_psi = _fisher_embedding(field, mapped, target_time)[:, genes]
        loss = (target_psi - source_psi).square().mean()
        (grad,) = torch.autograd.grad(loss, logits)
        logits.grad = grad
        optimizer.step()

    with torch.no_grad():
        target_knots = _target_knots_from_logits(
            logits,
            lower=lower,
            upper=upper,
        ).detach()

    return MonotonePiecewiseLinearFlow(
        source_knots=source_knots.detach(),
        target_knots=target_knots,
        source_time=float(source_time),
        target_time=float(target_time),
    )


def _per_gene_loss(
    field: ExpressionField,
    flow: MonotonePiecewiseLinearFlow,
    *,
    z_grid: Tensor,
    genes: Tensor,
) -> Tensor:
    with torch.no_grad():
        source_psi = _fisher_embedding(field, z_grid, flow.source_time)[:, genes]
        target_psi = _fisher_embedding(
            field,
            flow(z_grid),
            flow.target_time,
        )[:, genes]
        return (target_psi - source_psi).square().mean(dim=0)


def fit_finite_time_interval(
    field: ExpressionField,
    *,
    source_time: float,
    target_time: float,
    z_grid: Tensor,
    config: DynamicsConfig | None = None,
) -> FiniteTimeIntervalFit:
    """Fit one observed-time flow map using held-out-gene complexity selection."""
    config = DynamicsConfig() if config is None else config
    if target_time <= source_time:
        raise ValueError("target_time must exceed source_time")
    if z_grid.ndim != 1 or len(z_grid) < 2:
        raise ValueError("z_grid must be a one-dimensional grid with at least 2 points")
    if not torch.all(z_grid[1:] > z_grid[:-1]):
        raise ValueError("z_grid must be strictly increasing")

    with torch.no_grad():
        n_genes = int(_fisher_embedding(field, z_grid[:1], source_time).shape[1])

    if n_genes < 2:
        raise ValueError("finite-time CV requires at least two genes")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.seed))
    perm = torch.randperm(n_genes, generator=generator)
    n_val = max(1, int(round(config.validation_fraction * n_genes)))
    n_val = min(n_val, n_genes - 1)
    val_genes = perm[:n_val].to(z_grid.device)
    train_genes = perm[n_val:].to(z_grid.device)
    all_genes = torch.arange(n_genes, device=z_grid.device)

    rows: list[FiniteTimeCVRow] = []

    for k in sorted(set(int(v) for v in config.k_candidates)):
        flow = _fit_fixed_k(
            field,
            source_time=source_time,
            target_time=target_time,
            z_grid=z_grid,
            genes=train_genes,
            k=k,
            steps=config.optimizer_steps,
            learning_rate=config.learning_rate,
        )
        losses = _per_gene_loss(field, flow, z_grid=z_grid, genes=val_genes)
        mean = float(losses.mean().cpu())
        if len(losses) > 1:
            se = float((losses.std(unbiased=True) / (len(losses) ** 0.5)).cpu())
        else:
            se = 0.0
        rows.append(FiniteTimeCVRow(k=k, mean_validation_loss=mean, validation_se=se))

    raw_index = min(range(len(rows)), key=lambda i: rows[i].mean_validation_loss)
    raw_best = rows[raw_index]
    threshold = raw_best.mean_validation_loss + raw_best.validation_se
    selected_index = next(
        i for i, row in enumerate(rows)
        if row.mean_validation_loss <= threshold
    )
    selected_k = rows[selected_index].k

    final_flow = _fit_fixed_k(
        field,
        source_time=source_time,
        target_time=target_time,
        z_grid=z_grid,
        genes=all_genes,
        k=selected_k,
        steps=config.optimizer_steps,
        learning_rate=config.learning_rate,
    )

    return FiniteTimeIntervalFit(
        flow=final_flow,
        selected_k=selected_k,
        raw_best_k=raw_best.k,
        cv=tuple(rows),
    )


def fit_finite_time_dynamics(
    field: ExpressionField,
    *,
    observed_times: Sequence[float],
    z_grid: Tensor,
    a: Tensor,
    config: DynamicsConfig | None = None,
) -> FiniteTimeTrajectory:
    """Fit adjacent finite-time flows and propagate anchors through observed ages."""
    config = DynamicsConfig() if config is None else config
    times = [float(t) for t in observed_times]
    if len(times) < 2 or any(b <= a_ for a_, b in zip(times[:-1], times[1:])):
        raise ValueError("observed_times must be strictly increasing with at least 2 entries")

    fits = tuple(
        fit_finite_time_interval(
            field,
            source_time=s,
            target_time=t,
            z_grid=z_grid,
            config=config,
        )
        for s, t in zip(times[:-1], times[1:])
    )
    flows = tuple(fit.flow for fit in fits)

    current = a.detach().clone()
    positions = [current.clone()]
    rhos = [torch.zeros_like(current)]
    velocities = []

    for fit in fits:
        flow = fit.flow
        next_position = flow(current)
        velocity = (next_position - current) / flow.dt
        psi0 = _fisher_embedding(field, current, flow.source_time)
        psi1 = _fisher_embedding(field, next_position, flow.target_time)
        chord = torch.linalg.vector_norm(psi1 - psi0, dim=1)
        positions.append(next_position.clone())
        rhos.append(rhos[-1] + chord.reshape_as(current))
        velocities.append(velocity.clone())
        current = next_position

    return FiniteTimeTrajectory(
        time=torch.as_tensor(times, dtype=a.dtype, device=a.device),
        position=torch.stack(positions, dim=0),
        rho=torch.stack(rhos, dim=0),
        interval_velocity=torch.stack(velocities, dim=0),
        flows=flows,
        interval_fits=fits,
    )


def infer_dynamics(
    field: ExpressionField,
    *,
    a: Tensor,
    observed_times: Sequence[float],
    z_grid: Tensor | None = None,
    config: DynamicsConfig | None = None,
) -> DynamicsResult:
    """Infer dynamics with finite-time matching by default or continuous RK4."""
    config = DynamicsConfig() if config is None else config
    times = [float(t) for t in observed_times]
    if len(times) < 2:
        raise ValueError("observed_times must contain at least two entries")

    if config.mode == "finite_time":
        if z_grid is None:
            raise ValueError("z_grid is required in finite_time mode")
        trajectory = fit_finite_time_dynamics(
            field,
            observed_times=times,
            z_grid=z_grid,
            a=a,
            config=config,
        )
        return DynamicsResult(mode="finite_time", finite_time=trajectory)

    from .trajectory import integrate_flow_and_rho

    trajectory = integrate_flow_and_rho(
        field,
        a,
        t0=times[0],
        t1=times[-1],
        n_steps=config.continuous_steps,
        eps=config.fisher_eps,
    )
    return DynamicsResult(mode="continuous", continuous=trajectory)
