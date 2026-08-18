"""Synthetic identifiability test for registered transcriptomic arclength.

Truth construction
------------------
The material/reference coordinate is ``a in (0,1)``.  We prescribe the exact
boundary-preserving logistic flow

    F(a,t) = sigmoid(logit(a) + k t),

so the Eulerian velocity is

    V(z,t) = k z (1-z),

which vanishes exactly at z=0 and z=1.  Its inverse is

    A(z,t) = sigmoid(logit(z) - k t).

The first half of the genes are *transport genes*: their log-mean expression is
an anatomical pattern evaluated at A(z,t).  Therefore they move with the flow
and satisfy M_t + V M_z = 0 exactly.

The second half are *development genes*: they are spatially constant but change
smoothly in time.  Consequently they contribute no M_z and cannot be explained
away by registration.  The registered residual is therefore known exactly and
its norm integrates to a known rho(t).

This deliberately easy decomposition is an identifiability positive control,
not a realistic biological simulator.  Once this passes, later tests can mix
spatial and developmental effects within genes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..simulated_atlas_positive_control.geometry import (
    bend_tissue_coordinates,
    sample_intrinsic_coordinates,
)


_EPS = 1e-7


@dataclass(frozen=True)
class ArclengthSyntheticConfig:
    n_timepoints: int = 6
    n_cells_per_tissue: int = 2500
    n_genes: int = 20
    tissue_width: float = 4.0
    tissue_height: float = 1.0
    maximum_bend_deg: float = 120.0
    flow_strength: float = 1.25
    base_log_mean: float = 2.0
    spatial_amplitude: float = 0.75
    developmental_amplitude: float = 0.65
    library_size_log_sd: float = 0.15
    random_seed: int = 1729

    def validate(self) -> None:
        if self.n_timepoints < 3:
            raise ValueError("n_timepoints must be at least 3")
        if self.n_cells_per_tissue <= 0:
            raise ValueError("n_cells_per_tissue must be positive")
        if self.n_genes < 4 or self.n_genes % 2:
            raise ValueError("n_genes must be even and at least 4")
        if self.tissue_width <= 0 or self.tissue_height <= 0:
            raise ValueError("tissue dimensions must be positive")
        if self.maximum_bend_deg < 0:
            raise ValueError("maximum_bend_deg must be nonnegative")
        if not np.isfinite(self.flow_strength):
            raise ValueError("flow_strength must be finite")

    @property
    def tissue_times(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.n_timepoints)

    @property
    def bend_angles_deg(self) -> np.ndarray:
        return np.linspace(0.0, self.maximum_bend_deg, self.n_timepoints)

    @property
    def n_transport_genes(self) -> int:
        return self.n_genes // 2

    @property
    def n_development_genes(self) -> int:
        return self.n_genes - self.n_transport_genes


@dataclass(frozen=True)
class ArclengthSyntheticAtlas:
    obs: pd.DataFrame
    counts: pd.DataFrame
    expected_mean: pd.DataFrame
    log_mean: pd.DataFrame
    gene_metadata: pd.DataFrame
    config: ArclengthSyntheticConfig


def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def _logit(x):
    x = np.clip(np.asarray(x, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(x) - np.log1p(-x)


def logistic_flow(a, t, flow_strength: float = 1.25):
    """Exact F(a,t)=sigmoid(logit(a)+k t), fixing endpoints 0 and 1."""
    a, t = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(t, dtype=float))
    out = _sigmoid(_logit(a) + float(flow_strength) * t)
    out = np.where(a <= 0.0, 0.0, out)
    out = np.where(a >= 1.0, 1.0, out)
    return out


def inverse_logistic_flow(z, t, flow_strength: float = 1.25):
    """Exact inverse A(z,t)=F_t^{-1}(z)."""
    z, t = np.broadcast_arrays(np.asarray(z, dtype=float), np.asarray(t, dtype=float))
    out = _sigmoid(_logit(z) - float(flow_strength) * t)
    out = np.where(z <= 0.0, 0.0, out)
    out = np.where(z >= 1.0, 1.0, out)
    return out


def logistic_velocity(z, t=None, flow_strength: float = 1.25):
    """Eulerian velocity V(z,t)=k z(1-z); independent of t here."""
    z = np.asarray(z, dtype=float)
    return float(flow_strength) * z * (1.0 - z)


def _spatial_program(a: np.ndarray, j: int, n: int) -> np.ndarray:
    """Smooth bounded transport-gene patterns on the material coordinate."""
    phase = 2.0 * np.pi * j / max(n, 1)
    if j % 3 == 0:
        return np.sin(np.pi * a + phase)
    if j % 3 == 1:
        center = 0.15 + 0.70 * ((j + 0.5) / n)
        width = 0.10 + 0.025 * (j % 4)
        return 2.0 * np.exp(-0.5 * ((a - center) / width) ** 2) - 0.65
    return np.cos(2.0 * np.pi * a + phase)


def _development_shape(t: np.ndarray, j: int, n: int) -> np.ndarray:
    """Low-order smooth temporal programs representable by the model basis."""
    u = (j + 1) / (n + 1)
    if j % 4 == 0:
        return t
    if j % 4 == 1:
        return t**2
    if j % 4 == 2:
        return t**3
    # Monotone cubic smootherstep-like polynomial without a separate spline knot.
    return (3.0 - 2.0 * u) * t**2 + (2.0 * u - 2.0) * t**3


def _development_shape_derivative(t: np.ndarray, j: int, n: int) -> np.ndarray:
    u = (j + 1) / (n + 1)
    if j % 4 == 0:
        return np.ones_like(t)
    if j % 4 == 1:
        return 2.0 * t
    if j % 4 == 2:
        return 3.0 * t**2
    return 2.0 * (3.0 - 2.0 * u) * t + 3.0 * (2.0 * u - 2.0) * t**2


def true_log_mean(z, t, config: ArclengthSyntheticConfig) -> np.ndarray:
    """Evaluate the exact G-dimensional M(z,t) on paired arrays z,t."""
    config.validate()
    z, t = np.broadcast_arrays(np.asarray(z, dtype=float), np.asarray(t, dtype=float))
    flat_z = z.reshape(-1)
    flat_t = t.reshape(-1)
    a = inverse_logistic_flow(flat_z, flat_t, config.flow_strength)
    out = np.full((len(flat_z), config.n_genes), config.base_log_mean, dtype=float)

    for g in range(config.n_transport_genes):
        out[:, g] += config.spatial_amplitude * _spatial_program(
            a, g, config.n_transport_genes
        )

    for j in range(config.n_development_genes):
        g = config.n_transport_genes + j
        sign = -1.0 if j % 2 else 1.0
        gene_scale = 0.70 + 0.60 * (j + 1) / config.n_development_genes
        out[:, g] += (
            sign
            * config.developmental_amplitude
            * gene_scale
            * _development_shape(flat_t, j, config.n_development_genes)
        )
    return out.reshape(z.shape + (config.n_genes,))


def developmental_derivative(t, config: ArclengthSyntheticConfig) -> np.ndarray:
    """Exact registered residual dM/dt along material paths, shape [...,G]."""
    config.validate()
    t = np.asarray(t, dtype=float)
    flat_t = t.reshape(-1)
    out = np.zeros((len(flat_t), config.n_genes), dtype=float)
    for j in range(config.n_development_genes):
        g = config.n_transport_genes + j
        sign = -1.0 if j % 2 else 1.0
        gene_scale = 0.70 + 0.60 * (j + 1) / config.n_development_genes
        out[:, g] = (
            sign
            * config.developmental_amplitude
            * gene_scale
            * _development_shape_derivative(flat_t, j, config.n_development_genes)
        )
    return out.reshape(t.shape + (config.n_genes,))


def true_q(t, config: ArclengthSyntheticConfig, gene_weights=None) -> np.ndarray:
    """Exact registered transcriptomic speed ||DM/Dt||_W."""
    r = developmental_derivative(t, config)
    if gene_weights is None:
        w = np.ones(config.n_genes, dtype=float)
    else:
        w = np.asarray(gene_weights, dtype=float).reshape(config.n_genes)
    return np.sqrt(np.einsum("...g,g,...g->...", r, w, r))


def true_rho(t, config: ArclengthSyntheticConfig, gene_weights=None, n_quad: int = 1025) -> np.ndarray:
    """Numerically integrate exact q from 0 to each requested time."""
    t = np.asarray(t, dtype=float)
    flat = t.reshape(-1)
    out = np.empty_like(flat)
    for i, tv in enumerate(flat):
        if tv <= 0:
            out[i] = 0.0
            continue
        grid = np.linspace(0.0, float(tv), n_quad)
        out[i] = np.trapz(true_q(grid, config, gene_weights), grid)
    return out.reshape(t.shape)


def generate_arclength_synthetic_atlas(
    config: ArclengthSyntheticConfig = ArclengthSyntheticConfig(),
) -> ArclengthSyntheticAtlas:
    """Generate bent tissues, exact M(z,t), library sizes, and Poisson counts."""
    config.validate()
    rng = np.random.default_rng(config.random_seed)
    n_total = config.n_timepoints * config.n_cells_per_tissue

    tissue_index = np.repeat(np.arange(config.n_timepoints), config.n_cells_per_tissue)
    time = np.repeat(config.tissue_times, config.n_cells_per_tissue)
    bends = np.repeat(config.bend_angles_deg, config.n_cells_per_tissue)

    s_intrinsic, z_current = sample_intrinsic_coordinates(
        n_total, width=config.tissue_width, rng=rng
    )
    x, y = bend_tissue_coordinates(
        s_intrinsic,
        z_current,
        width=config.tissue_width,
        height=config.tissue_height,
        bend_angle_deg=bends,
    )

    a_true = inverse_logistic_flow(z_current, time, config.flow_strength)
    M = true_log_mean(z_current, time, config)

    # Cell-specific depth factors make the raw-count test closer to the packaged
    # atlas simulation while remaining exactly removable by a known offset.
    log_library_size = rng.normal(0.0, config.library_size_log_sd, size=n_total)
    mean = np.exp(M + log_library_size[:, None])
    counts = rng.poisson(mean).astype(np.int32, copy=False)

    q = true_q(time, config)
    rho = true_rho(time, config)
    V = logistic_velocity(z_current, time, config.flow_strength)

    cell_ids = np.asarray([
        f"tissue_{ti:02d}_cell_{ci:05d}"
        for ti in range(config.n_timepoints)
        for ci in range(config.n_cells_per_tissue)
    ], dtype=object)
    tissue_ids = np.asarray([f"tissue_{i:02d}" for i in tissue_index], dtype=object)

    obs = pd.DataFrame({
        "cell_id": cell_ids,
        "tissue_id": tissue_ids,
        "tissue_index": tissue_index,
        "time": time,
        "bend_angle_deg": bends,
        "s_intrinsic": s_intrinsic,
        "z_true": z_current,
        "a_true": a_true,
        "x_extrinsic": x,
        "y_extrinsic": y,
        "log_library_size": log_library_size,
        "V_true": V,
        "q_true": q,
        "rho_true": rho,
    }).set_index("cell_id", drop=False)

    names = [f"transport_{g:02d}" for g in range(config.n_transport_genes)] + [
        f"development_{g:02d}" for g in range(config.n_development_genes)
    ]
    gene_metadata = pd.DataFrame({
        "gene": names,
        "program": ["transport"] * config.n_transport_genes
        + ["development"] * config.n_development_genes,
    }).set_index("gene", drop=False)

    return ArclengthSyntheticAtlas(
        obs=obs,
        counts=pd.DataFrame(counts, index=cell_ids, columns=names),
        expected_mean=pd.DataFrame(mean, index=cell_ids, columns=names),
        log_mean=pd.DataFrame(M, index=cell_ids, columns=names),
        gene_metadata=gene_metadata,
        config=config,
    )
