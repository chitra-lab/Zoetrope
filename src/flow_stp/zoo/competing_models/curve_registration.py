"""Expression-first Fisher principal-curve registration estimator.

Pipeline:
    local population mean-expression estimates
      -> Poisson Fisher-flat coordinates Y = 2 sqrt(M)
      -> shared expression PCA
      -> tissue-specific principal curves
      -> spatially regularized curve coordinate u_i
      -> atlas-wide orientation synchronization
      -> monotone cross-tissue registration
      -> common coordinate z_i = h_i(u_i)

H1 labels are not used anywhere in fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import itertools

import numpy as np
import torch
import torch.nn.functional as F

from scipy import sparse
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import cg

from sklearn.decomposition import PCA
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge

from ..common import AtlasData, ZooResult, set_seed


@dataclass(frozen=True)
class Config:
    local_k: int = 16
    pseudocount: float = 0.5

    pca_rank: int = 12
    pca_fit_cells_per_tissue: int = 10000

    curve_knots: int = 12
    curve_degree: int = 3
    curve_rounds: int = 3
    curve_grid_size: int = 256
    curve_ridge: float = 1e-3

    spatial_graph_k: int = 8
    spatial_lambda: float = 0.3

    lower_quantile: float = 0.005
    upper_quantile: float = 0.995

    warp_segments: int = 8
    warp_roughness: float = 0.3
    identity_penalty: float = 0.03
    gauge_penalty: float = 10.0

    surface_degree_z: int = 6
    surface_degree_t: int = 3
    surface_ridge: float = 1e-4

    registration_steps: int = 2000
    registration_lr: float = 1e-2
    print_every: int = 250

    workers: int = 1
    seed: int = 20260908


def _np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _robust_unit_interval(x, lower, upper):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    q0, q1 = np.quantile(x, [float(lower), float(upper)])
    span = float(q1 - q0)
    if not np.isfinite(span) or span < 1e-10:
        raise RuntimeError("degenerate one-dimensional coordinate")
    return np.clip((x - q0) / span, 0.0, 1.0)


def _local_fisher_coordinates(tissue, *, k, pseudocount, workers):
    """Estimate local mean rates and return Y=2*sqrt(M_hat)."""
    xy = np.asarray(_np(tissue.coords), dtype=np.float64)
    counts = np.asarray(_np(tissue.counts), dtype=np.float32)
    exposure = np.exp(
        np.asarray(_np(tissue.log_library_size), dtype=np.float64)
    )

    n = len(xy)
    kk = min(max(int(k), 2), n)
    tree = cKDTree(xy)
    _, nbr = tree.query(xy, k=kk, workers=int(workers))
    if nbr.ndim == 1:
        nbr = nbr[:, None]

    rows = np.repeat(np.arange(n, dtype=np.int64), nbr.shape[1])
    cols = nbr.reshape(-1).astype(np.int64)
    W = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, n),
        dtype=np.float32,
    )

    numerator = W @ counts
    denominator = np.asarray(
        W @ exposure.astype(np.float32)
    ).reshape(-1)

    # One weak pseudocount per local population estimate, not per cell.
    rate = (numerator + float(pseudocount)) / np.maximum(
        denominator[:, None], 1e-8
    )
    rate = np.maximum(rate, 1e-8)
    return np.asarray(2.0 * np.sqrt(rate), dtype=np.float32)


def _fit_shared_pca(Y_by_tissue, *, rank, fit_cells_per_tissue, seed):
    rng = np.random.default_rng(int(seed))
    samples = []
    for Y in Y_by_tissue:
        m = min(len(Y), int(fit_cells_per_tissue))
        if m < len(Y):
            idx = rng.choice(len(Y), size=m, replace=False)
            samples.append(Y[idx])
        else:
            samples.append(Y)

    fit_Y = np.concatenate(samples, axis=0)
    rank = min(int(rank), fit_Y.shape[1], fit_Y.shape[0] - 1)
    pca = PCA(
        n_components=rank,
        svd_solver="randomized",
        random_state=int(seed),
    )
    pca.fit(fit_Y)
    scores = [
        np.asarray(pca.transform(Y), dtype=np.float32)
        for Y in Y_by_tissue
    ]
    return pca, scores


def _normalized_spatial_laplacian(xy, *, k, workers):
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    kk = min(max(int(k), 1) + 1, n)
    tree = cKDTree(xy)
    _, nbr = tree.query(xy, k=kk, workers=int(workers))
    if nbr.ndim == 1:
        nbr = nbr[:, None]
    nbr = nbr[:, 1:]

    rows = np.repeat(np.arange(n, dtype=np.int64), nbr.shape[1])
    cols = nbr.reshape(-1).astype(np.int64)
    W = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(n, n),
    )
    W = W.maximum(W.T)

    degree = np.asarray(W.sum(axis=1)).reshape(-1)
    invsqrt = np.zeros_like(degree)
    good = degree > 0
    invsqrt[good] = 1.0 / np.sqrt(degree[good])
    Dinv = sparse.diags(invsqrt)
    return sparse.eye(n, format="csr", dtype=np.float64) - Dinv @ W @ Dinv


def _smooth_scalar(target, L, lam):
    target = np.asarray(target, dtype=np.float64)
    if float(lam) <= 0:
        return target.copy()
    A = sparse.eye(len(target), format="csr", dtype=np.float64) + float(lam) * L
    result, info = cg(A, target, rtol=1e-6, atol=0.0, maxiter=1000)
    if info != 0:
        raise RuntimeError(f"spatial smoothing CG did not converge; info={info}")
    return result


def _initial_curve_coordinate(scores, *, lower, upper):
    centered = scores - scores.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return _robust_unit_interval(centered @ vt[0], lower, upper)


def _fit_spline_curve(u, scores, *, grid, n_knots, degree, ridge):
    transformer = SplineTransformer(
        n_knots=int(n_knots),
        degree=int(degree),
        include_bias=True,
        knots="quantile",
    )
    B = transformer.fit_transform(np.asarray(u)[:, None])
    model = Ridge(alpha=float(ridge), fit_intercept=False)
    model.fit(B, scores)
    curve = model.predict(transformer.transform(grid[:, None]))
    return np.asarray(curve, dtype=np.float64)


def _curve_self_intersection_score(curve, exclusion=8):
    D = cdist(curve, curve, metric="euclidean")
    ii = np.arange(len(curve))
    D[np.abs(ii[:, None] - ii[None, :]) <= int(exclusion)] = np.inf
    return float(np.min(D))


def _fit_principal_curve(scores, xy, config):
    grid = np.linspace(0.0, 1.0, int(config.curve_grid_size), dtype=np.float64)
    u = _initial_curve_coordinate(
        scores,
        lower=config.lower_quantile,
        upper=config.upper_quantile,
    )

    L = None
    if float(config.spatial_lambda) > 0:
        L = _normalized_spatial_laplacian(
            xy,
            k=config.spatial_graph_k,
            workers=config.workers,
        )

    for _ in range(int(config.curve_rounds)):
        curve = _fit_spline_curve(
            u,
            scores,
            grid=grid,
            n_knots=config.curve_knots,
            degree=config.curve_degree,
            ridge=config.curve_ridge,
        )
        tree = cKDTree(curve)
        _, nearest = tree.query(scores, k=1, workers=int(config.workers))
        u_projected = grid[np.asarray(nearest, dtype=np.int64)]
        u = (
            _smooth_scalar(u_projected, L, config.spatial_lambda)
            if L is not None
            else u_projected
        )
        u = _robust_unit_interval(
            u,
            config.lower_quantile,
            config.upper_quantile,
        )

    curve = _fit_spline_curve(
        u,
        scores,
        grid=grid,
        n_knots=config.curve_knots,
        degree=config.curve_degree,
        ridge=config.curve_ridge,
    )
    tree = cKDTree(curve)
    dist, _ = tree.query(scores, k=1, workers=int(config.workers))

    diagnostics = {
        "curve_reconstruction_mse": float(np.mean(np.asarray(dist) ** 2)),
        "spatial_roughness": (
            float(u @ (L @ u) / max(len(u), 1)) if L is not None else np.nan
        ),
        "curve_nonlocal_min_distance": _curve_self_intersection_score(curve),
    }
    return u, curve, grid, diagnostics


def _curve_direction(curve):
    n = len(curve)
    tail = max(3, int(round(0.05 * n)))
    d = curve[-tail:].mean(axis=0) - curve[:tail].mean(axis=0)
    norm = np.linalg.norm(d)
    return np.zeros_like(d) if norm < 1e-10 else d / norm


def _orientation_weights(times):
    unique_t = np.unique(times)
    time_rank = {float(t): i for i, t in enumerate(unique_t)}

    def weight(t1, t2):
        if np.isclose(t1, t2):
            return 4.0
        gap = abs(time_rank[float(t1)] - time_rank[float(t2)])
        return 1.0 if gap == 1 else 0.25

    return weight


def _synchronize_orientations(curves, times):
    n = len(curves)
    dirs = [_curve_direction(c) for c in curves]
    weight_fn = _orientation_weights(times)
    pair_data = []

    for i in range(n):
        for j in range(i + 1, n):
            ni = np.linalg.norm(dirs[i])
            nj = np.linalg.norm(dirs[j])
            cosine = (
                0.0
                if ni < 1e-10 or nj < 1e-10
                else float(np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0))
            )
            pair_data.append((i, j, cosine, weight_fn(float(times[i]), float(times[j]))))

    # Global sign is arbitrary; fix tissue 0 to +1 and solve the remaining
    # 2^(N-1) binary synchronization problem exactly (512 states for N=10).
    best_cost = np.inf
    best_signs = None
    for bits in itertools.product((-1, 1), repeat=n - 1):
        signs = np.asarray((1,) + bits, dtype=np.int8)
        cost = 0.0
        for i, j, cosine, w in pair_data:
            cost += w * (1.0 - float(signs[i] * signs[j]) * cosine)
        if cost < best_cost:
            best_cost = cost
            best_signs = signs

    return best_signs, float(best_cost)


def _cheb_numpy(x, degree):
    return np.polynomial.chebyshev.chebvander(
        np.asarray(x, dtype=np.float64), int(degree)
    )


def _cheb_torch(x, degree):
    values = [torch.ones_like(x)]
    if int(degree) >= 1:
        values.append(x)
    for _ in range(2, int(degree) + 1):
        values.append(2.0 * x * values[-1] - values[-2])
    return torch.stack(values, dim=-1)


def _initial_surface_beta(curves, times, grid, *, degree_z, degree_t, ridge):
    X_parts = []
    Y_parts = []
    bz = _cheb_numpy(2.0 * grid - 1.0, degree_z)

    for i, t in enumerate(times):
        bt = _cheb_numpy(np.asarray([2.0 * t - 1.0]), degree_t)[0]
        design = np.einsum("ma,b->mab", bz, bt).reshape(len(grid), -1)
        X_parts.append(design)
        Y_parts.append(curves[i])

    X = np.concatenate(X_parts, axis=0)
    Y = np.concatenate(Y_parts, axis=0)
    lhs = X.T @ X + float(ridge) * np.eye(X.shape[1], dtype=np.float64)
    beta = np.linalg.solve(lhs, X.T @ Y)
    return beta.reshape(int(degree_z) + 1, int(degree_t) + 1, curves.shape[-1])


def _fit_registration(curves, times, grid, config, *, device):
    curves_np = np.stack(curves, axis=0).astype(np.float32)
    n_tissues, n_grid, _ = curves_np.shape

    beta0 = _initial_surface_beta(
        curves_np,
        times,
        grid,
        degree_z=config.surface_degree_z,
        degree_t=config.surface_degree_t,
        ridge=config.surface_ridge,
    )

    curves_t = torch.tensor(curves_np, dtype=torch.float32, device=device)
    times_t = torch.tensor(times, dtype=torch.float32, device=device)
    ugrid_t = torch.tensor(grid, dtype=torch.float32, device=device)
    beta = torch.nn.Parameter(torch.tensor(beta0, dtype=torch.float32, device=device))
    theta = torch.nn.Parameter(
        torch.zeros(
            n_tissues,
            int(config.warp_segments),
            dtype=torch.float32,
            device=device,
        )
    )
    optimizer = torch.optim.Adam([beta, theta], lr=float(config.registration_lr))

    aa = torch.arange(
        int(config.surface_degree_z) + 1, dtype=torch.float32, device=device
    )
    bb = torch.arange(
        int(config.surface_degree_t) + 1, dtype=torch.float32, device=device
    )
    coefficient_weight = aa[:, None].square() + bb[None, :].square()
    history = []

    for step in range(int(config.registration_steps)):
        increments = F.softplus(theta) + 1e-4
        increments = increments / increments.sum(dim=1, keepdim=True)
        knots = torch.cat(
            [
                torch.zeros(n_tissues, 1, dtype=theta.dtype, device=device),
                torch.cumsum(increments, dim=1),
            ],
            dim=1,
        )
        h = F.interpolate(
            knots[:, None, :],
            size=n_grid,
            mode="linear",
            align_corners=True,
        )[:, 0, :]

        bz = _cheb_torch(2.0 * h - 1.0, config.surface_degree_z)
        bt = _cheb_torch(2.0 * times_t - 1.0, config.surface_degree_t)
        prediction = torch.einsum("ima,ib,abr->imr", bz, bt, beta)

        reconstruction = (prediction - curves_t).square().mean()
        log_d = torch.log(increments.clamp_min(1e-8))
        warp_rough = (
            (log_d[:, 1:] - log_d[:, :-1]).square().mean()
            if increments.shape[1] >= 2
            else torch.zeros((), device=device)
        )
        identity = (h - ugrid_t[None, :]).square().mean()
        gauge = (h.mean(dim=0) - ugrid_t).square().mean()
        surface_penalty = (
            beta.square() * coefficient_weight[:, :, None]
        ).mean()

        loss = (
            reconstruction
            + float(config.warp_roughness) * warp_rough
            + float(config.identity_penalty) * identity
            + float(config.gauge_penalty) * gauge
            + float(config.surface_ridge) * surface_penalty
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([beta, theta], max_norm=10.0)
        optimizer.step()

        if (
            step == 0
            or step == int(config.registration_steps) - 1
            or (int(config.print_every) > 0 and step % int(config.print_every) == 0)
        ):
            rec = {
                "step": float(step),
                "loss": float(loss.detach().cpu()),
                "reconstruction": float(reconstruction.detach().cpu()),
                "warp_roughness": float(warp_rough.detach().cpu()),
                "identity": float(identity.detach().cpu()),
                "gauge": float(gauge.detach().cpu()),
            }
            history.append(rec)

    with torch.no_grad():
        increments = F.softplus(theta) + 1e-4
        increments = increments / increments.sum(dim=1, keepdim=True)
        knots = torch.cat(
            [
                torch.zeros(n_tissues, 1, dtype=theta.dtype, device=device),
                torch.cumsum(increments, dim=1),
            ],
            dim=1,
        )
        h = F.interpolate(
            knots[:, None, :],
            size=n_grid,
            mode="linear",
            align_corners=True,
        )[:, 0, :]
        h_np = h.cpu().numpy()
        increments_np = increments.cpu().numpy()
        beta_np = beta.cpu().numpy()

    derivative = increments_np * int(config.warp_segments)
    diagnostics = {
        "warp_derivative_min": float(derivative.min()),
        "warp_derivative_max": float(derivative.max()),
        "warp_derivative_median": float(np.median(derivative)),
        "mean_absolute_warp": float(np.mean(np.abs(h_np - grid[None, :]))),
        "registration_reconstruction": history[-1]["reconstruction"],
        "registration_loss": history[-1]["loss"],
    }
    return h_np, increments_np, beta_np, history, diagnostics


def fit(
    atlas: AtlasData,
    config: Config = Config(),
    *,
    device: Optional[torch.device] = None,
) -> ZooResult:
    set_seed(config.seed)
    registration_device = torch.device("cpu") if device is None else torch.device(device)

    Y_by_tissue = [
        _local_fisher_coordinates(
            tissue,
            k=config.local_k,
            pseudocount=config.pseudocount,
            workers=config.workers,
        )
        for tissue in atlas.tissues
    ]

    pca, scores_by_tissue = _fit_shared_pca(
        Y_by_tissue,
        rank=config.pca_rank,
        fit_cells_per_tissue=config.pca_fit_cells_per_tissue,
        seed=config.seed,
    )
    del Y_by_tissue

    u_by_tissue = []
    curves = []
    curve_diagnostics = []
    common_grid = None

    for tissue, scores in zip(atlas.tissues, scores_by_tissue):
        u, curve, grid, diag = _fit_principal_curve(
            scores,
            np.asarray(_np(tissue.coords), dtype=np.float64),
            config,
        )
        if common_grid is None:
            common_grid = grid
        elif not np.allclose(common_grid, grid):
            raise RuntimeError("principal-curve grids differ unexpectedly")
        u_by_tissue.append(u)
        curves.append(curve)
        curve_diagnostics.append(diag)

    times = np.asarray([t.time for t in atlas.tissues], dtype=np.float64)
    signs, orientation_cost = _synchronize_orientations(curves, times)

    oriented_u = []
    oriented_curves = []
    for sign, u, curve in zip(signs, u_by_tissue, curves):
        if int(sign) == 1:
            oriented_u.append(np.asarray(u, dtype=np.float64))
            oriented_curves.append(np.asarray(curve, dtype=np.float64))
        else:
            oriented_u.append(1.0 - np.asarray(u, dtype=np.float64))
            oriented_curves.append(np.asarray(curve[::-1], dtype=np.float64))

    (
        warp_grid,
        warp_increments,
        surface_beta,
        registration_history,
        registration_diagnostics,
    ) = _fit_registration(
        oriented_curves,
        times,
        common_grid,
        config,
        device=registration_device,
    )

    z_by_tissue = [
        np.asarray(np.interp(u, common_grid, warp_grid[i]), dtype=np.float64)
        for i, u in enumerate(oriented_u)
    ]

    diagnostics = {
        "orientation_cost": float(orientation_cost),
        "orientation_signs": signs.tolist(),
        "curve": curve_diagnostics,
        "registration": registration_diagnostics,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_explained_variance_total": float(
            np.sum(pca.explained_variance_ratio_)
        ),
    }

    result = ZooResult(
        method="secondary_fisher_principal_curve",
        z_by_tissue=z_by_tissue,
        raw_coordinate_by_tissue=[
            np.asarray(x, dtype=np.float64) for x in oriented_u
        ],
        auxiliary={
            "diagnostics": diagnostics,
            "curve_grid": common_grid,
            "curves": oriented_curves,
            "orientation_signs": signs,
            "warp_grid": warp_grid,
            "warp_increments": warp_increments,
            "surface_beta": surface_beta,
            "pca_components": pca.components_,
            "pca_mean": pca.mean_,
        },
        history=registration_history,
    )
    result.check()
    return result


__all__ = ["Config", "fit"]
