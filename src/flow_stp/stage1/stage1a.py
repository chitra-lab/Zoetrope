"""Selected Stage 1a anatomical-axis implementation.

This module is the package implementation selected after benchmarking against
the Qian PFC atlas and the default positive-control simulator.

Coordinate model for tissue i:
    u_i(x, y) = raw spatial neural field
    z_local   = robust affine normalization of u_i on fixed spatial references
    z         = a_i + (b_i - a_i) * z_local

Coverage:
    a_i = missing_left_i
    b_i = 1 - missing_right_i

Expression model:
    log mu_gc = beta_g + F_g(z_c, t_i) + eta_gi + log_library_size_c

Geometry:
    * coverage-invariant submersion is imposed on z_local;
    * relative normal-scale variation is
          (grad z)^T H_z (grad z) / ||grad z||^3.

The historical names FlowSTPConfig / FlowSTPModel are retained so downstream
Stage 1b code can continue using result.model.coord_maps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import gc
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F


METHOD_VERSION = "primary_stage1a_v1_frozen"
Tensor = torch.Tensor


# =============================================================================
# Model
# =============================================================================

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int],
        activation: str,
    ):
        super().__init__()

        activations = {
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")

        Act = activations[activation]

        layers = []
        d = int(in_dim)
        for width in hidden:
            layers.extend([nn.Linear(d, int(width)), Act()])
            d = int(width)

        layers.append(nn.Linear(d, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass
class FlowSTPConfig:
    """Model configuration.

    Historical fields are retained for backward compatibility with older
    configuration construction. The old flow/q expression model is no longer
    used by Stage 1a.
    """

    num_slices: int
    num_genes: int

    coord_hidden: Tuple[int, ...] = (64, 64)
    decoder_hidden: Tuple[int, ...] = (128, 128)
    slice_effect_dim: int = 2

    coord_grad_min: float = 0.05
    coord_grad_eps: float = 1e-12
    scale_denominator_eps: float = 1e-8

    coverage_prior_sd: float = 0.05
    max_missing_each_end: float = 0.25
    initial_missing: float = 0.005

    # Historical compatibility fields.
    field_hidden: Tuple[int, ...] = (128, 128)
    n_flow_steps: int = 16
    coord_min_width: float = 0.05
    q_epsilon: float = 0.0
    clamp_eps: float = 1e-5
    coord_submersion_weight: float = 0.11364637
    coord_grad_samples_per_slice: Optional[int] = 128


class SliceCoordinateMaps(nn.Module):
    def __init__(self, config: FlowSTPConfig):
        super().__init__()

        self.num_slices = int(config.num_slices)
        self.coverage_prior_sd = float(config.coverage_prior_sd)
        self.max_missing_each_end = float(config.max_missing_each_end)

        self.maps = nn.ModuleList(
            [
                MLP(
                    2,
                    1,
                    config.coord_hidden,
                    "tanh",
                )
                for _ in range(self.num_slices)
            ]
        )

        self.missing_left = nn.Parameter(
            torch.full(
                (self.num_slices,),
                float(config.initial_missing),
            )
        )
        self.missing_right = nn.Parameter(
            torch.full(
                (self.num_slices,),
                float(config.initial_missing),
            )
        )

        # Detached robust affine gauge, refreshed by the runner.
        self.register_buffer(
            "gauge_low",
            torch.zeros(self.num_slices),
        )
        self.register_buffer(
            "gauge_high",
            torch.ones(self.num_slices),
        )

    def raw_one_slice(
        self,
        coords: Tensor,
        slice_id: int,
    ) -> Tensor:
        return self.maps[int(slice_id)](coords).squeeze(-1)

    def coverage_intervals(
        self,
    ) -> Tuple[Tensor, Tensor]:
        return (
            self.missing_left,
            1.0 - self.missing_right,
        )

    def intervals(
        self,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Stage 1b compatibility interface."""
        a, b = self.coverage_intervals()
        return a, b, b - a

    def local_and_global_one_slice(
        self,
        coords: Tensor,
        slice_id: int,
    ) -> Tuple[Tensor, Tensor]:
        u = self.raw_one_slice(coords, slice_id)

        low = self.gauge_low[int(slice_id)]
        high = self.gauge_high[int(slice_id)]
        span = torch.clamp(high - low, min=1e-4)

        z_local = (u - low) / span

        a, b = self.coverage_intervals()
        width = b[int(slice_id)] - a[int(slice_id)]
        z_global = a[int(slice_id)] + width * z_local

        return z_local, z_global

    def one_slice(
        self,
        coords: Tensor,
        slice_id: int,
    ) -> Tensor:
        return self.local_and_global_one_slice(
            coords,
            slice_id,
        )[1]

    def forward(
        self,
        coords: Tensor,
        slice_idx,
    ) -> Tensor:
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must have shape [B,2].")

        # Stage 1b sometimes calls coord_maps(coords, integer_slice_id).
        if isinstance(slice_idx, (int, np.integer)):
            return self.one_slice(
                coords,
                int(slice_idx),
            )[:, None]

        if not torch.is_tensor(slice_idx):
            slice_idx = torch.as_tensor(
                slice_idx,
                device=coords.device,
            )

        slice_idx = slice_idx.long().reshape(-1)

        if len(slice_idx) != len(coords):
            raise ValueError(
                "slice_idx must have one entry per coordinate."
            )

        out = torch.empty(
            len(coords),
            dtype=coords.dtype,
            device=coords.device,
        )

        for s_tensor in torch.unique(
            slice_idx,
            sorted=True,
        ):
            s = int(s_tensor.item())
            idx = torch.nonzero(
                slice_idx == s_tensor,
                as_tuple=False,
            ).flatten()

            out = out.index_copy(
                0,
                idx,
                self.one_slice(coords[idx], s),
            )

        return out[:, None]

    def coverage_penalty(self) -> Tensor:
        sd = max(self.coverage_prior_sd, 1e-12)

        return 0.5 * (
            (self.missing_left / sd).square()
            + (self.missing_right / sd).square()
        ).mean()

    @torch.no_grad()
    def project_endpoints_(self) -> None:
        self.missing_left.clamp_(
            0.0,
            self.max_missing_each_end,
        )
        self.missing_right.clamp_(
            0.0,
            self.max_missing_each_end,
        )


class FlowSTPModel(nn.Module):
    """Selected Stage 1a model; historical class name retained."""

    def __init__(self, config: FlowSTPConfig):
        super().__init__()

        self.config = config
        self.coordinate_maps = SliceCoordinateMaps(config)

        G = int(config.num_genes)

        self.beta = nn.Parameter(torch.zeros(G))

        self.decoder = MLP(
            2,
            G,
            config.decoder_hidden,
            "gelu",
        )

        rank = int(config.slice_effect_dim)

        if rank > 0:
            self.slice_embedding = nn.Embedding(
                int(config.num_slices),
                rank,
            )
            nn.init.zeros_(self.slice_embedding.weight)

            self.nuisance_loadings = nn.Parameter(
                0.01 * torch.randn(G, rank)
            )
        else:
            self.slice_embedding = None
            self.nuisance_loadings = None

    @property
    def coord_maps(self) -> SliceCoordinateMaps:
        """Backward-compatible alias used by Stage 1b."""
        return self.coordinate_maps

    def nuisance_effect_matrix(
        self,
    ) -> Optional[Tensor]:
        if (
            self.slice_embedding is None
            or self.nuisance_loadings is None
        ):
            return None

        effect = (
            self.slice_embedding.weight
            @ self.nuisance_loadings.T
        )

        # z-independent nuisance cannot absorb the shared intercept.
        return effect - effect.mean(
            dim=0,
            keepdim=True,
        )

    def nuisance_penalty(self) -> Tensor:
        effect = self.nuisance_effect_matrix()

        if effect is None:
            return torch.zeros(
                (),
                device=self.beta.device,
            )

        return effect.square().mean()

    def forward(
        self,
        *,
        coords: Tensor,
        slice_idx: Tensor,
        time: Tensor,
        log_library_size: Optional[Tensor] = None,
    ) -> Mapping[str, Tensor]:
        slice_idx = slice_idx.long().reshape(-1)
        time = time.reshape(-1).to(
            device=coords.device,
            dtype=coords.dtype,
        )

        z = self.coordinate_maps(
            coords,
            slice_idx,
        ).reshape(-1)

        log_mu = (
            self.beta[None, :]
            + self.decoder(
                torch.stack([z, time], dim=1)
            )
        )

        if log_library_size is not None:
            log_mu = (
                log_mu
                + log_library_size.reshape(-1, 1).to(
                    device=log_mu.device,
                    dtype=log_mu.dtype,
                )
            )

        nuisance = self.nuisance_effect_matrix()
        if nuisance is not None:
            log_mu = log_mu + nuisance[slice_idx]

        return {
            "z": z[:, None],
            "log_mu": log_mu,
        }

    @staticmethod
    def poisson_nll(
        counts: Tensor,
        log_mu: Tensor,
        *,
        full: bool = True,
    ) -> Tensor:
        return F.poisson_nll_loss(
            input=log_mu,
            target=counts,
            log_input=True,
            full=full,
            reduction="mean",
        )

    def coordinate_spatial_gradients(
        self,
        coords: Tensor,
        slice_idx: Tensor,
        *,
        samples_per_slice: Optional[int] = None,
        create_graph: bool = True,
    ):
        """Backward-compatible global-z gradient helper."""
        slice_idx = slice_idx.long().reshape(-1)

        selected = []
        if samples_per_slice is None or int(samples_per_slice) <= 0:
            selected = torch.arange(
                len(coords),
                device=coords.device,
            )
        else:
            for s in torch.unique(
                slice_idx,
                sorted=True,
            ):
                idx = torch.nonzero(
                    slice_idx == s,
                    as_tuple=False,
                ).flatten()

                if idx.numel() > int(samples_per_slice):
                    perm = torch.randperm(
                        idx.numel(),
                        device=idx.device,
                    )[: int(samples_per_slice)]
                    idx = idx[perm]

                selected.append(idx)

            selected = torch.cat(selected)

        coords_reg = (
            coords[selected]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        slice_reg = slice_idx[selected]

        z = self.coordinate_maps(
            coords_reg,
            slice_reg,
        ).reshape(-1)

        grad = torch.autograd.grad(
            z.sum(),
            coords_reg,
            create_graph=create_graph,
            retain_graph=create_graph,
            only_inputs=True,
        )[0]

        return (
            z[:, None],
            grad,
            coords_reg,
            slice_reg,
        )

    def coordinate_submersion_penalty(
        self,
        coords: Tensor,
        slice_idx: Tensor,
        *,
        min_grad: Optional[float] = None,
        samples_per_slice: Optional[int] = None,
        create_graph: bool = True,
    ):
        """Coverage-invariant submersion penalty on local observed z."""
        slice_idx = slice_idx.long().reshape(-1)

        threshold = (
            float(self.config.coord_grad_min)
            if min_grad is None
            else float(min_grad)
        )

        terms = []
        norms = []

        for s_tensor in torch.unique(
            slice_idx,
            sorted=True,
        ):
            s = int(s_tensor.item())
            idx = torch.nonzero(
                slice_idx == s_tensor,
                as_tuple=False,
            ).flatten()

            if (
                samples_per_slice is not None
                and int(samples_per_slice) > 0
                and idx.numel() > int(samples_per_slice)
            ):
                perm = torch.randperm(
                    idx.numel(),
                    device=idx.device,
                )[: int(samples_per_slice)]
                idx = idx[perm]

            x = (
                coords[idx]
                .detach()
                .clone()
                .requires_grad_(True)
            )

            z_local, _ = (
                self.coordinate_maps
                .local_and_global_one_slice(
                    x,
                    s,
                )
            )

            grad = torch.autograd.grad(
                z_local.sum(),
                x,
                create_graph=create_graph,
                retain_graph=create_graph,
                only_inputs=True,
            )[0]

            norm = torch.sqrt(
                grad.square().sum(dim=1)
                + float(self.config.coord_grad_eps)
            )

            deficit = F.relu(
                (threshold - norm)
                / max(threshold, 1e-12)
            )

            terms.append(
                deficit.square().mean()
            )
            norms.append(norm)

        all_norm = torch.cat(norms)

        stats = {
            "grad_norm_mean": all_norm.mean().detach(),
            "grad_norm_min": all_norm.min().detach(),
            "grad_violation_fraction": (
                (all_norm < threshold)
                .float()
                .mean()
                .detach()
            ),
        }

        return torch.stack(terms).mean(), stats

    def coordinate_geometry_penalties(
        self,
        coords_by_slice,
    ) -> Tuple[Tensor, Tensor]:
        """Submersion on local z and relative scale variation on global z."""
        sub_terms = []
        scale_terms = []

        for s, coords in enumerate(coords_by_slice):
            x = (
                coords.detach()
                .clone()
                .requires_grad_(True)
            )

            z_local, z_global = (
                self.coordinate_maps
                .local_and_global_one_slice(
                    x,
                    s,
                )
            )

            grad_local = torch.autograd.grad(
                z_local.sum(),
                x,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            local_norm = torch.sqrt(
                grad_local.square().sum(dim=1)
                + float(self.config.coord_grad_eps)
            )

            threshold = max(
                float(self.config.coord_grad_min),
                1e-12,
            )

            deficit = F.relu(
                (threshold - local_norm)
                / threshold
            )

            sub_terms.append(
                deficit.square().mean()
            )

            grad_global = torch.autograd.grad(
                z_global.sum(),
                x,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            grad_sq = grad_global.square().sum(dim=1)
            grad_norm = torch.sqrt(
                grad_sq
                + float(self.config.coord_grad_eps)
            )

            hessian_times_grad = torch.autograd.grad(
                0.5 * grad_sq.sum(),
                x,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            relative_scale = (
                (grad_global * hessian_times_grad)
                .sum(dim=1)
                / (
                    grad_norm.pow(3)
                    + float(
                        self.config.scale_denominator_eps
                    )
                )
            )

            scale_terms.append(
                relative_scale.square().mean()
            )

        return (
            torch.stack(sub_terms).mean(),
            torch.stack(scale_terms).mean(),
        )


# =============================================================================
# Public run config/result
# =============================================================================

@dataclass(frozen=True)
class Stage1ARunConfig:
    output_dir: Path = Path(
        "flow_stp_stage1a_direct_coverage_v1"
    )

    # Selected objective.
    submersion_weight: float = 0.11364637
    scale_variation_weight: float = 1e-3
    coverage_prior_weight: float = 1e-4
    coverage_prior_sd: float = 0.05
    nuisance_weight: float = 5e-3

    # Optimization.
    epochs: int = 1500
    batch_size_per_slice: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: Optional[float] = 5.0
    print_every: int = 100
    checkpoint_every: int = 100
    validation_fraction: float = 0.10
    eval_batch_size: int = 8192
    diagnostic_cells_per_slice: int = 256
    seed: int = 20260622
    deterministic: bool = True
    resume: bool = True

    # Geometry / coordinate map.
    coord_grad_min: float = 0.05
    coord_grad_samples_per_slice: Optional[int] = 128
    coord_grad_eps: float = 1e-12
    scale_denominator_eps: float = 1e-8
    coord_hidden: tuple[int, ...] = (64, 64)
    decoder_hidden: tuple[int, ...] = (128, 128)

    # Fixed spatially balanced robust gauge.
    gauge_reference_points: int = 4096
    gauge_spatial_bins: int = 36
    gauge_lower_quantile: float = 0.005
    gauge_upper_quantile: float = 0.995

    # Direct coverage endpoints.
    max_missing_each_end: float = 0.25
    initial_missing: float = 0.005

    # Real-data default selected in the PFC benchmark.
    # For the clean simulator, explicitly set slice_effect_dim=0.
    slice_effect_dim: int = 2

    use_library_size_offset: bool = True

    # Historical compatibility fields accepted but unused by the new
    # expression architecture.
    field_hidden: tuple[int, ...] = (128, 128)
    n_flow_steps: int = 16
    coord_min_width: float = 0.05


@dataclass
class Stage1AResult:
    model: FlowSTPModel
    summary: dict[str, Any]
    history: pd.DataFrame
    coordinate_diagnostics: pd.DataFrame
    cells: pd.DataFrame
    obs: Optional[pd.DataFrame]
    xy_norm_params: pd.DataFrame
    panel_df: pd.DataFrame
    normalized_times: np.ndarray
    output_dir: Path


# =============================================================================
# Runner helpers
# =============================================================================

def _set_all_seeds(
    seed: int,
    *,
    deterministic: bool = True,
) -> None:
    """Seed all stochastic backends and optionally require deterministic PyTorch ops."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Required by deterministic CUDA GEMM on supported CUDA versions.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        torch.use_deterministic_algorithms(False)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        getattr(torch.backends, "mps", None)
        is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def _to_dense_float_tensor(x) -> Tensor:
    if torch.is_tensor(x):
        return (
            x.detach()
            .cpu()
            .float()
            .contiguous()
        )

    if hasattr(x, "toarray"):
        x = x.toarray()

    return torch.as_tensor(
        np.asarray(x),
        dtype=torch.float32,
    ).contiguous()


def _maybe_pin(
    x: Tensor,
    device: torch.device,
) -> Tensor:
    x = x.contiguous().cpu()

    if device.type == "cuda":
        return x.pin_memory()

    return x


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(v)
            for v in value
        ]

    if isinstance(value, dict):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    return value


def _atomic_json_dump(
    payload,
    path: Path,
) -> None:
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        tmp,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            _json_safe(payload),
            handle,
            indent=2,
        )

    tmp.replace(path)


def _atomic_torch_save(
    payload,
    path: Path,
) -> None:
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(payload, tmp)
    tmp.replace(path)


def _load_torch(
    path: Path,
    map_location,
):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def _state_dict_cpu(model) -> dict:
    return {
        key: value.detach().cpu()
        for key, value
        in model.state_dict().items()
    }


def _optimizer_to_device(
    optimizer,
    device,
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _spatial_reference_indices(
    coords: Tensor,
    target_n: int,
    bins: int,
    seed: int,
) -> Tensor:
    """Spatially balanced fixed reference cells; not cell-density quantiles."""
    xy = coords.detach().cpu().numpy()
    n = len(xy)

    if n <= int(target_n):
        return torch.arange(
            n,
            dtype=torch.long,
        )

    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(
        hi - lo,
        1e-8,
    )
    unit = (xy - lo) / span

    bins = int(bins)

    bx = np.clip(
        np.floor(
            unit[:, 0] * bins
        ).astype(int),
        0,
        bins - 1,
    )
    by = np.clip(
        np.floor(
            unit[:, 1] * bins
        ).astype(int),
        0,
        bins - 1,
    )

    bin_id = bx * bins + by
    bin_counts = np.bincount(
        bin_id,
        minlength=bins * bins,
    )

    weights = (
        1.0
        / np.maximum(
            bin_counts[bin_id],
            1,
        )
    )
    weights /= weights.sum()

    rng = np.random.default_rng(seed)

    idx = np.sort(
        rng.choice(
            n,
            size=int(target_n),
            replace=False,
            p=weights,
        )
    )

    return torch.as_tensor(
        idx,
        dtype=torch.long,
    )


def _build_library_offsets(
    counts_by_slice,
    use_library_size_offset: bool,
):
    if not use_library_size_offset:
        return [
            torch.zeros(
                x.shape[0],
                dtype=torch.float32,
            )
            for x in counts_by_slice
        ]

    libraries = [
        x.sum(dim=1).float()
        for x in counts_by_slice
    ]

    all_library = torch.cat(libraries)
    positive = all_library[
        all_library > 0
    ]

    median_library = (
        positive.median()
        if positive.numel()
        else torch.tensor(1.0)
    )

    offsets = []

    for lib in libraries:
        sf = (
            lib
            / torch.clamp(
                median_library,
                min=1.0,
            )
        )

        sf = torch.clamp(
            sf,
            min=0.05,
            max=20.0,
        )

        offsets.append(
            torch.log(sf)
            .float()
            .contiguous()
        )

    return offsets


def _prepare_data(
    *,
    counts_by_slice,
    spatial_by_slice,
    panel_df,
    gene_names,
    config: Stage1ARunConfig,
    device: torch.device,
):
    n_slices = len(counts_by_slice)

    if n_slices == 0:
        raise ValueError(
            "counts_by_slice cannot be empty."
        )

    if len(spatial_by_slice) != n_slices:
        raise ValueError(
            "counts_by_slice and spatial_by_slice "
            "must have the same length."
        )

    n_genes = len(gene_names)

    counts = []
    coords = []
    xy_records = []

    for s, (c_raw, xy_raw) in enumerate(
        zip(
            counts_by_slice,
            spatial_by_slice,
        )
    ):
        c = _to_dense_float_tensor(c_raw)

        xy = np.asarray(
            xy_raw,
            dtype=np.float32,
        )[:, :2]

        if c.ndim != 2 or c.shape[1] != n_genes:
            raise ValueError(
                f"Slice {s}: invalid count shape {tuple(c.shape)}."
            )

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(
                f"Slice {s}: XY must have shape [N,2]."
            )

        if len(c) != len(xy):
            raise ValueError(
                f"Slice {s}: count/XY row mismatch."
            )

        if (c < 0).any():
            raise ValueError(
                f"Slice {s}: counts contain negatives."
            )

        if not np.isfinite(xy).all():
            raise ValueError(
                f"Slice {s}: XY contains non-finite values."
            )

        center = xy.mean(axis=0)
        scale = xy.std(axis=0)

        scale = np.where(
            scale > 1e-6,
            scale,
            1.0,
        ).astype(np.float32)

        xy_std = (
            (xy - center)
            / scale
        ).astype(np.float32)

        counts.append(c)
        coords.append(
            torch.from_numpy(xy_std)
        )

        xy_records.append(
            {
                "slice_id": s,
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "scale_x": float(scale[0]),
                "scale_y": float(scale[1]),
            }
        )

    panel = panel_df.copy()

    if "slice_id" not in panel.columns:
        panel["slice_id"] = np.arange(
            len(panel),
            dtype=int,
        )

    panel = (
        panel.sort_values("slice_id")
        .reset_index(drop=True)
    )

    expected = np.arange(
        n_slices,
        dtype=int,
    )

    if not np.array_equal(
        panel["slice_id"].to_numpy(
            dtype=int
        ),
        expected,
    ):
        raise ValueError(
            "panel_df must have exactly one row for "
            "slice_id 0..n_slices-1."
        )

    if "time" not in panel.columns:
        default_time_map = {
            "GW15": 15.0,
            "GW20": 20.0,
            "GW22": 22.0,
            "GW34": 34.0,
        }

        if "gw" in panel.columns:
            panel["time"] = (
                panel["gw"]
                .astype(str)
                .map(default_time_map)
            )
        elif "GW" in panel.columns:
            panel["time"] = (
                panel["GW"]
                .astype(str)
                .map(default_time_map)
            )
        else:
            raise KeyError(
                "panel_df needs time, gw, or GW."
            )

    raw_times = panel[
        "time"
    ].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(raw_times).all():
        raise ValueError(
            "panel_df time values must be finite."
        )

    t_min = float(raw_times.min())
    t_max = float(raw_times.max())

    normalized_times = (
        (raw_times - t_min)
        / max(
            t_max - t_min,
            1e-8,
        )
    ).astype(np.float32)

    log_library = _build_library_offsets(
        counts,
        bool(
            config.use_library_size_offset
        ),
    )

    split_rng = np.random.default_rng(
        int(config.seed) + 101
    )

    train_idx = []
    val_idx = []

    for matrix in counts:
        n = matrix.shape[0]
        perm = split_rng.permutation(n)

        n_val = int(
            round(
                float(config.validation_fraction)
                * n
            )
        )
        n_val = min(
            max(n_val, 1),
            max(n - 1, 1),
        )

        val = np.sort(
            perm[:n_val]
        )
        train = np.sort(
            perm[n_val:]
        )

        train_idx.append(
            torch.as_tensor(
                train,
                dtype=torch.long,
            )
        )
        val_idx.append(
            torch.as_tensor(
                val,
                dtype=torch.long,
            )
        )

    gauge_idx = [
        _spatial_reference_indices(
            coords[s],
            int(config.gauge_reference_points),
            int(config.gauge_spatial_bins),
            int(config.seed) + 2000 + s,
        )
        for s in range(n_slices)
    ]

    diag_rng = np.random.default_rng(
        int(config.seed) + 202
    )
    diagnostic_idx = []

    for s in range(n_slices):
        n = len(coords[s])
        k = min(
            int(config.diagnostic_cells_per_slice),
            n,
        )

        idx = np.sort(
            diag_rng.choice(
                n,
                size=k,
                replace=False,
            )
        )

        diagnostic_idx.append(
            torch.as_tensor(
                idx,
                dtype=torch.long,
            )
        )

    counts = [
        _maybe_pin(x, device)
        for x in counts
    ]
    coords = [
        _maybe_pin(
            x.float(),
            device,
        )
        for x in coords
    ]
    log_library = [
        _maybe_pin(
            x.float(),
            device,
        )
        for x in log_library
    ]

    return {
        "n_slices": n_slices,
        "n_genes": n_genes,
        "counts": counts,
        "coords": coords,
        "log_library": log_library,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "gauge_idx": gauge_idx,
        "diagnostic_idx": diagnostic_idx,
        "xy_norm_params": pd.DataFrame(
            xy_records
        ),
        "panel": panel,
        "raw_times": raw_times,
        "normalized_times": normalized_times,
        "time_min": t_min,
        "time_max": t_max,
    }


def _make_model_config(
    data,
    config: Stage1ARunConfig,
) -> FlowSTPConfig:
    return FlowSTPConfig(
        num_slices=int(
            data["n_slices"]
        ),
        num_genes=int(
            data["n_genes"]
        ),
        coord_hidden=tuple(
            config.coord_hidden
        ),
        decoder_hidden=tuple(
            config.decoder_hidden
        ),
        slice_effect_dim=int(
            config.slice_effect_dim
        ),
        coord_grad_min=float(
            config.coord_grad_min
        ),
        coord_grad_eps=float(
            config.coord_grad_eps
        ),
        scale_denominator_eps=float(
            config.scale_denominator_eps
        ),
        coverage_prior_sd=float(
            config.coverage_prior_sd
        ),
        max_missing_each_end=float(
            config.max_missing_each_end
        ),
        initial_missing=float(
            config.initial_missing
        ),
        field_hidden=tuple(
            config.field_hidden
        ),
        n_flow_steps=int(
            config.n_flow_steps
        ),
        coord_min_width=float(
            config.coord_min_width
        ),
        coord_submersion_weight=float(
            config.submersion_weight
        ),
        coord_grad_samples_per_slice=(
            config.coord_grad_samples_per_slice
        ),
    )


@torch.no_grad()
def _refresh_gauge(
    model: FlowSTPModel,
    data,
    config: Stage1ARunConfig,
    device: torch.device,
) -> None:
    model.eval()

    for s in range(
        data["n_slices"]
    ):
        idx = data[
            "gauge_idx"
        ][s]

        x = (
            data["coords"][s][idx]
            .to(
                device,
                non_blocking=(
                    device.type == "cuda"
                ),
            )
        )

        u = (
            model.coordinate_maps
            .raw_one_slice(
                x,
                s,
            )
        )

        low = torch.quantile(
            u,
            float(
                config.gauge_lower_quantile
            ),
        )
        high = torch.quantile(
            u,
            float(
                config.gauge_upper_quantile
            ),
        )

        if (
            not torch.isfinite(low)
            or not torch.isfinite(high)
            or high <= low + 1e-5
        ):
            raise RuntimeError(
                f"Degenerate robust gauge for slice {s}."
            )

        model.coordinate_maps.gauge_low[s] = low
        model.coordinate_maps.gauge_high[s] = high


def _balanced_batch(
    data,
    config: Stage1ARunConfig,
    generator: torch.Generator,
    device: torch.device,
):
    coords = []
    counts = []
    loglib = []
    slice_idx = []
    times = []

    for s in range(
        data["n_slices"]
    ):
        train = data[
            "train_idx"
        ][s]

        batch_n = min(
            int(
                config.batch_size_per_slice
            ),
            train.numel(),
        )

        pos = torch.randint(
            0,
            train.numel(),
            (batch_n,),
            generator=generator,
        )
        idx = train[pos]

        coords.append(
            data["coords"][s][idx]
        )
        counts.append(
            data["counts"][s][idx]
        )
        loglib.append(
            data["log_library"][s][idx]
        )
        slice_idx.append(
            torch.full(
                (batch_n,),
                s,
                dtype=torch.long,
            )
        )
        times.append(
            torch.full(
                (batch_n,),
                float(
                    data[
                        "normalized_times"
                    ][s]
                ),
                dtype=torch.float32,
            )
        )

    nb = device.type == "cuda"

    return {
        "coords": torch.cat(coords).to(
            device,
            non_blocking=nb,
        ),
        "counts": torch.cat(counts).to(
            device,
            non_blocking=nb,
        ),
        "log_library_size": (
            torch.cat(loglib).to(
                device,
                non_blocking=nb,
            )
        ),
        "slice_idx": torch.cat(slice_idx).to(
            device,
            non_blocking=nb,
        ),
        "time": torch.cat(times).to(
            device,
            non_blocking=nb,
        ),
    }


def _geometry_sample(
    data,
    config: Stage1ARunConfig,
    epoch: int,
    device: torch.device,
):
    rng = np.random.default_rng(
        int(config.seed)
        + 4000
        + int(epoch)
    )

    sample = []

    for s in range(
        data["n_slices"]
    ):
        n = len(
            data["coords"][s]
        )

        requested = (
            config.coord_grad_samples_per_slice
        )

        if (
            requested is None
            or int(requested) <= 0
        ):
            k = n
        else:
            k = min(
                int(requested),
                n,
            )

        idx = np.sort(
            rng.choice(
                n,
                size=k,
                replace=False,
            )
        )

        sample.append(
            data["coords"][s][
                torch.as_tensor(
                    idx,
                    dtype=torch.long,
                )
            ].to(
                device,
                non_blocking=(
                    device.type == "cuda"
                ),
            )
        )

    return sample


@torch.no_grad()
def _validation_nll(
    model,
    data,
    config,
    device,
) -> float:
    model.eval()

    total = 0.0
    elements = 0
    nb = device.type == "cuda"

    for s in range(
        data["n_slices"]
    ):
        indices = data[
            "val_idx"
        ][s]

        for start in range(
            0,
            len(indices),
            int(config.eval_batch_size),
        ):
            idx = indices[
                start:
                start
                + int(
                    config.eval_batch_size
                )
            ]

            n = len(idx)

            output = model(
                coords=data["coords"][s][idx].to(
                    device,
                    non_blocking=nb,
                ),
                slice_idx=torch.full(
                    (n,),
                    s,
                    dtype=torch.long,
                    device=device,
                ),
                time=torch.full(
                    (n,),
                    float(
                        data[
                            "normalized_times"
                        ][s]
                    ),
                    dtype=torch.float32,
                    device=device,
                ),
                log_library_size=(
                    data[
                        "log_library"
                    ][s][idx].to(
                        device,
                        non_blocking=nb,
                    )
                ),
            )

            target = data[
                "counts"
            ][s][idx].to(
                device,
                non_blocking=nb,
            )

            nll = F.poisson_nll_loss(
                input=output["log_mu"],
                target=target,
                log_input=True,
                full=True,
                reduction="sum",
            )

            total += float(
                nll.cpu()
            )
            elements += int(
                target.numel()
            )

    return total / max(
        elements,
        1,
    )


def _coordinate_diagnostics(
    model,
    data,
    config,
    device,
):
    """Memory-safe one-slice-at-a-time second-derivative audit."""
    model.eval()
    rows = []

    all_local = []
    all_global = []
    all_scale = []

    for s in range(
        data["n_slices"]
    ):
        idx = data[
            "diagnostic_idx"
        ][s]

        x = (
            data["coords"][s][idx]
            .to(
                device,
                non_blocking=(
                    device.type == "cuda"
                ),
            )
            .detach()
            .clone()
            .requires_grad_(True)
        )

        z_local, z_global = (
            model.coordinate_maps
            .local_and_global_one_slice(
                x,
                s,
            )
        )

        g_local = torch.autograd.grad(
            z_local.sum(),
            x,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        local_norm = torch.sqrt(
            g_local.square().sum(dim=1)
            + float(config.coord_grad_eps)
        )

        g = torch.autograd.grad(
            z_global.sum(),
            x,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        g2 = g.square().sum(dim=1)
        global_norm = torch.sqrt(
            g2
            + float(config.coord_grad_eps)
        )

        Hg = torch.autograd.grad(
            0.5 * g2.sum(),
            x,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]

        relative_scale = (
            (g * Hg).sum(dim=1)
            / (
                global_norm.pow(3)
                + float(
                    config.scale_denominator_eps
                )
            )
        )

        local_np = (
            local_norm.detach()
            .cpu()
            .numpy()
        )
        global_np = (
            global_norm.detach()
            .cpu()
            .numpy()
        )
        scale_np = (
            relative_scale.detach()
            .abs()
            .cpu()
            .numpy()
        )
        z_np = (
            z_global.detach()
            .cpu()
            .numpy()
        )
        g_np = (
            g.detach()
            .cpu()
            .numpy()
        )
        x_np = (
            x.detach()
            .cpu()
            .numpy()
        )

        neighbor_negative = np.nan
        neighbor_cos_p01 = np.nan

        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(x_np)
            _, nn_idx = tree.query(
                x_np,
                k=2,
            )
            neighbor = nn_idx[:, 1]

            denom = (
                np.linalg.norm(
                    g_np,
                    axis=1,
                )
                * np.linalg.norm(
                    g_np[neighbor],
                    axis=1,
                )
                + 1e-12
            )

            cosine = (
                (g_np * g_np[neighbor])
                .sum(axis=1)
                / denom
            )

            neighbor_negative = float(
                np.mean(
                    cosine < 0
                )
            )
            neighbor_cos_p01 = float(
                np.quantile(
                    cosine,
                    0.01,
                )
            )
        except Exception:
            pass

        with torch.no_grad():
            a, b, width = (
                model.coord_maps
                .intervals()
            )

            a_s = float(
                a[s].cpu()
            )
            b_s = float(
                b[s].cpu()
            )
            width_s = float(
                width[s].cpu()
            )

        rows.append(
            {
                "slice_id": s,
                "n_diagnostic": len(local_np),
                "interval_a": a_s,
                "interval_b": b_s,
                "interval_width": width_s,
                "z_min": float(
                    np.min(z_np)
                ),
                "z_p01": float(
                    np.quantile(
                        z_np,
                        0.01,
                    )
                ),
                "z_median": float(
                    np.median(z_np)
                ),
                "z_p99": float(
                    np.quantile(
                        z_np,
                        0.99,
                    )
                ),
                "z_max": float(
                    np.max(z_np)
                ),
                "local_grad_p05": float(
                    np.quantile(
                        local_np,
                        0.05,
                    )
                ),
                "fraction_local_grad_below_005":
                    float(
                        np.mean(
                            local_np < 0.05
                        )
                    ),
                "fraction_local_grad_below_001":
                    float(
                        np.mean(
                            local_np < 0.01
                        )
                    ),
                "global_grad_p05": float(
                    np.quantile(
                        global_np,
                        0.05,
                    )
                ),
                "relative_scale_abs_p95":
                    float(
                        np.quantile(
                            scale_np,
                            0.95,
                        )
                    ),
                "neighbor_gradient_cosine_p01":
                    neighbor_cos_p01,
                "neighbor_gradient_negative_fraction":
                    neighbor_negative,
            }
        )

        all_local.append(local_np)
        all_global.append(global_np)
        all_scale.append(scale_np)

        del (
            x,
            z_local,
            z_global,
            g_local,
            local_norm,
            g,
            g2,
            global_norm,
            Hg,
            relative_scale,
        )

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)

    local_all = np.concatenate(
        all_local
    )
    global_all = np.concatenate(
        all_global
    )
    scale_all = np.concatenate(
        all_scale
    )

    neighbor_values = frame[
        "neighbor_gradient_negative_fraction"
    ].to_numpy(dtype=float)

    summary = {
        "diagnostic_n": int(
            len(local_all)
        ),
        "coord_grad_min_target": float(
            config.coord_grad_min
        ),
        "local_grad_p05": float(
            np.quantile(
                local_all,
                0.05,
            )
        ),
        "local_grad_violation_fraction":
            float(
                np.mean(
                    local_all
                    < float(
                        config.coord_grad_min
                    )
                )
            ),
        "global_grad_p05": float(
            np.quantile(
                global_all,
                0.05,
            )
        ),
        "relative_scale_abs_p95": float(
            np.quantile(
                scale_all,
                0.95,
            )
        ),
        "mean_relative_scale_p95": float(
            frame[
                "relative_scale_abs_p95"
            ].mean()
        ),
        "worst_relative_scale_p95": float(
            frame[
                "relative_scale_abs_p95"
            ].max()
        ),
        "worst_fraction_local_grad_below_005":
            float(
                frame[
                    "fraction_local_grad_below_005"
                ].max()
            ),
        "worst_fraction_local_grad_below_001":
            float(
                frame[
                    "fraction_local_grad_below_001"
                ].max()
            ),
        "worst_neighbor_gradient_negative_fraction":
            (
                float(
                    np.nanmax(
                        neighbor_values
                    )
                )
                if np.isfinite(
                    neighbor_values
                ).any()
                else np.nan
            ),
        "mean_interval_width": float(
            frame[
                "interval_width"
            ].mean()
        ),

        # Compatibility aliases used by older notebooks.
        "grad_std_p05": float(
            np.quantile(
                global_all,
                0.05,
            )
        ),
        "grad_std_violation_fraction":
            float(
                np.mean(
                    local_all
                    < float(
                        config.coord_grad_min
                    )
                )
            ),
    }

    return summary, frame


@torch.no_grad()
def _infer_cells(
    model,
    data,
    spatial_by_slice,
    config,
    device,
) -> pd.DataFrame:
    model.eval()
    frames = []

    for s in range(
        data["n_slices"]
    ):
        coords_cpu = data[
            "coords"
        ][s]

        z_parts = []

        for start in range(
            0,
            len(coords_cpu),
            int(config.eval_batch_size),
        ):
            coords = coords_cpu[
                start:
                start
                + int(
                    config.eval_batch_size
                )
            ].to(
                device,
                non_blocking=(
                    device.type == "cuda"
                ),
            )

            z_parts.append(
                model.coord_maps(
                    coords,
                    s,
                )
                .reshape(-1)
                .cpu()
                .numpy()
            )

        z_slice = np.concatenate(
            z_parts
        )

        xy = np.asarray(
            spatial_by_slice[s],
            dtype=np.float32,
        )[:, :2]

        frames.append(
            pd.DataFrame(
                {
                    "slice_id": s,
                    "cell_index": np.arange(
                        len(z_slice),
                        dtype=int,
                    ),
                    "x": xy[:, 0],
                    "y": xy[:, 1],
                    "z_stage1a": z_slice,
                    "time_raw": float(
                        data["raw_times"][s]
                    ),
                    "time_normalized":
                        float(
                            data[
                                "normalized_times"
                            ][s]
                        ),
                }
            )
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# =============================================================================
# Public runner
# =============================================================================

def run_stage1a(
    *,
    counts_by_slice: Sequence[Any],
    spatial_by_slice: Sequence[Any],
    panel_df: pd.DataFrame,
    gene_names: Sequence[str],
    obs_df: Optional[pd.DataFrame] = None,
    config: Optional[Stage1ARunConfig] = None,
) -> Stage1AResult:
    """Train, resume, or reload the selected Stage 1a model."""
    config = (
        Stage1ARunConfig()
        if config is None
        else config
    )

    # Seed before model construction or any stochastic backend work.
    _set_all_seeds(
        int(config.seed),
        deterministic=bool(config.deterministic),
    )

    output_dir = (
        Path(config.output_dir)
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    tables_dir = (
        output_dir / "tables"
    )
    tables_dir.mkdir(
        exist_ok=True
    )

    device = _device()

    if device.type == "cuda" and not bool(config.deterministic):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print("Stage 1a method:", METHOD_VERSION)
    print("Output:", output_dir)
    print("Device:", device)

    data = _prepare_data(
        counts_by_slice=counts_by_slice,
        spatial_by_slice=spatial_by_slice,
        panel_df=panel_df,
        gene_names=gene_names,
        config=config,
        device=device,
    )

    data[
        "xy_norm_params"
    ].to_csv(
        tables_dir
        / "xy_normalization_params.csv",
        index=False,
    )

    data["panel"].to_csv(
        tables_dir / "panel_df.csv",
        index=False,
    )

    split_payload = {}

    for s in range(
        data["n_slices"]
    ):
        split_payload[
            f"train_{s}"
        ] = data["train_idx"][s].numpy()

        split_payload[
            f"validation_{s}"
        ] = data["val_idx"][s].numpy()

        split_payload[
            f"diagnostic_{s}"
        ] = data[
            "diagnostic_idx"
        ][s].numpy()

        split_payload[
            f"gauge_reference_{s}"
        ] = data[
            "gauge_idx"
        ][s].numpy()

    np.savez_compressed(
        tables_dir
        / "fixed_splits_diagnostics_and_gauge.npz",
        **split_payload,
    )

    manifest_path = (
        output_dir
        / "training_manifest.json"
    )

    if (
        config.resume
        and manifest_path.exists()
    ):
        with open(
            manifest_path,
            encoding="utf-8",
        ) as handle:
            existing = json.load(handle)

        if (
            existing.get(
                "method_version"
            )
            != METHOD_VERSION
        ):
            raise RuntimeError(
                "output_dir contains a different Stage 1a "
                "method. Choose a new output_dir."
            )

    manifest = {
        "method_version":
            METHOD_VERSION,
        "run_config":
            asdict(config),
        "num_slices":
            data["n_slices"],
        "num_genes":
            data["n_genes"],
        "time_min":
            data["time_min"],
        "time_max":
            data["time_max"],
        "raw_times":
            data["raw_times"],
        "normalized_times":
            data["normalized_times"],
    }

    _atomic_json_dump(
        manifest,
        manifest_path,
    )

    model_config = _make_model_config(
        data,
        config,
    )

    checkpoint_path = (
        output_dir
        / "checkpoint_last.pt"
    )
    final_model_path = (
        output_dir
        / "model_final.pt"
    )
    summary_path = (
        output_dir
        / "summary.json"
    )
    history_path = (
        output_dir
        / "training_history.csv"
    )
    diagnostics_path = (
        output_dir
        / "coordinate_diagnostics_by_slice.csv"
    )
    done_path = (
        output_dir
        / "done.json"
    )

    model = None
    history = []

    completed = (
        config.resume
        and done_path.exists()
        and final_model_path.exists()
        and summary_path.exists()
    )

    if completed:
        print(
            "Run already complete; loading saved model."
        )

        payload = _load_torch(
            final_model_path,
            map_location="cpu",
        )

        if (
            payload.get(
                "method_version"
            )
            != METHOD_VERSION
        ):
            raise RuntimeError(
                "Saved final model is not the selected "
                "Stage 1a method."
            )

        saved_model_config = (
            FlowSTPConfig(
                **payload[
                    "model_config"
                ]
            )
        )

        model = (
            FlowSTPModel(
                saved_model_config
            )
            .to(device)
        )

        model.load_state_dict(
            payload[
                "model_state_dict"
            ],
            strict=True,
        )

        _refresh_gauge(
            model,
            data,
            config,
            device,
        )

        if history_path.exists():
            history = (
                pd.read_csv(
                    history_path
                )
                .to_dict(
                    orient="records"
                )
            )

    else:
        _set_all_seeds(
            int(config.seed),
            deterministic=bool(config.deterministic),
        )

        model = (
            FlowSTPModel(
                model_config
            )
            .to(device)
        )

        _refresh_gauge(
            model,
            data,
            config,
            device,
        )

        endpoint_params = [
            model.coordinate_maps
            .missing_left,

            model.coordinate_maps
            .missing_right,
        ]

        endpoint_ids = {
            id(p)
            for p in endpoint_params
        }

        network_params = [
            p
            for p in model.parameters()
            if (
                p.requires_grad
                and id(p)
                not in endpoint_ids
            )
        ]

        optimizer = torch.optim.AdamW(
            [
                {
                    "params":
                        network_params,
                    "weight_decay":
                        float(
                            config.weight_decay
                        ),
                },
                {
                    "params":
                        endpoint_params,
                    # No hidden L2 pull toward full coverage.
                    "weight_decay":
                        0.0,
                },
            ],
            lr=float(
                config.learning_rate
            ),
        )

        generator = torch.Generator(
            device="cpu"
        )
        generator.manual_seed(
            int(config.seed)
            + 303
        )

        start_epoch = 1
        elapsed_before = 0.0

        if (
            config.resume
            and checkpoint_path.exists()
        ):
            checkpoint = _load_torch(
                checkpoint_path,
                map_location=device,
            )

            if (
                checkpoint.get(
                    "method_version"
                )
                != METHOD_VERSION
            ):
                raise RuntimeError(
                    "Checkpoint belongs to a different "
                    "Stage 1a method."
                )

            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ],
                strict=True,
            )
            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )
            _optimizer_to_device(
                optimizer,
                device,
            )

            if (
                "batch_generator_state"
                in checkpoint
            ):
                generator.set_state(
                    checkpoint[
                        "batch_generator_state"
                    ].cpu()
                )

            history = list(
                checkpoint.get(
                    "history",
                    [],
                )
            )
            start_epoch = int(
                checkpoint["epoch"]
            ) + 1
            elapsed_before = float(
                checkpoint.get(
                    "elapsed_sec",
                    0.0,
                )
            )

            print(
                f"Resuming at epoch {start_epoch}."
            )

        started = time.time()

        for epoch in range(
            start_epoch,
            int(config.epochs) + 1,
        ):
            model.train()

            _refresh_gauge(
                model,
                data,
                config,
                device,
            )

            batch = _balanced_batch(
                data,
                config,
                generator,
                device,
            )

            geometry_coords = (
                _geometry_sample(
                    data,
                    config,
                    epoch,
                    device,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(
                coords=batch["coords"],
                slice_idx=batch[
                    "slice_idx"
                ],
                time=batch["time"],
                log_library_size=batch[
                    "log_library_size"
                ],
            )

            data_nll = F.poisson_nll_loss(
                input=output["log_mu"],
                target=batch["counts"],
                log_input=True,
                full=True,
                reduction="mean",
            )

            (
                submersion,
                scale_variation,
            ) = (
                model
                .coordinate_geometry_penalties(
                    geometry_coords
                )
            )

            coverage_prior = (
                model.coordinate_maps
                .coverage_penalty()
            )

            nuisance = (
                model.nuisance_penalty()
            )

            total = (
                data_nll
                + float(
                    config.submersion_weight
                )
                * submersion
                + float(
                    config.scale_variation_weight
                )
                * scale_variation
                + float(
                    config.coverage_prior_weight
                )
                * coverage_prior
                + float(
                    config.nuisance_weight
                )
                * nuisance
            )

            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"Non-finite Stage 1a loss at epoch {epoch}."
                )

            total.backward()

            if (
                config.grad_clip_norm
                is not None
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(
                        config.grad_clip_norm
                    ),
                )

            optimizer.step()
            model.coordinate_maps.project_endpoints_()

            if (
                epoch == 1
                or epoch
                % int(
                    config.print_every
                )
                == 0
                or epoch
                == int(config.epochs)
            ):
                with torch.no_grad():
                    _, _, width = (
                        model.coord_maps
                        .intervals()
                    )

                    mean_width = float(
                        width.mean().cpu()
                    )

                row = {
                    "epoch": epoch,
                    "data_nll": float(
                        data_nll.detach().cpu()
                    ),
                    "submersion": float(
                        submersion.detach().cpu()
                    ),
                    "scale_variation": float(
                        scale_variation
                        .detach()
                        .cpu()
                    ),
                    "coverage_prior": float(
                        coverage_prior
                        .detach()
                        .cpu()
                    ),
                    "nuisance": float(
                        nuisance.detach().cpu()
                    ),
                    "loss": float(
                        total.detach().cpu()
                    ),
                    "mean_coverage_width":
                        mean_width,
                }

                history.append(row)

                print(
                    f"epoch {epoch:4d} | "
                    f"NLL={row['data_nll']:.5f} | "
                    f"sub={row['submersion']:.5f} | "
                    f"scale={row['scale_variation']:.4f} | "
                    f"coverage={mean_width:.3f}"
                )

            if (
                epoch
                % int(
                    config.checkpoint_every
                )
                == 0
                or epoch
                == int(config.epochs)
            ):
                elapsed = (
                    elapsed_before
                    + time.time()
                    - started
                )

                _atomic_torch_save(
                    {
                        "method_version":
                            METHOD_VERSION,
                        "epoch":
                            epoch,
                        "model_config":
                            asdict(
                                model_config
                            ),
                        "model_state_dict":
                            _state_dict_cpu(
                                model
                            ),
                        "optimizer_state_dict":
                            optimizer.state_dict(),
                        "batch_generator_state":
                            generator.get_state(),
                        "history":
                            history,
                        "elapsed_sec":
                            elapsed,
                    },
                    checkpoint_path,
                )

        _refresh_gauge(
            model,
            data,
            config,
            device,
        )

        validation_nll = (
            _validation_nll(
                model,
                data,
                config,
                device,
            )
        )

        (
            diagnostic_summary,
            diagnostic_frame,
        ) = (
            _coordinate_diagnostics(
                model,
                data,
                config,
                device,
            )
        )

        diagnostic_frame.to_csv(
            diagnostics_path,
            index=False,
        )

        with torch.no_grad():
            a, b, width = (
                model.coord_maps
                .intervals()
            )

            intervals = (
                data["panel"]
                .copy()
            )
            intervals[
                "interval_a"
            ] = a.cpu().numpy()
            intervals[
                "interval_b"
            ] = b.cpu().numpy()
            intervals[
                "interval_width"
            ] = width.cpu().numpy()

        intervals.to_csv(
            tables_dir
            / "coverage_intervals.csv",
            index=False,
        )

        elapsed_total = (
            elapsed_before
            + time.time()
            - started
        )

        summary = {
            "method_version":
                METHOD_VERSION,
            "validation_poisson_nll":
                float(validation_nll),
            "training_elapsed_sec":
                float(elapsed_total),
            "submersion_weight":
                float(
                    config.submersion_weight
                ),
            "scale_variation_weight":
                float(
                    config.scale_variation_weight
                ),
            "coverage_prior_weight":
                float(
                    config.coverage_prior_weight
                ),
            "coverage_prior_sd":
                float(
                    config.coverage_prior_sd
                ),
            "nuisance_weight":
                float(
                    config.nuisance_weight
                ),
            "slice_effect_dim":
                int(
                    config.slice_effect_dim
                ),
            **diagnostic_summary,
        }

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
        )

        _atomic_json_dump(
            summary,
            summary_path,
        )

        _atomic_torch_save(
            {
                "method_version":
                    METHOD_VERSION,
                "model_config":
                    asdict(
                        model_config
                    ),
                "model_state_dict":
                    _state_dict_cpu(
                        model
                    ),
                "summary":
                    summary,
            },
            final_model_path,
        )

        _atomic_json_dump(
            {
                "method_version":
                    METHOD_VERSION,
                "completed":
                    True,
                "validation_poisson_nll":
                    validation_nll,
            },
            done_path,
        )

    _refresh_gauge(
        model,
        data,
        config,
        device,
    )

    with open(
        summary_path,
        encoding="utf-8",
    ) as handle:
        summary = json.load(handle)

    history_df = (
        pd.read_csv(history_path)
        if history_path.exists()
        else pd.DataFrame()
    )

    diagnostic_df = (
        pd.read_csv(diagnostics_path)
        if diagnostics_path.exists()
        else pd.DataFrame()
    )

    cells = _infer_cells(
        model,
        data,
        spatial_by_slice,
        config,
        device,
    )

    cells.to_csv(
        tables_dir
        / "stage1a_all_cells.csv.gz",
        index=False,
        compression="gzip",
    )

    obs_result = None

    if obs_df is not None:
        metadata = obs_df.copy()

        metadata[
            "slice_id"
        ] = metadata[
            "slice_id"
        ].astype(int)

        metadata[
            "cell_index"
        ] = (
            metadata.groupby(
                "slice_id",
                sort=False,
            )
            .cumcount()
            .astype(int)
        )

        obs_result = metadata.merge(
            cells[
                [
                    "slice_id",
                    "cell_index",
                    "z_stage1a",
                    "time_raw",
                    "time_normalized",
                ]
            ],
            on=[
                "slice_id",
                "cell_index",
            ],
            how="left",
            validate="one_to_one",
        )

        obs_result.to_csv(
            tables_dir
            / "obs_with_stage1a.csv.gz",
            index=False,
            compression="gzip",
        )

    return Stage1AResult(
        model=model,
        summary=summary,
        history=history_df,
        coordinate_diagnostics=
            diagnostic_df,
        cells=cells,
        obs=obs_result,
        xy_norm_params=
            data[
                "xy_norm_params"
            ].copy(),
        panel_df=
            data["panel"].copy(),
        normalized_times=
            np.asarray(
                data[
                    "normalized_times"
                ],
                dtype=np.float32,
            ).copy(),
        output_dir=output_dir,
    )


__all__ = [
    "FlowSTPConfig",
    "FlowSTPModel",
    "Stage1ARunConfig",
    "Stage1AResult",
    "run_stage1a",
]
