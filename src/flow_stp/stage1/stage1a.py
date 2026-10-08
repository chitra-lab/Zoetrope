"""Selected Stage 1: shared coordinate (time-only or tissue-conditioned), global gauge,
log E(z,t) Poisson decoder, and differentiably derived Fisher-flat Psi.
Historic direct-Psi and independent models are under stage1/archive/.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import gc
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F


METHOD_VERSION = "stage1a_shared_logE_global_gauge_v1"
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

    Only the selected shared-coordinate log-expression model is configurable.
    """

    num_slices: int
    num_genes: int
    seed: int = 20260622

    # Resolved coordinate choice: time-only or tissue-conditioned.
    coordinate_decoder: str = "shared_time"
    gauge_mode: str = "global"
    slice_times: Tuple[float, ...] = ()

    coord_hidden: Tuple[int, ...] = (64, 64)
    decoder_hidden: Tuple[int, ...] = (128, 128)

    coord_grad_min: float = 0.05
    coord_grad_eps: float = 1e-12
    scale_denominator_eps: float = 1e-8


class SliceCoordinateMaps(nn.Module):
    def __init__(self, config: FlowSTPConfig):
        super().__init__()

        self.num_slices = int(config.num_slices)
        self.coordinate_decoder = str(config.coordinate_decoder)
        self.gauge_mode = str(config.gauge_mode)
        if self.gauge_mode != "global":
            raise ValueError("The selected coordinate model requires the global gauge.")

        slice_times = tuple(float(t) for t in config.slice_times)
        if len(slice_times) != self.num_slices:
            raise ValueError(
                "FlowSTPConfig.slice_times must contain one normalized "
                "time per slice."
            )
        self.register_buffer(
            "slice_times",
            torch.as_tensor(slice_times, dtype=torch.float32),
        )

        if self.coordinate_decoder in ("shared_time", "shared_time_tissue"):
            input_dim = 3 + (self.num_slices if self.coordinate_decoder == "shared_time_tissue" else 0)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(config.seed) + 704113)
                self.shared_map = MLP(input_dim, 1, config.coord_hidden, "tanh")
                if self.coordinate_decoder == "shared_time_tissue":
                    torch.manual_seed(int(config.seed) + 704113)
                    base = MLP(3, 1, config.coord_hidden, "tanh")
                    with torch.no_grad():
                        for dst, src in zip(self.shared_map.net, base.net):
                            if isinstance(dst, nn.Linear):
                                if dst.in_features != src.in_features:
                                    dst.weight.zero_()
                                    dst.weight[:, :src.in_features].copy_(src.weight)
                                else:
                                    dst.weight.copy_(src.weight)
                                dst.bias.copy_(src.bias)
        else:
            raise ValueError(
                "coordinate_decoder must be 'shared_time' or 'shared_time_tissue'; "
                f"got {self.coordinate_decoder!r}."
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
        s = int(slice_id)

        if self.shared_map is None:
            raise RuntimeError("shared_time coordinate decoder is missing.")

        time_value = self.slice_times[s].to(
            device=coords.device,
            dtype=coords.dtype,
        )
        time_column = (
            time_value.reshape(1, 1)
            .expand(coords.shape[0], 1)
        )
        inputs = [coords, time_column]
        if self.coordinate_decoder == "shared_time_tissue":
            tissue_id = coords.new_zeros((len(coords), self.num_slices))
            tissue_id[:, s] = 1.0
            inputs.append(tissue_id)
        return self.shared_map(torch.cat(inputs, dim=1)).squeeze(-1)


    def one_slice(self, coords: Tensor, slice_id: int) -> Tensor:
        raw = self.raw_one_slice(coords, slice_id)
        low = self.gauge_low[int(slice_id)]
        span = torch.clamp(self.gauge_high[int(slice_id)] - low, min=1e-4)
        return (raw - low) / span

    def forward(
        self,
        coords: Tensor,
        slice_idx,
    ) -> Tensor:
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must have shape [B,2].")

        # Evaluate one physical tissue or a batch of tissue indices.
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


class FlowSTPModel(nn.Module):
    """Shared-coordinate model: fit log E, derive Fisher-flat Psi."""

    def __init__(self, config: FlowSTPConfig):
        super().__init__()

        self.config = config
        self.coordinate_maps = SliceCoordinateMaps(config)

        G = int(config.num_genes)

        self.log_expression_baseline = nn.Parameter(torch.zeros(G))

        # Decoder initialization is independent of coordinate input width.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(config.seed) + 71718)
            self.decoder = MLP(2, G, config.decoder_hidden, "gelu")

    def log_expression(self, z: Tensor, time: Tensor) -> Tensor:
        """Learned eta(z,t)=log E(z,t), excluding cell library exposure."""
        zc = z.reshape(-1).to(
            device=self.log_expression_baseline.device,
            dtype=self.log_expression_baseline.dtype,
        )
        tc = time.reshape(-1).to(
            device=self.log_expression_baseline.device,
            dtype=self.log_expression_baseline.dtype,
        )
        if zc.shape[0] != tc.shape[0]:
            raise ValueError("z and time must have matching rows")
        return self.log_expression_baseline[None, :] + self.decoder(
            torch.stack([zc, tc], dim=1)
        )

    def expression(self, z: Tensor, time: Tensor) -> Tensor:
        return torch.exp(self.log_expression(z, time))

    def fisher_flat(self, z: Tensor, time: Tensor) -> Tensor:
        """Derived positive Fisher embedding Psi=2 sqrt(E)."""
        return 2.0 * torch.exp(0.5 * self.log_expression(z, time))

    def psi(self, z: Tensor, time: Tensor) -> Tensor:
        return self.fisher_flat(z, time)


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

        eta = self.log_expression(z, time)
        log_mu = eta
        expression = torch.exp(eta)
        psi = 2.0 * torch.exp(0.5 * eta)

        if log_library_size is not None:
            log_mu = (
                log_mu
                + log_library_size.reshape(-1, 1).to(
                    device=log_mu.device,
                    dtype=log_mu.dtype,
                )
            )

        return {
            "z": z[:, None],
            "psi": psi,
            "expression": expression,
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


# =============================================================================
# Public run config/result
# =============================================================================

@dataclass(frozen=True)
class Stage1ARunConfig:
    output_dir: Path = Path(
        "flow_stp_stage1a_shared_logE_global_gauge_v1"
    )

    # Selected log-expression Poisson model.

    # Optimization.
    epochs: int = 1500
    batch_size_per_slice: int = 1024
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    grad_clip_norm: Optional[float] = 5.0
    print_every: int = 100
    checkpoint_every: int = 100
    validation_fraction: float = 0.10
    eval_batch_size: int = 8192
    diagnostic_cells_per_slice: int = 256
    seed: int = 20260622
    resume: bool = True

    # Active shared-coordinate model chooses tissue conditioning automatically
    # from the number of physical slices at each observed time.
    coordinate_decoder: str = "shared_auto"
    gauge_mode: str = "global"

    coord_grad_min: float = 0.05
    coord_grad_eps: float = 1e-12
    scale_denominator_eps: float = 1e-8
    coord_hidden: tuple[int, ...] = (64, 64)
    decoder_hidden: tuple[int, ...] = (128, 128)

    # Fixed spatially balanced robust gauge.
    gauge_reference_points: int = 4096
    gauge_spatial_bins: int = 36
    gauge_lower_quantile: float = 0.005
    gauge_upper_quantile: float = 0.995


    # Essential observation-model correction selected by the PFC ablation.
    use_library_size_offset: bool = True


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

def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _load_torch(path: Path, map_location):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("method_version") == METHOD_VERSION:
        cfg = payload.get("model_config", {})
        for name in ("slice_effect_dim", "max_missing_each_end", "initial_missing",
                     "coord_submersion_weight"):
            if cfg.get(name, 0) != 0:
                raise RuntimeError(f"Checkpoint has unsupported {name}; use its historical code.")
        payload["model_config"] = {
            key: value for key, value in cfg.items()
            if key in FlowSTPConfig.__dataclass_fields__
        }
        state = payload.get("model_state_dict", {})
        for key in ("coordinate_maps.missing_left", "coordinate_maps.missing_right"):
            if key in state:
                if torch.count_nonzero(state[key]).item():
                    raise RuntimeError("Checkpoint has nonzero coverage endpoints.")
                del state[key]
        opt = payload.get("optimizer_state_dict")
        if opt and len(opt["param_groups"]) == 2:
            old_endpoints = opt["param_groups"].pop()
            for pid in old_endpoints["params"]:
                opt["state"].pop(pid, None)
    return payload


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
        seed=int(config.seed),
        coordinate_decoder=(
            "shared_time_tissue" if len(np.unique(data["raw_times"])) < data["n_slices"]
            else "shared_time"
        ),
        gauge_mode=str(
            config.gauge_mode
        ),
        slice_times=tuple(
            float(t)
            for t in data["normalized_times"]
        ),
        coord_hidden=tuple(
            config.coord_hidden
        ),
        decoder_hidden=tuple(
            config.decoder_hidden
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
    )


@torch.no_grad()
def _refresh_gauge(
    model: FlowSTPModel,
    data,
    config: Stage1ARunConfig,
    device: torch.device,
) -> None:
    """Refresh one detached affine ruler from pooled spatial references."""
    model.eval()
    if config.gauge_mode != "global":
        raise ValueError("The selected model requires the global gauge.")

    raw_parts = []

    for s in range(data["n_slices"]):
        idx = data["gauge_idx"][s]
        x = data["coords"][s][idx].to(
            device,
            non_blocking=(device.type == "cuda"),
        )
        raw_parts.append(
            model.coordinate_maps.raw_one_slice(
                x,
                s,
            )
        )

    pooled = torch.cat(raw_parts, dim=0)
    low = torch.quantile(
        pooled,
        float(config.gauge_lower_quantile),
    )
    high = torch.quantile(
        pooled,
        float(config.gauge_upper_quantile),
    )

    if (
        not torch.isfinite(low)
        or not torch.isfinite(high)
        or high <= low + 1e-5
    ):
        raise RuntimeError(
            "Degenerate global robust coordinate gauge."
        )

    model.coordinate_maps.gauge_low.copy_(
        low.expand_as(model.coordinate_maps.gauge_low)
    )
    model.coordinate_maps.gauge_high.copy_(
        high.expand_as(model.coordinate_maps.gauge_high)
    )

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

        # Retain established diagnostic column names; both now refer to the
        # same global coordinate because coverage endpoints were removed.
        z_local, z_global = (
            (model.coordinate_maps.one_slice(x, s),) * 2
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

        rows.append(
            {
                "slice_id": s,
                "n_diagnostic": len(local_np),
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
                model.coordinate_maps(
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

    if not config.use_library_size_offset:
        raise ValueError("The selected model requires the library-size offset.")

    coordinate_choice = (
        str(config.coordinate_decoder),
        str(config.gauge_mode),
    )
    if coordinate_choice != ("shared_auto", "global"):
        raise ValueError(
            "The frozen runner requires shared_auto/global; "
            "Historical alternatives remain available in stage1/archive. "
            f"Received {coordinate_choice!r}."
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

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print("Stage 1a method:", METHOD_VERSION)
    print(
        "Coordinate decoder:",
        config.coordinate_decoder,
        "| gauge:",
        config.gauge_mode,
    )
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

        if (
            saved_model_config.coordinate_decoder
            != model_config.coordinate_decoder
            or saved_model_config.gauge_mode
            != str(config.gauge_mode)
        ):
            raise RuntimeError(
                "Saved Stage-1 model uses coordinate choice "
                f"{saved_model_config.coordinate_decoder}/"
                f"{saved_model_config.gauge_mode}, but the requested "
                f"run uses {config.coordinate_decoder}/{config.gauge_mode}."
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
            int(config.seed)
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

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
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

            checkpoint_model_config = FlowSTPConfig(
                **checkpoint["model_config"]
            )
            if (
                checkpoint_model_config.coordinate_decoder
                != model_config.coordinate_decoder
                or checkpoint_model_config.gauge_mode
                != str(config.gauge_mode)
            ):
                raise RuntimeError(
                    "Checkpoint uses coordinate choice "
                    f"{checkpoint_model_config.coordinate_decoder}/"
                    f"{checkpoint_model_config.gauge_mode}, but the "
                    f"requested run uses {config.coordinate_decoder}/"
                    f"{config.gauge_mode}."
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

            total = data_nll

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
                row = {
                    "epoch": epoch,
                    "data_nll": float(data_nll.detach().cpu()),
                    "loss": float(total.detach().cpu()),
                }

                history.append(row)

                print(f"epoch {epoch:4d} | NLL={row['data_nll']:.5f}")

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
            "expression_parameterization":
                "log_E",
            "coordinate_decoder":
                str(model.config.coordinate_decoder),
            "coordinate_gauge":
                str(config.gauge_mode),
            "coordinate_hidden":
                list(config.coord_hidden),
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
    summary["coordinate_decoder"] = model.config.coordinate_decoder

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

