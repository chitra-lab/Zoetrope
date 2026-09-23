"""Shared infrastructure for the Stage-1 anatomical-process estimator zoo.

The zoo deliberately keeps a common input/output contract so that methods which
make different statistical approximations can be compared against the same
estimand: a one-dimensional common process coordinate for each tissue.

The code in this package is research-oriented. It is intentionally explicit
about the geometry and keeps method-specific assumptions local to each module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import math
import os
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

Tensor = torch.Tensor


def set_seed(seed: int) -> None:
    """Seed retained zoo estimators and require deterministic PyTorch operations."""
    # Must be set before CUDA BLAS is initialized.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def as_float_tensor(x: Any) -> Tensor:
    if torch.is_tensor(x):
        return x.detach().float().contiguous()
    if hasattr(x, "toarray"):
        x = x.toarray()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32).contiguous()


@dataclass
class TissueData:
    """One tissue slice in the common zoo format."""

    coords: Tensor
    raw_coords: Tensor
    counts: Tensor
    time: float
    log_library_size: Tensor
    center: Tensor
    scale: Tensor
    name: str

    @property
    def n_cells(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.counts.shape[1])


@dataclass
class AtlasData:
    tissues: list[TissueData]
    gene_names: tuple[str, ...]

    @property
    def n_tissues(self) -> int:
        return len(self.tissues)

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)

    @property
    def times(self) -> Tensor:
        return torch.tensor([t.time for t in self.tissues], dtype=torch.float32)

    def to(self, device: torch.device | str) -> "AtlasData":
        device = torch.device(device)
        tissues = []
        for t in self.tissues:
            tissues.append(
                TissueData(
                    coords=t.coords.to(device),
                    raw_coords=t.raw_coords.to(device),
                    counts=t.counts.to(device),
                    time=t.time,
                    log_library_size=t.log_library_size.to(device),
                    center=t.center.to(device),
                    scale=t.scale.to(device),
                    name=t.name,
                )
            )
        return AtlasData(tissues=tissues, gene_names=self.gene_names)


def prepare_atlas(
    *,
    counts_by_slice: Sequence[Any],
    spatial_by_slice: Sequence[Any],
    times: Sequence[float],
    gene_names: Optional[Sequence[str]] = None,
    tissue_names: Optional[Sequence[str]] = None,
    standardize_coords: bool = True,
    preserve_physical_metric: bool = True,
    use_library_size_offset: bool = True,
) -> AtlasData:
    """Convert common Stage-1 arrays into the zoo representation.

    Times are normalized to [0, 1] exactly once here. Raw physical spatial
    coordinates are retained in addition to fitting coordinates.
    """

    if len(counts_by_slice) == 0:
        raise ValueError("counts_by_slice cannot be empty")
    if len(spatial_by_slice) != len(counts_by_slice):
        raise ValueError("counts_by_slice and spatial_by_slice must match")
    if len(times) != len(counts_by_slice):
        raise ValueError("times must have one entry per tissue")

    counts = [as_float_tensor(x).cpu() for x in counts_by_slice]
    raw_xy = [as_float_tensor(x).cpu()[:, :2] for x in spatial_by_slice]
    g = int(counts[0].shape[1])

    for i, (c, xy) in enumerate(zip(counts, raw_xy)):
        if c.ndim != 2 or c.shape[1] != g:
            raise ValueError(f"tissue {i}: counts must have shape [N,{g}]")
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"tissue {i}: coordinates must have shape [N,2]")
        if c.shape[0] != xy.shape[0]:
            raise ValueError(f"tissue {i}: count/coordinate row mismatch")
        if torch.any(c < 0):
            raise ValueError(f"tissue {i}: counts must be non-negative")
        if not torch.isfinite(xy).all():
            raise ValueError(f"tissue {i}: coordinates contain non-finite values")

    if gene_names is None:
        gene_names = [f"gene_{j}" for j in range(g)]
    if len(gene_names) != g:
        raise ValueError("gene_names has the wrong length")

    if tissue_names is None:
        tissue_names = [f"tissue_{i}" for i in range(len(counts))]
    if len(tissue_names) != len(counts):
        raise ValueError("tissue_names must have one entry per tissue")

    raw_times = np.asarray(times, dtype=np.float32)
    if not np.isfinite(raw_times).all():
        raise ValueError("times must be finite")
    t0 = float(raw_times.min())
    t1 = float(raw_times.max())
    denom = max(t1 - t0, 1e-8)
    norm_times = (raw_times - t0) / denom

    libraries = [c.sum(dim=1) for c in counts]
    positive = torch.cat([x[x > 0] for x in libraries])
    median_lib = positive.median() if positive.numel() else torch.tensor(1.0)
    median_lib = torch.clamp(median_lib, min=1.0)

    tissues: list[TissueData] = []
    for i, (c, xy, lib) in enumerate(zip(counts, raw_xy, libraries)):
        center = xy.mean(dim=0)
        axis_scale = xy.std(dim=0, unbiased=False)
        axis_scale = torch.where(axis_scale > 1e-6, axis_scale, torch.ones_like(axis_scale))
        if preserve_physical_metric:
            scalar = torch.sqrt(torch.mean(axis_scale.square()))
            scale = torch.full_like(axis_scale, torch.clamp(scalar, min=1e-6))
        else:
            scale = axis_scale
        fit_xy = (xy - center) / scale if standardize_coords else xy.clone()

        if use_library_size_offset:
            sf = torch.clamp(lib / median_lib, min=0.05, max=20.0)
            log_lib = torch.log(sf)
        else:
            log_lib = torch.zeros_like(lib)

        tissues.append(
            TissueData(
                coords=fit_xy.float().contiguous(),
                raw_coords=xy.float().contiguous(),
                counts=c.float().contiguous(),
                time=float(norm_times[i]),
                log_library_size=log_lib.float().contiguous(),
                center=center.float(),
                scale=scale.float(),
                name=str(tissue_names[i]),
            )
        )

    return AtlasData(tissues=tissues, gene_names=tuple(str(x) for x in gene_names))


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int] = (64, 64),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        acts = {"tanh": nn.Tanh, "gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}
        if activation not in acts:
            raise ValueError(f"unknown activation {activation!r}")
        Act = acts[activation]
        layers: list[nn.Module] = []
        d = int(in_dim)
        for width in hidden:
            layers.extend([nn.Linear(d, int(width)), Act()])
            d = int(width)
        layers.append(nn.Linear(d, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CoordinateMaps(nn.Module):
    """Smooth tissue-specific maps into a shared interval."""

    def __init__(
        self,
        n_tissues: int,
        hidden: Sequence[int] = (64, 64),
        max_missing_each_end: float = 0.30,
        initial_missing: float = 0.01,
    ) -> None:
        super().__init__()
        self.n_tissues = int(n_tissues)
        self.max_missing_each_end = float(max_missing_each_end)
        self.maps = nn.ModuleList([MLP(2, 1, hidden=hidden, activation="tanh") for _ in range(self.n_tissues)])

        frac = np.clip(float(initial_missing) / max(float(max_missing_each_end), 1e-8), 1e-5, 1.0 - 1e-5)
        init = float(math.log(frac / (1.0 - frac)))
        self.left_logits = nn.Parameter(torch.full((self.n_tissues,), init))
        self.right_logits = nn.Parameter(torch.full((self.n_tissues,), init))

    def coverage_intervals(self) -> tuple[Tensor, Tensor]:
        left = self.max_missing_each_end * torch.sigmoid(self.left_logits)
        right_missing = self.max_missing_each_end * torch.sigmoid(self.right_logits)
        right = 1.0 - right_missing
        return left, right

    def local_one(self, coords: Tensor, tissue_id: int) -> Tensor:
        raw = self.maps[int(tissue_id)](coords).squeeze(-1)
        return torch.sigmoid(raw)

    def one(self, coords: Tensor, tissue_id: int) -> Tensor:
        local = self.local_one(coords, tissue_id)
        left, right = self.coverage_intervals()
        a = left[int(tissue_id)]
        b = right[int(tissue_id)]
        return a + (b - a) * local

    def forward(self, coords: Tensor, tissue_idx: Tensor) -> Tensor:
        tissue_idx = tissue_idx.long().reshape(-1)
        if len(tissue_idx) != len(coords):
            raise ValueError("tissue_idx must have one entry per coordinate")
        out = torch.empty(len(coords), device=coords.device, dtype=coords.dtype)
        for tid_tensor in torch.unique(tissue_idx, sorted=True):
            tid = int(tid_tensor.item())
            idx = torch.nonzero(tissue_idx == tid_tensor, as_tuple=False).flatten()
            out[idx] = self.one(coords[idx], tid)
        return out

    def coverage_prior(self, sd: float = 0.08) -> Tensor:
        left, right = self.coverage_intervals()
        missing = torch.cat([left, 1.0 - right])
        return 0.5 * (missing / max(float(sd), 1e-8)).square().mean()


class LocalMeanField(nn.Module):
    """Smooth per-tissue log mean-expression rate M_i(s)."""

    def __init__(self, n_genes: int, hidden: Sequence[int] = (96, 96)) -> None:
        super().__init__()
        self.net = MLP(2, int(n_genes), hidden=hidden, activation="silu")
        self.intercept = nn.Parameter(torch.zeros(int(n_genes)))

    def forward(self, coords: Tensor) -> Tensor:
        return self.intercept[None, :] + self.net(coords)


class SharedLogRateDecoder(nn.Module):
    def __init__(self, n_genes: int, hidden: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(int(n_genes)))
        self.net = MLP(2, int(n_genes), hidden=hidden, activation="gelu")

    def forward(self, z: Tensor, time: Tensor) -> Tensor:
        x = torch.stack([z.reshape(-1), time.reshape(-1)], dim=1)
        return self.beta[None, :] + self.net(x)


class SharedSqrtDecoder(nn.Module):
    """Shared expression map in Poisson Fisher-flat coordinates Y=2 sqrt(E)."""

    def __init__(self, n_genes: int, hidden: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        self.net = MLP(2, int(n_genes), hidden=hidden, activation="gelu")
        self.base = nn.Parameter(torch.ones(int(n_genes)))

    def forward(self, z: Tensor, time: Tensor) -> Tensor:
        x = torch.stack([z.reshape(-1), time.reshape(-1)], dim=1)
        return F.softplus(self.base[None, :] + self.net(x)) + 1e-4


class LowRankSliceEffect(nn.Module):
    def __init__(self, n_tissues: int, n_genes: int, rank: int = 2) -> None:
        super().__init__()
        rank = int(rank)
        self.rank = rank
        if rank > 0:
            self.embedding = nn.Embedding(int(n_tissues), rank)
            nn.init.zeros_(self.embedding.weight)
            self.loadings = nn.Parameter(0.01 * torch.randn(int(n_genes), rank))
        else:
            self.embedding = None
            self.loadings = None

    def matrix(self) -> Optional[Tensor]:
        if self.embedding is None or self.loadings is None:
            return None
        effect = self.embedding.weight @ self.loadings.T
        return effect - effect.mean(dim=0, keepdim=True)

    def penalty(self) -> Tensor:
        mat = self.matrix()
        if mat is None:
            p = next(self.parameters(), None)
            device = p.device if p is not None else torch.device("cpu")
            return torch.zeros((), device=device)
        return mat.square().mean()


@dataclass
class ZooResult:
    method: str
    z_by_tissue: list[np.ndarray]
    raw_coordinate_by_tissue: Optional[list[np.ndarray]] = None
    model: Optional[nn.Module] = None
    auxiliary: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)

    def check(self) -> None:
        for i, z in enumerate(self.z_by_tissue):
            arr = np.asarray(z)
            if arr.ndim != 1:
                raise ValueError(f"tissue {i}: z must be one-dimensional")
            if not np.isfinite(arr).all():
                raise ValueError(f"tissue {i}: z contains non-finite values")


@dataclass
class OptimConfig:
    epochs: int = 800
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: Optional[float] = 5.0
    seed: int = 20260908
    print_every: int = 100


def sample_indices(n: int, batch_size: int, *, device: torch.device) -> Tensor:
    if n <= int(batch_size) or int(batch_size) <= 0:
        return torch.arange(n, device=device)
    return torch.randperm(n, device=device)[: int(batch_size)]


def poisson_nll(counts: Tensor, log_rate: Tensor, log_library_size: Tensor) -> Tensor:
    log_mu = log_rate + log_library_size.reshape(-1, 1)
    return F.poisson_nll_loss(log_mu, counts, log_input=True, full=False, reduction="mean")


def normalized_rate(tissue: TissueData, pseudocount: float = 0.0) -> Tensor:
    exposure = torch.exp(tissue.log_library_size).reshape(-1, 1)
    return (tissue.counts + float(pseudocount)) / torch.clamp(exposure, min=1e-6)


def sqrt_fisher_coordinate(rate: Tensor, eps: float = 1e-8) -> Tensor:
    return 2.0 * torch.sqrt(torch.clamp(rate, min=float(eps)))


def fit_coordinate_regressor(
    coords: Tensor,
    target: Tensor,
    *,
    hidden: Sequence[int] = (64, 64),
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: Optional[torch.device] = None,
) -> nn.Module:
    if device is None:
        device = default_device()
    x = coords.to(device)
    y = target.to(device).reshape(-1)
    model = MLP(2, 1, hidden=hidden, activation="tanh").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(int(epochs)):
        pred = torch.sigmoid(model(x).squeeze(-1))
        loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def evaluate_coordinate_maps(maps: CoordinateMaps, atlas: AtlasData, device: torch.device) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    maps.eval()
    for i, tissue in enumerate(atlas.tissues):
        z = maps.one(tissue.coords.to(device), i)
        out.append(z.detach().cpu().numpy())
    return out


def finite_mean(values: Iterable[Tensor], *, device: Optional[torch.device] = None) -> Tensor:
    vals = list(values)
    if not vals:
        if device is None:
            device = torch.device("cpu")
        return torch.zeros((), device=device)
    return torch.stack(vals).mean()
