# ============================================================
# Flow-STP Stage 1a: selected library-offset-only training run
#
# Run this cell AFTER:
#   1. the Qian PFC data-loading cell, which creates
#      counts_by_slice, spatial_by_slice, panel_df, obs_df, gene_names
#   2. the standalone Stage 1a model-definition cell, which defines
#      FlowSTPConfig and FlowSTPModel.
#
# Selected primary model:
#     expression decoder + library-size offset
#
# Coordinate regularization and slice nuisance effects are disabled in the
# primary configuration after Qian PFC ablations showed no consistent
# anatomical benefit from positive submersion weights.
#
# Main outputs left in memory:
#   model_flow_stp
#   xy_norm_params
#   stage1a_training_summary
#   stage1a_training_history_df
#   stage1a_coordinate_diagnostics_df
#   stage1a_cells
#   obs_flow_stp  (when obs_df exists)
# ============================================================

from __future__ import annotations

from pathlib import Path
import copy
import gc
import json
import math
import random
import time
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Required objects
# ------------------------------------------------------------

_REQUIRED = [
    "FlowSTPConfig",
    "FlowSTPModel",
    "counts_by_slice",
    "spatial_by_slice",
    "panel_df",
    "gene_names",
]
_missing = [name for name in _REQUIRED if name not in globals()]
if _missing:
    raise NameError(
        "Missing required objects: "
        + ", ".join(_missing)
        + ". Run the PFC loader and standalone Stage 1a model cells first."
    )

# ------------------------------------------------------------
# Fixed-run settings
# ------------------------------------------------------------

# Primary Stage 1a uses no submersion regularization.
SUBMERSION_WEIGHT = 0.0

# Change only this path if a different output location is preferred.
STAGE1A_ROOT = Path.cwd() / "flow_stp_stage1a_library_offset_only"

# Kept identical to the full sweep for direct comparability.
STAGE1A_EPOCHS = 1500
STAGE1A_BATCH_SIZE_PER_SLICE = 1024
STAGE1A_LR = 1e-3
STAGE1A_WEIGHT_DECAY = 1e-5
STAGE1A_GRAD_CLIP = 5.0
STAGE1A_PRINT_EVERY = 25
STAGE1A_CHECKPOINT_EVERY = 100
STAGE1A_VALIDATION_FRACTION = 0.10
STAGE1A_EVAL_BATCH_SIZE = 8192
STAGE1A_DIAGNOSTIC_CELLS_PER_SLICE = 4096
STAGE1A_SEED = 20260622
RESUME_EXISTING_RUN = True

# Coordinate-map condition and model architecture.
COORD_GRAD_MIN = 0.05
COORD_GRAD_SAMPLES_PER_SLICE = 256
COORD_GRAD_EPS = 1e-12
COORD_HIDDEN = (64, 64)
FIELD_HIDDEN = (128, 128)
DECODER_HIDDEN = (128, 128)
N_FLOW_STEPS = 16
COORD_MIN_WIDTH = 0.05
SLICE_EFFECT_DIM = 0
USE_LIBRARY_SIZE_OFFSET = True

# Device
# ------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif (
    getattr(torch.backends, "mps", None) is not None
    and torch.backends.mps.is_available()
):
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

print("Selected library-offset Stage 1a run output:", STAGE1A_ROOT)
print("Device:", DEVICE)
print("Submersion weight:", SUBMERSION_WEIGHT)
print("Library-size offset:", USE_LIBRARY_SIZE_OFFSET)

# Reproducibility helpers
# ------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def atomic_torch_save(payload, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def atomic_json_dump(payload, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(json_safe(payload), f, indent=2)
    tmp.replace(path)


def load_torch_checkpoint(path: Path, map_location):
    # PyTorch 2.6+ defaults to weights_only=True, but these resumable
    # checkpoints intentionally contain optimizer and RNG states too.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch versions lacking weights_only.
        return torch.load(path, map_location=map_location)




def show_figure_in_notebook_or_close(fig) -> None:
    try:
        get_ipython  # noqa: F821
    except NameError:
        plt.close(fig)
    else:
        plt.show()

# Validate and prepare the PFC data
# ------------------------------------------------------------

n_slices = len(counts_by_slice)
n_genes = len(gene_names)

if len(spatial_by_slice) != n_slices:
    raise ValueError("spatial_by_slice and counts_by_slice have different lengths.")
if n_slices == 0:
    raise ValueError("No slices were loaded.")


def to_dense_float_tensor(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.detach().cpu().float().contiguous()
    if hasattr(x, "toarray"):
        x = x.toarray()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32).contiguous()


counts_by_slice_stage1a: List[torch.Tensor] = []
coords_by_slice_stage1a: List[torch.Tensor] = []
xy_norm_records: List[dict] = []

for s, (counts_raw, xy_raw) in enumerate(zip(counts_by_slice, spatial_by_slice)):
    counts = to_dense_float_tensor(counts_raw)
    xy = np.asarray(xy_raw, dtype=np.float32)

    if counts.ndim != 2:
        raise ValueError(f"Slice {s}: counts must be 2D, got {counts.shape}.")
    if counts.shape[1] != n_genes:
        raise ValueError(
            f"Slice {s}: counts has {counts.shape[1]} genes; expected {n_genes}."
        )
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"Slice {s}: spatial coordinates must be [N,2].")
    if counts.shape[0] != xy.shape[0]:
        raise ValueError(
            f"Slice {s}: {counts.shape[0]} count rows but {xy.shape[0]} coordinates."
        )
    if not np.isfinite(xy).all():
        raise ValueError(f"Slice {s}: non-finite spatial coordinates remain.")
    if (counts < 0).any():
        raise ValueError(f"Slice {s}: counts contain negative values.")

    center = xy.mean(axis=0)
    scale = xy.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    xy_norm = ((xy - center[None, :]) / scale[None, :]).astype(np.float32)

    counts_by_slice_stage1a.append(counts)
    coords_by_slice_stage1a.append(torch.from_numpy(xy_norm).contiguous())
    xy_norm_records.append(
        {
            "slice_id": int(s),
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "scale_x": float(scale[0]),
            "scale_y": float(scale[1]),
        }
    )

xy_norm_params_stage1a = pd.DataFrame(xy_norm_records)

# Normalize gestational times to [0,1], ordered by slice_id.
panel_df_stage1a = panel_df.copy()
if "slice_id" not in panel_df_stage1a.columns:
    panel_df_stage1a["slice_id"] = np.arange(len(panel_df_stage1a), dtype=int)

panel_df_stage1a["slice_id"] = panel_df_stage1a["slice_id"].astype(int)
panel_df_stage1a = panel_df_stage1a.sort_values("slice_id").reset_index(drop=True)

expected_slice_ids = np.arange(n_slices, dtype=int)
if not np.array_equal(panel_df_stage1a["slice_id"].to_numpy(), expected_slice_ids):
    raise ValueError(
        "panel_df must contain exactly one row for every slice_id from 0 to "
        f"{n_slices - 1}."
    )

if "time" not in panel_df_stage1a.columns:
    time_map_default = {"GW15": 15.0, "GW20": 20.0, "GW22": 22.0, "GW34": 34.0}
    if "gw" in panel_df_stage1a.columns:
        panel_df_stage1a["time"] = panel_df_stage1a["gw"].astype(str).map(time_map_default)
    elif "GW" in panel_df_stage1a.columns:
        panel_df_stage1a["time"] = panel_df_stage1a["GW"].astype(str).map(time_map_default)
    else:
        raise KeyError("panel_df needs a 'time', 'gw', or 'GW' column.")

raw_times = panel_df_stage1a["time"].to_numpy(dtype=np.float32)
if not np.isfinite(raw_times).all():
    raise ValueError("panel_df time values contain missing/non-finite entries.")

time_min = float(raw_times.min())
time_max = float(raw_times.max())
times_norm_stage1a = (raw_times - time_min) / max(time_max - time_min, 1e-8)

# Library-size offsets centered at the global median library size.
def build_log_library_offsets(
    matrices: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    if not USE_LIBRARY_SIZE_OFFSET:
        return [torch.zeros(x.shape[0], dtype=torch.float32) for x in matrices]

    library_sizes = [x.sum(dim=1).float() for x in matrices]
    all_library_sizes = torch.cat(library_sizes)
    positive = all_library_sizes[all_library_sizes > 0]
    median_library = positive.median() if positive.numel() else torch.tensor(1.0)

    offsets = []
    for lib in library_sizes:
        size_factor = lib / torch.clamp(median_library, min=1.0)
        size_factor = torch.clamp(size_factor, min=0.05, max=20.0)
        offsets.append(torch.log(size_factor).float().contiguous())
    return offsets


log_library_by_slice_stage1a = build_log_library_offsets(counts_by_slice_stage1a)

# Fixed train/validation split shared across all 20 runs.
split_rng = np.random.default_rng(STAGE1A_SEED + 101)
train_indices_by_slice: List[torch.Tensor] = []
val_indices_by_slice: List[torch.Tensor] = []

for s, counts in enumerate(counts_by_slice_stage1a):
    n = counts.shape[0]
    perm = split_rng.permutation(n)
    n_val = int(round(STAGE1A_VALIDATION_FRACTION * n))
    n_val = min(max(n_val, 1), max(n - 1, 1))

    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])

    if len(train_idx) == 0:
        raise ValueError(f"Slice {s} has too few observations for a split.")

    train_indices_by_slice.append(torch.as_tensor(train_idx, dtype=torch.long))
    val_indices_by_slice.append(torch.as_tensor(val_idx, dtype=torch.long))

# Fixed coordinate diagnostic subsets shared across runs.
diag_rng = np.random.default_rng(STAGE1A_SEED + 202)
diagnostic_indices_by_slice: List[torch.Tensor] = []
for s, counts in enumerate(counts_by_slice_stage1a):
    n = counts.shape[0]
    k = min(int(STAGE1A_DIAGNOSTIC_CELLS_PER_SLICE), n)
    idx = np.sort(diag_rng.choice(n, size=k, replace=False))
    diagnostic_indices_by_slice.append(torch.as_tensor(idx, dtype=torch.long))

# Pin long-lived CPU tensors once when CUDA is available.
def maybe_pin(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous().cpu()
    return x.pin_memory() if DEVICE.type == "cuda" else x


coords_by_slice_stage1a = [maybe_pin(x.float()) for x in coords_by_slice_stage1a]
counts_by_slice_stage1a = [maybe_pin(x.float()) for x in counts_by_slice_stage1a]
log_library_by_slice_stage1a = [maybe_pin(x.float()) for x in log_library_by_slice_stage1a]
train_indices_by_slice = [x.cpu() for x in train_indices_by_slice]
val_indices_by_slice = [x.cpu() for x in val_indices_by_slice]
diagnostic_indices_by_slice = [x.cpu() for x in diagnostic_indices_by_slice]

print(
    f"Prepared {n_slices} slices, {n_genes} genes, "
    f"{sum(x.shape[0] for x in counts_by_slice_stage1a):,} observations."
)
print("Raw times:", raw_times)
print("Normalized times:", times_norm_stage1a)

# ------------------------------------------------------------
# Output structure and manifest
# ------------------------------------------------------------

STAGE1A_ROOT.mkdir(parents=True, exist_ok=True)
(STAGE1A_ROOT / "tables").mkdir(exist_ok=True)
(STAGE1A_ROOT / "figures").mkdir(exist_ok=True)

xy_norm_params_stage1a.to_csv(
    STAGE1A_ROOT / "tables" / "xy_normalization_params.csv",
    index=False,
)
panel_df_stage1a.to_csv(
    STAGE1A_ROOT / "tables" / "panel_df.csv",
    index=False,
)

split_payload = {}
for s in range(n_slices):
    split_payload[f"train_{s}"] = train_indices_by_slice[s].numpy()
    split_payload[f"validation_{s}"] = val_indices_by_slice[s].numpy()
    split_payload[f"diagnostic_{s}"] = diagnostic_indices_by_slice[s].numpy()
np.savez_compressed(
    STAGE1A_ROOT / "tables" / "fixed_splits_and_diagnostics.npz",
    **split_payload,
)

manifest = {
    "submersion_weight": float(SUBMERSION_WEIGHT),
    "epochs": STAGE1A_EPOCHS,
    "batch_size_per_slice": STAGE1A_BATCH_SIZE_PER_SLICE,
    "learning_rate": STAGE1A_LR,
    "weight_decay": STAGE1A_WEIGHT_DECAY,
    "gradient_clip": STAGE1A_GRAD_CLIP,
    "validation_fraction": STAGE1A_VALIDATION_FRACTION,
    "seed": STAGE1A_SEED,
    "coord_grad_min": COORD_GRAD_MIN,
    "coord_grad_samples_per_slice": COORD_GRAD_SAMPLES_PER_SLICE,
    "coord_hidden": COORD_HIDDEN,
    "field_hidden": FIELD_HIDDEN,
    "decoder_hidden": DECODER_HIDDEN,
    "n_flow_steps": N_FLOW_STEPS,
    "coord_min_width": COORD_MIN_WIDTH,
    "slice_effect_dim": SLICE_EFFECT_DIM,
    "use_library_size_offset": USE_LIBRARY_SIZE_OFFSET,
    "num_slices": n_slices,
    "num_genes": n_genes,
    "time_min": time_min,
    "time_max": time_max,
    "raw_times": raw_times,
    "normalized_times": times_norm_stage1a,
}
atomic_json_dump(manifest, STAGE1A_ROOT / "training_manifest.json")

# Model/config helpers
# ------------------------------------------------------------

def make_config(weight: float) -> "FlowSTPConfig":
    return FlowSTPConfig(
        num_slices=n_slices,
        num_genes=n_genes,
        coord_hidden=COORD_HIDDEN,
        field_hidden=FIELD_HIDDEN,
        decoder_hidden=DECODER_HIDDEN,
        n_flow_steps=N_FLOW_STEPS,
        coord_min_width=COORD_MIN_WIDTH,
        slice_effect_dim=SLICE_EFFECT_DIM,
        coord_grad_min=COORD_GRAD_MIN,
        coord_submersion_weight=float(weight),
        coord_grad_samples_per_slice=COORD_GRAD_SAMPLES_PER_SLICE,
        coord_grad_eps=COORD_GRAD_EPS,
    )


# All fresh runs begin from exactly the same parameters.
set_all_seeds(STAGE1A_SEED)
_initial_model = FlowSTPModel(make_config(float(SUBMERSION_WEIGHT))).cpu()
INITIAL_STATE = {
    key: value.detach().cpu().clone()
    for key, value in _initial_model.state_dict().items()
}
del _initial_model


# ------------------------------------------------------------
# Batch construction and evaluation
# ------------------------------------------------------------

def make_balanced_training_batch(
    generator: torch.Generator,
) -> Dict[str, torch.Tensor]:
    batch_coords = []
    batch_counts = []
    batch_loglib = []
    batch_slice_idx = []
    batch_time = []

    for s in range(n_slices):
        train_idx = train_indices_by_slice[s]
        n_train = train_idx.numel()
        bsz = min(int(STAGE1A_BATCH_SIZE_PER_SLICE), n_train)
        positions = torch.randint(
            low=0,
            high=n_train,
            size=(bsz,),
            generator=generator,
        )
        idx = train_idx[positions]

        batch_coords.append(coords_by_slice_stage1a[s][idx])
        batch_counts.append(counts_by_slice_stage1a[s][idx])
        batch_loglib.append(log_library_by_slice_stage1a[s][idx])
        batch_slice_idx.append(torch.full((bsz,), s, dtype=torch.long))
        batch_time.append(
            torch.full((bsz,), float(times_norm_stage1a[s]), dtype=torch.float32)
        )

    def move(x: torch.Tensor) -> torch.Tensor:
        return x.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))

    return {
        "coords": move(torch.cat(batch_coords, dim=0)),
        "counts": move(torch.cat(batch_counts, dim=0)),
        "log_library_size": move(torch.cat(batch_loglib, dim=0)),
        "slice_idx": move(torch.cat(batch_slice_idx, dim=0)),
        "time": move(torch.cat(batch_time, dim=0)),
    }


@torch.no_grad()
def evaluate_validation_poisson_nll(model: "FlowSTPModel") -> float:
    model.eval()
    total_nll = 0.0
    total_elements = 0

    for s in range(n_slices):
        idx_all = val_indices_by_slice[s]
        for start in range(0, idx_all.numel(), STAGE1A_EVAL_BATCH_SIZE):
            idx = idx_all[start : start + STAGE1A_EVAL_BATCH_SIZE]
            bsz = idx.numel()

            coords = coords_by_slice_stage1a[s][idx].to(
                DEVICE, non_blocking=(DEVICE.type == "cuda")
            )
            counts = counts_by_slice_stage1a[s][idx].to(
                DEVICE, non_blocking=(DEVICE.type == "cuda")
            )
            loglib = log_library_by_slice_stage1a[s][idx].to(
                DEVICE, non_blocking=(DEVICE.type == "cuda")
            )
            slice_idx = torch.full((bsz,), s, dtype=torch.long, device=DEVICE)
            t = torch.full(
                (bsz,),
                float(times_norm_stage1a[s]),
                dtype=torch.float32,
                device=DEVICE,
            )

            out = model(
                coords=coords,
                slice_idx=slice_idx,
                time=t,
                log_library_size=loglib,
            )
            nll_sum = F.poisson_nll_loss(
                input=out["log_mu"],
                target=counts,
                log_input=True,
                full=True,
                reduction="sum",
            )
            total_nll += float(nll_sum.cpu())
            total_elements += int(counts.numel())

    return total_nll / max(total_elements, 1)


def _stats_from_array(values: np.ndarray, prefix: str) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_p01": float(np.quantile(values, 0.01)),
        f"{prefix}_p05": float(np.quantile(values, 0.05)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
        f"{prefix}_max": float(np.max(values)),
    }


def evaluate_coordinate_diagnostics(
    model: "FlowSTPModel",
) -> Tuple[dict, pd.DataFrame]:
    """
    Evaluate z and ||grad z|| on fixed cells shared across every run.

    The regularizer uses gradients with respect to standardized XY. Native-XY
    gradient magnitudes are also recorded for later geometric interpretation.
    """
    model.eval()

    all_g_std = []
    all_g_native = []
    all_z = []
    rows = []

    with torch.no_grad():
        a_all, b_all, width_all = model.coord_maps.intervals()
        a_all = a_all.detach().cpu().numpy()
        b_all = b_all.detach().cpu().numpy()
        width_all = width_all.detach().cpu().numpy()

    for s in range(n_slices):
        idx = diagnostic_indices_by_slice[s]
        coords = coords_by_slice_stage1a[s][idx].to(
            DEVICE, non_blocking=(DEVICE.type == "cuda")
        )
        slice_idx = torch.full(
            (idx.numel(),), s, dtype=torch.long, device=DEVICE
        )

        # This method needs input autograd, even though no parameter gradients
        # are retained because create_graph=False.
        with torch.enable_grad():
            z, grad_std, _, _ = model.coordinate_spatial_gradients(
                coords=coords,
                slice_idx=slice_idx,
                samples_per_slice=0,  # use the complete fixed diagnostic subset
                create_graph=False,
            )

        z_np = z.detach().cpu().numpy().reshape(-1)
        grad_std_np = grad_std.detach().cpu().numpy()
        g_std = np.sqrt(np.sum(grad_std_np ** 2, axis=1) + COORD_GRAD_EPS)

        scale_x = float(xy_norm_params_stage1a.loc[s, "scale_x"])
        scale_y = float(xy_norm_params_stage1a.loc[s, "scale_y"])
        grad_native_np = grad_std_np / np.array(
            [scale_x, scale_y], dtype=np.float32
        )[None, :]
        g_native = np.sqrt(np.sum(grad_native_np ** 2, axis=1) + COORD_GRAD_EPS)

        row = {
            "slice_id": int(s),
            "n_diagnostic": int(len(g_std)),
            "interval_a": float(a_all[s]),
            "interval_b": float(b_all[s]),
            "interval_width": float(width_all[s]),
            "z_min": float(np.min(z_np)),
            "z_p01": float(np.quantile(z_np, 0.01)),
            "z_p05": float(np.quantile(z_np, 0.05)),
            "z_median": float(np.median(z_np)),
            "z_p95": float(np.quantile(z_np, 0.95)),
            "z_p99": float(np.quantile(z_np, 0.99)),
            "z_max": float(np.max(z_np)),
            "z_std": float(np.std(z_np)),
            "grad_std_violation_fraction": float(np.mean(g_std < COORD_GRAD_MIN)),
            **_stats_from_array(g_std, "grad_std"),
            **_stats_from_array(g_native, "grad_native"),
        }
        rows.append(row)

        all_g_std.append(g_std)
        all_g_native.append(g_native)
        all_z.append(z_np)

    all_g_std_np = np.concatenate(all_g_std)
    all_g_native_np = np.concatenate(all_g_native)
    all_z_np = np.concatenate(all_z)

    global_summary = {
        "diagnostic_n": int(len(all_g_std_np)),
        "coord_grad_min_target": float(COORD_GRAD_MIN),
        "grad_std_violation_fraction": float(
            np.mean(all_g_std_np < COORD_GRAD_MIN)
        ),
        **_stats_from_array(all_g_std_np, "grad_std"),
        **_stats_from_array(all_g_native_np, "grad_native"),
        "z_global_min": float(np.min(all_z_np)),
        "z_global_p05": float(np.quantile(all_z_np, 0.05)),
        "z_global_median": float(np.median(all_z_np)),
        "z_global_p95": float(np.quantile(all_z_np, 0.95)),
        "z_global_max": float(np.max(all_z_np)),
        "z_global_std": float(np.std(all_z_np)),
        "interval_width_min": float(np.min(width_all)),
        "interval_width_median": float(np.median(width_all)),
        "interval_width_max": float(np.max(width_all)),
    }

    return global_summary, pd.DataFrame(rows)

def train_fixed_stage1a() -> dict:
    run_index = 0
    weight = float(SUBMERSION_WEIGHT)
    run_name = "stage1a_lambda_0p11364637"
    run_dir = STAGE1A_ROOT
    run_dir.mkdir(parents=True, exist_ok=True)

    done_path = run_dir / "done.json"
    checkpoint_path = run_dir / "checkpoint_last.pt"
    history_path = run_dir / "training_history.csv"
    summary_path = run_dir / "summary.json"
    diagnostics_path = run_dir / "coordinate_diagnostics_by_slice.csv"
    final_model_path = run_dir / "model_final.pt"

    if (
        RESUME_EXISTING_RUN
        and done_path.exists()
        and summary_path.exists()
        and final_model_path.exists()
    ):
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"\nAlready complete: {run_name}")
        return summary

    print("\n" + "=" * 78)
    print(f"Stage 1a fixed run | lambda_sub = {weight:.8g}")
    print("=" * 78)

    # Same initialization and random sequence for every lambda.
    set_all_seeds(STAGE1A_SEED)
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(STAGE1A_SEED + 303)

    config = make_config(weight)
    model = FlowSTPModel(config).to(DEVICE)
    model.load_state_dict(INITIAL_STATE, strict=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=STAGE1A_LR,
        weight_decay=STAGE1A_WEIGHT_DECAY,
    )

    history: List[dict] = []
    start_epoch = 1
    elapsed_before_resume = 0.0

    if RESUME_EXISTING_RUN and checkpoint_path.exists():
        ckpt = load_torch_checkpoint(checkpoint_path, map_location=DEVICE)
        if not math.isclose(float(ckpt["weight"]), float(weight), rel_tol=0, abs_tol=1e-15):
            raise ValueError(f"Checkpoint weight mismatch in {run_dir}")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        history = list(ckpt.get("history", []))
        elapsed_before_resume = float(ckpt.get("elapsed_sec", 0.0))

        # Restore RNG states so a resumed run follows the same sequence.
        if "python_rng_state" in ckpt:
            random.setstate(ckpt["python_rng_state"])
        if "numpy_rng_state" in ckpt:
            np.random.set_state(ckpt["numpy_rng_state"])
        if "torch_rng_state" in ckpt:
            torch.random.set_rng_state(ckpt["torch_rng_state"].cpu())
        if DEVICE.type == "cuda" and ckpt.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])
        if "batch_generator_state" in ckpt:
            batch_generator.set_state(ckpt["batch_generator_state"].cpu())

        print(f"Resuming at epoch {start_epoch} from {checkpoint_path}")

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    wall_start = time.time()

    for epoch in range(start_epoch, STAGE1A_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        batch = make_balanced_training_batch(batch_generator)
        loss, parts = model.loss(batch)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch}, lambda={weight}: {loss}"
            )

        loss.backward()

        if STAGE1A_GRAD_CLIP is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), STAGE1A_GRAD_CLIP
            )
        else:
            grad_norm = torch.tensor(float("nan"), device=DEVICE)

        optimizer.step()

        should_log = (
            epoch == 1
            or epoch == STAGE1A_EPOCHS
            or epoch % STAGE1A_PRINT_EVERY == 0
        )
        elapsed = elapsed_before_resume + (time.time() - wall_start)

        if should_log:
            row = {
                "run_index": int(run_index),
                "submersion_weight": float(weight),
                "epoch": int(epoch),
                "loss": float(parts["total"].cpu()),
                "data_nll": float(parts["data"].cpu()),
                "submersion_penalty": float(parts["coord_submersion"].cpu()),
                "weighted_submersion_penalty": float(
                    parts["coord_submersion_weighted"].cpu()
                ),
                "batch_grad_min": float(parts["coord_grad_min"].cpu()),
                "batch_grad_p05": float(parts["coord_grad_p05"].cpu()),
                "batch_grad_median": float(parts["coord_grad_median"].cpu()),
                "batch_grad_mean": float(parts["coord_grad_mean"].cpu()),
                "batch_grad_violation_fraction": float(
                    parts["coord_grad_violation_frac"].cpu()
                ),
                "z_mean": float(parts["z_mean"].cpu()),
                "a_mean": float(parts["a_mean"].cpu()),
                "rho_mean": float(parts["rho_mean"].cpu()),
                "parameter_grad_norm_before_clip": float(
                    grad_norm.detach().cpu()
                ),
                "elapsed_sec": float(elapsed),
            }
            history.append(row)
            pd.DataFrame(history).to_csv(history_path, index=False)

            msg = (
                f"epoch {epoch:05d} | total={row['loss']:.5f} | "
                f"data={row['data_nll']:.5f} | "
                f"sub={row['submersion_penalty']:.5f} | "
                f"lambda*sub={row['weighted_submersion_penalty']:.5f} | "
                f"g_p05={row['batch_grad_p05']:.5f} | "
                f"viol={100.0 * row['batch_grad_violation_fraction']:.1f}% | "
                f"elapsed={elapsed / 60.0:.1f} min"
            )
            if DEVICE.type == "cuda":
                msg += (
                    f" | peak GPU={torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
                )
            print(msg)

        if (
            epoch % STAGE1A_CHECKPOINT_EVERY == 0
            or epoch == STAGE1A_EPOCHS
        ):
            checkpoint = {
                "run_index": int(run_index),
                "weight": float(weight),
                "epoch": int(epoch),
                "model_state_dict": {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "elapsed_sec": float(elapsed),
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.random.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all() if DEVICE.type == "cuda" else None
                ),
                "batch_generator_state": batch_generator.get_state(),
                "config": config.__dict__,
            }
            atomic_torch_save(checkpoint, checkpoint_path)

        del batch, loss, parts

    training_elapsed = elapsed_before_resume + (time.time() - wall_start)

    validation_nll = evaluate_validation_poisson_nll(model)
    coord_summary, coord_by_slice = evaluate_coordinate_diagnostics(model)
    coord_by_slice.insert(0, "submersion_weight", float(weight))
    coord_by_slice.insert(0, "run_index", int(run_index))
    coord_by_slice.to_csv(diagnostics_path, index=False)

    final_history = pd.DataFrame(history)
    final_row = final_history.iloc[-1].to_dict() if len(final_history) else {}

    summary = {
        "run_index": int(run_index),
        "run_name": run_name,
        "submersion_weight": float(weight),
        "epochs": int(STAGE1A_EPOCHS),
        "training_elapsed_sec": float(training_elapsed),
        "validation_poisson_nll": float(validation_nll),
        "final_training_total": float(final_row.get("loss", np.nan)),
        "final_training_data_nll": float(final_row.get("data_nll", np.nan)),
        "final_submersion_penalty": float(
            final_row.get("submersion_penalty", np.nan)
        ),
        "final_weighted_submersion_penalty": float(
            final_row.get("weighted_submersion_penalty", np.nan)
        ),
        **coord_summary,
        "run_directory": str(run_dir),
        "model_path": str(final_model_path),
        "history_path": str(history_path),
        "diagnostics_path": str(diagnostics_path),
    }

    final_payload = {
        "model_state_dict": {
            k: v.detach().cpu() for k, v in model.state_dict().items()
        },
        "config": config.__dict__,
        "run_summary": summary,
        "gene_names": list(gene_names),
        "panel_df": panel_df_stage1a.to_dict(orient="list"),
        "xy_norm_params": xy_norm_params_stage1a.to_dict(orient="list"),
        "time_metadata": {
            "time_min": time_min,
            "time_max": time_max,
            "times_raw": raw_times.tolist(),
            "times_norm": times_norm_stage1a.tolist(),
        },
        "training_settings": manifest,
    }
    atomic_torch_save(final_payload, final_model_path)
    atomic_json_dump(summary, summary_path)
    atomic_json_dump(
        {"complete": True, "completed_at_unix": time.time()},
        done_path,
    )

    print(
        f"Completed lambda={weight:.8g} | "
        f"validation NLL={validation_nll:.6f} | "
        f"fixed g_p05={coord_summary['grad_std_p05']:.6f} | "
        f"fixed violation={100.0 * coord_summary['grad_std_violation_fraction']:.2f}%"
    )

    del model, optimizer
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return summary

# ------------------------------------------------------------
# Execute or resume the selected fixed run
# ------------------------------------------------------------

training_start = time.time()

try:
    stage1a_training_summary = train_fixed_stage1a()
except KeyboardInterrupt:
    print(
        "Training interrupted by user. Rerun this cell to resume from "
        "checkpoint_last.pt."
    )
    raise

print(
    "\nFixed Stage 1a training finished in "
    f"{(time.time() - training_start) / 60.0:.2f} minutes."
)


# ------------------------------------------------------------
# Load the final model and expose Stage 1a objects for Stage 1b
# ------------------------------------------------------------

FINAL_MODEL_PATH = STAGE1A_ROOT / "model_final.pt"
if not FINAL_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Expected final model at {FINAL_MODEL_PATH}; training did not finish."
    )

final_payload = load_torch_checkpoint(FINAL_MODEL_PATH, map_location=DEVICE)
final_config = FlowSTPConfig(**final_payload["config"])

model_flow_stp = FlowSTPModel(final_config).to(DEVICE)
model_flow_stp.load_state_dict(final_payload["model_state_dict"], strict=True)
model_flow_stp.eval()

# Names expected by the Stage 1b cells.
xy_norm_params = xy_norm_params_stage1a.copy()
panel_df_flow_stp = panel_df_stage1a.copy()
times_norm_flow_stp = np.asarray(times_norm_stage1a, dtype=np.float32).copy()

history_path = STAGE1A_ROOT / "training_history.csv"
diagnostics_path = STAGE1A_ROOT / "coordinate_diagnostics_by_slice.csv"

stage1a_training_history_df = (
    pd.read_csv(history_path)
    if history_path.exists()
    else pd.DataFrame()
)
stage1a_coordinate_diagnostics_df = (
    pd.read_csv(diagnostics_path)
    if diagnostics_path.exists()
    else pd.DataFrame()
)


# ------------------------------------------------------------
# Cell-level Stage 1a coordinate inference
# ------------------------------------------------------------

@torch.no_grad()
def infer_stage1a_coordinates(
    model: "FlowSTPModel",
    eval_batch_size: int = STAGE1A_EVAL_BATCH_SIZE,
) -> pd.DataFrame:
    model.eval()
    frames = []

    for s in range(n_slices):
        coords_cpu = coords_by_slice_stage1a[s]
        z_parts = []

        for start in range(0, coords_cpu.shape[0], int(eval_batch_size)):
            coords = coords_cpu[start : start + int(eval_batch_size)].to(
                DEVICE,
                non_blocking=(DEVICE.type == "cuda"),
            )
            slice_idx = torch.full(
                (coords.shape[0],),
                int(s),
                dtype=torch.long,
                device=DEVICE,
            )
            z = model.coord_maps(coords, slice_idx)
            if isinstance(z, tuple):
                z = z[0]
            z_parts.append(z.reshape(-1).detach().cpu().numpy())

        z_slice = np.concatenate(z_parts)
        xy_raw = np.asarray(spatial_by_slice[s], dtype=np.float32)[:, :2]

        frames.append(
            pd.DataFrame(
                {
                    "slice_id": int(s),
                    "cell_index": np.arange(len(z_slice), dtype=int),
                    "x": xy_raw[:, 0],
                    "y": xy_raw[:, 1],
                    "z_stage1a": z_slice,
                    "time_raw": float(raw_times[s]),
                    "time_normalized": float(times_norm_stage1a[s]),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


stage1a_cells = infer_stage1a_coordinates(model_flow_stp)
stage1a_cells.to_csv(
    STAGE1A_ROOT / "tables" / "stage1a_all_cells.csv.gz",
    index=False,
    compression="gzip",
)

# Merge coordinates into the loader metadata when available.
if "obs_df" in globals():
    metadata = obs_df.copy()
    metadata["slice_id"] = metadata["slice_id"].astype(int)
    metadata["cell_index"] = (
        metadata.groupby("slice_id", sort=False).cumcount().astype(int)
    )

    obs_flow_stp = metadata.merge(
        stage1a_cells[
            [
                "slice_id",
                "cell_index",
                "z_stage1a",
                "time_raw",
                "time_normalized",
            ]
        ],
        on=["slice_id", "cell_index"],
        how="left",
        validate="one_to_one",
    )
    obs_flow_stp.to_csv(
        STAGE1A_ROOT / "tables" / "pfc_obs_with_stage1a.csv.gz",
        index=False,
        compression="gzip",
    )


# ------------------------------------------------------------
# Compact training diagnostics
# ------------------------------------------------------------

if len(stage1a_training_history_df):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(
        stage1a_training_history_df["epoch"],
        stage1a_training_history_df["data_nll"],
        label="Poisson data NLL",
    )
    ax.plot(
        stage1a_training_history_df["epoch"],
        stage1a_training_history_df["loss"],
        label="Total loss",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(
        "Stage 1a training "
        f"(lambda_sub={SUBMERSION_WEIGHT:.8g})"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        STAGE1A_ROOT / "figures" / "training_loss.png",
        dpi=250,
    )
    show_figure_in_notebook_or_close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(
        stage1a_training_history_df["epoch"],
        stage1a_training_history_df["batch_grad_p05"],
        label="Batch gradient p05",
    )
    ax.axhline(
        COORD_GRAD_MIN,
        linestyle="--",
        label=f"Target m={COORD_GRAD_MIN:g}",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("||grad z|| in standardized XY")
    ax.set_title("Stage 1a submersion diagnostic")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        STAGE1A_ROOT / "figures" / "gradient_p05_training.png",
        dpi=250,
    )
    show_figure_in_notebook_or_close(fig)

print("\nStage 1a model is ready for Stage 1b.")
print("Model:", type(model_flow_stp).__name__)
print("Device:", next(model_flow_stp.parameters()).device)
print("Submersion weight:", SUBMERSION_WEIGHT)
print("Output root:", STAGE1A_ROOT)
print("Validation Poisson NLL:", stage1a_training_summary["validation_poisson_nll"])
print("Fixed gradient p05:", stage1a_training_summary["grad_std_p05"])
print(
    "Fixed violation fraction:",
    stage1a_training_summary["grad_std_violation_fraction"],
)
