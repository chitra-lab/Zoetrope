"""Qian et al. MERFISH fetal cortex PFC loader for Flow-STP.

This module does not store or copy the atlas data. It reads selected PFC
panels from the shared lab dataset folder and returns objects used by the
Flow-STP notebooks.

Default data root:
    /scratch/uchitra1/datasets/spatial-RNA/merfish/merfish_cortex_atlas/zenodo_14422018_processed
"""

# ============================================================
# Qian PFC 10-slice data-loading cell for Flow-STP prototype
#
# No GASTON loading.
# No GASTON comparison metadata.
#
# Produces:
# counts_by_slice : list[torch.Tensor], each [N_i, G]
# obs_df : cell/spot metadata dataframe
# panel_df : one row per tissue slice/panel
# gene_names : list[str]
# spatial_by_slice : list[np.ndarray], each [N_i, 2]
#
# These names are compatible with the Flow-STP training cell.
# ============================================================

from pathlib import Path
import re
import gc
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
import scanpy as sc
import torch


# ============================================================
# Paths and constants
# ============================================================

DATA_ROOT = Path(
    "/scratch/uchitra1/datasets/spatial-RNA/merfish/"
    "merfish_cortex_atlas/zenodo_14422018_processed"
)

GW_FILES = {
    "GW15": DATA_ROOT / "gw15.h5ad",
    "GW20": DATA_ROOT / "gw20.h5ad",
    "GW22": DATA_ROOT / "gw22.h5ad",
    "GW34": DATA_ROOT / "gw34.h5ad",
}

# PFC atlas panels.
# With sample-splitting enabled, this should give the 10-slice PFC dataset.
TARGETS = [
    {"gw": "GW15", "regions": ["F1a", "F1b"]},
    {"gw": "GW20", "regions": ["F1", "F2a", "F2b"]},
    {"gw": "GW22", "regions": ["F1", "F2", "F3"]},
    {"gw": "GW34", "regions": ["BA9"]},
]

TIME_MAP = {
    "GW15": 15.0,
    "GW20": 20.0,
    "GW22": 22.0,
    "GW34": 34.0,
}

LABEL_COL = "H1_annotation"
REGION_COL = "region"
SAMPLE_COL = "sample"
SPATIAL_KEY = "spatial"

# Use None to read X. Set to a layer name if the count matrix lives in a layer.
LAYER_KEY_FOR_COUNTS = None

# Set to an integer for quick debugging, otherwise None.
MAX_CELLS_PER_SLICE = None
MAX_GENES_FOR_DEBUG = None

SPLIT_MULTI_SAMPLE_REGIONS = True
RANDOM_SEED = 20260622

# Optional label cleanup.
H1_RELABEL_MAP = {
    "EN-IT-1": "EN-IT",
    "EN-IT-2": "EN-IT",
}

EXPECTED_NUM_PFC_SLICES = 10


# ============================================================
# Basic helpers
# ============================================================

def sanitize_name(x):
    x = str(x)
    x = re.sub(r"[^\w\-.]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def safe_close_backed(adata_b):
    try:
        adata_b.file.close()
    except Exception:
        pass


def normalize_h1(labels):
    s = pd.Series(labels)
    s = s.where(~s.isna(), "Unknown").astype(str)
    s = s.replace(H1_RELABEL_MAP)
    return s.to_numpy(dtype=str)


def get_panel_mask(obs: pd.DataFrame, spec: dict) -> np.ndarray:
    mask = obs[REGION_COL].astype(str).values == str(spec["region"])

    if spec.get("sample", None) is not None:
        mask = mask & (obs[SAMPLE_COL].astype(str).values == str(spec["sample"]))

    return mask


def as_dense_float32(X):
    if sp.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)

    return X.astype(np.float32, copy=False)


def _decode_h5_attr(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return x


def _get_h5_matrix_object(f, layer_key=None):
    if layer_key is None:
        return f["X"]
    return f["layers"][layer_key]


def _get_h5_shape(X_obj):
    shape = X_obj.attrs.get("shape", None)

    if shape is None and "shape" in X_obj:
        shape = X_obj["shape"][()]

    if shape is None:
        raise ValueError("Could not determine matrix shape from h5ad object.")

    return tuple(int(x) for x in shape)


def read_h5ad_rows_direct(filename, idx, layer_key=None):
    """
    Read selected h5ad rows directly from X or a layer.
    Handles CSR, CSC, and dense h5ad storage.
    Returns dense float32 array with rows in the same order as idx.
    """
    filename = str(filename)
    idx = np.asarray(idx, dtype=np.int64)

    if len(idx) == 0:
        raise ValueError("Requested zero rows from expression matrix.")

    order = np.argsort(idx)
    idx_sorted = idx[order]

    with h5py.File(filename, "r") as f:
        X_obj = _get_h5_matrix_object(f, layer_key=layer_key)

        if isinstance(X_obj, h5py.Group):
            encoding_type = _decode_h5_attr(X_obj.attrs.get("encoding-type", ""))
            shape = _get_h5_shape(X_obj)
            n_obs, n_vars = shape

            data_ds = X_obj["data"]
            indices_ds = X_obj["indices"]
            indptr_ds = X_obj["indptr"]

            if encoding_type == "csr_matrix":
                breaks = np.where(np.diff(idx_sorted) != 1)[0] + 1
                runs = np.split(idx_sorted, breaks)

                sparse_pieces = []

                for rows in runs:
                    row_start = int(rows[0])
                    row_stop = int(rows[-1]) + 1

                    indptr_block = indptr_ds[row_start : row_stop + 1].astype(np.int64)
                    data_start = int(indptr_block[0])
                    data_stop = int(indptr_block[-1])

                    data_block = data_ds[data_start:data_stop]
                    indices_block = indices_ds[data_start:data_stop].astype(np.int64)
                    indptr_block = indptr_block - indptr_block[0]

                    X_block = sp.csr_matrix(
                        (data_block, indices_block, indptr_block),
                        shape=(row_stop - row_start, n_vars),
                    )
                    sparse_pieces.append(X_block)

                X_sorted = (
                    sparse_pieces[0]
                    if len(sparse_pieces) == 1
                    else sp.vstack(sparse_pieces, format="csr")
                )
                X_sorted = as_dense_float32(X_sorted)

            elif encoding_type == "csc_matrix":
                data = data_ds[:]
                indices = indices_ds[:].astype(np.int64)
                indptr = indptr_ds[:].astype(np.int64)

                X_full = sp.csc_matrix((data, indices, indptr), shape=(n_obs, n_vars))
                X_sorted = as_dense_float32(X_full[idx_sorted, :])

            else:
                raise ValueError(f"Unsupported sparse encoding-type: {encoding_type}")

        elif isinstance(X_obj, h5py.Dataset):
            X_sorted = as_dense_float32(X_obj[idx_sorted, :])

        else:
            raise TypeError(f"Unsupported HDF5 object type: {type(X_obj)}")

    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(len(order))

    return X_sorted[inverse_order].astype(np.float32, copy=False)


# ============================================================
# Build PFC 10-slice panel dataframe
# ============================================================

def build_qian_pfc_panel_specs():
    """
    Build PFC panel specs from GW files and target regions.
    Multi-sample regions are split when SPLIT_MULTI_SAMPLE_REGIONS=True.
    """
    panel_specs = []

    for target in TARGETS:
        gw = target["gw"]
        path = GW_FILES[gw]

        if not Path(path).exists():
            raise FileNotFoundError(f"Missing {gw} h5ad file: {path}")

        print(f"Reading obs for {gw}: {path}")

        a = sc.read_h5ad(path, backed="r")
        obs = a.obs

        for c in [SAMPLE_COL, REGION_COL]:
            if c not in obs.columns:
                safe_close_backed(a)
                raise KeyError(f"{gw} is missing obs['{c}']")

        for region in target["regions"]:
            region_mask = obs[REGION_COL].astype(str).values == str(region)

            if int(region_mask.sum()) == 0:
                warnings.warn(f"{gw} region {region} has zero cells; skipping.")
                continue

            samples_here = sorted(pd.unique(obs.loc[region_mask, SAMPLE_COL].astype(str)))

            if SPLIT_MULTI_SAMPLE_REGIONS and len(samples_here) > 1:
                for sample in samples_here:
                    mask = region_mask & (
                        obs[SAMPLE_COL].astype(str).values == str(sample)
                    )
                    n_cells = int(mask.sum())

                    panel_specs.append(
                        {
                            "gw": gw,
                            "time": TIME_MAP[gw],
                            "path": Path(path),
                            "sample": sample,
                            "region": region,
                            "panel_key": f"{gw} | {sample} | {region}",
                            "n_full": n_cells,
                        }
                    )

            else:
                sample_value = samples_here[0] if len(samples_here) == 1 else None
                sample_label = samples_here[0] if len(samples_here) == 1 else "pooled_samples"
                n_cells = int(region_mask.sum())

                panel_specs.append(
                    {
                        "gw": gw,
                        "time": TIME_MAP[gw],
                        "path": Path(path),
                        "sample": sample_value,
                        "region": region,
                        "panel_key": f"{gw} | {sample_label} | {region}",
                        "n_full": n_cells,
                    }
                )

        safe_close_backed(a)

    panel_df = pd.DataFrame(panel_specs)
    panel_df = (
        panel_df
        .sort_values(["time", "gw", "sample", "region", "panel_key"])
        .reset_index(drop=True)
    )
    panel_df["slice_id"] = np.arange(len(panel_df), dtype=int)

    return panel_df


# ============================================================
# Load counts, spatial coordinates, and metadata
# ============================================================

def load_qian_pfc_10slice_for_flow_stp(panel_df):
    counts_by_slice = []
    spatial_by_slice = []
    obs_chunks = []
    gene_names = None

    print("\nLoading Qian PFC panels for Flow-STP")
    print("====================================")

    for _, row in panel_df.iterrows():
        slice_id = int(row["slice_id"])
        gw = str(row["gw"])
        path = Path(row["path"])
        sample = row["sample"]
        region = row["region"]
        panel_key = str(row["panel_key"])

        print(f"\nLoading slice {slice_id:02d}: {panel_key}")
        print(f" h5ad: {path}")

        a = sc.read_h5ad(path, backed="r")
        obs = a.obs

        for c in [SAMPLE_COL, REGION_COL]:
            if c not in obs.columns:
                safe_close_backed(a)
                raise KeyError(f"{gw} is missing obs['{c}']")

        if SPATIAL_KEY not in a.obsm:
            safe_close_backed(a)
            raise KeyError(f"Missing obsm['{SPATIAL_KEY}'] in {path}")

        spec = {
            "gw": gw,
            "path": path,
            "sample": sample,
            "region": region,
            "panel_key": panel_key,
        }

        mask = get_panel_mask(obs, spec)
        idx = np.where(mask)[0].astype(np.int64)
        idx = np.sort(idx)

        if len(idx) == 0:
            safe_close_backed(a)
            raise ValueError(f"Slice {slice_id} has zero cells: {panel_key}")

        if MAX_CELLS_PER_SLICE is not None and len(idx) > MAX_CELLS_PER_SLICE:
            rng = np.random.default_rng(RANDOM_SEED + slice_id)
            idx = np.sort(rng.choice(idx, size=MAX_CELLS_PER_SLICE, replace=False))
            print(f" downsampled to {len(idx):,} cells")

        spatial_all = np.asarray(a.obsm[SPATIAL_KEY], dtype=np.float32)
        spatial = spatial_all[idx]

        finite_spatial = np.isfinite(spatial).all(axis=1)

        if finite_spatial.sum() < len(finite_spatial):
            n_drop = len(finite_spatial) - int(finite_spatial.sum())
            print(f" dropping {n_drop:,} cells with non-finite spatial coords")

            idx = idx[finite_spatial]
            spatial = spatial[finite_spatial]

        X = read_h5ad_rows_direct(path, idx, layer_key=LAYER_KEY_FOR_COUNTS)
        X = np.maximum(X, 0).astype(np.float32, copy=False)

        gene_names_arr = np.asarray(a.var_names).astype(str)

        if MAX_GENES_FOR_DEBUG is not None and X.shape[1] > MAX_GENES_FOR_DEBUG:
            X = X[:, :MAX_GENES_FOR_DEBUG]
            gene_names_arr = gene_names_arr[:MAX_GENES_FOR_DEBUG]

        if gene_names is None:
            gene_names = gene_names_arr.tolist()
        else:
            if len(gene_names) != len(gene_names_arr) or not np.array_equal(
                np.asarray(gene_names), gene_names_arr
            ):
                safe_close_backed(a)
                raise ValueError(
                    "Gene names/order differ across h5ad files. "
                    "The current prototype assumes identical gene order across slices."
                )

        if LABEL_COL in obs.columns:
            h1 = normalize_h1(obs.iloc[idx][LABEL_COL].values)
        else:
            h1 = np.array(["Unknown"] * len(idx), dtype=str)

        obs_names = np.asarray(a.obs_names.astype(str))
        cell_ids = obs_names[idx].astype(str)

        obs_i = pd.DataFrame(
            {
                "cell_uid": [f"{gw}:{cid}" for cid in cell_ids],
                "cell_id": cell_ids,
                "row_index": idx,
                "slice_id": slice_id,
                "panel_key": panel_key,
                "stage": gw,
                "GW": gw,
                "gw": gw,
                "time": float(row["time"]),
                "sample": "" if pd.isna(sample) else str(sample),
                "region": "" if pd.isna(region) else str(region),
                "H1_annotation_relabel": h1,
                "H1": h1,
                "spatial_x": spatial[:, 0],
                "spatial_y": spatial[:, 1],
            }
        )

        counts_by_slice.append(torch.from_numpy(X))
        spatial_by_slice.append(spatial.astype(np.float32, copy=False))
        obs_chunks.append(obs_i)

        print(f" cells: {len(idx):,}")
        print(f" expression shape: {X.shape}")
        print(f" spatial shape: {spatial.shape}")
        print(f" count mean: {float(X.mean()):.4f}")
        print(f" count max: {float(X.max()):.4f}")

        safe_close_backed(a)

        del X, spatial_all, spatial
        gc.collect()

    obs_df = pd.concat(obs_chunks, axis=0, ignore_index=True)

    print("\nFinished loading Qian PFC Flow-STP input data")
    print(f" total cells: {obs_df.shape[0]:,}")
    print(f" total slices: {len(counts_by_slice)}")
    print(f" genes: {len(gene_names):,}")

    return counts_by_slice, obs_df, spatial_by_slice, gene_names


def load_qian_pfc_for_flow_stp(
    data_root=None,
    max_cells_per_slice=None,
    max_genes_for_debug=None,
    layer_key_for_counts=None,
    split_multi_sample_regions=True,
    random_seed=20260622,
    expected_num_pfc_slices=10,
):
    """
    Convenience loader for the Qian et al. MERFISH fetal cortex PFC subset.

    Returns:
        counts_by_slice, obs_df, spatial_by_slice, gene_names, panel_df

    Notes:
        The source h5ad files remain in the shared lab dataset folder.
        This function reads the selected PFC panels and returns in-memory
        objects compatible with the Flow-STP training notebooks.
    """
    global DATA_ROOT
    global GW_FILES
    global MAX_CELLS_PER_SLICE
    global MAX_GENES_FOR_DEBUG
    global LAYER_KEY_FOR_COUNTS
    global SPLIT_MULTI_SAMPLE_REGIONS
    global RANDOM_SEED
    global EXPECTED_NUM_PFC_SLICES

    if data_root is not None:
        DATA_ROOT = Path(data_root)
        GW_FILES = {
            "GW15": DATA_ROOT / "gw15.h5ad",
            "GW20": DATA_ROOT / "gw20.h5ad",
            "GW22": DATA_ROOT / "gw22.h5ad",
            "GW34": DATA_ROOT / "gw34.h5ad",
        }

    MAX_CELLS_PER_SLICE = max_cells_per_slice
    MAX_GENES_FOR_DEBUG = max_genes_for_debug
    LAYER_KEY_FOR_COUNTS = layer_key_for_counts
    SPLIT_MULTI_SAMPLE_REGIONS = bool(split_multi_sample_regions)
    RANDOM_SEED = int(random_seed)
    EXPECTED_NUM_PFC_SLICES = int(expected_num_pfc_slices)

    panel_df = build_qian_pfc_panel_specs()

    print("\nPFC panels selected:")
    print(
        panel_df[
            [
                "slice_id",
                "panel_key",
                "gw",
                "time",
                "sample",
                "region",
                "n_full",
                "path",
            ]
        ].to_string(index=False)
    )

    if len(panel_df) != EXPECTED_NUM_PFC_SLICES:
        warnings.warn(
            f"Expected {EXPECTED_NUM_PFC_SLICES} PFC slices, but built {len(panel_df)}. "
            "This may be fine if the dataset/sample-splitting differs from expectation."
        )

    counts_by_slice, obs_df, spatial_by_slice, gene_names = (
        load_qian_pfc_10slice_for_flow_stp(panel_df)
    )

    print("\nReady for Flow-STP training.")
    print(f"num_slices = {len(counts_by_slice)}")
    print(f"num_genes = {len(gene_names)}")
    print(f"obs_df shape = {obs_df.shape}")
    print(f"First count matrix shape = {tuple(counts_by_slice[0].shape)}")

    return counts_by_slice, obs_df, spatial_by_slice, gene_names, panel_df


__all__ = [
    "build_qian_pfc_panel_specs",
    "load_qian_pfc_10slice_for_flow_stp",
    "load_qian_pfc_for_flow_stp",
]

