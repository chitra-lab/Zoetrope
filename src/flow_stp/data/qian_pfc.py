"""Qian et al. PFC loading utilities for the Flow-STP Stage 1 pipeline.

The public loader returns the exact objects consumed by ``run_stage1a``:
counts_by_slice, spatial_by_slice, panel_df, gene_names, and obs_df.

Shared H1/H2/H3 metadata may be supplied as a dataframe or as a CSV, TSV,
Parquet, or Feather file. The merge is explicit so a lab-specific path is
never guessed silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import gc
import re
import warnings

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


DEFAULT_DATA_ROOT = Path(
    "/scratch/uchitra1/datasets/spatial-RNA/merfish/"
    "merfish_cortex_atlas/zenodo_14422018_processed"
)

DEFAULT_TARGETS = (
    ("GW15", ("F1a", "F1b")),
    ("GW20", ("F1", "F2a", "F2b")),
    ("GW22", ("F1", "F2", "F3")),
    ("GW34", ("BA9",)),
)

DEFAULT_TIME_MAP = {
    "GW15": 15.0,
    "GW20": 20.0,
    "GW22": 22.0,
    "GW34": 34.0,
}

DEFAULT_H1_RELABEL_MAP = {
    "EN-IT-1": "EN-IT",
    "EN-IT-2": "EN-IT",
}


@dataclass(frozen=True)
class QianPFCConfig:
    """Configuration for the ten-slice Qian PFC loader."""

    data_root: Path = DEFAULT_DATA_ROOT
    targets: tuple[tuple[str, tuple[str, ...]], ...] = DEFAULT_TARGETS
    time_map: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_TIME_MAP)
    )

    region_column: str = "region"
    sample_column: str = "sample"
    spatial_key: str = "spatial"
    # Count source:
    #   "raw" -> /raw/X       (original QC-retained MERFISH counts)
    #   None  -> /X           (processed/log-normalized expression)
    #   other -> /layers/<key>
    layer_key_for_counts: Optional[str] = "raw"

    # Annotation columns read directly from each h5ad when available.
    h1_source_columns: tuple[str, ...] = (
        "H1_annotation",
        "H1",
    )
    h2_source_columns: tuple[str, ...] = (
        "H2_annotation",
        "H2",
    )
    h3_source_columns: tuple[str, ...] = (
        "H3_annotation",
        "H3",
    )
    h1_relabel_map: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_H1_RELABEL_MAP)
    )

    max_cells_per_slice: Optional[int] = None
    max_genes_for_debug: Optional[int] = None
    split_multi_sample_regions: bool = True
    random_seed: int = 20260622
    expected_num_slices: Optional[int] = 10

    # Optional lab-shared annotation table. This must be an exact file path,
    # not a directory whose contents would have to be guessed.
    shared_metadata_path: Optional[Path] = None
    shared_metadata_join_on: Optional[tuple[str, ...]] = None
    shared_metadata_columns: tuple[str, ...] = (
        "H1",
        "H2",
        "H3",
        "H1_annotation",
        "H2_annotation",
        "H3_annotation",
    )
    shared_metadata_overwrite: bool = True

    def resolved_data_root(self) -> Path:
        return Path(self.data_root).expanduser().resolve()

    def gw_files(self) -> dict[str, Path]:
        root = self.resolved_data_root()
        return {
            "GW15": root / "gw15.h5ad",
            "GW20": root / "gw20.h5ad",
            "GW22": root / "gw22.h5ad",
            "GW34": root / "gw34.h5ad",
        }


@dataclass
class QianPFCData:
    """Stage 1-ready Qian PFC atlas data."""

    counts_by_slice: list[torch.Tensor]
    obs_df: pd.DataFrame
    panel_df: pd.DataFrame
    gene_names: list[str]
    spatial_by_slice: list[np.ndarray]

    @property
    def num_slices(self) -> int:
        return len(self.counts_by_slice)

    @property
    def num_genes(self) -> int:
        return len(self.gene_names)

    @property
    def num_cells(self) -> int:
        return int(sum(int(x.shape[0]) for x in self.counts_by_slice))

    def as_stage1_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted directly by ``run_stage1a``."""
        return {
            "counts_by_slice": self.counts_by_slice,
            "spatial_by_slice": self.spatial_by_slice,
            "panel_df": self.panel_df,
            "gene_names": self.gene_names,
            "obs_df": self.obs_df,
        }



def sanitize_name(value: Any) -> str:
    value = str(value)
    value = re.sub(r"[^\w\-.]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def safe_close_backed(adata: Any) -> None:
    try:
        adata.file.close()
    except Exception:
        pass


def _first_existing_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:
    for column in candidates:
        if column in frame.columns:
            return str(column)
    return None


def _normalize_annotation(
    values: Sequence[Any],
    *,
    relabel_map: Optional[Mapping[str, str]] = None,
) -> np.ndarray:
    series = pd.Series(values)
    series = series.where(~series.isna(), "Unknown").astype(str)
    if relabel_map:
        series = series.replace(dict(relabel_map))
    return series.to_numpy(dtype=str)


def get_panel_mask(
    obs: pd.DataFrame,
    *,
    region: str,
    sample: Optional[str],
    region_column: str,
    sample_column: str,
) -> np.ndarray:
    mask = obs[region_column].astype(str).to_numpy() == str(region)

    if sample is not None and not pd.isna(sample):
        mask &= (
            obs[sample_column].astype(str).to_numpy()
            == str(sample)
        )

    return mask


def as_dense_float32(matrix: Any) -> np.ndarray:
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    else:
        matrix = np.asarray(matrix)
    return matrix.astype(np.float32, copy=False)


def _decode_h5_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _get_h5_matrix_object(
    file_handle: h5py.File,
    layer_key: Optional[str] = None,
) -> Any:
    """Return X, raw/X, or a named AnnData layer."""

    if layer_key is None:
        return file_handle["X"]

    if layer_key == "raw":
        if "raw" not in file_handle or "X" not in file_handle["raw"]:
            raise KeyError(
                "Raw counts were requested, but this h5ad has no /raw/X."
            )
        return file_handle["raw"]["X"]

    return file_handle["layers"][layer_key]


def _get_h5_shape(matrix_object: Any) -> tuple[int, int]:
    shape = matrix_object.attrs.get("shape", None)

    if shape is None and "shape" in matrix_object:
        shape = matrix_object["shape"][()]

    if shape is None:
        raise ValueError(
            "Could not determine matrix shape from h5ad object."
        )

    return tuple(int(value) for value in shape)


def read_h5ad_rows_direct(
    filename: Path | str,
    indices: Sequence[int],
    layer_key: Optional[str] = None,
) -> np.ndarray:
    """Read selected rows directly from a dense, CSR, or CSC h5ad matrix."""
    filename = str(filename)
    indices = np.asarray(indices, dtype=np.int64)

    if len(indices) == 0:
        raise ValueError("Requested zero rows from expression matrix.")

    order = np.argsort(indices)
    sorted_indices = indices[order]

    with h5py.File(filename, "r") as file_handle:
        matrix_object = _get_h5_matrix_object(
            file_handle,
            layer_key=layer_key,
        )

        if isinstance(matrix_object, h5py.Group):
            encoding_type = _decode_h5_attr(
                matrix_object.attrs.get("encoding-type", "")
            )
            n_obs, n_vars = _get_h5_shape(matrix_object)

            data_dataset = matrix_object["data"]
            indices_dataset = matrix_object["indices"]
            indptr_dataset = matrix_object["indptr"]

            if encoding_type == "csr_matrix":
                breaks = np.where(np.diff(sorted_indices) != 1)[0] + 1
                runs = np.split(sorted_indices, breaks)
                sparse_pieces = []

                for rows in runs:
                    row_start = int(rows[0])
                    row_stop = int(rows[-1]) + 1

                    indptr_block = indptr_dataset[
                        row_start : row_stop + 1
                    ].astype(np.int64)
                    data_start = int(indptr_block[0])
                    data_stop = int(indptr_block[-1])

                    data_block = data_dataset[data_start:data_stop]
                    column_indices = indices_dataset[
                        data_start:data_stop
                    ].astype(np.int64)
                    indptr_block = indptr_block - indptr_block[0]

                    sparse_pieces.append(
                        sp.csr_matrix(
                            (
                                data_block,
                                column_indices,
                                indptr_block,
                            ),
                            shape=(row_stop - row_start, n_vars),
                        )
                    )

                sorted_matrix = (
                    sparse_pieces[0]
                    if len(sparse_pieces) == 1
                    else sp.vstack(sparse_pieces, format="csr")
                )
                sorted_matrix = as_dense_float32(sorted_matrix)

            elif encoding_type == "csc_matrix":
                data = data_dataset[:]
                row_indices = indices_dataset[:].astype(np.int64)
                indptr = indptr_dataset[:].astype(np.int64)

                full_matrix = sp.csc_matrix(
                    (data, row_indices, indptr),
                    shape=(n_obs, n_vars),
                )
                sorted_matrix = as_dense_float32(
                    full_matrix[sorted_indices, :]
                )

            else:
                raise ValueError(
                    "Unsupported sparse encoding-type: "
                    f"{encoding_type!r}"
                )

        elif isinstance(matrix_object, h5py.Dataset):
            sorted_matrix = as_dense_float32(
                matrix_object[sorted_indices, :]
            )

        else:
            raise TypeError(
                "Unsupported HDF5 object type: "
                f"{type(matrix_object)!r}"
            )

    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(len(order))

    return sorted_matrix[inverse_order].astype(
        np.float32,
        copy=False,
    )


def build_qian_pfc_panel(
    config: Optional[QianPFCConfig] = None,
) -> pd.DataFrame:
    """Build the ordered Qian PFC panel table."""
    if config is None:
        config = QianPFCConfig()

    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError(
            "Loading Qian h5ad files requires scanpy."
        ) from exc

    gw_files = config.gw_files()
    panel_specs: list[dict[str, Any]] = []

    for gw, regions in config.targets:
        if gw not in gw_files:
            raise KeyError(f"No h5ad filename configured for {gw!r}.")
        if gw not in config.time_map:
            raise KeyError(f"No time configured for {gw!r}.")

        path = gw_files[gw]

        if not path.exists():
            raise FileNotFoundError(f"Missing {gw} h5ad file: {path}")

        print(f"Reading obs for {gw}: {path}")
        adata = sc.read_h5ad(path, backed="r")

        try:
            obs = adata.obs

            for column in (
                config.sample_column,
                config.region_column,
            ):
                if column not in obs.columns:
                    raise KeyError(
                        f"{gw} is missing obs[{column!r}]"
                    )

            for region in regions:
                region_mask = (
                    obs[config.region_column]
                    .astype(str)
                    .to_numpy()
                    == str(region)
                )

                if int(region_mask.sum()) == 0:
                    warnings.warn(
                        f"{gw} region {region} has zero cells; "
                        "skipping."
                    )
                    continue

                samples_here = sorted(
                    pd.unique(
                        obs.loc[
                            region_mask,
                            config.sample_column,
                        ].astype(str)
                    )
                )

                if (
                    config.split_multi_sample_regions
                    and len(samples_here) > 1
                ):
                    for sample in samples_here:
                        mask = region_mask & (
                            obs[config.sample_column]
                            .astype(str)
                            .to_numpy()
                            == str(sample)
                        )

                        panel_specs.append(
                            {
                                "gw": gw,
                                "time": float(config.time_map[gw]),
                                "path": path,
                                "sample": sample,
                                "region": region,
                                "panel_key": (
                                    f"{gw} | {sample} | {region}"
                                ),
                                "n_full": int(mask.sum()),
                            }
                        )
                else:
                    sample_value = (
                        samples_here[0]
                        if len(samples_here) == 1
                        else None
                    )
                    sample_label = (
                        samples_here[0]
                        if len(samples_here) == 1
                        else "pooled_samples"
                    )

                    panel_specs.append(
                        {
                            "gw": gw,
                            "time": float(config.time_map[gw]),
                            "path": path,
                            "sample": sample_value,
                            "region": region,
                            "panel_key": (
                                f"{gw} | {sample_label} | {region}"
                            ),
                            "n_full": int(region_mask.sum()),
                        }
                    )
        finally:
            safe_close_backed(adata)

    panel_df = pd.DataFrame(panel_specs)

    if panel_df.empty:
        raise RuntimeError("No Qian PFC panels were selected.")

    panel_df = (
        panel_df
        .sort_values(
            ["time", "gw", "sample", "region", "panel_key"],
            na_position="last",
        )
        .reset_index(drop=True)
    )
    panel_df["slice_id"] = np.arange(
        len(panel_df),
        dtype=int,
    )

    if (
        config.expected_num_slices is not None
        and len(panel_df) != int(config.expected_num_slices)
    ):
        warnings.warn(
            f"Expected {config.expected_num_slices} PFC slices, "
            f"but built {len(panel_df)}. This may be valid when "
            "sample splitting differs from the expected atlas."
        )

    return panel_df


def read_shared_metadata(
    source: Path | str | pd.DataFrame,
) -> pd.DataFrame:
    """Read a supported shared annotation table."""
    if isinstance(source, pd.DataFrame):
        return source.copy()

    path = Path(source).expanduser().resolve()

    if path.is_dir():
        raise IsADirectoryError(
            "shared_metadata_path must identify the exact metadata "
            f"file, not only its directory: {path}"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"Shared metadata file was not found: {path}"
        )

    name = path.name.lower()

    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return pd.read_csv(path)
    if name.endswith(".tsv") or name.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t")
    if name.endswith(".parquet"):
        return pd.read_parquet(path)
    if name.endswith(".feather"):
        return pd.read_feather(path)

    raise ValueError(
        "Unsupported shared metadata format. Use CSV, TSV, "
        "Parquet, or Feather."
    )


def _choose_metadata_join_columns(
    obs_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    requested: Optional[Sequence[str]],
) -> tuple[str, ...]:
    if requested is not None:
        columns = tuple(str(x) for x in requested)
        missing_obs = [x for x in columns if x not in obs_df.columns]
        missing_metadata = [
            x for x in columns if x not in metadata_df.columns
        ]

        if missing_obs or missing_metadata:
            raise KeyError(
                "Requested metadata join columns are unavailable. "
                f"Missing from obs: {missing_obs}; "
                f"missing from metadata: {missing_metadata}."
            )
        return columns

    candidate_sets = (
        ("cell_uid",),
        ("gw", "cell_id"),
        ("GW", "cell_id"),
        ("stage", "cell_id"),
        ("cell_id",),
    )

    for columns in candidate_sets:
        if all(x in obs_df.columns for x in columns) and all(
            x in metadata_df.columns for x in columns
        ):
            return columns

    raise KeyError(
        "Could not infer a shared-metadata join key. Supply "
        "shared_metadata_join_on explicitly. Preferred keys are "
        "cell_uid or the pair (gw, cell_id)."
    )


def merge_shared_metadata(
    obs_df: pd.DataFrame,
    metadata: Path | str | pd.DataFrame,
    *,
    join_on: Optional[Sequence[str]] = None,
    columns: Optional[Sequence[str]] = None,
    overwrite: bool = True,
) -> pd.DataFrame:
    """Merge H1/H2/H3 metadata into the cell-level observation table."""
    metadata_df = read_shared_metadata(metadata)
    join_columns = _choose_metadata_join_columns(
        obs_df,
        metadata_df,
        join_on,
    )

    if metadata_df.duplicated(list(join_columns)).any():
        duplicate_count = int(
            metadata_df.duplicated(
                list(join_columns),
                keep=False,
            ).sum()
        )
        raise ValueError(
            "Shared metadata contains duplicate join keys "
            f"({duplicate_count} duplicate rows) for "
            f"{join_columns}."
        )

    requested_columns = (
        tuple(columns)
        if columns is not None
        else tuple(
            x
            for x in (
                "H1",
                "H2",
                "H3",
                "H1_annotation",
                "H2_annotation",
                "H3_annotation",
            )
            if x in metadata_df.columns
        )
    )

    available_columns = [
        x
        for x in requested_columns
        if x in metadata_df.columns
        and x not in join_columns
    ]

    if not available_columns:
        raise KeyError(
            "The shared metadata table contains none of the "
            "requested H1/H2/H3 annotation columns."
        )

    right = metadata_df[
        [*join_columns, *available_columns]
    ].copy()

    if overwrite:
        left = obs_df.drop(
            columns=[
                x for x in available_columns
                if x in obs_df.columns
            ],
            errors="ignore",
        )
    else:
        left = obs_df.copy()
        right = right.rename(
            columns={
                x: f"{x}_shared"
                for x in available_columns
            }
        )

    merged = left.merge(
        right,
        on=list(join_columns),
        how="left",
        validate="many_to_one",
        sort=False,
    )

    for level in ("H1", "H2", "H3"):
        annotation_column = f"{level}_annotation"

        if level not in merged.columns:
            if annotation_column in merged.columns:
                merged[level] = _normalize_annotation(
                    merged[annotation_column]
                )
            else:
                merged[level] = "Unknown"
        else:
            merged[level] = _normalize_annotation(
                merged[level]
            )

    return merged


def validate_qian_pfc_data(data: QianPFCData) -> None:
    """Validate the invariants required by the Stage 1 API."""
    n_slices = len(data.counts_by_slice)

    if n_slices == 0:
        raise ValueError("No count matrices were loaded.")
    if len(data.spatial_by_slice) != n_slices:
        raise ValueError(
            "counts_by_slice and spatial_by_slice lengths differ."
        )
    if len(data.panel_df) != n_slices:
        raise ValueError(
            "panel_df row count does not match the number of slices."
        )
    if not data.gene_names:
        raise ValueError("gene_names is empty.")

    total_cells = 0

    for slice_id, (counts, spatial) in enumerate(
        zip(data.counts_by_slice, data.spatial_by_slice)
    ):
        if not torch.is_tensor(counts):
            raise TypeError(
                f"counts_by_slice[{slice_id}] is not a tensor."
            )
        if counts.ndim != 2:
            raise ValueError(
                f"counts_by_slice[{slice_id}] must be 2D."
            )
        if counts.shape[1] != len(data.gene_names):
            raise ValueError(
                f"Slice {slice_id} gene dimension differs from "
                "gene_names."
            )

        spatial_array = np.asarray(spatial)
        if spatial_array.ndim != 2 or spatial_array.shape[1] < 2:
            raise ValueError(
                f"spatial_by_slice[{slice_id}] must have shape "
                "[N, 2+]."
            )
        if spatial_array.shape[0] != counts.shape[0]:
            raise ValueError(
                f"Slice {slice_id} count/spatial row mismatch."
            )
        if not np.isfinite(spatial_array[:, :2]).all():
            raise ValueError(
                f"Slice {slice_id} contains non-finite coordinates."
            )

        total_cells += int(counts.shape[0])

    if len(data.obs_df) != total_cells:
        raise ValueError(
            "obs_df row count does not equal the total number of "
            "loaded cells."
        )


def load_qian_pfc_for_stage1(
    config: Optional[QianPFCConfig] = None,
    *,
    panel_df: Optional[pd.DataFrame] = None,
    shared_metadata: Optional[
        Path | str | pd.DataFrame
    ] = None,
) -> QianPFCData:
    """Load Qian PFC slices and return Stage 1-ready data."""
    if config is None:
        config = QianPFCConfig()

    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError(
            "Loading Qian h5ad files requires scanpy."
        ) from exc

    if panel_df is None:
        panel_df = build_qian_pfc_panel(config)
    else:
        panel_df = panel_df.copy().reset_index(drop=True)

    counts_by_slice: list[torch.Tensor] = []
    spatial_by_slice: list[np.ndarray] = []
    obs_chunks: list[pd.DataFrame] = []
    gene_names: Optional[list[str]] = None

    print("\nLoading Qian PFC panels for Flow-STP")
    print("====================================")

    for _, row in panel_df.iterrows():
        slice_id = int(row["slice_id"])
        gw = str(row["gw"])
        path = Path(row["path"])
        sample = row["sample"]
        region = str(row["region"])
        panel_key = str(row["panel_key"])

        print(f"\nLoading slice {slice_id:02d}: {panel_key}")
        print(f" h5ad: {path}")

        adata = sc.read_h5ad(path, backed="r")

        try:
            obs = adata.obs

            for column in (
                config.sample_column,
                config.region_column,
            ):
                if column not in obs.columns:
                    raise KeyError(
                        f"{gw} is missing obs[{column!r}]"
                    )

            if config.spatial_key not in adata.obsm:
                raise KeyError(
                    f"Missing obsm[{config.spatial_key!r}] "
                    f"in {path}"
                )

            mask = get_panel_mask(
                obs,
                region=region,
                sample=sample,
                region_column=config.region_column,
                sample_column=config.sample_column,
            )
            indices = np.sort(
                np.where(mask)[0].astype(np.int64)
            )

            if len(indices) == 0:
                raise ValueError(
                    f"Slice {slice_id} has zero cells: "
                    f"{panel_key}"
                )

            if (
                config.max_cells_per_slice is not None
                and len(indices) > config.max_cells_per_slice
            ):
                rng = np.random.default_rng(
                    config.random_seed + slice_id
                )
                indices = np.sort(
                    rng.choice(
                        indices,
                        size=config.max_cells_per_slice,
                        replace=False,
                    )
                )
                print(
                    f" downsampled to {len(indices):,} cells"
                )

            spatial_all = np.asarray(
                adata.obsm[config.spatial_key],
                dtype=np.float32,
            )
            spatial = spatial_all[indices, :2]

            finite_spatial = np.isfinite(spatial).all(axis=1)

            if not finite_spatial.all():
                dropped = int((~finite_spatial).sum())
                print(
                    f" dropping {dropped:,} cells with "
                    "non-finite spatial coordinates"
                )
                indices = indices[finite_spatial]
                spatial = spatial[finite_spatial]

            expression = read_h5ad_rows_direct(
                path,
                indices,
                layer_key=config.layer_key_for_counts,
            ).astype(np.float32, copy=False)

            # raw/X owns its own var index. Use it whenever raw
            # counts are requested so matrix columns and gene names
            # remain formally aligned.
            if config.layer_key_for_counts == "raw":
                if adata.raw is None:
                    raise ValueError(
                        f"{path} has no AnnData .raw object."
                    )
                gene_names_array = np.asarray(
                    adata.raw.var_names
                ).astype(str)
            else:
                gene_names_array = np.asarray(
                    adata.var_names
                ).astype(str)

            if (
                config.max_genes_for_debug is not None
                and expression.shape[1]
                > config.max_genes_for_debug
            ):
                expression = expression[
                    :, : config.max_genes_for_debug
                ]
                gene_names_array = gene_names_array[
                    : config.max_genes_for_debug
                ]

            if gene_names is None:
                gene_names = gene_names_array.tolist()
            elif (
                len(gene_names) != len(gene_names_array)
                or not np.array_equal(
                    np.asarray(gene_names),
                    gene_names_array,
                )
            ):
                raise ValueError(
                    "Gene names or ordering differ across h5ad "
                    "files. Stage 1 requires a common ordered "
                    "gene panel."
                )

            selected_obs = obs.iloc[indices]

            h1_column = _first_existing_column(
                selected_obs,
                config.h1_source_columns,
            )
            h2_column = _first_existing_column(
                selected_obs,
                config.h2_source_columns,
            )
            h3_column = _first_existing_column(
                selected_obs,
                config.h3_source_columns,
            )

            h1 = (
                _normalize_annotation(
                    selected_obs[h1_column].to_numpy(),
                    relabel_map=config.h1_relabel_map,
                )
                if h1_column is not None
                else np.full(
                    len(indices),
                    "Unknown",
                    dtype=str,
                )
            )
            h2 = (
                _normalize_annotation(
                    selected_obs[h2_column].to_numpy()
                )
                if h2_column is not None
                else np.full(
                    len(indices),
                    "Unknown",
                    dtype=str,
                )
            )
            h3 = (
                _normalize_annotation(
                    selected_obs[h3_column].to_numpy()
                )
                if h3_column is not None
                else np.full(
                    len(indices),
                    "Unknown",
                    dtype=str,
                )
            )

            obs_names = np.asarray(
                adata.obs_names.astype(str)
            )
            cell_ids = obs_names[indices].astype(str)

            obs_i = pd.DataFrame(
                {
                    "cell_uid": [
                        f"{gw}:{cell_id}"
                        for cell_id in cell_ids
                    ],
                    "cell_id": cell_ids,
                    "row_index": indices,
                    "slice_id": slice_id,
                    "panel_key": panel_key,
                    "stage": gw,
                    "GW": gw,
                    "gw": gw,
                    "time": float(row["time"]),
                    "sample": (
                        ""
                        if pd.isna(sample)
                        else str(sample)
                    ),
                    "region": region,
                    "H1_annotation_relabel": h1,
                    "H1": h1,
                    "H2": h2,
                    "H3": h3,
                    "spatial_x": spatial[:, 0],
                    "spatial_y": spatial[:, 1],
                }
            )

            counts_by_slice.append(
                torch.from_numpy(expression)
            )
            spatial_by_slice.append(
                spatial.astype(np.float32, copy=False)
            )
            obs_chunks.append(obs_i)

            print(f" cells: {len(indices):,}")
            print(f" expression shape: {expression.shape}")
            print(f" spatial shape: {spatial.shape}")
            print(
                f" count mean: {float(expression.mean()):.4f}"
            )
            print(
                f" count max: {float(expression.max()):.4f}"
            )

        finally:
            safe_close_backed(adata)

        del expression, spatial_all, spatial
        gc.collect()

    if gene_names is None:
        raise RuntimeError("No gene names were loaded.")

    obs_df = pd.concat(
        obs_chunks,
        axis=0,
        ignore_index=True,
    )

    metadata_source = (
        shared_metadata
        if shared_metadata is not None
        else config.shared_metadata_path
    )

    if metadata_source is not None:
        obs_df = merge_shared_metadata(
            obs_df,
            metadata_source,
            join_on=config.shared_metadata_join_on,
            columns=config.shared_metadata_columns,
            overwrite=config.shared_metadata_overwrite,
        )

    data = QianPFCData(
        counts_by_slice=counts_by_slice,
        obs_df=obs_df,
        panel_df=panel_df,
        gene_names=gene_names,
        spatial_by_slice=spatial_by_slice,
    )
    validate_qian_pfc_data(data)

    print("\nFinished loading Qian PFC Flow-STP input data")
    print(f" total cells: {data.num_cells:,}")
    print(f" total slices: {data.num_slices}")
    print(f" genes: {data.num_genes:,}")

    return data


# Concise public alias.
load_qian_pfc = load_qian_pfc_for_stage1


__all__ = [
    "DEFAULT_DATA_ROOT",
    "QianPFCConfig",
    "QianPFCData",
    "build_qian_pfc_panel",
    "load_qian_pfc",
    "load_qian_pfc_for_stage1",
    "merge_shared_metadata",
    "read_h5ad_rows_direct",
    "read_shared_metadata",
    "validate_qian_pfc_data",
]

