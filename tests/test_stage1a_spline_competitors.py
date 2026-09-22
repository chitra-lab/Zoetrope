from __future__ import annotations

import numpy as np

from flow_stp.zoo.common import prepare_atlas
from flow_stp.zoo.competing_models.stage1a_spline import (
    Stage1aSplineJointConfig,
    Stage1aSplineModelConfig,
    Stage1aSplineProfiledConfig,
    fit_stage1a_spline_joint,
    fit_stage1a_spline_profiled,
)


def _tiny_atlas():
    rng = np.random.default_rng(7)
    counts_by_slice = []
    spatial_by_slice = []

    for tissue_id, time in enumerate((0.0, 1.0)):
        n = 24
        z = np.linspace(0.05, 0.95, n)
        x = z + 0.05 * rng.normal(size=n)
        y = 0.2 * rng.normal(size=n)
        spatial_by_slice.append(np.column_stack([x, y]).astype(np.float32))

        mean = np.column_stack(
            [
                2.0 + 3.0 * z,
                4.0 - 2.0 * z + time,
                1.5 + 2.0 * z * time,
            ]
        )
        counts_by_slice.append(rng.poisson(mean).astype(np.float32))

    return prepare_atlas(
        counts_by_slice=counts_by_slice,
        spatial_by_slice=spatial_by_slice,
        times=(0.0, 1.0),
        gene_names=("g0", "g1", "g2"),
        preserve_physical_metric=True,
        use_library_size_offset=False,
    )


def _model_config() -> Stage1aSplineModelConfig:
    return Stage1aSplineModelConfig(
        coord_hidden=(8, 8),
        spatial_basis_dim=4,
        temporal_basis_dim=3,
        spline_degree=3,
        coordinate_domain=(-0.5, 1.5),
        slice_effect_dim=0,
    )


def test_stage1a_spline_joint_smoke() -> None:
    result = fit_stage1a_spline_joint(
        _tiny_atlas(),
        Stage1aSplineJointConfig(
            model=_model_config(),
            epochs=2,
            batch_size_per_tissue=12,
            geometry_cells_per_tissue=8,
            print_every=1,
            seed=11,
        ),
    )

    assert result.method == "stage1a_nn_spline_joint"
    assert len(result.z_by_tissue) == 2
    assert all(np.isfinite(z).all() for z in result.z_by_tissue)
    assert result.auxiliary["expression_backend"] == "tensor_product_bspline"


def test_stage1a_spline_profiled_smoke() -> None:
    result = fit_stage1a_spline_profiled(
        _tiny_atlas(),
        Stage1aSplineProfiledConfig(
            model=_model_config(),
            outer_iterations=1,
            coordinate_steps_per_outer=2,
            batch_size_per_tissue=12,
            geometry_cells_per_tissue=8,
            gauge_refresh_every=1,
            irls_max_iter=3,
            irls_gene_chunk_size=2,
            seed=13,
        ),
    )

    assert result.method == "stage1a_nn_spline_profiled"
    assert len(result.z_by_tissue) == 2
    assert all(np.isfinite(z).all() for z in result.z_by_tissue)
    assert len(result.history) == 2
