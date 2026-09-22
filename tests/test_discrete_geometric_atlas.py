import numpy as np
import torch

from flow_stp.zoo import METHODS
from flow_stp.zoo.competing_models.discrete_geometric_atlas import (
    AdjacentTransportMap,
    Config,
    DiscreteGeometricAtlas,
    fit_adjacent_transport,
)


def _pair(source_tissue, target_tissue, source, target, increment):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    increment = np.asarray(increment, dtype=float)
    return AdjacentTransportMap(
        source_tissue=source_tissue,
        target_tissue=target_tissue,
        source_time=float(source_tissue),
        target_time=float(target_tissue),
        source_coordinate=source,
        target_coordinate=target,
        progression_increment=increment,
        progression_rate=increment,
        diagnostics={},
    )


def test_discrete_atlas_registered_in_zoo():
    assert "discrete_geometric_atlas" in METHODS


def test_adjacent_map_composition_and_progression():
    grid = np.linspace(0.1, 0.9, 17)
    first = _pair(
        0,
        1,
        grid,
        grid + 0.05,
        np.full_like(grid, 0.2),
    )
    second = _pair(
        1,
        2,
        grid + 0.05,
        grid + 0.10,
        np.full_like(grid, 0.3),
    )
    atlas = DiscreteGeometricAtlas(
        times=np.array([0.0, 1.0, 2.0]),
        ordered_tissues=(0, 1, 2),
        adjacent_maps=(first, second),
    )

    x = np.array([0.2, 0.4, 0.6])
    transported = atlas.transport(
        x,
        source_tissue=0,
        target_tissue=2,
    )
    rho = atlas.progression(
        x,
        source_tissue=0,
        target_tissue=2,
    )

    assert np.allclose(transported, x + 0.10)
    assert np.allclose(rho, 0.5)
    assert np.allclose(
        atlas.transport(
            transported,
            source_tissue=2,
            target_tissue=0,
        ),
        x,
    )


def test_adjacent_fit_identity_fisher_curves():
    grid = np.linspace(0.0, 1.0, 64)
    curve = np.stack(
        [
            3.0 + grid,
            4.0 - 0.5 * grid,
            2.5 + 0.25 * grid**2,
        ],
        axis=1,
    ).astype(np.float32)

    config = Config(
        curve_grid_size=64,
        registration_grid_size=48,
        registration_steps=20,
        registration_lr=1e-2,
        warp_degree=3,
        warp_quadrature_points=16,
        max_missing_each_end=0.10,
        initial_missing=0.0,
        warp_roughness_weight=1e-3,
        warp_magnitude_weight=1e-3,
        overlap_penalty_weight=0.1,
        print_every=100,
    )

    transport, history = fit_adjacent_transport(
        curve,
        curve,
        source_tissue=0,
        target_tissue=1,
        source_time=0.0,
        target_time=0.5,
        config=config,
        device=torch.device("cpu"),
    )

    assert history
    assert np.all(np.diff(transport.source_coordinate) > 0)
    assert np.all(np.diff(transport.target_coordinate) > 0)
    assert np.max(
        np.abs(
            transport.target_coordinate
            - transport.source_coordinate
        )
    ) < 5e-3
    assert np.max(np.abs(transport.progression_increment)) < 1e-3
