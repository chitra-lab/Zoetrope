import numpy as np
import torch

from flow_stp.zoo import METHODS, prepare_atlas
from flow_stp.zoo import fisher_diffusion as diffusion
from flow_stp.zoo import geometry_anchored_lowrank as lowrank
from flow_stp.zoo import multiblock_kernel_pls as pls
from flow_stp.zoo import multiscale_score_rayleigh as multiscale
from flow_stp.zoo import rkhs_score_rayleigh as rkhs
from flow_stp.zoo import score_rayleigh_warp as score
from flow_stp.zoo.warps import MonotoneWarpFamily


def _atlas():
    rng = np.random.default_rng(4)
    counts = []
    xy = []
    for t in (0.0, 1.0):
        n = 14
        s = rng.uniform(0.0, 1.0, size=(n, 2)).astype(np.float32)
        z = np.clip(s[:, 0] + 0.15 * s[:, 1], 0.0, 1.0)
        rate = np.stack(
            [
                2.0 + 3.0 * z + 0.5 * t,
                2.0 + 2.0 * (1.0 - z),
            ],
            axis=1,
        )
        counts.append(rng.poisson(rate).astype(np.float32))
        xy.append(s)
    return prepare_atlas(
        counts_by_slice=counts,
        spatial_by_slice=xy,
        times=[0.0, 1.0],
        gene_names=["g0", "g1"],
    )


def _small_geometry(cfg):
    cfg.epochs = 1
    cfg.batch_size = 32
    cfg.coord_hidden = (8,)
    cfg.decoder_hidden = (8,)
    cfg.print_every = 1
    cfg.weights.corrected_scale = 1e-4


def _check(result):
    assert len(result.z_by_tissue) == 2
    for z in result.z_by_tissue:
        assert z.shape == (14,)
        assert np.isfinite(z).all()


def test_identity_warp_is_monotone_and_centered():
    warps = MonotoneWarpFamily(3, degree=3)
    z = torch.linspace(0, 1, 20)
    for i in range(3):
        assert torch.all(warps.derivative(z, i) > 0)
    assert torch.allclose(
        warps.consensus_gauge_penalty(), torch.tensor(0.0)
    )


def test_active_zoo_registry_replaces_retired_methods():
    assert set(METHODS) == {
        "score_rayleigh_warp",
        "multiscale_score_rayleigh",
        "rkhs_score_rayleigh",
        "fisher_diffusion",
        "geometry_anchored_lowrank",
        "multiblock_kernel_pls",
    }


def test_active_zoo_entrypoints_smoke():
    atlas = _atlas()
    device = torch.device("cpu")

    c_score = score.Config(k_neighbors=5)
    _small_geometry(c_score.geometry_fit)
    _check(score.fit(atlas, c_score, device=device))

    c_multi = multiscale.Config(k_neighbors=(4, 5), reconstruction_scale=5)
    _small_geometry(c_multi.geometry_fit)
    _check(multiscale.fit(atlas, c_multi, device=device))

    c_rkhs = rkhs.Config(
        k_neighbors=5,
        n_centers=4,
        epochs=1,
        batch_size=32,
        decoder_hidden=(8,),
        print_every=1,
    )
    c_rkhs.weights.corrected_scale = 1e-4
    _check(rkhs.fit(atlas, c_rkhs, device=device))

    c_diff = diffusion.Config(
        score_k_neighbors=5,
        graph_k_neighbors=5,
        orientation_cells_per_tissue=14,
    )
    _check(diffusion.fit(atlas, c_diff, device=device))

    c_lowrank = lowrank.Config(
        k_neighbors=5,
        latent_rank=1,
        basis_cells_per_tissue=14,
        degree_z=2,
        degree_t=1,
        epochs=1,
        batch_size=32,
        coord_hidden=(8,),
        print_every=1,
    )
    c_lowrank.weights.corrected_scale = 1e-4
    _check(lowrank.fit(atlas, c_lowrank, device=device))

    c_pls = pls.Config(
        n_centers=4,
        fit_cells_per_tissue=14,
    )
    _check(pls.fit(atlas, c_pls, device=device))
