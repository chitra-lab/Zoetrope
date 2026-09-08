import numpy as np
import torch

from flow_stp.zoo import prepare_atlas
from flow_stp.zoo import joint_likelihood_fisher_warp as m1
from flow_stp.zoo import meanfield_rayleigh_warp as m2
from flow_stp.zoo import score_rayleigh_warp as m3
from flow_stp.zoo import independent_register as m5
from flow_stp.zoo import eigenfield_warp as m6
from flow_stp.zoo import principal_arclength_warp as m7
from flow_stp.zoo.warps import MonotoneWarpFamily


def _atlas():
    rng = np.random.default_rng(4)
    counts = []
    xy = []
    for t in (0.0, 1.0):
        n = 14
        s = rng.uniform(0.0, 1.0, size=(n, 2)).astype(np.float32)
        z = np.clip(s[:, 0] + 0.15 * s[:, 1], 0.0, 1.0)
        rate = np.stack([2.0 + 3.0 * z + 0.5 * t, 2.0 + 2.0 * (1.0 - z)], axis=1)
        counts.append(rng.poisson(rate).astype(np.float32))
        xy.append(s)
    return prepare_atlas(counts_by_slice=counts, spatial_by_slice=xy, times=[0.0, 1.0], gene_names=["g0", "g1"])


def _small_mean(cfg):
    cfg.epochs = 1; cfg.batch_size = 32; cfg.hidden = (8,)


def _small_geometry(cfg):
    cfg.epochs = 1; cfg.batch_size = 32; cfg.coord_hidden = (8,); cfg.decoder_hidden = (8,); cfg.print_every = 1; cfg.weights.corrected_scale = 1e-4


def _small_joint(cfg):
    cfg.epochs = 1; cfg.batch_size = 32; cfg.coord_hidden = (8,); cfg.decoder_hidden = (8,); cfg.print_every = 1; cfg.weights.corrected_scale = 1e-4


def _small_registration(cfg):
    cfg.epochs_soft_orientation = 1; cfg.epochs_hard_refine = 1; cfg.batch_size = 32; cfg.decoder_hidden = (8,)


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
    assert torch.allclose(warps.consensus_gauge_penalty(), torch.tensor(0.0))


def test_ranked_zoo_entrypoints_smoke():
    atlas = _atlas(); device = torch.device("cpu")

    c1 = m1.Config(); _small_mean(c1.mean_field); _small_joint(c1.joint); c1.fisher_batch_size = 32
    _check(m1.fit(atlas, c1, device=device))

    c2 = m2.Config(); _small_mean(c2.mean_field); _small_geometry(c2.geometry_fit); c2.fisher_batch_size = 32
    _check(m2.fit(atlas, c2, device=device))

    c3 = m3.Config(k_neighbors=5); _small_geometry(c3.geometry_fit)
    _check(m3.fit(atlas, c3, device=device))

    c5 = m5.Config(); _small_mean(c5.mean_field); c5.independent.epochs = 1; c5.independent.batch_size = 32; c5.independent.hidden = (8,); _small_registration(c5.registration); c5.fisher_batch_size = 32
    _check(m5.fit(atlas, c5, device=device))

    c6 = m6.Config(); _small_mean(c6.mean_field); _small_registration(c6.registration); c6.fisher_batch_size = 32; c6.k_neighbors = 5; c6.integration_epochs = 3; c6.regressor_epochs = 2
    _check(m6.fit(atlas, c6, device=device))

    c7 = m7.Config(); _small_mean(c7.mean_field); _small_registration(c7.registration); c7.fisher_batch_size = 32; c7.k_neighbors = 5; c7.regressor_epochs = 2
    _check(m7.fit(atlas, c7, device=device))
