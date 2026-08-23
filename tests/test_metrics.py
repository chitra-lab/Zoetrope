import numpy as np
import torch
from torch import nn

from flow_stp.metrics import compute_rpce


class _LinearMaps(nn.Module):
    def __init__(self, lengths):
        super().__init__()
        self.lengths = nn.Parameter(
            torch.as_tensor(lengths, dtype=torch.float32),
            requires_grad=False,
        )

    def one_slice(self, coords, tissue_id):
        return coords[:, 0] / self.lengths[int(tissue_id)]


class _WarpedMaps(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(0.0))

    def one_slice(self, coords, tissue_id):
        if int(tissue_id) == 0:
            return coords[:, 0]
        return coords[:, 0].square()


def _rectangle(rng, length, half_width, n=2000):
    return np.column_stack(
        [
            rng.uniform(0.0, length, n),
            rng.uniform(-half_width, half_width, n),
        ]
    )


def test_rpce_zero_for_consistent_relative_position_up_to_scale():
    rng = np.random.default_rng(7)
    spatial = [
        _rectangle(rng, 2.0, 1.0),
        _rectangle(rng, 5.0, 2.0),
    ]
    result = compute_rpce(
        _LinearMaps([2.0, 5.0]),
        spatial,
        n_z=41,
        n_paths_per_tissue=8,
    )

    assert result.summary["rpce"] < 1e-5
    assert result.summary["rpce_within"] < 1e-5
    assert result.summary["rpce_between"] < 1e-5
    assert result.summary["coordinate_error_max_abs"] < 1e-4


def test_rpce_detects_between_tissue_relative_position_warp():
    rng = np.random.default_rng(11)
    spatial = [
        _rectangle(rng, 1.0, 1.0),
        _rectangle(rng, 1.0, 1.0),
    ]
    result = compute_rpce(
        _WarpedMaps(),
        spatial,
        n_z=41,
        n_paths_per_tissue=8,
        min_grad_norm=1e-5,
    )

    assert result.summary["rpce"] > 0.03
    assert result.summary["rpce_between"] > 0.03
    assert result.summary["rpce_within"] < 1e-4
    assert result.summary["variance_decomposition_error"] < 1e-10
