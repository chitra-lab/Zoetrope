"""Regression gates for the single frozen Stage-1 scientific model."""

import numpy as np
import pytest
import torch

from flow_stp.stage1 import (
    METHOD_VERSION, FlowSTPConfig, FlowSTPModel, Stage1ARunConfig, run_stage1a,
)
from flow_stp.stage1.stage1a import _make_model_config


def test_selected_defaults():
    cfg = Stage1ARunConfig()
    assert METHOD_VERSION == "stage1a_shared_logE_global_gauge_v1"
    assert cfg.coordinate_decoder == "shared_auto"
    assert cfg.gauge_mode == "global"
    assert cfg.epochs == 1500
    assert cfg.batch_size_per_slice == 1024
    assert cfg.learning_rate == 2e-3
    assert cfg.coord_hidden == (64, 64)
    assert cfg.decoder_hidden == (128, 128)
    assert cfg.use_library_size_offset is True


@pytest.mark.parametrize(
    "times,expected",
    [
        ([0.0, 0.3, 1.0], "shared_time"),
        ([0.0, 0.0, 1.0], "shared_time_tissue"),
        ([0.0, 0.5, 0.5, 1.0], "shared_time_tissue"),
    ],
)
def test_tissue_multiplicity_automatically_selects_map(times, expected):
    data = {
        "n_slices": len(times),
        "n_genes": 3,
        "raw_times": np.array(times),
        "normalized_times": np.array(times),
    }
    config = _make_model_config(data, Stage1ARunConfig())
    assert config.coordinate_decoder == expected
    assert config.gauge_mode == "global"
    model = FlowSTPModel(config)
    expected_input = 3 + (len(times) if expected == "shared_time_tissue" else 0)
    assert model.coordinate_maps.shared_map.net[0].in_features == expected_input


def test_time_only_and_tissue_conditioned_have_matching_initial_coordinates():
    base = dict(
        num_slices=2, num_genes=3, seed=20260622,
        slice_times=(0.0, 1.0), coord_hidden=(16, 16),
        decoder_hidden=(16, 16),
    )
    time_only = FlowSTPModel(FlowSTPConfig(coordinate_decoder="shared_time", **base))
    with_id = FlowSTPModel(FlowSTPConfig(coordinate_decoder="shared_time_tissue", **base))
    xy = torch.tensor([[0.1, -0.2], [0.6, 0.5], [-1.0, 1.2]])
    for sid in (0, 1):
        assert torch.allclose(
            time_only.coordinate_maps.raw_one_slice(xy, sid),
            with_id.coordinate_maps.raw_one_slice(xy, sid),
            atol=1e-7, rtol=1e-7,
        )
    for a, b in zip(time_only.decoder.parameters(), with_id.decoder.parameters()):
        assert torch.equal(a, b)


def test_log_expression_poisson_field_matches_derived_fisher_coordinates():
    model = FlowSTPModel(
        FlowSTPConfig(num_slices=1, num_genes=2, slice_times=(0.0,),
                      coord_hidden=(8,), decoder_hidden=(8,))
    )
    with torch.no_grad():
        model.log_expression_baseline.copy_(torch.log(torch.tensor([4.0, 9.0])))
        for parameter in model.decoder.parameters():
            parameter.zero_()
    z = torch.tensor([0.2, 0.8])
    times = torch.tensor([0.1, 0.9])
    assert torch.allclose(model.expression(z, times),
                          torch.tensor([[4.0, 9.0], [4.0, 9.0]]))
    assert torch.allclose(model.fisher_flat(z, times),
                          torch.tensor([[4.0, 6.0], [4.0, 6.0]]))
    out = model(
        coords=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        slice_idx=torch.zeros(2, dtype=torch.long),
        time=times,
        log_library_size=torch.log(torch.tensor([2.0, 0.5])),
    )
    assert torch.allclose(
        torch.exp(out["log_mu"]),
        torch.tensor([[8.0, 18.0], [2.0, 4.5]]),
        atol=1e-5, rtol=1e-5,
    )


def test_archived_coordinate_modes_cannot_be_selected_from_active_runner(tmp_path):
    with pytest.raises(ValueError, match="shared_auto/global"):
        run_stage1a(
            counts_by_slice=[], spatial_by_slice=[],
            panel_df=None, gene_names=[],
            config=Stage1ARunConfig(
                output_dir=tmp_path,
                coordinate_decoder="independent",
                gauge_mode="per_tissue",
            ),
        )

