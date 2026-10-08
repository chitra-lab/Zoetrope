import torch
import pytest
from flow_stp.stage1 import FlowSTPConfig, FlowSTPModel, METHOD_VERSION
from flow_stp.stage1.stage1a import _load_torch
from dataclasses import asdict


def test_old_selected_checkpoint_migrates_inert_endpoints_and_optimizer(tmp_path):
    model = FlowSTPModel(FlowSTPConfig(num_slices=1,num_genes=2,slice_times=(0.,)))
    state = model.state_dict()
    state['coordinate_maps.missing_left'] = torch.zeros(1)
    state['coordinate_maps.missing_right'] = torch.zeros(1)
    config = asdict(model.config) | dict(slice_effect_dim=0,max_missing_each_end=0,
        initial_missing=0,field_hidden=(128,128),n_flow_steps=16)
    path = tmp_path/'old.pt'
    torch.save(dict(method_version=METHOD_VERSION,model_config=config,model_state_dict=state,
        optimizer_state_dict={'param_groups':[{'params':[0]},{'params':[1,2]}],
                              'state':{0:{},1:{},2:{}}}),path)
    payload = _load_torch(path,'cpu')
    restored = FlowSTPModel(FlowSTPConfig(**payload['model_config']))
    restored.load_state_dict(payload['model_state_dict'],strict=True)
    assert len(payload['optimizer_state_dict']['param_groups']) == 1
    assert set(payload['optimizer_state_dict']['state']) == {0}
    for a,b in zip(model.parameters(),restored.parameters()):
        assert torch.equal(a,b)


def test_nonzero_coverage_is_rejected_instead_of_silently_changing_predictions(tmp_path):
    path = tmp_path/'bad.pt'
    torch.save(dict(method_version=METHOD_VERSION,model_config={},
        model_state_dict={'coordinate_maps.missing_left':torch.ones(1)}),path)
    with pytest.raises(RuntimeError,match='nonzero coverage'):
        _load_torch(path,'cpu')
