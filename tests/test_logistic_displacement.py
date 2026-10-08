import numpy as np
import pytest
import torch

from flow_stp.simulations.logistic_displacement import (
    LogisticDisplacementConfig, generate_logistic_tissues, true_flow,
    contour_lift, displacement_metrics, evaluate_displacement,
)


def test_generator_is_deterministic_and_flow_matches_material_labels():
    cfg = LogisticDisplacementConfig(n_times=3, cells_per_tissue=30)
    a, b = generate_logistic_tissues(cfg), generate_logistic_tissues(cfg)
    for i, t in enumerate(a.times):
        np.testing.assert_array_equal(a.counts[i], b.counts[i])
        np.testing.assert_allclose(a.xy[i][:,0], true_flow(a.material_labels[i],t), atol=1e-7)
    assert set(a.stage1_kwargs()) == {'counts_by_slice','spatial_by_slice','panel_df','gene_names'}


def test_no_motion_and_perfect_endpoint_controls():
    source = np.array([[0.,0.],[1.,0.]])
    truth = source + [0.1,0.]
    assert displacement_metrics(source,truth,source)['pooled_displacement_nrmse'] == pytest.approx(1)
    assert displacement_metrics(truth,truth,source)['pooled_motion_skill'] == 1
    xy = np.array([[0.,0.],[0.5,0.1],[1.,0.]])
    np.testing.assert_array_equal(contour_lift(xy[:,0],xy[:,1],xy[:,0],xy,k=1),xy)


def test_oracle_logistic_flow_recovers_both_backward_and_forward_endpoints():
    data = generate_logistic_tissues(LogisticDisplacementConfig(
        n_times=5,cells_per_tissue=1000,n_source=8,contour_k=1,max_dt=0.05))
    # Isolate axial integration from finite transverse contour-lift resolution.
    for xy in data.xy:
        xy[:,1] = 0
    h = torch.tensor([1.,-1.])
    def field(z,t):
        z,t = z.reshape(-1),t.reshape(-1)
        e = torch.exp(0.8*t)
        a = z/(e-z*(e-1))
        # Pure material expression: Fisher V equals physical logistic velocity.
        return 10 + a[:,None]*h
    summary, by = evaluate_displacement(data,[xy[:,0] for xy in data.xy],field)
    assert by.time.min() < data.times[summary['source_id']] < by.time.max()
    assert summary['pooled_displacement_nrmse'] < 0.1
    assert summary['pooled_displacement_nrmse'] == pytest.approx(summary['oracle_lift_nrmse'],abs=1e-5)


def test_zero_motion_has_undefined_normalized_error():
    points = np.zeros((2,2))
    assert np.isnan(displacement_metrics(points,points,points)['pooled_displacement_nrmse'])
