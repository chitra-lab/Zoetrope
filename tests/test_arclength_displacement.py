"""Paired arclength and oracle displacement metrics."""
import numpy as np
import pytest
from flow_stp.tissue_coordinates import AnatomicalContourField
from flow_stp.simulations.arclength_displacement import evaluate_arclength_vs_oracle


def field(shift=0., polygon=False):
    x = np.linspace(-1, 1, 101) + shift
    y = np.linspace(0, 1, 71)
    z = np.meshgrid(x,y)[1]
    kwargs = dict(assume_rectangular_domain=True)
    if polygon:
        pytest.importorskip("shapely")
        kwargs = dict(tissue_polygon=[[x[0],0],[x[-1],0],[x[-1],1],[x[0],1]])
    return AnatomicalContourField(x,y,z,**kwargs)


@pytest.mark.parametrize("polygon",[False,True])
def test_paired_truth_evaluation_does_not_leak_oracle_information(polygon):
    source,target = field(polygon=polygon),field(.2,polygon=polygon)
    initial=np.array([[.25,.2],[-.6,.3]])
    truth=initial+np.array([.2,.5])
    first=evaluate_arclength_vs_oracle(
        initial,truth,[.2,.3],[.7,.8],source,target,cell_ids=[4,7],
    )
    assert first.summary["n_success"] == 2
    assert first.summary["oracle_contour_displacement_nrmse"] == pytest.approx(0.,abs=1e-12)
    assert first.summary["arclength_displacement_nrmse"] == pytest.approx(0.,abs=1e-12)
    altered = truth.copy();altered[:,0]+=.1
    second=evaluate_arclength_vs_oracle(
        initial,altered,[.2,.3],[.7,.8],source,target,
    )
    np.testing.assert_allclose(
        first.per_cell[["predicted_x","predicted_y"]],
        second.per_cell[["predicted_x","predicted_y"]],
    )
    assert second.summary["arclength_displacement_nrmse"]>0


def test_fail_closed_and_recorded_coverage_modes():
    source,target=field(),field(.2)
    initial=np.array([[.2,.2],[500.,500.]])
    truth=np.array([[.4,.8],[500.,500.]])
    with pytest.raises(ValueError,match="Cell 1"):
        evaluate_arclength_vs_oracle(
            initial,truth,[.2,.2],[.8,.8],source,target,
        )
    result=evaluate_arclength_vs_oracle(
        initial,truth,[.2,.2],[.8,.8],source,target,on_error="record",
    )
    assert result.summary["n_failure"]==1
    assert result.summary["success_fraction"]==.5
    assert result.per_cell.success.sum()==1
