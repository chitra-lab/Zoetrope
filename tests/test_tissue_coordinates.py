"""Tests for oriented normalized-arclength reconstruction and oracle evaluation."""
import numpy as np
import pytest
from flow_stp.tissue_coordinates import (
    AnatomicalContourField, OrientedContour,
    reconstruct_by_arclength, oracle_nearest_contour_point,
)


def plane(*, flipped=False, shift=0.0, warped=False, mask=None):
    x = np.linspace(-1., 1., 101)
    y = np.linspace(0., 1., 71)
    xx, yy = np.meshgrid(x, y)
    z = yy if not warped else yy - .15 * xx * xx
    if flipped:
        z = -z
    return AnatomicalContourField(x + shift, y, z, tissue_mask=mask)


def test_orientation_from_rotated_anatomical_gradient():
    c = plane().unique_contour(.5)
    assert c.xy[0, 0] > c.xy[-1, 0]  # J grad(z)=(-1,0)
    np.testing.assert_allclose(c.at_fraction(.25), [.5, .5], atol=1e-12)
    p, fraction, distance = c.project([.25, .5])
    np.testing.assert_allclose(p, [.25, .5], atol=1e-12)
    assert fraction == pytest.approx(.375, abs=1e-12)
    assert distance == pytest.approx(0.)


def test_transfer_fraction_not_oracle_destination():
    source, target = plane(), plane(shift=.4)
    result = reconstruct_by_arclength([.25, .2], .2, .8, source, target)
    assert result.source_fraction == pytest.approx(.375, abs=1e-12)
    np.testing.assert_allclose(result.xy, [.65, .8], atol=1e-12)
    assert result.source_contour_distance < 1e-12
    point, error = oracle_nearest_contour_point([.8, .8], .8, target)
    np.testing.assert_allclose(point, [.8, .8], atol=1e-12)
    assert error < 1e-12
    assert np.linalg.norm(result.xy - [.8, .8]) > .1


def test_orientation_switch_when_anatomical_sign_flipped():
    c = plane(flipped=True).unique_contour(-.5)
    assert c.xy[0, 0] < c.xy[-1, 0]
    np.testing.assert_allclose(c.at_fraction(.25), [-.5, .5], atol=1e-12)


def test_curved_contours_and_arclength_interpolation():
    c = plane(warped=True).unique_contour(.35)
    assert c.length > 1.5
    point = c.at_fraction(.3)
    _, fraction, distance = c.project(point)
    assert fraction == pytest.approx(.3, abs=1e-6)
    assert distance < 1e-10
    assert abs(point[1] - .35) > 1e-4


def test_reject_closed_contours():
    x = np.linspace(-1., 1., 101)
    y = np.linspace(-1., 1., 101)
    xx, yy = np.meshgrid(x, y)
    f = AnatomicalContourField(x, y, xx*xx + yy*yy)
    with pytest.raises(ValueError, match="Closed contours"):
        f.unique_contour(.25)


def test_reject_disconnected_contours():
    x = np.linspace(-1., 1., 101)
    y = np.linspace(-1., 1., 101)
    xx, yy = np.meshgrid(x, y)
    f = AnatomicalContourField(x, y, xx*xx)
    with pytest.raises(ValueError, match="2 contours"):
        f.unique_contour(.5)


def test_tissue_mask_and_unattained_depth():
    x = np.linspace(-1., 1., 101)
    y = np.linspace(0., 1., 71)
    xx, yy = np.meshgrid(x, y)
    f = AnatomicalContourField(x, y, yy, tissue_mask=np.abs(xx) < .6)
    c = f.unique_contour(.5)
    assert np.max(np.abs(c.xy[:, 0])) <= .61
    with pytest.raises(ValueError, match="0 contours"):
        f.unique_contour(2.)


def test_invalid_inputs():
    with pytest.raises(ValueError, match="strictly increasing"):
        AnatomicalContourField([0, 1, 1], [0, 1, 2], np.zeros((3, 3)))
    with pytest.raises(ValueError, match="Closed contours"):
        OrientedContour.from_xy([[0,0],[1,0],[0,0]])
    with pytest.raises(ValueError, match="finite XY"):
        plane().unique_contour(.5).project([np.nan, .5])
