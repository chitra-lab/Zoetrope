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
    return AnatomicalContourField(x + shift, y, z, tissue_mask=mask, assume_rectangular_domain=(mask is None))


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
    f = AnatomicalContourField(x, y, xx*xx + yy*yy, assume_rectangular_domain=True)
    with pytest.raises(ValueError, match="Closed contours"):
        f.unique_contour(.25)


def test_reject_disconnected_contours():
    x = np.linspace(-1., 1., 101)
    y = np.linspace(-1., 1., 101)
    xx, yy = np.meshgrid(x, y)
    f = AnatomicalContourField(x, y, xx*xx, assume_rectangular_domain=True)
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
        AnatomicalContourField([0, 1, 1], [0, 1, 2], np.zeros((3, 3)), assume_rectangular_domain=True)
    with pytest.raises(ValueError, match="Closed contours"):
        OrientedContour.from_xy([[0,0],[1,0],[0,0]])
    with pytest.raises(ValueError, match="finite XY"):
        plane().unique_contour(.5).project([np.nan, .5])


def test_missing_tissue_mask_requires_explicit_rectangular_assumption():
    x = np.linspace(-1., 1., 21)
    y = np.linspace(0., 1., 21)
    xx, yy = np.meshgrid(x, y)
    with pytest.raises(ValueError, match="tissue_mask"):
        AnatomicalContourField(x, y, yy)


def test_rejects_far_away_and_off_tissue_sources():
    source = plane()
    target = plane(shift=.4)
    with pytest.raises(ValueError, match="outside"):
        reconstruct_by_arclength([200., 200.], .2, .8, source, target)
    with pytest.raises(ValueError, match="Source cell"):
        reconstruct_by_arclength([.2, .8], .2, .8, source, target)


def test_minimum_gradient_is_absolute_not_relative_to_itself():
    x = np.linspace(-1., 1., 101)
    y = np.linspace(0., 1., 71)
    xx, yy = np.meshgrid(x, y)
    f = AnatomicalContourField(x, y, 1e-12 * yy, assume_rectangular_domain=True)
    with pytest.raises(ValueError, match="min_gradient_norm"):
        f.unique_contour(.5e-12)


def test_contour_mask_changes_correspondence_in_wedge():
    x = np.linspace(-1., 1., 201)
    y = np.linspace(0., 1., 101)
    xx, yy = np.meshgrid(x, y)
    mask = np.abs(xx) < (.25 + .7 * yy)
    source = AnatomicalContourField(x,y,yy,tissue_mask=mask)
    target = AnatomicalContourField(x,y,yy,tissue_mask=mask)
    src = np.array([0.1,.2])
    pred = reconstruct_by_arclength(src,.2,.8,source,target)
    wrong = reconstruct_by_arclength(src,.2,.8,
        AnatomicalContourField(x,y,yy,assume_rectangular_domain=True),
        AnatomicalContourField(x,y,yy,assume_rectangular_domain=True))
    assert abs(pred.xy[0] - wrong.xy[0]) > .1
