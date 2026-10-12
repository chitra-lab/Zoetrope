"""Tissue-space reconstruction via oriented normalized contour arclength.

The default procedure predicts XY from the source cell and inferred depths,
without using the target true endpoint. The oracle projector is for evaluation
only. All distances are in supplied physical XY units.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass(frozen=True)
class OrientedContour:
    xy: np.ndarray
    length: float
    cumulative_length: np.ndarray

    @classmethod
    def from_xy(cls, points: np.ndarray) -> "OrientedContour":
        xy = np.asarray(points, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
            raise ValueError("Contour must have at least two XY vertices.")
        if not np.isfinite(xy).all():
            raise ValueError("Contour contains nonfinite vertices.")
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        xy = xy[np.r_[True, seg > 1e-12]]
        if len(xy) < 2:
            raise ValueError("Contour has zero length.")
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        arc = np.r_[0., np.cumsum(seg)]
        if np.linalg.norm(xy[0] - xy[-1]) <= 1e-9 * max(1., arc[-1]):
            raise ValueError("Closed contours need an origin landmark; arclength is ambiguous.")
        return cls(xy=xy, length=float(arc[-1]), cumulative_length=arc)

    def at_fraction(self, u: float) -> np.ndarray:
        if not np.isfinite(u) or u < 0 or u > 1:
            raise ValueError("Relative arclength u must be in [0, 1].")
        distance = float(u) * self.length
        return np.array([
            np.interp(distance, self.cumulative_length, self.xy[:, dim])
            for dim in (0, 1)
        ])

    def project(self, point: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Closest polyline point, fractional arclength, Euclidean distance."""
        p = np.asarray(point, dtype=float)
        if p.shape != (2,) or not np.isfinite(p).all():
            raise ValueError("Point must be a finite XY pair.")
        starts, vectors = self.xy[:-1], np.diff(self.xy, axis=0)
        denom = (vectors * vectors).sum(axis=1)
        weights = np.clip(np.einsum('ij,ij->i', p - starts, vectors) / denom, 0., 1.)
        projections = starts + weights[:, None] * vectors
        squared = ((projections - p) ** 2).sum(axis=1)
        k = int(np.argmin(squared))
        arc = self.cumulative_length[k] + weights[k] * np.sqrt(denom[k])
        return projections[k], float(arc / self.length), float(np.sqrt(squared[k]))


class AnatomicalContourField:
    """Contour extractor on a finite physical XY grid.

    x, y strictly increase; z_grid has shape (len(y), len(x)).
    tissue_mask optionally marks the valid tissue domain. Disconnected
    contours, closed loops, and ambiguous orientation are rejected.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z_grid: np.ndarray,
        *,
        tissue_mask: np.ndarray | None = None,
    ) -> None:
        try:
            import contourpy
        except ImportError as exc:
            raise ImportError("Install contourpy to extract iso-depth contours.") from exc
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        z = np.asarray(z_grid, dtype=float)
        if (self.x.ndim != 1 or self.y.ndim != 1 or
            len(self.x) < 3 or len(self.y) < 3 or
            not np.isfinite(self.x).all() or not np.isfinite(self.y).all() or
            np.any(np.diff(self.x) <= 0) or np.any(np.diff(self.y) <= 0)):
            raise ValueError("x and y must be strictly increasing finite arrays of length >=3.")
        if z.shape != (len(self.y), len(self.x)) or not np.isfinite(z).all():
            raise ValueError("z_grid must be finite and have shape (len(y), len(x)).")
        if tissue_mask is not None:
            mask = np.asarray(tissue_mask, dtype=bool)
            if mask.shape != z.shape:
                raise ValueError("tissue_mask shape must match z_grid.")
        else:
            mask = np.ones_like(z, dtype=bool)
        self.mask = mask
        masked_z = np.ma.array(z, mask=~mask)
        self._generator = contourpy.contour_generator(
            x=self.x, y=self.y, z=masked_z, name="serial",
            line_type="Separate", corner_mask=False,
        )
        dz_dy, dz_dx = np.gradient(z, self.y, self.x)
        self._grad_x = RegularGridInterpolator(
            (self.y, self.x), dz_dx, bounds_error=True,
        )
        self._grad_y = RegularGridInterpolator(
            (self.y, self.x), dz_dy, bounds_error=True,
        )

    def contours(self, depth: float) -> list[OrientedContour]:
        if not np.isfinite(depth):
            raise ValueError("Depth must be finite.")
        out = []
        for line in self._generator.lines(float(depth)):
            points = np.asarray(line, dtype=float)
            if len(points) < 2:
                continue
            contour = OrientedContour.from_xy(points)
            mid = (contour.xy[:-1] + contour.xy[1:]) * 0.5
            gradient = np.column_stack((
                self._grad_x(mid[:, ::-1]),
                self._grad_y(mid[:, ::-1]),
            ))
            tangent = np.diff(contour.xy, axis=0)
            rotated = np.column_stack((-gradient[:, 1], gradient[:, 0]))
            alignment = float(np.einsum('ij,ij->', tangent, rotated))
            norm_scale = float(
                np.linalg.norm(tangent, axis=1).sum() *
                max(np.linalg.norm(gradient, axis=1).mean(), 1e-12)
            )
            if not np.isfinite(alignment) or abs(alignment) <= 1e-10 * norm_scale:
                raise ValueError(
                    "Contour orientation is ambiguous (vanishing/inconsistent gradient)."
                )
            if alignment < 0:
                contour = OrientedContour.from_xy(contour.xy[::-1])
            out.append(contour)
        return out

    def unique_contour(self, depth: float) -> OrientedContour:
        contours = self.contours(depth)
        if len(contours) != 1:
            raise ValueError(
                f"Depth {depth:g} has {len(contours)} contours; expected exactly one. "
                "Check tissue mask or implement explicit component matching."
            )
        return contours[0]

    @classmethod
    def from_stage1a(
        cls,
        result: Any,
        tissue_id: int,
        x: np.ndarray,
        y: np.ndarray,
        *,
        tissue_mask: np.ndarray | None = None,
        batch_size: int = 8192,
    ) -> "AnatomicalContourField":
        """Evaluate fitted Stage1AResult alpha on a physical XY grid."""
        import torch
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        model = result.model.coordinate_maps
        params = result.xy_norm_params
        row = params.loc[params["slice_id"] == int(tissue_id)]
        if len(row) != 1:
            raise ValueError("Expected one XY-normalization row per tissue_id.")
        center = row[["center_x", "center_y"]].iloc[0].to_numpy(dtype=float)
        scale = row[["scale_x", "scale_y"]].iloc[0].to_numpy(dtype=float)
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Invalid XY normalization scale.")
        xx, yy = np.meshgrid(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float),
        )
        xy = np.column_stack((xx.ravel(), yy.ravel()))
        p = next(model.parameters(), None)
        device = p.device if p is not None else torch.device("cpu")
        dtype = p.dtype if p is not None else torch.float32
        was_training = model.training
        model.eval()
        values = []
        try:
            with torch.no_grad():
                for k in range(0, len(xy), batch_size):
                    coords = torch.as_tensor(
                        (xy[k:k+batch_size] - center) / scale,
                        device=device, dtype=dtype,
                    )
                    z = model.one_slice(coords, int(tissue_id))
                    values.append(z.detach().cpu().numpy().reshape(-1))
        finally:
            model.train(was_training)
        return cls(
            x, y, np.concatenate(values).reshape(xx.shape),
            tissue_mask=tissue_mask,
        )


@dataclass(frozen=True)
class ArclengthReconstruction:
    xy: np.ndarray
    source_fraction: float
    source_contour_distance: float


def reconstruct_by_arclength(
    source_xy: np.ndarray,
    source_depth: float,
    target_depth: float,
    source_field: AnatomicalContourField,
    target_field: AnatomicalContourField,
) -> ArclengthReconstruction:
    """Predict XY endpoint via constant normalized contour arclength.

    No true destination is used. Requires single open contours with
    consistent anatomical orientation and XY handedness.
    """
    initial = source_field.unique_contour(source_depth)
    target = target_field.unique_contour(target_depth)
    _, fraction, distance = initial.project(source_xy)
    return ArclengthReconstruction(
        xy=target.at_fraction(fraction),
        source_fraction=fraction,
        source_contour_distance=distance,
    )


def oracle_nearest_contour_point(
    true_destination_xy: np.ndarray,
    target_depth: float,
    target_field: AnatomicalContourField,
) -> tuple[np.ndarray, float]:
    """Evaluation ONLY: closest target contour point to true destination.

    Lower bound on contour-constrained endpoint error, not a prediction.
    """
    point, _, error = target_field.unique_contour(target_depth).project(
        true_destination_xy,
    )
    return point, error


__all__ = [
    "AnatomicalContourField", "ArclengthReconstruction", "OrientedContour",
    "reconstruct_by_arclength", "oracle_nearest_contour_point",
]
