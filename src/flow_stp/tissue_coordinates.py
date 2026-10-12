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
        assume_rectangular_domain: bool = False,
        min_gradient_norm: float = 1e-6,
        min_orientation_cosine: float = 0.5,
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
        if tissue_mask is None and not assume_rectangular_domain:
            raise ValueError(
                "Supply tissue_mask for the actual tissue domain, or explicitly set "
                "assume_rectangular_domain=True for a verified rectangular tissue."
            )
        if tissue_mask is not None:
            mask = np.asarray(tissue_mask, dtype=bool)
            if mask.shape != z.shape:
                raise ValueError("tissue_mask shape must match z_grid.")
        else:
            mask = np.ones_like(z, dtype=bool)
        if not mask.any():
            raise ValueError("tissue_mask must contain at least one tissue grid point.")
        if not np.isfinite(min_gradient_norm) or min_gradient_norm <= 0:
            raise ValueError("min_gradient_norm must be strictly positive and finite.")
        if not np.isfinite(min_orientation_cosine) or not 0 < min_orientation_cosine <= 1:
            raise ValueError("min_orientation_cosine must be in (0,1].")
        self.mask = mask
        self.min_gradient_norm = float(min_gradient_norm)
        self.min_orientation_cosine = float(min_orientation_cosine)
        self.max_grid_step = float(np.hypot(np.max(np.diff(self.x)), np.max(np.diff(self.y))))
        self._mask_at = RegularGridInterpolator(
            (self.y, self.x), mask.astype(float), method="nearest",
            bounds_error=False, fill_value=0.,
        )
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
            gradient_norm = np.linalg.norm(gradient, axis=1)
            if np.any(~np.isfinite(gradient_norm)) or np.min(gradient_norm) < self.min_gradient_norm:
                raise ValueError("Contour gradient is below min_gradient_norm; anatomical axis is poorly conditioned.")
            rotated = np.column_stack((-gradient[:, 1], gradient[:, 0]))
            signed_cosine = np.einsum('ij,ij->i', tangent, rotated) / (
                np.linalg.norm(tangent, axis=1) * gradient_norm
            )
            if (np.any(~np.isfinite(signed_cosine)) or
                np.min(np.abs(signed_cosine)) < self.min_orientation_cosine or
                np.any(np.sign(signed_cosine) != np.sign(signed_cosine[0]))):
                raise ValueError("Contour orientation is ambiguous or locally inconsistent.")
            alignment = float(np.dot(np.linalg.norm(tangent, axis=1), signed_cosine))
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
        assume_rectangular_domain: bool = False,
        min_gradient_norm: float = 1e-6,
        min_orientation_cosine: float = 0.5,
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
            assume_rectangular_domain=assume_rectangular_domain,
            min_gradient_norm=min_gradient_norm,
            min_orientation_cosine=min_orientation_cosine,
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
    *,
    max_source_distance: float | None = None,
) -> ArclengthReconstruction:
    """Predict XY endpoint via constant normalized contour arclength.

    No true destination is used. Requires single open contours with
    consistent anatomical orientation and XY handedness.
    """
    xy = np.asarray(source_xy, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("source_xy must be a finite physical XY position.")
    if source_field._mask_at(xy[::-1]).item() < 0.5:
        raise ValueError("Source XY lies outside the supplied tissue domain.")
    limit = 2. * source_field.max_grid_step if max_source_distance is None else float(max_source_distance)
    if not np.isfinite(limit) or limit < 0:
        raise ValueError("max_source_distance must be nonnegative and finite.")
    initial = source_field.unique_contour(source_depth)
    _, fraction, distance = initial.project(xy)
    if distance > limit:
        raise ValueError(
            f"Source cell is {distance:.5g} from its anatomical contour, "
            f"exceeding allowed distance {limit:.5g}; check depth, grid, and registration."
        )
    target = target_field.unique_contour(target_depth)
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
