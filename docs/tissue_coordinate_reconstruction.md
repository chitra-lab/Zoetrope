# Reconstructing tissue-space endpoints by normalized contour arclength

This module implements the **candidate canonical tissue-coordinate imputation**
introduced in Spatial TI Methods §2.5. It does not alter trained Stage 1,
the Fisher projection, or Stage 2 flow integration.

For an initial cell at physical XY location `s0`, let
`a = alpha_source(s0)` and let `z1 = F(a,t1)` be its predicted anatomical
depth from Stage 2. Let `C_i(z) = {s : alpha_i(s)=z}` be the iso-depth contour
in tissue `i`. Orient the contour in the XY plane by the vector field
`J grad(alpha)=(-d_y alpha,d_x alpha)`. The source cell receives a normalized
arclength fraction `u0 = length(C_source(a) up to s0)/length(C_source(a))`.
The predicted destination is the point at fraction `u0` of `C_target(z1)`.

## Usage with a trained Stage 1 result

```python
import numpy as np
import torch
from flow_stp.tissue_coordinates import (
    AnatomicalContourField,
    reconstruct_by_arclength,
    oracle_nearest_contour_point,
)

# User-defined physical XY grids (same handedness), and tissue-domain masks.
source = AnatomicalContourField.from_stage1a(
    result, tissue_id=0, x=x0, y=y0, tissue_mask=mask0,
)
target = AnatomicalContourField.from_stage1a(
    result, tissue_id=1, x=x1, y=y1, tissue_mask=mask1,
)

source_xy = np.array([x_cell, y_cell], dtype=float)
row = result.xy_norm_params.loc[
    result.xy_norm_params["slice_id"] == 0
].iloc[0]
standardized = (source_xy - row[["center_x", "center_y"]].to_numpy(float)) / (
    row[["scale_x", "scale_y"]].to_numpy(float)
)
maps = result.model.coordinate_maps
p = next(maps.parameters())
with torch.no_grad():
    a = float(maps.one_slice(
        torch.as_tensor(standardized[None], dtype=p.dtype, device=p.device), 0
    ).item())

# z_target is the F(a,t_target) produced by the Stage 2 RK4 flow.
prediction = reconstruct_by_arclength(
    source_xy, source_depth=a, target_depth=z_target,
    source_field=source, target_field=target,
)
predicted_xy = prediction.xy
print(prediction.source_fraction, predicted_xy)

# Synthetic EVALUATION ONLY. The oracle requires known hidden destination.
oracle_xy, oracle_endpoint_error = oracle_nearest_contour_point(
    true_destination_xy, z_target, target,
)
```

The `source_contour_distance` output measures how far the source position
is from the extracted source contour, serving as a numerical grid diagnostic.

## Assumptions and limitations

- All contours must be connected, open, regular and contained within a
  supplied tissue mask or polygon (if the bounding box contains non-tissue
  regions, supply the actual domain). Closed loops and multiple disconnected components are
  deliberately rejected instead of silently inventing a correspondence.
- Increasing anatomical `z` must mean the same thing across tissues; the
  physical XY coordinate planes must also have consistent handedness.
  Mirrored tissue images must be corrected.
- A contour direction follows from those orientations, but **material
  conservation of normalized arclength is an additional assumption**, not a
  consequence of the Fisher model. Uneven tangential tissue stretching and
  tangential migration can invalidate the correspondence.
- `contourpy` extracts approximate contours on a gridded neural coordinate
  field. Refine grids for accuracy; provide physical XY spacing, not tissue-
  normalized model coordinates. Target contours outside the learned spatial
  support cannot be reconstructed reliably.
- This provides a *tissue-space imputed endpoint*, not an identified cell
  lineage or a uniquely recovered physical migration trajectory.
- The oracle comparison selects the point on the **same predicted contour**
  minimizing distance to the **true target endpoint**. Its error is a lower
  bound for a fixed contour; it is not a prediction.
- Existing Regime B ground-truth geometry-only comparisons were encouraging,
  but recovery using held-out **learned** anatomical contours remains to be
  tested with the frozen Stage 1 checkpoint.

## Boundary-aware extraction

When an observed tissue boundary polygon is available, prefer polygon clipping:

```python
source = AnatomicalContourField.from_stage1a(
    result, tissue_id=0, x=x0, y=y0, tissue_polygon=boundary0,
)
target = AnatomicalContourField.from_stage1a(
    result, tissue_id=1, x=x1, y=y1, tissue_polygon=boundary1,
)
```

Install the optional dependency with `pip install -e '.[geometry]'`.
`tissue_polygon` is an ordered array of physical XY vertices describing one
valid simple polygon (holes are not supported by this array API). The grid
must cover the entire polygon; a small exterior margin is recommended. The
coordinate field must be finite and reliable up to and immediately outside
the tissue boundary. Polygon geometry must come from allowed observed tissue
information in a learned-model benchmark, never from hidden destinations.
Supplying both `tissue_mask` and `tissue_polygon` raises an error.

The extractor traces the level set on the enclosing grid, intersects it with
the polygon, and joins only exactly shared endpoints produced by geometric
intersection. It does not bridge contour gaps, select the longest component,
or replace normalized arclength with a material transverse coordinate.
The original orientation, gradient, source-distance, and topology checks
remain in force. Point membership allows only a machine-roundoff tolerance
at polygon edges; contours themselves are not buffered. A level set that
touches the tissue only at an isolated point is not a reconstructable contour.

Mask mode remains available and retains `corner_mask=False`. Its fully masked
grid-square rule can erase thin boundary strips or truncate contour endpoints.
`corner_mask=True` is an optional diagnostic alternative, but it does not
recover an accurate observed boundary and is not the recommended replacement.

Each field caches up to 512 exact floating-point depth values; depths are not
rounded or interpolated between cached contours. Set `cache_size=0` to disable
caching or a smaller positive value to bound memory more tightly. Cached
contour arrays and input-grid copies are read-only. Construct a new field if
the coordinate grid, boundary, or validation thresholds change. Calls remain
sequential; the mutable cache is not intended as a shared concurrent service.

Reproducible geometry-only experiments and the comparison report are in
`benchmarks/arclength/`. They use the frozen geometry recovered from the
previous Regime B/D1 archive, without training or modifying Stage 1 or Stage 2.

## API

- `AnatomicalContourField(x,y,z_grid,tissue_mask=...)`
- `AnatomicalContourField.from_stage1a(result,tissue_id,x,y,tissue_mask=...)`
- `AnatomicalContourField.unique_contour(z)`
- `OrientedContour.at_fraction(u)` and `OrientedContour.project(xy)`
- `reconstruct_by_arclength(source_xy,source_depth,target_depth,source_field,target_field)`
- `oracle_nearest_contour_point(true_destination_xy,target_depth,target_field)`

Tests: `pytest -q tests/test_tissue_coordinates.py`.

## Before running learned-contour displacement benchmarks

The source and target fields must be restricted to actual tissue domains:

```python
source = AnatomicalContourField.from_stage1a(
    result, tissue_id=0, x=x0, y=y0, tissue_mask=mask0,
)
target = AnatomicalContourField.from_stage1a(
    result, tissue_id=1, x=x1, y=y1, tissue_mask=mask1,
)
```

A tissue mask or polygon is required by default. For a genuinely rectangular domain,
pass `assume_rectangular_domain=True` explicitly. The extractor rejects
contours with gradient norm below `min_gradient_norm` (default `1e-6`)
or tangents inconsistent with the rotated anatomical gradient. Source
cells must lie in the source tissue domain and no farther than two grid
diagonals from their declared level set (override with
`max_source_distance` when the coordinate-grid accuracy justifies it).
These checks detect poor learned-coordinate geometry; they do not repair it.

The paired benchmark helper evaluates both methods on **the same**
predicted contours and exact same held-out cells:

```python
from flow_stp.simulations.arclength_displacement import (
    evaluate_arclength_vs_oracle,
)
comparison = evaluate_arclength_vs_oracle(
    source_xy, true_target_xy,
    source_depth, predicted_target_depth,
    source, target,
    on_error="record",  # or "raise" to fail on first invalid cell
)
print(comparison.summary)
comparison.per_cell.to_csv("paired_contour_endpoints.csv", index=False)
```

When `on_error="record"`, the success fraction must be reported with the
paired NRMSE values, since selective contour failures can bias metrics.
No prediction uses the true destination; only the oracle lower bound
and evaluation use it.

For Fisher velocity diagnostics without modifying the ODE:

```python
from flow_stp.stage2 import audit_flow_observability
report = audit_flow_observability(
    result.model.fisher_flat, trajectory, eps=1e-8,
)
print(report.summary())
```

The default observability threshold detects when epsilon is at least 1% of
the Fisher tangent-energy denominator. This audit checks saved trajectory
nodes only, not every intermediate RK4 substep. Report the flagged fraction
before interpreting low-observability velocities as migration.

