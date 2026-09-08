"""Reference-normalized coordinate maps for zoo estimators.

The learned coverage interval [a_i, b_i] should describe the portion of the
common process axis represented by tissue i.  To make that statement true in
practice, the tissue-local neural coordinate is put in a detached robust affine
gauge before applying the learned coverage endpoints.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .common import CoordinateMaps, Tensor


class ReferenceNormalizedCoordinateMaps(CoordinateMaps):
    """Coordinate maps with a fixed-within-step robust affine local gauge.

    ``gauge_low`` and ``gauge_high`` are refreshed from spatial reference
    points during optimization.  They are buffers, not parameters: gradients
    flow through the raw coordinate field but not through the empirical gauge
    statistics.  This mirrors the selected Stage-1a strategy and prevents a
    tissue from using only an arbitrary narrow subinterval of its nominal
    coverage interval.
    """

    def __init__(self, *args, gauge_eps: float = 1e-4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gauge_eps = float(gauge_eps)
        self.register_buffer("gauge_low", torch.zeros(self.n_tissues))
        self.register_buffer("gauge_high", torch.ones(self.n_tissues))

    def raw_one(self, coords: Tensor, tissue_id: int) -> Tensor:
        return self.maps[int(tissue_id)](coords).squeeze(-1)

    def local_one(self, coords: Tensor, tissue_id: int) -> Tensor:
        raw = self.raw_one(coords, tissue_id)
        low = self.gauge_low[int(tissue_id)].to(raw)
        high = self.gauge_high[int(tissue_id)].to(raw)
        span = torch.clamp(high - low, min=self.gauge_eps)
        return (raw - low) / span

    @torch.no_grad()
    def refresh_gauge_(
        self,
        coords_by_tissue: Sequence[Tensor],
        *,
        max_points: int = 4096,
        lower_quantile: float = 0.005,
        upper_quantile: float = 0.995,
    ) -> None:
        if len(coords_by_tissue) != self.n_tissues:
            raise ValueError("coords_by_tissue must have one entry per tissue")
        qlo = float(lower_quantile)
        qhi = float(upper_quantile)
        if not 0.0 <= qlo < qhi <= 1.0:
            raise ValueError("gauge quantiles must satisfy 0 <= low < high <= 1")

        for tid, coords in enumerate(coords_by_tissue):
            x = coords
            if max_points > 0 and len(x) > int(max_points):
                # Deterministic spatially agnostic reference subset.  The
                # atlas preprocessing already uses an isotropic physical
                # similarity transform, so this only fixes the 1-D gauge.
                idx = torch.linspace(
                    0,
                    len(x) - 1,
                    int(max_points),
                    device=x.device,
                ).round().long()
                x = x[idx]
            raw = self.raw_one(x, tid)
            low = torch.quantile(raw, qlo)
            high = torch.quantile(raw, qhi)
            if not torch.isfinite(low) or not torch.isfinite(high):
                raise RuntimeError(f"tissue {tid}: non-finite coordinate gauge")
            if float(high - low) < self.gauge_eps:
                center = 0.5 * (high + low)
                low = center - 0.5 * self.gauge_eps
                high = center + 0.5 * self.gauge_eps
            self.gauge_low[tid].copy_(low)
            self.gauge_high[tid].copy_(high)
