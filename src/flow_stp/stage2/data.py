"""Data handoff utilities between Stage 1 and Stage 2."""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


ArrayLike = Union[np.ndarray, torch.Tensor]


def _float_tensor(x: ArrayLike, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(x, dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values.")
    return tensor.detach().cpu().contiguous()


def normalize_times(
    times: ArrayLike,
    *,
    time_min: Optional[float] = None,
    time_max: Optional[float] = None,
) -> torch.Tensor:
    """Normalize observation times to ``[0,1]``.

    The earliest observed time becomes exactly zero.
    """
    times_tensor = _float_tensor(times, "times").reshape(-1)

    if time_min is None:
        time_min = float(times_tensor.min())
    if time_max is None:
        time_max = float(times_tensor.max())
    if time_max <= time_min:
        raise ValueError("time_max must be greater than time_min.")

    normalized = (times_tensor - time_min) / (time_max - time_min)
    return normalized.clamp(0.0, 1.0)


class Stage2Dataset(Dataset):
    """Dataset containing frozen Stage 1 coordinates and Stage 2 observations."""

    def __init__(
        self,
        *,
        z: ArrayLike,
        time: ArrayLike,
        counts: ArrayLike,
        log_library_size: Optional[ArrayLike] = None,
    ) -> None:
        self.z = _float_tensor(z, "z").reshape(-1)
        self.time = _float_tensor(time, "time").reshape(-1)
        self.counts = _float_tensor(counts, "counts")

        if self.counts.ndim != 2:
            raise ValueError("counts must have shape [cells, genes].")

        n_cells = self.counts.shape[0]
        if self.z.numel() != n_cells or self.time.numel() != n_cells:
            raise ValueError("z, time, and counts must contain the same cells.")

        if torch.any(self.z < 0.0) or torch.any(self.z > 1.0):
            raise ValueError("z must lie in [0,1].")
        if torch.any(self.time < 0.0) or torch.any(self.time > 1.0):
            raise ValueError("time must lie in [0,1].")
        if torch.any(self.counts < 0.0):
            raise ValueError("counts must be nonnegative.")

        if log_library_size is None:
            self.log_library_size = None
        else:
            self.log_library_size = _float_tensor(
                log_library_size,
                "log_library_size",
            ).reshape(-1)
            if self.log_library_size.numel() != n_cells:
                raise ValueError(
                    "log_library_size must contain one value per cell."
                )

    def __len__(self) -> int:
        return self.counts.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {
            "z": self.z[index],
            "time": self.time[index],
            "counts": self.counts[index],
        }
        if self.log_library_size is not None:
            item["log_library_size"] = self.log_library_size[index]
        return item
