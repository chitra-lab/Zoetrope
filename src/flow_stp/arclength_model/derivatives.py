"""Forward-mode derivatives of the fitted mean-expression field M(z,t)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor


@dataclass
class ExpressionGrid:
    z: np.ndarray
    t: np.ndarray
    M: np.ndarray       # [nt,nz,G]
    M_z: np.ndarray     # [nt,nz,G]
    M_t: np.ndarray     # [nt,nz,G]


def expression_jvp(field, z: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Evaluate M, M_z and M_t with two forward-mode JVPs.

    ``torch.func.jvp`` differentiates the complete batched map at once.  Since
    the field is row-wise in (z,t), a tangent with ones only in the z column
    yields each row's dM/dz; the analogous t tangent yields dM/dt.
    """
    x = torch.stack([z.reshape(-1), t.reshape(-1)], dim=1)

    def fn(inp: Tensor) -> Tensor:
        return field(inp[:, 0], inp[:, 1])

    tangent_z = torch.zeros_like(x)
    tangent_z[:, 0] = 1.0
    tangent_t = torch.zeros_like(x)
    tangent_t[:, 1] = 1.0

    M, M_z = torch.func.jvp(fn, (x,), (tangent_z,))
    _, M_t = torch.func.jvp(fn, (x,), (tangent_t,))
    return M, M_z, M_t


@torch.no_grad()
def _as_numpy(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def evaluate_expression_grid(
    field,
    z_grid,
    t_grid,
    *,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
    chunk_size: int = 16384,
) -> ExpressionGrid:
    """Evaluate M and its first derivatives on a dense (t,z) grid."""
    z_np = np.asarray(z_grid, dtype=float)
    t_np = np.asarray(t_grid, dtype=float)
    zz, tt = np.meshgrid(z_np, t_np, indexing="xy")
    z_flat = torch.as_tensor(zz.reshape(-1), device=device, dtype=dtype)
    t_flat = torch.as_tensor(tt.reshape(-1), device=device, dtype=dtype)

    Ms, Mzs, Mts = [], [], []
    # JVP needs grad-enabled execution; do not wrap in no_grad.
    for start in range(0, z_flat.numel(), chunk_size):
        sl = slice(start, min(start + chunk_size, z_flat.numel()))
        M, M_z, M_t = expression_jvp(field, z_flat[sl], t_flat[sl])
        Ms.append(_as_numpy(M))
        Mzs.append(_as_numpy(M_z))
        Mts.append(_as_numpy(M_t))

    shape = (len(t_np), len(z_np), -1)
    return ExpressionGrid(
        z=z_np,
        t=t_np,
        M=np.concatenate(Ms, axis=0).reshape(shape),
        M_z=np.concatenate(Mzs, axis=0).reshape(shape),
        M_t=np.concatenate(Mts, axis=0).reshape(shape),
    )
