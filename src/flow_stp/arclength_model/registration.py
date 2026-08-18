"""Smooth registration for the arclength model.

For fixed M, solve

    min_V sum ||M_t + V M_z||_W^2
          + lambda_v ||V||^2
          + lambda_z ||d_z V||^2
          + lambda_t ||d_t V||^2

with Dirichlet boundary conditions V(0,t)=V(1,t)=0.

Writing

    A = <M_z, M_z>_W,
    B = <M_z, M_t>_W,

expands the data term to A V^2 + 2 B V + const.  Setting its derivative
to zero gives H V = -B.  Thus ``-B`` is not an extra model quantity; it is
the right-hand side induced by the cross term between spatial and temporal
expression derivatives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg


@dataclass
class RegistrationResult:
    z: np.ndarray
    t: np.ndarray
    V: np.ndarray                 # [nt,nz]
    A: np.ndarray                 # [nt,nz]
    B: np.ndarray                 # [nt,nz]
    residual: np.ndarray          # [nt,nz,G]
    q: np.ndarray                 # [nt,nz]
    q0: np.ndarray                # [nt,nz]
    explained_fraction: np.ndarray  # [nt,nz]


def _metric_weights(num_genes: int, weights: Optional[np.ndarray]) -> np.ndarray:
    if weights is None:
        return np.ones(num_genes, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if len(w) != num_genes:
        raise ValueError("gene_weights has the wrong length")
    if np.any(w < 0):
        raise ValueError("gene_weights must be nonnegative")
    return w


def _first_difference(n: int, spacing: float) -> sparse.csr_matrix:
    if n < 2:
        return sparse.csr_matrix((0, n))
    return sparse.diags(
        diagonals=[-np.ones(n - 1), np.ones(n - 1)],
        offsets=[0, 1],
        shape=(n - 1, n),
        format="csr",
    ) / float(spacing)


def solve_smooth_registration(
    M_z: np.ndarray,
    M_t: np.ndarray,
    z_grid,
    t_grid,
    *,
    gene_weights: Optional[np.ndarray] = None,
    lambda_v: float = 1e-4,
    lambda_z: float = 1e-2,
    lambda_t: float = 1e-2,
    solver: str = "spsolve",
    cg_rtol: float = 1e-8,
    cg_maxiter: Optional[int] = None,
    eps: float = 1e-12,
) -> RegistrationResult:
    """Solve the convex quadratic registration problem on a rectangular grid.

    Arrays must have shape [nt,nz,G].  z boundary nodes are fixed exactly to
    zero velocity.  Interior values are solved as one sparse linear system.
    """
    M_z = np.asarray(M_z, dtype=float)
    M_t = np.asarray(M_t, dtype=float)
    z = np.asarray(z_grid, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    if M_z.shape != M_t.shape or M_z.ndim != 3:
        raise ValueError("M_z and M_t must both have shape [nt,nz,G]")
    nt, nz, G = M_z.shape
    if len(z) != nz or len(t) != nt:
        raise ValueError("grid lengths do not match derivative arrays")
    if nz < 3:
        raise ValueError("need at least 3 z grid points for Dirichlet boundaries")
    if np.any(np.diff(z) <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError("z_grid and t_grid must be strictly increasing")

    w = _metric_weights(G, gene_weights)
    A = np.einsum("tzg,g,tzg->tz", M_z, w, M_z)
    B = np.einsum("tzg,g,tzg->tz", M_z, w, M_t)

    # Unknowns are only z=1,...,nz-2; boundary velocities are exactly zero.
    full_n = nt * nz
    mask2d = np.ones((nt, nz), dtype=bool)
    mask2d[:, 0] = False
    mask2d[:, -1] = False
    interior = np.flatnonzero(mask2d.reshape(-1))

    dz = float(np.mean(np.diff(z)))
    dt = float(np.mean(np.diff(t))) if nt > 1 else 1.0
    Dz = sparse.kron(sparse.eye(nt, format="csr"), _first_difference(nz, dz), format="csr")
    Dt = sparse.kron(_first_difference(nt, dt), sparse.eye(nz, format="csr"), format="csr") if nt > 1 else sparse.csr_matrix((0, full_n))
    Dz_i = Dz[:, interior]
    Dt_i = Dt[:, interior]

    diag = A.reshape(-1)[interior] + float(lambda_v)
    H = sparse.diags(diag, format="csr")
    if lambda_z:
        H = H + float(lambda_z) * (Dz_i.T @ Dz_i)
    if lambda_t and nt > 1:
        H = H + float(lambda_t) * (Dt_i.T @ Dt_i)
    rhs = -B.reshape(-1)[interior]

    if solver == "spsolve":
        v_i = splinalg.spsolve(H.tocsc(), rhs)
    elif solver == "cg":
        v_i, info = splinalg.cg(H, rhs, rtol=cg_rtol, maxiter=cg_maxiter)
        if info != 0:
            raise RuntimeError(f"scipy.sparse.linalg.cg failed with info={info}")
    else:
        raise ValueError("solver must be 'spsolve' or 'cg'")

    V = np.zeros((nt, nz), dtype=float)
    V.reshape(-1)[interior] = v_i

    residual = M_t + V[..., None] * M_z
    q2 = np.einsum("tzg,g,tzg->tz", residual, w, residual)
    q02 = np.einsum("tzg,g,tzg->tz", M_t, w, M_t)
    q = np.sqrt(np.maximum(q2, 0.0))
    q0 = np.sqrt(np.maximum(q02, 0.0))
    explained = 1.0 - q2 / np.maximum(q02, eps)

    return RegistrationResult(
        z=z,
        t=t,
        V=V,
        A=A,
        B=B,
        residual=residual,
        q=q,
        q0=q0,
        explained_fraction=explained,
    )
