"""Primary learned objects for Fisher-Lagrangian trajectory inference.

Only two scientific objects are estimated from the count data:

* tissue-specific neural anatomical coordinates ``z = alpha_i(x,y)``;
* a shared low-complexity mean-expression state map ``E(z,t)``.

The latter is parameterized on the log-mean scale by fixed cubic spline bases.
Velocity, flow, expression trajectories, and pseudotime are derived after fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .spline import AnchoredTemporalSplineBasis, FixedBSplineBasis

Tensor = torch.Tensor


def _as_col(x: Tensor) -> Tensor:
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2 and x.shape[1] == 1:
        return x
    raise ValueError(f"Expected [N] or [N,1], got {tuple(x.shape)}")


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int],
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation {activation!r}")
        act = activations[activation]
        dims = [in_dim, *hidden, out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TissueCoordinateMap(nn.Module):
    """Smooth tissue-specific map ``(x,y) -> z in (0,1)``."""

    def __init__(
        self,
        spatial_dim: int = 2,
        hidden: Sequence[int] = (64, 64),
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.raw = MLP(spatial_dim, 1, hidden, activation)

    def forward(self, xy: Tensor) -> Tensor:
        if xy.ndim != 2:
            raise ValueError(f"Expected [N,D] coordinates, got {tuple(xy.shape)}")
        return torch.sigmoid(self.raw(xy))


class SplineLogMeanExpressionField(nn.Module):
    r"""Low-complexity smooth field ``eta_g(z,t)=log E_g(z,t)``.

    The parameterization is

    ``eta_g(z,t) = sum_j beta_gj B_j(z)
                    + sum_jk theta_gjk B_j(z) T_k(t)``.

    ``B_j`` are fixed spatial B-splines and the fixed temporal contrasts
    ``T_k`` satisfy ``T_k(t0)=0``. Therefore the baseline anatomical profile
    at the reference time is identified exactly by ``beta``.
    """

    def __init__(
        self,
        num_genes: int,
        *,
        spatial_basis_dim: int = 10,
        temporal_basis_dim: int = 3,
        spline_degree: int = 3,
        coordinate_domain: tuple[float, float] = (0.0, 1.0),
        time_domain: tuple[float, float] = (0.0, 1.0),
        reference_time: float = 0.0,
    ) -> None:
        super().__init__()
        if num_genes <= 0:
            raise ValueError("num_genes must be positive")
        self.num_genes = int(num_genes)
        self.spatial_basis_dim = int(spatial_basis_dim)
        self.temporal_basis_dim = int(temporal_basis_dim)
        self.reference_time = float(reference_time)
        self.spatial_basis = FixedBSplineBasis(
            self.spatial_basis_dim,
            degree=spline_degree,
            domain=coordinate_domain,
        )
        self.temporal_basis = AnchoredTemporalSplineBasis(
            self.temporal_basis_dim,
            degree=spline_degree,
            domain=time_domain,
            t0=reference_time,
        )
        self.baseline_coeff = nn.Parameter(
            torch.zeros(self.num_genes, self.spatial_basis_dim)
        )
        self.temporal_coeff = nn.Parameter(
            torch.zeros(
                self.num_genes,
                self.spatial_basis_dim,
                self.temporal_basis_dim,
            )
        )

    @property
    def num_coefficients_per_gene(self) -> int:
        return self.spatial_basis_dim * (1 + self.temporal_basis_dim)

    def basis_values(self, z: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
        zc = _as_col(z)
        tc = _as_col(time)
        if zc.shape[0] != tc.shape[0]:
            raise ValueError("z and time must have matching rows")
        return self.spatial_basis(zc), self.temporal_basis(tc)

    def design_matrix(self, z: Tensor, time: Tensor) -> Tensor:
        bz, ct = self.basis_values(z, time)
        interaction = torch.einsum("nj,nk->njk", bz, ct).reshape(bz.shape[0], -1)
        return torch.cat((bz, interaction), dim=1)

    def coefficient_matrix(self) -> Tensor:
        return torch.cat(
            (
                self.baseline_coeff,
                self.temporal_coeff.reshape(self.num_genes, -1),
            ),
            dim=1,
        )

    @torch.no_grad()
    def set_coefficient_matrix(self, coef: Tensor) -> None:
        expected = (self.num_genes, self.num_coefficients_per_gene)
        if tuple(coef.shape) != expected:
            raise ValueError(f"Expected coefficients {expected}, got {tuple(coef.shape)}")
        j = self.spatial_basis_dim
        self.baseline_coeff.copy_(coef[:, :j])
        self.temporal_coeff.copy_(coef[:, j:].reshape_as(self.temporal_coeff))

    def forward(self, z: Tensor, time: Tensor) -> Tensor:
        bz, ct = self.basis_values(z, time)
        baseline = torch.einsum("nj,gj->ng", bz, self.baseline_coeff)
        temporal = torch.einsum("nj,nk,gjk->ng", bz, ct, self.temporal_coeff)
        return baseline + temporal

    def mean(self, z: Tensor, time: Tensor) -> Tensor:
        return torch.exp(self.forward(z, time))

    def log_mean_derivatives(self, z: Tensor, time: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(eta, eta_z, eta_t)`` using analytic spline derivatives."""
        zc = _as_col(z)
        tc = _as_col(time)
        if zc.shape[0] != tc.shape[0]:
            raise ValueError("z and time must have matching rows")
        bz = self.spatial_basis(zc)
        dbz = self.spatial_basis.derivative(zc)
        ct = self.temporal_basis(tc)
        dct = self.temporal_basis.derivative(tc)
        eta = (
            torch.einsum("nj,gj->ng", bz, self.baseline_coeff)
            + torch.einsum("nj,nk,gjk->ng", bz, ct, self.temporal_coeff)
        )
        eta_z = (
            torch.einsum("nj,gj->ng", dbz, self.baseline_coeff)
            + torch.einsum("nj,nk,gjk->ng", dbz, ct, self.temporal_coeff)
        )
        eta_t = torch.einsum("nj,nk,gjk->ng", bz, dct, self.temporal_coeff)
        return eta, eta_z, eta_t


# Backwards-compatible public name used by geometry/trajectory code.
LogMeanExpressionField = SplineLogMeanExpressionField


@dataclass(frozen=True)
class FisherLagrangianConfig:
    num_tissues: int
    num_genes: int
    spatial_dim: int = 2
    coordinate_hidden: tuple[int, ...] = (64, 64)
    coordinate_activation: str = "gelu"
    spatial_basis_dim: int = 10
    temporal_basis_dim: int = 3
    spline_degree: int = 3
    time_domain: tuple[float, float] = (0.0, 1.0)
    reference_time: float = 0.0
    submersion_floor: float = 1e-3
    geometry_eps: float = 1e-8


class FisherLagrangianModel(nn.Module):
    """Joint anatomical-coordinate and smooth mean-expression model."""

    def __init__(self, config: FisherLagrangianConfig) -> None:
        super().__init__()
        if config.num_tissues <= 0:
            raise ValueError("num_tissues must be positive")
        self.config = config
        self.coordinate_maps = nn.ModuleList(
            [
                TissueCoordinateMap(
                    spatial_dim=config.spatial_dim,
                    hidden=config.coordinate_hidden,
                    activation=config.coordinate_activation,
                )
                for _ in range(config.num_tissues)
            ]
        )
        self.expression_field = SplineLogMeanExpressionField(
            num_genes=config.num_genes,
            spatial_basis_dim=config.spatial_basis_dim,
            temporal_basis_dim=config.temporal_basis_dim,
            spline_degree=config.spline_degree,
            coordinate_domain=(0.0, 1.0),
            time_domain=config.time_domain,
            reference_time=config.reference_time,
        )

    def coordinate(self, xy: Tensor, tissue_index: Tensor) -> Tensor:
        if xy.ndim != 2:
            raise ValueError("xy must be [N,D]")
        if tissue_index.ndim != 1 or tissue_index.shape[0] != xy.shape[0]:
            raise ValueError("tissue_index must be [N]")
        valid = (tissue_index >= 0) & (tissue_index < len(self.coordinate_maps))
        if not bool(valid.all()):
            raise ValueError("tissue_index contains an out-of-range value")
        out = torch.empty((xy.shape[0], 1), dtype=xy.dtype, device=xy.device)
        for idx, mapping in enumerate(self.coordinate_maps):
            mask = tissue_index == idx
            if mask.any():
                out[mask] = mapping(xy[mask])
        return out

    def log_mean(self, z: Tensor, time: Tensor) -> Tensor:
        return self.expression_field(z, time)

    def mean(self, z: Tensor, time: Tensor) -> Tensor:
        return self.expression_field.mean(z, time)

    def forward(
        self,
        xy: Tensor,
        tissue_index: Tensor,
        time: Tensor,
        log_exposure: Tensor | None = None,
    ) -> dict[str, Tensor]:
        z = self.coordinate(xy, tissue_index)
        eta = self.log_mean(z, time)
        if log_exposure is None:
            log_mu = eta
        else:
            le = _as_col(log_exposure)
            if le.shape[0] != eta.shape[0]:
                raise ValueError("log_exposure must have one row per cell")
            log_mu = eta + le
        return {
            "z": z,
            "log_lambda": eta,
            "lambda": torch.exp(eta),
            "log_mu": log_mu,
            "mu": torch.exp(log_mu),
        }

    @staticmethod
    def poisson_nll(counts: Tensor, log_mu: Tensor, reduction: str = "mean") -> Tensor:
        return F.poisson_nll_loss(
            log_mu,
            counts.float(),
            log_input=True,
            full=False,
            reduction=reduction,
        )

    def coordinate_geometry_penalties(
        self,
        xy: Tensor,
        tissue_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""Return submersion and relative-position gauge penalties.

        The relative-position term penalizes

        ``R_alpha = grad(alpha)^T H_alpha grad(alpha) / ||grad(alpha)||^3``.

        ``R_alpha=0`` means the gradient magnitude is constant along each
        gradient streamline. With ``alpha in (0,1)`` this is the constant-speed
        relative-arclength gauge: equal coordinate increments correspond to equal
        fractions of physical path length along a streamline.
        """
        xy_req = xy.detach().clone().requires_grad_(True)
        z = self.coordinate(xy_req, tissue_index)
        grad = torch.autograd.grad(z.sum(), xy_req, create_graph=True)[0]
        norm_sq = grad.square().sum(dim=1) + float(self.config.geometry_eps)
        norm = torch.sqrt(norm_sq)
        sub = F.relu(self.config.submersion_floor - norm).square().mean()

        h_rows = []
        for d in range(xy_req.shape[1]):
            row = torch.autograd.grad(
                grad[:, d].sum(),
                xy_req,
                create_graph=True,
                retain_graph=True,
            )[0]
            h_rows.append(row)
        hessian = torch.stack(h_rows, dim=1)
        quad = torch.einsum("ni,nij,nj->n", grad, hessian, grad)
        relative_rate = quad / norm_sq.pow(1.5)
        relative = relative_rate.square().mean()
        return sub, relative

    def loss(
        self,
        *,
        xy: Tensor,
        tissue_index: Tensor,
        time: Tensor,
        counts: Tensor,
        log_exposure: Tensor | None = None,
        submersion_weight: float = 0.0,
        relative_position_weight: float = 0.0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        out = self.forward(xy, tissue_index, time, log_exposure)
        data = self.poisson_nll(counts, out["log_mu"])
        if submersion_weight > 0 or relative_position_weight > 0:
            sub, rel = self.coordinate_geometry_penalties(xy, tissue_index)
        else:
            sub = data.new_zeros(())
            rel = data.new_zeros(())
        total = (
            data
            + float(submersion_weight) * sub
            + float(relative_position_weight) * rel
        )
        return total, {
            "data_nll": data.detach(),
            "submersion": sub.detach(),
            "relative_position": rel.detach(),
            "z_min": out["z"].min().detach(),
            "z_max": out["z"].max().detach(),
        }
