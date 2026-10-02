"""Method-independent biological simulator for comparative spatial TI benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import BenchmarkSimulationConfig


@dataclass(frozen=True)
class BenchmarkTruth:
    """Ground truth retained only for evaluation, never exposed to methods."""

    a_grid: np.ndarray
    time_grid: np.ndarray
    gene_names: tuple[str, ...]
    psi_lagrangian: np.ndarray
    material_velocity: np.ndarray
    flow: np.ndarray
    physical_velocity: np.ndarray
    psi_z_along_flow: np.ndarray
    fisher_velocity: np.ndarray
    material_speed: np.ndarray
    fisher_speed: np.ndarray
    fisher_observability: np.ndarray
    rho_material: np.ndarray
    rho_fisher: np.ndarray
    spatial_loadings: np.ndarray
    molecular_loadings: np.ndarray
    off_axis_loadings: np.ndarray
    molecular_program_parameters: dict[str, np.ndarray]
    off_axis_program_parameters: dict[str, np.ndarray]


@dataclass(frozen=True)
class BenchmarkSimulationResult:
    """Observed benchmark data plus hidden truth used for scoring."""

    config: BenchmarkSimulationConfig
    obs: pd.DataFrame
    counts: pd.DataFrame
    expected_mean: pd.DataFrame
    cell_truth: pd.DataFrame
    gene_metadata: pd.DataFrame
    truth: BenchmarkTruth



def physical_velocity(z, t, config: BenchmarkSimulationConfig) -> np.ndarray:
    """Independently specified axial physical velocity field V*(z,t).

    Spatial structure is deliberately nonlinear, while the explicit temporal
    factor is affine in time.  This keeps the within-tissue velocity field
    challenging without asking sparse developmental snapshots to identify
    hidden temporal oscillations.
    """
    z_arr, t_arr = np.broadcast_arrays(np.asarray(z, float), np.asarray(t, float))

    temporal = 1.0 + config.velocity_temporal_modulation * (
        2.0 * t_arr - 1.0
    )
    spatial = 1.0 + config.velocity_spatial_modulation * np.cos(
        2.0 * np.pi * z_arr
    )

    return (
        config.velocity_scale
        * 4.0
        * z_arr
        * (1.0 - z_arr)
        * temporal
        * spatial
    )


def _rk4_step(y, t, dt, rhs):
    k1 = rhs(y, t)
    k2 = rhs(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs(y + dt * k3, t + dt)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _integrate_flow(config):
    a = config.a_grid
    times = config.time_grid
    flow = np.empty((len(times), len(a)), float)
    flow[0] = a

    for k in range(len(times) - 1):
        t = float(times[k])
        dt = float(times[k + 1] - times[k])
        flow[k + 1] = _rk4_step(
            flow[k],
            t,
            dt,
            lambda z, s: physical_velocity(z, s, config),
        )
        flow[k + 1] = np.clip(flow[k + 1], 0.0, 1.0)

    if np.any(np.diff(flow, axis=1) <= 0.0):
        raise RuntimeError("Physical flow ceased to be order preserving.")

    return flow


def _normalize_columns(x):
    return x / np.maximum(
        np.linalg.norm(x, axis=0, keepdims=True),
        1e-12,
    )



def _initial_expression_curve(config, rng):
    a = config.a_grid
    k = config.n_spatial_programs

    centers = np.linspace(0.08, 0.92, k)
    centers += rng.normal(0.0, 0.035, size=k)
    centers = np.clip(centers, 0.03, 0.97)
    widths = rng.uniform(0.10, 0.28, size=k)

    raw_basis = np.stack(
        [
            np.exp(-0.5 * ((a - center) / width) ** 2)
            for center, width in zip(centers, widths)
        ],
        axis=1,
    )

    raw_basis_a = np.stack(
        [
            raw_basis[:, j]
            * (-(a - center) / (width**2))
            for j, (center, width)
            in enumerate(zip(centers, widths))
        ],
        axis=1,
    )

    centered = raw_basis - raw_basis.mean(axis=0, keepdims=True)
    column_norms = np.maximum(
        np.linalg.norm(centered, axis=0, keepdims=True),
        1e-12,
    )

    basis = centered / column_norms
    basis_a = raw_basis_a / column_norms

    loadings = _normalize_columns(
        rng.normal(size=(config.n_genes, k))
    )
    loadings *= (
        config.spatial_program_scale
        * np.sqrt(config.n_genes)
    )

    baseline = (
        config.psi_baseline
        + rng.normal(
            0.0,
            config.psi_baseline_jitter,
            size=config.n_genes,
        )
    )

    psi0 = baseline[None, :] + basis @ loadings.T
    psi0_a = basis_a @ loadings.T

    psi0 += np.maximum(
        config.psi_floor - psi0.min(axis=0),
        0.0,
    )[None, :]

    return psi0, psi0_a, loadings



def _draw_molecular_programs(config, rng, spatial_loadings):
    """Draw a spatially complex molecular slope with constant Fisher norm.

    The molecular directions live in the orthogonal complement of the initial
    spatial-expression subspace.  Each latent program rotates in its own
    two-dimensional gene-space plane as a function of anatomical position.
    The resulting material-expression slope can vary strongly with anatomy
    while retaining constant norm, which makes Psi(a,t) exactly affine in time
    in the structured regime and keeps the Fisher identification exact.
    """
    k = config.n_molecular_programs
    needed = 2 * k

    spatial_basis, _ = np.linalg.qr(
        spatial_loadings,
        mode="reduced",
    )

    candidate = rng.normal(
        size=(config.n_genes, needed)
    )
    candidate -= spatial_basis @ (
        spatial_basis.T @ candidate
    )

    molecular_directions, _ = np.linalg.qr(
        candidate,
        mode="reduced",
    )

    molecular_directions = molecular_directions[:, :needed]

    amplitudes = rng.uniform(
        0.5,
        1.5,
        size=k,
    )
    amplitudes /= np.maximum(
        np.linalg.norm(amplitudes),
        1e-12,
    )
    amplitudes *= (
        config.molecular_program_scale
        * np.sqrt(config.n_genes)
    )

    params = {
        "frequency_a": rng.integers(
            1,
            4,
            size=k,
        ).astype(float),
        "phase_a": rng.uniform(
            0.0,
            2.0 * np.pi,
            size=k,
        ),
        "amplitude": amplitudes,
    }

    return molecular_directions, params



def _molecular_raw_field(
    a,
    t,
    loadings,
    params,
):
    """Affine-time molecular expression slope used by the benchmark.

    The returned field is independent of t: Psi(a,t) is therefore a straight
    line in Fisher-flat expression space along each hidden Lagrangian label.
    Spatial dependence remains nonlinear through rotating gene-space programs.
    """
    del t

    a = np.asarray(
        a,
        float,
    ).reshape(-1)

    k = len(
        params["frequency_a"]
    )

    cos_dirs = loadings[
        :,
        0 : 2 * k : 2,
    ]
    sin_dirs = loadings[
        :,
        1 : 2 * k : 2,
    ]

    frequency = params[
        "frequency_a"
    ][None, :]

    phase = params[
        "phase_a"
    ][None, :]

    amplitude = params[
        "amplitude"
    ][None, :]

    angle = (
        np.pi
        * a[:, None]
        * frequency
        + phase
    )

    cos_term = (
        amplitude
        * np.cos(angle)
    )
    sin_term = (
        amplitude
        * np.sin(angle)
    )

    return (
        cos_term @ cos_dirs.T
        + sin_term @ sin_dirs.T
    )



def _expression_rhs(
    psi,
    t,
    config,
    molecular_loadings,
    molecular_params,
):
    """Material expression velocity for the affine-time benchmark."""
    del psi

    a = config.a_grid

    perpendicular = _molecular_raw_field(
        a,
        t,
        molecular_loadings,
        molecular_params,
    )

    if config.parallel_strength == 0.0:
        return perpendicular

    # The violation remains affine in time at the level of Psi because this
    # driver is time independent.  It intentionally injects molecular change
    # in the initial spatial-expression tangent direction.
    psi0_a = molecular_params[
        "psi0_a_for_parallel"
    ]
    tangent_norm = np.sqrt(
        np.maximum(
            np.sum(
                psi0_a * psi0_a,
                axis=1,
                keepdims=True,
            ),
            1e-12,
        )
    )
    tangent_unit = psi0_a / tangent_norm

    parallel_driver = (
        config.molecular_program_scale
        * np.sqrt(config.n_genes)
        * np.sin(np.pi * a)[:, None]
        * tangent_unit
    )

    return (
        perpendicular
        + config.parallel_strength
        * parallel_driver
    )



def _integrate_expression(
    config,
    rng,
):
    (
        psi0,
        psi0_a,
        spatial_loadings,
    ) = _initial_expression_curve(
        config,
        rng,
    )

    (
        molecular_loadings,
        molecular_params,
    ) = _draw_molecular_programs(
        config,
        rng,
        spatial_loadings,
    )

    molecular_params = {
        **molecular_params,
        "psi0_a_for_parallel": psi0_a,
    }

    base_slope = _expression_rhs(
        psi0,
        0.0,
        config,
        molecular_loadings,
        molecular_params,
    )

    # Keep the entire affine path safely positive without changing its
    # molecular slope. Because the path is affine in time, its minimum occurs
    # at one of the two endpoints. A gene-wise constant shift changes neither
    # spatial derivatives nor material velocity.
    psi1_unshifted = (
        psi0
        + base_slope
    )

    endpoint_min = np.minimum(
        psi0.min(axis=0),
        psi1_unshifted.min(axis=0),
    )

    gene_shift = np.maximum(
        config.psi_floor
        - endpoint_min,
        0.0,
    )

    psi0 = (
        psi0
        + gene_shift[None, :]
    )

    material_slope = base_slope

    times = config.time_grid

    psi = (
        psi0[None, :, :]
        + times[:, None, None]
        * material_slope[None, :, :]
    )

    if np.min(psi) < config.psi_floor - 1e-10:
        raise RuntimeError(
            "Affine expression trajectory violated the Fisher-coordinate floor."
        )

    material_velocity = np.broadcast_to(
        material_slope[None, :, :],
        psi.shape,
    ).copy()

    # In the structured regime, compute the anatomical derivative analytically
    # for the rotating molecular programs.  This avoids turning a deliberately
    # exact positive control into a finite-difference boundary artifact.
    a = config.a_grid
    k = config.n_molecular_programs

    cos_dirs = molecular_loadings[
        :,
        0 : 2 * k : 2,
    ]
    sin_dirs = molecular_loadings[
        :,
        1 : 2 * k : 2,
    ]

    frequency = molecular_params[
        "frequency_a"
    ][None, :]
    phase = molecular_params[
        "phase_a"
    ][None, :]
    amplitude = molecular_params[
        "amplitude"
    ][None, :]

    angle = (
        np.pi
        * a[:, None]
        * frequency
        + phase
    )

    slope_a = (
        (
            -amplitude
            * np.pi
            * frequency
            * np.sin(angle)
        )
        @ cos_dirs.T
        + (
            amplitude
            * np.pi
            * frequency
            * np.cos(angle)
        )
        @ sin_dirs.T
    )

    if config.parallel_strength != 0.0:
        # The exact Fisher identity is intentionally broken in this regime,
        # so a numerical derivative of the added violation is sufficient.
        perpendicular = _molecular_raw_field(
            a,
            0.0,
            molecular_loadings,
            molecular_params,
        )
        parallel_part = (
            material_slope
            - perpendicular
        )
        slope_a += np.gradient(
            parallel_part,
            a,
            axis=0,
            edge_order=2,
        )

    psi_a = (
        psi0_a[None, :, :]
        + times[:, None, None]
        * slope_a[None, :, :]
    )

    stored_params = {
        key: value
        for key, value
        in molecular_params.items()
        if key != "psi0_a_for_parallel"
    }
    stored_params[
        "gene_shift"
    ] = gene_shift.astype(float)

    return (
        psi,
        material_velocity,
        psi_a,
        spatial_loadings,
        molecular_loadings,
        stored_params,
    )



def _draw_off_axis_programs(
    config,
    rng,
):
    k = config.n_off_axis_programs

    loadings = _normalize_columns(
        rng.normal(
            size=(
                config.n_genes,
                k,
            )
        )
    )

    loadings *= (
        config.off_axis_program_scale
        * np.sqrt(
            config.n_genes
        )
    )

    params = {
        "frequency_a": rng.integers(
            1,
            4,
            size=k,
        ).astype(float),
        "phase": rng.uniform(
            0.0,
            2.0 * np.pi,
            size=k,
        ),
        "u_power": rng.integers(
            1,
            3,
            size=k,
        ).astype(int),
        "amplitude_start": rng.uniform(
            -1.0,
            1.0,
            size=k,
        ),
        "amplitude_end": rng.uniform(
            -1.0,
            1.0,
            size=k,
        ),
    }

    return loadings, params



def _off_axis_effect(
    a,
    u,
    t,
    loadings,
    params,
):
    a = np.asarray(
        a,
        float,
    ).reshape(-1, 1)

    u = np.asarray(
        u,
        float,
    ).reshape(-1, 1)

    t = np.broadcast_to(
        np.asarray(
            t,
            float,
        ),
        (len(a),),
    ).reshape(-1, 1)

    fa = params[
        "frequency_a"
    ][None, :]

    phase = params[
        "phase"
    ][None, :]

    power = params[
        "u_power"
    ][None, :]

    amplitude_start = params[
        "amplitude_start"
    ][None, :]

    amplitude_end = params[
        "amplitude_end"
    ][None, :]

    temporal_amplitude = (
        (1.0 - t)
        * amplitude_start
        + t
        * amplitude_end
    )

    u_term = np.where(
        power == 1,
        u,
        u**2 - 1.0 / 3.0,
    )

    latent = (
        u_term
        * np.sin(
            np.pi
            * fa
            * a
            + phase
        )
        * temporal_amplitude
    )

    return latent @ loadings.T


def _cumulative_trapezoid(
    y,
    x,
):
    out = np.zeros_like(
        y,
        float,
    )

    dt = np.diff(
        x
    )[:, None]

    out[1:] = np.cumsum(
        0.5
        * (
            y[1:]
            + y[:-1]
        )
        * dt,
        axis=0,
    )

    return out


def _derive_truth(
    config,
    rng,
):
    flow = _integrate_flow(
        config
    )

    (
        psi,
        material_velocity,
        psi_a,
        spatial_loadings,
        molecular_loadings,
        molecular_params,
    ) = _integrate_expression(
        config,
        rng,
    )

    (
        off_axis_loadings,
        off_axis_params,
    ) = _draw_off_axis_programs(
        config,
        rng,
    )

    a = config.a_grid
    times = config.time_grid

    flow_a = np.gradient(
        flow,
        a,
        axis=1,
        edge_order=2,
    )

    if np.any(
        flow_a <= 0.0
    ):
        raise RuntimeError(
            "Flow derivative must stay positive."
        )

    # Along z = F(a,t), dPsi/dz = (dPsi/da)/(dF/da).
    psi_z = (
        psi_a
        / flow_a[:, :, None]
    )

    physical_v = physical_velocity(
        flow,
        times[:, None],
        config,
    )

    # Eulerian temporal derivative at fixed z.
    eulerian_t = (
        material_velocity
        - physical_v[:, :, None]
        * psi_z
    )

    observability = np.sum(
        psi_z * psi_z,
        axis=2,
    )

    if np.any(
        observability <= 0.0
    ):
        raise RuntimeError(
            "Fisher spatial observability "
            "must be positive everywhere."
        )

    # The Fisher projection is applied only for evaluation.
    fisher_v = (
        -np.sum(
            eulerian_t
            * psi_z,
            axis=2,
        )
        / observability
    )

    fisher_residual = (
        eulerian_t
        + fisher_v[:, :, None]
        * psi_z
    )

    material_speed = np.linalg.norm(
        material_velocity,
        axis=2,
    )

    fisher_speed = np.linalg.norm(
        fisher_residual,
        axis=2,
    )

    rho_material = _cumulative_trapezoid(
        material_speed,
        times,
    )

    rho_fisher = _cumulative_trapezoid(
        fisher_speed,
        times,
    )

    if (
        config.parallel_strength
        == 0.0
    ):
        max_error = float(
            np.max(
                np.abs(
                    fisher_v
                    - physical_v
                )
            )
        )

        if max_error > 5e-5:
            raise RuntimeError(
                "Structured benchmark failed its "
                "identification check: "
                f"max |V_fisher - V_physical|="
                f"{max_error:.3e}."
            )

    gene_names = tuple(
        f"GENE_{g + 1:03d}"
        for g
        in range(
            config.n_genes
        )
    )

    return BenchmarkTruth(
        a_grid=a.copy(),
        time_grid=times.copy(),
        gene_names=gene_names,
        psi_lagrangian=psi.astype(
            np.float32
        ),
        material_velocity=material_velocity.astype(
            np.float32
        ),
        flow=flow.astype(
            np.float32
        ),
        physical_velocity=physical_v.astype(
            np.float32
        ),
        psi_z_along_flow=psi_z.astype(
            np.float32
        ),
        fisher_velocity=fisher_v.astype(
            np.float32
        ),
        material_speed=material_speed.astype(
            np.float32
        ),
        fisher_speed=fisher_speed.astype(
            np.float32
        ),
        fisher_observability=observability.astype(
            np.float32
        ),
        rho_material=rho_material.astype(
            np.float32
        ),
        rho_fisher=rho_fisher.astype(
            np.float32
        ),
        spatial_loadings=spatial_loadings.astype(
            np.float32
        ),
        molecular_loadings=molecular_loadings.astype(
            np.float32
        ),
        off_axis_loadings=off_axis_loadings.astype(
            np.float32
        ),
        molecular_program_parameters={
            key: np.asarray(value)
            for key, value
            in molecular_params.items()
        },
        off_axis_program_parameters={
            key: np.asarray(value)
            for key, value
            in off_axis_params.items()
        },
    )


def _interp_time_a(
    array,
    times,
    a_grid,
    t,
    a,
):
    """Bilinear interpolation on a regular (time, a, ...) grid."""

    t = np.asarray(
        t,
        float,
    ).reshape(-1)

    a = np.asarray(
        a,
        float,
    ).reshape(-1)

    if len(t) != len(a):
        raise ValueError(
            "t and a must have equal length."
        )

    ti1 = np.clip(
        np.searchsorted(
            times,
            t,
            side="right",
        ),
        1,
        len(times) - 1,
    )

    ti0 = ti1 - 1

    ai1 = np.clip(
        np.searchsorted(
            a_grid,
            a,
            side="right",
        ),
        1,
        len(a_grid) - 1,
    )

    ai0 = ai1 - 1

    wt = (
        t - times[ti0]
    ) / np.maximum(
        times[ti1]
        - times[ti0],
        1e-12,
    )

    wa = (
        a - a_grid[ai0]
    ) / np.maximum(
        a_grid[ai1]
        - a_grid[ai0],
        1e-12,
    )

    trailing = (
        (1,)
        * (
            array.ndim
            - 2
        )
    )

    wt = wt.reshape(
        (-1,)
        + trailing
    )

    wa = wa.reshape(
        (-1,)
        + trailing
    )

    v00 = array[
        ti0,
        ai0,
    ]

    v01 = array[
        ti0,
        ai1,
    ]

    v10 = array[
        ti1,
        ai0,
    ]

    v11 = array[
        ti1,
        ai1,
    ]

    return (
        (1.0 - wt)
        * (
            (1.0 - wa)
            * v00
            + wa
            * v01
        )
        + wt
        * (
            (1.0 - wa)
            * v10
            + wa
            * v11
        )
    )


def _warp_tissue(
    z,
    u,
    *,
    t,
    replicate,
    config,
    rng,
):
    curvature = (
        config.curvature_base
        + config.curvature_time_trend
        * float(t)
        + rng.normal(
            0.0,
            config.curvature_jitter,
        )
    )

    if rng.random() < 0.5:
        curvature *= -1.0

    width_phase = rng.uniform(
        0.0,
        2.0 * np.pi,
    )

    center_x = z

    center_y = (
        curvature
        * np.sin(
            np.pi * z
        )
    )

    tangent_y = (
        curvature
        * np.pi
        * np.cos(
            np.pi * z
        )
    )

    tangent_norm = np.sqrt(
        1.0
        + tangent_y**2
    )

    normal_x = (
        -tangent_y
        / tangent_norm
    )

    normal_y = (
        1.0
        / tangent_norm
    )

    half_width = (
        config.tissue_half_width
        * (
            1.0
            + config.width_modulation
            * np.sin(
                2.0
                * np.pi
                * z
                + width_phase
            )
        )
    )

    x = (
        center_x
        + half_width
        * u
        * normal_x
    )

    y = (
        center_y
        + half_width
        * u
        * normal_y
    )

    reflection = (
        -1.0
        if (
            config.random_reflection
            and rng.random()
            < 0.5
        )
        else 1.0
    )

    x = reflection * x

    angle = (
        rng.uniform(
            -np.pi,
            np.pi,
        )
        if config.random_rotation
        else 0.0
    )

    cos_angle = np.cos(
        angle
    )

    sin_angle = np.sin(
        angle
    )

    x_rotated = (
        cos_angle * x
        - sin_angle * y
    )

    y_rotated = (
        sin_angle * x
        + cos_angle * y
    )

    (
        translation_x,
        translation_y,
    ) = rng.normal(
        0.0,
        config.translation_sd,
        size=2,
    )

    x_rotated += translation_x
    y_rotated += translation_y

    metadata = {
        "curvature": float(
            curvature
        ),
        "rotation_rad": float(
            angle
        ),
        "reflection": float(
            reflection
        ),
        "translation_x": float(
            translation_x
        ),
        "translation_y": float(
            translation_y
        ),
        "width_phase": float(
            width_phase
        ),
        "replicate": int(
            replicate
        ),
    }

    return (
        x_rotated,
        y_rotated,
        metadata,
    )


def _sample_counts(
    mu,
    dispersion,
    model,
    rng,
):
    if model == "poisson":
        return rng.poisson(
            mu
        ).astype(
            np.int32
        )

    theta = dispersion[
        None,
        :,
    ]

    latent_rate = rng.gamma(
        shape=theta,
        scale=(
            np.maximum(
                mu,
                0.0,
            )
            / theta
        ),
    )

    return rng.poisson(
        latent_rate
    ).astype(
        np.int32
    )


def run_benchmark_simulation(
    config: BenchmarkSimulationConfig | None = None,
) -> BenchmarkSimulationResult:
    """Generate destructive spatial-transcriptomic snapshots and hidden truth."""

    config = (
        BenchmarkSimulationConfig()
        if config is None
        else config
    )

    config.validate()

    rng = np.random.default_rng(
        config.random_seed
    )

    truth = _derive_truth(
        config,
        rng,
    )

    population_n = (
        config.resolved_latent_population_size
    )

    particle_a = rng.uniform(
        0.0,
        1.0,
        size=population_n,
    )

    particle_u = rng.uniform(
        -1.0,
        1.0,
        size=population_n,
    )

    selected = rng.permutation(
        population_n
    )[
        : config.n_observed_cells
    ]

    dispersion = np.exp(
        rng.normal(
            config.nb_dispersion_log_mean,
            config.nb_dispersion_log_sd,
            size=config.n_genes,
        )
    )

    obs_frames = []
    truth_frames = []
    counts_parts = []
    mean_parts = []
    tissue_meta_rows: list[
        dict[str, Any]
    ] = []

    cursor = 0
    cell_counter = 0

    for (
        time_index,
        t,
    ) in enumerate(
        config.tissue_times
    ):
        for replicate in range(
            config.replicates_per_time
        ):
            particle_ids = selected[
                cursor
                : cursor
                + config.cells_per_tissue
            ]

            cursor += (
                config.cells_per_tissue
            )

            a = particle_a[
                particle_ids
            ]

            u = particle_u[
                particle_ids
            ]

            tt = np.full(
                len(particle_ids),
                float(t),
            )

            z = _interp_time_a(
                truth.flow,
                truth.time_grid,
                truth.a_grid,
                tt,
                a,
            )

            psi_axial = _interp_time_a(
                truth.psi_lagrangian,
                truth.time_grid,
                truth.a_grid,
                tt,
                a,
            )

            off_axis = _off_axis_effect(
                a,
                u,
                tt,
                truth.off_axis_loadings,
                truth.off_axis_program_parameters,
            )

            psi_cell = (
                psi_axial
                + config.off_axis_strength
                * off_axis
            )

            if np.any(
                psi_cell <= 0.0
            ):
                raise RuntimeError(
                    "Off-axis perturbation made "
                    "Fisher-flat expression "
                    "nonpositive."
                )

            biological_mean = (
                0.25
                * psi_cell**2
            )

            library_factor = np.exp(
                rng.normal(
                    0.0,
                    config.library_log_sd,
                    size=len(
                        particle_ids
                    ),
                )
            )

            library_factor /= np.exp(
                0.5
                * config.library_log_sd**2
            )

            expected_count_mean = (
                biological_mean
                * library_factor[:, None]
            )

            counts = _sample_counts(
                expected_count_mean,
                dispersion,
                config.observation_model,
                rng,
            )

            (
                x,
                y,
                geometry,
            ) = _warp_tissue(
                z,
                u,
                t=float(t),
                replicate=replicate,
                config=config,
                rng=rng,
            )

            tissue_id = (
                f"t{time_index:02d}"
                f"_r{replicate:02d}"
            )

            cell_ids = np.asarray(
                [
                    f"cell_{index:08d}"
                    for index
                    in range(
                        cell_counter,
                        cell_counter
                        + len(
                            particle_ids
                        ),
                    )
                ]
            )

            cell_counter += len(
                particle_ids
            )

            obs_frames.append(
                pd.DataFrame(
                    {
                        "cell_id": cell_ids,
                        "tissue_id": tissue_id,
                        "time": float(t),
                        "time_index": int(
                            time_index
                        ),
                        "replicate": int(
                            replicate
                        ),
                        "x": x,
                        "y": y,
                    }
                )
            )

            def get_truth_field(
                array,
            ):
                return _interp_time_a(
                    array,
                    truth.time_grid,
                    truth.a_grid,
                    tt,
                    a,
                )

            truth_frames.append(
                pd.DataFrame(
                    {
                        "cell_id": cell_ids,
                        "particle_id": particle_ids,
                        "a_true": a,
                        "u_true": u,
                        "z_true": z,
                        "library_factor_true": library_factor,
                        "V_physical_true": get_truth_field(
                            truth.physical_velocity
                        ),
                        "V_fisher_true": get_truth_field(
                            truth.fisher_velocity
                        ),
                        "q_material_true": get_truth_field(
                            truth.material_speed
                        ),
                        "q_fisher_true": get_truth_field(
                            truth.fisher_speed
                        ),
                        "rho_material_true": get_truth_field(
                            truth.rho_material
                        ),
                        "rho_fisher_true": get_truth_field(
                            truth.rho_fisher
                        ),
                    }
                )
            )

            counts_parts.append(
                counts
            )

            mean_parts.append(
                expected_count_mean.astype(
                    np.float32
                )
            )

            tissue_meta_rows.append(
                {
                    "tissue_id": tissue_id,
                    "time": float(t),
                    "time_index": int(
                        time_index
                    ),
                    **geometry,
                }
            )

    obs = pd.concat(
        obs_frames,
        ignore_index=True,
    )

    cell_truth = pd.concat(
        truth_frames,
        ignore_index=True,
    )

    counts_df = pd.DataFrame(
        np.vstack(
            counts_parts
        ),
        columns=truth.gene_names,
    )

    counts_df.insert(
        0,
        "cell_id",
        obs[
            "cell_id"
        ].to_numpy(),
    )

    expected_mean_df = pd.DataFrame(
        np.vstack(
            mean_parts
        ),
        columns=truth.gene_names,
    )

    expected_mean_df.insert(
        0,
        "cell_id",
        obs[
            "cell_id"
        ].to_numpy(),
    )

    gene_metadata = pd.DataFrame(
        {
            "gene": truth.gene_names,
        }
    )

    gene_metadata.attrs[
        "hidden_gene_truth"
    ] = pd.DataFrame(
        {
            "gene": truth.gene_names,
            "nb_dispersion": dispersion,
            "spatial_loading_norm": np.linalg.norm(
                truth.spatial_loadings,
                axis=1,
            ),
            "molecular_loading_norm": np.linalg.norm(
                truth.molecular_loadings,
                axis=1,
            ),
            "off_axis_loading_norm": np.linalg.norm(
                truth.off_axis_loadings,
                axis=1,
            ),
        }
    )

    obs.attrs[
        "tissue_geometry"
    ] = pd.DataFrame(
        tissue_meta_rows
    )

    return BenchmarkSimulationResult(
        config=config,
        obs=obs,
        counts=counts_df,
        expected_mean=expected_mean_df,
        cell_truth=cell_truth,
        gene_metadata=gene_metadata,
        truth=truth,
    )


def _json_safe(value):
    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        tuple,
    ):
        return [
            _json_safe(item)
            for item
            in value
        ]

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _json_safe(
                item
            )
            for key, item
            in value.items()
        }

    return value


def save_benchmark_simulation(
    result,
    output_dir,
    *,
    save_expected_mean=True,
) -> Path:
    """Save observed inputs separately from hidden benchmark truth."""

    output = Path(
        output_dir
    ).expanduser().resolve()

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.obs.to_csv(
        output
        / "observed_cell_metadata.csv.gz",
        index=False,
    )

    result.counts.to_csv(
        output
        / "counts.csv.gz",
        index=False,
    )

    result.gene_metadata.to_csv(
        output
        / "gene_metadata.csv",
        index=False,
    )

    hidden_gene_truth = (
        result.gene_metadata.attrs.get(
            "hidden_gene_truth"
        )
    )

    if (
        hidden_gene_truth
        is not None
    ):
        hidden_gene_truth.to_csv(
            output
            / "hidden_gene_truth.csv",
            index=False,
        )

    if save_expected_mean:
        result.expected_mean.to_csv(
            output
            / "hidden_expected_count_mean.csv.gz",
            index=False,
        )

    result.cell_truth.to_csv(
        output
        / "hidden_cell_truth.csv.gz",
        index=False,
    )

    geometry = result.obs.attrs.get(
        "tissue_geometry"
    )

    if geometry is not None:
        geometry.to_csv(
            output
            / "hidden_tissue_geometry.csv",
            index=False,
        )

    truth = result.truth

    payload = {
        "a_grid": truth.a_grid,
        "time_grid": truth.time_grid,
        "psi_lagrangian": truth.psi_lagrangian,
        "material_velocity": truth.material_velocity,
        "flow": truth.flow,
        "physical_velocity": truth.physical_velocity,
        "psi_z_along_flow": truth.psi_z_along_flow,
        "fisher_velocity": truth.fisher_velocity,
        "material_speed": truth.material_speed,
        "fisher_speed": truth.fisher_speed,
        "fisher_observability": truth.fisher_observability,
        "rho_material": truth.rho_material,
        "rho_fisher": truth.rho_fisher,
        "spatial_loadings": truth.spatial_loadings,
        "molecular_loadings": truth.molecular_loadings,
        "off_axis_loadings": truth.off_axis_loadings,
    }

    for prefix, params in (
        (
            "molecular",
            truth.molecular_program_parameters,
        ),
        (
            "off_axis",
            truth.off_axis_program_parameters,
        ),
    ):
        for (
            key,
            value,
        ) in params.items():
            payload[
                f"{prefix}_{key}"
            ] = np.asarray(
                value
            )

    np.savez_compressed(
        output
        / "hidden_truth_arrays.npz",
        **payload,
    )

    with (
        output
        / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            _json_safe(
                asdict(
                    result.config
                )
            ),
            handle,
            indent=2,
        )

    summary = {
        "n_tissues": int(
            result.config.n_tissues
        ),
        "n_cells": int(
            len(
                result.obs
            )
        ),
        "n_genes": int(
            result.config.n_genes
        ),
        "observation_model": (
            result.config.observation_model
        ),
        "off_axis_strength": float(
            result.config.off_axis_strength
        ),
        "parallel_strength": float(
            result.config.parallel_strength
        ),
        "max_structured_velocity_gap": float(
            np.max(
                np.abs(
                    result.truth.fisher_velocity
                    - result.truth.physical_velocity
                )
            )
        ),
    }

    with (
        output
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    return output
