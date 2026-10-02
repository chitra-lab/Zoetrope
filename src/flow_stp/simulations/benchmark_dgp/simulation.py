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


def physical_velocity(
    z: np.ndarray | float,
    t: np.ndarray | float,
    config: BenchmarkSimulationConfig,
) -> np.ndarray:
    """Independently specified axial physical velocity field V*(z,t)."""

    z_arr, t_arr = np.broadcast_arrays(
        np.asarray(z, dtype=float),
        np.asarray(t, dtype=float),
    )
    temporal = 1.0 + config.velocity_temporal_modulation * np.sin(2.0 * np.pi * t_arr)
    spatial = 1.0 + config.velocity_spatial_modulation * np.cos(2.0 * np.pi * z_arr)
    return (
        config.velocity_scale
        * 4.0
        * z_arr
        * (1.0 - z_arr)
        * temporal
        * spatial
    )


def _rk4_step(y: np.ndarray, t: float, dt: float, rhs) -> np.ndarray:
    k1 = rhs(y, t)
    k2 = rhs(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs(y + dt * k3, t + dt)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _integrate_flow(config: BenchmarkSimulationConfig) -> np.ndarray:
    a = config.a_grid
    times = config.time_grid
    flow = np.empty((len(times), len(a)), dtype=float)
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


def _normalize_columns(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=0, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def _initial_expression_curve(
    config: BenchmarkSimulationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    a = config.a_grid
    k = config.n_spatial_programs

    centers = np.linspace(0.08, 0.92, k)
    centers += rng.normal(0.0, 0.035, size=k)
    centers = np.clip(centers, 0.03, 0.97)
    widths = rng.uniform(0.10, 0.28, size=k)

    basis = np.stack(
        [np.exp(-0.5 * ((a - c) / w) ** 2) for c, w in zip(centers, widths)],
        axis=1,
    )
    basis -= basis.mean(axis=0, keepdims=True)
    basis = _normalize_columns(basis)

    loadings = rng.normal(size=(config.n_genes, k))
    loadings /= np.maximum(np.linalg.norm(loadings, axis=0, keepdims=True), 1e-12)
    loadings *= config.spatial_program_scale * np.sqrt(config.n_genes)

    baseline = config.psi_baseline + rng.normal(
        0.0,
        config.psi_baseline_jitter,
        size=config.n_genes,
    )
    psi0 = baseline[None, :] + basis @ loadings.T

    minimum = psi0.min(axis=0)
    shift = np.maximum(config.psi_floor - minimum, 0.0)
    psi0 += shift[None, :]

    return psi0, loadings


def _draw_molecular_programs(
    config: BenchmarkSimulationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    k = config.n_molecular_programs
    loadings = rng.normal(size=(config.n_genes, k))
    loadings /= np.maximum(np.linalg.norm(loadings, axis=0, keepdims=True), 1e-12)
    loadings *= config.molecular_program_scale * np.sqrt(config.n_genes)

    params = {
        "frequency_a": rng.integers(1, 4, size=k).astype(float),
        "frequency_t": rng.integers(1, 4, size=k).astype(float),
        "phase_a": rng.uniform(0.0, 2.0 * np.pi, size=k),
        "phase_t": rng.uniform(0.0, 2.0 * np.pi, size=k),
        "mixture": rng.uniform(0.25, 0.75, size=k),
    }
    return loadings, params


def _molecular_raw_field(
    a: np.ndarray,
    t: float,
    loadings: np.ndarray,
    params: dict[str, np.ndarray],
) -> np.ndarray:
    aa = np.asarray(a, dtype=float)[:, None]
    fa = params["frequency_a"][None, :]
    ft = params["frequency_t"][None, :]
    pa = params["phase_a"][None, :]
    pt = params["phase_t"][None, :]
    mix = params["mixture"][None, :]

    spatial_wave = np.sin(np.pi * fa * aa + pa)
    temporal_wave = np.cos(np.pi * ft * float(t) + pt)
    transient = np.exp(-0.5 * ((float(t) - mix) / 0.22) ** 2)
    latent = spatial_wave * (0.65 * temporal_wave + 0.35 * transient)
    return latent @ loadings.T


def _expression_rhs(
    psi: np.ndarray,
    t: float,
    config: BenchmarkSimulationConfig,
    molecular_loadings: np.ndarray,
    molecular_params: dict[str, np.ndarray],
) -> np.ndarray:
    a = config.a_grid
    psi_a = np.gradient(psi, a, axis=0, edge_order=2)
    tangent_norm_sq = np.sum(psi_a * psi_a, axis=1, keepdims=True)
    tangent_norm = np.sqrt(np.maximum(tangent_norm_sq, 1e-12))
    tangent_unit = psi_a / tangent_norm

    raw = _molecular_raw_field(a, t, molecular_loadings, molecular_params)
    tangent_component = np.sum(raw * tangent_unit, axis=1, keepdims=True)
    perpendicular = raw - tangent_component * tangent_unit

    # A separate smooth scalar driver creates a controlled violation of local
    # molecular persistence.  At parallel_strength=0 the material velocity is
    # orthogonal to the expression-space spatial tangent by construction.
    parallel_driver = (
        config.molecular_program_scale
        * np.sin(np.pi * a)[:, None]
        * np.cos(2.0 * np.pi * float(t))
    )
    parallel = parallel_driver * tangent_unit

    return perpendicular + config.parallel_strength * parallel


def _integrate_expression(
    config: BenchmarkSimulationConfig,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
]:
    psi0, spatial_loadings = _initial_expression_curve(config, rng)
    molecular_loadings, molecular_params = _draw_molecular_programs(config, rng)

    times = config.time_grid
    psi = np.empty((len(times), config.n_a_grid, config.n_genes), dtype=float)
    psi[0] = psi0

    rhs = lambda state, t: _expression_rhs(
        state,
        t,
        config,
        molecular_loadings,
        molecular_params,
    )

    for k in range(len(times) - 1):
        dt = float(times[k + 1] - times[k])
        psi[k + 1] = _rk4_step(psi[k], float(times[k]), dt, rhs)
        if np.min(psi[k + 1]) <= config.psi_floor * 0.25:
            raise RuntimeError(
                "Expression trajectory approached zero in Fisher coordinates. "
                "Increase psi_baseline or reduce program scales."
            )

    material_velocity = np.stack(
        [rhs(psi[k], float(times[k])) for k in range(len(times))],
        axis=0,
    )
    return psi, material_velocity, spatial_loadings, molecular_loadings, molecular_params


def _draw_off_axis_programs(
    config: BenchmarkSimulationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    k = config.n_off_axis_programs
    loadings = rng.normal(size=(config.n_genes, k))
    loadings /= np.maximum(np.linalg.norm(loadings, axis=0, keepdims=True), 1e-12)
    loadings *= config.off_axis_program_scale * np.sqrt(config.n_genes)
    params = {
        "frequency_a": rng.integers(1, 4, size=k).astype(float),
        "frequency_t": rng.integers(1, 4, size=k).astype(float),
        "phase": rng.uniform(0.0, 2.0 * np.pi, size=k),
        "u_power": rng.integers(1, 3, size=k).astype(int),
    }
    return loadings, params


def _off_axis_effect(
    a: np.ndarray,
    u: np.ndarray,
    t: np.ndarray | float,
    loadings: np.ndarray,
    params: dict[str, np.ndarray],
) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1, 1)
    u = np.asarray(u, dtype=float).reshape(-1, 1)
    t = np.broadcast_to(np.asarray(t, dtype=float), (len(a),)).reshape(-1, 1)

    fa = params["frequency_a"][None, :]
    ft = params["frequency_t"][None, :]
    phase = params["phase"][None, :]
    power = params["u_power"][None, :]

    u_term = np.where(power == 1, u, u**2 - 1.0 / 3.0)
    latent = (
        u_term
        * np.sin(np.pi * fa * a + phase)
        * np.cos(np.pi * ft * t + 0.5 * phase)
    )
    return latent @ loadings.T


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    dt = np.diff(x)[:, None]
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dt, axis=0)
    return out


def _derive_truth(
    config: BenchmarkSimulationConfig,
    rng: np.random.Generator,
) -> BenchmarkTruth:
    flow = _integrate_flow(config)
    (
        psi,
        material_velocity,
        spatial_loadings,
        molecular_loadings,
        molecular_params,
    ) = _integrate_expression(config, rng)
    off_axis_loadings, off_axis_params = _draw_off_axis_programs(config, rng)

    a = config.a_grid
    times = config.time_grid
    psi_a = np.gradient(psi, a, axis=1, edge_order=2)
    flow_a = np.gradient(flow, a, axis=1, edge_order=2)
    if np.any(flow_a <= 0.0):
        raise RuntimeError("Flow derivative must stay positive.")

    psi_z = psi_a / flow_aShêaÊÈj[é+"q©ozZÜ-g¬6Þrß
±êïi¸¥Ü§¦Ë¦¦È³¦È³k¬Úèç-ç¦Ë¦«^®&¥½éhr+r¦È³k¬Ûùé¬bé¨nÇ«½¦â+rÕív~+!z»é+"q©o¦º#yËb¢wâ²«­ë"væ¥«^®&¥½éhr+r¦º#yËb¢shêl9µêâj[)yçg¦X§jX'¢¹¦j×«©ozZÜÆ+6~+!z»)yçg¦X§jX'¢¹È^®·¬Û¬b³já¢f­z¸Ë¦ºV­÷­­ª^Î«^®&¥²vØ¦zÊá¡ø¬êÜºk¥jØ¯zÚÚ¥ìè×â²«²vØ¦zÈrß
Z­©ez[-­éà¶4¬^®º+~Z¶zfkéi»È^®úaÊÈj[â~f±zºè¯¹­¨¬y§¶)ºè­+k¹Ën­ç[zw!ªä}¨¥yØ­²'^ØÆ­Üç$~f±Uø¬êÕ¦¬Æ¥¬^®º+ÝèéÚë-º^|aµÓw_¢¸"¶§ç(ø §zÊÞ¶êçéÜf«:î¶ ®'Zrr¶)¸¶)±Ê)Ê§zv¦zÈéÚë)²)Z¶§&§¦È²Ü©zz_­ßiµêâj[Þ"·)µêâj[Þ"·&¬·*^å¡«wÙùhÁùhÁ«-Ê§¥ùhjÝö¦¬Æ¥½éhr+r¦¬Æ¥½«-Ê§¥ùhjÝö¦È³jZ'ùhÂ"Í«-Ê§¥ùhjÝö~+!z»Þ"·'â²«½«-Ê§¥ùhjÝö«^®&¥²vf­z¸Ê^yÖ¬·*^å¡«wÙø¬êì¥ç~+!z»)yçZ²Ü©zz_­ßgâ²«¡»®öX­Êìz»Ún)b·&¬·*^å¡«wÚ¸h«^®&¥®&j×«©Z²Ü©zz_­ßjá¡ø¬êëâ²«jËr¥éé~Z·}¬¥«bjYhiØ§Ë)jØZv)à±«-Ê§¥ùhjÝö^réZ®Zv)à²j%yË¥j¹hiØ§Æ¬·*^å¡«wÚßk¬x,¡÷ÚÆ+%¡§b²Ü©zz_­ßi¨ç.ªé®+jjZ­©µêìzZ±ªëk+ß¢¹/y¨ç.ªéj¶¦²+^Ê}¬b²è¶¦¥ªÚë^®É'¥«®¶²½ú+ø§¡÷ÚÆ+)j¶¦²+^Ç^~)ízºmgjºÚÊzguªëk+bë'¦wZ®¶²j
âvzguªëk+Z)bæ«{^®%jØ¨Ú­è.ªíg¸¶zZ±ªëk+]·*^~Z¶·¬ª^Õ©éjÆ«­¬vÜ©yùhjÚÞ²©{XéíéÚ­¨¬yV¥¹á+®íjwZë-«Þz«W§ØmYé±æ«r(®×¶)²Û"uêâmYérX©¶-uéíg¬ÖØ´¶-uj-g¦Ç­Èl¢»^u¨+Ö¬×«mj-g¥Éb¥¨µÖW§j
âwV¢Ñ¨µ×m¶)²Ø´þzfk¦ºkbë-[bë-M^×li¨+Ö¢Óùé¬bé¸j-Z¸j-5{]­­¨¥x5jºÚÊwbl-ÂÚÞ²©{_­­¨¥x0k«zÈZ¥í~¶¶¢)à¿M®¶²¶-KôÕªëk+bÑ¨µ¿]®¶²¶-ZKõÕªëk+bÕ¨µ­ën®}tÂÝtÁ«ôÓì¿M~ÂÝtÁ«õÓì¿]]yü®b²ËÎzguªëk+§¦wZ®¶²µùhjÚÞ¦Xj×¢×(ø éÜf«(¦ºV­Â¢wâ¹àÚÚ&éÞ­«h®Û©ééÖ«­¬§¦wZ®¶²v'-²Úß­rêïjÛ«yÊ'~(º»Ú¶êÞm«ùÊ'~(º»Ú¶êÞ¶)¶·§uùhjÛ~®x'¢¹M¢wâË«½«n­èâ¶×«rêïjÛ«{]"~¹à­©Ý¢m9z[×L"vØi«®x.'è®m4ÛIé¦'1ÍÌº»Ú¶êÞ"zi;Z§·'.®ö­º·§¦§¥Ê,bÎÖ§éíæ*®ÝtúÖ§éíËiñµ©àz{rþÖ§éíæ-tþÖ§éíæ©_Â'mÊ'~(-Ë.z¥¶tùÊ'~(0Ûan«b¢yé²)öÒzi?°Ûa¦¬{1ú¥¶§Ç'2ú¥¶§Ê·ç-õÒ'Ü¢wâ¶§v«yù^rØ¨©Ý®x+jwhN^ÇµÓÞ~W¶*'Å©àêçéâ~æbb÷(ø ­©Ý¢jèµ«b¢w¥±í4q«(±©àéé²)Ú	^Æ·ÆÆ²Ê»ÇçÊÜmÊ¹àæj]4rßkj{%jØ¨Çl7¶Æ¿­Ç*þ·)µ§.®ö­º·­rêïjÛ«zº-jØ¨¶~Zµ©àêÞ~W¶*'~Z¶·ç-í­©ì«b¢|_­·kj{%jØ¨'å¡«mË¶ajÇ­Â'mZ±êÞ¦Xj×¢ÚÞ¦Xj×«zÛ«ò®g­i×±©©ç(º{lééÖ«­¬Ê^®È¨zguªëk)¨uél¶ºçzkjwhg§z¶­¢¹éÖ«­¬¢~jzZhË(·­º¹ë
hË(k²Ü©zzbÝö©¦hú"²Ê'­êkzÇ§µ«b¢z4"¶Z­®úk¶þØ^µ«azÖÊ^®È¨Ú'z¶­z¹à©¦jÈZ¥ëazÖ¬q©^Æ)®k´ÓûazÖ«zÛ«¹à¦¬²ëj×²Ü©zzbÝöuçëºvÞÈfj¹,k¥jØ¨Ê'~(zw!ªäJ)®«b¢p¨ø 6Þ6ÞéÜf«(¦ºV­ÑzË¥´g§z¶­y×¬¶»¶+Þ²­©m­©ìr¸©¶¢rÉÚ¦Èh¶Æ§vuéí®ëarß ^Èfj¹k¥jØ¨*'~("}Ê'~("°Ú'yélyÊ'~(¢wâö¥Ö­z¹àÚÚ&uçÚº[k(ø ­©Ý¢kyÛkºØ]z¸¯zÚî¶(ø ®x)¢¥jØ¨w(ø ­ë(÷«^Úh¦éZ¶*'²,Þ¥ªíÉ^j¹àºx¢¹´Ó],7©¢¥jØ¨zZ®Øë«§ú+]5ÒÈ³z)ºV­çÈ§W«jwh«ëZ¶*'ÊZ®Ø­Þv)í¢Ø¬²ç¬æ«j{^z)àuë-®ç-÷¬ªllëÞ­ìi¢Ç­ëéZµéí¥ªíÉ^·zÇ¥yË^v¹à¥êæºÖ­é¢¥jØ¨w(ø ìz»ÞuÇ¥Ç.®Ê+ÑØ¬¥êìç¥ìi®x'¢¹Ê'~('mØ¬¥êìå¢	jw(ø ·b²«²*',vÈ³yÊ'~('éÞ²ì~¶¦zÉb²Ú]«Z¶¦zÚî¶ëjg¬+-¥ÐÚµ¡kjg¢éí²«¶Éb²ÙéÖ«­¬¦y©éj»l+-ÝjºÚÊØ¬²ç¦zÖ«£%Ë]Ël¶°'ÉÇ¥Ê.×«ÒØ¬²ç¢éíz½¢»bè§uìmw§ºg«j×¢wâØ¬²ç­g¬~ëz§­©àyÊ'~(+zbq«^²«¶)Û,zWµç\º»(­Ë«²þrß[)z»b²Ërêì¢¿¢wâÇ¥Ê^®Ø¬²ç¥ªíÉ^j'lº«¶'%zè²Ûg¥û¥W§Û­·8§µêé¶)jÚî¶å£kºØmg ®'m®ëaj
âvÛZ¦ÈÆ&¥{^®bæ­®ëa¦È¥j
ÚÚî¶bè+ÛkºØZ¸¶Ö¨}ö±Ê}¬b±çßyËZºÛm®ëa¡÷ÚÆ+%¡§b-®ëa¡÷ÚÆ+)®+jjZ­©µêì¦ÈzYi²&±©~rß
}¬b²Ëkzx-ßk¬ùéj|©²']4­¨¬y§¶)ºè¬çßk¬¥êíº¶Ú¶*'§^+!z·åj×±¦·¬²*'é¢È­÷«yÛzßk¬²ÚÞa¢º}¬b²è¶¦±Æ¥y¸¨"q©fy©ôÛlÇ¥ibn¶«Éö¶ç¥ìi®x'¢¹M¢wâX­ªò,vÈ³zW§Û%ºÚ¯'ÚrÚ+þz^Æ9rß	bn¶«ÊZ ±Ý¦¹¸¨"q©fy©åºÚ¯'ÚrÚ+6ÞÊÆ¦¦W¢éí²kÊ^®È¨Ê'~((nÇ«½«b¢y¨uékrr+v¨«¦Ø¬²ç³º×å¡«m­êeÆ­z·©rß(ø ®x+b²Ë×í¶)w^ÇM®·©Óg\zYbvÉéjºÚÉ÷Y4ñ×è®H§­©àyÇ¥Ê.×«qéer§µêþéâvÇW(º{^¯é^'l¡»­©±ªizwit6­hZÚçXqéeÛ-Ë.z'mË.z'mg­¶Ø¦z)Ý{bè§uìkzbq«^­êqÇ,¢Ý¦¬Æ¥¾)ízºmg¶»­r²'÷¥¡È­ÊÚî¶bè+ÛkºØZ¸¶ÖÈ^®ø§µêé¶)jÚî¶â²«½éhr+r¶»­Ø¦z
âvÚî¶ ®'mµª¦j×«©b×«¦Ø¦y«kºØfj×«©l¥ç¶»­Ø¦z
âvÚî¶ ®'mµªÈ^®)ízºmg¶»­ø¬êì¥ç¶»­Ø¦z
âvÚî¶ ®'mµªá¢f­z¸)ízºmg¶»­¸h«^®&¥¶»­Ø¦z
âvÚî¶ ®'mµªá¡ø¬êâ×«¦Ø¦y«kºØkâ²«¶»­Ø¦z
âvÚî¶ ®'mµ«kºØ_­©±ªizwit6­hZÚçXqéeÛ)j»brW¢v'ljÚîy«­®ç®ÎÚî{9bn¶«Éö¶í®ç¥ºÚ¯'ÚrÚ+Vr²'Úîzr²'õ_È^®Úîyø¬êïªf­z¸Úîz©µêâjZÈ^®Úîz§â²«®&j×«©m®ç«µêâjZá¡ø¬êí®ç«â²«r§¶ÊZ®Û¦§w)jzZ®Û¦§vk²Ü©zz_­ßkb²ËëZ®,j^Ûb²ËÛb²ËÛbçå¡«m¶)w^ÆØ¦z)Ý{¢kb²Ër§µêþÖì¥×(Æ­¡»­©²('¢·¢×±N»qée¶»­\¢w¶Úî¶ëjg¬	è­è§uìS®ç¢éí±ªëk)é¾ËZrG(º{l¥ªí²gªëk)é¾ËZrIjzZ®Û¢éí±×ét6­hZÚç(º{ljºÚÉÊ%ºiì¶»­§zv¦zÇ(º{luø§±êíÑÇ¥'hnÇX¶îÆµçfy©Ý~CjÖ­©æ§jºÚÉÊ%ºiì¶»­§zv¦zÇ±¥ç-yÙjw_{®ÝzYbvìqéeÛhé©ÈÇ­oÈèéµ§Zµ¨§µéíÚ\¢{Z{'¢V­z{i®+jh§~æjØ¨ènÇ«½«b¢y¨uéij¶¦z×«²§zg­iÖ­jCjÖ­©éÞ¶»­§zv¦zÁâu×§éÞëÞÚî¶¬²Ú+yØ§jÛk±©ÝÂ¸­µéìz«j×¥É¼­ë½è^«²ÞÈfj¹+ºyÞ®Çz-iÇ"uéíjYr{h±è­éÞëZu«ZjÛk²uéàzw­®ëa¥ÐÚµ¡kjg zw­®ëaéÞ©²vÝÊ^®È¨Ø¬¥êìì¥«bjYhiØ§z+zev¥z+Úî¶)jØZv)à±¬b³Y¨ç.ªå¡§b	è®ié)Ú	è®kkºØf¢WºV«x,k¬Ößk¬x'¢¹§¦X§jX'¢¹­®ëa¡÷ÚÆ+%¡§bÆ+5JÚ+zØ¬²ç­­©ì~æjØ¨g­iÖ­j)Ú¶Úì­«az»ajw±¦"­¶(­x&zØhvÆ¬¥ªí¡ûazìz»Þu·§r®KZnW¨nÆ­¶»-Ë.z¨ëkÊCjÖ­©¶+,¹éµªèÂÊÞ¶êçéÜf«(¦ºV­ÑzË¥µÊ'~(¢wâì¡»¢éí±Ê.Û}ìiyË^vgìiyË^vg×ÜzYm®ëaqée¶»­§zg­iÖ­j§zg­iÖ­jÚî¶kºØ]yøì¢{}ëÚç¢~+"ËZÇ¯j[ÝjºÚÊ·­º¹ïj[¶b²ØÈ§²Ö§qëÚç§¦§z¸­ën®{Úç¢µé¢~+"ËZÇ¯j[¶êez·­º¹åËoj[ø¬{-jw½©nyØ¶·­º¹ì¶¹#²ìi÷¯~ä¾)ïj[×¦²·­º¹ïj[uçìj÷zw!ªä²)®«b¢zÞ²éméÜf«(¦ºV­ÑzË¥¶­¦ë]»-¬ö­Æ¯yìiyË^vgº(:îxö­&¯zìz»Þv)éºÛ,z«j×¥ÉúèuéÛzw!ªä¶»­­¦ëOjØhºÚnµØ«{ZÛ¬zºÞ²oz­¦ëfØ«¥ªÞÛ®çÆ+-¢Dë¹êÞ²ém¡»-¡Ë/¢ëiºßènÇ«½ç\zYfzÖjÖ²ø3w^ÄV¥±êÞ²émr§¶ËhrËèºÚn·÷(º{lrËàÎ)Ý{ZÇ«zË¥¶§zg­iÖ­jÚ²ú.¶­þ§zg­iÖ­iË/w^ÄV¥±èbu×§éÞ¶»­·¬º[`zw¦zÖjÖ¶Úìëa×^§zÚî¶']zxëkºØb²z-6Þ']zxëkºØm¡Ë/¢ëiºßá×^§zÚî¶,¾)Ý{ZÇ¢~Æ¯yìiyË^vg·¬º[^Æµçfy©í¡Ë/¢ëiºßá×^ìiyË^uÊ.Ùjw,¾â×±©lxxuéí®ëaÇ^&Þ­«^+· j×~º&ëa¡Ûâ²&åz)éºÛ+zË¥µÇ¥Úî¶hrËèºÚn·øbu×§qée¶»­Ë/8§uìEj[ê&zÚò­ë.Ú±«m®È¶Ø¬²ç z¶¼¢~¨ëkÊ+'¢Óhè¢g­¯+hrËèºÚn·øbu×§¶+,¹è¢g­¯',¾)Ý{ZÇ­®ëa­ë.ÛkºØm®ëa¥¬¥¡§Z¸¶»­¨+Ûbè+ÛkºØmg ®'i²)Z¶§&§¶»­"¨+jx"jyµêâj[Þ"·+kºØfj×«©ozZÜ-®ëa~Z0¦¬Æ¥½éhr+r¶»­r²'÷¥¡È­Ê"Í©hå£kºØi²,Úà~Z0~+!z»Þ"·+kºØ_È^®÷¥¡È­Êf­z¸Ê^yÛkºØfj×«©l¥ç~+!z»)yçm®ëa~+!z»)yç_È^®ìz»Ún)b·+kºØ_È^®ìz»Ún)b·*á¢f­z¸Úî¶á¢f­z¸¸h~+!z»kºØkâ²«²­©e¡§b-®ëa²­©e¡§b&¢WºV«x,¶»­j%yË¥j¹hiØ§Ê}¬b²Zv)à²Úî¶}¬b²Zv)à±ú+¦·Z­©¬y¨ç.ªí®ëa^réZ®è¶¦¥ªÚë^®Ê}¬b²Úî¶}¬b²è¶¦¥ªÚë^®Çè®G²½©nz)éj¶¦²+^ËkºØik)hi×é­çâÆG²½©nzzlj÷³r©­ë,yÚ.¶­þuéí®ëajºÚÊÉéÎÚî¶ZÊZw­^­¦ërßì¢|Êx.µÿ²§vW£²Ýºjc²ìi÷±Ø¶·¬º[\¢wâ§vW¢×§·k.f«Ê{b²Ë²)í­ë.×(ø Ø¬²ç¬Ç¥È§¶W§­ë.Ú²xë"ÚÞ²émrß	àzw¬¡»®ö­æ¡×¥­ë.×(ø ¡»®ö­æ¡×¥¡÷ÚÆ+,¶·§Ø_­­ë.×(ø ¡÷ÚÆ+,¶·§Øij¶¥él¶·§Ø_­­ë.×(ø ¥ªÚW¥²ÚÞa¬l¶»¶êÞv÷¥¡È­Ê©~Z¶zfkéi»+zË¥¶Úî¶â²«½éhr+r­ë.ÛkºØi+"q©ozZÜ°Øh¥éèºÚn·û.f«Ê;(§rb­Æ¬©Ýèì¢wn.f«Ê§vW¢×§·jÞ¶êç¢ëiº