"""
黏度计算模块

基于斯托克斯公式计算液体黏度，支持壁面修正（Ladenburg 径向修正）和
雷诺数修正（Oseen），两者可独立开关、叠加生效。
（注：液柱高度修正暂未纳入壁面修正公式。）

【符号约定】
  公式中出现的 r、R、h 均为半径/高度值（非直径）。

  基础公式（斯托克斯定律）：
      η_basic = 2 * r² * g * (ρ_s - ρ_l) / (9 * v_t)

  壁面修正（Ladenburg 径向修正）：
      k_wall = 1 + 2.4 * r / R
      η_wall = η_basic / k_wall

    其中：
      r — 小球半径 (m)
      R — 量筒内半径 (m)
      注：液柱高度 h 暂未纳入壁面修正公式。

  雷诺数修正（Oseen 修正，迭代求解隐式方程）：
      k_Re = 1 + 3/16 * Re
      Re = 2 * r * v_t * ρ_l / η

    其中 η 和 Re 互为隐式依赖，通过迭代收敛。
    Oseen 修正适用于 Re ≲ 1 的范围。

  综合修正（两者叠加）：
      η_final = η_basic / (k_wall * k_Re)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _oseen_iteration(eta_basic, r, v_t, rho_l, wall_factor=1.0,
                     max_iter=20, tol=1e-12):
    """迭代求解 Oseen 修正的隐式方程（可与壁面修正叠加）。

    求解:
      η = η_basic / (wall_factor * (1 + 3/16 * Re))
      Re = 2 * r * v_t * ρ_l / η

    参数:
      eta_basic:  Stokes 理想黏度
      wall_factor: 壁面修正因子（>= 1，不启用时 = 1.0）

    返回 (eta_corrected, re_factor, Re, iterations)。
    """
    eta = eta_basic / wall_factor  # 初始猜测（已考虑壁面修正）
    for i in range(max_iter):
        Re = 2.0 * r * v_t * rho_l / eta
        re_factor = 1.0 + 3.0 / 16.0 * Re
        eta_new = eta_basic / (wall_factor * re_factor)
        if abs(eta_new - eta) < tol:
            return eta_new, re_factor, Re, i + 1
        eta = eta_new
    return eta_new, 1.0 + 3.0 / 16.0 * Re, Re, max_iter


def compute_viscosity(
    terminal_velocity_m_s: float,
    ball_radius_m: float,
    ball_density_kg_m3: float,
    liquid_density_kg_m3: float,
    cylinder_radius_m: float = 0.02,
    liquid_height_m: float = 0.35,
    g_m_s2: float = 9.8,
    temperature_c: Optional[float] = None,
    enable_wall_correction: bool = True,
    enable_reynolds_correction: bool = True,
) -> dict:
    """计算液体黏度。

    参数：
        ball_radius_m: 小球半径 (m)
        ball_density_kg_m3: 小球密度 (kg/m³)
        liquid_density_kg_m3: 液体密度 (kg/m³)
        cylinder_radius_m: 量筒内半径 (m)，壁面修正所需
        liquid_height_m: 液柱高度 (m)，壁面修正所需
        g_m_s2: 重力加速度 (m/s²)
        temperature_c: 温度 (°C)，可选
        enable_wall_correction: 是否启用壁面修正，默认 True
        enable_reynolds_correction: 是否启用雷诺数修正，默认 True

    返回 dict：
        eta_basic_pa_s: float        — 理想 Stokes 黏度 (Pa·s)
        wall_correction_factor: float — 壁面修正因子
        enable_wall_correction: bool  — 是否启用壁面修正
        eta_wall_pa_s: float         — 壁面修正后黏度 (Pa·s)
        r_over_R: float              — 球径/筒径比
        r_over_h: float              — 球径/液高比
        R_m: float                   — 量筒内半径 (m)
        h_m: float                   — 液柱高度 (m)
        reynolds_factor: float       — 雷诺数修正因子 (= 1 + 3/16 Re)
        enable_reynolds_correction: bool — 是否启用雷诺数修正
        eta_reynolds_pa_s: float     — 仅雷诺数修正后黏度 (Pa·s，不含壁面修正)
        reynolds_number: float       — 雷诺数
        reynolds_iterations: int     — Oseen 迭代收敛次数
        r_m: float                   — 小球半径 (m)
        eta_final_pa_s: float        — 最终输出黏度（综合修正）
        warnings: list[str]          — 警告信息
    """
    warnings = []

    # ---- 输入验证 ----
    if terminal_velocity_m_s <= 0:
        raise ValueError(
            f"终端速度必须为正数 (当前值: {terminal_velocity_m_s} m/s)。\n"
            f"请检查速度计算或方向约定是否正确。"
        )

    if ball_radius_m <= 0:
        raise ValueError(f"小球半径必须为正数 (当前值: {ball_radius_m} m)")

    if ball_density_kg_m3 <= liquid_density_kg_m3:
        warnings.append(
            f"小球密度 ({ball_density_kg_m3} kg/m³) 不大于液体密度 "
            f"({liquid_density_kg_m3} kg/m³)。\n"
            f"  小球可能无法下沉，请检查参数。"
        )

    if cylinder_radius_m <= 0:
        raise ValueError(f"量筒半径必须为正数 (当前值: {cylinder_radius_m} m)")

    if liquid_height_m <= 0:
        raise ValueError(f"液柱高度必须为正数 (当前值: {liquid_height_m} m)")

    # ---- 基础黏度 ----
    r = ball_radius_m
    rho_s = ball_density_kg_m3
    rho_l = liquid_density_kg_m3
    g = g_m_s2
    v_t = terminal_velocity_m_s

    eta_basic = (2.0 * r**2 * g * (rho_s - rho_l)) / (9.0 * v_t)

    # ---- 壁面修正（Ladenburg） ----
    R = cylinder_radius_m
    h = liquid_height_m

    if enable_wall_correction:
        wall_factor = (1.0 + 2.4 * r / R)
        if wall_factor > 1.20:
            logger.warning(
                "Wall correction factor is large (correction=%.4f > 1.20). "
                "Check cylinder/ball dimensions or eccentric fall.",
                wall_factor,
            )
    else:
        wall_factor = 1.0

    r_over_R = r / R
    r_over_h = r / h
    eta_wall = eta_basic / wall_factor

    if enable_wall_correction:
        logger.info(
            "Wall correction (Ladenburg, divide correction):\n"
            "  r = %.6f m,  R = %.6f m,  h = %.6f m\n"
            "  r/R = %.6f,  r/h = %.6f\n"
            "  wall_factor = (1+2.4*r/R) = %.6f\n"
            "  eta_wall = eta_basic / wall_factor = %.6f Pa·s",
            r, R, h, r_over_R, r_over_h, wall_factor, eta_wall,
        )

    # ---- 雷诺数修正（Oseen，可与壁面修正叠加） ----
    if enable_reynolds_correction:
        eta_combined, re_factor, Re, iterations = _oseen_iteration(
            eta_basic, r, v_t, rho_l, wall_factor=wall_factor,
        )
        # Re-only value (diagnostic: without wall factor)
        eta_reynolds = eta_combined * wall_factor  # = eta_basic / re_factor

        logger.info(
            "Reynolds correction (Oseen, iterative):\n"
            "  Re (corrected) = %.6f\n"
            "  re_factor = 1 + 3/16·Re = %.6f\n"
            "  eta_combined = eta_basic / (%.4f × %.6f) = %.6f Pa·s\n"
            "  iterations = %d",
            Re, re_factor, wall_factor, re_factor, eta_combined, iterations,
        )

        if Re > 1.0:
            warnings.append(
                f"雷诺数 Re = {Re:.2f} > 1，超出 Oseen 修正的有效范围。\n"
                f"  Oseen 修正适用于 Re ≲ 1。\n"
                f"  建议: 使用更小直径的小球或更高黏度的液体。\n"
                f"  当前结果需要谨慎解释。"
            )
        elif Re > 0.1:
            warnings.append(
                f"雷诺数 Re = {Re:.3f}，略高于 Stokes 层流极限。\n"
                f"  已应用 Oseen 修正，结果在工程精度内可用。"
            )
    else:
        re_factor = 1.0
        Re = 2.0 * r * v_t * rho_l / eta_wall
        eta_combined = eta_wall
        eta_reynolds = eta_basic
        iterations = 0

        if Re > 0.1:
            warnings.append(
                f"雷诺数 Re = {Re:.3f}，建议启用雷诺数修正以获得更准确结果。"
            )
        logger.info(
            "Reynolds correction disabled.\n"
            "  eta_final (wall only) = %.6f Pa·s\n"
            "  Re = %.4f",
            eta_combined, Re,
        )

    # ---- 温度信息 ----
    if temperature_c is not None:
        logger.info("Liquid temperature: %.1f C", temperature_c)

    logger.info(
        "Viscosity results:\n"
        "  η_basic  = %.6f Pa·s  (Stokes)\n"
        "  k_wall   = %.6f  (wall corr.)\n"
        "  η_wall   = %.6f Pa·s  (after wall)\n"
        "  k_Re     = %.6f  (Re corr.)\n"
        "  η_final  = %.6f Pa·s  (combined)\n"
        "  Re       = %.6f\n"
        "  iterations = %d",
        eta_basic, wall_factor, eta_wall, re_factor,
        eta_combined, Re, iterations,
    )

    return {
        "eta_basic_pa_s": float(eta_basic),

        # 壁面修正
        "wall_correction_factor": float(wall_factor),
        "enable_wall_correction": enable_wall_correction,
        "eta_wall_pa_s": float(eta_wall),
        "r_over_R": float(r_over_R),
        "r_over_h": float(r_over_h),
        "R_m": float(R),
        "h_m": float(h),

        # 雷诺数修正
        "reynolds_factor": float(re_factor),
        "enable_reynolds_correction": enable_reynolds_correction,
        "eta_reynolds_pa_s": float(eta_reynolds),
        "reynolds_number": float(Re),
        "reynolds_iterations": iterations,

        # 综合
        "eta_final_pa_s": float(eta_combined),

        "r_m": float(r),
        "warnings": warnings,
    }
