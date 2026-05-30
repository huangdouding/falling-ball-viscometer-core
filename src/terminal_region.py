"""
终端速度区自动判定模块（v2 — 基于窗口 y-t 拟合速度）

v2 核心改进：
  1. Cv 由相邻滑动窗口的 y-t 拟合斜率序列计算，而非逐帧差分速度
  2. 不再依赖 velocity_df 进行判定（velocity_df 保留参数仅用于兼容）
  3. 增加兜底逻辑：当 y-t 线性度高但窗口速度有波动时仍可接受
  4. 输出 window_velocities 用于 v-t 图显示窗口拟合速度

统一字段约定（全模块一致）：
  - _fit_y_t / _fit_y_t_range 返回 {"v_t": slope, "intercept": ..., "r2": ...}
  - 候选字典包含 "v_t" 和 "terminal_velocity_m_s"（二者值相同）
  - 最终输出字典同时含 "v_t" 和 "terminal_velocity_m_s"
  - success/found 双字段（found 保持向后兼容）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def find_terminal_region(
    traj_df: pd.DataFrame,
    velocity_df: pd.DataFrame,
    config: dict,
) -> dict:
    """自动寻找终端速度区（v2 算法）。

    参数：
        traj_df:     轨迹 DataFrame（需含 time_s, y_m, x_px, valid, point_type）
        velocity_df: 速度 DataFrame（仅用于兼容，不作判定依据）
        config:      配置字典

    返回 dict（关键字段）:
        success: bool               — 是否成功
        found: bool                 — 同上，向后兼容
        v_t: float | None           — 终端速度（y-t 拟合斜率）
        terminal_velocity_m_s: float | None  — 同上，GUI/黏度用
        start_time_s / end_time_s   — 终端区间
        start_frame / end_frame     — 起止帧号
        r2 / cv                     — 拟合质量
        method: str                 — 判定方式
        window_velocities: list     — 窗口拟合速度（用于 v-t 图）
        candidates_table: list      — 候选窗口表
        message: str                — 说明信息
    """
    # ---- 配置参数 ----
    window_sec = config.get("terminal_window_sec", 0.8)
    cv_threshold = config.get("cv_threshold", 0.08)
    r2_threshold = config.get("r2_threshold", 0.990)
    min_group_windows = config.get("terminal_min_group_windows", 5)
    min_duration = config.get("terminal_min_duration", 0.5)
    ignore_start_s = config.get("terminal_ignore_start_sec", 0.3)
    ignore_end_s = config.get("terminal_ignore_end_sec", 0.5)
    min_real_rate = float(config.get("terminal_min_real_detection_rate", 0.80))

    # ---- 手动区间优先 ----
    manual_start = int(config.get("manual_start_frame", 0))
    manual_end = int(config.get("manual_end_frame", 0))
    manual_override = (manual_start > 0 or manual_end > 0)

    if manual_override:
        n_total = len(traj_df)
        if manual_start <= 0:
            manual_start = 0
        if manual_end <= 0:
            manual_end = n_total - 1
        manual_start = max(0, min(manual_start, n_total - 1))
        manual_end = max(0, min(manual_end, n_total - 1))
        if manual_start >= manual_end:
            return _not_found(
                f"手动区间无效: start_frame={manual_start} >= end_frame={manual_end}"
            )

        # 限制 valid_idx 到手动区间
        frame_arr = traj_df["frame"].values
        in_manual = (frame_arr[valid_idx] >= manual_start) & (frame_arr[valid_idx] <= manual_end)
        valid_idx = valid_idx[in_manual]
        if len(valid_idx) < 5:
            return _not_found(
                f"手动区间 {manual_start}-{manual_end} 内有效点不足 ({len(valid_idx)} 个)。"
            )

    # ---- 有效数据准备 ----
    valid_mask = traj_df["valid"].values.copy()
    if "point_type" in traj_df.columns:
        valid_mask &= traj_df["point_type"].isin(["raw", "interpolated"]).values
    t = traj_df["time_s"].values
    y = traj_df["y_m"].values
    x = traj_df["x_px"].values if "x_px" in traj_df.columns else None

    if not manual_override:
        valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 5:
        return _not_found(f"有效数据点太少 ({len(valid_idx)} 个)，至少需要 5 个。")

    # 忽略轨迹首尾段（加速段和底部碰撞段）
    t_min = t[valid_idx[0]] + ignore_start_s
    t_max = t[valid_idx[-1]] - ignore_end_s
    valid_idx = valid_idx[(t[valid_idx] >= t_min) & (t[valid_idx] <= t_max)]
    if len(valid_idx) < 5:
        return _not_found(
            f"排除起止段 ({ignore_start_s}s/{ignore_end_s}s) 后有效点不足 "
            f"({len(valid_idx)} 个)。"
        )

    fps = _estimate_fps(t)
    window_frames = max(3, int(round(window_sec * fps)))
    if window_frames > len(valid_idx):
        return _not_found(
            f"滑动窗口大小 ({window_frames} 帧 ≈ {window_sec:.2f}s) "
            f"超过有效点数 ({len(valid_idx)})。"
        )

    # ================================================================
    # Phase 1: 对每个滑动窗口做 y-t 线性拟合 → v_t, R²
    # ================================================================
    window_data = []
    for i in range(len(valid_idx) - window_frames + 1):
        idx_win = valid_idx[i:i + window_frames]
        fit = _fit_y_t(t[idx_win], y[idx_win])
        if fit is None:
            continue
        fit["pos"] = i
        fit["start_idx"] = idx_win[0]
        fit["end_idx"] = idx_win[-1]
        window_data.append(fit)

    if len(window_data) < min_group_windows:
        return _not_found(
            f"滑动窗口有效拟合数 ({len(window_data)}) 不足 "
            f"({min_group_windows} 个)。"
        )

    # 收集所有窗口速度（用于绘图展示）
    window_velocities = []
    for w in window_data:
        window_velocities.append({
            "time": (t[w["start_idx"]] + t[w["end_idx"]]) / 2.0,
            "v": w["v_t"],
            "r2": w["r2"],
            "start_time": t[w["start_idx"]],
            "end_time": t[w["end_idx"]],
        })

    # ================================================================
    # Phase 2: 连续 R² 合格的窗口组成群组
    # ================================================================
    good_mask = np.array([w["r2"] >= r2_threshold for w in window_data])

    groups = []
    lo = 0
    while lo < len(window_data):
        if good_mask[lo]:
            hi = lo
            while hi < len(window_data) and good_mask[hi]:
                hi += 1
            count = hi - lo
            if count >= min_group_windows:
                groups.append(window_data[lo:hi])
            lo = hi
        else:
            lo += 1

    if not groups:
        logger.info("无连续 R² 合格窗口群组，尝试兜底搜索...")
        return _fallback_search(
            traj_df, valid_idx, t, y, x,
            window_data, window_velocities,
            config, r2_threshold, min_duration,
            manual_override=manual_override,
        )

    # ================================================================
    # Phase 3: 对每个群组用组内 v_t 计算 Cv
    # ================================================================
    candidates = []
    for group in groups:
        v_arr = np.array([w["v_t"] for w in group])
        v_mn = np.mean(v_arr)
        if v_mn <= 0:
            continue
        v_sd = np.std(v_arr, ddof=1)
        cv = v_sd / v_mn
        duration = group[-1]["end_time"] - group[0]["start_time"]
        quality = _region_quality_stats(
            traj_df, group[0]["start_idx"], group[-1]["end_idx"], v_arr
        )

        if cv > cv_threshold:
            continue
        if duration < min_duration:
            continue
        if quality["real_detection_rate"] < min_real_rate:
            continue

        # 在群组全覆盖区间上做一次 y-t 拟合，得最终 v_t
        region_result = _fit_y_t_range(
            t, y,
            start_idx=group[0]["start_idx"],
            end_idx=group[-1]["end_idx"],
        )
        if region_result is None:
            continue

        v_t = region_result["v_t"]

        candidates.append({
            "start_idx": group[0]["start_idx"],
            "end_idx": group[-1]["end_idx"],
            "start_time": t[group[0]["start_idx"]],
            "end_time": t[group[-1]["end_idx"]],
            "n_points": region_result["n"],
            "duration": duration,
            "v_t": v_t,
            "terminal_velocity_m_s": v_t,
            "r2": region_result["r2"],
            "cv": cv,
            "velocity_std": float(v_sd),
            "fit_intercept": region_result["intercept"],
            "n_windows": len(group),
            **quality,
        })

    # ================================================================
    # Phase 4: 选优 / 兜底
    # ================================================================
    if candidates:
        best = _select_best_region(candidates, valid_idx, len(t))
        candidates_table = _build_candidates_table(candidates, best)
        warnings = []
        if best.get("real_detection_rate", 0.0) < min_real_rate:
            warnings.append(
                f"terminal real detection rate {best.get('real_detection_rate', 0.0):.1%} "
                f"is below {min_real_rate:.0%}"
            )
        if best.get("predicted_count", 0) > 0:
            warnings.append(
                f"terminal interval contains {best.get('predicted_count', 0)} predicted points; "
                "they were not used for fitting"
            )

        start_frame = int(traj_df.iloc[best["start_idx"]]["frame"])
        end_frame = int(traj_df.iloc[best["end_idx"]]["frame"])

        logger.info(
            "终端速度区: %.3f-%.3f s (%d-%d) | "
            "v_t = %.6f m/s | R² = %.6f | Cv = %.4f | 候选组: %d",
            best["start_time"], best["end_time"],
            start_frame, end_frame,
            best["terminal_velocity_m_s"], best["r2"], best["cv"],
            len(candidates),
        )

        return {
            "success": True,
            "found": True,
            "v_t": best["terminal_velocity_m_s"],
            "terminal_velocity_m_s": best["terminal_velocity_m_s"],
            "start_time_s": best["start_time"],
            "end_time_s": best["end_time"],
            "start_frame": start_frame,
            "end_frame": end_frame,
            "r2": best["r2"],
            "cv": best["cv"],
            "velocity_std": best.get("velocity_std"),
            "point_count": best.get("point_count"),
            "real_detection_count": best.get("real_detection_count"),
            "predicted_count": best.get("predicted_count"),
            "effective_rate": best.get("effective_rate"),
            "real_detection_rate": best.get("real_detection_rate"),
            "selection_reason": (
                "highest score among sliding windows passing R2/CV/detection-rate checks"
            ),
            "warning_list": warnings,
            "method": "sliding_window_linear_fit",
            "interval_source": "manual_override" if manual_override else "auto_track",
            "fit_slope": best["terminal_velocity_m_s"],
            "fit_intercept": best["fit_intercept"],
            "window_velocities": window_velocities,
            "candidates_table": candidates_table,
            "message": (
                f"找到终端速度区: {best['start_time']:.3f}s - {best['end_time']:.3f}s, "
                f"v_t = {best['terminal_velocity_m_s']:.6f} m/s, "
                f"R² = {best['r2']:.6f}, Cv = {best['cv']:.4f}"
            ),
        }

    # ---- 兜底 ----
    logger.info("Cv 判据未通过，尝试兜底搜索...")
    return _fallback_search(
        traj_df, valid_idx, t, y, x,
        window_data, window_velocities,
        config, r2_threshold, min_duration,
        manual_override=manual_override,
    )


# ================================================================
#  拟合工具
# ================================================================

def _region_quality_stats(
    traj_df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    velocities: np.ndarray | None = None,
) -> dict:
    """Count real/predicted/interpolated points in a candidate terminal region."""
    region = traj_df.iloc[start_idx:end_idx + 1]
    total = len(region)
    if total <= 0:
        return {
            "point_count": 0,
            "real_detection_count": 0,
            "predicted_count": 0,
            "interpolated_count": 0,
            "effective_rate": 0.0,
            "real_detection_rate": 0.0,
            "velocity_std": None,
        }

    if "point_type" in region.columns:
        pt = region["point_type"]
        real = int((pt == "raw").sum())
        predicted = int((pt == "predicted").sum())
        interpolated = int((pt == "interpolated").sum())
    else:
        real = int(region["valid"].sum()) if "valid" in region.columns else total
        predicted = 0
        interpolated = 0

    effective = real + interpolated
    velocity_std = None
    if velocities is not None and len(velocities) > 1:
        velocity_std = float(np.std(velocities, ddof=1))

    return {
        "point_count": int(total),
        "real_detection_count": real,
        "predicted_count": predicted,
        "interpolated_count": interpolated,
        "effective_rate": effective / total,
        "real_detection_rate": real / total,
        "velocity_std": velocity_std,
    }

def _fit_y_t(t_arr: np.ndarray, y_arr: np.ndarray) -> dict | None:
    """对一段 (t, y) 做线性拟合。

    Returns:
        {"v_t": slope, "intercept": ..., "r2": ..., "residual_std": ...,
         "n": ..., "start_time": ..., "end_time": ...}
        or None 拟合失败
    """
    finite = np.isfinite(t_arr) & np.isfinite(y_arr)
    if finite.sum() < 3:
        return None
    t_f = t_arr[finite]
    y_f = y_arr[finite]

    A = np.vstack([t_f, np.ones_like(t_f)]).T
    try:
        slope, intercept = np.linalg.lstsq(A, y_f, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    if slope <= 0:
        return None

    y_pred = slope * t_f + intercept
    ss_res = np.sum((y_f - y_pred) ** 2)
    ss_tot = np.sum((y_f - np.mean(y_f)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    residual_std = np.std(y_f - y_pred, ddof=1) if len(y_f) > 2 else 0.0

    return {
        "v_t": slope,
        "intercept": intercept,
        "r2": r2,
        "residual_std": residual_std,
        "n": len(t_f),
        "start_time": t_f[0],
        "end_time": t_f[-1],
    }


def _fit_y_t_range(
    t_full: np.ndarray,
    y_full: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> dict | None:
    """对 t_full[start_idx:end_idx+1] 区间做 y-t 拟合。

    Returns: 同 _fit_y_t（包含 "v_t" 字段）
    """
    idx = np.arange(start_idx, end_idx + 1)
    return _fit_y_t(t_full[idx], y_full[idx])


# ================================================================
#  兜底搜索（宽松判据，忽略 Cv）
# ================================================================

def _fallback_search(
    traj_df: pd.DataFrame,
    valid_idx: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    x: np.ndarray | None,
    window_data: list,
    window_velocities: list,
    config: dict,
    r2_threshold: float,
    min_duration: float,
    manual_override: bool = False,
) -> dict:
    """兜底：在有效轨迹中寻找 y-t 线性度好的连续段。

    判据（比主流程宽松）：
      - y-t 拟合 R² >= r2_threshold（与主流程一致，重点在线性度）
      - x 坐标波动小（标准差 < 阈值）
      - y 单调下落（回跳帧占比 < 20%）
      - 段长度 >= min_duration
    """
    logger.info("执行兜底搜索: R² >= %.4f, min_duration=%.2fs", r2_threshold, min_duration)

    fps = _estimate_fps(t)
    min_frames = max(3, int(round(min_duration * fps)))

    # 取 x 通道半宽作为 x 稳定性判据
    x_gate_px = config.get("search_win_w", 60) * 0.25
    x_std_max = x_gate_px * 0.5  # x 标准差容忍上限
    min_real_rate = float(config.get("terminal_min_real_detection_rate", 0.80))

    # 在 valid_idx 中找连续段
    segments = _split_consecutive(valid_idx)

    candidates = []
    for seg in segments:
        t_seg = t[seg]
        if t_seg[-1] - t_seg[0] < min_duration:
            continue
        if len(seg) < min_frames:
            continue

        fit = _fit_y_t(t_seg, y[seg])
        if fit is None:
            continue
        if fit["r2"] < r2_threshold * 0.995:  # 稍微放宽
            continue

        # x 稳定性检查
        if x is not None:
            x_seg = x[seg]
            x_finite = x_seg[np.isfinite(x_seg)]
            if len(x_finite) > 5 and np.std(x_finite, ddof=1) > x_std_max:
                continue

        # y 单调性检查
        y_seg = y[seg]
        finite_mask = np.isfinite(y_seg)
        y_clean = y_seg[finite_mask]
        if len(y_clean) >= 5:
            dy = np.diff(y_clean)
            backward_ratio = np.sum(dy <= 0) / len(dy)
            if backward_ratio > 0.2:
                continue

        # 通过
        v_t = fit["v_t"]
        intercept = fit["intercept"]
        duration = t_seg[-1] - t_seg[0]
        quality = _region_quality_stats(traj_df, seg[0], seg[-1], None)
        if quality["real_detection_rate"] < min_real_rate:
            continue

        candidates.append({
            "start_idx": seg[0],
            "end_idx": seg[-1],
            "start_time": t_seg[0],
            "end_time": t_seg[-1],
            "n_points": fit["n"],
            "duration": duration,
            "v_t": v_t,
            "terminal_velocity_m_s": v_t,
            "r2": fit["r2"],
            "cv": fit.get("residual_std", 0.0) / v_t if v_t > 0 else 1.0,
            "fit_intercept": intercept,
            "n_windows": 0,
            "_fallback": True,
            **quality,
        })

    if candidates:
        best = _select_best_region(candidates, valid_idx, len(t))
        start_frame = int(traj_df.iloc[best["start_idx"]]["frame"])
        end_frame = int(traj_df.iloc[best["end_idx"]]["frame"])

        logger.info(
            "[兜底] 终端速度区: %.3f-%.3f s (%d-%d) | "
            "v_t = %.6f m/s | R² = %.6f",
            best["start_time"], best["end_time"],
            start_frame, end_frame,
            best["terminal_velocity_m_s"], best["r2"],
        )

        candidates_table = _build_candidates_table(candidates, best)
        warnings = []
        if best.get("predicted_count", 0) > 0:
            warnings.append(
                f"terminal interval contains {best.get('predicted_count', 0)} predicted points; "
                "they were not used for fitting"
            )
        return {
            "success": True,
            "found": True,
            "v_t": best["terminal_velocity_m_s"],
            "terminal_velocity_m_s": best["terminal_velocity_m_s"],
            "start_time_s": best["start_time"],
            "end_time_s": best["end_time"],
            "start_frame": start_frame,
            "end_frame": end_frame,
            "r2": best["r2"],
            "cv": best["cv"],
            "velocity_std": best.get("velocity_std"),
            "point_count": best.get("point_count"),
            "real_detection_count": best.get("real_detection_count"),
            "predicted_count": best.get("predicted_count"),
            "effective_rate": best.get("effective_rate"),
            "real_detection_rate": best.get("real_detection_rate"),
            "selection_reason": "fallback region passing relaxed linearity and detection-rate checks",
            "warning_list": warnings,
            "method": "fallback_linear_search",
            "interval_source": "manual_override" if manual_override else "auto_track",
            "fit_slope": best["terminal_velocity_m_s"],
            "fit_intercept": best["fit_intercept"],
            "window_velocities": window_velocities,
            "candidates_table": candidates_table,
            "message": (
                f"[兜底] 找到终端速度区: {best['start_time']:.3f}s - {best['end_time']:.3f}s, "
                f"v_t = {best['terminal_velocity_m_s']:.6f} m/s, "
                f"R² = {best['r2']:.6f}"
            ),
        }

    # 兜底也失败 → best_effort: 找有效轨迹中最长段，即使 R² 不达标也输出 + 警告
    if len(valid_idx) >= 5:
        t_valid = t[valid_idx]
        y_valid = y[valid_idx]
        fit = _fit_y_t(t_valid, y_valid)
        if fit is not None and fit["v_t"] > 0:
            duration = t_valid[-1] - t_valid[0]
            quality = _region_quality_stats(traj_df, valid_idx[0], valid_idx[-1], None)
            start_frame = int(traj_df.iloc[valid_idx[0]]["frame"])
            end_frame = int(traj_df.iloc[valid_idx[-1]]["frame"])
            best_effort_warnings = [
                f"R²={fit['r2']:.4f} 未达到阈值 {r2_threshold}，结果仅供参考",
                "请检查轨迹质量或手动指定分析区间",
            ]
            return {
                "success": True,
                "found": True,
                "v_t": fit["v_t"],
                "terminal_velocity_m_s": fit["v_t"],
                "start_time_s": t_valid[0],
                "end_time_s": t_valid[-1],
                "start_frame": start_frame,
                "end_frame": end_frame,
                "r2": fit["r2"],
                "cv": fit.get("residual_std", 0.0) / fit["v_t"] if fit["v_t"] > 0 else 1.0,
                "velocity_std": None,
                "point_count": quality.get("point_count", 0),
                "real_detection_count": quality.get("real_detection_count", 0),
                "predicted_count": quality.get("predicted_count", 0),
                "effective_rate": quality.get("effective_rate", 0.0),
                "real_detection_rate": quality.get("real_detection_rate", 0.0),
                "selection_reason": "best_effort（质量未达标，仅供参考）",
                "warning_list": best_effort_warnings,
                "method": "best_effort_low_quality",
                "interval_source": "manual_override" if manual_override else "auto_track",
                "fit_slope": fit["v_t"],
                "fit_intercept": fit["intercept"],
                "window_velocities": window_velocities,
                "candidates_table": [],
                "message": (
                    f"[低质量] 最佳可用段: {t_valid[0]:.3f}s - {t_valid[-1]:.3f}s, "
                    f"v_t = {fit['v_t']:.6f} m/s, "
                    f"R² = {fit['r2']:.4f} (未达标 {r2_threshold})"
                ),
            }

    # 彻底失败
    return _not_found(
        f"所有搜索方式均未找到终端速度区。\n"
        f"  请检查轨迹质量：需有一段连续 {min_duration:.1f}s 以上、\n"
        f"  y-t 接近直线 (R² ≥ {r2_threshold}) 且 x 稳定的段。"
    )


# ================================================================
#  工具函数
# ================================================================

def _split_consecutive(idx: np.ndarray) -> list:
    """将 index 数组按连续步长分组。"""
    if len(idx) == 0:
        return []
    groups = []
    start = 0
    for i in range(1, len(idx)):
        if idx[i] - idx[i - 1] != 1:
            groups.append(idx[start:i])
            start = i
    groups.append(idx[start:])
    return [g for g in groups if len(g) >= 2]


def _estimate_fps(t: np.ndarray) -> float:
    """从时间数组估计帧率。"""
    dt = np.diff(t)
    dt_valid = dt[dt > 0]
    if len(dt_valid) == 0:
        return 30.0
    return 1.0 / np.median(dt_valid)


def _select_best_region(
    candidates: list,
    valid_idx: np.ndarray,
    total_frames: int,
) -> dict:
    """多规则综合评分选择最佳终端速度区。"""
    total_duration = (
        valid_idx[-1] - valid_idx[0] if len(valid_idx) > 1 else 1
    )

    scored = []
    for c in candidates:
        score = 0.0

        # 点数评分 (0~20)
        n_points = c.get("n_points", 1)
        n_ratio = n_points / max(total_duration, 1)
        score += min(n_ratio * 20, 20.0)

        # 持续时间评分 (0~20)
        dur = c.get("duration", 0.0)
        dur_ratio = dur / max(total_duration, 1e-6)
        score += min(dur_ratio * 20, 20.0)

        # 位置评分 (0~25)
        mid_pos = (c["start_idx"] + c["end_idx"]) / 2.0
        total_mid = (valid_idx[0] + valid_idx[-1]) / 2.0
        pos_score = max(0, 25.0 - abs(mid_pos - total_mid) / max(total_frames, 1) * 50)
        score += pos_score

        # R² 评分 (0~20)
        score += min(c.get("r2", 0) * 20, 20.0)

        # Cv 评分 (0~15)
        cv = c.get("cv", 1.0)
        cv_score = max(0, 15.0 - cv / 0.05 * 15)
        score += cv_score

        c["_score"] = score
        scored.append(c)

    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[0]


def _build_candidates_table(candidates: list, best: dict) -> list[dict]:
    """构建候选窗口表（用于 CSV 导出 / 报告展示）。"""
    table = []
    for c in candidates:
        passed = (c["start_idx"] == best["start_idx"] and
                  c["end_idx"] == best["end_idx"])
        v = c.get("v_t", c.get("terminal_velocity_m_s", None))
        table.append({
            "start_time_s": round(c["start_time"], 4),
            "end_time_s": round(c["end_time"], 4),
            "n_points": c.get("n_points", c.get("n_windows", 0)),
            "duration_s": round(c.get("duration", 0), 4),
            "slope_m_s": round(v, 6) if v is not None else None,
            "r2": round(c.get("r2", 0), 6),
            "cv": round(c.get("cv", 0), 4),
            "velocity_std": round(c.get("velocity_std", 0), 6)
                if c.get("velocity_std") is not None else None,
            "real_detection_count": c.get("real_detection_count"),
            "predicted_count": c.get("predicted_count"),
            "real_detection_rate": round(c.get("real_detection_rate", 0), 4),
            "effective_rate": round(c.get("effective_rate", 0), 4),
            "score": round(c.get("_score", 0), 1),
            "passed": passed,
            "method": "fallback" if c.get("_fallback") else "cv_group",
        })
    return table


def _not_found(message: str) -> dict:
    """返回未找到的完整失败结果（包含所有必要字段）。"""
    return {
        "success": False,
        "found": False,
        "v_t": None,
        "terminal_velocity_m_s": None,
        "start_time_s": None,
        "end_time_s": None,
        "start_frame": None,
        "end_frame": None,
        "r2": None,
        "cv": None,
        "velocity_std": None,
        "point_count": 0,
        "real_detection_count": 0,
        "predicted_count": 0,
        "effective_rate": 0.0,
        "real_detection_rate": 0.0,
        "selection_reason": "",
        "warning_list": [],
        "method": "sliding_window_linear_fit",
        "fit_slope": None,
        "fit_intercept": None,
        "window_velocities": [],
        "candidates_table": [],
        "message": message,
    }
