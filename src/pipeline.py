"""
主分析管线 — 集成所有模块的新版 process_video

替代 tracking.py:process_video() 的完整分析管线。

流程：
  1. 读取视频 → 构建背景模型
  2. 初始化 ROI 管理器、候选检测器、追踪器
  3. 逐帧处理：ROI 裁剪 → 搜索窗口 → 候选检测 → 评分 → 追踪更新
  4. 后处理：插值、异常剔除
  5. 区间选择 → 速度计算 → 终端速度拟合 → 黏度计算
  6. 输出：CSV、图表、报告

与旧版 tracking.py 的关键区别：
  - 使用 BallTracker 状态机替代直接 BallDetector 调用
  - 使用 CandidateDetector（3 方法）替代原 5 方法
  - 使用 interval_selector（无 0-0 回退）
  - 物理约束内嵌在追踪器中，而非后处理
"""

from __future__ import annotations

import os
import cv2
import numpy as np
import pandas as pd
import logging
from typing import Callable

from src.background_model import build_background_median
from src.candidate_detector import CandidateDetector
from src.ball_tracker import BallTracker, TrackerState
from src.roi_manager import ROIManager
from src.trajectory_filter import (
    interpolate_short_gaps,
    remove_outliers as remove_traj_outliers,
    normalize_y,
    compute_displacement,
)
from src.interval_selector import select_interval
from src.terminal_velocity import find_terminal_velocity
from src.velocity import compute_velocity
from src.viscosity import compute_viscosity
from src.video_io import VideoReader
from src.utils import mm_per_px_to_m_per_px

logger = logging.getLogger(__name__)


def run_pipeline(
    video_path: str,
    config: dict,
    output_dir: str | None = None,
    *,
    debug_callback: Callable | None = None,
    progress_callback: Callable | None = None,
) -> dict:
    """运行完整分析管线。

    Args:
        video_path: 视频文件路径
        config: 配置字典
        output_dir: 输出目录（None=不保存文件）
        debug_callback: 调试日志回调
        progress_callback: 进度回调 (current, total)

    Returns:
        dict: {
            "success": bool,
            "traj_df": 轨迹 DataFrame,
            "velocity_df": 速度 DataFrame,
            "terminal_region": 终端速度区结果,
            "viscosity_result": 黏度计算结果,
            "track_interval": 区间选择结果,
            "stats": 统计信息,
        }
    """
    _log = debug_callback or (lambda msg: logger.info(msg))
    _prog = progress_callback or (lambda c, t: None)

    # ── 1. 打开视频 ──
    _log("[INFO] 打开视频...")
    reader = VideoReader(video_path)
    reader.open()

    fps = reader.fps
    if config.get("manual_fps") is not None:
        fps = float(config["manual_fps"])
        reader.fps = fps

    total_frames = reader.total_frames
    frame_width = reader.width
    frame_height = reader.height

    _log(f"[INFO] 视频: {os.path.basename(video_path)} | "
         f"{frame_width}x{frame_height} | {fps:.2f} fps | {total_frames} 帧")

    # 比例尺
    scale_mm_per_px = config.get("scale_mm_per_px")
    if scale_mm_per_px is None:
        raise ValueError("scale_mm_per_px 未设置")
    scale_m_per_px = mm_per_px_to_m_per_px(scale_mm_per_px)

    # ── 2. 构建背景模型 ──
    _log("[INFO] 构建背景模型...")
    reader.reset()
    bg_frames = []
    for _ in range(min(50, total_frames)):
        ret, frame = reader.read_frame()
        if not ret:
            break
        bg_frames.append(frame)
    background = build_background_median(bg_frames) if bg_frames else None
    _log(f"[INFO] 背景模型: {len(bg_frames)} 帧中值合成")

    # ── 3. 初始化组件 ──
    reader.reset()

    # ROI 管理器
    roi_mgr = ROIManager.from_config(config, frame_width, frame_height)
    if roi_mgr.has_roi:
        _log(f"[INFO] ROI: {roi_mgr.state.display_roi}")
    if roi_mgr.has_init_point:
        _log(f"[INFO] 初始点: {roi_mgr.state.init_point}")

    # 候选检测器。CandidateDetector works in ROI-local coordinates, so the
    # fallback pipeline must crop the background and translate fall_axis_x too.
    detector_config = dict(config)
    detector_background = background
    if roi_mgr.has_roi and background is not None:
        rx, ry, rw, rh = [int(v) for v in roi_mgr.state.detection_roi]
        rx = max(0, rx); ry = max(0, ry)
        rw = min(rw, background.shape[1] - rx)
        rh = min(rh, background.shape[0] - ry)
        if rw > 0 and rh > 0:
            detector_background = background[ry:ry + rh, rx:rx + rw].copy()
            if detector_config.get("fall_axis_x") is not None:
                detector_config["fall_axis_x"] = float(detector_config["fall_axis_x"]) - rx
            _log("[INFO] fallback pipeline background/axis source=roi_local")
    detector = CandidateDetector(detector_config, detector_background)

    # 追踪器
    tracker = BallTracker(config)

    # ── 4. 逐帧处理 ──
    _log("[INFO] 开始逐帧追踪...")

    records = []  # [{frame, x_px, y_px, state, point_type, score, circularity}, ...]
    preview_frames = []  # 标注帧（用于输出标记视频）

    for frame_idx in range(total_frames):
        ret, frame = reader.read_frame()
        if not ret:
            break

        # ROI 裁剪
        if roi_mgr.has_roi:
            rx, ry, rw, rh = [int(v) for v in roi_mgr.state.detection_roi]
            rx = max(0, rx); ry = max(0, ry)
            rw = min(rw, frame.shape[1] - rx)
            rh = min(rh, frame.shape[0] - ry)
            roi_frame = frame[ry:ry + rh, rx:rx + rw].copy()
            roi_offset_x, roi_offset_y = rx, ry
        else:
            roi_frame = frame.copy()
            roi_offset_x, roi_offset_y = 0, 0

        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        # 搜索窗口（从预测位置）
        if tracker.is_active:
            sx, sy, sw, sh = tracker.get_search_window(
                roi_frame.shape[1], roi_frame.shape[0],
                expanded=(tracker.state == TrackerState.REACQUIRE),
            )
            search_rect = (sx, sy, sw, sh)
            predict_pos = tracker.predict_next()
        else:
            search_rect = None
            predict_pos = None

        # 检测
        candidates = detector.detect(
            gray,
            search_rect=search_rect,
            predict_pos=predict_pos,
            frame_idx=frame_idx,
        )

        if not tracker.is_active:
            # 首次检测到 → 启动追踪
            if candidates:
                best = candidates[0]
                tracker.start(best["cx"], best["cy"], frame_idx)
                _log(f"[INFO] 帧 {frame_idx}: 追踪器启动 ({best['cx']:.0f}, {best['cy']:.0f})")
                records.append({
                    "frame": frame_idx,
                    "x_px": best["cx"],
                    "y_px": best["cy"],
                    "state": tracker.state,
                    "point_type": "raw",
                    "confidence": 1.0,
                    "score": best["score"],
                    "circularity": best.get("circularity", 0),
                })
                _draw_detection(frame, best, roi_offset_x, roi_offset_y)
                preview_frames.append(frame.copy())
        else:
            # 追踪模式
            best_candidate = candidates[0] if candidates else None

            if best_candidate is not None:
                # 物理约束检查
                is_ok, reason = tracker.check_physical(
                    best_candidate["cx"], best_candidate["cy"]
                )

                if is_ok:
                    pt = tracker.update(
                        best_candidate["cx"], best_candidate["cy"],
                        frame_idx,
                        score=best_candidate["score"],
                        circularity=best_candidate.get("circularity", 0),
                    )
                    records.append({
                        "frame": frame_idx,
                        "x_px": best_candidate["cx"],
                        "y_px": best_candidate["cy"],
                        "state": pt.state,
                        "point_type": "raw",
                        "confidence": 1.0,
                        "score": best_candidate["score"],
                        "circularity": best_candidate.get("circularity", 0),
                    })
                    _draw_detection(frame, best_candidate, roi_offset_x, roi_offset_y)
                else:
                    # 不满足物理约束 → 预测补位
                    pt = tracker.predict_fallback(frame_idx, reason=reason)
                    records.append({
                        "frame": frame_idx,
                        "x_px": pt.x_px,
                        "y_px": pt.y_px,
                        "state": pt.state,
                        "point_type": "predicted",
                        "confidence": pt.confidence,
                        "score": 0,
                        "circularity": 0,
                    })
                    _draw_prediction(frame, pt, roi_offset_x, roi_offset_y)
            else:
                # 无候选 → 预测补位
                pt = tracker.predict_fallback(frame_idx, reason="无检测候选")
                records.append({
                    "frame": frame_idx,
                    "x_px": pt.x_px,
                    "y_px": pt.y_px,
                    "state": pt.state,
                    "point_type": "predicted",
                    "confidence": pt.confidence,
                    "score": 0,
                    "circularity": 0,
                })
                _draw_prediction(frame, pt, roi_offset_x, roi_offset_y)

            # 检查终止：超出底部
            if roi_mgr.has_roi and tracker.is_active:
                cy = records[-1]["y_px"]
                if tracker.is_beyond_stop_y(cy, roi_mgr.state.detection_roi):
                    tracker.stop("超出底部终止线")
                    _log(f"[INFO] 帧 {frame_idx}: 超出底部终止线，追踪停止")

            _draw_info(frame, frame_idx, tracker)
            preview_frames.append(frame.copy())

        _prog(frame_idx + 1, total_frames)

    _log(f"[INFO] 逐帧处理完成: {len(records)} 帧记录"
         f" ({sum(1 for r in records if r['point_type']=='raw')} raw + "
         f"{sum(1 for r in records if r['point_type']=='predicted')} predicted)")

    # ── 5. 构建 DataFrame ──
    traj_df = _build_traj_df(records, fps, scale_m_per_px, roi_offset_x, roi_offset_y)

    # 更新下落中心线
    raw_records = [r for r in records if r["point_type"] == "raw"]
    if len(raw_records) >= 5:
        axis_x = np.median([r["x_px"] for r in raw_records])
        roi_mgr.update_fall_axis(axis_x)

    # ── 6. 后处理 ──
    _log("[INFO] 后处理: 插值 + 异常剔除 + 归一化...")
    traj_df = interpolate_short_gaps(traj_df, max_gap=3)
    traj_df = remove_traj_outliers(
        traj_df,
        dx_max=float(config.get("dx_max", 20)),
        dy_back_tol=float(config.get("dy_back_tol", 5)),
        dist_max=float(config.get("max_jump_px", 80)),
    )
    traj_df = normalize_y(traj_df)

    # 统计
    stats = compute_displacement(traj_df)
    valid_raw = sum(1 for r in records if r["point_type"] == "raw")
    stats["total_frames_all"] = total_frames
    stats["valid_raw_all"] = valid_raw
    stats["valid_raw_ratio"] = valid_raw / total_frames if total_frames > 0 else 0.0

    _log(f"[INFO] 有效识别: {valid_raw}/{total_frames} ({valid_raw/total_frames*100:.1f}%)")

    # ── 7. 保存 CSV ──
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "trajectory.csv")
        traj_df.to_csv(csv_path, index=False, float_format="%.8f", encoding="utf-8")
        _log(f"[INFO] 轨迹已保存: {csv_path}")

    # ── 8. 区间选择 ──
    _log("[INFO] 选择分析区间...")
    track_interval = select_interval(
        traj_df, config, roi_mgr.state.detection_roi,
        debug_callback=_log,
    )

    if not track_interval.success:
        return {
            "success": False,
            "traj_df": traj_df,
            "velocity_df": None,
            "terminal_region": {
                "found": False,
                "message": f"区间选择失败: {track_interval.termination_reason}",
            },
            "viscosity_result": None,
            "track_interval": track_interval,
            "stats": stats,
            "output_dir": output_dir,
            "preview_frames": preview_frames if config.get("save_marked_video", False) else [],
        }

    _log(f"[INFO] 分析区间: 帧 {track_interval.start_frame}~{track_interval.end_frame}, "
         f"时长 {track_interval.duration_s:.3f}s")

    # 切分
    df_analysis = traj_df.iloc[track_interval.start_frame:track_interval.end_frame + 1].copy()

    # ── 9. 速度 ──
    _log("[INFO] 计算速度...")
    velocity_df = compute_velocity(df_analysis, config)

    if output_dir:
        vel_csv = os.path.join(output_dir, "velocity.csv")
        velocity_df.to_csv(vel_csv, index=False, float_format="%.8f", encoding="utf-8")

    # ── 10. 终端速度 ──
    _log("[INFO] 自动判定终端速度区...")
    terminal_region = find_terminal_velocity(
        df_analysis, velocity_df, config,
        debug_callback=_log,
    )

    if not terminal_region["found"]:
        _log(f"[WARN] {terminal_region['message']}")
    else:
        _log(f"[INFO] 终端速度: v_t = {terminal_region['terminal_velocity_m_s']:.6f} m/s, "
             f"R² = {terminal_region['r2']:.6f}")

    # ── 11. 黏度 ──
    viscosity_result = None
    if terminal_region["found"]:
        _log("[INFO] 计算黏度...")
        try:
            viscosity_result = compute_viscosity(
                terminal_velocity_m_s=terminal_region["terminal_velocity_m_s"],
                ball_radius_m=config["ball_radius_mm"] / 1000.0,
                ball_density_kg_m3=config["ball_density_kg_m3"],
                liquid_density_kg_m3=config["liquid_density_kg_m3"],
                cylinder_radius_m=config["cylinder_radius_mm"] / 1000.0,
                liquid_height_m=config["liquid_height_mm"] / 1000.0,
                g_m_s2=config.get("gravity_m_s2", 9.8),
                temperature_c=config.get("temperature_c"),
                enable_wall_correction=config.get("enable_wall_correction", True),
                enable_reynolds_correction=config.get("enable_reynolds_correction", True),
            )
            viscosity_result["reference_viscosity_pa_s"] = config.get("reference_viscosity_pa_s")
            viscosity_result["temperature_c"] = config.get("temperature_c", "?")

            _log(f"[INFO] 黏度: η_basic={viscosity_result['eta_basic_pa_s']:.6f} Pa·s, "
                 f"η_final={viscosity_result['eta_final_pa_s']:.6f} Pa·s, "
                 f"Re={viscosity_result['reynolds_number']:.4f}")
        except (ValueError, ZeroDivisionError) as e:
            _log(f"[ERROR] 黏度计算失败: {e}")

    # ── 12. 输出 ──
    result = {
        "success": True,
        "traj_df": df_analysis,
        "traj_df_full": traj_df,
        "velocity_df": velocity_df,
        "terminal_region": terminal_region,
        "viscosity_result": viscosity_result,
        "track_interval": track_interval,
        "stats": stats,
        "output_dir": output_dir,
        "preview_frames": preview_frames if config.get("save_marked_video", False) else [],
    }

    _log("[INFO] 分析完成")
    return result


# ── 辅助函数 ──

def _build_traj_df(
    records: list[dict],
    fps: float,
    scale_m_per_px: float,
    roi_offset_x: int,
    roi_offset_y: int,
) -> pd.DataFrame:
    """从追踪记录构建 DataFrame。"""
    n = len(records)
    if n == 0:
        return pd.DataFrame(columns=[
            "frame", "time_s", "x_px", "y_px", "y_m",
            "valid", "point_type", "circularity",
        ])

    frames = np.array([r["frame"] for r in records], dtype=int)
    x_px = np.array([r["x_px"] for r in records], dtype=float)
    y_px = np.array([r["y_px"] for r in records], dtype=float)
    point_types = np.array([r["point_type"] for r in records])
    circularity = np.array([r.get("circularity", 0) for r in records], dtype=float)

    # 全局坐标
    x_global = x_px + roi_offset_x
    y_global = y_px + roi_offset_y

    time_s = frames / fps
    y_m = y_global * scale_m_per_px

    # —— 归一化 y_m 为相对于首个 raw 检测帧的物理位移 ——
    y0_px = None
    for i, pt in enumerate(point_types):
        if pt == "raw":
            y0_px = y_global[i]
            break
    if y0_px is not None:
        y_m = (y_global - y0_px) * scale_m_per_px

    valid = np.array([r["point_type"] in ("raw", "predicted", "interpolated")
                      for r in records])

    return pd.DataFrame({
        "frame": frames,
        "time_s": time_s,
        "x_px": x_global,
        "y_px": y_global,
        "y_m": y_m,
        "valid": valid,
        "point_type": point_types,
        "circularity": circularity,
    })


def _draw_detection(frame, candidate: dict, ox: int, oy: int):
    """在帧上画检测标注。"""
    cx = int(candidate["cx"]) + ox
    cy = int(candidate["cy"]) + oy
    r = max(1, int(candidate.get("radius", 5)))
    cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)


def _draw_prediction(frame, pt, ox: int, oy: int):
    """在帧上画预测标注（橙色）。"""
    cx = int(pt.x_px) + ox
    cy = int(pt.y_px) + oy
    cv2.circle(frame, (cx, cy), 3, (0, 165, 255), -1)
    cv2.circle(frame, (cx, cy), 5, (0, 165, 255), 1)


def _draw_info(frame, frame_idx: int, tracker: BallTracker):
    """在帧上叠加追踪状态信息。"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (2, 2), (200, 50), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    cv2.putText(frame, f"Frame: {frame_idx} State: {tracker.state}",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, f"Lost: {tracker.lost_count} Raw: {tracker.raw_count}",
                (6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
