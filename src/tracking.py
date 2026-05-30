"""
视频追踪模块（重构版）

负责：
- 逐帧处理视频，调用 BallDetector 检测小球
- 记录轨迹 DataFrame（含像素坐标 + 实际距离）
- 帧间连续性追踪（previous_center 反馈）
- 丢帧计数 & 恢复机制
- 异常跳变检测（max_jump_px + y 方向单调性）
- 自动估计下落中心线（fall_axis_x）用于排除刻度线伪目标
- 保存标注视频
- 统计有效识别率、连续轨迹长度
"""

import os
import cv2
import numpy as np
import pandas as pd
import logging

from src.ball_detector import BallDetector
from src.video_io import VideoReader, VideoWriterWrapper
from src.utils import mm_per_px_to_m_per_px

logger = logging.getLogger(__name__)


def _build_background(reader: VideoReader, n_frames: int = 50,
                      roi: list = None) -> np.ndarray:
    """读取前 n_frames 帧，用中值法构建背景模型。

    使用更多帧（50）让运动中的小球更好地被中值滤除。
    roi: 可选 [x, y, w, h]，构建 ROI 裁剪后的背景。
    """
    frames = []
    for _ in range(n_frames):
        ret, frame = reader.read_frame()
        if not ret:
            break
        if roi is not None:
            x, y, w, h = [int(v) for v in roi]
            frame = frame[y:y + h, x:x + w]
        frames.append(frame)
    if not frames:
        return None
    # 中值合成
    bg = np.median(np.array(frames), axis=0).astype(np.uint8)
    return bg


def _estimate_fall_axis_x(traj_df: pd.DataFrame, roi, frame_width: int) -> float:
    """从有效轨迹点估计下落中心线（全局 X 坐标）。

    使用有效点的 X 中位数作为 fall_axis_x 估计值。
    """
    valid = traj_df[traj_df["valid"]]
    if len(valid) < 5:
        # 默认 ROI 中心
        if roi is not None:
            return roi[0] + roi[2] / 2.0
        return frame_width / 2.0
    return float(valid["x_px"].median())


def _compute_dynamic_detection_params(config: dict):
    """根据 ball_radius_mm 和 scale_mm_per_px 动态计算识别参数。

    只有当 auto_size_params=True 时才覆盖用户参数。
    auto_size_params 默认为 False，此时完全使用用户界面设置的参数。

    当 auto_size_params=True 时覆盖：
      min_area_px, max_area_px, expected_radius_px_min, expected_radius_px_max, min_circularity
    """
    ball_radius_mm = config.get("ball_radius_mm")
    scale_mm_per_px = config.get("scale_mm_per_px")
    if not ball_radius_mm or not scale_mm_per_px or ball_radius_mm <= 0 or scale_mm_per_px <= 0:
        config["_param_source"] = "manual（数据不足以自动计算）"
        return

    expected_radius_px = ball_radius_mm / scale_mm_per_px
    expected_area_px = np.pi * expected_radius_px ** 2
    auto_size_params = config.get("auto_size_params", True)
    manual_min = float(config.get("expected_radius_px_min", 1.0))
    manual_max = float(config.get("expected_radius_px_max", 4.0))
    stale_radius_range = expected_radius_px < manual_min * 0.7 or expected_radius_px > manual_max * 1.3

    if not auto_size_params and not stale_radius_range:
        config["_expected_radius_px"] = expected_radius_px
        config["_expected_area_px"] = expected_area_px
        config["_param_source"] = "manual（用户界面设置）"
        return

    # The detector must follow the physical ball size selected in the UI.
    # Old saved values such as radius 1-4 px are treated as stale once the
    # physical radius/scale imply a different image size.
    #
    # The ranges are intentionally not huge: too wide a radius gate lets scale
    # marks, cap shadows, and merged threshold blobs beat the real ball.
    config["min_area_px"] = max(2, int(0.35 * expected_area_px))
    config["max_area_px"] = max(config["min_area_px"] + 5, int(3.2 * expected_area_px))
    config["expected_radius_px_min"] = max(1.0, 0.50 * expected_radius_px)
    config["expected_radius_px_max"] = max(
        config["expected_radius_px_min"] + 1.0,
        1.65 * expected_radius_px,
    )
    if config.get("min_circularity", 0.5) > 0.4:
        config["min_circularity"] = 0.35

    config["_expected_radius_px"] = expected_radius_px
    config["_expected_area_px"] = expected_area_px
    if stale_radius_range and not auto_size_params:
        config["_param_source"] = "auto（手动半径范围与球径/比例尺明显不匹配，已保护性重算）"
    else:
        config["_param_source"] = "auto（从 ball_radius_mm / scale_mm_per_px 计算）"


def _compute_dynamic_tracking_params(config: dict):
    """根据球径动态计算追踪参数（搜索窗口、运动约束等）。

    large_ball_mode 自动启用条件: ball_radius_mm >= 1.0 或 config 显式要求。
    在 large_ball_mode 下，搜索窗口、跳变容限等明显放宽。
    这些参数只有在用户没有手动设置时才会覆盖。
    如果 auto_size_params=False，radius/area 等识别参数不受影响。
    """
    ball_radius_mm = config.get("ball_radius_mm")
    scale_mm_per_px = config.get("scale_mm_per_px")
    if not ball_radius_mm or not scale_mm_per_px or ball_radius_mm <= 0 or scale_mm_per_px <= 0:
        return

    expected_radius_px = ball_radius_mm / scale_mm_per_px
    large_ball_mode = config.get("large_ball_mode")
    if large_ball_mode is None:
        large_ball_mode = bool(ball_radius_mm >= 1.0 or expected_radius_px >= 5.0)
    config["large_ball_mode"] = large_ball_mode

    if large_ball_mode:
        config["search_win_w"] = max(70, int(expected_radius_px * 10))
        config["search_window_radius"] = config.get("search_win_w", max(70, int(expected_radius_px * 10)))
        config["search_win_y_up"] = max(25, int(expected_radius_px * 4))
        config["search_win_y_down"] = max(120, int(expected_radius_px * 20))
        config["dist_max"] = max(55, int(expected_radius_px * 8))
        config["dx_max"] = max(25, int(expected_radius_px * 3.5))
        config["dy_back_tol"] = max(6, int(expected_radius_px * 1.0))
    else:
        config["search_win_w"] = max(45, int(expected_radius_px * 8))
        config["search_window_radius"] = config.get("search_win_w", max(45, int(expected_radius_px * 8)))
        config["search_win_y_up"] = max(15, int(expected_radius_px * 3))
        config["search_win_y_down"] = max(80, int(expected_radius_px * 14))
        config["dist_max"] = max(30, int(expected_radius_px * 6))
        config["dx_max"] = max(12, int(expected_radius_px * 2.5))
        config["dy_back_tol"] = max(3, int(expected_radius_px * 0.5))

    logger.info(
        "追踪参数 (large_ball_mode=%s):\n"
        "  expected_radius_px = %.1f\n"
        "  search_win         = ±%d x (+%d, -%d)\n"
        "  dist_max           = %d px\n"
        "  dx_max             = %d px\n"
        "  dy_back_tol        = %d px",
        large_ball_mode, expected_radius_px,
        config["search_win_w"], config["search_win_y_up"], config["search_win_y_down"],
        config["dist_max"], config["dx_max"], config["dy_back_tol"],
    )


def process_video(video_path: str, config: dict, *, frame_callback=None) -> dict:
    """处理视频，逐帧追踪小球，返回轨迹 DataFrame 和统计量。

    追踪策略（v5 预测容错版）：
      1. 基于速度和预测设置非对称搜索窗口 (x ± 60, y: -20 ~ +120)
      2. 连续丢失帧时用速度预测补位（up to 10 帧）
      3. 丢失超过 10 帧才重置追踪状态
      4. 后处理：线性插值填充短间隙 (≤3 帧)
      5. 评估最长连续有效段的 R² 和 Cv
      6. 输出 point_type 列: "raw" / "predicted" / "interpolated"

    Returns:
        dict: {
            "traj_df": 轨迹 DataFrame (含 point_type 列),
            "valid_frames": 检测到的有效帧数,
            "total_frames": 总帧数,
            "valid_ratio": 有效比例,
            "longest_valid_segment_frames": 最长连续有效段帧数,
            "longest_valid_segment_sec": 最长连续有效段时间(秒),
            "longest_segment_r2": 最长段 y-t 线性拟合 R²,
            "longest_segment_cv": 最长段速度 Cv,
            "has_quality_segment": 是否存在满足质量标准的段,
        }
    """
    reader = VideoReader(video_path)
    reader.open()

    fps = reader.fps
    if config.get("manual_fps") is not None:
        fps = config["manual_fps"]
        reader.fps = fps  # ★ 覆盖 reader.fps，让 get_frame_time() 使用正确帧率

    logger.info(
        "视频信息: %s | %d x %d | %.2f fps | %d 帧",
        os.path.basename(video_path),
        reader.width, reader.height, fps, reader.total_frames
    )

    # 比例尺
    scale_mm_per_px = config.get("scale_mm_per_px")
    if scale_mm_per_px is None:
        raise ValueError("scale_mm_per_px 未设置，无法进行坐标换算。")
    scale_m_per_px = mm_per_px_to_m_per_px(scale_mm_per_px)

    # ---- 动态计算识别参数（基于小球半径和比例尺） ----
    _compute_dynamic_detection_params(config)
    logger.info(
        "识别参数来源: %s\n"
        "  ball_radius_mm=%.3f, scale=%.6f mm/px\n"
        "  expected_radius=%.1f px, expected_area=%.0f px²\n"
        "  min_area=%d, max_area=%d\n"
        "  radius_range=[%.1f, %.1f]\n"
        "  circularity=%.2f",
        config.get("_param_source", "?"),
        config.get("ball_radius_mm", 0),
        scale_mm_per_px,
        config.get("_expected_radius_px", 0),
        config.get("_expected_area_px", 0),
        config.get("min_area_px", 0),
        config.get("max_area_px", 0),
        config.get("expected_radius_px_min", 0),
        config.get("expected_radius_px_max", 0),
        config.get("min_circularity", 0),
    )

    # ---- 动态计算追踪参数（搜索窗口、运动约束） ----
    _compute_dynamic_tracking_params(config)

    # ---- 检测区域 detect_roi ----
    # auto_shrink_roi: 默认关闭。开启后会围绕下落中心线缩窄检测区域以排除管壁等干扰。
    auto_shrink_roi = config.get("auto_shrink_roi", False)
    roi = config.get("roi")
    user_roi = list(roi) if roi else None

    fall_axis_x = config.get("fall_axis_x")
    if fall_axis_x is None and roi is not None:
        fall_axis_x = roi[0] + roi[2] / 2.0
        config["fall_axis_x"] = fall_axis_x

    if auto_shrink_roi and fall_axis_x is not None and fall_axis_x > 0 and roi is not None:
        drw = config.get("detect_roi_width", 80)
        detect_roi = [
            max(0, int(fall_axis_x - drw)),
            0,
            int(drw * 2),
            reader.height,
        ]
        if detect_roi[0] + detect_roi[2] > reader.width:
            excess = detect_roi[0] + detect_roi[2] - reader.width
            detect_roi[0] = max(0, detect_roi[0] - excess)
            detect_roi[2] = min(detect_roi[2], reader.width - detect_roi[0])
        config["detect_roi"] = detect_roi
    else:
        # Use the user-drawn ROI as the true detection region.  Extending it to
        # full frame height admits too many bottom/side structures in this
        # experiment setup and causes late-stage drift.
        config["detect_roi"] = list(roi) if roi else None

    logger.info(
        "检测区域:\n"
        "  user_roi           = %s\n"
        "  actual_detect_roi  = %s\n"
        "  auto_shrink_roi    = %s\n"
        "  fall_axis_x        = %s",
        user_roi, config.get("detect_roi"), auto_shrink_roi,
        f"{fall_axis_x:.0f}" if fall_axis_x else "None",
    )

    # 构建背景模型（基于 detect_roi，确保 bg_sub 与检测区域一致）
    background = None
    detect_method = config.get("detect_method", "auto")
    if detect_method in ("auto", "background_subtraction"):
        bg_roi = config.get("detect_roi") or config.get("roi")
        background = _build_background(reader, n_frames=50, roi=bg_roi)
        logger.info("背景模型已构建: 50 帧中值合成 (ROI: %s)", bg_roi)
        reader.reset()  # 回到开头

    # 传递 image_mode=False（视频模式，允许 bg_sub）
    config["image_mode"] = False
    detector = BallDetector(config, background=background)
    detector.reset()

    # ---- 坐标偏移：detect_roi → frame（set_prev 需要用 ROI 坐标） ----
    crop_roi = config.get("detect_roi") or config.get("roi")
    if crop_roi is not None:
        crop_offset_x, crop_offset_y = int(crop_roi[0]), int(crop_roi[1])
    else:
        crop_offset_x, crop_offset_y = 0, 0

    # ---- 初始位置（手动指定或首帧检测） ----
    init_x = config.get("init_ball_x")
    init_y = config.get("init_ball_y")
    if init_x is not None and init_y is not None:
        # 转换为 crop-ROI 坐标（BallDetector 内部使用）
        init_x_roi = init_x - crop_offset_x
        init_y_roi = init_y - crop_offset_y
        detector.set_prev(init_x_roi, init_y_roi)
        detector.set_search_center(init_x_roi, init_y_roi)
        logger.info("使用手动指定的初始小球位置: (%.0f, %.0f) → ROI (%.0f, %.0f)",
                    init_x, init_y, init_x_roi, init_y_roi)

    # ★ 初始位置等待：需连续多帧向下运动才确认小球到达，防止静止特征误确认
    _init_waiting = init_x is not None
    _init_confirm_frames = int(config.get("init_ball_confirm_frames", 5))
    _init_confirm_buf = []  # [{x, y, frame}], 累积在 init_ball 附近的候选点
    # 默认阈值 = 5 × ball_radius_px（用户要求最多 5 个球半径），最小 15px
    _exp_r = config.get("_expected_radius_px", 5.0)
    _init_dist_thresh = float(config.get("init_ball_distance_px",
                                          max(15.0, 5.0 * _exp_r)))

    # ---- 追踪参数 ----
    max_lost_frames = config.get("max_lost_frames", 30)

    # ---- 物理约束过滤器 ----
    dx_max = config.get("dx_max", 15)
    dy_back_tol = config.get("dy_back_tol", 3)
    dist_max = config.get("dist_max", 35)

    # ---- 中心通道约束 ----
    _dr = config.get("detect_roi")
    if _dr is not None:
        x_center = _dr[0] + _dr[2] / 2.0
        x_gate = _dr[2] * config.get("x_gate_ratio", 0.25)
    else:
        x_center = None
        x_gate = None

    # 保存原始搜索窗口大小（用于动态扩展）
    _orig_sw_w = detector.search_win_w
    _orig_sw_y_up = detector.search_win_y_up
    _orig_sw_y_down = detector.search_win_y_down

    # 下落中心线估计（前 15 个有效点的 X 中位数）
    axis_estimate_xs = []
    axis_estimated = False
    axis_estimate_needed = config.get("fall_axis_x") is None

    # ---- 帧状态定义 ----
    STATE_TRACKING = "tracking"
    STATE_LOST = "lost"
    STATE_REACQUIRE = "reacquire"
    STATE_STOPPED = "stopped"

    def _accept_prediction(record: dict, reason: str) -> bool:
        """使用短期运动预测保持搜索窗口移动。

        Predicted 点标记为 "predicted"，不计入 raw detection rate。
        仅用于在短暂丢帧时保持追踪连续性。

        ★ 连续 predicted ≥ max_consecutive_pred 帧后强制进 REACQUIRE，
        并在整个 ROI 内重新搜索真实 raw 点。
        Predicted 点不更新 detector 的搜索中心/预测状态，
        避免预测位置污染真实追踪。
        """
        nonlocal last_valid_x, last_valid_y, last_valid_frame, lost_counter, frame_state
        nonlocal consecutive_predicted

        if not bool(config.get("enable_prediction_fill", False)):
            return False

        max_pred = int(config.get("predict_max_frames", 18))
        if lost_counter >= max_pred:
            return False

        max_consecutive_pred = int(config.get("max_consecutive_predicted", 4))
        if consecutive_predicted >= max_consecutive_pred:
            # 连续预测太多，强制全 ROI 搜索
            detector.clear_search_center()
            frame_state = STATE_REACQUIRE
            return False

        pred = _predict_from_recent_points(records, frame_idx, config)
        if pred is None:
            return False

        px, py, pred_conf = pred
        record["x_px"] = px
        record["y_px"] = py
        record["x_m"] = px * scale_m_per_px
        record["y_m"] = py * scale_m_per_px
        record["radius_px"] = None
        record["area_px"] = None
        record["circularity"] = None
        record["confidence"] = pred_conf
        record["valid"] = True
        record["point_type"] = "predicted"
        record["rejection"] = reason
        record["frame_state"] = STATE_LOST if consecutive_predicted < max_consecutive_pred else STATE_REACQUIRE

        last_valid_x = px
        last_valid_y = py
        last_valid_frame = frame_idx
        # ★ predicted 点不更新 detector 搜索中心，不污染真实追踪状态
        consecutive_predicted += 1
        return True

    # 标注视频
    save_video = config.get("save_marked_video", True)
    writer = None
    if save_video:
        out_dir = config.get("output_dir", "data/results")
        os.makedirs(out_dir, exist_ok=True)
        out_video_path = os.path.join(out_dir, "marked_video.mp4")
        writer = VideoWriterWrapper(out_video_path, fps, reader.width, reader.height)
        logger.info("标注视频将保存至: %s", out_video_path)

    # ---- 逐帧处理 ----
    records = []
    frame_debug_records = []  # 增强逐帧 debug 数据
    reject_reason_counter = {}  # 拒绝原因累计计数
    frame_idx = 0
    lost_counter = 0
    last_valid_x = None
    last_valid_y = None
    last_valid_frame = None
    total_found = 0
    total_anomaly = 0
    frame_state = STATE_TRACKING
    # 首次丢失时的帧号 + 最后一个有效点
    first_lost_frame = None
    last_valid_before_loss = (None, None)
    # 重新捕获跟踪
    reacquire_count = 0
    reacquire_success = 0
    startup_candidate = None
    startup_candidate_count = 0
    # ★ 启动确认缓冲区：收集首帧候选点用于 tracklet 验证
    consecutive_predicted = 0  # ★ 连续 predicted 帧计数，超限进 REACQUIRE
    startup_confirmed = False  # ★ 启动 tracklet 已确认/超时回退
    startup_buffer = []  # list of {x, y, diff, frame}
    # 函数：验证启动 tracklet 是否有持续向下运动
    def _startup_tracklet_confirmed(buf: list) -> bool:
        min_frames = int(config.get("startup_confirm_frames", 5))
        if len(buf) < min_frames:
            return False
        recent = buf[-min_frames:]
        ys = [p["y"] for p in recent]
        xs = [p["x"] for p in recent]
        # 大部分帧应向下运动 (dy > -3 容忍微抖)
        dy_list = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        if sum(1 for d in dy_list if d > -3) < len(dy_list) * 0.6:
            return False
        # 总向下位移 ≥ 阈值
        if ys[-1] - ys[0] < float(config.get("min_startup_dy_px", 8.0)):
            return False
        # x 不剧烈漂移
        if max(xs) - min(xs) > float(config.get("startup_max_x_drift_px", 35.0)):
            return False
        # 非纯静止点 (y 变化 ≤ 2px)
        if max(ys) - min(ys) < 3.0:
            return False
        return True
    # 拒绝原因日志（按帧输出）
    reject_log = []
    # 逐帧候选点收集（用于 candidate_debug.csv）
    candidate_records = []

    # ---- 辅助函数：动态搜索窗口 ----
    def _adjust_search_window(lc: int):
        """根据连续丢失帧数动态扩展搜索窗口。"""
        if lc <= 0:
            detector.set_search_window(_orig_sw_w, _orig_sw_y_up, _orig_sw_y_down)
        elif lc <= 3:
            factor = 1.5
            detector.set_search_window(
                int(_orig_sw_w * factor),
                int(_orig_sw_y_up * factor),
                int(_orig_sw_y_down * factor),
            )
        elif lc <= 8:
            factor = 2.5
            detector.set_search_window(
                int(_orig_sw_w * factor),
                int(_orig_sw_y_up * factor),
                int(_orig_sw_y_down * factor),
            )
        elif lc <= 20:
            factor = 4.0
            detector.set_search_window(
                int(_orig_sw_w * factor),
                int(_orig_sw_y_up * factor),
                int(_orig_sw_y_down * factor),
            )
        else:
            # 丢失太久，清除搜索中心，全 detect_roi 搜索
            detector.set_search_window(
                _orig_sw_w * 6,
                _orig_sw_y_up * 6,
                int(_orig_sw_y_down * 6),
            )

    while True:
        ret, frame = reader.read_frame()
        if not ret:
            break

        time_s = reader.get_frame_time(frame_idx)

        # ★ 根据 lost_counter 动态调整搜索窗口
        _adjust_search_window(lost_counter)

        # ★ 如果连续丢失太多帧，清除搜索中心（全 detect_roi 搜索）
        if lost_counter >= max_lost_frames:
            detector.clear_search_center()
            # 重置丢失计数器避免反复触发
            frame_state = STATE_STOPPED
            logger.debug("帧 %d: 连续丢失 %d 帧，切换到全 ROI 搜索。",
                         frame_idx, lost_counter)

        # 执行检测
        result = detector.detect(frame, frame_idx)

        # ★ 强制 ROI 边界过滤：拒绝 user_roi 外的候选（belt-and-suspenders 保护）
        if user_roi is not None and result["found"]:
            _rx, _ry, _rw, _rh = user_roi
            _rx2, _ry2 = _rx + _rw, _ry + _rh
            _filtered = [c for c in result.get("candidates", [])
                         if _rx <= c.get("cx", 0) <= _rx2
                         and _ry <= c.get("cy", 0) <= _ry2]
            if not _filtered:
                result["found"] = False
                result["candidates"] = []
            else:
                result["candidates"] = _filtered
                # 更新最佳候选为 ROI 内第一候选
                result["x_px"] = _filtered[0]["cx"]
                result["y_px"] = _filtered[0]["cy"]

        # ★ 初始位置等待：需要连续 N 帧在 init_ball 附近 + 向下运动才确认
        # 只靠距离判断会被静止特征（刻度线/反光）误确认
        if _init_waiting and result["found"]:
            if result.get("fallback", False):
                result["found"] = False
                detector.set_prev(init_x_roi, init_y_roi)
                detector.clear_search_center()
                _init_confirm_buf.clear()
            else:
                _bc = result.get("candidates", [{}])[0] if result.get("candidates") else {}
                _cx = float(_bc.get("cx", 0))
                _cy = float(_bc.get("cy", 0))
                _ud = np.hypot(_cx - init_x, _cy - init_y)
                if _ud > _init_dist_thresh:
                    # 不在范围内 → 清空缓冲区，继续等待
                    result["found"] = False
                    detector.set_prev(init_x_roi, init_y_roi)
                    detector.clear_search_center()
                    _init_confirm_buf.clear()
                else:
                    # 在范围内 → 累积到确认缓冲区
                    _init_confirm_buf.append({"x": _cx, "y": _cy, "frame": frame_idx})
                    # 缓冲区过长（>15帧静止不动）→ 清空，等真小球
                    if len(_init_confirm_buf) > 15:
                        ys_stale = [p["y"] for p in _init_confirm_buf]
                        if max(ys_stale) - min(ys_stale) < 3.0:
                            if frame_idx < 100 or frame_idx % 100 == 0:
                                logger.debug("帧 %d: 静止特征在 init_ball 附近(disp=%.1fpx)，清空缓冲区等待真小球",
                                            frame_idx, max(ys_stale) - min(ys_stale))
                            _init_confirm_buf.clear()
                    # 检查是否满足确认条件：连续 N 帧持续向下运动
                    if len(_init_confirm_buf) >= _init_confirm_frames:
                        recent = _init_confirm_buf[-_init_confirm_frames:]
                        ys = [p["y"] for p in recent]
                        # 逐帧检查：相邻帧 dy > -0.5px（允许微抖但不允许明显回退）
                        dy_list = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
                        n_down = sum(1 for d in dy_list if d > -0.5)
                        y_std = float(np.std(ys))
                        # 确认条件：绝大部分帧向下 + 位置有变化（非静止） + 总体向下
                        if n_down >= len(dy_list) and y_std > 0.5 and ys[-1] > ys[0]:
                            _init_waiting = False
                            logger.info("帧 %d: 初始小球已确认 (dist=%.0fpx, dy=%.1fpx, σ=%.1fpx over %d帧)，开始追踪",
                                        frame_idx, _ud, ys[-1] - ys[0], y_std, _init_confirm_frames)
                            _init_confirm_buf.clear()
                        else:
                            result["found"] = False
                    else:
                        # 缓冲区未满 → 本帧不算找到
                        result["found"] = False

        # ★ 首次检测到小球时设置模板（须 init_ball 已确认 + 不在等待阶段）
        if result["found"] and detector._template_gray is None and not _init_waiting:
            _best_cand = result.get("candidates", [{}])[0]
            _cx = _best_cand.get("cx")
            _cy = _best_cand.get("cy")
            if _cx is not None and _cy is not None:
                detector.set_template(frame, float(_cx), float(_cy))
                logger.info("帧 %d: 首次检测 → 模板已设置 (cx=%.1f, cy=%.1f)", frame_idx, float(_cx), float(_cy))

        # ★ 调试：打印头 30 帧的模板匹配分
        if frame_idx < 30 and result["found"]:
            _top = result.get("candidates", [{}])[0]
            _tm = _top.get("template_match_score", -1.0)
            _sc = _top.get("score", 0)
            _src = _top.get("source", "?")
            logger.info("帧 %d: tm_score=%.3f  ball_score=%.0f  src=%s  cx=%.0f cy=%.0f",
                        frame_idx, _tm, _sc, _src,
                        _top.get("cx", 0), _top.get("cy", 0))

        # ---- 记录本帧拒绝原因 ----
        rejection = ""
        is_fallback = result.get("fallback", False)

        # 初始化记录
        has_prev = last_valid_x is not None and last_valid_y is not None
        record = {
            "frame": frame_idx,
            "time_s": time_s,
            "x_px": None, "y_px": None,
            "x_m": None, "y_m": None,
            "radius_px": None, "area_px": None,
            "circularity": None, "confidence": None,
            "valid": False,
            "point_type": None,
            "lost_counter": lost_counter,
            "frame_state": frame_state,
            "rejection": "",
            # ---- 增强调试字段 ----
            "raw_candidate_count": len(result.get("candidates", [])),
            "method_used": result.get("method_used", ""),
            "is_selected": int(result.get("found", False)),
            "dx_from_prev": None,
            "dy_from_prev": None,
            "jump_px": None,
        }

        if result["found"]:
            motion_choice = _select_motion_consistent_detection(
                result, records, crop_offset_x, crop_offset_y, config
            )
            if motion_choice is not None:
                x_px = motion_choice["x_px"]
                y_px = motion_choice["y_px"]
                selected_diff_score = motion_choice.get("diff_score", 0.0)
                result["x_px"] = x_px
                result["y_px"] = y_px
                result["radius_px"] = motion_choice.get("radius_px", result.get("radius_px"))
                result["area_px"] = motion_choice.get("area_px", result.get("area_px"))
                result["circularity"] = motion_choice.get("circularity", result.get("circularity"))
                result["confidence"] = motion_choice.get("confidence", result.get("confidence", 0.0))
                if motion_choice.get("_rescued"):
                    result["rescued_motion_candidate"] = True
                    result["motion_reject_reason"] = motion_choice.get("_motion_reject_reason", "")
                if motion_choice.get("rank", 1) != 1:
                    result["method_used"] = f"{result.get('method_used', '')}|motion_rank={motion_choice['rank']}"
            else:
                x_px = result["x_px"]
                y_px = result["y_px"]
                # fallback: 从 result 的候选列表中获取 diff_score
                _fc = result.get("candidates", [{}])
                selected_diff_score = _fc[0].get("diff_score", 0.0) if _fc else 0.0
            confidence = result.get("confidence", 0.0)

            # ★ 模板匹配分数提取（用于后续跳过运动门控）
            _best_cand = result.get("candidates", [{}])[0] if result.get("candidates") else {}
            tm_score = _best_cand.get("template_match_score", -1.0)
            if motion_choice is not None:
                tm_score = motion_choice.get("template_match_score", tm_score)
            tm_trust = float(config.get("template_trust_threshold", 0.65))

            # ---- 异常检测 ----
            is_anomaly = False
            reject_reasons = []
            startup_confirm_frames = int(config.get("startup_confirm_frames", 3))
            startup_confirm_dx = float(config.get("startup_confirm_dx_px", max(12.0, dx_max * 0.8)))
            startup_confirm_dy = float(config.get("startup_confirm_dy_px", max(18.0, dist_max * 0.35)))

            if not is_fallback:
                if motion_choice is None and len(_recent_valid_points(records, 3)) >= 1:
                    # ★ startup/reacquire 阶段不过硬拒绝，降级为 warning
                    if frame_state in (STATE_LOST, STATE_REACQUIRE) or last_valid_frame is None or (frame_idx - (last_valid_frame or 0)) > 3:
                        pass  # 宽松：允许进入后续检查
                    else:
                        reject_reasons.append("no_motion_consistent_candidate")

                # 1. 自适应跳变检测（基于帧间距 dt）
                if last_valid_x is not None and last_valid_y is not None:
                    dx = abs(x_px - last_valid_x)
                    dy = y_px - last_valid_y
                    dist = np.hypot(x_px - last_valid_x, y_px - last_valid_y)
                    gap_frames = frame_idx - last_valid_frame if last_valid_frame is not None else 1
                    gap_frames = max(1, gap_frames)
                    gap_factor = max(1.0, gap_frames * 1.5)  # 丢帧越多门控越宽

                    # startup/reacquire 阶段用宽松门控
                    if frame_state in (STATE_LOST, STATE_REACQUIRE) or last_valid_frame is None or (frame_idx - (last_valid_frame or 0)) > 5:
                        _dx_max = max(dx_max * 2.5, 50.0)
                        _dy_back_tol = max(dy_back_tol * 2.5, 12.0)
                        _dist_max = max(dist_max * 2.5, 120.0)
                    else:
                        _dx_max = dx_max * gap_factor
                        _dy_back_tol = max(dy_back_tol, 5.0)
                        _dist_max = dist_max * gap_factor

                    if dx > _dx_max:
                        reject_reasons.append(f"dx={dx:.1f}>{_dx_max:.1f}")
                    if dy < -_dy_back_tol:
                        reject_reasons.append(f"dy_back={dy:.1f}")
                    if dist > _dist_max:
                        reject_reasons.append(f"dist={dist:.1f}>{_dist_max:.1f}")

                    # 保存帧间运动数据到 record
                    record["dx_from_prev"] = x_px - last_valid_x  # signed dx
                    record["dy_from_prev"] = dy                   # signed dy (y_px - last_valid_y)
                    record["jump_px"] = dist

                # 2. 中心通道约束
                if x_center is not None:
                    d_from_center = abs(x_px - x_center)
                    gate_limit = x_gate
                    if last_valid_x is None or last_valid_y is None:
                        gate_limit = x_gate * float(config.get("startup_gate_scale", 2.0))
                    if d_from_center > gate_limit:
                        reject_reasons.append(
                            f"outside_gate(d={d_from_center:.0f}>{gate_limit:.0f})"
                        )

                # Full-height detection is only allowed as a continuation of an
                # existing track.  It must not reacquire a static object outside
                # the user-drawn ROI.
                #
                # Direction-aware logic:
                #   - Ball below ROI bottom moving downward → lenient (natural exit)
                #   - Ball above ROI top or to the side → strict (likely interference)
                #   - Static point (barely moves frame-to-frame) → always reject
                if user_roi is not None:
                    urx, ury, urw, urh = [float(v) for v in user_roi]
                    inside_user_roi = (
                        urx <= x_px <= urx + urw
                        and ury <= y_px <= ury + urh
                    )
                    if not inside_user_roi:
                        if last_valid_x is None or last_valid_y is None:
                            startup_x_margin = max(
                                dx_max * float(config.get("startup_roi_x_margin_scale", 3.0)),
                                x_gate if x_gate is not None else 0,
                            )
                            startup_y_above = max(
                                float(config.get("search_win_y_up", 20)) * float(config.get("startup_roi_y_above_scale", 3.0)),
                                float(config.get("startup_roi_y_above_px", 80)),
                            )
                            startup_y_below = float(config.get("startup_roi_y_below_px", 12))
                            near_start_zone = (
                                (urx - startup_x_margin <= x_px <= urx + urw + startup_x_margin)
                                and (ury - startup_y_above <= y_px <= ury + urh + startup_y_below)
                            )
                            if not near_start_zone:
                                reject_reasons.append("outside_user_roi_without_track")
                        else:
                            # Determine exit direction relative to ROI
                            below_roi = (y_px > ury + urh)
                            within_x_band = (urx - dx_max * 1.25 <= x_px <= urx + urw + dx_max * 1.25)
                            moving_down = (y_px > last_valid_y - dy_back_tol)

                            if below_roi and within_x_band and moving_down:
                                # Natural downward exit: use generous lost tolerance
                                max_lost_below = int(config.get("outside_roi_max_lost_below", 8))
                                if lost_counter > max_lost_below:
                                    reject_reasons.append(
                                        f"outside_user_roi_below_after_lost({lost_counter}>{max_lost_below})"
                                    )
                                else:
                                    # Still check continuity with relaxed constraints
                                    out_dx = abs(x_px - last_valid_x)
                                    out_dy = y_px - last_valid_y
                                    out_dist = np.hypot(out_dx, out_dy)
                                    # Relax dist_max by 50% for below-ROI continuation
                                    relaxed_dist = dist_max * 1.75
                                    relaxed_dx = dx_max * 1.75
                                    if out_dx > relaxed_dx or out_dy < -dy_back_tol or out_dist > relaxed_dist:
                                        reject_reasons.append(
                                            "outside_user_roi_below_not_continuous"
                                            f"(dx={out_dx:.1f},dy={out_dy:.1f},dist={out_dist:.1f})"
                                        )
                            else:
                                # Not a natural downward exit: use strict tolerance
                                max_lost_strict = int(config.get("outside_roi_max_lost", 3))
                                if lost_counter > max_lost_strict:
                                    reject_reasons.append(
                                        f"outside_user_roi_after_lost({lost_counter}>{max_lost_strict})"
                                    )
                                else:
                                    out_dx = abs(x_px - last_valid_x)
                                    out_dy = y_px - last_valid_y
                                    out_dist = np.hypot(out_dx, out_dy)
                                    if out_dx > dx_max * 1.25 or out_dy < -dy_back_tol or out_dist > dist_max * 1.25:
                                        reject_reasons.append(
                                            "outside_user_roi_not_continuous"
                                            f"(dx={out_dx:.1f},dy={out_dy:.1f},dist={out_dist:.1f})"
                                        )

                # ★ 3. 启动确认：首次建立轨迹时用 tracklet 验证（连续 N 帧向下运动）
                # 不再按单帧最高 diff_score 直接锁定，避免首帧锁定到刻度线/反光等伪目标。
                # startup_confirmed=False 时累积候选缓冲区，验证通过或超时后置 True。
                if not reject_reasons and last_valid_x is None and last_valid_y is None:
                    if not startup_confirmed:
                        frame_center_x = reader.width / 2.0
                        center_margin = reader.width * 0.40
                        if not (frame_center_x - center_margin <= x_px <= frame_center_x + center_margin):
                            reject_reasons.append(
                                f"startup_outside_center(x={x_px:.0f},center={frame_center_x:.0f})"
                            )
                            startup_buffer.clear()
                        else:
                            startup_buffer.append({
                                "x": x_px, "y": y_px, "diff": selected_diff_score,
                                "frame": frame_idx,
                            })
                            startup_confirm_frames = int(config.get("startup_confirm_frames", 5))
                            startup_fb = int(config.get("startup_fallback_frames", 25))
                            buf_max = max(startup_confirm_frames + 5, startup_fb)
                            if len(startup_buffer) > buf_max:
                                startup_buffer.pop(0)
                            if _startup_tracklet_confirmed(startup_buffer):
                                startup_confirmed = True
                            elif len(startup_buffer) >= startup_fb:
                                logger.info(
                                    "启动确认超时 (%d 帧未确认)，回退接受候选",
                                    len(startup_buffer),
                                )
                                startup_confirmed = True
                            else:
                                reject_reasons.append("startup_not_confirmed")
                    # 已确认或已超时回退 → 正常接受此点

                if frame_state in (STATE_REACQUIRE, STATE_STOPPED):
                    # reacquire 时使用更宽的跳变门控（已在上面设置），但不完全跳过。
                    # 完全跳过会导致 tracker 锁定到任意噪声点（如帧间距过大时的速度异常）。
                    # 只放宽那些略超门控的微小违规，严重跳变仍需拒绝。
                    if reject_reasons:
                        still_blocked = []
                        for r in reject_reasons:
                            if r.startswith("dx="):
                                val = float(r.split("=")[1].split(">")[0])
                                limit = float(r.split(">")[1])
                                if val > limit * 3.0:
                                    still_blocked.append(r)
                            elif r.startswith("dist="):
                                val = float(r.split("=")[1].split(">")[0])
                                limit = float(r.split(">")[1])
                                if val > limit * 3.0:
                                    still_blocked.append(r)
                            elif r.startswith("dy_back="):
                                val = float(r.split("=")[-1])
                                if val < -20.0:
                                    still_blocked.append(r)
                            else:
                                still_blocked.append(r)
                        reject_reasons = still_blocked

                # ★ 4. large_ball_mode 下额外放宽约束
                if config.get("large_ball_mode", False):
                    if reject_reasons:
                        # 大球模式：只保留中心通道+最严重的跳变
                        filtered = [
                            r for r in reject_reasons
                            if r.startswith("outside_gate")
                            or r.startswith("outside_user_roi")
                            or r.startswith("dist=")
                        ]
                        # dist > max_dist * 1.5 才拒绝
                        filtered = [r for r in filtered
                                    if not r.startswith("dist=") or any(
                                        float(r.split("=")[-1].split(">")[0]) > config.get("dist_max", 80) * 1.5
                                        for _ in [1]
                                    )]
                        reject_reasons = filtered

                # ★ 5. diff_score 低不能单独硬拒绝（终端速度阶段 diff 天然低）
                # 只有同时满足 (a) 极低 diff 或 (b) 长时间接近静止 才判定为静态噪声
                if not reject_reasons and last_valid_x is not None:
                    min_track_diff = float(config.get("min_track_diff_score", 3.0))
                    # 取最近 valid 点的累计位移
                    recent_valid = [
                        r for r in records[-10:]
                        if r.get("valid") and r.get("y_px") is not None
                    ]
                    recent_n = len(recent_valid)
                    if recent_n >= 3:
                        recent_ys = [r["y_px"] for r in recent_valid]
                        recent_xs = [r["x_px"] for r in recent_valid]
                        y_disp = max(recent_ys) - min(recent_ys)
                        x_disp = max(recent_xs) - min(recent_xs)
                        # 条件 1: 极低 diff + 几乎静止
                        if selected_diff_score < 1.0 and y_disp < 1.5:
                            reject_reasons.append(
                                f"static_noise(diff={selected_diff_score:.1f},y_disp={y_disp:.1f})"
                            )
                        # 条件 2: 超过 5 帧累计位移 < 2px → 可能锁定静态噪点
                        elif recent_n >= 5 and y_disp < 2.0 and x_disp < 2.0 and selected_diff_score < 2.5:
                            reject_reasons.append(
                                f"static_lock(y_disp={y_disp:.1f},x_disp={x_disp:.1f},diff={selected_diff_score:.1f})"
                            )
                    if selected_diff_score < 1.0 and not reject_reasons:
                        # diff 低但小球仍在运动 → 仅记录 warning，不拒绝
                        record["_low_diff_warning"] = True

                # ★ 模板匹配高分 → 信任检测，清空拒绝原因
                if tm_score >= tm_trust and reject_reasons:
                    reject_reasons.clear()
                    if frame_idx < 100:
                        logger.info("帧 %d: 模板信任跳过拒绝 (tm=%.3f)", frame_idx, tm_score)

                if reject_reasons:
                    is_anomaly = True
                    rejection = " | ".join(reject_reasons)

            # ★ 如果 fallback（预测补位）的置信度太低也拒绝
            if is_fallback and last_valid_y is not None:
                pred_dy = y_px - last_valid_y
                if pred_dy < -dy_back_tol * 2:
                    is_anomaly = True
                    rejection = f"fallback_dy_back={pred_dy:.1f}"
                    # 清扫 detector 的预测状态
                    detector.lost_frame()

            if not is_anomaly:
                record["x_px"] = x_px
                record["y_px"] = y_px
                record["x_m"] = x_px * scale_m_per_px
                record["y_m"] = y_px * scale_m_per_px
                record["radius_px"] = result.get("radius_px")
                record["area_px"] = result.get("area_px")
                record["circularity"] = result.get("circularity")
                record["confidence"] = confidence
                record["diff_score"] = selected_diff_score
                record["valid"] = True
                record["point_type"] = "predicted" if is_fallback else "raw"
                if frame_state in (STATE_REACQUIRE, STATE_STOPPED):
                    reacquire_success += 1
                record["frame_state"] = STATE_TRACKING
                frame_state = STATE_TRACKING

                # 反馈追踪状态（仅 raw 点更新 detector 内部状态）
                if not is_fallback:
                    detector.set_prev(x_px - crop_offset_x, y_px - crop_offset_y)
                    # ★ 模板进化：每次原始检测确认后更新模板（首次自动设置）
                    if bool(config.get("template_enabled", True)):
                        detector.evolve_template(frame, x_px, y_px)
                # ★ predicted 点不更新 detector 状态，避免预测位置污染真实追踪
                last_valid_x = x_px
                last_valid_y = y_px
                last_valid_frame = frame_idx
                lost_counter = 0
                consecutive_predicted = 0  # ★ raw 点接入后重置连续预测计数
                total_found += 1
                startup_candidate = None
                startup_candidate_count = 0
                first_lost_frame = None
                last_valid_before_loss = (None, None)

                # 收集用于估计中心线的 X 坐标
                if axis_estimate_needed and not axis_estimated:
                    axis_estimate_xs.append(x_px)
                    if len(axis_estimate_xs) >= 15:
                        estimated_axis = float(np.median(axis_estimate_xs))
                        detector.set_fall_axis_x(estimated_axis)
                        axis_estimated = True
                        logger.info("下落中心线已自动估计: fall_axis_x = %.1f px",
                                    estimated_axis)
            else:
                total_anomaly += 1
                lost_counter += 1
                if not _accept_prediction(record, "wrong_detection: " + rejection):
                    # ★ 同步通知检测器追踪丢失
                    _restore_detector_to_last_good(detector, last_valid_x, last_valid_y,
                                                   crop_offset_x, crop_offset_y)
                    detector.lost_frame()
                    if frame_state == STATE_TRACKING:
                        frame_state = STATE_LOST
                        first_lost_frame = frame_idx
                        last_valid_before_loss = (last_valid_x, last_valid_y)
                        detector.clear_search_center()  # ★ LOST 状态禁用旧搜索窗口
                    elif lost_counter >= 5 and frame_state != STATE_REACQUIRE:
                        frame_state = STATE_REACQUIRE
                        reacquire_count += 1
                        detector.clear_search_center()  # ★ REACQUIRE 回到全 ROI 搜索
        else:
            lost_counter += 1
            if not _accept_prediction(record, result.get("fail_reason", "no_detection")):
                detector.lost_frame()  # ★ 同步通知检测器本帧丢失
                if frame_state == STATE_TRACKING:
                    frame_state = STATE_LOST
                    first_lost_frame = frame_idx
                    last_valid_before_loss = (last_valid_x, last_valid_y)
                    detector.clear_search_center()  # ★ LOST 状态禁用旧搜索窗口
                elif lost_counter >= 5 and frame_state != STATE_REACQUIRE:
                    frame_state = STATE_REACQUIRE
                    reacquire_count += 1
                    detector.clear_search_center()  # ★ REACQUIRE 回到全 ROI 搜索

        record["rejection"] = rejection
        records.append(record)

        # ★ 等待阶段不受 lost_counter / STATE_LOST 影响
        if _init_waiting:
            lost_counter = 0
            frame_state = STATE_TRACKING
            detector.set_prev(init_x_roi, init_y_roi)
            detector.clear_search_center()  # 全 ROI 搜索，不限制窗口

        # ---- 收集逐帧候选点信息（用于 candidate_debug.csv） ----
        candidates_list = result.get("candidates", [])
        if not candidates_list:
            candidate_records.append({
                "frame": frame_idx, "time_s": time_s,
                "candidate_id": 0, "x": None, "y": None,
                "radius_px": None, "area_px": None,
                "circularity": None, "score": None,
                "size_score": None, "contrast_score": None,
                "diff_score": None, "motion_bonus": None,
                "continuity": None, "isolation_score": None,
                "center_band_score": None, "axis_score": None,
                "long_line_penalty": None, "source_method": "",
                "dist_to_pred": None, "reject_reason": "no_candidate",
                "accepted": 0,
            })
        else:
            # fallback/预測候选
            is_fallback = result.get("fallback", False)
            accepted_x = record.get("x_px")
            accepted_y = record.get("y_px")
            for ci, cand in enumerate(candidates_list):
                dist_pred = cand.get("dist_to_axis", 0)
                is_acc = (
                    record["valid"] and not is_fallback
                    and record["x_px"] is not None
                    and abs(record["x_px"] - cand.get("cx", -9999)) < 0.5
                    and abs(record["y_px"] - cand.get("cy", -9999)) < 0.5
                )
                candidate_records.append({
                    "frame": frame_idx, "time_s": time_s,
                    "candidate_id": ci + 1,
                    "x": cand.get("cx"), "y": cand.get("cy"),
                    "radius_px": cand.get("radius"),
                    "area_px": cand.get("area"),
                    "circularity": cand.get("circularity"),
                    "score": cand.get("score"),
                    "size_score": cand.get("size_score"),
                    "contrast_score": cand.get("contrast", 0),
                    "diff_score": cand.get("diff_score"),
                    "prediction_score": cand.get("prediction_score"),
                    "motion_continuity": cand.get("motion_continuity"),
                    "motion_bonus": cand.get("motion_bonus"),
                    "continuity": cand.get("continuity"),
                    "isolation_score": cand.get("isolation_score"),
                    "center_band_score": cand.get("center_band_score"),
                    "axis_score": cand.get("axis_score"),
                    "long_line_penalty": cand.get("long_line_penalty"),
                    "source_method": cand.get("source_method", ""),
                    "dist_to_pred": cand.get("dist_to_axis"),
                    "contrast": cand.get("contrast"),
                    "inner_mean": cand.get("inner_mean"),
                    "outer_mean": cand.get("outer_mean"),
                    "reject_reason": cand.get("reject_reason", ""),
                    "accepted": 1 if is_acc else 0,
                })

        # ---- 收集增强逐帧 debug 数据 ----
        _axis = config.get("fall_axis_x")
        _x = record.get("x_px")
        _dx = record.get("dx_from_prev")
        _dy = record.get("dy_from_prev")
        _jump = record.get("jump_px")
        _pt = record.get("point_type")
        frame_debug_records.append({
            "frame": frame_idx,
            "time_s": time_s,
            "raw_candidate_count": record.get("raw_candidate_count", 0),
            "pre_search_candidates": result.get("pre_search_candidates", 0),
            "search_window_active": int(result.get("search_window_active", False)),
            "selected_x": _x,
            "selected_y": record.get("y_px"),
            "selected_radius": record.get("radius_px"),
            "selected_confidence": record.get("confidence"),
            "selected_diff_score": record.get("diff_score", 0.0),
            "method_used": record.get("method_used", ""),
            "is_selected": int(result.get("found", False)),
            "is_predicted": int(_pt == "predicted"),
            "is_valid_real": int(_pt == "raw" and record.get("valid", False)),
            "is_interpolated": int(_pt == "interpolated"),
            "frame_state": record.get("frame_state", ""),
            "tracking_state": frame_state,
            "lost_counter": lost_counter,
            "reject_reason": record.get("rejection", ""),
            "dx_from_prev": _dx,
            "dy_from_prev": _dy,
            "jump_px": _jump,
            "instant_velocity_px_s": (_jump * fps) if _jump is not None else None,
            "x_drift_px": abs(_x - _axis) if (_x is not None and _axis is not None) else None,
            "y_monotonic_ok": int(_dy >= -dy_back_tol) if _dy is not None else 1,
            "confidence_ok": int(record.get("confidence", 0) >= 0.5) if record.get("confidence") is not None else 0,
            "radius_ok": int(
                config.get("expected_radius_px_min", 0) <= record.get("radius_px", 0) <= config.get("expected_radius_px_max", 999)
            ) if record.get("radius_px") is not None else 0,
        })

        # ★ 实时回调：每帧处理完毕后通知调用者
        if frame_callback is not None:
            frame_callback(
                frame_idx=frame_idx,
                x_px=record.get("x_px"),
                y_px=record.get("y_px"),
                time_s=record["time_s"],
                radius_px=record.get("radius_px"),
                is_valid=bool(record["valid"] and record.get("x_px") is not None),
                is_fallback=(record.get("point_type") == "predicted"),
            )

        frame_idx += 1

        # ---- 标注视频 ----
        if writer is not None:
            marked = frame.copy()
            if record["valid"] and record["x_px"] is not None:
                cx = int(round(record["x_px"]))
                cy = int(round(record["y_px"]))
                r = int(round(record["radius_px"])) if record["radius_px"] else 10
                pt = record.get("point_type", "raw")
                if pt == "raw":
                    color = (0, 255, 0)
                    label = f"({cx}, {cy})"
                elif pt == "predicted":
                    color = (0, 165, 255)
                    label = "PRED"
                else:
                    color = (255, 165, 0)
                    label = "INT"
                cv2.circle(marked, (cx, cy), r, color, 2)
                cv2.circle(marked, (cx, cy), 3, (255, 0, 0), -1)
                cv2.putText(marked, label, (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            # ★ 显示追踪状态
            state_colors = {
                STATE_TRACKING: (0, 255, 0),
                STATE_LOST: (0, 165, 255),
                STATE_REACQUIRE: (255, 0, 255),
                STATE_STOPPED: (0, 0, 255),
            }
            sc = state_colors.get(frame_state, (128, 128, 128))
            cv2.putText(marked, f"Frame: {frame_idx}  Time: {time_s:.3f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(marked, f"State: {frame_state.upper()}  Lost: {lost_counter}",
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sc, 2)
            writer.write_frame(marked)

    reader.release()
    if writer is not None:
        writer.release()

    # ---- 构建 DataFrame ----
    df = pd.DataFrame(records)

    # ---- 后处理：先剔除离群点，再插值，再复查 ----
    # 这样远处误识别点不会把插值轨迹拉偏。
    df = _remove_trajectory_outliers(df, config)
    df = _trim_after_breakaway(df, config)
    df = _interpolate_short_gaps(df, max_gap_frames=3, scale_m_per_px=scale_m_per_px)
    df = _remove_trajectory_outliers(df, config)
    df = _trim_after_breakaway(df, config)

    # ---- 后处理：归一化 y_m 为相对于首帧物理位移 ----
    raw_first = df[df["point_type"] == "raw"].iloc[0] if (df["point_type"] == "raw").any() else None
    if raw_first is not None:
        y0_px = raw_first["y_px"]
        # 对所有有效点重新计算 y_m = (y_px - y0_px) * scale_m_per_px
        valid_idx = df["valid"]
        df.loc[valid_idx, "y_m"] = (df.loc[valid_idx, "y_px"] - y0_px) * scale_m_per_px
        logger.info("y_m 已归一化: y0_px = %.1f, scale = %.8f m/px", y0_px, scale_m_per_px)
    else:
        logger.warning("无原始检测点，无法归一化 y_m")

    # ---- 统计（valid_mask 仅基于 raw detection，排除 predicted/interpolated） ----
    x_arr = pd.to_numeric(df["x_px"], errors="coerce").values
    y_arr = pd.to_numeric(df["y_px"], errors="coerce").values
    valid_mask = (df["point_type"] == "raw").values & np.isfinite(x_arr) & np.isfinite(y_arr)
    valid_count = int(valid_mask.sum())
    total_count = len(df)
    ratio = valid_count / total_count if total_count > 0 else 0.0

    # 最长连续有效段
    segment_gap_frames = int(config.get("segment_gap_frames", 3))
    segments = _find_valid_segments(valid_mask, max_gap=segment_gap_frames)
    longest_seg_frames = 0
    longest_seg_sec = 0.0
    longest_seg_r2 = 0.0
    longest_seg_cv = 1.0
    has_quality = False

    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start
        if seg_len > longest_seg_frames:
            longest_seg_frames = seg_len
            seg_times = df.iloc[seg_start:seg_end]["time_s"]
            if len(seg_times) > 1:
                longest_seg_sec = seg_times.iloc[-1] - seg_times.iloc[0]

            # 评估该段质量
            seg_df = df.iloc[seg_start:seg_end]
            r2, cv = _assess_segment_quality(seg_df)
            longest_seg_r2 = r2 if r2 is not None else 0.0
            longest_seg_cv = cv if cv is not None else 1.0

            # 质量标准：≥80 帧、R²≥0.995、Cv<5%
            quality_threshold_r2 = config.get("quality_r2_threshold", 0.995)
            quality_threshold_cv = config.get("quality_cv_threshold", 0.05)
            quality_min_frames = config.get("quality_min_frames", 80)
            if (seg_len >= quality_min_frames
                    and r2 is not None and r2 >= quality_threshold_r2
                    and cv is not None and cv < quality_threshold_cv):
                has_quality = True

    # 统计各点类型数量
    raw_count = int((df["point_type"] == "raw").sum())
    pred_count = int((df["point_type"] == "predicted").sum())
    interp_count = int((df["point_type"] == "interpolated").sum())
    logger.info(
        "追踪完成: 总计 %d 帧, raw=%d, predicted=%d, interpolated=%d",
        total_count, raw_count, pred_count, interp_count
    )
    logger.info("最长连续有效段: %d 帧 (%.3f s), R²=%.6f, Cv=%.4f",
                longest_seg_frames, longest_seg_sec, longest_seg_r2, longest_seg_cv)
    if has_quality:
        logger.info("存在满足质量标准的轨迹段！")

    if axis_estimated:
        logger.info("下落中心线: fall_axis_x = %.1f px", detector.get_fall_axis_x())

    # ---- 分段追踪统计 ----
    third = total_count // 3
    logger.info("分段追踪统计 (raw 点识别率):")
    for seg_name, seg_start, seg_end in [
        ("前1/3", 0, third),
        ("中1/3", third, 2 * third),
        ("后1/3", 2 * third, total_count),
    ]:
        seg_mask = valid_mask[seg_start:seg_end]
        seg_valid = int(seg_mask.sum())
        seg_total = seg_end - seg_start
        if seg_total > 0:
            logger.info("  %s: %d/%d (%.1f%%)",
                        seg_name, seg_valid, seg_total,
                        seg_valid / seg_total * 100)
        else:
            logger.info("  %s: 空区间", seg_name)

    # ---- 丢失/重新捕获日志 ----
    if first_lost_frame is not None:
        logger.info("首次追踪中断: 帧 %d (time=%.3fs)", first_lost_frame,
                    df.iloc[first_lost_frame]["time_s"]
                    if first_lost_frame < len(df) else 0)
        lvx, lvy = last_valid_before_loss
        if lvx is not None:
            logger.info("中断前最后有效点: (x=%.0f, y=%.0f)", lvx, lvy)

    if reacquire_count > 0 or reacquire_success > 0:
        logger.info("重新捕获: 尝试 %d 次, 成功 %d 次",
                    reacquire_count, reacquire_success)

    # ---- 拒绝原因统计（从 df 最终状态收集） ----
    reject_reason_counter = {}
    if "rejection" in df.columns and "point_type" in df.columns:
        outlier_mask = df["point_type"] == "outlier"
        invalid_raw = (df["point_type"] == "raw") & ~df["valid"]
        tracked_rej = outlier_mask | invalid_raw
        for _, row in df[tracked_rej].iterrows():
            rej_str = str(row.get("rejection", ""))
            if not rej_str:
                continue
            # 提取主要拒绝种类
            for part in rej_str.split(" | "):
                key = part.split("(")[0].split("=")[0].strip()
                if key:
                    reject_reason_counter[key] = reject_reason_counter.get(key, 0) + 1

    # ---- 统计检测率（统一分母：总帧数，统一数据源：df["point_type"]） ----
    # 逻辑链: raw <= raw+pred (detector_involved) <= raw+pred+interp (valid)
    # 所有计数从 df["point_type"] 推导，保证自洽
    raw_total = len(df)
    raw_cnt = int((df["point_type"] == "raw").sum())
    pred_cnt = int((df["point_type"] == "predicted").sum())
    interp_cnt = int((df["point_type"] == "interpolated").sum())
    outlier_cnt = int((df["point_type"] == "outlier").sum())
    trimmed_cnt = int((df["point_type"] == "trimmed").sum())
    # detector_involved = 检测器参与了输出的帧（raw 或 predicted）
    detector_involved_cnt = raw_cnt + pred_cnt
    valid_real_cnt = raw_cnt
    detector_involved_rate = detector_involved_cnt / raw_total * 100 if raw_total > 0 else 0.0
    valid_real_rate = valid_real_cnt / raw_total * 100 if raw_total > 0 else 0.0
    # 候选检测帧数（BallDetector 找到至少一个候选），来自逐帧 debug 记录
    raw_candidate_cnt = sum(r.get("is_selected", 0) for r in frame_debug_records)
    logger.info(
        "检测率统计（统一分母=总帧数 %d）:\n"
        "  candidate_detection_rate（BallDetector找到候选）: %d/%d = %.1f%%\n"
        "  detector_involved_rate（raw+predicted，检测器参与输出）: %d/%d = %.1f%%\n"
        "  valid_real_point_rate（仅raw，通过motion gate）: %d/%d = %.1f%%\n"
        "  逻辑自洽: valid(%.1f%%) <= detector_involved(%.1f%%)  ✓\n"
        "  类型分布: raw=%d, predicted=%d, interpolated=%d, outlier=%d, trimmed=%d",
        raw_total,
        raw_candidate_cnt, raw_total,
        raw_candidate_cnt / raw_total * 100 if raw_total > 0 else 0.0,
        detector_involved_cnt, raw_total, detector_involved_rate,
        valid_real_cnt, raw_total, valid_real_rate,
        valid_real_rate, detector_involved_rate,
        raw_cnt, pred_cnt, interp_cnt, outlier_cnt, trimmed_cnt,
    )
    if reject_reason_counter:
        sorted_reasons = sorted(reject_reason_counter.items(), key=lambda x: -x[1])
        logger.info("拒绝原因统计（Top 10）:")
        for reason, count in sorted_reasons[:10]:
            logger.info("  %s: %d 次 (%.1f%%)", reason, count,
                        count / max(len(reject_reason_counter), 1) * 100)

    # ---- 保存逐帧检测 debug CSV ----
    output_dir = config.get("output_dir", "data/results")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        debug_csv_path = os.path.join(output_dir, "frame_debug.csv")
        _save_frame_debug_csv(df, records, debug_csv_path)
        cand_csv_path = os.path.join(output_dir, "candidate_debug.csv")
        _save_candidate_debug_csv(candidate_records, cand_csv_path)
        # 增强调试 CSV
        if frame_debug_records:
            enhanced_csv_path = os.path.join(output_dir, "frame_debug_detailed.csv")
            _save_frame_debug_detailed_csv(frame_debug_records, enhanced_csv_path)
        # ★ tracker_debug.csv — 每帧追踪决策明细
        tracker_csv_path = os.path.join(output_dir, "tracker_debug.csv")
        _save_tracker_debug_csv(frame_debug_records, tracker_csv_path)

    return {
        "traj_df": df,
        "valid_frames": valid_count,
        "total_frames": total_count,
        "valid_ratio": ratio,
        "longest_valid_segment_frames": longest_seg_frames,
        "longest_valid_segment_sec": longest_seg_sec,
        "longest_segment_r2": longest_seg_r2,
        "longest_segment_cv": longest_seg_cv,
        "has_quality_segment": has_quality,
        "first_lost_frame": first_lost_frame,
        "last_valid_before_loss": list(last_valid_before_loss) if last_valid_before_loss[0] is not None else None,
        "reacquire_count": reacquire_count,
        "reacquire_success": reacquire_success,
        # ---- 详细统计（统一分母，逻辑自洽） ----
        "raw_candidate_rate": detector_involved_rate,       # 兼容旧字段名
        "selected_detection_rate": detector_involved_rate,  # 兼容旧字段名
        "valid_real_point_rate": valid_real_rate,
        "detector_involved_rate": detector_involved_rate,
        "raw_candidate_count": raw_candidate_cnt,
        "detector_involved_count": detector_involved_cnt,
        "valid_real_count": valid_real_cnt,
        "reject_reason_counts": dict(sorted_reasons) if reject_reason_counter else {},
        "predicted_count": pred_cnt,
        "interpolated_count": interp_cnt,
        "outlier_count": outlier_cnt,
        "trimmed_count": trimmed_cnt,
    }


def _find_valid_segments(valid_mask, max_gap: int = 0) -> list:
    """找出连续 True 段（允许 max_gap 个连续 False 容差），返回 [(start, end), ...] 左闭右开。

    max_gap=0 时行为与原始版一致（不允许多帧间隙）。
    max_gap>0 时允许短间隙不截断轨迹（避免单帧预测/插值打断连续性）。
    """
    n = len(valid_mask)
    if n == 0:
        return []
    if max_gap <= 0:
        # 原始逻辑：严格连续
        segments = []
        in_seg = False
        start = 0
        for i, v in enumerate(valid_mask):
            if v and not in_seg:
                start = i
                in_seg = True
            elif not v and in_seg:
                segments.append((start, i))
                in_seg = False
        if in_seg:
            segments.append((start, n))
        return segments

    # 带容差的版本：允许 max_gap 帧间隙
    segments = []
    seg_start = None
    gap_start = None

    for i in range(n):
        if valid_mask[i]:
            if seg_start is None:
                seg_start = i
            gap_start = None  # 重置间隙计数
        else:
            if seg_start is not None:
                if gap_start is None:
                    gap_start = i
                elif i - gap_start + 1 > max_gap:
                    # 间隙超过容差，截断
                    segments.append((seg_start, gap_start))
                    seg_start = None
                    gap_start = None

    if seg_start is not None:
        if gap_start is not None:
            segments.append((seg_start, gap_start))
        else:
            segments.append((seg_start, n))

    return segments


def _recent_valid_points(records: list, limit: int = 3) -> list:
    """Return recent accepted trajectory points, newest last."""
    pts = []
    for r in reversed(records):
        if not r.get("valid"):
            continue
        if r.get("point_type") not in ("raw", "predicted", "interpolated"):
            continue
        x = r.get("x_px")
        y = r.get("y_px")
        if x is None or y is None or pd.isna(x) or pd.isna(y):
            continue
        pts.append({
            "frame": int(r.get("frame", 0)),
            "x": float(x),
            "y": float(y),
        })
        if len(pts) >= limit:
            break
    return list(reversed(pts))


def _predict_from_recent_points(records: list, frame_idx: int, config: dict):
    """Predict the next ball position from recent accepted trajectory points."""
    history = _recent_valid_points(records, limit=4)
    if len(history) < 2:
        return None

    last = history[-1]
    prev = history[-2]
    dt = max(1, last["frame"] - prev["frame"])
    vx = (last["x"] - prev["x"]) / dt
    vy = (last["y"] - prev["y"]) / dt

    # Falling in image coordinates should not move upward during prediction.
    min_vy = float(config.get("min_predict_vy_px_frame", 0.3))
    max_vy = float(config.get("max_predict_vy_px_frame", config.get("search_win_y_down", 120) / 3.0))
    vy = min(max(vy, min_vy), max_vy)

    max_vx = float(config.get("max_predict_vx_px_frame", config.get("dx_max", 15)))
    vx = float(np.clip(vx, -max_vx, max_vx))

    gap = max(1, frame_idx - last["frame"])
    px = last["x"] + vx * gap
    py = last["y"] + vy * gap
    confidence = max(0.05, 1.0 / (1.0 + gap))
    return px, py, confidence


def _tracking_tolerance_scale(history: list, config: dict) -> float:
    """Relax continuity gates slightly once a stable falling track exists."""
    scale = 1.0
    if len(history) >= 2:
        scale += 0.20
    if len(history) >= 3:
        scale += 0.15
    if bool(config.get("large_ball_mode", False)):
        scale += 0.15
    return float(config.get("tracking_tolerance_scale", scale))


def _can_rescue_motion_candidate(candidate: dict, history: list, config: dict) -> bool:
    """Keep plausible detections when strict continuity gates are slightly too hard."""
    if not history:
        return True

    last = history[-1]
    cx = float(candidate["x_px"])
    cy = float(candidate["y_px"])
    dx = abs(cx - last["x"])
    dy = cy - last["y"]
    dist = float(np.hypot(cx - last["x"], cy - last["y"]))

    # 大间隙：重捕获场景全部放行（motion_gate 已处理）
    frame_gap = int(candidate.get("frame", 0)) - int(last.get("frame", 0))
    if frame_gap > 10:
        return True

    dx_max = float(config.get("dx_max", 15))
    dy_back_tol = float(config.get("dy_back_tol", 3))
    dist_max = float(config.get("dist_max", config.get("max_jump_px", 35)))
    rescue_scale = float(config.get("tracking_rescue_scale", 1.8))
    # 高 diff_score 候选放宽 rescue 门控（真实运动的小球值得更多容忍）
    cand_diff = float(candidate.get("diff_score", 0.0))
    if cand_diff > 15.0:
        rescue_scale = rescue_scale * 1.4

    if dy < -dy_back_tol * 1.2:
        return False
    if dx > dx_max * rescue_scale:
        return False
    if dist > dist_max * rescue_scale:
        return False

    if len(history) >= 2:
        prev = history[-2]
        frame_gap = max(1, last["frame"] - prev["frame"])
        vx = (last["x"] - prev["x"]) / frame_gap
        vy = max(0.0, (last["y"] - prev["y"]) / frame_gap)
        pred_frame_gap = max(1, int(candidate.get("frame", last["frame"] + 1)) - last["frame"])
        pred_x = last["x"] + vx * pred_frame_gap
        pred_y = last["y"] + vy * pred_frame_gap
        pred_error = float(np.hypot(cx - pred_x, cy - pred_y))
        pred_gate = float(config.get(
            "trajectory_prediction_gate_px",
            max(dist_max, 2.5 * max(1.0, np.hypot(vx, vy))),
        ))
        if pred_error > pred_gate * float(config.get("tracking_rescue_pred_scale", 1.6)):
            return False

    return True


def _motion_gate(candidate: dict, history: list, config: dict) -> tuple:
    """Check whether a candidate can belong to the current falling-ball track."""
    if not history:
        return True, "", 0.0

    cx = float(candidate["x_px"])
    cy = float(candidate["y_px"])
    last = history[-1]
    dx = abs(cx - last["x"])
    dy = cy - last["y"]
    dist = float(np.hypot(cx - last["x"], cy - last["y"]))

    # 如果上次有效帧和当前候选帧差距太大（重捕获场景），
    # 说明球已经掉出去很远，用紧的门控只会误杀，全部放行。
    # ★ 注意：x 约束不放松（防止跳到旁边的干扰特征），只放松 y/距离。
    frame_gap = int(candidate.get("frame", 0)) - int(last.get("frame", 0))
    if frame_gap > 10:
        tol_scale = 3.5
        dx_max = float(config.get("dx_max", 15)) * tol_scale
        dy_back_tol = float(config.get("dy_back_tol", 3)) * tol_scale
        dist_max = float(config.get("dist_max", 35)) * tol_scale
        reasons = []
        if dx > dx_max:
            reasons.append(f"dx={dx:.1f}>{dx_max:.1f}")
        if dy < -dy_back_tol:
            reasons.append(f"dy_back={dy:.1f}<-{dy_back_tol:.1f}")
        if dist > dist_max:
            reasons.append(f"dist={dist:.1f}>{dist_max:.1f}")
        return len(reasons) == 0, " | ".join(reasons), dist

    tol_scale = _tracking_tolerance_scale(history, config)
    dx_max = float(config.get("dx_max", 15))
    dy_back_tol = float(config.get("dy_back_tol", 3))
    dist_max = float(config.get("dist_max", config.get("max_jump_px", 35)))
    dx_limit = dx_max * tol_scale
    dist_limit = dist_max * tol_scale

    reasons = []
    if dx > dx_limit:
        reasons.append(f"dx={dx:.1f}>{dx_limit:.1f}")
    if dy < -dy_back_tol:
        reasons.append(f"dy_back={dy:.1f}<-{dy_back_tol:.1f}")
    if dist > dist_limit:
        reasons.append(f"dist={dist:.1f}>{dist_limit:.1f}")

    pred_error = dist
    if len(history) >= 2:
        prev = history[-2]
        frame_gap = max(1, last["frame"] - prev["frame"])
        vx = (last["x"] - prev["x"]) / frame_gap
        vy = max(0.0, (last["y"] - prev["y"]) / frame_gap)
        pred_frame_gap = max(1, int(candidate.get("frame", last["frame"] + 1)) - last["frame"])
        pred_x = last["x"] + vx * pred_frame_gap
        pred_y = last["y"] + vy * pred_frame_gap
        pred_error = float(np.hypot(cx - pred_x, cy - pred_y))
        pred_gate = float(config.get(
            "trajectory_prediction_gate_px",
            max(dist_max, 2.5 * max(1.0, np.hypot(vx, vy))),
        )) * float(config.get("tracking_prediction_gate_scale", max(1.10, tol_scale)))
        if pred_error > pred_gate:
            reasons.append(f"pred_error={pred_error:.1f}>{pred_gate:.1f}")

    return len(reasons) == 0, " | ".join(reasons), pred_error


def _select_motion_consistent_detection(
    result: dict,
    records: list,
    crop_offset_x: int,
    crop_offset_y: int,
    config: dict,
):
    """Pick a candidate that obeys trajectory continuity before accepting it."""
    history = _recent_valid_points(records, limit=3)
    frame_idx = len(records)

    candidates = []
    rescued_candidates = []
    for c in result.get("candidates", []):
        x = float(c.get("cx", np.nan))
        y = float(c.get("cy", np.nan))
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        cand = {
            "frame": frame_idx,
            "x_px": x,
            "y_px": y,
            "radius_px": c.get("radius"),
            "area_px": c.get("area"),
            "circularity": c.get("circularity"),
            "confidence": c.get("score", 0.0),
            "rank": c.get("rank", 999),
            "score": c.get("score", 0.0),
            "diff_score": c.get("diff_score", 0.0),
            "mean_diff": c.get("mean_diff", 0.0),
            "source_method": c.get("source_method", ""),
            "template_match_score": c.get("template_match_score", -1.0),
        }
        # ★ 模板匹配高分 → 信任检测结果，跳过运动门控
        tm_trust = float(config.get("template_trust_threshold", 0.65))
        if cand["template_match_score"] >= tm_trust:
            cand["_tm_trusted"] = True
            candidates.append(cand)
            if frame_idx < 100:
                logger.info("帧 %d: 运动门控被模板信任跳过 (tm=%.3f)", frame_idx, cand["template_match_score"])
            continue
        ok, reason, pred_error = _motion_gate(cand, history, config)
        cand["_motion_reject_reason"] = reason
        cand["_pred_error"] = pred_error
        cand["_tolerance_scale"] = _tracking_tolerance_scale(history, config)
        if ok:
            candidates.append(cand)
        elif _can_rescue_motion_candidate(cand, history, config):
            cand["_rescued"] = True
            rescued_candidates.append(cand)

    if not candidates:
        if rescued_candidates:
            rescued_candidates.sort(
                key=lambda c: (c["_pred_error"], abs(float(c.get("rank", 999))), -float(c.get("score", 0.0)))
            )
            return rescued_candidates[0]
        return None if history else {
            "x_px": result["x_px"],
            "y_px": result["y_px"],
            "radius_px": result.get("radius_px"),
            "area_px": result.get("area_px"),
            "circularity": result.get("circularity"),
            "confidence": result.get("confidence", 0.0),
            "rank": 1,
            "diff_score": 0.0,
            "mean_diff": 0.0,
        }

    # ★ 无历史时：综合 score + diff_score 排序，不单靠 diff_score（刻度线/反光也有高 diff）
    if not history:
        candidates.sort(key=lambda c: (
            -float(c.get("score", 0.0)),
            -float(c.get("diff_score", 0.0)),
        ))
    else:
        last_y = history[-1]["y"]
        for c in candidates:
            c["_dy"] = float(c["y_px"]) - last_y
        candidates.sort(key=lambda c: (
            c["_pred_error"],
            # ★ y 单调：优先选 dy>0（球往下掉）的候选，排除 y 回跳的
            0 if c["_dy"] >= -float(config.get("dy_back_tol", 3)) * 0.5 else 1,
            -float(c.get("diff_score", 0.0)),
            -float(c.get("score", 0.0)),
        ))
    return candidates[0]


def _restore_detector_to_last_good(detector: BallDetector, x: float, y: float,
                                   crop_offset_x: int, crop_offset_y: int):
    """Undo detector state pollution caused by a rejected best candidate."""
    if x is None or y is None or pd.isna(x) or pd.isna(y):
        detector.clear_search_center()
        return
    lx = float(x) - crop_offset_x
    ly = float(y) - crop_offset_y
    detector.set_prev(lx, ly)
    detector.set_search_center(lx, ly)


def _interpolate_short_gaps(df: pd.DataFrame, max_gap_frames: int = 3,
                           scale_m_per_px: float | None = None) -> pd.DataFrame:
    """对 ≤ max_gap_frames 帧的短间隙做线性插值补全。

    插值后的帧 point_type="interpolated"。
    scale_m_per_px: 全局比例尺 (m/px)，传了则直接使用，不反推。
    """
    valid = df["valid"].values
    n = len(df)

    # 找所有有效段
    segs = _find_valid_segments(valid)
    if len(segs) < 2:
        return df  # 最多一段，无需插值

    # 对每对相邻段之间的间隙做插值
    for i in range(len(segs) - 1):
        seg_a_end = segs[i][1]      # 前段结束（开区间）
        seg_b_start = segs[i + 1][0]  # 后段开始
        gap = seg_b_start - seg_a_end

        if gap <= 0 or gap > max_gap_frames:
            continue

        # 间隙前后帧索引
        a_idx = seg_a_end - 1
        b_idx = seg_b_start

        if a_idx < 0 or b_idx >= n:
            continue

        x_a = df.loc[a_idx, "x_px"]
        y_a = df.loc[a_idx, "y_px"]
        x_b = df.loc[b_idx, "x_px"]
        y_b = df.loc[b_idx, "y_px"]

        if pd.isna(x_a) or pd.isna(x_b) or pd.isna(y_a) or pd.isna(y_b):
            continue

        # 线性插值
        for g in range(1, gap + 1):
            t = g / (gap + 1)
            x_interp = x_a + (x_b - x_a) * t
            y_interp = y_a + (y_b - y_a) * t
            idx = a_idx + g

            # ★ 优先使用全局比例尺，不回退反推（反推易受单帧误差污染）
            sc = scale_m_per_px
            if sc is None or sc <= 0:
                # 回退：从已有数据反推比例尺（兼容无全局 scale_m_per_px 的旧调用）
                try:
                    for search_back in range(idx - 1, -1, -1):
                        sx = df.loc[search_back, "x_m"]
                        spx = df.loc[search_back, "x_px"]
                        if pd.notna(sx) and pd.notna(spx) and spx != 0:
                            sc = sx / spx
                            break
                except (ZeroDivisionError, TypeError):
                    pass

            df.loc[idx, "x_px"] = x_interp
            df.loc[idx, "y_px"] = y_interp
            if sc and sc > 0:
                df.loc[idx, "x_m"] = x_interp * sc
                df.loc[idx, "y_m"] = y_interp * sc
            df.loc[idx, "valid"] = True
            df.loc[idx, "point_type"] = "interpolated"

    return df


def _remove_trajectory_outliers(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """后处理：逐帧检查轨迹点，剔除不符合物理约束的离群点。

    约束:
      - x 方向跳变 ≤ dx_max (px)
      - y 方向不回跳 (或 ≤ dy_back_tol)
      - 帧间距离 ≤ dist_max (px)
      - 超出下落中心线/ROI 中心通道的剔除
      - y 单调性检查：在滑动窗口内 y 应持续下降（ball falling），
        不允许出现 y 明显回跳（>dy_back_tol）后又被接受的点。
        此约束防止两个 x 位置来回跳时把上方的错误点也纳入轨迹。

    剔除的点被设 x_px/y_px = NaN, valid = False，不参与后续统计和绘图。
    """
    dx_max = config.get("dx_max", 15)
    dy_back_tol = config.get("dy_back_tol", 3)
    dist_max = config.get("dist_max", 35)
    max_x_drift_px = float(
        config.get("max_x_drift_px",
                   config.get("allowed_axis_deviation_px", 50))
    )

    detect_roi = config.get("detect_roi")
    if detect_roi is not None:
        roi_center = detect_roi[0] + detect_roi[2] / 2.0
        roi_gate = min(
            max_x_drift_px,
            detect_roi[2] * float(config.get("x_gate_ratio", 0.25)),
        )
    else:
        roi_center = None
        roi_gate = None

    x_num = pd.to_numeric(df["x_px"], errors="coerce")
    y_num = pd.to_numeric(df["y_px"], errors="coerce")
    valid_xy = df["valid"] & np.isfinite(x_num) & np.isfinite(y_num)
    raw_xy = valid_xy & (df["point_type"] == "raw")
    if raw_xy.sum() >= 5:
        axis_x = float(np.median(x_num.loc[raw_xy]))
    elif config.get("fall_axis_x") is not None:
        axis_x = float(config["fall_axis_x"])
    else:
        axis_x = roi_center

    idx_all = df[valid_xy].index
    if len(idx_all) < 2:
        return df

    last_good_x = None
    last_good_y = None
    last_good_frame = None
    removed = 0
    reason_counts = {}
    # ★ y 单调滑动窗口：记录最近接受的 y 值，用于趋势检查
    _recent_accepted_ys = []
    _max_trend_window = int(config.get("trend_window_frames", 10))

    for idx in idx_all:
        x = x_num.at[idx]
        y = y_num.at[idx]
        if pd.isna(x) or pd.isna(y):
            continue

        outlier = False
        reasons = []

        # ROI 中心通道约束：先排除明显跑到管壁/标尺附近的点。
        if roi_center is not None and roi_gate is not None and abs(x - roi_center) > roi_gate:
            outlier = True
            reasons.append(f"outside_roi_center(|x-{roi_center:.1f}|>{roi_gate:.1f})")

        # 稳健下落轴约束：用 raw 点中位数作为中心线，偏移过大直接剔除。
        if axis_x is not None and abs(x - axis_x) > max_x_drift_px:
            outlier = True
            reasons.append(f"outside_fall_axis(|x-{axis_x:.1f}|>{max_x_drift_px:.1f})")

        # ★ x 稳定性约束：在滑动窗口内 x 不应大幅跳变，
        #   防止两个特征交替出现时把错误点混入轨迹。
        if not outlier and len(_recent_accepted_ys) >= 5:
            _recent_xs = [
                x_num.at[oidx] for oidx in idx_all
                if oidx < idx and x_num.at[oidx] is not None and not pd.isna(x_num.at[oidx])
            ][-10:]
            if len(_recent_xs) >= 5:
                _x_median = float(np.median(_recent_xs))
                _x_mad = float(np.median([abs(v - _x_median) for v in _recent_xs]))
                # 如果 x 偏离中位数超过 dx_max*1.5 且远超 MAD → 大概率跳到了错误特征
                _x_dev = abs(x - _x_median)
                if _x_dev > max(dx_max * 3.0, _x_mad * 6.0, 30.0):
                    outlier = True
                    reasons.append(f"x_jitter(x={x:.0f},median={_x_median:.0f},dev={_x_dev:.1f})")

        # ★ y 单调趋势约束：在滑动窗口内 y 应有持续下降趋势（ball falling）。
        #   如果当前 y 明显高于窗口内最低 y（爬升），说明可能跳到了上方的错误点。
        if not outlier and len(_recent_accepted_ys) >= 5:
            _min_y = min(_recent_accepted_ys[-5:])
            if y < _min_y - dy_back_tol * 3:
                outlier = True
                reasons.append(f"y_climb(y={y:.0f},min_recent={_min_y:.0f},diff={_min_y - y:.0f})")

        # 帧间跳变 — 大间隙后重置参考点，避免与过期位置比较
        gap_frames = idx - last_good_frame if last_good_frame is not None else 0
        if last_good_x is not None and gap_frames <= 30:
            dx = abs(x - last_good_x)
            dy = y - last_good_y
            dist = np.hypot(x - last_good_x, y - last_good_y)
            if dx > dx_max * 1.25:
                outlier = True
                reasons.append(f"dx={dx:.1f}>{dx_max * 1.25:.1f}")
            if dy < -dy_back_tol:
                outlier = True
                reasons.append(f"dy_back={dy:.1f}")
            if dist > dist_max * 1.25:
                outlier = True
                reasons.append(f"dist={dist:.1f}>{dist_max * 1.25:.1f}")
        elif last_good_x is not None and gap_frames > 30:
            # 大间隙：只检查下落轴约束，不检查帧间跳变
            pass

        if outlier:
            df.at[idx, "x_px"] = np.nan
            df.at[idx, "y_px"] = np.nan
            df.at[idx, "x_m"] = np.nan
            df.at[idx, "y_m"] = np.nan
            df.at[idx, "valid"] = False
            df.at[idx, "point_type"] = "outlier"
            df.at[idx, "radius_px"] = np.nan
            df.at[idx, "area_px"] = np.nan
            df.at[idx, "circularity"] = np.nan
            df.at[idx, "confidence"] = np.nan
            df.at[idx, "rejection"] = " | ".join(reasons)
            for r in reasons:
                key = r.split("(")[0].split("=")[0]
                reason_counts[key] = reason_counts.get(key, 0) + 1
            removed += 1
        else:
            last_good_x = x
            last_good_y = y
            last_good_frame = idx
            _recent_accepted_ys.append(y)
            if len(_recent_accepted_ys) > _max_trend_window:
                _recent_accepted_ys.pop(0)

    if removed > 0:
        logger.info(
            "[后处理] 剔除 %d 个离群点: axis_x=%s, max_x_drift=%.1f, "
            "dx_max=%s, dy_back_tol=%s, dist_max=%s, reasons=%s",
            removed,
            f"{axis_x:.1f}" if axis_x is not None else "None",
            max_x_drift_px, dx_max, dy_back_tol, dist_max, reason_counts,
        )
    return df


def _trim_after_breakaway(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Stop the track at a hard break instead of drawing a fake connection.

    If several consecutive frames after a good track are rejected as impossible
    jumps, the remaining tail is usually another object or scale/reflection.
    Keeping it creates the misleading pink kink shown in the UI.
    """
    consecutive_outliers = int(config.get("breakaway_consecutive_outliers", 2))
    min_raw_before = int(config.get("breakaway_min_raw_before", 8))
    raw_seen = 0
    bad_run = 0
    trim_from = None

    for idx, row in df.iterrows():
        pt = row.get("point_type")
        if pt == "raw" and bool(row.get("valid")):
            raw_seen += 1
            bad_run = 0
            continue

        rejection = str(row.get("rejection", ""))
        hard_bad = (
            pt == "outlier"
            and (
                "dx=" in rejection
                or "dist=" in rejection
                or "pred_error=" in rejection
                or "no_motion_consistent_candidate" in rejection
                or "outside_fall_axis" in rejection
            )
        )
        if raw_seen >= min_raw_before and hard_bad:
            bad_run += 1
            if bad_run >= consecutive_outliers:
                trim_from = idx - consecutive_outliers + 1
                break
        elif pt not in ("predicted", "interpolated"):
            bad_run = 0

    if trim_from is None:
        return df

    mask = df.index >= trim_from
    df.loc[mask, ["x_px", "y_px", "x_m", "y_m", "radius_px",
                  "area_px", "circularity", "confidence"]] = np.nan
    df.loc[mask, "valid"] = False
    df.loc[mask, "point_type"] = "trimmed"
    df.loc[mask, "rejection"] = "track_breakaway_trimmed"
    logger.info("[后处理] 轨迹在帧 %d 后发生断裂，已截断后续尾段", int(trim_from))
    return df


def _save_frame_debug_csv(df: pd.DataFrame, records: list, output_path: str):
    """保存逐帧检测 debug CSV。

    包含字段：frame, time_s, detected, x_px, y_px, radius_px, area_px,
    circularity, confidence, valid, point_type, rejection
    """
    try:
        cols = [
            "frame", "time_s", "x_px", "y_px",
            "radius_px", "area_px", "circularity", "confidence",
            "valid", "point_type", "rejection",
        ]
        # 从 df 提取（含后处理后的 point_type 和 rejection）
        export = df[[c for c in cols if c in df.columns]].copy()
        export["detected"] = export["valid"].astype(int)
        ordered = ["frame", "time_s", "detected"] + [
            c for c in cols if c not in ("frame", "time_s")
        ]
        export = export[[c for c in ordered if c in export.columns]]
        export.to_csv(output_path, index=False, float_format="%.8f", encoding="utf-8")
        logger.info("逐帧 debug CSV 已保存: %s (%d 行)", output_path, len(export))
    except Exception as e:
        logger.warning("保存 frame debug CSV 失败: %s", e)


def _save_candidate_debug_csv(candidate_records: list, output_path: str):
    """保存逐帧候选点 debug CSV。

    包含每帧所有候选点的评分明细，用于分析失败原因。
    """
    try:
        if not candidate_records:
            logger.warning("无候选点记录，跳过 candidate_debug.csv")
            return
        df = pd.DataFrame(candidate_records)
        df.to_csv(output_path, index=False, float_format="%.4f", encoding="utf-8")
        logger.info("候选点 debug CSV 已保存: %s (%d 行)", output_path, len(df))
    except Exception as e:
        logger.warning("保存候选点 debug CSV 失败: %s", e)


def _save_frame_debug_detailed_csv(debug_records: list, output_path: str):
    """保存增强逐帧 debug CSV。

    包含每帧完整检测-过滤链路数据，用于分析检测层-追踪层-终端窗口各层统计口径差异。
    """
    try:
        if not debug_records:
            logger.warning("无增强 debug 记录，跳过 frame_debug_detailed.csv")
            return
        cols = [
            "frame", "time_s",
            "raw_candidate_count", "pre_search_candidates", "search_window_active",
            "selected_x", "selected_y",
            "selected_radius", "selected_confidence",
            "method_used",
            "is_selected", "is_predicted", "is_valid_real", "is_interpolated",
            "frame_state", "tracking_state", "lost_counter",
            "reject_reason",
            "dx_from_prev", "dy_from_prev", "jump_px",
            "instant_velocity_px_s",
            "x_drift_px",
            "y_monotonic_ok",
            "confidence_ok",
            "radius_ok",
        ]
        df = pd.DataFrame(debug_records)
        ordered = [c for c in cols if c in df.columns]
        df[ordered].to_csv(output_path, index=False, float_format="%.6f", encoding="utf-8")
        logger.info("增强逐帧 debug CSV 已保存: %s (%d 行)", output_path, len(df))
    except Exception as e:
        logger.warning("保存增强 frame debug CSV 失败: %s", e)


def _save_tracker_debug_csv(debug_records: list, output_path: str):
    """保存 tracker_debug.csv — 每帧追踪决策明细。

    字段:
      frame                    帧号
      single_frame_found       BallDetector 是否找到候选 (0/1)
      tracking_selected        追踪层是否接受此帧 (0/1)
      point_type               点类型: raw / predicted / interpolated / (空)
      reject_reason            拒绝原因
      dx                       与上一有效点 x 位移 (px)
      dy                       与上一有效点 y 位移 (px)
      dist                     与上一有效点距离 (px)
      diff_score               被选中候选的 diff_score
      state                    追踪状态: TRACKING / LOST / REACQUIRE / STOPPED
    """
    try:
        if not debug_records:
            logger.warning("无 tracker debug 记录，跳过 tracker_debug.csv")
            return
        rows = []
        for d in debug_records:
            # point_type
            pt = ""
            if d.get("is_valid_real"):
                pt = "raw"
            elif d.get("is_predicted"):
                pt = "predicted"
            elif d.get("is_interpolated"):
                pt = "interpolated"

            rows.append({
                "frame": d.get("frame", 0),
                "single_frame_found": d.get("is_selected", 0),
                "tracking_selected": 1 if pt in ("raw", "predicted") else 0,
                "point_type": pt,
                "reject_reason": d.get("reject_reason", ""),
                "dx": d.get("dx_from_prev"),
                "dy": d.get("dy_from_prev"),
                "dist": d.get("jump_px"),
                "diff_score": d.get("selected_diff_score"),
                "state": d.get("tracking_state", ""),
            })
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, float_format="%.3f", encoding="utf-8")
        logger.info("tracker_debug.csv 已保存: %s (%d 行)", output_path, len(df))
    except Exception as e:
        logger.warning("保存 tracker_debug.csv 失败: %s", e)


def detect_track_interval(
    traj_df: pd.DataFrame,
    config: dict,
    roi: list = None,
    debug_callback=None,
) -> dict:
    """从轨迹数据检测最佳有效追踪区间。

    优先级:
      1. 如果 config 中 manual_start_frame > 0 或 manual_end_frame > 0，
         直接返回手动区间 (interval_source = "manual_override")
      2. 否则自动扫描全视频找出所有候选段，选择最长物理合理段
      3. 如果没有候选段满足最低时长要求，返回空结果并提示用户

    质量判定:
      - 段连续性 (display_count) 仅用于 segment boundary 判定，
        允许少量 predicted/interpolated 桥接
      - raw_count / raw_rate / r2(仅raw点) 用于真实质量评估
      - 段时长必须 ≥ ignore_start_s + ignore_end_s + terminal_window_sec
        否则终端速度判定无法进行
    """
    n = len(traj_df)
    if n == 0:
        return _empty_track_result()

    # ---- 手动区间优先 ----
    manual_start = int(config.get("manual_start_frame", 0))
    manual_end = int(config.get("manual_end_frame", 0))

    # ★ 初始位置锚点：用户点击了初始位置 → 自动找到该位置附近的首帧有效检测作为起始帧
    _init_x = config.get("init_ball_x")
    _init_y = config.get("init_ball_y")
    if (_init_x is not None and _init_y is not None
            and manual_start <= 0 and manual_end <= 0):
        _dist_thresh = float(config.get("init_ball_distance_px", 120.0))
        _raw_mask_local = (traj_df["point_type"] == "raw").values
        _raw_idx_arr = np.where(_raw_mask_local)[0]
        _x_vals = pd.to_numeric(traj_df["x_px"], errors="coerce").values
        _y_vals = pd.to_numeric(traj_df["y_px"], errors="coerce").values
        for _idx in _raw_idx_arr:
            if (np.isfinite(_x_vals[_idx]) and np.isfinite(_y_vals[_idx])
                    and abs(_x_vals[_idx] - _init_x) < _dist_thresh
                    and abs(_y_vals[_idx] - _init_y) < _dist_thresh):
                manual_start = int(_idx)
                if debug_callback:
                    debug_callback(
                        f"[INFO] 初位置锚点: 帧 {manual_start} 在 "
                        f"({_init_x:.0f}, {_init_y:.0f}) 附近检测到小球 → 设为起始帧"
                    )
                break
        else:
            if debug_callback:
                debug_callback(
                    f"[WARN] 初位置锚点: 在 ({_init_x:.0f}, {_init_y:.0f}) 附近 "
                    f"{_dist_thresh:.0f}px 内未找到有效检测点，将使用自动区间"
                )

    if manual_start > 0 or manual_end > 0:
        _roi = roi if roi is not None else config.get("roi")
        rx, ry, rw, rh = (float(_roi[0]), float(_roi[1]),
                          float(_roi[2]), float(_roi[3])) if _roi else (0, 0, 9999, 9999)

        if manual_start <= 0:
            manual_start = 0
        if manual_end <= 0:
            manual_end = n - 1

        manual_start = max(0, min(manual_start, n - 1))
        manual_end = max(manual_start, min(manual_end, n - 1))

        seg_df = traj_df.iloc[manual_start:manual_end + 1]
        raw_in_manual = int((seg_df["point_type"] == "raw").values.sum())
        pred_in_manual = int((seg_df["point_type"] == "predicted").values.sum())
        interp_in_manual = int((seg_df["point_type"] == "interpolated").values.sum())

        t_start = float(traj_df.iloc[manual_start]["time_s"])
        t_end = float(traj_df.iloc[manual_end]["time_s"])
        duration = t_end - t_start

        ignore_start = float(config.get("terminal_ignore_start_sec", 0.3))
        ignore_end = float(config.get("terminal_ignore_end_sec", 0.5))
        min_window = float(config.get("terminal_window_sec", 0.8))
        min_req = ignore_start + ignore_end + min_window

        if debug_callback:
            debug_callback(
                f"[INFO] 手动分析区间已设置:\n"
                f"  manual_start_frame = {manual_start}\n"
                f"  manual_end_frame = {manual_end}\n"
                f"  手动区间时长 = {duration:.3f}s\n"
                f"  区间内: raw={raw_in_manual}, predicted={pred_in_manual}, "
                f"interpolated={interp_in_manual}\n"
                f"  最低要求时长 = {min_req:.1f}s (ignore_start={ignore_start}s "
                f"+ ignore_end={ignore_end}s + window={min_window}s)"
            )
            if duration < min_req:
                debug_callback(
                    f"[WARN] 手动区间时长 ({duration:.3f}s) < 最低要求 ({min_req:.1f}s)，"
                    f"终端速度判定可能失败"
                )

        result = _build_track_result(
            traj_df, manual_start, manual_end,
            "手动指定区间",
        )
        result["interval_source"] = "manual_override"
        result["candidates_summary"] = (
            f"手动区间: 帧 {manual_start}~{manual_end}, "
            f"时长 {duration:.3f}s, "
            f"raw={raw_in_manual}, predicted={pred_in_manual}, "
            f"interpolated={interp_in_manual}"
        )
        result["candidates_table"] = []
        result["candidates_table_all"] = []
        return result

    # ---- 自动检测 ----
    # 读取参数
    lost_threshold = int(config.get("lost_threshold", 15))
    outside_threshold = int(config.get("outside_threshold", 10))
    stop_y_ratio = float(config.get("stop_y_ratio", 0.95))
    bottom_margin = float(config.get("bottom_margin_ratio", 0.03))

    # 终端速度相关参数（用于硬性时长下限）
    ignore_start_s = float(config.get("terminal_ignore_start_sec", 0.3))
    ignore_end_s = float(config.get("terminal_ignore_end_sec", 0.5))
    terminal_window_sec = float(config.get("terminal_window_sec", 0.8))
    # 总轨迹时长最低要求（含 ignore 首尾 + 终端窗口）
    min_total_track_duration = ignore_start_s + ignore_end_s + terminal_window_sec

    # 自动段质量门槛
    min_raw_count = int(config.get("min_track_valid_points", 30))
    min_raw_rate = float(config.get("min_track_raw_rate", 0.30))
    min_displacement_px = float(config.get("min_track_displacement_px", 30.0))
    min_r2_raw = float(config.get("quality_r2_threshold", 0.990))

    # ROI
    _roi = roi if roi is not None else config.get("roi")
    if _roi is None or len(_roi) != 4:
        return _track_full_video_fallback(traj_df)

    rx, ry, rw, rh = float(_roi[0]), float(_roi[1]), float(_roi[2]), float(_roi[3])
    # ROI is a detection crop, not the whole physical fall distance.  The ball
    # may leave the drawn ROI while the tracker still follows it correctly, so
    # interval selection must not cut the trajectory at the ROI bottom.
    use_roi_for_interval = bool(config.get("use_roi_for_interval", False))
    stop_y = ry + rh * stop_y_ratio if use_roi_for_interval else np.inf
    bottom_y = ry + rh * (1.0 - bottom_margin) if use_roi_for_interval else np.inf

    x_arr = pd.to_numeric(traj_df["x_px"], errors="coerce").values
    y_arr = pd.to_numeric(traj_df["y_px"], errors="coerce").values
    raw_mask = (traj_df["point_type"] == "raw").values
    pred_mask = (traj_df["point_type"] == "predicted").values
    interp_mask = (traj_df["point_type"] == "interpolated").values

    has_coord = np.isfinite(x_arr) & np.isfinite(y_arr)

    in_roi = np.zeros(n, dtype=bool)
    if has_coord.any() and use_roi_for_interval:
        in_roi[has_coord] = (
            (x_arr[has_coord] >= rx) & (x_arr[has_coord] <= rx + rw) &
            (y_arr[has_coord] >= ry) & (y_arr[has_coord] <= ry + rh)
        )
    elif has_coord.any():
        in_roi[has_coord] = True

    raw_in_roi = raw_mask & in_roi

    # ---- 找所有候选段 ----
    candidates = _find_candidate_segments(
        raw_in_roi, raw_mask, has_coord, in_roi,
        x_arr, y_arr, n, lost_threshold, outside_threshold,
        stop_y, bottom_y,
    )

    if debug_callback:
        if not use_roi_for_interval:
            debug_callback(
                "[DEBUG] ROI 仅限制检测裁剪；轨迹区间使用所有已识别坐标，不再被 ROI 底部截断"
            )
        debug_callback(
            f"[DEBUG] 全视频扫描: 共发现 {len(candidates)} 个候选段"
            f" (基于显示连续性 valid_in_roi = raw+predicted+interpolated)"
        )

    # ---- 详细评估每个候选段 ----
    for c in candidates:
        seg_start, seg_end = c["start_frame"], c["end_frame"]
        seg_len = seg_end - seg_start + 1
        seg_ts = float(traj_df.iloc[seg_start]["time_s"])
        seg_te = float(traj_df.iloc[seg_end]["time_s"])
        seg_dur = seg_te - seg_ts

        # 按类型统计（均在 ROI 内）
        seg_raw = int(raw_in_roi[seg_start:seg_end + 1].sum())
        seg_pred = int((pred_mask & in_roi)[seg_start:seg_end + 1].sum())
        seg_interp = int((interp_mask & in_roi)[seg_start:seg_end + 1].sum())
        seg_display = seg_raw + seg_pred + seg_interp
        raw_rate = seg_raw / seg_len if seg_len > 0 else 0.0

        # y 位移（仅 raw 点）
        ys = y_arr[seg_start:seg_end + 1]
        raw_seg = raw_in_roi[seg_start:seg_end + 1]
        raw_ys = ys[raw_seg]
        y_disp = abs(float(raw_ys[-1] - raw_ys[0])) if len(raw_ys) >= 2 else 0.0

        # x 标准差（仅 raw 点）
        xs = x_arr[seg_start:seg_end + 1]
        raw_xs = xs[raw_seg]
        x_std = float(np.std(raw_xs)) if len(raw_xs) >= 3 else 0.0

        # y-t 拟合 R²（仅 raw 点）
        seg_df = traj_df.iloc[seg_start:seg_end + 1]
        r2, _cv = _assess_segment_quality(seg_df)
        r2 = r2 if r2 is not None else 0.0

        # 可用时长（排除 ignore 首尾后）
        usable_after_ignore = seg_dur - ignore_start_s - ignore_end_s

        # ---- 拒绝判定 ----
        reject_reasons = []
        # 检查1: 总轨迹时长（含 ignore 首尾 + 终端窗口）是否足够
        if seg_dur < min_total_track_duration:
            reject_reasons.append(
                f"总轨迹时长不足 ({seg_dur:.3f}s < {min_total_track_duration:.1f}s"
                f" = ignore_start={ignore_start_s}s + ignore_end={ignore_end_s}s"
                f" + window={terminal_window_sec}s)"
            )
        # 检查2: 排除 ignore 首尾后的可用时长是否仍 ≥ 终端窗口
        elif usable_after_ignore < terminal_window_sec:
            reject_reasons.append(
                f"排除起止段后可用时长不足 ({usable_after_ignore:.3f}s"
                f" < window={terminal_window_sec}s)"
            )
        if seg_raw < min_raw_count:
            reject_reasons.append(f"raw点数不足 ({seg_raw} < {min_raw_count})")
        if raw_rate < min_raw_rate:
            reject_reasons.append(f"raw识别率过低 ({raw_rate:.1%} < {min_raw_rate:.0%})")
        if y_disp < min_displacement_px:
            reject_reasons.append(f"位移过小 ({y_disp:.0f}px < {min_displacement_px:.0f}px)")
        if r2 < min_r2_raw and seg_raw >= 10:
            # R² 不足 → 降级为警告，不阻塞分析
            c["_r2_warning"] = f"R²(仅raw)偏低 ({r2:.4f} < {min_r2_raw})"

        # 就地丰富 c
        c["duration_s"] = seg_dur
        c["display_count"] = seg_display
        c["raw_count"] = seg_raw
        c["predicted_count"] = seg_pred
        c["interpolated_count"] = seg_interp
        c["raw_rate"] = raw_rate
        c["y_displacement_px"] = y_disp
        c["x_std_px"] = x_std
        c["r2"] = r2
        c["usable_after_ignore_s"] = usable_after_ignore
        c["termination_reason"] = c.get("termination_reason", "")
        c["rejected"] = len(reject_reasons) > 0
        c["reject_reasons"] = reject_reasons

        if debug_callback:
            status = " [拒绝]" if c["rejected"] else " [通过]"
            debug_callback(
                f"  段 帧{seg_start}~{seg_end}: "
                f"时长={seg_dur:.3f}s, "
                f"display={seg_display}, raw={seg_raw}, pred={seg_pred}, "
                f"interp={seg_interp}, "
                f"raw_rate={raw_rate:.1%}, "
                f"位移={y_disp:.0f}px, "
                f"x_std={x_std:.1f}px, "
                f"R²={r2:.4f}, "
                f"可用={usable_after_ignore:.3f}s, "
                f"终止: {c['termination_reason']}"
                f"{status}"
            )
            if reject_reasons:
                for rr in reject_reasons:
                    debug_callback(f"    -> 拒绝: {rr}")

    valid_candidates = [c for c in candidates if not c["rejected"]]

    # ---- 选择最佳段 ----
    if not valid_candidates:
        # 无合格段 → 不静默回退，明确告知用户
        if debug_callback:
            debug_callback(
                f"[INFO] 自动检测失败: {len(candidates)} 个候选段均不满足质量要求\n"
                f"  最低要求: 时长≥{min_total_track_duration:.1f}s, "
                f"raw≥{min_raw_count}, raw_rate≥{min_raw_rate:.0%}, "
                f"位移≥{min_displacement_px:.0f}px, R²≥{min_r2_raw}"
            )
            if candidates:
                longest = max(candidates, key=lambda c: c["duration_s"])
                rej_list = longest.get("reject_reasons", [])
                rej_str = ""
                if rej_list:
                    rej_str = "; " + "; ".join(
                        rej_list if isinstance(rej_list, list) else [rej_list]
                    )
                debug_callback(
                    f"  最长候选段: {longest['duration_s']:.3f}s"
                    f" (需≥{min_total_track_duration:.1f}s 总轨迹时长)"
                    f"{rej_str}\n"
                    f"  → 建议: 设置 manual_start_frame / manual_end_frame 手动指定分析区间，\n"
                    f"    或检查 ROI 范围、检测参数以提升轨迹识别质量"
                )
            else:
                debug_callback(
                    f"  → 未发现任何候选段，请检查 ROI 设置和检测参数"
                )

        result = _empty_track_result()
        result["candidates_summary"] = _build_candidates_summary(candidates, valid_candidates)
        result["candidates_table"] = []
        result["candidates_table_all"] = [dict(c) for c in candidates]
        if candidates:
            longest = max(candidates, key=lambda c: c["duration_s"])
            rej_list = longest.get("reject_reasons", [])
            rej_str = ""
            if rej_list:
                rej_str = "; 具体原因: " + "; ".join(
                    rej_list if isinstance(rej_list, list) else [rej_list]
                )
            result["termination_reason"] = (
                f"自动段均不满足质量要求: "
                f"最长段={longest['duration_s']:.3f}s"
                f"(需≥{min_total_track_duration:.1f}s 总轨迹时长"
                f" = ignore_start+ignore_end+window)"
                f"{rej_str}"
            )
        return result

    # 评分: 时长(35%) + raw点数(30%) + raw_rate(15%) + R²(10%) + 位移(5%) + x稳定(5%)
    if len(valid_candidates) > 1:
        max_dur = max(c["duration_s"] for c in valid_candidates) or 0.001
        max_raw = max(c["raw_count"] for c in valid_candidates) or 1
        max_rate = max(c["raw_rate"] for c in valid_candidates) or 0.001
        max_r2 = max(c["r2"] for c in valid_candidates) or 0.001
        max_disp = max(c["y_displacement_px"] for c in valid_candidates) or 1.0
        max_xstd = max(c["x_std_px"] for c in valid_candidates) or 1.0
        min_xstd = min(c["x_std_px"] for c in valid_candidates) or 0.0

        for c in valid_candidates:
            dur_score = c["duration_s"] / max_dur * 35
            raw_score = c["raw_count"] / max_raw * 30
            rate_score = c["raw_rate"] / max_rate * 15
            r2_score = c["r2"] / max_r2 * 10
            disp_score = c["y_displacement_px"] / max_disp * 5
            x_range = max_xstd - min_xstd if max_xstd > min_xstd else 1.0
            x_score = (1.0 - (c["x_std_px"] - min_xstd) / x_range) * 5
            c["score"] = dur_score + raw_score + rate_score + r2_score + disp_score + x_score

        best = max(valid_candidates, key=lambda c: c["score"])
    else:
        best = valid_candidates[0]
        best["score"] = 100.0

    if debug_callback:
        debug_callback(
            f"[INFO] 选中最佳自动段: 帧 {best['start_frame']}~{best['end_frame']}, "
            f"时长 {best['duration_s']:.3f}s, "
            f"raw={best['raw_count']}, raw_rate={best['raw_rate']:.1%}, "
            f"R²={best['r2']:.4f}, "
            f"得分 {best.get('score', 0):.1f}"
        )

    result = _build_track_result(
        traj_df, best["start_frame"], best["end_frame"],
        best.get("termination_reason", "自动检测"),
    )
    result["interval_source"] = "auto_track"
    result["candidates_summary"] = _build_candidates_summary(candidates, valid_candidates)
    result["candidates_table"] = [dict(c) for c in valid_candidates]
    result["candidates_table_all"] = [dict(c) for c in candidates]
    return result


def _track_full_video_fallback(traj_df) -> dict:
    """无 ROI 时的回退：使用全视频首末 raw 帧。"""
    raw_mask = (traj_df["point_type"] == "raw").values
    raw_frames = np.where(raw_mask)[0]
    if len(raw_frames) == 0:
        return _empty_track_result()
    return _build_track_result(
        traj_df, int(raw_frames[0]), int(raw_frames[-1]),
        "无ROI（全视频首末raw帧）",
    )


def _find_candidate_segments(
    raw_in_roi, raw_mask, has_coord, in_roi,
    x_arr, y_arr, n, lost_threshold, outside_threshold,
    stop_y, bottom_y,
) -> list:
    """扫描全视频，找出所有候选轨迹段。

    规则：
      - 段由 valid_in_roi 帧组成（has_coord & in_roi，含 raw/predicted/interpolated）
        → 仅用于判断"轨迹是否连续显示"，不等于真实识别质量
      - 允许 ≤ lost_threshold 帧的连续完全丢失（无任何坐标）
      - 超出间隙上限时结束当前段，继续扫描后续帧形成新候选段
      - 当出现连续 outside_threshold 帧在 ROI 外时结束当前段
      - 超过 stop_y / bottom_y 时结束当前段
      - raw_count 用于质量评估，不影响段边界判定
    """
    valid_in_roi = has_coord & in_roi

    candidates = []
    seg_start = None
    seg_end = None
    seg_raw_count = 0
    seg_display_count = 0  # valid_in_roi 帧数（显示用连续性）
    gap_count = 0
    outside_count = 0

    for i in range(n):
        if valid_in_roi[i]:
            if seg_start is None:
                seg_start = i
                seg_raw_count = 0
                seg_display_count = 0
            seg_end = i
            seg_display_count += 1
            if raw_in_roi[i]:
                seg_raw_count += 1
            gap_count = 0
            outside_count = 0
        else:
            if seg_start is None:
                continue

            gap_count += 1

            if has_coord[i] and not in_roi[i]:
                outside_count += 1

            if outside_count >= outside_threshold:
                candidates.append({
                    "start_frame": seg_start, "end_frame": seg_end,
                    "termination_reason": f"离开ROI(连续{outside_count}帧在ROI外)",
                    "raw_count": seg_raw_count,
                    "display_count": seg_display_count,
                })
                seg_start = seg_end = None
                seg_raw_count = seg_display_count = 0
                gap_count = outside_count = 0
                continue

            if has_coord[i] and not np.isnan(y_arr[i]) and y_arr[i] > stop_y:
                candidates.append({
                    "start_frame": seg_start, "end_frame": seg_end,
                    "termination_reason": f"超过终止线(y={y_arr[i]:.0f})",
                    "raw_count": seg_raw_count,
                    "display_count": seg_display_count,
                })
                seg_start = seg_end = None
                seg_raw_count = seg_display_count = 0
                gap_count = outside_count = 0
                continue

            if has_coord[i] and not np.isnan(y_arr[i]) and y_arr[i] > bottom_y:
                candidates.append({
                    "start_frame": seg_start, "end_frame": seg_end,
                    "termination_reason": f"接近底部干扰区(y={y_arr[i]:.0f})",
                    "raw_count": seg_raw_count,
                    "display_count": seg_display_count,
                })
                seg_start = seg_end = None
                seg_raw_count = seg_display_count = 0
                gap_count = outside_count = 0
                continue

            if gap_count > lost_threshold:
                candidates.append({
                    "start_frame": seg_start, "end_frame": seg_end,
                    "termination_reason": f"连续丢失超过{lost_threshold}帧",
                    "raw_count": seg_raw_count,
                    "display_count": seg_display_count,
                })
                seg_start = seg_end = None
                seg_raw_count = seg_display_count = 0
                gap_count = outside_count = 0
                continue

    if seg_start is not None and seg_end is not None:
        candidates.append({
            "start_frame": seg_start, "end_frame": seg_end,
            "termination_reason": "到达视频末帧",
            "raw_count": seg_raw_count,
            "display_count": seg_display_count,
        })

    return candidates


def _empty_track_result() -> dict:
    return {
        "track_start_frame": 0,
        "track_end_frame": 0,
        "track_start_time_s": 0.0,
        "track_end_time_s": 0.0,
        "track_total_frames": 0,
        "track_duration_s": 0.0,
        "valid_frames_in_track": 0,
        "termination_reason": "无有效轨迹",
        "interval_source": "auto_track",
        "candidates_summary": "",
        "success": False,          # ★ 明确失败标志
    }


def _build_candidates_summary(all_candidates: list, valid_candidates: list) -> str:
    """构建候选段摘要文本，用于日志和结果显示。

    每个段显示:
      duration_s, display_count, raw_count, predicted_count,
      interpolated_count, raw_rate, y_displacement_px, x_std_px,
      r2 (仅raw点), reject_reason
    """
    valid_keys = {(c["start_frame"], c["end_frame"]) for c in valid_candidates}
    lines = [
        f"全视频共发现 {len(all_candidates)} 个候选轨迹段"
        f" (基于显示连续性=raw+predicted+interpolated)，"
        f"其中 {len(valid_candidates)} 个通过质量筛选："
    ]
    for i, c in enumerate(all_candidates):
        is_valid = (c["start_frame"], c["end_frame"]) in valid_keys
        tag = "[通过]" if is_valid else "[拒绝]"
        lines.append(
            f"  段{i + 1} {tag}: 帧 {c['start_frame']}~{c['end_frame']}, "
            f"时长={c.get('duration_s', 0):.3f}s, "
            f"display={c.get('display_count', c.get('raw_count', 0))}, "
            f"raw={c.get('raw_count', 0)}, "
            f"pred={c.get('predicted_count', 0)}, "
            f"interp={c.get('interpolated_count', 0)}, "
            f"raw_rate={c.get('raw_rate', 0):.1%}, "
            f"位移={c.get('y_displacement_px', 0):.0f}px, "
            f"x_std={c.get('x_std_px', 0):.1f}px, "
            f"R²(仅raw)={c.get('r2', 0):.4f}, "
            f"可用={c.get('usable_after_ignore_s', 0):.3f}s, "
            f"终止: {c.get('termination_reason', c.get('termination', ''))}"
        )
        if c.get("reject_reasons"):
            for rr in c["reject_reasons"]:
                lines.append(f"        {rr}")
    return "\n".join(lines)


def _build_track_result(
    traj_df, start: int, end: int, reason: str
) -> dict:
    n = len(traj_df)
    start = max(0, min(start, n - 1))
    end = max(start, min(end, n - 1))
    t_start = float(traj_df.iloc[start]["time_s"])
    t_end = float(traj_df.iloc[end]["time_s"])
    total_frames = end - start + 1
    duration = t_end - t_start
    raw_in_track = int(
        ((traj_df["point_type"] == "raw").values)[start:end + 1].sum()
    )
    return {
        "track_start_frame": start,
        "track_end_frame": end,
        "track_start_time_s": t_start,
        "track_end_time_s": t_end,
        "track_total_frames": total_frames,
        "track_duration_s": duration,
        "valid_frames_in_track": raw_in_track,
        "termination_reason": reason,
        "interval_source": "auto_track",
        "success": True,            # ★ 明确成功标志
    }


def _assess_segment_quality(seg_df: pd.DataFrame) -> tuple:
    """评估一个轨迹段的 y-t 线性拟合 R² 和速度 Cv。

    仅使用 "raw" 类型点进行拟合，确保质量评估反映真实检测质量。

    Returns:
        (r2, cv): (float or None, float or None)
    """
    # ★ 只使用 raw 点评估质量
    raw_mask = (seg_df["point_type"] == "raw").values
    if raw_mask.sum() < 10:
        return None, None

    y = seg_df["y_m"].values
    t = seg_df["time_s"].values

    valid_both = raw_mask & np.isfinite(y) & np.isfinite(t)
    if valid_both.sum() < 10:
        return None, None

    y_valid = y[valid_both]
    t_valid = t[valid_both]

    # 线性拟合 y = a*t + b
    A = np.vstack([t_valid, np.ones_like(t_valid)]).T
    try:
        slope, intercept = np.linalg.lstsq(A, y_valid, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None, None

    # R²
    y_pred = slope * t_valid + intercept
    ss_res = np.sum((y_valid - y_pred) ** 2)
    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Cv: 从 y-t 斜率≈速度，计算窗口内速度变异系数
    # 用一阶差分估算瞬时速度
    if len(t_valid) >= 3:
        v = np.diff(y_valid) / np.diff(t_valid)
        v_mean = np.mean(v)
        if v_mean > 0:
            v_std = np.std(v, ddof=1)
            cv = v_std / v_mean
        else:
            cv = 1.0
    else:
        cv = 1.0

    return r2, cv
