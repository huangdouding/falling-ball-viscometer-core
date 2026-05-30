"""
主窗口

三段式布局：
  左侧：视频预览
  右侧：参数设置与操作按钮
  下方：结果展示（Tab 页）
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QMessageBox, QFileDialog, QApplication,
)
from PySide6.QtCore import Qt, QTimer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.gui.video_widget import VideoWidget
from src.gui.parameter_panel import ParameterPanel
from src.gui.result_tabs import ResultTabs
from src.gui.worker import AnalysisWorker
from src.gui.dialogs import ScaleCalibrationDialog, LoadConfigDialog, BallCalibrateDialog
from src.utils import load_config
from src.ball_detector import BallDetector
from src.viscosity import compute_viscosity


class MainWindow(QMainWindow):
    """应用程序主窗口。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("落球法液体黏度自动分析系统")
        self.setMinimumSize(1280, 800)
        self.resize(1400, 900)

        # 状态
        self._worker = None
        self._analysis_result = None
        self._media_type = None      # "image" | "video" | None
        self._video_path = ""
        self._image_path = ""
        self._traj_df = None
        self._velocity_df = None
        self._terminal_region = {"found": False}
        self._viscosity_result = None
        self._output_dir = "data/results"
        self._init_ball_pos = None   # 用户手动点击的初始位置 (x, y)

        # 实时绘图（分析过程中增量更新）
        self._live_traj_points = []   # [(t_s, y_mm), ...]
        self._live_scale_mm_px = 1.0
        self._live_update_timer = QTimer(self)
        self._live_update_timer.setInterval(200)  # 200ms
        self._live_update_timer.timeout.connect(self._update_live_plot)

        # 中央 widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)

        # 使用两个 QSplitter 实现三段式
        # 上: video | params
        # 下: results
        top_splitter = QSplitter(Qt.Horizontal)
        bottom_splitter = QSplitter(Qt.Vertical)

        # 左侧视频
        self._video_widget = VideoWidget()
        top_splitter.addWidget(self._video_widget)

        # 右侧参数
        self._param_panel = ParameterPanel()
        top_splitter.addWidget(self._param_panel)
        top_splitter.setStretchFactor(0, 6)  # 视频 60%
        top_splitter.setStretchFactor(1, 4)  # 参数 40%

        # 下方结果
        self._result_tabs = ResultTabs()
        bottom_splitter.addWidget(top_splitter)
        bottom_splitter.addWidget(self._result_tabs)
        bottom_splitter.setStretchFactor(0, 7)  # 上 70%
        bottom_splitter.setStretchFactor(1, 3)  # 下 30%

        main_layout.addWidget(bottom_splitter)

        # 连接信号
        self._connect_signals()

        # 默认加载 config.yaml
        self._load_default_config()
        # 加载持久化 settings.json（覆盖 yaml 中的 UI 参数）
        self._load_persisted_settings()

    def _connect_signals(self):
        """连接各组件信号。"""
        # 参数面板按钮
        self._param_panel.open_video_requested.connect(self._on_open_video)
        self._param_panel.open_image_requested.connect(self._on_open_image)
        self._param_panel.test_frame_requested.connect(self._on_test_frame)
        self._param_panel.analyze_requested.connect(self._on_analyze)
        self._param_panel.stop_requested.connect(self._on_stop_analysis)
        self._param_panel.export_requested.connect(self._on_export)
        self._param_panel.clear_requested.connect(self._on_clear)

        # 视频控件
        self._video_widget.roi_changed.connect(self._on_roi_changed)
        self._video_widget.points_selected.connect(self._on_calibration_points)
        self._video_widget.ball_position_set.connect(self._on_ball_position_set)
        self._video_widget.ball_calibrate_requested.connect(self._on_ball_calibrate)
        self._video_widget.video_loaded.connect(self._on_video_loaded)

        # 参数面板信号
        self._param_panel.config_saved.connect(self._on_config_saved)
        self._param_panel.profile_changed.connect(self._on_profile_changed)

    def _load_default_config(self):
        """尝试加载 config.yaml 默认值。"""
        from src.utils import normalize_config_keys
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config.yaml")
        cfg_path = os.path.normpath(cfg_path)
        if os.path.exists(cfg_path):
            try:
                cfg = load_config(cfg_path)
                cfg = normalize_config_keys(cfg)
                self._param_panel.set_config(cfg)
                self._result_tabs.append_log(f"[INFO] 已加载默认配置: {cfg_path}")
            except Exception as e:
                self._result_tabs.append_log(f"[WARN] 加载默认配置失败: {e}")
        else:
            self._result_tabs.append_log(f"[INFO] 无 config.yaml，使用默认参数")

    def _load_persisted_settings(self):
        """从 config/settings.json 加载持久化参数。"""
        from src.gui.parameter_panel import _SETTINGS_PATH
        # ★ 记录 config.yaml 比例尺（直接从 spinner 读，避免 get_config() 触发动参计算）
        yaml_scale = self._param_panel._scale.value()
        loaded = self._param_panel.load_settings()
        if loaded:
            json_scale = self._param_panel._scale.value()
            if yaml_scale > 0 and json_scale > 0 and abs(yaml_scale - json_scale) > 1e-6:
                self._result_tabs.append_log(
                    f"[INFO] 比例尺已从 config.yaml 的 {yaml_scale:.6f} "
                    f"更新为 settings.json 的 {json_scale:.6f} mm/px"
                )
            self._result_tabs.append_log(f"[INFO] 已加载持久化参数: {_SETTINGS_PATH}")
        else:
            self._result_tabs.append_log(f"[INFO] 首次启动，保存当前参数到 {_SETTINGS_PATH}")
            self._param_panel.save_settings()

    # ================================================================
    #  统一文件加载入口
    # ================================================================
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".m4v"}

    def _reset_media_state(self):
        """打开新文件前统一清除旧状态。"""
        self._media_type = None
        self._image_path = ""
        self._video_path = ""
        self._current_roi = None
        self._traj_df = None
        self._velocity_df = None
        self._terminal_region = {"found": False}
        self._viscosity_result = None
        self._analysis_result = None
        self._init_ball_pos = None
        self._video_widget._ball_pos = None
        self._video_widget.close_video()
        self._video_widget.clear_debug_images()
        self._video_widget.clear_background()

    def load_media(self, path: str):
        """统一文件加载入口（图片/视频）。根据后缀自动判断类型。"""
        if not os.path.exists(path):
            QMessageBox.warning(self, "打开失败",
                               "文件不存在，请重新选择。")
            return

        ext = os.path.splitext(path)[1].lower()
        if ext in self._IMAGE_EXTS:
            self._do_open_image(path)
        elif ext in self._VIDEO_EXTS:
            self._do_open_video(path)
        else:
            QMessageBox.warning(self, "不支持的文件格式",
                               f"不支持的文件格式: {ext}\n"
                               "支持的图片格式: {', '.join(sorted(self._IMAGE_EXTS))}\n"
                               "支持的视频格式: {', '.join(sorted(self._VIDEO_EXTS))}")

    # ---- 视频操作 ----
    def _on_open_video(self):
        """打开视频文件对话框。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.mov *.avi *.mkv *.wmv *.m4v);;所有文件 (*)"
        )
        if path:
            self.load_media(path)

    def _do_open_video(self, path: str):
        """加载视频文件。"""
        self._reset_media_state()

        if not self._video_widget.load_video(path):
            QMessageBox.warning(self, "打开失败",
                               "无法打开视频文件，请检查文件是否损坏。")
            return

        self._media_type = "video"
        self._video_path = path
        self._result_tabs.append_log(f"[INFO] 已读取视频: {os.path.basename(path)}")
        self._result_tabs.append_log(
            f"[INFO] 分辨率: {self._video_widget._video_w}x{self._video_widget._video_h}"
            f"  {self._video_widget._fps:.1f}fps"
            f"  {self._video_widget._total_frames}帧"
        )

        saved_roi = self._param_panel.get_config().get("roi")
        if saved_roi and len(saved_roi) == 4:
            x, y, w, h = [int(v) for v in saved_roi]
            if x >= 0 and y >= 0 and w > 10 and h > 10 \
                    and x + w <= self._video_widget._video_w \
                    and y + h <= self._video_widget._video_h:
                self._video_widget.set_roi([x, y, w, h], emit_signal=False)
                self._current_roi = [x, y, w, h]
                self._result_tabs.append_log(
                    f"[CONFIG] ROI source=saved_config value={[x, y, w, h]}"
                )
            else:
                self._result_tabs.append_log(
                    f"[WARN] saved_config ROI {saved_roi} exceeds this video size; ROI ignored"
                )

        # 从参数面板同步 FPS 到视频控件（覆盖 OpenCV 误读值）
        from src.utils import normalize_config_keys
        cfg = normalize_config_keys(self._param_panel.get_config())
        manual_fps = cfg.get("manual_fps")
        if manual_fps is not None and manual_fps > 0:
            self._video_widget.set_fps(float(manual_fps))
            self._result_tabs.append_log(
                f"[CONFIG] fps source=saved_config/manual_override value={manual_fps} fps"
                f" (覆盖 OpenCV 读取的 {self._video_widget._fps:.1f} fps)"
            )
        self._result_tabs.append_log(
            "[CONFIG] physical parameters source=saved_config/manual_override; "
            "video metadata did not overwrite density/diameter/scale"
        )

    def _on_open_image(self):
        """打开图片文件对话框。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片文件", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;所有文件 (*)"
        )
        if path:
            self.load_media(path)

    def _do_open_image(self, path: str):
        """加载图片并显示当前帧（不自动检测）。"""
        self._reset_media_state()

        if not self._video_widget.load_image(path):
            QMessageBox.warning(self, "打开失败", "无法加载图片文件。")
            return

        self._media_type = "image"
        self._image_path = path
        self._result_tabs.append_log(f"[INFO] 已加载图片: {os.path.basename(path)}")
        self._result_tabs.append_log(
            f"[INFO] 分辨率: {self._video_widget._video_w}x{self._video_widget._video_h}"
        )
        self._result_tabs.append_log("[INFO] 当前输入：图片  仅支持单帧检测")
        self._result_tabs.append_log("[INFO] 请框选 ROI 后，点击「测试当前帧识别」检测小球。")

    def _on_video_loaded(self, path: str):
        """视频控件完成加载后更新状态（来自拖拽等非对话框入口）。"""
        if self._media_type != "video":
            # 拖拽进入的视频，状态尚未设置
            self._media_type = "video"
            self._video_path = path
            self._image_path = ""
        self._result_tabs.append_log(
            f"[INFO] 当前输入：视频  "
            f"{self._video_widget._video_w}x{self._video_widget._video_h}"
            f"  {self._video_widget._fps:.1f}fps"
            f"  {self._video_widget._total_frames}帧"
        )

    def _on_roi_changed(self, roi):
        """ROI 变更时清空旧识别结果，提示用户重新检测。"""
        if roi is not None:
            self._current_roi = list(roi)
            self._param_panel.set_runtime_config({"roi": list(roi)})
            self._result_tabs.append_log(f"[INFO] ROI 已设置: {roi}")
        else:
            self._current_roi = None
            self._param_panel.set_runtime_config({"roi": None})
            self._result_tabs.append_log("[INFO] ROI 已清除")

        # 清空旧检测结果（ROI 变了，旧结果不再有效）
        self._video_widget.set_detection(None)
        self._video_widget.set_debug_images({})
        # 清空基于旧 ROI 构建的背景模型
        self._video_widget.clear_background()
        self._result_tabs.append_log("[INFO] ROI 已更新，请点击「测试当前帧识别」重新检测。")

    def _on_calibration_points(self, p1, p2):
        """比例尺标定点已选。"""
        px_dist = abs(p2[1] - p1[1])  # 仅垂直距离（刻度在竖直方向）
        dialog = ScaleCalibrationDialog(p1, p2, px_dist, self)
        if dialog.exec() == ScaleCalibrationDialog.Accepted:
            scale = dialog.get_scale()
            if scale > 0:
                self._param_panel.set_config({"scale_mm_per_px": scale})
                # ★ 立即落盘，防止用户在 autosave 触发前切档位导致标定丢失
                self._param_panel.save_settings()
                self._result_tabs.append_log(
                    f"[INFO] 比例尺标定完成\n"
                    f"  第一点: ({p1[0]}, {p1[1]})\n"
                    f"  第二点: ({p2[0]}, {p2[1]})\n"
                    f"  垂直像素距离: {px_dist:.1f} px\n"
                    f"  比例尺: {scale:.6f} mm/px"
                )

    def _on_ball_calibrate(self):
        """小球自标定：只在 ROI 内逐帧检测，质量过滤后取稳定均值。

        1. 从当前帧开始向后扫描，直到小球离开 ROI 底部
        2. 每帧检测 + 质量验证（确保是小球不是噪点）
        3. 迭代剔除离群值，取最稳定的平均值
        """
        import cv2

        roi = self._video_widget.get_roi()
        if roi is None:
            QMessageBox.warning(self, "提示",
                                "请先用「框选 ROI」圈出小球下落区域，\n"
                                "再点击小球标定。")
            return

        roi_x, roi_y, roi_w, roi_h = [int(v) for v in roi]

        if self._media_type == "image":
            frames_to_scan = [(self._video_widget._current_frame_idx,
                               self._video_widget.get_current_frame_bgr())]
        elif self._video_widget._cap is not None:
            cap = self._video_widget._cap
            current = self._video_widget._current_frame_idx
            total = self._video_widget._total_frames
            # 从当前帧往后读，最多 200 帧（覆盖典型下落过程）
            max_scan = min(total - current, 200)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current)
            frames_to_scan = []
            for _ in range(max_scan):
                ret, frame = cap.read()
                if not ret:
                    break
                fi = current + len(frames_to_scan)
                frames_to_scan.append((fi, frame))
            # 恢复位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, current)
            cap.read()
        else:
            QMessageBox.warning(self, "提示", "请先加载视频或图片。")
            return

        if not frames_to_scan:
            QMessageBox.warning(self, "提示", "无法读取视频帧。")
            return

        cfg = self._param_panel.get_config()
        cfg["roi"] = roi
        cfg["detect_roi"] = list(roi)
        cfg["image_mode"] = (self._media_type == "image")

        # ★ 标定时不依赖当前比例尺（当前比例尺可能本身就是错的）。
        #    使用极宽的半径/面积范围，让检测器自己找最优候选。
        cfg["auto_size_params"] = False
        cfg["expected_radius_px_min"] = 1.0
        cfg["expected_radius_px_max"] = 30.0
        cfg["min_area_px"] = 2
        cfg["max_area_px"] = 3000
        cfg["min_circularity"] = 0.30

        # 构建背景模型
        background = None
        detect_method = cfg.get("detect_method", "auto")
        if not cfg["image_mode"] and detect_method in ("auto", "background_subtraction"):
            try:
                from src.video_io import VideoReader
                reader = VideoReader(self._video_path)
                from src.tracking import _build_background
                bg_roi = cfg.get("detect_roi") or cfg.get("roi")
                background = _build_background(reader, n_frames=50, roi=bg_roi)
                reader.reset()
            except Exception:
                pass

        # 小球直径
        ball_diam = None
        radius_mm = cfg.get("ball_radius_mm")
        if radius_mm and radius_mm > 0:
            ball_diam = radius_mm * 2.0
        if ball_diam is None:
            ball_diam = 1.5

        # ★ 逐帧检测，信任单帧识别逻辑（不额外加质量过滤）
        detector = BallDetector(cfg, background=background)
        raw_detections = []  # [(frame_idx, cx, cy, radius_px), ...]
        ball_left_roi = False
        n_not_found = 0

        for fi, frame in frames_to_scan:
            detector.reset()
            result = detector.detect(frame, fi)

            if not result.get("found"):
                n_not_found += 1
                continue

            cx = result.get("x_px")
            cy = result.get("y_px")
            r = result.get("radius_px")

            # 仅校验：位置在 ROI 内 + 半径有效
            if not (roi_x <= cx <= roi_x + roi_w and roi_y <= cy <= roi_y + roi_h):
                continue
            if r is None or r <= 0:
                continue

            raw_detections.append((fi, cx, cy, r))

            # 小球离开 ROI 底部 → 停止扫描
            if cy > roi_y + roi_h + 20:
                ball_left_roi = True
                break

        n_scanned = len(frames_to_scan)
        n_found = len(raw_detections)

        if n_found < 3:
            self._result_tabs.append_log(
                f"[BALL-CALIB] ROI内扫描 {n_scanned} 帧, "
                f"检出 {n_found} 帧（需≥3帧）\n"
                f"  BallDetector未检出={n_not_found} 帧\n"
                f"  请确保: ROI框住了小球, 小球在当前帧可见"
            )
            dialog = BallCalibrateDialog(
                ball_diameter_mm=ball_diam,
                detected_radius_px=None,
                detection_ok=False,
                parent=self,
            )
            dialog.exec()
            return

        radii = np.array([d[3] for d in raw_detections])

        # ★ 密度峰值法：小球会被稳定检测到某个值附近，噪点偶尔出现。
        #    找直方图中计数最多的 bin，取该 bin 周围窗口内的均值。
        #    这比中位数/均值更抗噪——异常值再多也影响不了峰值位置。
        bin_width = 0.15  # px
        r_min, r_max = radii.min(), radii.max()
        n_bins = max(5, int((r_max - r_min) / bin_width) + 1)
        hist, bin_edges = np.histogram(radii, bins=n_bins)

        # 找密度最高的 bin
        peak_bin = int(np.argmax(hist))
        peak_center = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0

        # 取峰值窗口 ±2 bin 内的所有样本
        win_start = max(0, peak_bin - 2)
        win_end = min(n_bins, peak_bin + 3)
        win_lo = bin_edges[win_start]
        win_hi = bin_edges[win_end]
        in_window = (radii >= win_lo) & (radii <= win_hi)
        radii_stable = radii[in_window]

        detected_r = float(np.mean(radii_stable))
        detected_ok = True
        std_r = float(np.std(radii_stable))
        n_dropped = n_found - len(radii_stable)

        self._result_tabs.append_log(
            f"[BALL-CALIB] ROI 内扫描 {n_scanned} 帧 → 检出 {n_found} 帧\n"
            f"  半径直方图峰值 @ {peak_center:.2f}px (计数={hist[peak_bin]})\n"
            f"  峰值窗口 [{win_lo:.2f}, {win_hi:.2f}] → {len(radii_stable)} 帧 "
            f"(丢弃 {n_dropped})\n"
            f"  密度均值: {detected_r:.2f}px, std={std_r:.2f}px"
            + (f"\n  小球{'已' if ball_left_roi else '未'}离开 ROI 底部")
        )

        dialog = BallCalibrateDialog(
            ball_diameter_mm=ball_diam,
            detected_radius_px=detected_r,
            detection_ok=detected_ok,
            parent=self,
        )
        if dialog.exec() == BallCalibrateDialog.Accepted:
            scale = dialog.get_scale()
            if scale > 0:
                self._param_panel.set_config({"scale_mm_per_px": scale})
                self._param_panel.save_settings()
                self._result_tabs.append_log(
                    f"[INFO] 小球自标定完成\n"
                    f"  小球直径: {ball_diam:.2f} mm\n"
                    f"  像素直径: {detected_r * 2:.2f} px "
                    f"(ROI内 {len(radii_stable)} 帧稳定均值)\n"
                    f"  比例尺: {scale:.6f} mm/px"
                )

    def _on_ball_position_set(self, pos):
        """用户手动设置了初始小球位置。"""
        self._init_ball_pos = pos
        self._result_tabs.append_log(
            f"[INFO] 初始小球位置已设置: ({pos[0]}, {pos[1]})"
        )

    # ---- 单帧测试 ----
    def _on_test_frame(self):
        """测试当前帧识别，输出完整调试信息。"""
        frame = self._video_widget.get_current_frame_bgr()
        if frame is None:
            QMessageBox.warning(self, "提示", "请先加载视频或图片。")
            return

        frame_idx = self._video_widget._current_frame_idx

        cfg = self._param_panel.get_config()
        roi = self._video_widget.get_roi()
        if roi is not None:
            cfg["roi"] = roi
            cfg["detect_roi"] = list(roi)
        else:
            cfg["roi"] = None
            cfg["detect_roi"] = None

        # 传递媒体类型给检测器（图片模式禁止 bg_sub）
        cfg["image_mode"] = (self._media_type == "image")

        # 构建背景模型（仅视频模式 + auto/bg_sub 检测方法）
        background = None
        detect_method = cfg.get("detect_method", "auto")
        if not cfg["image_mode"] and detect_method in ("auto", "background_subtraction"):
            try:
                from src.video_io import VideoReader
                reader = VideoReader(self._video_path)
                from src.tracking import _build_background
                bg_roi = cfg.get("detect_roi") or cfg.get("roi")
                background = _build_background(reader, n_frames=50, roi=bg_roi)
                reader.reset()
            except Exception as e:
                self._result_tabs.append_log(f"[WARN] 背景模型构建失败: {e}")

        # Keep the single-frame debug path aligned with full analysis. Without
        # this, 2.0/2.5 mm balls can still be judged by a stale 1-4 px radius
        # range and the preview becomes misleading.
        from src.tracking import (
            _compute_dynamic_detection_params,
            _compute_dynamic_tracking_params,
        )
        _compute_dynamic_detection_params(cfg)
        _compute_dynamic_tracking_params(cfg)

        detector = BallDetector(cfg, background=background)
        detector.reset()

        result = detector.detect(frame, frame_idx)

        # 存储检测结果 + 调试图片
        self._video_widget.set_detection(result)
        dbg = result.get("debug_images") or {}
        self._video_widget.set_debug_images(dbg)

        # 检查缺失的调试图
        expected_keys = ["roi_frame", "gray", "binary_mask", "candidate_preview", "final_preview"]
        missing = [k for k in expected_keys if k not in dbg or dbg[k] is None]
        if missing:
            self._result_tabs.append_log(f"[DEBUG] 缺少调试图: {', '.join(missing)}")

        # ---- 调试输出 ----
        lines = []
        lines.append(
            "[DEBUG] ball-size: "
            f"expected_r={cfg.get('_expected_radius_px', 0):.2f}px, "
            f"range={cfg.get('expected_radius_px_min', 0):.2f}-"
            f"{cfg.get('expected_radius_px_max', 0):.2f}px, "
            f"source={cfg.get('_param_source', '?')}"
        )
        lines.append(f"[DEBUG] 帧 {frame_idx} | ROI: {cfg['roi']}")
        lines.append(f"[DEBUG] 检测方法: {result.get('method_used', '?')}")

        # fall_axis_x 信息
        fall_axis = cfg.get("fall_axis_x")
        if fall_axis:
            lines.append(f"[DEBUG] 下落中心线: fall_axis_x={fall_axis}")
        else:
            lines.append(f"[DEBUG] 下落中心线: 未设置（使用 ROI 中心）")

        if dbg:
            lines.append(f"[DEBUG] 总轮廓数: {dbg.get('total_contours', 0)}")
            lines.append(f"[DEBUG] 原始 blob 数: {dbg.get('raw_blob_count', 0)}")
            sw_filtered = dbg.get("search_window_filtered", 0)
            if sw_filtered:
                lines.append(f"[DEBUG] 搜索窗口过滤: {sw_filtered} 个")
            axis_rej = dbg.get("axis_rejected", 0)
            if axis_rej:
                rej_details = dbg.get("axis_rejected_details", [])
                lines.append(f"[DEBUG] 中心线过滤: {axis_rej} 个")
                for rcx, rcy, rd in rej_details[:3]:
                    lines.append(f"  - ({rcx:.0f},{rcy:.0f}) dist={rd:.0f}px")
            bt = dbg.get("bg_sub_threshold")
            if bt is not None and bt > 0:
                lines.append(f"[DEBUG] bg_sub 阈值: {bt}")
            tv = dbg.get("threshold_value")
            if tv is not None and tv > 0:
                lines.append(f"[DEBUG] Otsu 阈值: {tv:.0f}")

        # 显示所有候选点（v3 详细输出）
        candidates = result.get("candidates", [])
        if candidates:
            # 统计过滤信息
            dbg = result.get("debug_images", {})
            axis_rej = dbg.get("axis_rejected", 0)
            sw_filtered = dbg.get("search_window_filtered", 0)
            sw_expanded = dbg.get("search_window_expanded", False)
            filters = []
            if axis_rej:
                filters.append(f"中心线过滤={axis_rej}")
            if sw_filtered:
                filters.append(f"搜索窗过滤={sw_filtered}")
            if sw_expanded:
                filters.append(f"搜索窗扩大={dbg.get('search_window_expanded','?')}")
            if filters:
                lines.append(f"[DEBUG] 过滤: {' | '.join(filters)}")

            lines.append(f"[DEBUG] 候选点 (共 {len(candidates)} 个):")
            for c in candidates[:10]:  # 前 10 个
                rank = c.get("rank", 0)
                flag = "✓" if rank == 1 else " "
                dist_axis = c.get("dist_to_axis", 0)
                source = c.get("source_method", "?")
                motion = c.get("motion_bonus", 0)
                ll_penalty = c.get("long_line_penalty", 0)
                reject = c.get("reject_reason", "")
                reject_tag = f" [{reject}]" if reject else ""
                lines.append(
                    f"  {flag} #{rank}: ({c['cx']:.0f},{c['cy']:.0f})"
                    f" r={c['radius']:.1f}"
                    f" score={c['score']:.0f}"
                    f" ctrst={c.get('contrast', 0):.0f}"
                    f" ax={dist_axis:.0f}"
                    f" m={motion:.0f}"
                    f" diff={c.get('diff_score', 0):.0f}"
                    f" src={source}"
                    f" ll={ll_penalty:.0f}"
                    f"{reject_tag}"
                )
            if len(candidates) > 10:
                lines.append(f"  ... 还有 {len(candidates) - 10} 个")

            # 显示各维度得分 (v5)
            best = candidates[0]
            md = best.get("mean_diff", 0)
            lines.append(
                f"[DEBUG] 最优 #1 得分构成 (v5):"
                f" diff={best.get('diff_score',0):.0f}(md={md:.1f})"
                f" pred={best.get('prediction_score',0):.0f}"
                f" cont={best.get('continuity',0):.0f}"
                f" iso={best.get('isolation',0):.0f}"
                f" size={best.get('size_score',0):.0f}"
                f" axis={best.get('axis_score',0):.0f}"
                f" band={best.get('center_band',0):.0f}"
                f" mot={best.get('motion_bonus',0):.0f}"
                f" ll_p={best.get('long_line_penalty',0):.0f}"
                f" = {best['score']:.0f}"
            )

        if result["found"]:
            lines.append(
                f"[INFO] 识别成功!"
                f"  center=({result['x_px']:.0f}, {result['y_px']:.0f})"
                f"  r={result['radius_px']:.1f}px"
                f"  area={result['area_px']:.0f}px2"
                f"  circ={result['circularity']:.3f}"
                f"  conf={result.get('confidence', 0):.2f}"
            )
            # 提示用户可用调试视图查看
            lines.append("[INFO] 可切换下方「调试视图」查看 Mask/轮廓/标注图")
        else:
            lines.append(f"[WARN] 识别失败!")
            fail_reason = result.get("fail_reason", "")
            if fail_reason:
                for line in fail_reason.split("\n"):
                    lines.append(f"  {line}")
            else:
                lines.append("  - 请检查 ROI 是否框住小球运动区域")
                lines.append("  - 检查颜色模式是否匹配")

            # 候选统计
            cand = result.get("candidates", [])
            if cand:
                lines.append(f"[DEBUG] 共有 {len(cand)} 个候选轮廓但被筛选条件过滤")
                # 显示前 3 个候选的信息
                for i, c in enumerate(cand[:3]):
                    lines.append(
                        f"  候选{i + 1}: area={c['area']:.0f}px2"
                        f"  circ={c['circularity']:.3f}"
                        f"  r={c['radius']:.1f}px"
                    )

        self._result_tabs.append_log("\n".join(lines))

    # ---- 完整分析 ----
    def _on_analyze(self):
        """启动完整视频分析（后台线程）。"""
        from src.utils import normalize_config_keys

        if self._media_type != "video":
            QMessageBox.warning(self, "提示",
                                "当前为图片模式，完整视频分析需要加载视频文件。"
                                if self._media_type == "image"
                                else "请先加载视频文件。")
            return
        if not self._video_path or not os.path.exists(self._video_path):
            QMessageBox.warning(self, "提示",
                                "视频文件不存在，请重新加载视频。")
            return

        # ★ 从 UI 实时读取当前参数并归一化
        cfg = self._param_panel.get_config()
        cfg = normalize_config_keys(cfg)

        # 检查比例尺
        if cfg.get("scale_mm_per_px") is None:
            QMessageBox.warning(self, "提示",
                                "缺少比例尺，无法将像素坐标换算为实际距离。\n"
                                "请使用「标定比例尺」功能或手动输入。")
            return

        roi = self._video_widget.get_roi()
        if roi is not None:
            cfg["roi"] = roi

        cfg["video_path"] = self._video_path
        cfg["output_dir"] = self._output_dir
        if self._init_ball_pos is not None:
            cfg["init_ball_x"] = float(self._init_ball_pos[0])
            cfg["init_ball_y"] = float(self._init_ball_pos[1])

        # ★ 自动保存当前参数到 settings.json（使用归一化后的 config）
        self._param_panel.save_settings()

        # ★ 读取刚保存的文件做一致性验证
        from src.gui.parameter_panel import _SETTINGS_PATH
        import json
        try:
            with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
                file_cfg = json.load(f)
            file_cfg = normalize_config_keys(file_cfg)
        except Exception:
            file_cfg = {}

        # ★ 一致性自检输出
        log_lines = ["=" * 45]
        log_lines.append("  CONFIG CONSISTENCY CHECK")
        log_lines.append("=" * 45)
        check_keys = [
            ("scale_mm_per_px", "mm/px"),
            ("ball_radius_mm", "mm"),
            ("ball_density_kg_m3", "kg/m³"),
            ("liquid_density_kg_m3", "kg/m³"),
            ("cylinder_radius_mm", "mm"),
            ("liquid_height_mm", "mm"),
            ("temperature_c", "°C"),
            ("reference_viscosity_pa_s", "Pa·s"),
            ("enable_wall_correction", ""),
            ("enable_reynolds_correction", ""),
            ("gravity_m_s2", "m/s²"),
            ("manual_fps", "fps"),
        ]
        all_ok = True
        for key, unit in check_keys:
            uv = cfg.get(key, "?")
            fv = file_cfg.get(key, "?")
            tag = "✓" if str(uv) == str(fv) else "✗ MISMATCH"
            if tag != "✓":
                all_ok = False
            unit_str = f" {unit}" if unit else ""
            log_lines.append(f"  {key:30s}  UI={uv}{unit_str}  FILE={fv}{unit_str}  {tag}")
        log_lines.append("=" * 45)
        if all_ok:
            log_lines.append("  一致性检查: 全部通过 ✓")
        else:
            log_lines.append("  一致性检查: 发现不一致 ✗ — 请检查后重试")
        log_lines.append("=" * 45)
        for line in log_lines:
            self._result_tabs.append_log(line)

        # ★ 日志输出当前实际使用的关键参数
        log_lines2 = ["========== 当前实际分析参数 =========="]
        _keys = [
            ("scale_mm_per_px", "mm/px"),
            ("ball_radius_mm", "mm"),
            ("ball_density_kg_m3", "kg/m³"),
            ("liquid_density_kg_m3", "kg/m³"),
            ("cylinder_radius_mm", "mm"),
            ("liquid_height_mm", "mm"),
            ("temperature_c", "°C"),
            ("reference_viscosity_pa_s", "Pa·s"),
            ("enable_wall_correction", ""),
            ("enable_reynolds_correction", ""),
            ("gravity_m_s2", "m/s²"),
            ("manual_fps", "fps"),
            ("color_mode", ""),
            ("threshold_method", ""),
            ("min_area_px", "px²"),
            ("max_area_px", "px²"),
            ("min_circularity", ""),
            ("velocity_window_sec", "s"),
            ("auto_size_params", ""),
            ("large_ball_mode", ""),
            ("auto_shrink_roi", ""),
            ("terminal_window_sec", "s"),
            ("cv_threshold", ""),
            ("r2_threshold", ""),
            ("terminal_ignore_start_sec", "s"),
            ("terminal_ignore_end_sec", "s"),
        ]
        for key, unit in _keys:
            v = cfg.get(key, "?")
            if unit:
                log_lines2.append(f"  {key} = {v} {unit}")
            else:
                log_lines2.append(f"  {key} = {v}")
        log_lines2.append("=" * 40)
        for line in log_lines2:
            self._result_tabs.append_log(line)

        # 禁用按钮
        self._param_panel.set_analyzing(True)
        self._result_tabs.append_log("[INFO] 开始分析完整视频...")

        # 创建工作线程（传递归一化后的 config）
        self._worker = AnalysisWorker(cfg, self._video_path, self._output_dir, self)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.frame_data.connect(self._on_frame_data)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        # 实时绘图初始化
        self._live_traj_points.clear()
        self._live_scale_mm_px = float(cfg.get("scale_mm_per_px", 1.0))
        self._result_tabs.clear_trajectory_plot()
        self._live_update_timer.start()
        self._worker.start()

    def _on_stop_analysis(self):
        """停止分析。"""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.quit()
            self._worker.wait(3000)
            self._param_panel.set_analyzing(False)
            self._live_update_timer.stop()
            self._result_tabs.append_log("[INFO] 分析已停止。")

    def _on_worker_progress(self, msg: str):
        """工作线程进度更新。"""
        self._result_tabs.append_log(msg)

    def _on_frame_data(self, frame_idx, x_px, y_px, time_s, radius_px, is_valid, is_fallback):
        """实时接收逐帧检测结果（工作线程跨线程信号）。"""
        if is_valid and y_px is not None:
            y_mm = y_px * self._live_scale_mm_px
            self._live_traj_points.append((time_s, y_mm))

    def _update_live_plot(self):
        """定时刷新 y-t 实时曲线。"""
        if not self._live_traj_points:
            return
        self._result_tabs.update_trajectory_plot_incremental(self._live_traj_points)

    def _on_worker_finished(self, result: dict):
        """工作线程完成。"""
        self._param_panel.set_analyzing(False)
        self._live_update_timer.stop()
        # 最终刷新实时图（确保最后一帧数据也显示）
        self._update_live_plot()

        self._traj_df = result.get("traj_df")
        self._traj_df_full = result.get("traj_df_full")
        self._velocity_df = result.get("velocity_df")
        self._terminal_region = result.get("terminal_region", {"found": False})
        self._viscosity_result = result.get("viscosity_result")
        self._analysis_result = result

        # 更新轨迹点 — 全视频轨迹（紫色显示）
        # 注意：紫色轨迹含 raw + predicted + interpolated 所有有效点
        # 用于显示轨迹连续性，不代表全部参与分析
        # 终端速度拟合仅使用 raw 检测点
        full_traj = result.get("traj_df_full")
        if full_traj is not None:
            valid = full_traj[full_traj["valid"]]
            if len(valid) > 0:
                trail = list(zip(valid["x_px"].values, valid["y_px"].values))
                self._video_widget.set_trail(trail)
                raw_count = int((full_traj["point_type"] == "raw").sum())
                pred_count = int((full_traj["point_type"] == "predicted").sum())
                interp_count = int((full_traj["point_type"] == "interpolated").sum())
                self._result_tabs.append_log(
                    f"[INFO] 叠加显示轨迹 (紫色): {len(valid)} 个有效点 "
                    f"(raw={raw_count}, predicted={pred_count}, interpolated={interp_count})"
                )
        elif self._traj_df is not None:
            valid = self._traj_df[self._traj_df["valid"]]
            trail = list(zip(valid["x_px"].values, valid["y_px"].values))
            self._video_widget.set_trail(trail)

        # 更新图表
        if self._traj_df is not None:
            self._result_tabs.update_trajectory_plot(self._traj_df, self._terminal_region)
        if self._velocity_df is not None:
            self._result_tabs.update_velocity_plot(self._velocity_df, self._terminal_region)
        if self._traj_df is not None:
            self._result_tabs.update_fit_plot(self._traj_df, self._terminal_region)

        # 综合修正调试输出到日志
        if self._viscosity_result:
            vr = self._viscosity_result
            wall_on = vr.get('enable_wall_correction', True)
            re_on = vr.get('enable_reynolds_correction', True)
            tag_parts = []
            if wall_on: tag_parts.append("壁面")
            if re_on: tag_parts.append("雷诺数")
            corr_tag = "+".join(tag_parts) + "修正" if tag_parts else "无修正"
            log_lines = [
                f"[DEBUG] 修正详情 ({corr_tag}):",
                f"  r = {vr.get('r_m', '?'):.6f} m",
                f"  η_basic = {vr.get('eta_basic_pa_s', '?'):.6f} Pa·s",
                f"  R = {vr.get('R_m', '?'):.6f} m",
                f"  h = {vr.get('h_m', '?'):.6f} m",
                f"  r/R = {vr.get('r_over_R', '?'):.6f}",
                f"  r/h = {vr.get('r_over_h', '?'):.6f}",
                f"  wall_correction_factor = {vr.get('wall_correction_factor', '?'):.6f}",
                f"  η_wall = {vr.get('eta_wall_pa_s', '?'):.6f} Pa·s",
                f"  Re = {vr.get('reynolds_number', '?'):.6f}",
                f"  reynolds_factor = 1 + 3/16·Re = {vr.get('reynolds_factor', '?'):.6f}",
                f"  η_re = {vr.get('eta_reynolds_pa_s', '?'):.6f} Pa·s",
                f"  η_final = {vr.get('eta_final_pa_s', '?'):.6f} Pa·s",
            ]
            ref_v = vr.get('reference_viscosity_pa_s')
            if ref_v and ref_v > 0:
                basic_err = abs(vr['eta_basic_pa_s'] - ref_v) / ref_v * 100
                final_err = abs(vr['eta_final_pa_s'] - ref_v) / ref_v * 100
                log_lines.append(f"  reference_viscosity = {ref_v:.6f} Pa·s")
                log_lines.append(f"  basic_error_percent = {basic_err:.2f}%")
                log_lines.append(f"  final_error_percent = {final_err:.2f}%")
            for line in log_lines:
                self._result_tabs.append_log(line)

        # 更新结果表（使用分析区间统计）
        stats = self._analysis_result or {}
        tr = self._terminal_region or {}
        target_summary = {
            "video_total_frames": stats.get("video_total_frames", 0),
            "interval_source": stats.get("interval_source", "auto_track"),
            "termination_reason": stats.get("termination_reason", ""),
            "analysis_start_frame": stats.get("analysis_start_frame", 0),
            "analysis_end_frame": stats.get("analysis_end_frame", 0),
            "analysis_total_frames": stats.get("analysis_total_frames", 0),
            "analysis_duration_s": stats.get("analysis_duration_s", 0),
            "valid_frames_in_analysis": stats.get("valid_frames_in_analysis", 0),
            "valid_rate_in_analysis": stats.get("valid_rate_in_analysis", 0),
            "valid_frames_all_video": stats.get("valid_frames_all_video", 0),
            "valid_rate_all_video": stats.get("valid_rate_all_video", 0),
        }
        self._result_tabs.update_result_table(
            self._terminal_region, self._viscosity_result,
            target_summary,
        )

    def _on_worker_error(self, msg: str):
        """工作线程错误。"""
        self._param_panel.set_analyzing(False)
        self._result_tabs.append_log(f"[ERROR] {msg}")
        QMessageBox.warning(self, "分析错误", msg)

    # ---- 导出 ----
    def _on_export(self):
        """导出分析结果。"""
        if self._analysis_result is None:
            QMessageBox.warning(
                self, "提示",
                "当前没有可导出的分析结果，请先完成视频分析。"
            )
            return

        export_dir = QFileDialog.getExistingDirectory(
            self, "选择导出文件夹", self._output_dir
        )
        if not export_dir:
            return

        try:
            # 复制已有结果
            import shutil
            src_dir = self._output_dir

            files_to_export = [
                "trajectory.csv", "velocity.csv",
                "marked_video.mp4",
                "trajectory_plot.png", "velocity_plot.png", "fit_plot.png",
                "result_summary.txt", "config_used.yaml",
            ]

            exported = []
            for fname in files_to_export:
                src = os.path.join(src_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(export_dir, fname))
                    exported.append(fname)

            self._result_tabs.append_log(
                f"[INFO] 已导出 {len(exported)} 个文件至: {export_dir}"
            )
            QMessageBox.information(
                self, "导出完成",
                f"已导出 {len(exported)} 个文件至:\n{export_dir}\n\n"
                + "\n".join(f"  - {f}" for f in exported)
            )

        except Exception as e:
            QMessageBox.warning(self, "导出失败", f"导出过程中发生错误:\n{e}")

    # ---- 清空 ----
    def _on_clear(self):
        """清空当前结果。"""
        self._analysis_result = None
        self._traj_df = None
        self._velocity_df = None
        self._terminal_region = {"found": False}
        self._viscosity_result = None
        self._video_widget.set_trail([])
        self._video_widget.set_detection({"found": False})
        self._video_widget.clear_debug_images()
        self._video_widget.clear_background()
        self._current_roi = None
        self._result_tabs.clear_all()
        self._result_tabs.append_log("[INFO] 结果已清空。")

    def _on_config_saved(self, path: str):
        """参数已保存到文件。"""
        self._result_tabs.append_log(f"[CONFIG] 参数已保存到: {os.path.normpath(path)}")

    def _on_profile_changed(self, key: str):
        """分辨率档位切换时，尝试加载该档位保存的比例尺。"""
        if key == "custom":
            return
        try:
            import json
            from src.gui.parameter_panel import _SETTINGS_PATH
            with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
                settings = json.load(f)
            per_res = settings.get("scale_per_resolution", {})
            saved_scale = per_res.get(key)
            if saved_scale and saved_scale > 0:
                self._param_panel.set_config({"scale_mm_per_px": saved_scale})
                self._result_tabs.append_log(
                    f"[CONFIG] 档位 {key} → 加载保存的比例尺: {saved_scale:.6f} mm/px"
                )
            else:
                self._result_tabs.append_log(
                    f"[CONFIG] 档位 {key} 无保存的比例尺，请重新标定"
                )
        except Exception as e:
            self._result_tabs.append_log(f"[WARN] 加载档位比例尺失败: {e}")

    # ---- 窗口事件 ----
    def closeEvent(self, event):
        """关闭窗口时清理并保存参数。"""
        self._on_stop_analysis()
        self._video_widget.close_video()
        self._param_panel.save_settings()
        event.accept()

    # ---- 文件拖入（窗口级别） ----
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            self.load_media(path)
            return
