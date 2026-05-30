"""
视频预览组件（重构版）

使用 QPainter 直接在 widget 上绘制视频帧和叠加层。
所有叠加数据以原始视频像素坐标保存，绘制时转换到显示坐标。

坐标系统：
  - 视频坐标 (vx, vy)：原始视频帧的像素坐标
  - 显示坐标 (dx, dy)：widget 上的像素坐标
"""

import os
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider,
    QLabel, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QTimer, QPointF, QRect, QRectF
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont,
    QMouseEvent, QPaintEvent, QWheelEvent,
)


class VideoWidget(QWidget):
    """视频预览组件（QPainter 直接绘制）。"""

    frame_changed = Signal(int, float)
    roi_changed = Signal(list)
    points_selected = Signal(tuple, tuple)
    ball_position_set = Signal(tuple)   # 用户手动点击了小球位置 (vx, vy)
    video_loaded = Signal(str)     # 成功加载视频时发出，附带路径
    image_loaded = Signal(str)     # 成功加载图片时发出，附带路径

    MODE_NONE = 0
    MODE_ROI = 1
    MODE_CALIBRATE = 2
    MODE_SET_BALL = 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ---- 视频源 ----
        self._cap = None
        self._video_path = ""
        self._total_frames = 0
        self._fps = 30.0
        self._video_w = 0
        self._video_h = 0
        self._current_frame_idx = 0
        self._frame_buffer = None   # (H,W,3) RGB uint8
        self._current_frame_bgr = None  # 原始 BGR 帧（用于检测）
        self._qimage = None         # 缓存的 QImage
        self._image_mode = False    # True=静态图片, False=视频

        # ---- 坐标变换缓存 ----
        self._scale = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0

        # ---- 背景模型 ----
        self._bg_model = None          # 背景图（用于 bg_sub）
        self._background_ready = False

        # ---- 叠加层（全部以原始视频坐标保存） ----
        self._roi = None            # [x, y, w, h]
        self._detection = None      # dict from ball_detector
        self._trail = []            # [(x, y), ...]
        self._terminal_region = {"found": False}

        # ---- 调试视图 ----
        self._debug_images = None   # dict from ball_detector debug_images
        self._debug_view_mode = 0   # 0=原图, 1=ROI, 2=灰度, 3=Mask, 4=轮廓, 5=最终标注

        # ---- 缩放 ----
        self._zoom_factor = 1.0        # 1.0 = 适配显示，>1 = 放大
        self._pan_x = 0.0              # 平移偏移（显示坐标）
        self._pan_y = 0.0
        self._is_panning = False
        self._pan_start_mouse = None   # 拖拽平移起点（屏幕坐标）
        self._pan_start_xy = None      # 拖拽平移起点（pan_x, pan_y）

        # ---- 交互状态 ----
        self._mode = self.MODE_NONE
        self._drag_start = None     # QPointF 在 widget 坐标
        self._drag_end = None
        self._calib_points = []     # [(vx, vy), ...]
        self._is_playing = False

        # ---- 播放定时器 ----
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._next_frame)

        # ---- 控件 ----
        self._setup_ui()

    # ================================================================
    #  控件布局
    # ================================================================
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # 视频显示区域（自定义绘制）
        self._video_area = _VideoArea(self)
        self._video_area.setMinimumSize(480, 320)
        self._video_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._video_area.mouse_pressed.connect(self._on_mouse_press)
        self._video_area.mouse_moved.connect(self._on_mouse_move)
        self._video_area.mouse_released.connect(self._on_mouse_release)
        self._video_area.wheel_zoomed.connect(self._on_wheel_zoomed)
        layout.addWidget(self._video_area, 1)

        # 播放控制栏
        ctrl = QHBoxLayout()
        ctrl.setSpacing(4)
        self._btn_play = QPushButton("▶")
        self._btn_play.setFixedSize(32, 24)
        self._btn_play.clicked.connect(self.toggle_play)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.valueChanged.connect(self._on_slider_changed)

        self._frame_label = QLabel("0 / 0")
        self._frame_label.setFixedWidth(100)
        self._frame_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        ctrl.addWidget(self._btn_play)
        ctrl.addWidget(self._slider, 1)
        ctrl.addWidget(self._frame_label)
        layout.addLayout(ctrl)

        # 工具按钮
        tools = QHBoxLayout()
        tools.setSpacing(4)
        self._btn_roi = QPushButton("框选 ROI")
        self._btn_roi.setCheckable(True)
        self._btn_roi.clicked.connect(self._toggle_roi_mode)

        self._btn_clear_roi = QPushButton("清除 ROI")
        self._btn_clear_roi.clicked.connect(self._clear_roi)

        self._btn_calib = QPushButton("标定比例尺")
        self._btn_calib.clicked.connect(self._start_calibrate)

        self._btn_set_ball = QPushButton("设置初始位置")
        self._btn_set_ball.setCheckable(True)
        self._btn_set_ball.clicked.connect(self._toggle_set_ball_mode)

        self._ball_pos = None   # 存储用户设置的初始位置 (vx, vy)

        self._btn_prev = QPushButton("◀")
        self._btn_prev.setFixedWidth(28)
        self._btn_prev.clicked.connect(self._prev_frame)

        self._btn_next = QPushButton("▶")
        self._btn_next.setFixedWidth(28)
        self._btn_next.clicked.connect(self._next_frame)

        tools.addWidget(self._btn_roi)
        tools.addWidget(self._btn_clear_roi)
        tools.addWidget(self._btn_calib)
        tools.addWidget(self._btn_set_ball)

        # 调试视图选择
        from PySide6.QtWidgets import QComboBox
        self._debug_combo = QComboBox()
        self._debug_combo.addItems(["原图", "ROI 图", "灰度图", "差分图", "候选区域", "轮廓图", "最终标注"])
        self._debug_combo.setMinimumWidth(100)
        self._debug_combo.currentIndexChanged.connect(self._on_debug_view_changed)
        tools.addWidget(self._debug_combo)

        # 缩放控制
        self._btn_zoom_in = QPushButton("⊕")
        self._btn_zoom_in.setFixedSize(28, 24)
        self._btn_zoom_in.setToolTip("放大（标定时用）")
        self._btn_zoom_in.clicked.connect(lambda: self._zoom_relative(1.4))
        self._btn_zoom_out = QPushButton("⊖")
        self._btn_zoom_out.setFixedSize(28, 24)
        self._btn_zoom_out.setToolTip("缩小")
        self._btn_zoom_out.clicked.connect(lambda: self._zoom_relative(1 / 1.4))
        self._btn_zoom_reset = QPushButton("1:1")
        self._btn_zoom_reset.setFixedSize(32, 24)
        self._btn_zoom_reset.setToolTip("重置缩放")
        self._btn_zoom_reset.clicked.connect(self._zoom_reset)
        self._zoom_label = QLabel("100%")
        self._zoom_label.setFixedWidth(36)
        self._zoom_label.setAlignment(Qt.AlignCenter)
        tools.addWidget(self._btn_zoom_out)
        tools.addWidget(self._btn_zoom_in)
        tools.addWidget(self._btn_zoom_reset)
        tools.addWidget(self._zoom_label)

        tools.addStretch()
        tools.addWidget(self._btn_prev)
        tools.addWidget(self._btn_next)
        layout.addLayout(tools)

        # 状态栏
        self._status = QLabel("等待导入视频...")
        self._status.setStyleSheet("color: #888; padding: 2px 0;")
        layout.addWidget(self._status)

    # ================================================================
    #  坐标变换（核心修复）
    # ================================================================
    def _update_transform(self):
        """根据 widget 当前尺寸和视频原始分辨率刷新缩放参数。"""
        vw, vh = self._video_w, self._video_h
        dw = self._video_area.width()
        dh = self._video_area.height()
        if vw <= 0 or vh <= 0 or dw <= 0 or dh <= 0:
            self._scale = 1.0
            self._offset_x = 0.0
            self._offset_y = 0.0
            return
        base_scale = min(dw / vw, dh / vh)
        self._scale = base_scale * self._zoom_factor
        self._offset_x = (dw - vw * self._scale) / 2.0 + self._pan_x
        self._offset_y = (dh - vh * self._scale) / 2.0 + self._pan_y

    def _zoom_relative(self, factor: float, cursor_pos: QPointF = None):
        """相对缩放，可指定光标位置作为缩放中心。"""
        old_zoom = self._zoom_factor
        new_zoom = old_zoom * factor
        new_zoom = max(1.0, min(new_zoom, 20.0))
        if abs(new_zoom - 1.0) < 0.001:
            self._zoom_reset()
            return

        # 计算缩放中心对应的视频坐标
        if cursor_pos is not None:
            dw = self._video_area.width()
            dh = self._video_area.height()
            base = min(dw / max(self._video_w, 1), dh / max(self._video_h, 1))
            cx, cy = cursor_pos.x(), cursor_pos.y()
            # 缩放前该点对应的视频坐标
            old_scale = base * old_zoom
            old_ox = (dw - self._video_w * old_scale) / 2.0 + self._pan_x
            old_oy = (dh - self._video_h * old_scale) / 2.0 + self._pan_y
            vx = (cx - old_ox) / old_scale if old_scale > 0 else 0
            vy = (cy - old_oy) / old_scale if old_scale > 0 else 0
            # 缩放后调整 pan 使该视频点仍在光标位置
            new_scale = base * new_zoom
            new_ox = (dw - self._video_w * new_scale) / 2.0
            new_oy = (dh - self._video_h * new_scale) / 2.0
            self._pan_x = cx - vx * new_scale - new_ox
            self._pan_y = cy - vy * new_scale - new_oy

        self._zoom_factor = new_zoom
        self._update_zoom_label()
        # 更新光标
        if self._mode == self.MODE_NONE:
            self._video_area.setCursor(Qt.OpenHandCursor if new_zoom > 1.0 else Qt.ArrowCursor)
        self._video_area.update()

    def _zoom_reset(self):
        """重置缩放到 1.0（适配显示）。"""
        self._zoom_factor = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._update_zoom_label()
        if self._mode == self.MODE_NONE:
            self._video_area.setCursor(Qt.ArrowCursor)
        self._video_area.update()

    def _update_zoom_label(self):
        self._zoom_label.setText(f"{int(self._zoom_factor * 100)}%")

    def video_to_display(self, vx: float, vy: float):
        """视频坐标 → widget 显示坐标。"""
        self._update_transform()
        return vx * self._scale + self._offset_x, vy * self._scale + self._offset_y

    def display_to_video(self, dx: float, dy: float):
        """widget 显示坐标 → 视频坐标。"""
        self._update_transform()
        if self._scale == 0:
            return 0.0, 0.0
        return (dx - self._offset_x) / self._scale, (dy - self._offset_y) / self._scale

    # ================================================================
    #  视频加载
    # ================================================================
    def load_video(self, path: str) -> bool:
        # 先关闭之前的视频/图片资源
        self.close_video()
        self._image_mode = False

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False

        self._cap = cap
        self._video_path = path
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._slider.setMaximum(max(self._total_frames - 1, 0))
        self._current_frame_idx = 0
        self._trail = []
        self._detection = None
        self._calib_points = []

        ret, frame = cap.read()
        if ret:
            self._set_frame(frame)

        self._update_frame_label()
        self._sync_ui()
        self._status.setText(
            f"{path.split('/')[-1].split(chr(92))[-1]}  "
            f"{self._video_w}x{self._video_h}  "
            f"{self._fps:.1f}fps  "
            f"{self._total_frames}帧"
        )
        self.frame_changed.emit(0, 0.0)
        self.video_loaded.emit(path)
        return True

    def load_image(self, path: str) -> bool:
        """加载静态图片（非视频）并显示。支持中文路径。"""
        # 使用 np.fromfile 兼容 Windows 中文路径
        img_data = np.fromfile(path, dtype=np.uint8)
        frame_bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return False

        # 关闭之前打开的视频
        self.close_video()

        self._image_mode = True
        self._video_path = path
        self._video_w = frame_bgr.shape[1]
        self._video_h = frame_bgr.shape[0]
        self._total_frames = 0
        self._fps = 30.0
        self._current_frame_idx = 0
        self._trail = []
        self._detection = None
        self._calib_points = []

        self._set_frame(frame_bgr)
        self._sync_ui()
        self._status.setText(
            f"图片: {os.path.basename(path)}  "
            f"{self._video_w}x{self._video_h}"
        )
        self.image_loaded.emit(path)
        return True

    def _set_frame(self, frame_bgr: np.ndarray):
        """缓存当前帧（保留 BGR 原版用于检测 + RGB 用于显示）。"""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).copy()
        self._current_frame_bgr = frame_bgr.copy()
        self._frame_buffer = rgb
        h, w = rgb.shape[:2]
        self._qimage = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self._video_area.update()

    def get_current_frame_bgr(self):
        """获取当前帧 BGR 数组（用于 BallDetector）。"""
        return self._current_frame_bgr

    # ================================================================
    #  播放控制
    # ================================================================
    def toggle_play(self):
        if self._cap is None:
            return
        self._is_playing = not self._is_playing
        if self._is_playing:
            self._play_timer.start(int(1000 / max(self._fps, 1)))
            self._btn_play.setText("⏸")
        else:
            self._play_timer.stop()
            self._btn_play.setText("▶")

    def seek_to(self, frame_idx: int):
        if self._cap is None:
            return
        idx = max(0, min(frame_idx, self._total_frames - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if ret:
            self._set_frame(frame)
            self._current_frame_idx = idx
            self._update_frame_label()
            self.frame_changed.emit(idx, idx / self._fps if self._fps > 0 else 0)

    def _prev_frame(self):
        if self._cap is not None:
            self.seek_to(self._current_frame_idx - 1)

    def _next_frame(self):
        if self._cap is not None:
            self.seek_to(self._current_frame_idx + 1)

    def _on_slider_changed(self, value):
        if self._cap is not None and not self._slider.isSliderDown():
            self.seek_to(value)

    def _update_frame_label(self):
        total = max(self._total_frames - 1, 0)
        if self._fps > 0:
            cur_time = self._current_frame_idx / self._fps
        else:
            cur_time = 0.0
        self._frame_label.setText(f"{self._current_frame_idx} / {total}  [{cur_time:.3f}s]")
        self._slider.blockSignals(True)
        self._slider.setValue(self._current_frame_idx)
        self._slider.blockSignals(False)

    def update_frame(self, frame_bgr: np.ndarray, frame_idx: int, detection: dict = None):
        self._set_frame(frame_bgr)
        self._current_frame_idx = frame_idx
        self._detection = detection
        self._update_frame_label()

    def set_fps(self, fps: float):
        """强制覆盖显示帧率（用于 OpenCV 读取帧率不正确的情况）。"""
        if fps > 0:
            self._fps = fps
            self._update_frame_label()

    def close_video(self):
        self._play_timer.stop()
        self._is_playing = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frame_buffer = None
        self._qimage = None
        self._image_mode = False
        self.clear_background()
        self._video_area.update()

    # ---- 背景模型管理 ----
    def clear_background(self):
        """清空背景模型（打开新文件/切换模式/ROI 改变时调用）。"""
        self._bg_model = None
        self._background_ready = False

    def set_background(self, bg: np.ndarray):
        """设置背景模型。"""
        if bg is not None:
            self._bg_model = bg.copy()
            self._background_ready = True

    def get_background(self):
        """获取背景模型。"""
        return self._bg_model

    def is_background_ready(self) -> bool:
        """背景模型是否就绪。"""
        return self._background_ready

    def _update_cursor(self):
        """根据当前模式和缩放状态设置光标。"""
        if self._mode != self.MODE_NONE:
            self._video_area.setCursor(Qt.CrossCursor if self._mode in (self.MODE_ROI, self.MODE_CALIBRATE, self.MODE_SET_BALL) else Qt.ArrowCursor)
        elif self._zoom_factor > 1.0:
            self._video_area.setCursor(Qt.OpenHandCursor)
        else:
            self._video_area.setCursor(Qt.ArrowCursor)

    # ================================================================
    #  ROI
    # ================================================================
    def _toggle_roi_mode(self):
        if self._frame_buffer is None:
            self._status.setText("请先加载图片或视频")
            return
        if self._mode == self.MODE_ROI:
            self._mode = self.MODE_NONE
            self._btn_roi.setChecked(False)
            self._update_cursor()
        else:
            self._mode = self.MODE_ROI
            self._btn_roi.setChecked(True)
            self._update_cursor()
            self._status.setText("在画面上拖动鼠标框选 ROI 区域")

    def _clear_roi(self):
        self._roi = None
        self.clear_background()
        self.roi_changed.emit(None)
        self._status.setText("ROI 已清除")
        self._video_area.update()

    def get_roi(self):
        return list(self._roi) if self._roi else None

    def set_roi(self, roi, *, emit_signal: bool = False):
        """Set ROI in original video coordinates."""
        if roi is None:
            self._roi = None
        else:
            x, y, w, h = [int(v) for v in roi]
            if w <= 0 or h <= 0:
                self._roi = None
            else:
                self._roi = [x, y, w, h]
        self.clear_background()
        if emit_signal:
            self.roi_changed.emit(self.get_roi())
        self._video_area.update()

    # ================================================================
    #  标定比例尺
    # ================================================================
    def _start_calibrate(self):
        if self._frame_buffer is None:
            return
        self._mode = self.MODE_CALIBRATE
        self._calib_points = []
        self._update_cursor()
        self._status.setText("依次点击标尺的两个端点（起点 → 终点）")

    def _toggle_set_ball_mode(self):
        if self._frame_buffer is None:
            self._btn_set_ball.setChecked(False)
            return
        if self._mode == self.MODE_SET_BALL:
            self._mode = self.MODE_NONE
            self._btn_set_ball.setChecked(False)
            self._update_cursor()
            self._status.setText("已退出「设置初始位置」模式")
        else:
            self._mode = self.MODE_SET_BALL
            self._btn_set_ball.setChecked(True)
            self._update_cursor()
            self._status.setText("请在小球中心位置点击一下")

    # ================================================================
    #  外部设定
    # ================================================================
    def set_detection(self, result: dict):
        self._detection = result
        self._video_area.update()

    def set_trail(self, points: list):
        self._trail = points
        self._video_area.update()

    def set_terminal_region(self, region: dict):
        self._terminal_region = region

    # ---- 调试视图 ----
    def set_debug_images(self, debug_images: dict):
        """存储检测器的调试图片。"""
        self._debug_images = debug_images

    def clear_debug_images(self):
        self._debug_images = None
        self._debug_view_mode = 0
        self._debug_combo.setCurrentIndex(0)
        self._video_area.update()

    def _on_debug_view_changed(self, idx: int):
        self._debug_view_mode = idx
        self._video_area.update()

    def set_status_text(self, text: str):
        self._status.setText(text)

    def sync_ui(self):
        self._sync_ui()

    def _sync_ui(self):
        """同步控件状态。图片模式下隐藏视频播放控件。"""
        is_video = not self._image_mode and self._cap is not None
        self._btn_play.setVisible(is_video)
        self._slider.setVisible(is_video)
        self._frame_label.setVisible(is_video)
        self._btn_prev.setVisible(is_video)
        self._btn_next.setVisible(is_video)
        self._video_area.update()

    def is_image_mode(self) -> bool:
        """当前是否为静态图片模式。"""
        return self._image_mode

    # ================================================================
    #  鼠标事件 → 转发到 _video_area
    # ================================================================
    def _on_mouse_press(self, pos: QPointF):
        if self._frame_buffer is None:
            return

        # 平移模式（放大后且无其他交互模式）
        if self._zoom_factor > 1.0 and self._mode == self.MODE_NONE:
            self._is_panning = True
            self._pan_start_mouse = pos
            self._pan_start_xy = (self._pan_x, self._pan_y)
            self._video_area.setCursor(Qt.ClosedHandCursor)
            return

        vx, vy = self.display_to_video(pos.x(), pos.y())
        if vx < 0 or vy < 0 or vx >= self._video_w or vy >= self._video_h:
            return

        if self._mode == self.MODE_ROI:
            self._drag_start = pos
            self._drag_end = pos

        elif self._mode == self.MODE_SET_BALL:
            # 单点点击设置初始小球位置
            self._ball_pos = (int(vx), int(vy))
            self.ball_position_set.emit((int(vx), int(vy)))
            self._mode = self.MODE_NONE
            self._btn_set_ball.setChecked(False)
            self._update_cursor()
            self._status.setText(f"初始小球位置已设置: ({int(vx)}, {int(vy)})")
            self._video_area.update()

        elif self._mode == self.MODE_CALIBRATE:
            self._calib_points.append((int(vx), int(vy)))
            if len(self._calib_points) == 1:
                self._status.setText(f"第一点: ({int(vx)}, {int(vy)})  请点击第二点")
            elif len(self._calib_points) == 2:
                p1, p2 = self._calib_points
                px_dist = abs(p2[1] - p1[1])  # 仅垂直距离
                if px_dist < 5:
                    self._calib_points = []
                    self._status.setText("两点距离太近，请重新选择")
                    return
                self.points_selected.emit(p1, p2)
                self._mode = self.MODE_NONE
                self._update_cursor()
            self._video_area.update()

    def _on_mouse_move(self, pos: QPointF):
        if self._is_panning and self._pan_start_mouse is not None:
            dx = pos.x() - self._pan_start_mouse.x()
            dy = pos.y() - self._pan_start_mouse.y()
            self._pan_x = self._pan_start_xy[0] + dx
            self._pan_y = self._pan_start_xy[1] + dy
            self._video_area.update()
            return
        if self._mode == self.MODE_ROI and self._drag_start is not None:
            self._drag_end = pos
            self._video_area.update()

    def _on_mouse_release(self, pos: QPointF):
        if self._is_panning:
            self._is_panning = False
            self._pan_start_mouse = None
            self._pan_start_xy = None
            self._update_cursor()
            return
        if self._mode == self.MODE_ROI and self._drag_start is not None and self._drag_end is not None:
            x1, y1 = self.display_to_video(self._drag_start.x(), self._drag_start.y())
            x2, y2 = self.display_to_video(pos.x(), pos.y())
            x = int(min(x1, x2))
            y = int(min(y1, y2))
            w = int(abs(x2 - x1))
            h = int(abs(y2 - y1))
            if w > 10 and h > 10:
                self._roi = [x, y, w, h]
                self.roi_changed.emit([x, y, w, h])
                self._status.setText(f"ROI: ({x}, {y})  {w}x{h}")
            else:
                self._status.setText("ROI 区域太小，请重新框选")
            self._drag_start = None
            self._drag_end = None
            self._mode = self.MODE_NONE
            self._btn_roi.setChecked(False)
            self._update_cursor()
            self._video_area.update()

    def _on_wheel_zoomed(self, event: QWheelEvent):
        """滚轮缩放，以鼠标位置为中心。"""
        if self._frame_buffer is None:
            return
        angle = event.angleDelta().y()
        if angle == 0:
            return
        factor = 1.15 if angle > 0 else 1 / 1.15
        self._zoom_relative(factor, event.position())

# ====================================================================
#  _VideoArea — 实际绘制区域
# ====================================================================
class _VideoArea(QWidget):
    """负责视频帧和叠加层的 QPainter 绘制。"""

    mouse_pressed = Signal(object)
    mouse_moved = Signal(object)
    mouse_released = Signal(object)
    wheel_zoomed = Signal(object)

    def __init__(self, parent: VideoWidget):
        super().__init__(parent)
        self._parent = parent
        self.setMouseTracking(True)
        self.setCursor(Qt.ArrowCursor)

    def paintEvent(self, event: QPaintEvent):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.setRenderHint(QPainter.Antialiasing)

        # 背景
        p.fillRect(self.rect(), QColor(26, 26, 26))

        vw = self._parent
        if vw._qimage is None:
            font = QFont("Segoe UI", 13)
            p.setFont(font)
            p.setPen(QColor(136, 136, 136))
            p.drawText(self.rect(), Qt.AlignCenter, "拖拽视频到此处\n或点击「打开视频」")
            return

        vw._update_transform()
        s = vw._scale
        ox, oy = vw._offset_x, vw._offset_y

        # ---- 判断是否显示调试视图 ----
        dbg = vw._debug_images
        dbg_mode = vw._debug_view_mode
        show_debug = dbg_mode > 0 and dbg is not None

        if show_debug:
            # 调试视图映射
            _KEY_MAP = {
                1: "roi_frame",
                2: "gray",
                3: "diff_map",
                4: "binary_mask",
                5: "candidate_preview",
                6: "final_preview",
            }
            _LABEL_MAP = {
                1: "ROI 图 (裁剪区域)",
                2: "灰度图",
                3: "背景差分图 (运动区域)",
                4: "候选区域 (Blob/Mask)",
                5: "轮廓图",
                6: "最终标注",
            }
            img_key = _KEY_MAP.get(dbg_mode)
            img_data = dbg.get(img_key) if img_key else None

            if img_data is not None:
                # 调试图可能是单通道（灰度/mask）或3通道（BGR/RGB）
                if len(img_data.shape) == 2:
                    h, w = img_data.shape
                    qimg = QImage(img_data.data, w, h, w, QImage.Format_Grayscale8)
                elif img_data.shape[2] == 3:
                    # 可能是 BGR 或 RGB，大部分来自 OpenCV 的是 BGR
                    h, w = img_data.shape[:2]
                    rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
                else:
                    qimg = None

                if qimg is not None:
                    iw, ih = qimg.width(), qimg.height()
                    tr = min(self.width() / max(iw, 1), self.height() / max(ih, 1))
                    sw = int(iw * tr)
                    sh = int(ih * tr)
                    ix = (self.width() - sw) / 2.0
                    iy = (self.height() - sh) / 2.0
                    scaled = qimg.scaled(sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    p.drawImage(QPointF(ix, iy), scaled)

                    # 调试视图标签（左上角）
                    label_text = _LABEL_MAP.get(dbg_mode, "调试视图")
                    p.setPen(QColor(0, 255, 255))
                    p.setFont(QFont("Segoe UI", 11, QFont.Bold))
                    p.drawText(QPointF(ix + 6, iy + 22), label_text)

                    # 空 Mask 提示
                    if dbg_mode == 3 and dbg.get("binary_mask_empty"):
                        p.setPen(QColor(255, 165, 0))
                        p.setFont(QFont("Segoe UI", 9))
                        p.drawText(QPointF(ix + 6, iy + 40),
                                   "当前方法未生成有效 Mask")

                    # 如果是最终标注或轮廓图，显示坐标信息
                    if dbg_mode == 5 and vw._detection and vw._detection.get("found"):
                        info = (f"  ({vw._detection['x_px']:.0f}, {vw._detection['y_px']:.0f})"
                                f"  r={vw._detection['radius_px']:.1f}px"
                                f"  area={vw._detection['area_px']:.0f}px2"
                                f"  circ={vw._detection['circularity']:.3f}"
                                f"  conf={vw._detection.get('confidence', 0):.2f}")
                        p.setPen(QColor(0, 255, 0))
                        p.setFont(QFont("Consolas", 9))
                        p.drawText(QPointF(ix + 6, iy + 42), info)
            else:
                # 调试图不存在 → 显示提示，不静默回退原图
                missing_label = _LABEL_MAP.get(dbg_mode, "未知视图")
                p.setPen(QColor(255, 80, 80))
                font_err = QFont("Segoe UI", 13, QFont.Bold)
                p.setFont(font_err)
                p.drawText(self.rect(), Qt.AlignCenter,
                           f"当前没有生成 {missing_label}\n请检查检测器是否返回 debug_images")
                p.setPen(QColor(200, 200, 200))
                font_hint = QFont("Segoe UI", 9)
                p.setFont(font_hint)
                p.drawText(QPointF(16, self.height() - 16),
                           f"缺失 key: {img_key}")
                return

        if not show_debug:
            # ---- 正常视频帧渲染 ----
            sw = int(vw._video_w * s)
            sh = int(vw._video_h * s)
            scaled = vw._qimage.scaled(sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            p.drawImage(QPointF(ox, oy), scaled)

            # ---- 叠加层 ----
            pen_roi = QPen(QColor(255, 165, 0), 2)
            pen_det = QPen(QColor(0, 255, 0), 2)
            pen_trail = QPen(QColor(255, 0, 255), 1.5)
            pen_calib = QPen(QColor(0, 255, 0), 2)
            font_small = QFont("Segoe UI", 9)

            # 1. ROI
            if vw._roi is not None:
                rx, ry, rw, rh = vw._roi
                dx1, dy1 = vw.video_to_display(rx, ry)
                dx2, dy2 = vw.video_to_display(rx + rw, ry + rh)
                p.setPen(pen_roi)
                p.drawRect(QRectF(dx1, dy1, dx2 - dx1, dy2 - dy1))
                p.setFont(font_small)
                p.drawText(QPointF(dx1 + 4, dy1 - 4), "ROI")

            # 2. 拖拽中的 ROI
            if vw._mode == vw.MODE_ROI and vw._drag_start is not None and vw._drag_end is not None:
                p.setPen(QPen(QColor(0, 255, 255), 2))
                r = QRectF(vw._drag_start, vw._drag_end)
                p.drawRect(r.normalized())

            # 3. 初始小球位置标记
            if vw._ball_pos is not None:
                vx, vy = vw._ball_pos
                dx, dy = vw.video_to_display(vx, vy)
                p.setPen(QPen(QColor(255, 0, 0), 2))
                p.setBrush(QColor(255, 0, 0, 60))
                p.drawEllipse(QPointF(dx, dy), 8, 8)
                p.setBrush(Qt.NoBrush)
                p.setFont(font_small)
                p.setPen(QColor(255, 100, 100))
                p.drawText(QPointF(dx + 12, dy - 8), f"起点({vx},{vy})")

            # 4. 标定点
            if vw._mode == vw.MODE_CALIBRATE or len(vw._calib_points) > 0:
                for i, (vx, vy) in enumerate(vw._calib_points):
                    dx, dy = vw.video_to_display(vx, vy)
                    p.setPen(pen_calib)
                    p.setBrush(QColor(0, 255, 0, 80))
                    p.drawEllipse(QPointF(dx, dy), 6, 6)
                    p.setBrush(Qt.NoBrush)
                    p.setFont(font_small)
                    label = f"P{i + 1}({vx},{vy})"
                    p.drawText(QPointF(dx + 10, dy - 5), label)

            # 4. 检测结果
            if vw._detection and vw._detection.get("found"):
                cx = vw._detection["x_px"]
                cy = vw._detection["y_px"]
                r = vw._detection.get("radius_px", 10)
                dx, dy = vw.video_to_display(cx, cy)
                dr = r * s

                p.setPen(pen_det)
                p.setBrush(Qt.NoBrush)
                if dr > 1:
                    p.drawEllipse(QPointF(dx, dy), dr, dr)
                p.setBrush(QColor(255, 0, 0))
                p.drawEllipse(QPointF(dx, dy), 3, 3)
                p.setBrush(Qt.NoBrush)
                p.setFont(font_small)
                p.setPen(QColor(0, 255, 0))
                p.drawText(QPointF(dx + 8, dy - 8),
                           f"({vw._detection['x_px']:.0f}, {vw._detection['y_px']:.0f})")

            # 5. 轨迹（按连续有效段分段绘制，不连接跨间隔点）
            if len(vw._trail) > 1:
                p.setPen(pen_trail)
                seg_pts = []
                for vx, vy in vw._trail:
                    if vx is None or vy is None or (isinstance(vx, float) and np.isnan(vx)):
                        # 遇到无效点：绘制当前段并重置
                        if len(seg_pts) > 1:
                            for i in range(len(seg_pts) - 1):
                                p.drawLine(seg_pts[i], seg_pts[i + 1])
                        seg_pts = []
                        continue
                    dx, dy = vw.video_to_display(vx, vy)
                    seg_pts.append(QPointF(dx, dy))
                # 绘制最后一段
                if len(seg_pts) > 1:
                    for i in range(len(seg_pts) - 1):
                        p.drawLine(seg_pts[i], seg_pts[i + 1])

        # ---- 帧号（所有模式下都显示） ----
        p.setPen(QColor(255, 255, 255))
        font_info = QFont("Consolas", 10)
        p.setFont(font_info)
        time_s = vw._current_frame_idx / vw._fps if vw._fps > 0 else 0
        p.drawText(QPointF(ox + 8, oy + 22),
                   f"Frame: {vw._current_frame_idx}  Time: {time_s:.3f}s")

    def mousePressEvent(self, event: QMouseEvent):
        self.mouse_pressed.emit(event.position())

    def mouseMoveEvent(self, event: QMouseEvent):
        self.mouse_moved.emit(event.position())

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.mouse_released.emit(event.position())

    def wheelEvent(self, event: QWheelEvent):
        self.wheel_zoomed.emit(event)
