"""
右侧参数面板（重构版）

分组：
  A. 实验参数
  B. 小球识别参数
  C. 终端速度判定参数
  [高级设置] 折叠面板

关键改动：
  - 鼠标滚轮不影响数值（NoWheel 组件）
  - 单位适合实验使用（mm, cm, m/s, Pa·s）
  - 内部自动转换为 SI 单位
"""

import os
import sys
import json
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLabel,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QPushButton,
    QHBoxLayout, QFileDialog, QMessageBox, QScrollArea,
)
from PySide6.QtCore import Qt, Signal, QTimer


_SETTINGS_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../config/settings.json")
)


def _safe_cfg_print(text: str):
    """安全打印配置日志到控制台，兼容 GBK 编码。"""
    try:
        print(text)
    except UnicodeEncodeError:
        safe = text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
            sys.stdout.encoding or "utf-8", errors="replace"
        )
        print(safe)

# 默认参数（程序启动/恢复默认时使用）
DEFAULT_CONFIG = {
    # — 实验参数 —
    "scale_mm_per_px": 0.071160,
    "ball_radius_mm": 0.750,
    "ball_density_kg_m3": 7850.0,
    "liquid_density_kg_m3": 950.0,
    "cylinder_radius_mm": 10.0,
    "liquid_height_mm": 400.0,
    "temperature_c": 40.0,
    "reference_viscosity_pa_s": 0.231,
    "enable_wall_correction": True,
    "enable_reynolds_correction": True,
    "gravity_m_s2": 9.80,
    "manual_fps": 240.0,
    # — 小球识别参数 —
    "color_mode": "dark_ball_on_bright_bg",
    "threshold_method": "otsu",
    "manual_threshold": 0,
    "min_area_px": 20,
    "max_area_px": 2000,
    "min_circularity": 0.350,
    "velocity_window_sec": 0.50,
    # — 高级识别选项 —
    "auto_size_params": True,
    "large_ball_mode": False,
    "auto_shrink_roi": False,
    "use_new_pipeline": False,
    # — 终端速度判定参数 —
    "terminal_window_sec": 0.80,
    "terminal_min_real_detection_rate": 0.80,
    "cv_threshold": 0.08,
    "r2_threshold": 0.990,
    "terminal_ignore_start_sec": 0.30,
    "terminal_ignore_end_sec": 0.50,
    # 分析区间手动覆盖（0=自动检测有效轨迹区间）
    "manual_start_frame": 0,
    "manual_end_frame": 0,
}


# ====================================================================
#  禁止滚轮组件
# ====================================================================
class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


# ====================================================================
#  带标签的数值行（方便复用）
# ====================================================================
class ParamRow(QWidget):
    """一行：标签 + 输入框 + 单位，禁止滚轮。"""

    def __init__(self, label: str, unit: str = "",
                 min_v: float = 0, max_v: float = 999999,
                 decimals: int = 2, default: float = 0,
                 parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.label = QLabel(label)
        self.label.setFixedWidth(72)
        layout.addWidget(self.label)

        self.spin = NoWheelDoubleSpinBox()
        self.spin.setRange(min_v, max_v)
        self.spin.setDecimals(decimals)
        self.spin.setValue(default)
        self.spin.setSingleStep((max_v - min_v) / 100)
        layout.addWidget(self.spin, 1)

        if unit:
            self.unit = QLabel(unit)
            self.unit.setFixedWidth(50)
            layout.addWidget(self.unit)

    def value(self):
        return self.spin.value()

    def setValue(self, v):
        self.spin.setValue(v)


class ParamIntRow(QWidget):
    """整数参数行。"""

    def __init__(self, label: str, unit: str = "",
                 min_v: int = 0, max_v: int = 999999,
                 default: int = 0, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.label = QLabel(label)
        self.label.setFixedWidth(72)
        layout.addWidget(self.label)

        self.spin = NoWheelSpinBox()
        self.spin.setRange(min_v, max_v)
        self.spin.setValue(default)
        layout.addWidget(self.spin, 1)

        if unit:
            self.unit = QLabel(unit)
            self.unit.setFixedWidth(50)
            layout.addWidget(self.unit)

    def value(self):
        return self.spin.value()

    def setValue(self, v):
        self.spin.setValue(v)


# ====================================================================
#  主面板
# ====================================================================
class ParameterPanel(QScrollArea):
    """右侧参数面板。"""

    open_video_requested = Signal()
    open_image_requested = Signal()
    test_frame_requested = Signal()
    analyze_requested = Signal()
    stop_requested = Signal()
    export_requested = Signal()
    clear_requested = Signal()
    config_saved = Signal(str)
    config_loaded = Signal(dict)
    profile_changed = Signal(str)  # "720x1280" | "1080x1920" | "custom"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loading_config = False
        self._runtime_config = {}
        self.setWidgetResizable(True)
        self.setMinimumWidth(300)
        self.setMaximumWidth(380)

        container = QWidget()
        self.setWidget(container)
        layout = QVBoxLayout(container)
        layout.setSpacing(6)

        # ============================================================
        #  操作按钮（顶部）
        # ============================================================
        self._btn_open = QPushButton("打开视频")
        self._btn_image = QPushButton("打开图片")
        self._btn_test = QPushButton("测试当前帧")
        self._btn_analyze = QPushButton("分析完整视频")
        self._btn_stop = QPushButton("停止")
        self._btn_stop.setEnabled(False)

        self._btn_open.clicked.connect(self.open_video_requested)
        self._btn_image.clicked.connect(self.open_image_requested)
        self._btn_test.clicked.connect(self.test_frame_requested)
        self._btn_analyze.clicked.connect(self.analyze_requested)
        self._btn_stop.clicked.connect(self.stop_requested)

        r1 = QHBoxLayout()
        r1.addWidget(self._btn_open)
        r1.addWidget(self._btn_image)
        r1.addWidget(self._btn_test)
        layout.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(self._btn_analyze, 3)
        r2.addWidget(self._btn_stop, 1)
        layout.addLayout(r2)

        # ============================================================
        #  A. 实验参数
        # ============================================================
        grp_a = QGroupBox("A. 实验参数")
        fa = QFormLayout(grp_a)
        fa.setSpacing(4)

        self._scale = ParamRow("比例尺", "mm/px", 0, 10000, 6, 0)

        # 分辨率档位：切换时自动加载该档位保存的比例尺
        self._profile_combo = NoWheelComboBox()
        self._profile_combo.addItem("720×1280 (竖屏)", "720x1280")
        self._profile_combo.addItem("1080×1920 (竖屏)", "1080x1920")
        self._profile_combo.addItem("自定义", "custom")
        self._profile_combo.setToolTip("切换分辨率档位，自动加载该档位保存的比例尺")
        self._profile_combo.currentIndexChanged.connect(self._on_profile_changed)

        self._radius = ParamRow("小球半径", "mm", 0.01, 100, 3, 1.0)
        self._ball_dens = ParamRow("小球密度", "kg/m³", 100, 20000, 0, 7800)
        self._liq_dens = ParamRow("液体密度", "kg/m³", 100, 20000, 0, 1260)
        self._cyl_radius = ParamRow("量筒半径", "mm", 0.1, 500, 1, 25.0)
        self._liq_height = ParamRow("液柱高度", "mm", 1, 2000, 0, 400)
        self._temp = ParamRow("温度", "°C", -10, 100, 1, 40)
        self._ref_visc = ParamRow("参考黏度", "Pa·s", 0, 100, 6, 0.231)
        self._ref_visc.setToolTip("用于相对误差计算（40°C蓖麻油 ≈ 0.231 Pa·s）")
        self._wall_corr = QCheckBox("启用壁面修正 (Ladenburg)")
        self._wall_corr.setChecked(True)
        self._reynolds_corr = QCheckBox("启用雷诺数修正 (Oseen)")
        self._reynolds_corr.setChecked(True)
        self._g = ParamRow("重力加速度", "m/s²", 1, 20, 2, 9.80)
        self._fps = ParamRow("视频帧率", "fps", 1, 10000, 0, 240)
        self._fps.setToolTip("iPhone 慢动作视频可能被 OpenCV 读为 30fps，请设为实际拍摄帧率 240")

        def add_row(f, row):
            f.addRow(row.label, row)

        add_row(fa, self._scale)
        fa.addRow("档位:", self._profile_combo)
        add_row(fa, self._radius)
        add_row(fa, self._ball_dens)
        add_row(fa, self._liq_dens)
        add_row(fa, self._cyl_radius)
        add_row(fa, self._liq_height)
        add_row(fa, self._temp)
        add_row(fa, self._ref_visc)
        fa.addRow("", self._wall_corr)
        fa.addRow("", self._reynolds_corr)
        add_row(fa, self._g)
        add_row(fa, self._fps)
        layout.addWidget(grp_a)

        # ============================================================
        #  B. 小球识别参数
        # ============================================================
        grp_b = QGroupBox("B. 小球识别参数")
        fb = QFormLayout(grp_b)
        fb.setSpacing(4)

        self._color_mode = NoWheelComboBox()
        self._color_mode.addItem("深色小球 / 浅色背景", "dark_ball_on_bright_bg")
        self._color_mode.addItem("浅色小球 / 深色背景", "bright_ball_on_dark_bg")
        fb.addRow("颜色模式:", self._color_mode)

        self._thresh_method = NoWheelComboBox()
        self._thresh_method.addItem("自动 Otsu", "otsu")
        self._thresh_method.addItem("手动阈值", "manual")
        self._thresh_method.currentIndexChanged.connect(self._on_thresh_method_change)
        fb.addRow("阈值方式:", self._thresh_method)

        self._thresh_val = NoWheelSpinBox()
        self._thresh_val.setRange(0, 255)
        self._thresh_val.setValue(0)
        self._thresh_val.setEnabled(False)
        fb.addRow("手动阈值:", self._thresh_val)

        self._min_area = ParamIntRow("最小面积", "px²", 1, 100000, 20)
        add_row(fb, self._min_area)

        self._max_area = ParamIntRow("最大面积", "px²", 1, 100000, 2000)
        add_row(fb, self._max_area)

        self._min_circ = ParamRow("最小圆度", "", 0.01, 1.0, 3, 0.50)
        add_row(fb, self._min_circ)

        layout.addWidget(grp_b)

        # ============================================================
        #  C. 终端速度判定参数
        # ============================================================
        grp_c = QGroupBox("C. 终端速度判定参数")
        fc = QFormLayout(grp_c)
        fc.setSpacing(4)

        self._term_win = ParamRow("速度判定窗口", "s", 0.05, 10, 2, 0.80)
        self._vel_win = ParamRow("速度平滑窗口", "s", 0.05, 10, 2, 0.50)
        self._vel_win.setToolTip("滑动窗口 y-t 拟合的窗口时长，仅影响速度图显示")
        self._cv_th = ParamRow("Cv 阈值", "", 0.001, 1.0, 4, 0.030)
        self._r2_th = ParamRow("R² 阈值", "", 0.5, 1.0, 4, 0.995)
        self._ignore_start = ParamRow("忽略起始段", "s", 0, 100, 2, 0)
        self._ignore_end = ParamRow("忽略结束段", "s", 0, 100, 2, 0)

        add_row(fc, self._term_win)
        add_row(fc, self._vel_win)
        add_row(fc, self._cv_th)
        add_row(fc, self._r2_th)
        add_row(fc, self._ignore_start)
        add_row(fc, self._ignore_end)
        layout.addWidget(grp_c)

        # ============================================================
        #  高级设置（折叠）
        # ============================================================
        self._adv_btn = QPushButton("▶ 高级设置")
        self._adv_btn.setStyleSheet("text-align: left; padding: 4px;")
        self._adv_btn.setCheckable(True)
        self._adv_btn.clicked.connect(self._toggle_advanced)
        layout.addWidget(self._adv_btn)

        self._adv_panel = QWidget()
        self._adv_panel.setVisible(False)
        adv_layout = QFormLayout(self._adv_panel)
        adv_layout.setSpacing(3)

        # ---- 手动覆盖分析区间（0=自动） ----
        self._analysis_start = NoWheelSpinBox()
        self._analysis_start.setRange(0, 999999)
        self._analysis_start.setValue(0)
        self._analysis_start.setToolTip("0=自动检测有效轨迹起始帧；非0=手动覆盖起始帧")
        self._analysis_end = NoWheelSpinBox()
        self._analysis_end.setRange(0, 999999)
        self._analysis_end.setValue(0)
        self._analysis_end.setToolTip("0=自动检测有效轨迹结束帧；非0=手动覆盖结束帧")
        adv_layout.addRow("手动覆盖起始帧:", self._analysis_start)
        adv_layout.addRow("手动覆盖结束帧:", self._analysis_end)

        self._adaptive_th = QCheckBox("启用自适应阈值")
        adv_layout.addRow("", self._adaptive_th)

        self._blur_ksize = NoWheelSpinBox()
        self._blur_ksize.setRange(1, 31)
        self._blur_ksize.setSingleStep(2)
        self._blur_ksize.setValue(5)

        self._morph_ksize = NoWheelSpinBox()
        self._morph_ksize.setRange(1, 15)
        self._morph_ksize.setSingleStep(2)
        self._morph_ksize.setValue(3)

        self._max_jump = NoWheelSpinBox()
        self._max_jump.setRange(1, 1000)
        self._max_jump.setValue(80)

        self._min_valid = NoWheelSpinBox()
        self._min_valid.setRange(3, 10000)
        self._min_valid.setValue(5)

        self._smooth_method = NoWheelComboBox()
        self._smooth_method.addItem("moving_average")
        self._smooth_method.addItem("savitzky_golay")

        self._smooth_win = NoWheelSpinBox()
        self._smooth_win.setRange(3, 31)
        self._smooth_win.setSingleStep(2)
        self._smooth_win.setValue(5)

        # ---- 检测方法 ----
        self._detect_method = NoWheelComboBox()
        self._detect_method.addItem("auto（自动回退）", "auto")
        self._detect_method.addItem("blob", "blob")
        self._detect_method.addItem("threshold_contour", "threshold_contour")
        self._detect_method.addItem("background_subtraction", "background_subtraction")

        # ---- 中心运动带 ----
        self._center_band = QCheckBox("启用中心运动带")
        self._center_band.setChecked(True)
        self._center_band_left = ParamRow("中心带左边界", "ratio", 0, 1, 2, 0.30)
        self._center_band_right = ParamRow("中心带右边界", "ratio", 0, 1, 2, 0.70)

        # ---- 半径范围 ----
        self._exp_rad_min = ParamRow("预期半径下限", "px", 0.1, 50, 1, 1.0)
        self._exp_rad_max = ParamRow("预期半径上限", "px", 0.1, 50, 1, 4.0)

        # ---- 搜索窗口 ----
        self._search_win = NoWheelSpinBox()
        self._search_win.setRange(10, 500)
        self._search_win.setValue(60)

        # ---- 非对称搜索窗口（v5 追踪） ----
        self._search_win_y_up = NoWheelSpinBox()
        self._search_win_y_up.setRange(0, 200)
        self._search_win_y_up.setValue(20)
        self._search_win_y_down = NoWheelSpinBox()
        self._search_win_y_down.setRange(10, 500)
        self._search_win_y_down.setValue(120)

        # ---- 轨迹质量标准（v5 新增） ----
        self._quality_min_frames = NoWheelSpinBox()
        self._quality_min_frames.setRange(20, 5000)
        self._quality_min_frames.setValue(80)
        self._quality_r2 = ParamRow("段 R² 阈值", "", 0.9, 1.0, 4, 0.995)
        self._quality_cv = ParamRow("段 Cv 阈值", "", 0.001, 1.0, 4, 0.05)

        # ---- 下落中心线（v3） ----
        self._fall_axis_x = NoWheelSpinBox()
        self._fall_axis_x.setRange(0, 10000)
        self._fall_axis_x.setValue(0)
        self._fall_axis_x.setToolTip("0=自动估计，非0=手动指定全局X坐标")

        self._axis_deviation = NoWheelSpinBox()
        self._axis_deviation.setRange(10, 500)
        self._axis_deviation.setValue(50)

        self._enable_long_line = QCheckBox("启用长线排除")
        self._enable_long_line.setChecked(True)

        self._auto_size = QCheckBox("自动识别参数")
        self._auto_size.setToolTip("启用后自动根据小球半径设定面积/半径/圆度范围，覆盖下方手动设置的识别参数")
        self._auto_size.setChecked(True)
        self._auto_size.setEnabled(False)
        self._large_ball = QCheckBox("大球模式")
        self._large_ball.setToolTip("适配较大球径的检测参数")
        self._large_ball.setEnabled(False)
        self._auto_shrink_roi = QCheckBox("自动缩窄检测区域")
        self._auto_shrink_roi.setToolTip("围绕下落中心线缩窄检测区域以排除管壁等干扰")

        # ---- 新版管线 ----
        self._use_new_pipeline = QCheckBox("新管线 (pipeline.py)")
        self._use_new_pipeline.setToolTip(
            "启用新版简化检测管线 (CandidateDetector+BallTracker)，\n"
            "替代旧版 BallDetector。需要重新分析才能生效。"
        )
        self._use_new_pipeline.setChecked(False)
        self._use_new_pipeline.setVisible(False)

        adv_layout.addRow("模糊核大小:", self._blur_ksize)
        adv_layout.addRow("形态学核:", self._morph_ksize)
        adv_layout.addRow("检测方法:", self._detect_method)
        adv_layout.addRow("最大跳变(px):", self._max_jump)
        adv_layout.addRow("搜索窗口(px):", self._search_win)
        adv_layout.addRow("搜索窗向上(px):", self._search_win_y_up)
        adv_layout.addRow("搜索窗向下(px):", self._search_win_y_down)
        adv_layout.addRow("最少有效点:", self._min_valid)
        adv_layout.addRow("质量最少帧数:", self._quality_min_frames)
        add_row(adv_layout, self._quality_r2)
        add_row(adv_layout, self._quality_cv)
        adv_layout.addRow("平滑方式:", self._smooth_method)
        adv_layout.addRow("平滑窗口:", self._smooth_win)
        adv_layout.addRow("", self._center_band)
        add_row(adv_layout, self._center_band_left)
        add_row(adv_layout, self._center_band_right)
        adv_layout.addRow("预期半径:", self._exp_rad_min)
        adv_layout.addRow("预期半径:", self._exp_rad_max)
        adv_layout.addRow("下落中心线 X:", self._fall_axis_x)
        adv_layout.addRow("中心线偏差(px):", self._axis_deviation)
        adv_layout.addRow("", self._enable_long_line)
        adv_layout.addRow("", self._auto_size)
        adv_layout.addRow("", self._large_ball)
        adv_layout.addRow("", self._auto_shrink_roi)
        adv_layout.addRow("", self._use_new_pipeline)
        layout.addWidget(self._adv_panel)

        # ============================================================
        #  底部按钮
        # ============================================================
        btn_row = QHBoxLayout()
        self._btn_save_cfg = QPushButton("保存配置")
        self._btn_load_cfg = QPushButton("加载配置")
        self._btn_export = QPushButton("导出结果")
        self._btn_clear = QPushButton("清空结果")

        self._btn_save_cfg.clicked.connect(self._on_save_yaml)
        self._btn_load_cfg.clicked.connect(self._on_load_yaml)
        self._btn_export.clicked.connect(self.export_requested)
        self._btn_clear.clicked.connect(self.clear_requested)

        btn_row.addWidget(self._btn_save_cfg)
        btn_row.addWidget(self._btn_load_cfg)
        btn_row.addWidget(self._btn_export)
        btn_row.addWidget(self._btn_clear)
        layout.addLayout(btn_row)

        # — 保存参数 / 恢复默认 —
        persist_row = QHBoxLayout()
        self._btn_save = QPushButton("保存参数")
        self._btn_save.setToolTip("将当前 UI 参数保存到 config/settings.json")
        self._btn_reset = QPushButton("恢复默认")
        self._btn_reset.setToolTip("恢复出厂默认参数并覆盖 settings.json")

        self._btn_save.clicked.connect(self._on_save_clicked)
        self._btn_reset.clicked.connect(self._on_reset_default)

        persist_row.addWidget(self._btn_save, 1)
        persist_row.addWidget(self._btn_reset, 1)
        layout.addLayout(persist_row)

        layout.addStretch()

        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(2000)
        self._autosave_timer.timeout.connect(self._autosave_now)
        self._connect_autosave_signals()

    def _connect_autosave_signals(self):
        """Save edited parameters shortly after the user changes them."""
        for spin in self.findChildren(QSpinBox):
            spin.valueChanged.connect(self._schedule_autosave)
        for spin in self.findChildren(QDoubleSpinBox):
            spin.valueChanged.connect(self._schedule_autosave)
        for combo in self.findChildren(QComboBox):
            combo.currentIndexChanged.connect(self._schedule_autosave)
        for checkbox in self.findChildren(QCheckBox):
            checkbox.toggled.connect(self._schedule_autosave)


    def _schedule_autosave(self, *args):
        if not self._loading_config:
            self._autosave_timer.start()

    def _autosave_now(self):
        if self._loading_config:
            return
        try:
            self.save_settings()
        except Exception:
            pass

    def set_runtime_config(self, updates: dict, *, autosave: bool = True):
        """Store non-widget parameters such as ROI in settings.json."""
        for key, value in updates.items():
            if value is None:
                self._runtime_config.pop(key, None)
            else:
                self._runtime_config[key] = value
        if autosave:
            self._schedule_autosave()

    # ================================================================
    #  阈值方式切换
    # ================================================================
    def _on_thresh_method_change(self, idx):
        self._thresh_val.setEnabled(idx == 1)  # Manual

    # ================================================================
    #  高级设置折叠
    # ================================================================
    def _toggle_advanced(self, checked):
        self._adv_panel.setVisible(checked)
        self._adv_btn.setText("▼ 高级设置" if checked else "▶ 高级设置")

    # ================================================================
    #  分辨率档位
    # ================================================================

    def get_profile_key(self) -> str:
        """返回当前档位键: "720x1280" / "1080x1920" / "custom" """
        return self._profile_combo.currentData()

    def set_profile_by_key(self, key: str):
        """按 key 设置档位。"""
        idx = self._profile_combo.findData(key)
        if idx >= 0:
            self._profile_combo.setCurrentIndex(idx)

    def _on_profile_changed(self, idx: int):
        key = self._profile_combo.itemData(idx)
        if key:
            self.profile_changed.emit(key)

    # ================================================================
    #  强制提交所有 SpinBox 当前文本（解决 Qt 失焦前 value() 旧值问题）
    # ================================================================
    def commit_editors(self):
        """强制所有数值输入框确认当前编辑文本。"""
        for spin in self.findChildren(QSpinBox):
            spin.interpretText()
        for spin in self.findChildren(QDoubleSpinBox):
            spin.interpretText()

    # ================================================================
    #  参数读取（只输出标准字段名）
    # ================================================================
    def get_config(self) -> dict:
        """从 UI 控件读取当前值，返回只含标准字段名的配置字典。"""
        self.commit_editors()
        scale = self._scale.value() if self._scale.value() > 0 else None
        color_mode_data = self._color_mode.currentData()
        threshold_method_data = self._thresh_method.currentData()
        cfg = {
            # — 实验参数（标准字段，仅 mm 单位，无 SI 副本） —
            "scale_mm_per_px": scale,
            "profile": self.get_profile_key(),
            "ball_radius_mm": self._radius.value(),
            "ball_density_kg_m3": self._ball_dens.value(),
            "liquid_density_kg_m3": self._liq_dens.value(),
            "cylinder_radius_mm": self._cyl_radius.value(),
            "liquid_height_mm": self._liq_height.value(),
            "temperature_c": self._temp.value(),
            "reference_viscosity_pa_s": self._ref_visc.value(),
            "enable_wall_correction": self._wall_corr.isChecked(),
            "enable_reynolds_correction": self._reynolds_corr.isChecked(),
            "gravity_m_s2": self._g.value(),
            "manual_fps": self._fps.value(),

            # — 小球识别参数（保留双名兼容 detector） —
            "color_mode": color_mode_data,
            "threshold_mode": color_mode_data,
            "threshold_method": threshold_method_data,
            "threshold_value": None
                if self._thresh_method.currentIndex() == 0
                else self._thresh_val.value(),
            "manual_threshold": None
                if self._thresh_method.currentIndex() == 0
                else self._thresh_val.value(),
            "adaptive_threshold": self._adaptive_th.isChecked(),
            "min_area_px": self._min_area.value(),
            "max_area_px": self._max_area.value(),
            "min_circularity": self._min_circ.value(),
            "detect_method": self._detect_method.currentData(),
            "expected_radius_px_min": self._exp_rad_min.value(),
            "expected_radius_px_max": self._exp_rad_max.value(),
            "center_band_enabled": self._center_band.isChecked(),
            "center_band_left_ratio": self._center_band_left.value(),
            "center_band_right_ratio": self._center_band_right.value(),
            "search_window_radius": self._search_win.value(),
            "search_win_w": self._search_win.value(),
            "search_win_y_up": self._search_win_y_up.value(),
            "search_win_y_down": self._search_win_y_down.value(),
            "gaussian_blur_ksize": self._blur_ksize.value()
                if self._blur_ksize.value() % 2 == 1
                else self._blur_ksize.value() + 1,
            "morph_kernel_size": self._morph_ksize.value()
                if self._morph_ksize.value() % 2 == 1
                else self._morph_ksize.value() + 1,
            "max_jump_px": self._max_jump.value(),

            # — 终端速度判定参数 —
            "terminal_window_sec": self._term_win.value(),
            "terminal_min_real_detection_rate": 0.80,
            "cv_threshold": self._cv_th.value(),
            "r2_threshold": self._r2_th.value(),
            "terminal_ignore_start_sec": self._ignore_start.value(),
            "terminal_ignore_end_sec": self._ignore_end.value(),

            # — 手动覆盖分析区间（0=自动） —
            "manual_start_frame": self._analysis_start.value(),
            "manual_end_frame": self._analysis_end.value(),

            # — 轨迹质量 / 平滑 —
            "quality_min_frames": self._quality_min_frames.value(),
            "quality_r2_threshold": self._quality_r2.value(),
            "quality_cv_threshold": self._quality_cv.value(),
            "smooth_method": self._smooth_method.currentText(),
            "smooth_window": self._smooth_win.value(),
            "velocity_window_sec": self._vel_win.value(),
            "save_marked_video": True,
            "save_plots": True,

            # — 高级识别选项 —
            "auto_size_params": True,
            "large_ball_mode": self._radius.value() >= 1.0,
            "auto_shrink_roi": self._auto_shrink_roi.isChecked(),
            "use_new_pipeline": False,

            # — 运动优先参数 —
            "fall_axis_x": self._fall_axis_x.value()
                if self._fall_axis_x.value() > 0 else None,
            "allowed_axis_deviation_px": self._axis_deviation.value(),
            "enable_long_line_rejection": self._enable_long_line.isChecked(),
        }
        cfg.update(self._runtime_config)
        try:
            from src.tracking import (
                _compute_dynamic_detection_params,
                _compute_dynamic_tracking_params,
            )
            _compute_dynamic_detection_params(cfg)
            _compute_dynamic_tracking_params(cfg)
        except Exception:
            # Keep UI collection robust; full analysis will report detailed
            # parameter errors if radius/scale are invalid.
            pass
        return cfg

    # ================================================================
    #  参数写入（支持 SI 单位 / mm / 旧键名混合输入）
    # ================================================================
    def set_config(self, cfg: dict):
        """将配置字典回填到 UI 控件。

        同时支持:
          - SI 单位键 (ball_radius_m) — 旧版 config.yaml
          - mm 键 (ball_radius_mm) — settings.json / 新版
          - 旧键名 (ignore_start_sec) 和新键名 (terminal_ignore_start_sec)
        """
        self._loading_config = True
        self._runtime_config = {
            k: cfg[k]
            for k in ("roi", "detector_profile", "init_ball_x", "init_ball_y")
            if k in cfg and cfg.get(k) is not None
        }

        def _get(cfg, *keys, default=None):
            """从 cfg 中依次尝试各 key，返回第一个存在的值。"""
            for k in keys:
                v = cfg.get(k)
                if v is not None:
                    return v
            return default

        def _set_d(spin, *keys, default=None, **_):
            v = _get(cfg, *keys, default=default)
            if v is not None:
                spin.setValue(float(v))

        def _set_i(spin, *keys, default=None, **_):
            v = _get(cfg, *keys, default=default)
            if v is not None:
                spin.setValue(int(v))

        # — 实验参数 —
        _set_d(self._scale, "scale_mm_per_px", default=0)
        # 恢复档位
        prof = _get(cfg, "profile", default=None)
        if prof:
            self.set_profile_by_key(prof)
        # ball_radius: 优先 mm，其次 m
        mm = _get(cfg, "ball_radius_mm")
        if mm is not None:
            self._radius.setValue(float(mm))
        elif cfg.get("ball_radius_m") is not None:
            self._radius.setValue(cfg["ball_radius_m"] * 1000)
        _set_d(self._ball_dens, "ball_density_kg_m3", default=7800)
        _set_d(self._liq_dens, "liquid_density_kg_m3", default=1260)
        # cylinder_radius: 优先 mm，其次 m
        mm = _get(cfg, "cylinder_radius_mm")
        if mm is not None:
            self._cyl_radius.setValue(float(mm))
        elif cfg.get("cylinder_radius_m") is not None:
            self._cyl_radius.setValue(cfg["cylinder_radius_m"] * 1000)
        # liquid_height: 优先 mm，其次 m
        mm = _get(cfg, "liquid_height_mm")
        if mm is not None:
            self._liq_height.setValue(float(mm))
        elif cfg.get("liquid_height_m") is not None:
            self._liq_height.setValue(cfg["liquid_height_m"] * 1000)
        _set_d(self._temp, "temperature_c", default=40)
        _set_d(self._ref_visc, "reference_viscosity_pa_s", default=0.231)
        self._wall_corr.setChecked(cfg.get("enable_wall_correction", True))
        self._reynolds_corr.setChecked(cfg.get("enable_reynolds_correction", True))
        _set_d(self._g, "g_m_s2", "gravity_m_s2", default=9.80)
        _set_d(self._fps, "manual_fps", "fps", default=240)

        # — 小球识别参数 —
        color_mode = _get(cfg, "color_mode", "threshold_mode", default="dark_ball_on_bright_bg")
        idx = self._color_mode.findData(color_mode)
        if idx >= 0:
            self._color_mode.setCurrentIndex(idx)
        else:
            # 尝试按文本匹配（settings 存中文显示名）
            txt_idx = self._color_mode.findText(str(color_mode))
            if txt_idx >= 0:
                self._color_mode.setCurrentIndex(txt_idx)

        # 阈值方式
        thresh_method = _get(cfg, "threshold_method", default="otsu")
        idx = self._thresh_method.findData(thresh_method)
        if idx >= 0:
            self._thresh_method.setCurrentIndex(idx)
        else:
            # 从旧版 threshold_value 推断
            tv = cfg.get("threshold_value")
            if tv is not None and tv > 0:
                self._thresh_method.setCurrentIndex(1)

        tv = cfg.get("threshold_value") or cfg.get("manual_threshold")
        if tv is not None and tv > 0:
            self._thresh_val.setValue(int(tv))

        self._adaptive_th.setChecked(cfg.get("adaptive_threshold", False))
        _set_i(self._min_area, "min_area_px", default=20)
        _set_i(self._max_area, "max_area_px", default=2000)
        _set_d(self._min_circ, "min_circularity", default=0.50)
        _set_i(self._blur_ksize, "gaussian_blur_ksize", default=5)
        _set_i(self._morph_ksize, "morph_kernel_size", default=3)
        _set_i(self._max_jump, "max_jump_px", default=80)
        _set_i(self._min_valid, "min_valid_points", default=5)

        dm = _get(cfg, "detect_method", default="auto")
        idx = self._detect_method.findData(dm)
        if idx >= 0:
            self._detect_method.setCurrentIndex(idx)
        _set_d(self._exp_rad_min, "expected_radius_px_min", default=1.0)
        _set_d(self._exp_rad_max, "expected_radius_px_max", default=4.0)
        self._center_band.setChecked(cfg.get("center_band_enabled", True))
        _set_d(self._center_band_left, "center_band_left_ratio", default=0.30)
        _set_d(self._center_band_right, "center_band_right_ratio", default=0.70)
        _set_i(self._search_win, "search_window_radius", "search_win_w", default=60)
        _set_i(self._search_win_y_up, "search_win_y_up", default=20)
        _set_i(self._search_win_y_down, "search_win_y_down", default=120)
        _set_i(self._quality_min_frames, "quality_min_frames", default=80)
        _set_d(self._quality_r2, "quality_r2_threshold", default=0.995)
        _set_d(self._quality_cv, "quality_cv_threshold", default=0.05)

        # — 终端速度判定参数 —
        _set_d(self._term_win, "terminal_window_sec", default=0.80)
        _set_d(self._vel_win, "velocity_window_sec", default=0.50)
        _set_d(self._cv_th, "cv_threshold", default=0.08)
        _set_d(self._r2_th, "r2_threshold", default=0.990)
        _set_d(self._ignore_start,
               "terminal_ignore_start_sec", "ignore_start_sec",
               default=0.30)
        _set_d(self._ignore_end,
               "terminal_ignore_end_sec", "ignore_end_sec",
               default=0.50)

        # — 手动覆盖分析区间 —
        _set_i(self._analysis_start, "manual_start_frame", "analysis_start_frame", default=0)
        _set_i(self._analysis_end, "manual_end_frame", "analysis_end_frame", default=0)

        sm = _get(cfg, "smooth_method", default="moving_average")
        idx = self._smooth_method.findText(sm)
        if idx >= 0:
            self._smooth_method.setCurrentIndex(idx)
        _set_i(self._smooth_win, "smooth_window", default=5)

        # — 运动优先参数 —
        fa = _get(cfg, "fall_axis_x", default=None)
        if fa is not None and fa > 0:
            self._fall_axis_x.setValue(int(fa))
        else:
            self._fall_axis_x.setValue(0)
        _set_i(self._axis_deviation, "allowed_axis_deviation_px", default=50)
        self._enable_long_line.setChecked(cfg.get("enable_long_line_rejection", True))
        self._auto_size.setChecked(True)
        self._large_ball.setChecked(self._radius.value() >= 1.0)
        self._auto_shrink_roi.setChecked(cfg.get("auto_shrink_roi", False))
        self._use_new_pipeline.setChecked(False)

        self._on_thresh_method_change(self._thresh_method.currentIndex())
        self._loading_config = False

    # ================================================================
    #  按钮状态
    # ================================================================
    def set_analyzing(self, analyzing: bool):
        for w in [self._btn_open, self._btn_image, self._btn_test,
                  self._btn_analyze, self._btn_save_cfg, self._btn_load_cfg,
                  self._btn_export, self._btn_save, self._btn_reset]:
            w.setEnabled(not analyzing)
        self._btn_stop.setEnabled(analyzing)

    # ================================================================
    #  保存参数 / 加载参数（自动 config/settings.json）
    # ================================================================
    def save_settings(self):
        """将当前 UI 参数保存到 config/settings.json，写后验证并回填 UI。"""
        from src.utils import normalize_config_keys

        cfg = self.get_config()
        cfg = normalize_config_keys(cfg)

        # 合并已保存的 scale_per_resolution，避免覆盖丢失
        try:
            with open(_SETTINGS_PATH, "r", encoding="utf-8") as _f:
                _old = json.load(_f)
            if "scale_per_resolution" in _old and "scale_per_resolution" not in cfg:
                cfg["scale_per_resolution"] = _old["scale_per_resolution"]
        except Exception:
            pass

        # ★ 保存当前比例尺到当前分辨率档位
        try:
            _cur_scale = cfg.get("scale_mm_per_px")
            _cur_profile = self.get_profile_key()
            if _cur_scale and _cur_scale > 0 and _cur_profile and _cur_profile != "custom":
                if "scale_per_resolution" not in cfg or cfg["scale_per_resolution"] is None:
                    cfg["scale_per_resolution"] = {}
                cfg["scale_per_resolution"][_cur_profile] = _cur_scale
        except Exception:
            pass

        try:
            os.makedirs(os.path.dirname(_SETTINGS_PATH), exist_ok=True)
            with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)

            # ★ 写后验证：重新读取确认
            with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            saved = normalize_config_keys(saved)

            # Do not write back into widgets during autosave; doing so can
            # steal focus while the user is editing a numeric field.

            n_keys = len(saved)
            _safe_cfg_print(f"[CONFIG] saved & verified: {_SETTINGS_PATH} ({n_keys} keys)")
            _safe_cfg_print(f"[CONFIG]   scale_mm_per_px       = {saved.get('scale_mm_per_px', '?')}")
            _safe_cfg_print(f"[CONFIG]   ball_radius_mm        = {saved.get('ball_radius_mm', '?')} mm")
            _safe_cfg_print(f"[CONFIG]   ball_density_kg_m3    = {saved.get('ball_density_kg_m3', '?')} kg/m3")
            _safe_cfg_print(f"[CONFIG]   liquid_density_kg_m3  = {saved.get('liquid_density_kg_m3', '?')} kg/m3")
            _safe_cfg_print(f"[CONFIG]   manual_fps            = {saved.get('manual_fps', '?')} fps")
            _safe_cfg_print(f"[CONFIG]   verified")

            self.config_saved.emit(_SETTINGS_PATH)
        except Exception as e:
            QMessageBox.warning(self, "保存失败",
                                f"无法保存参数到 {_SETTINGS_PATH}:\n{e}")
            raise

    def _on_save_clicked(self):
        """「保存参数」按钮点击：保存并确认弹窗。"""
        self.save_settings()
        QMessageBox.information(self, "保存成功",
                               f"参数已保存并验证:\n{_SETTINGS_PATH}",
                               QMessageBox.Ok)

    def load_settings(self) -> bool:
        """从 config/settings.json 加载参数并回填 UI。

        Returns:
            True 加载成功, False 文件不存在或加载失败
        """
        from src.utils import normalize_config_keys

        if not os.path.exists(_SETTINGS_PATH):
            print(f"[CONFIG] settings.json 不存在: {_SETTINGS_PATH}")
            return False
        try:
            with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg = normalize_config_keys(cfg)
            self.set_config(cfg)
            n_keys = len(cfg)
            _safe_cfg_print(f"[CONFIG] loaded {n_keys} keys from {_SETTINGS_PATH}")
            _safe_cfg_print(f"[CONFIG]   scale_mm_per_px       = {cfg.get('scale_mm_per_px', '?')}")
            _safe_cfg_print(f"[CONFIG]   ball_radius_mm        = {cfg.get('ball_radius_mm', '?')} mm")
            _safe_cfg_print(f"[CONFIG]   ball_density_kg_m3    = {cfg.get('ball_density_kg_m3', '?')} kg/m3")
            _safe_cfg_print(f"[CONFIG]   liquid_density_kg_m3  = {cfg.get('liquid_density_kg_m3', '?')} kg/m3")
            _safe_cfg_print(f"[CONFIG]   cylinder_radius_mm    = {cfg.get('cylinder_radius_mm', '?')} mm")
            _safe_cfg_print(f"[CONFIG]   liquid_height_mm      = {cfg.get('liquid_height_mm', '?')} mm")
            _safe_cfg_print(f"[CONFIG]   temperature_c         = {cfg.get('temperature_c', '?')} C")
            _safe_cfg_print(f"[CONFIG]   reference_viscosity_pa_s = {cfg.get('reference_viscosity_pa_s', '?')} Pa.s")
            _safe_cfg_print(f"[CONFIG]   gravity_m_s2          = {cfg.get('gravity_m_s2', '?')} m/s2")
            _safe_cfg_print(f"[CONFIG]   manual_fps            = {cfg.get('manual_fps', '?')} fps")
            return True
        except Exception as e:
            print(f"[CONFIG] 加载 settings.json 失败: {e}")
            return False

    def _on_reset_default(self):
        """恢复默认参数并写回 settings.json。"""
        reply = QMessageBox.question(
            self, "恢复默认",
            "确定要恢复所有参数到出厂默认值吗？\n"
            "当前修改将丢失。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.set_config(DEFAULT_CONFIG)
        # 使用 save_settings 完成归一化→写入→验证→回填
        self.save_settings()
        QMessageBox.information(self, "恢复默认", "参数已恢复为出厂默认值并保存。")

    # ================================================================
    #  YAML 配置导入 / 导出（高级用户）
    # ================================================================
    def _on_save_yaml(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 YAML 配置", "config.yaml", "YAML (*.yaml)"
        )
        if path:
            cfg = self.get_config()
            import yaml
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
            self.config_saved.emit(path)

    def _on_load_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "导入 YAML 配置", "", "YAML (*.yaml)"
        )
        if path:
            try:
                import yaml
                with open(path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                self.set_config(cfg)
                self.config_loaded.emit(cfg)
            except Exception as e:
                QMessageBox.warning(self, "加载失败", f"无法加载配置文件:\n{e}")
