"""
对话框模块
"""

import math
import yaml
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QFormLayout, QDialogButtonBox, QFileDialog,
    QMessageBox, QGroupBox, QGridLayout,
)
from PySide6.QtCore import Qt


class ScaleCalibrationDialog(QDialog):
    """比例尺标定对话框，显示详细标定信息。"""

    def __init__(self, p1: tuple, p2: tuple, pixel_distance: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("标定比例尺")
        self.setModal(True)
        self.resize(380, 320)

        self._result_scale = 0.0

        # 计算距离信息
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        euclidean = math.hypot(dx, dy)

        layout = QVBoxLayout(self)

        # 标定点信息
        info_group = QGroupBox("标定信息")
        grid = QGridLayout(info_group)
        grid.addWidget(QLabel("第一点坐标:"), 0, 0)
        grid.addWidget(QLabel(f"({p1[0]}, {p1[1]})"), 0, 1)
        grid.addWidget(QLabel("第二点坐标:"), 1, 0)
        grid.addWidget(QLabel(f"({p2[0]}, {p2[1]})"), 1, 1)
        grid.addWidget(QLabel("水平偏移 ΔX:"), 2, 0)
        grid.addWidget(QLabel(f"{dx:.1f} px"), 2, 1)
        grid.addWidget(QLabel("竖直距离 ΔY:"), 3, 0)
        self._px_label = QLabel(f"{dy:.1f} px")
        self._px_label.setStyleSheet("font-weight: bold; color: #2ecc71;")
        grid.addWidget(self._px_label, 3, 1)
        grid.addWidget(QLabel("直线距离:"), 4, 0)
        grid.addWidget(QLabel(f"{euclidean:.1f} px"), 4, 1)
        layout.addWidget(info_group)

        # 水平偏移警告
        if dx > 5:
            warn = QLabel(
                f"⚠ 两点水平偏差 {dx:.0f} px，请尽量保持竖直对齐。\n"
                f"   比例尺使用竖直距离 ΔY = {dy:.1f} px 计算。"
            )
            warn.setStyleSheet("color: #e67e22; font-weight: bold; padding: 4px;")
            warn.setWordWrap(True)
            layout.addWidget(warn)

        # 输入真实距离
        input_group = QGroupBox("输入真实距离")
        input_form = QFormLayout(input_group)

        self._dist_spin = QDoubleSpinBox()
        self._dist_spin.setRange(0.001, 10000.0)
        self._dist_spin.setDecimals(2)
        self._dist_spin.setValue(50.0)
        self._dist_spin.setSuffix(" mm")
        self._dist_spin.valueChanged.connect(self._update)
        input_form.addRow("真实距离:", self._dist_spin)

        self._scale_display = QLabel("= 0.000000 mm/px")
        self._scale_display.setStyleSheet("font-weight: bold; font-size: 13px;")
        input_form.addRow("比例尺:", self._scale_display)
        layout.addWidget(input_group)

        self._px_dist = dy  # ★ 使用竖直距离计算比例尺
        self._update()

        # 按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update(self):
        if self._px_dist <= 0:
            return
        real_mm = self._dist_spin.value()
        scale = real_mm / self._px_dist
        self._result_scale = scale
        self._scale_display.setText(f"= {scale:.6f} mm/px")

    def get_scale(self) -> float:
        return self._result_scale


class LoadConfigDialog(QDialog):
    """加载配置确认对话框。"""

    def __init__(self, config_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("加载配置")
        self.setModal(True)
        self.resize(400, 300)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"将从以下文件加载配置:"))
        layout.addWidget(QLabel(f"  {config_path}"))

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()
            preview = QLabel(content[:600])
            preview.setWordWrap(True)
            preview.setStyleSheet(
                "background: #f5f5f5; padding: 8px; font-family: Consolas; font-size: 9px;"
            )
            layout.addWidget(preview)
        except Exception:
            pass

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class BallCalibrateDialog(QDialog):
    """小球自标定对话框。

    用已知直径的小球标定比例尺。小球在轨迹平面内，
    无深度视差误差，比在量筒表面标定更准确。
    """

    def __init__(self, ball_diameter_mm: float = 1.5,
                 detected_radius_px: float | None = None,
                 detection_ok: bool = False,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("小球自标定")
        self.setModal(True)
        self.setMinimumWidth(420)

        self._result_scale = 0.0

        layout = QVBoxLayout(self)

        # ── 说明 ──
        hint = QLabel(
            "小球和下落轨迹在同一平面，用它标定比例尺无深度视差。\n"
            "请先确保小球在当前帧中清晰可见，且 ROI 已正确框选。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555; padding: 4px;")
        layout.addWidget(hint)

        # ── 小球直径 ──
        input_group = QGroupBox("小球参数")
        input_form = QFormLayout(input_group)

        self._diam_spin = QDoubleSpinBox()
        self._diam_spin.setRange(0.01, 100.0)
        self._diam_spin.setDecimals(3)
        self._diam_spin.setValue(ball_diameter_mm)
        self._diam_spin.setSuffix(" mm")
        self._diam_spin.valueChanged.connect(self._recalc)
        input_form.addRow("小球直径:", self._diam_spin)

        layout.addWidget(input_group)

        # ── 检测结果 ──
        result_group = QGroupBox("检测结果")
        result_form = QFormLayout(result_group)

        status_row = QHBoxLayout()
        self._status_icon = QLabel("—")
        self._status_icon.setStyleSheet("font-size: 20px;")
        self._status_text = QLabel("等待检测...")
        status_row.addWidget(self._status_icon)
        status_row.addWidget(self._status_text, 1)
        result_form.addRow("状态:", status_row)

        self._px_diam_label = QLabel("—")
        result_form.addRow("像素直径:", self._px_diam_label)

        self._scale_label = QLabel("—")
        self._scale_label.setStyleSheet("font-weight: bold; font-size: 15px;")
        result_form.addRow("比例尺:", self._scale_label)

        layout.addWidget(result_group)

        # ── 设置初始检测结果 ──
        self._detected_radius_px = detected_radius_px
        self._detection_ok = detection_ok
        if detection_ok and detected_radius_px is not None:
            self._show_success(detected_radius_px)
        elif not detection_ok:
            self._show_failure()

        self._recalc()

        # ── 按钮 ──
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _show_success(self, radius_px: float):
        self._detected_radius_px = radius_px
        self._detection_ok = True
        self._status_icon.setText("✓")
        self._status_icon.setStyleSheet("font-size: 20px; color: #2ecc71;")
        diam_px = radius_px * 2
        self._px_diam_label.setText(f"{diam_px:.2f} px  (半径 {radius_px:.2f} px)")
        self._status_text.setText("检测成功 — 小球在轨迹平面内，无深度视差")
        self._recalc()

    def _show_failure(self):
        self._detection_ok = False
        self._status_icon.setText("✗")
        self._status_icon.setStyleSheet("font-size: 20px; color: #e74c3c;")
        self._status_text.setText(
            "未检测到小球。请调整 ROI 或换一帧小球更清晰的画面。"
        )
        self._px_diam_label.setText("—")
        self._scale_label.setText("—")

    def _recalc(self):
        if not self._detection_ok or self._detected_radius_px is None:
            self._result_scale = 0.0
            return
        diam_mm = self._diam_spin.value()
        diam_px = self._detected_radius_px * 2
        if diam_px <= 0:
            self._result_scale = 0.0
            return
        scale = diam_mm / diam_px
        self._result_scale = scale
        self._scale_label.setText(f"{scale:.6f} mm/px")

    def _on_accept(self):
        if self._result_scale <= 0:
            QMessageBox.warning(self, "无效比例尺",
                                "请先确保小球被成功检测到。")
            return
        self.accept()

    def get_scale(self) -> float:
        return self._result_scale

    def update_detection(self, detected_radius_px: float | None,
                         detection_ok: bool):
        """外部更新检测结果。"""
        if detection_ok and detected_radius_px is not None:
            self._show_success(detected_radius_px)
        else:
            self._show_failure()
