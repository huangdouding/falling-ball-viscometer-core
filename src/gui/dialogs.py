"""
对话框模块
"""

import math
import yaml
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QFormLayout, QDialogButtonBox, QFileDialog,
    QMessageBox, QGroupBox, QGridLayout, QCheckBox,
)
from PySide6.QtCore import Qt


class ScaleCalibrationDialog(QDialog):
    """比例尺标定对话框。

    管表面两点 → 真实距离 → 管表面比例尺。
    可选视差修正：输入深度差 → 自动换算到球轨迹平面。
    """

    def __init__(self, p1: tuple, p2: tuple, pixel_distance: float,
                 ball_diameter_mm: float = 1.5, parent=None):
        super().__init__(parent)
        self.setWindowTitle("标定比例尺")
        self.setModal(True)
        self.resize(400, 400)

        self._raw_scale = 0.0
        self._result_scale = 0.0

        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        euclidean = math.hypot(dx, dy)
        self._px_dist = dy
        self._ball_diam = ball_diameter_mm

        layout = QVBoxLayout(self)

        # ── 标定点信息 ──
        info_group = QGroupBox("标定信息")
        grid = QGridLayout(info_group)
        grid.addWidget(QLabel("第一点:"), 0, 0)
        grid.addWidget(QLabel(f"({p1[0]}, {p1[1]})"), 0, 1)
        grid.addWidget(QLabel("第二点:"), 1, 0)
        grid.addWidget(QLabel(f"({p2[0]}, {p2[1]})"), 1, 1)
        grid.addWidget(QLabel("ΔY:"), 2, 0)
        px_label = QLabel(f"{dy:.1f} px")
        px_label.setStyleSheet("font-weight: bold; color: #2ecc71;")
        grid.addWidget(px_label, 2, 1)
        grid.addWidget(QLabel("ΔX / 直线:"), 3, 0)
        grid.addWidget(QLabel(f"{dx:.1f} px / {euclidean:.1f} px"), 3, 1)
        layout.addWidget(info_group)

        if dx > 5:
            warn = QLabel(
                f"⚠ 水平偏差 {dx:.0f} px，请尽量竖直对齐。"
                f"比例尺使用 ΔY = {dy:.1f} px 计算。"
            )
            warn.setStyleSheet("color: #e67e22; font-weight: bold; padding: 4px;")
            warn.setWordWrap(True)
            layout.addWidget(warn)

        # ── 真实距离 → 管表面比例尺 ──
        dist_group = QGroupBox("管表面比例尺（两点标定）")
        dist_form = QFormLayout(dist_group)

        self._dist_spin = QDoubleSpinBox()
        self._dist_spin.setRange(0.001, 10000.0)
        self._dist_spin.setDecimals(2)
        self._dist_spin.setValue(50.0)
        self._dist_spin.setSuffix(" mm")
        self._dist_spin.valueChanged.connect(self._update)
        dist_form.addRow("真实距离:", self._dist_spin)

        self._raw_scale_label = QLabel("= 0.000000 mm/px")
        dist_form.addRow("管表面:", self._raw_scale_label)
        layout.addWidget(dist_group)

        # ── 视差修正 ──
        corr_group = QGroupBox("视差修正（管表面 → 球轨迹平面）")
        corr_form = QFormLayout(corr_group)

        self._corr_enabled = QCheckBox("启用视差修正")
        self._corr_enabled.setChecked(False)
        self._corr_enabled.toggled.connect(self._update)
        corr_form.addRow("", self._corr_enabled)

        self._depth_spin = QDoubleSpinBox()
        self._depth_spin.setRange(0.1, 100.0)
        self._depth_spin.setDecimals(1)
        self._depth_spin.setValue(8.0)
        self._depth_spin.setSuffix(" mm")
        self._depth_spin.setToolTip(
            "管表面到球轨迹的垂直距离\n"
            "≈ 管壁厚度 + 小球半径 (通常 5-12mm)"
        )
        self._depth_spin.valueChanged.connect(self._update)
        corr_form.addRow("深度差:", self._depth_spin)

        self._corr_scale_label = QLabel("= — mm/px")
        self._corr_scale_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #2ecc71;")
        corr_form.addRow("修正后:", self._corr_scale_label)

        layout.addWidget(corr_group)

        # ── 预览 ──
        preview_group = QGroupBox("预览")
        preview_form = QFormLayout(preview_group)
        self._preview_label = QLabel("小球直径 ≈ — px")
        preview_form.addRow("", self._preview_label)
        layout.addWidget(preview_group)

        self._update()

        # ── 按钮 ──
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update(self):
        if self._px_dist <= 0:
            return
        real_mm = self._dist_spin.value()
        raw = real_mm / self._px_dist
        self._raw_scale = raw
        self._raw_scale_label.setText(f"= {raw:.6f} mm/px")

        if self._corr_enabled.isChecked():
            depth = self._depth_spin.value()
            # 视差公式：scale_ball = scale_tube × (D + depth) / D
            # 近似为 scale_ball ≈ scale_tube × (1 + depth/D_typical)
            # D_typical ≈ 120mm (智能手机到量筒的典型距离)
            d_est = 120.0
            corrected = raw * (d_est + depth) / d_est
            self._result_scale = corrected
            self._corr_scale_label.setText(f"= {corrected:.6f} mm/px")
            # 质量指示
            pct = (corrected / raw - 1.0) * 100
            self._corr_scale_label.setText(
                f"= {corrected:.6f} mm/px  (+{pct:.1f}%)"
            )
        else:
            self._result_scale = raw
            self._corr_scale_label.setText("= — mm/px (未启用)")

        # 小球像素预览
        ball_px = self._ball_diam / self._result_scale
        if ball_px < 5:
            quality = "⚠ 太小，检测困难"
            color = "#e74c3c"
        elif ball_px < 8:
            quality = "可接受"
            color = "#f39c12"
        else:
            quality = "✓ 良好"
            color = "#2ecc71"
        self._preview_label.setText(
            f"小球直径 ≈ {ball_px:.1f} px  ({quality})"
        )
        self._preview_label.setStyleSheet(f"color: {color};")

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
