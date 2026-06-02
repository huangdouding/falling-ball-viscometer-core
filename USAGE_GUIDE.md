# 落球法液体黏度自动测量系统 — 使用指南

> 基于机器视觉自动追踪与误差修正的落球法液体黏度测量实验改进

---

## 目录

- [1. 环境准备与安装](#1-环境准备与安装)
- [2. 拍摄实验视频](#2-拍摄实验视频)
- [3. 配置参数](#3-配置参数)
- [4. GUI 图形界面模式（推荐）](#4-gui-图形界面模式推荐)
- [5. 命令行模式](#5-命令行模式)
- [6. 批量处理模式](#6-批量处理模式)
- [7. 输出文件解读](#7-输出文件解读)
- [8. 常见问题与参数调优](#8-常见问题与参数调优)

---

## 1. 环境准备与安装

### 1.1 系统要求

- **Python** ≥ 3.8（推荐 3.10 或 3.11）
- **操作系统**：Windows / macOS / Linux
- **可选**：PyInstaller（用于打包独立可执行文件）

### 1.2 安装步骤

```bash
# 克隆仓库（含子模块）
git clone --recursive https://github.com/huangdouding/falling-ball-viscometer-core.git
cd falling-ball-viscometer-core

# 安装依赖
pip install -r requirements.txt
```

核心依赖：

| 包名 | 用途 |
|------|------|
| `opencv-python` | 视频读取与图像处理 |
| `numpy`、`pandas` | 数值计算与数据管理 |
| `matplotlib`、`scipy` | 绘图与拟合 |
| `PySide6` | 图形界面 |
| `PyYAML` | 配置解析 |

### 1.3 验证安装

```bash
python -c "import cv2; import numpy; import PySide6; print('OK')"
```

输出 `OK` 即安装成功。

---

## 2. 拍摄实验视频

拍摄质量是测量精度的基础。请遵循以下规范：

### 基本要求

| 要求 | 说明 |
|------|------|
| **背景** | 白色哑光背景，与深色小球形成高对比度 |
| **光照** | 光线均匀柔和，避免强反光或过曝 |
| **相机固定** | **必须使用三脚架**，严禁手持拍摄 |
| **帧率** | ≥ 60 fps，推荐 120+ fps（手机慢动作模式） |
| **标尺** | 标尺贴在量筒外壁，与小球下落路径在同一平面 |
| **释放** | 小球从量筒中心轴线释放，避免碰壁 |
| **完整过程** | 视频应包含从释放到落底的完整过程 |
| **分辨率** | 确保小球像素直径 ≥ 12 px（对应 "good" 等级） |

### 拍摄检查清单

- [ ] 背景无反光
- [ ] 标尺清晰可见
- [ ] 相机水平放置
- [ ] 对焦准确
- [ ] 量筒垂直

---

## 3. 配置参数

系统通过 `config.yaml` 文件配置所有参数。该文件位于项目根目录。

### 3.1 必填物理参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `scale_mm_per_px` | 比例尺 (mm/像素) | `0.2869` |
| `ball_radius_mm` | 小球半径 (mm) | `0.75` |
| `ball_density_kg_m3` | 小球密度 (kg/m³) — 钢球约 7850 | `7850` |
| `liquid_density_kg_m3` | 液体密度 (kg/m³) — 甘油约 950 | `950` |
| `cylinder_radius_mm` | 量筒内半径 (mm) | `10.0` |
| `liquid_height_mm` | 液柱高度 (mm) | `335.0` |
| `temperature_c` | 液体温度 (°C) | `40.0` |

### 3.2 获取比例尺（关键步骤）

比例尺 `scale_mm_per_px` 直接影响黏度计算精度：

1. 拍摄一张**含标尺**的量筒照片（标尺与下落路径在同一平面）
2. 测量照片中标尺上已知距离对应的像素数
3. 计算：`scale_mm_per_px = 实际距离(mm) / 像素距离(px)`

> **示例**：标尺上 10 mm 对应 34.85 像素 → `scale_mm_per_px = 10 ÷ 34.85 = 0.2869`

### 3.3 首次使用建议调整的参数

| 参数 | 默认值 | 何时调整 |
|------|--------|----------|
| `detector_profile` | `stable_demo` | 正式分析切换为 `strict_physics` |
| `roi` | `null`（全帧） | 背景复杂时缩小搜索范围 |
| `expected_radius_px_max` | `4.0` | 球在视频中偏大时增加 |
| `enable_prediction_fill` | `false` | 追踪中断时设为 `true` |

> 完整参数表见项目根目录的 [README.md](README.md)。

---

## 4. GUI 图形界面模式（推荐）

GUI 模式集成视频预览、参数编辑、运行控制、结果展示于一体，是推荐的使用方式。

### 4.1 启动

```bash
python gui_app.py
```

或直接在文件管理器中双击 `gui_app.py`。

### 4.2 分步操作

#### 第 1 步：加载视频

点击 **「选择视频」** 按钮，选择实验视频文件（支持 `.mp4` / `.avi` / `.mov` / `.mkv`）。

#### 第 2 步：填写物理参数

在左侧参数面板中填写：

- 比例尺 `scale_mm_per_px`
- 小球半径 `ball_radius_mm`
- 小球密度 `ball_density_kg_m3`
- 液体密度 `liquid_density_kg_m3`
- 量筒内半径 `cylinder_radius_mm`
- 液柱高度 `liquid_height_mm`
- 液体温度 `temperature_c`

> 参数面板支持折叠分组，可按"物理参数"/"检测参数"/"追踪参数"分类查看。

#### 第 3 步：框选 ROI（可选）

在视频预览区**拖动鼠标**框选感兴趣区域（Region of Interest）。

- 缩小 ROI 可减少干扰（刻度线、量筒边缘等）
- 不框选则使用全帧检测

#### 第 4 步：设定小球初始位置（★ 关键步骤）

在视频预览区**右键点击**小球的起始位置。

为什么这一步很重要：
- 系统不会立即开始追踪，而是进入**等待阶段**
- 仅当连续多帧在初始位置附近检测到**持续向下运动**时才确认小球到达
- 有效防止静止特征（刻度线、反光、污渍）被误认为小球
- 未设初始位置时，系统可能从第 1 帧就开始追踪静止物体

#### 第 5 步：点击「开始分析」

点击 **「开始分析」** 按钮，系统自动执行完整管线：

```
视频输入 → 逐帧检测球心 → 坐标换算
→ y-t 轨迹 → v-t 速度 → 自动判稳
→ 终端速度 vt → 黏度计算（理想+修正）
→ 输出 CSV/图表/报告
```

分析过程中 **y-t 轨迹图实时更新**（每秒刷新 5 次），可即时观察追踪质量。

#### 第 6 步：查看结果

结果面板包含多个 Tab 页：

| Tab | 内容 |
|-----|------|
| **轨迹图** | y-t 散点图，标注终端速度区 |
| **速度曲线** | v-t 曲线，标注终端速度均值线 |
| **黏度结果** | 理想黏度、壁面修正因子、雷诺数修正因子、最终黏度 |
| **调试信息** | 逐帧候选数、状态机状态、跳变值 |

结果同时自动保存到 `data/results/` 目录。

### 4.3 GUI 高级功能

| 功能 | 说明 |
|------|------|
| **管线切换** | 勾选 `use_new_pipeline` 在旧版/新版管线间切换 |
| **手动区间** | 手动指定分析起止帧号，覆盖自动区间选择 |
| **参考黏度** | 输入参考黏度值，自动计算相对误差 |
| **参数预设** | 一键切换 `stable_demo` 和 `strict_physics` 预设 |
| **实时预览** | 视频播放/暂停/逐帧浏览 |

---

## 5. 命令行模式

适合无需图形界面的场景，或批量处理前的快速测试。

### 5.1 基本用法

```bash
python main.py --config config.yaml
```

此命令会弹出文件选择对话框选择视频。

### 5.2 指定视频文件

```bash
# 方式一：位置参数
python main.py --config config.yaml 视频文件路径.mp4

# 方式二：--video 参数
python main.py --config config.yaml --video 视频文件路径.mp4

# 方式三：拖放视频文件到 main.py 上
```

### 5.3 使用新版管线

```bash
python main.py --config config.yaml --new-pipeline
```

### 5.4 控制台输出解读

命令行模式按 7 个步骤顺序执行，控制台实时显示进度：

```
[1/7] 读取视频           → 加载视频，获取帧率/分辨率
[2/7] 逐帧追踪           → 检测球心，生成轨迹
[3/7] 计算速度           → 求导计算瞬时速度
[4/7] 自动判定终端速度区   → 滑动窗口判稳
[5/7] 计算黏度           → Stokes + 修正
[6/7] 生成图表           → 保存 y-t/v-t 图
[7/7] 生成结果摘要       → 汇总报告
```

---

## 6. 批量处理模式

处理大量实验视频时使用。

### 6.1 使用 batch_run.py

编辑 `batch_run.py` 中的配置（视频目录、球径、温度、输出路径等）：

```python
# 示例配置
BASE_VIDEO = "实验视频文件夹路径"
BALL_DENSITY = 7850.0      # kg/m³
LIQUID_DENSITY = 950.0     # kg/m³
DIAM_TO_RADIUS = {1.5: 0.75, 2.0: 1.0, 2.5: 1.25}  # 球径→半径
```

然后运行：

```bash
python batch_run.py
```

### 6.2 自定义批处理脚本

```python
from src.utils import load_config, normalize_config_keys, ensure_output_dir
from src.pipeline import run_pipeline

config = load_config("config.yaml")
config = normalize_config_keys(config)

config["video_path"] = "视频路径.mp4"
config["ball_radius_mm"] = 0.75
config["temperature_c"] = 40.0
# ... 设置其他参数

ensure_output_dir("输出目录")
result = run_pipeline("视频路径.mp4", config, "输出目录")
print(result)
```

---

## 7. 输出文件解读

分析完成后，结果默认保存到 `data/results/` 目录。

### 7.1 输出文件清单

| 文件 | 格式 | 内容 |
|------|------|------|
| `trajectory.csv` | CSV | 逐帧轨迹：帧号、时间(s)、x/y 坐标(px)、半径、diff_score、有效性 |
| `velocity.csv` | CSV | 速度：原始速度、平滑速度 (m/s)、有效性 |
| `marked_video.mp4` | 视频 | 标注视频（绿圈=检测点，蓝圈=预测点） |
| `trajectory_plot.png` | 图片 | y-t 轨迹图，标注终端速度区 |
| `velocity_plot.png` | 图片 | v-t 速度曲线，标注终端速度均值 |
| `result_summary.txt` | 文本 | 完整结果摘要 |
| `frame_debug.csv` | CSV | 逐帧调试信息 |
| `terminal_candidates.csv` | CSV | 终端速度区候选窗口明细 |

### 7.2 结果摘要示例

```
=========================================
  落球法液体黏度自动测量系统 - 结果摘要
=========================================

视频: videos/experiment_01.mp4
温度: 40.0 °C

[追踪统计]
  总帧数: 480
  有效帧: 456 (95.0 %)
  最长连续段: 450 帧 (1.875 s)

[终端速度区]
  范围: 0.800 s — 1.600 s
  终端速度 vt = 0.025341 m/s
  拟合 R² = 0.9985
  变异系数 Cv = 0.0324

[黏度结果]
  理想黏度 η_basic   = 0.2398 Pa·s
  壁面修正因子       = 1.1523
  雷诺数修正因子     = 1.0187
  最终黏度 η_final   = 0.2041 Pa·s
  雷诺数 Re          = 0.1245

[质量评估]
  参考值: 0.231 Pa·s (40 °C)
  相对误差: -11.6 %
```

---

## 8. 常见问题与参数调优

### 8.1 检测不到小球

| 可能原因 | 解决方法 |
|----------|----------|
| 背景对比度太低 | 使用白色背景 + 深色小球，增强光照 |
| 搜索范围太大 | **设置 `roi`** 缩小检测区域 |
| 球大小参数不匹配 | 增大 `expected_radius_px_max` |
| 阈值不合适 | 尝试不同的 `threshold_mode`（OTSU / 自适应） |

### 8.2 追踪不稳定或中途丢失

| 可能原因 | 解决方法 |
|----------|----------|
| 帧间跳变过大 | 增大 `search_win_y_down`（默认 120 → 180） |
| 图像噪声 | 增大 `gaussian_blur_ksize`（默认 5 → 7） |
| 加速阶段门控过严 | 确保 `template_enabled: true` |
| 短暂遮挡/模糊 | 开启 `enable_prediction_fill: true` |
| 检测质量波动 | 切换到 `detector_profile: strict_physics` |

### 8.3 静止噪点被误追踪（刻度线、反光）

| 解决方法 | 说明 |
|----------|------|
| **✅ 右键设定初始位置** | **最有效**：让系统等待小球真正到达才开始追踪 |
| 增大 `min_track_diff_score` | 默认 3.0 → 尝试 5.0 ~ 8.0 |
| 开启 `enable_long_line_rejection` | 自动抑制刻度线等直线结构的误检 |
| 缩小 `roi` | 避开量筒上的密集刻度区域 |

### 8.4 找不到终端速度区（匀速段）

| 解决方法 | 说明 |
|----------|------|
| 放宽 `cv_threshold` | 默认 0.08 → 尝试 0.12 |
| 增大 `terminal_window_sec` | 默认 0.8 s → 尝试 1.0 s |
| 检查视频完整性 | 确保包含从释放到落底的完整过程 |
| 检查有效帧数 | 确保有效帧数 > 50 |

### 8.5 黏度结果偏差大

| 可能原因 | 解决方法 |
|----------|----------|
| 比例尺不准 | 重新标定 `scale_mm_per_px` |
| 壁面修正未开启 | 确保 `enable_wall_correction: true` |
| 雷诺数修正未开启 | 确保 `enable_reynolds_correction: true` |
| 小球密度错误 | 确认 `ball_density_kg_m3`（钢球 7850） |
| 终端速度不准 | 检查 y-t 图终端区是否真的水平 |

---

## 附录

### 管线架构

| 管线 | 切换方式 | 检测器 | 追踪器 | 特点 |
|------|----------|--------|--------|------|
| **旧版（默认）** | `use_new_pipeline: false` | BallDetector（3 方法，130 分） | tracking.py 状态机 | 成熟稳定 |
| **新版** | `use_new_pipeline: true` 或 `--new-pipeline` | CandidateDetector（3 方法，100 分） | BallTracker alpha-beta 滤波 | 架构清晰 |

### 检测器预设

| 预设 | 适用场景 |
|------|----------|
| `stable_demo` | 稳定演示模式，放宽质量门槛，优先输出结果 |
| `strict_physics` | 严格物理模式，高标准质量要求，正式分析用 |

### 小球尺寸质量分级

| 等级 | 像素直径 | 含义 |
|------|----------|------|
| good | ≥ 12 px | 所有检测方法均可靠 |
| acceptable | ≥ 10 px | 多数方法可用 |
| poor | ≥ 5 px | 仅背景差分较可靠 |
| bad | < 5 px | 检测不可靠，建议调整拍摄 |

### 最常用调参路径速查

```
检测不到小球     → 设 roi + 调 expected_radius_px_max
追踪静止噪点     → 设初始位置 + 增大 min_track_diff_score
追踪中途丢失     → 增大 search_win_y_down + 开启 prediction_fill
找不到匀速段     → 放宽 cv_threshold + 检查视频完整性
黏度偏差大       → 检查比例尺 + 确认壁面修正已开启
```

---

> **仓库地址**：https://github.com/huangdouding/falling-ball-viscometer-core
> **详细技术文档**：[README.md](README.md) — 含完整参数表、算法说明、构建打包指南
