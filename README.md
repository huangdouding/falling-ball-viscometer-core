# 落球法液体黏度自动测量系统

基于**机器视觉自动追踪与误差修正**的落球法液体黏度测量实验改进。通过 OpenCV 逐帧识别透明液体中下落小球的球心位置，拟合下落轨迹，自动判稳终端速度区并计算液体黏度（含管壁修正）。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# GUI 模式（推荐）
python gui_app.py

# 命令行模式
python main.py --config config.yaml
```

## 项目结构

```
├── main.py                  # 命令行入口
├── gui_app.py               # PySide6 GUI 入口
├── config.yaml              # 默认配置文件
├── requirements.txt
│
├── src/
│   ├── pipeline.py              # 新版分析管线（一键 run_pipeline）
│   ├── video_io.py              # 视频读写
│   ├── ball_detector.py         # 单帧小球检测（blob / 阈值轮廓 / 背景差分 + 长直线抑制）
│   ├── candidate_detector.py    # 新版检测器（3 方法 + 5 维度评分，满分 100）
│   ├── background_model.py      # 背景建模（中值/均值/MOG2/滑动窗口）
│   ├── ball_geometry.py         # 小球几何参数计算与尺寸质量分级
│   ├── tracking.py              # 逐帧追踪 → 轨迹 DataFrame（状态机 + 运动门控 + 预测填充）
│   ├── ball_tracker.py          # 新版追踪器（alpha-beta 滤波 + 物理约束）
│   ├── trajectory_filter.py     # 轨迹后处理滤波器
│   ├── interval_selector.py     # 追踪区间选择器
│   ├── velocity.py              # 速度计算与平滑
│   ├── terminal_region.py       # 终端速度区自动判定（三级回退）
│   ├── terminal_velocity.py     # 新版终端速度拟合器
│   ├── viscosity.py             # 黏度计算（理想 + 管壁修正）
│   ├── plotting.py              # 图表输出（y-t、v-t）
│   ├── report.py                # 结果摘要报告
│   ├── utils.py                 # 配置读取、参数标准化、检测器预设
│   ├── roi_manager.py           # ROI 管理
│   └── gui/                     # GUI 组件
│       ├── main_window.py
│       ├── parameter_panel.py
│       ├── video_widget.py
│       ├── result_tabs.py
│       ├── dialogs.py
│       └── worker.py
│
├── tests/
│   └── test_core.py
│
├── config/
│   └── settings.json        # GUI 持久化设置
│
├── videos/                  # 输入视频
├── data/results/            # 默认输出目录
└── output/                  # 备用输出目录
```

## 管线架构

项目包含两套分析管线，通过 `use_new_pipeline` 切换：

| 管线 | 检测器 | 追踪器 | 终端速度 | 说明 |
|------|--------|--------|----------|------|
| **legacy**（默认） | `BallDetector`（3 方法，10 维评分/130 分） | `tracking.py` 状态机 | `terminal_region.py` 三级回退 | 成熟稳定，久经测试 |
| **new** | `CandidateDetector`（3 方法，5 维评分/100 分） | `BallTracker`（alpha-beta 滤波） | `terminal_velocity.py` | 架构更清晰，无静默回退 |

### 新版管线评分维度

| 维度 | 权重 | 说明 |
|------|------|------|
| pred_score | 30 | 位置预测一致性 |
| motion_score | 25 | 运动方向一致性 |
| size_score | 20 | 目标尺寸匹配度 |
| contrast_score | 15 | 与背景对比度 |
| axis_score | 10 | 偏离下落轴线程度 |

## 检测器预设

`config.yaml` 中通过 `detector_profile` 切换预设：

| 预设 | 说明 |
|------|------|
| `stable_demo` | 稳定演示模式，放宽质量门槛，优先输出结果 |
| `strict_physics` | 严格物理模式，高标准质量要求，用于正式分析 |

## 检测方法

### 单帧检测（BallDetector）

三种检测方法自动回退（`detect_method: auto`）：

1. **blob** — SimpleBlobDetector，按面积/圆度/惯性比筛选
2. **threshold_contour** — 自适应/OTSU 阈值分割 + 轮廓筛选
3. **background_subtraction** — 背景差分，对运动小球敏感度最高

### 长直线抑制

自动识别并抑制标尺刻度、量筒刻线等线性结构上的伪检测点。通过连通域分析 + Sobel 边缘检测 + 形态学直线检测，对落在长直线结构上的候选点施加最高 -20 分惩罚。由 `enable_long_line_rejection` 控制。

### 背景建模

`background_model.py` 支持四种背景构建策略：

| 策略 | 说明 |
|------|------|
| median | 中值背景（默认，50 帧采样） |
| mean | 均值背景 |
| MOG2 | 高斯混合模型 |
| sliding | 滑动窗口增量更新 |

### 小球尺寸质量分级

系统根据像素直径自动评定检测难度等级：

| 等级 | 像素直径 | 含义 |
|------|----------|------|
| good | ≥ 12 px | 所有检测方法均可靠 |
| acceptable | ≥ 10 px | 多数方法可用 |
| poor | ≥ 5 px | 仅背景差分较可靠 |
| bad | < 5 px | 检测不可靠，建议调整拍摄 |

## 追踪算法

### 旧版追踪（tracking.py）

逐帧追踪采用四层判定架构：

1. **单帧检测** — BallDetector 三方法合并候选，综合评分排序
2. **运动一致性筛选** — 遍历所有候选过运动门控（dx/dy/距离），优先选 diff_score 最高的候选。含 rescue 机制处理边界情况
3. **异常检测** — 帧间跳变、中心通道、ROI 约束、静止噪声（diff < 3.0）、首帧保护
4. **状态机** — TRACKING → LOST → REACQUIRE → STOPPED，自适应搜索窗口和门控

辅助机制：

- **初始位置等待** — 用户在视频上点击小球初始位置后，系统不会立即开始追踪。等待阶段持续累积候选点，仅当连续多帧检测到**持续向下运动**（非静止）时才确认小球到达并开始追踪。有效防止静止特征（刻度线、反光）被误认为小球
- **ROI 边界过滤** — 所有检测候选强制经过 `user_roi` 边界校验，ROI 外的候选直接丢弃
- **模板匹配（NCC 相关）** — 首次检测到小球后自动提取 ROI 内图像模板，后续帧对候选窗口计算归一化互相关分数。模板匹配高分（≥ 0.65）时自动信任检测结果并跳过运动门控，解决加速阶段门控过严问题。模板持续进化（α=0.20）并回拽原始帧（α=0.05）
- **动态参数计算** — 根据 `ball_radius_mm` 和 `scale_mm_per_px` 自动推算 `min_area_px`、`max_area_px`、搜索窗口尺寸、`dist_max` 等参数，无需手动调参
- **预测填充** — 检测短暂丢失时，基于加速度预测位置填补空白（`pos + v·dt + 0.5·a·dt²`）。`enable_prediction_fill` 控制开关，最多连续 18 帧
- **动态搜索窗口** — 模板高置信度时缩小搜索窗口，丢帧后逐步扩大（1.5× → 2.5× → 4× → 6×）
- **断尾修剪** — 检测到轨迹突然跳变至异常位置后，自动截断尾部伪影（连续 2 帧异常即触发）
- **大球模式** — 球半径 ≥ 1.0 mm 或像素半径 ≥ 5 px 时自动启用，放宽面积和搜索窗口

### 新版追踪（ball_tracker.py）

- **alpha-beta 滤波** — 位置平滑 α=0.7，速度平滑 β=0.3，可配置
- **物理约束搜索窗口** — 非对称窗口（上 20px / 下 120px / 宽 60px）
- **状态机** — tracking → lost → reacquire → stopped，reacquire 阶段搜索窗口放大 2 倍

## 终端速度区判定

### 旧版（terminal_region.py）

三级回退策略：

1. **主方案** — 滑动窗口 Cv 判稳：在轨迹上滑动固定时长窗口，逐窗口线性拟合计算变异系数 Cv，找出 Cv < 阈值的连续窗口组
2. **回退方案** — 放宽线性度检查 + x 稳定性 + y 单调性约束
3. **兜底方案** — 选取数据量最多的连续段

### 新版（terminal_velocity.py）

仅使用 raw 类型点（非预测填充点），无可用于拟合的段时明确报错，不静默回退。

## 黏度计算

### 理想黏度（斯托克斯定律）

$$ \eta_{\text{basic}} = \frac{2 \cdot r^2 \cdot g \cdot (\rho_s - \rho_l)}{9 \cdot v_t} $$

其中 r 为小球半径，ρ_s 为小球密度，ρ_l 为液体密度，v_t 为终端速度。

### 壁面修正（Ladenburg 公式）

补偿有限容器壁面对小球阻力的影响。Stokes 定律假设无限大流体，当小球在有限半径量筒中下落时，壁面会增大阻力，使表观黏度偏高。

壁面修正因子仅考虑量筒径向壁面效应（液柱高度修正暂未纳入）：

$$ k_{\text{wall}} = 1 + 2.4 \frac{r}{R} $$

其中 $r$ 为小球半径，$R$ 为量筒内半径。

由 `enable_wall_correction` 控制开关。

### 雷诺数修正（Oseen 公式）

补偿有限雷诺数下 Stokes 阻力公式的系统偏差。Stokes 阻力 $F_d = 6\pi\eta r v$ 仅严格适用于 Re → 0。
Oseen 修正给出更精确的阻力表达式：

$$ F_d = 6\pi\eta r v \left(1 + \frac{3}{16}Re \right) $$

其中雷诺数 $Re = \frac{2 r v \rho_l}{\eta}$。由终端速度平衡导出隐式方程并通过迭代求解：

$$ \eta_{\text{re}} = \frac{\eta_{\text{basic}}}{1 + \frac{3}{16}Re}, \quad Re = \frac{2 r v_t \rho_l}{\eta_{\text{re}}} $$

由 `enable_reynolds_correction` 控制开关。Re > 1 时输出警告。

### 综合修正

当同时启用壁面修正和雷诺数修正时，最终黏度为：

$$ \eta_{\text{final}} = \frac{\eta_{\text{basic}}}{k_{\text{wall}} \cdot k_{\text{Re}}} $$

### 修正符号表

| 符号 | 含义 |
|------|------|
| r | 小球半径 |
| R | 量筒内半径 |
| h | 液柱高度 |
| ρ_l | 液体密度 |
| v_t | 终端速度 |
| η_basic | Stokes 理想黏度 |
| η_re | Oseen 修正后黏度 |
| η_final | 综合修正后黏度 |
| k_wall | 壁面修正因子 |
| k_Re | 雷诺数修正因子 |

## 配置参数

### 必填参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `scale_mm_per_px` | 比例尺 (mm/px) | `0.2869` |
| `ball_radius_mm` | 小球半径 (mm) | `0.75` |
| `ball_density_kg_m3` | 小球密度 (kg/m³) | `7850` |
| `liquid_density_kg_m3` | 液体密度 (kg/m³) | `950` |
| `cylinder_radius_mm` | 量筒内半径 (mm) | `20.0` |
| `liquid_height_mm` | 液柱高度 (mm) | `335` |

### 实验/物理参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `g_m_s2` | 重力加速度 (m/s²) | `9.80` |
| `temperature_c` | 液体温度 (°C) | — |
| `reference_viscosity_pa_s` | 参考黏度值 (Pa·s)，用于对比 | — |
| `fps` | 视频帧率 | `240` |
| `manual_fps` | 手动指定帧率（覆盖视频元数据） | `240` |

### 检测参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `detector_profile` | 检测器预设 | `stable_demo` |
| `detect_method` | 检测方法：auto/blob/threshold_contour/bg_sub | `auto` |
| `color_mode` / `threshold_mode` | 颜色模式 | `dark_ball_on_bright_bg` |
| `roi` | 感兴趣区域 [x, y, w, h] | `null`（全帧） |
| `expected_radius_px_min` | 预期像素半径下限（auto_size_params 为 true 时自动计算） | `1.0` |
| `expected_radius_px_max` | 预期像素半径上限 | `4.0` |
| `min_area_px` | 最小 blob 面积 (px²) | 动态计算 |
| `max_area_px` | 最大 blob 面积 (px²) | 动态计算 |
| `min_circularity` | 最小圆度 | `0.35` |
| `gaussian_blur_ksize` | 高斯模糊核大小 | `5` |
| `morph_kernel_size` | 形态学操作核大小 | `3` |
| `contrast_threshold` | 最低对比度分 | `5.0` |
| `auto_size_params` | 根据球半径/比例尺自动推算尺寸参数 | `true` |
| `large_ball_mode` | 大球模式（自动启用：ball_radius_mm ≥ 1.0 或 radius_px ≥ 5） | auto |
| `image_mode` | 单图模式（禁用背景差分） | `false` |

### 中心通道/轴线约束

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `center_band_enabled` | 启用中心运动带约束 | `true` |
| `center_band_left_ratio` | 中心带左边界（ROI 宽度比例） | `0.30` |
| `center_band_right_ratio` | 中心带右边界 | `0.70` |
| `fall_axis_x` | 下落轴线 X 坐标（null 则自动估计） | `null` |
| `allowed_axis_deviation_px` | 允许偏离轴线最大像素 | `50` |
| `auto_shrink_roi` | 自动收缩检测 ROI 至轴线附近 | `false` |
| `detect_roi_width` | 自动收缩 ROI 宽度 | `80` |

### 追踪参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `max_jump_px` | 帧间最大跳变像素 | `80` |
| `dx_max` | 帧间最大水平位移 (px) | 动态计算 |
| `dy_back_tol` | 允许的 y 方向回跳 (px) | 动态计算 |
| `dist_max` | 帧间最大欧氏距离 (px) | 动态计算 |
| `search_win_w` | 搜索窗口半宽 (px) | `60` |
| `search_win_y_up` | 搜索窗口向上扩展 (px) | `20` |
| `search_win_y_down` | 搜索窗口向下扩展 (px) | `120` |
| `search_window_radius` | 搜索窗口半径（备用规格） | `60` |
| `lost_threshold` | 连续丢失帧数阈值 | `10` |
| `max_lost_frames` | 强制全 ROI 搜索前最大丢失帧数 | `30` |
| `stop_y_ratio` | 停止追踪的 y/ROI 高度比 | `0.92` |
| `min_track_diff_score` | 接受追踪点的最低 diff_score（抑制静止噪点） | `3.0` |
| `x_gate_ratio` | x 门控占 ROI 宽度比例 | `0.25` |

### 预测填充参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `enable_prediction_fill` | 启用速度预测填充检测缺口 | `false` |
| `predict_max_frames` | 最大连续预测填充帧数 | `18` |
| `trajectory_prediction_gate_px` | 预测误差门控 (px) | 动态计算 |
| `min_predict_vy_px_frame` | 最小预测下落速度 (px/frame) | `0.3` |
| `max_predict_vy_px_frame` | 最大预测下落速度 (px/frame) | search_win_y_down/3 |
| `max_predict_vx_px_frame` | 最大预测水平速度 (px/frame) | dx_max |
| `breakaway_consecutive_outliers` | 断尾判定连续异常帧数 | `2` |
| `breakaway_min_raw_before` | 断尾前最少 raw 点数 | `8` |

### 启动/恢复参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `start_confirm_frames` | 启动确认帧数 | `5` |
| `init_ball_distance_px` | 初始位置检测范围 (px) | 自动计算（5×ball_radius_px，最小15） |
| `init_ball_confirm_frames` | 初始位置确认所需连续向下运动帧数 | `5` |
| `startup_gate_scale` | 启动阶段 x 门控放宽倍数 | `2.0` |
| `tracking_tolerance_scale` | 稳定追踪连续性门控放宽系数 | auto |
| `tracking_rescue_scale` | Rescue 门控放宽系数 | `1.8` |
| `tracking_rescue_pred_scale` | Rescue 预测误差放宽系数 | `1.6` |
| `tracking_prediction_gate_scale` | 预测误差门控缩放 | `1.10` |

### 模板匹配参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `template_enabled` | 启用 NCC 模板匹配 | `true` |
| `template_match_threshold` | 模板匹配最低相关系数 (0~1) | `0.50` |
| `template_trust_threshold` | 信任跳过运动门控的模板分阈值 | `0.65` |
| `template_evolve_alpha` | 模板进化率 (0~1) | `0.20` |
| `template_tether_alpha` | 模板回拽率 (0~1)，拉回原始帧 | `0.05` |
| `template_size_factor` | 模板尺寸 = 球半径 × 系数 | `2.5` |

### 新版追踪参数（ball_tracker.py）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `tracker_alpha` | alpha-beta 滤波器位置平滑系数 | `0.7` |
| `tracker_beta` | alpha-beta 滤波器速度平滑系数 | `0.3` |
| `reacquire_search_scale` | reacquire 状态搜索窗口放大倍数 | `2.0` |

### 终端速度判定参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `cv_threshold` | 终端速度判稳 Cv 变异系数阈值 | `0.08` |
| `r2_threshold` | 线性拟合 R² 阈值 | `0.99` |
| `terminal_window_sec` | 滑动窗口时长 (s) | `0.8` |
| `terminal_min_group_windows` | 有效窗口组最少窗口数 | `5` |
| `terminal_min_duration` | 终端区最短时长 (s) | `0.5` |
| `terminal_ignore_start_sec` | 轨迹起始忽略时长 (s) | `0.3` |
| `terminal_ignore_end_sec` | 轨迹末尾忽略时长 (s) | `0.5` |
| `terminal_min_real_detection_rate` | 终端区内最低真实检测率 | `0.80` |
| `ignore_start_sec` | 起始忽略时长 (s) | `0.3` |
| `ignore_end_sec` | 末尾忽略时长 (s) | `3.5` |
| `manual_start_frame` | 手动指定起始帧（0=自动） | `0` |
| `manual_end_frame` | 手动指定结束帧（0=自动） | `0` |

### 质量评估参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `quality_r2_threshold` | 优质段 R² 阈值 | `0.995` |
| `quality_cv_threshold` | 优质段 Cv 阈值 | `0.05` |
| `quality_min_frames` | 优质段最少帧数 | `80` |
| `min_track_valid_points` | 候选段最少有效 raw 点数 | `30` |
| `min_track_raw_rate` | 候选段最低 raw 检测率 | `0.30` |
| `min_track_displacement_px` | 候选段最低 y 位移 (px) | `30.0` |

### 速度平滑参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `smooth_method` | 平滑方法 | `moving_average` |
| `velocity_window_sec` | 速度平滑窗口时长 (s) | `0.2` |

### 管线/输出控制

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `use_new_pipeline` | 启用新版管线 | `false` |
| `enable_wall_correction` | 壁面修正（Ladenburg） | `true` |
| `enable_reynolds_correction` | Oseen 雷诺数修正 | `true` |
| `enable_long_line_rejection` | 长直线结构抑制 | `true` |
| `save_marked_video` | 输出标注视频 | `true` |
| `save_plots` | 输出图表 | `true` |

## GUI 功能

GUI 基于 PySide6，主要功能区：

| 面板 | 功能 |
|------|------|
| 参数面板 | 编辑所有实验参数（物理量 / 检测 / 追踪 / 终端判定 / 模板匹配） |
| 视频预览 | 实时预览视频、ROI 框选、**点击设定小球初始位置** |
| 结果面板 | 多 Tab 展示轨迹图、速度曲线、黏度结果、调试信息 |
| **实时轨迹** | 分析过程中 y-t 图增量更新，每秒刷新 5 次，即时观察追踪质量 |
| 手动区间 | 支持手动指定分析起止帧，覆盖自动区间选择 |
| 管线切换 | `use_new_pipeline` 一键切换旧版/新版管线 |
| 质量评估 | 展示 min_frames / R² / Cv 质量指标，与阈值对比 |
| 参考黏度 | 输入参考黏度值进行实验误差分析 |

### 初始位置设定

在视频预览区**右键点击**小球初始位置，系统会记录该坐标。分析时：
1. 等待阶段：仅当连续多帧在初始位置附近检测到**持续向下运动**时才确认小球到达
2. 确认后开始追踪，之前的静止误检全部忽略
3. 初始位置的范围由 `init_ball_distance_px` 控制（默认 5 倍球半径）

## 输出

默认输出到 `data/results/`：

| 文件 | 说明 |
|------|------|
| `trajectory.csv` | 轨迹数据（帧号、时间、坐标、半径、diff_score、有效性） |
| `velocity.csv` | 速度数据（原始/平滑速度、有效性） |
| `marked_video.mp4` | 标注视频（绿圈 = raw 点，蓝圈 = 预测点） |
| `trajectory_plot.png` | y-t 轨迹图 |
| `velocity_plot.png` | v-t 速度曲线图 |
| `result_summary.txt` | 结果摘要（追踪统计、黏度结果、质量评估） |
| `frame_debug.csv` | 逐帧调试信息（候选数、跳变、状态机） |
| `frame_debug_detailed.csv` | 逐帧详细调试（完整检测→过滤链路数据） |
| `candidate_debug.csv` | 逐帧候选点评分明细 |
| `terminal_candidates.csv` | 终端区所有候选窗口及拟合结果 |

## 拍摄建议

1. 白色哑光背景，与深色小球形成对比
2. 光线均匀，避免强反光或过曝
3. 三脚架固定相机，避免抖动
4. 标尺与小球下落路径在同一平面
5. 帧率 ≥ 60 fps，推荐 120+ fps 慢动作
6. 小球从量筒中心轴线释放，避免碰壁
7. 视频包含从释放到落底的完整过程
8. 小球像素直径 ≥ 12 px 为最佳（`good` 等级）

## 参数调整

| 问题 | 建议 |
|------|------|
| 检测不到小球 | 设置 `roi` 缩小搜索区；调整 `expected_radius_px_max` 覆盖实际大小 |
| 识别不稳定 | 减小 `max_jump_px`；增大 `gaussian_blur_ksize` |
| **从视频开始就追踪到静止噪点** | **在视频上右键点击小球初始位置**；系统会等待小球真正到达才追踪 |
| 静止噪点被误追踪 | 增大 `min_track_diff_score`（默认 3.0）；确保已设置初始位置；开启 `enable_long_line_rejection` |
| 初始位置范围过大/过小 | 调整 `init_ball_distance_px`（默认 5×球半径，约 26 px） |
| 找不到终端区 | 放宽 `cv_threshold` 到 0.12；检查小球是否已进入匀速段 |
| 轨迹跟随伪影 | 切换到 `strict_physics` 预设；减小 `allowed_axis_deviation_px` |
| 刻度线被误检为小球 | 确保 `enable_long_line_rejection: true`；缩小 `roi` 避开密集刻度区 |
| 丢失帧太多导致断连 | 开启 `enable_prediction_fill: true`；增大 `predict_max_frames` |
| 加速阶段运动门控过严 | 确保 `template_enabled: true`；模板匹配高分自动跳过门控 |
| 小球像素太小（< 5 px） | 提高拍摄分辨率或拉近相机；确保小球像素直径 ≥ 12 px |

## 依赖

- Python >= 3.8
- opencv-python >= 4.8
- numpy >= 1.24
- pandas >= 2.0
- matplotlib >= 3.7
- scipy >= 1.10
- PyYAML >= 6.0
- PySide6 >= 6.11

## 构建与打包

项目使用 **PyInstaller** 构建跨平台可执行文件，通过 GitHub Actions CI 自动完成。

### 从源码运行

```bash
pip install -r requirements.txt
python gui_app.py     # GUI 模式
python main.py --config config.yaml   # 命令行模式
```

### 本地构建可执行文件

```bash
pip install pyinstaller

# Windows
pyinstaller --name "Viscometer" --onedir --add-data "config.yaml;." --collect-all PySide6 --collect-all cv2 gui_app.py

# macOS
pyinstaller --name "Viscometer" --onedir --windowed --add-data "config.yaml:." --collect-all PySide6 --collect-all cv2 gui_app.py
```

### CI 自动构建（推荐）

推送 tag 自动触发 GitHub Actions 构建：

```bash
git tag v1.0.0
git push origin v1.0.0
```

构建产物将自动发布到 GitHub Releases 页面。也可在 Actions 页面手动触发 `workflow_dispatch`。

### 下载预构建包

访问 [Releases](https://github.com/huangdouding/falling-ball-viscometer-/releases) 页面下载：

| 平台 | 文件 | 说明 |
|------|------|------|
| Windows | `Viscometer-Windows.zip` | 解压后运行 `Viscometer.exe` |
| macOS | `Viscometer-macOS.dmg` | 挂载后将 App 拖入 Applications 文件夹 |

> **macOS 注意**：首次运行需右键 → 打开以绕过 Gatekeeper 签名检查。
> **Windows 注意**：如遇 Windows Defender 提示，选择"仍要运行"即可。
