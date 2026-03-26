# 实现方案：基于 L515 + SAM3 的多目标深度轨迹采集系统

> 目标：采集 L515 视频 → 对多个 prompt 目标分割 → 提取各目标平均深度 → 可视化轨迹  
> 日期：2026-03-26

---

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         完整管线                                  │
│                                                                   │
│  [Step 1] L515 采集                                              │
│     RealSense SDK → RGB帧 + 深度帧（对齐到RGB坐标系）→ 保存       │
│           ↓                                                       │
│  [Step 2] SAM3 分割                                              │
│     video_sam3.py（改造） → 多个 prompt → 每帧每目标的 mask       │
│           ↓                                                       │
│  [Step 3] 深度抠像 + 三维坐标计算                                │
│     mask ∩ depth_map → 中位数像素反投影 → (X, Y, Z) mm          │
│           ↓                                                       │
│  [Step 4] 轨迹可视化                                              │
│     matplotlib → 3D 轨迹图 + X/Y/Z 分量图 → 输出图像             │
└─────────────────────────────────────────────────────────────────┘
```

**新增/修改的文件**：

```
collect_scripts/
├── video_sam3.py          ← ✅ 已完成（支持多 prompt、返回 mask 数据）
├── l515_capture.py        ← ✅ 已完成（L515 采集脚本）
├── depth_extract.py       ← ✅ 已完成（深度抠像 + 三维坐标）
├── visualize_trajectory.py ← ✅ 已完成（轨迹可视化）
└── pipeline.py            ← ✅ 已完成（Step 2~4 一键串联）
```

---

## 最终输出目录全景

整个管线运行完毕后，`collect_data/session_YYYYMMDD_HHMMSS/` 的完整结构如下。每项后面标注了它由哪个 Step 写入。

```
collect_data/session_YYYYMMDD_HHMMSS/
│
│  ── Step 1: l515_capture.py ──────────────────────────────────
├── color/
│   ├── frame_0000.png          # RGB 图（BGR，1280×720）
│   ├── frame_0001.png
│   └── ...
├── depth/
│   ├── frame_0000.npy          # 对齐后深度图（uint16，mm）
│   ├── frame_0001.npy
│   └── ...
├── intrinsics.json             # 相机内参 {fx,fy,cx,cy,width,height}
├── metadata.json               # 采集参数 {fps, total_frames, timestamp,...}
│
│  ── Step 2: video_sam3.py ────────────────────────────────────
├── masks/
│   ├── door_handle/
│   │   ├── frame_0000.npy      # 合并后 mask（bool, H×W）
│   │   └── ...
│   ├── person/
│   │   └── ...
│   └── ...                     # 每个 prompt 一个子目录
│
│  ── Step 3: depth_extract.py ─────────────────────────────────
├── trajectories_3d.json        # 三维轨迹 {prompt: [[X,Y,Z],...]}
│
│  ── vis/ 下均为 debug 可视化，各 Step 按需写入 ────────────────
└── vis/
    ├── mask/                   # ← Step 2 写入
    │   ├── door_handle/
    │   │   ├── frame_0000.png  # RGB+mask轮廓 | 深度伪彩色 | mask区域高亮
    │   │   └── ...
    │   └── person/
    │       └── ...
    ├── depth/                  # ← Step 3 写入
    │   ├── door_handle/
    │   │   ├── frame_0000.png  # 深度伪彩色 + 中位数点标注 + 三轴坐标文字
    │   │   └── ...
    │   └── person/
    │       └── ...
    └── trajectory/             # ← Step 4 写入
        ├── trajectory_3d.png           # 3D 空间轨迹图（Axes3D）
        └── trajectory_components.png  # X(t) / Y(t) / Z(t) 三子图
```

**各文件格式一览：**

| 文件 | 格式 | 内容 | 写入阶段 |
|---|---|---|---|
| `color/frame_XXXX.png` | PNG（8bit BGR） | RGB 帧 | Step 1 |
| `depth/frame_XXXX.npy` | NumPy uint16 | 对齐深度图（mm） | Step 1 |
| `intrinsics.json` | JSON | fx, fy, cx, cy, w, h | Step 1 |
| `metadata.json` | JSON | fps, 帧数, 时间戳 | Step 1 |
| `masks/{prompt}/frame_XXXX.npy` | NumPy bool (H,W) | 合并后 mask | Step 2 |
| `vis/mask/{prompt}/frame_XXXX.png` | PNG（8bit RGB） | mask+RGB+深度三合一 | Step 2 |
| `trajectories_3d.json` | JSON | 每帧 [X,Y,Z]（mm） | Step 3 |
| `vis/depth/{prompt}/frame_XXXX.png` | PNG（8bit RGB） | 中位数点标注深度图 | Step 3 |
| `vis/trajectory/trajectory_3d.png` | PNG（高DPI） | 3D 轨迹图 | Step 4 |
| `vis/trajectory/trajectory_components.png` | PNG（高DPI） | X/Y/Z 分量折线图 | Step 4 |

---

## 二、各步骤实现思路

---

### Step 1：✅ L515 数据采集（`l515_capture.py`）

#### 1.1 核心目标

- 同步采集 RGB + 深度帧
- 深度图必须**对齐到 RGB 坐标系**，保证每个像素位置的 RGB 和深度对应同一个空间点
- 保存到结构化目录，便于后续按帧索引读取

#### 1.2 对齐方案（重要）

L515 的 RGB 传感器和深度传感器有不同的内参和位置，直接使用两者的原始图像**像素不对应**。

**必须使用 `rs.align` 将深度图对齐到 RGB 坐标系**：

```
rs.align(rs.stream.color)
→ aligned_frames = align.process(frames)
→ aligned_depth_frame = aligned_frames.get_depth_frame()  # 已对齐的深度，分辨率与 RGB 相同
→ color_frame = aligned_frames.get_color_frame()
```

对齐后：
- RGB 图和深度图的**分辨率相同**（如 1280×720）
- 像素坐标 `(u, v)` 在两张图中对应**同一个空间点**
- 可以直接用 SAM3 输出的 mask（基于 RGB 坐标）索引深度图

#### 1.3 保存目录结构

```
collect_data/session_YYYYMMDD_HHMMSS/
├── color/
│   ├── frame_0000.png     # RGB 图（BGR，用于 SAM3 输入）
│   ├── frame_0001.png
│   └── ...
├── depth/
│   ├── frame_0000.npy     # 深度图（uint16，单位 mm）
│   ├── frame_0001.npy
│   └── ...
├── intrinsics.json        # RGB 相机内参（fx, fy, cx, cy, width, height）
└── metadata.json          # 采集参数（fps, 帧数, 时间戳等）
```

**深度图为什么保存 `.npy` 而非 `.png`？**

| 格式 | 优点 | 缺点 |
|---|---|---|
| `.npy` | 无损，保留 uint16 精度（0~65535 mm） | 不能直接用图像查看器打开 |
| `16bit PNG` | 可用图像工具查看 | 部分工具读取时会错误地归一化 |
| `8bit PNG` | 文件小、可视化方便 | **精度严重损失**（深度范围压缩到 0~255） |

**结论**：深度数据用 `.npy`；如需可视化则另外生成 `depth_vis/frame_XXXX.png`（伪彩色）。

#### 1.4 分辨率与帧率选择

L515 支持的典型配置：

| 配置 | RGB | Depth | FPS | 适用场景 |
|---|---|---|---|---|
| 高精度 | 1920×1080 | 1024×768 | 15fps | 静态场景、精细采集 |
| **均衡（推荐）** | **1280×720** | **1024×768** | **30fps** | 一般场景、实时采集 |
| 低延迟 | 640×480 | 640×480 | 60fps | 快速运动场景 |

注意：L515 的深度分辨率最高为 1024×768，**不是 1280×720**，`rs.align` 会自动处理分辨率差异。

---

### Step 2：✅ SAM3 多 Prompt 分割（改造 `video_sam3.py`）

#### 2.1 核心改动

当前 `video_sam3.py` 只支持**一个文本 prompt**，需要改为支持**多个 prompt 列表**，且每个 prompt 对应独立的一组 obj_id 和 mask 序列。

#### 2.2 多 Prompt 的关键问题：obj_id 混用

**问题**：SAM3 高层 API 在单个 session 中，所有 prompt 的检测结果会共享同一个 obj_id 空间，无法直接区分哪个 obj_id 属于哪个 prompt。

**方案对比**：

| 方案 | 思路 | 优点 | 缺点 |
|---|---|---|---|
| 方案 A：多 Session | 每个 prompt 开一个独立 session，分别 propagate | 结果完全隔离，obj_id 不混淆 | GPU 显存翻倍，推理速度慢 |
| **✅ 方案 B：单 Session 顺序添加（已选定）** | 同一 session 中先 reset，再依次加入每个 prompt | 节省显存，结果完全隔离 | 需多次 propagate，耗时 × N |
| 方案 C：单 Session 同时添加 | 同一 session 中同时加入所有 prompt | 最快 | obj_id 混用，需要依赖检测顺序推断归属，**不可靠** |

**决策**：采用**方案 B（单 Session 顺序推理）**。在显存受限（单卡 8GB）的情况下，每个 prompt 单独 reset 并 propagate，保证各 prompt 的 mask 结果完全独立、不混淆。

#### 2.3 输出数据结构设计

改造后的核心函数签名：

```python
def run_sam3_multi_prompt(
    video_path: str,          # RGB 视频路径（MP4 或帧目录）
    prompts: list[str],       # 多个文本 prompt，如 ["door handle", "person"]
    frame_index: int = 0,     # 在哪一帧添加 prompt
) -> dict[str, dict[int, np.ndarray]]:
    # 返回: {prompt_text: {frame_idx: merged_binary_mask (H, W, bool)}}
```

每个 prompt 对应的值是一个 `{frame_idx: mask}` 字典，其中 mask 是**该帧内该 prompt 所有检测到的实例合并后的单张 mask**（用 `np.any(masks, axis=0)`）。

#### 2.4 需要修改的文件

- **修改**：`collect_scripts/video_sam3.py`
  - 将 `text_prompt = 'door handle'` 改为接受 `prompts: list[str]` 参数
  - 将输出从"保存 PNG"改为"返回 mask 字典"，将可视化拆分为独立函数
  - 暴露 `run_sam3_multi_prompt()` 函数供其他模块调用

#### 2.5 Mask 的磁盘存储格式

**原始数据结构（内存中）：**

SAM3 每一帧的 propagate 输出中，`outputs["out_binary_masks"]` 是一个 `numpy.ndarray`，形状为 `(N, H, W)`，dtype 为 `bool`：
- `N`：本帧检测到的实例数量（同一 prompt 可能检测到多个实例）
- `H, W`：与 RGB 帧的分辨率一致（L515 默认 1280×720）
- 值域：`True` 表示属于该实例的前景像素，`False` 为背景

**合并逻辑（实例聚合）：**

对于同一 prompt 的 N 个实例，将其合并为单张 mask：

```python
# 合并该 prompt 所有实例的 mask，形状 (H, W)，dtype=bool
merged_mask = np.any(outputs["out_binary_masks"], axis=0)
```

合并后每帧每个 prompt 只有一张 `(H, W)` bool 数组。

**磁盘存储方式（推荐 `.npy`）：**

```
collect_data/session_YYYYMMDD_HHMMSS/
├── masks/
│   ├── door_handle/           # prompt 名称（空格替换为下划线）
│   │   ├── frame_0000.npy     # shape=(H, W), dtype=bool
│   │   ├── frame_0001.npy
│   │   └── ...
│   ├── person/
│   │   ├── frame_0000.npy
│   │   └── ...
│   └── ...
└── vis/
    └── mask/                  # SAM3 mask 相关可视化（debug 用）
        ├── door_handle/       # 每个 prompt 一个子目录
        │   ├── frame_0000.png # mask 轮廓叠加 RGB + 深度伪彩色的合并图
        │   ├── frame_0001.png
        │   └── ...
        ├── person/
        │   └── ...
        └── ...
```

> **vis 文件夹说明**：`vis/` 下按功能划分子目录，便于 debug。`vis/mask/{prompt}/` 存放每帧的三合一可视化图（左：RGB + mask 轮廓叠加；中：深度伪彩色；右：mask 区域高亮深度），仅在 debug 模式下生成，不影响主流程性能。

- **格式**：`numpy` 二进制格式（`.npy`），用 `np.save("frame_0000.npy", merged_mask)` 保存，`np.load("frame_0000.npy")` 加载
- **文件大小**：1280×720 bool 数组 = 921,600 字节 ≈ **0.88 MB/帧**（未压缩）；若用 `np.savez_compressed` 打包整个视频，压缩率约 10:1（因为 bool 数组稀疏）
- **与 PNG 对比**：PNG 单通道灰度图也可存储 mask（0=背景，255=前景），但加载后需手动 `> 128` 转回 bool；`.npy` 直接保持 bool 语义，无需转换，读写速度更快，推荐用于后处理流程

**加载示例：**

```python
import numpy as np
mask = np.load("collect_data/session_xxx/masks/door_handle/frame_0042.npy")
# mask.shape == (720, 1280), mask.dtype == bool
```

**说明**：Step 3（深度抠像）直接读取这些 `.npy` 文件，与深度图逐帧对齐使用。

---

### Step 3：✅ 深度抠像 + 平均深度计算（`depth_extract.py`）

#### 3.1 核心逻辑

对于每一帧、每一个 prompt：

```
输入:
  mask[H, W]         bool，来自 SAM3
  depth[H, W]        uint16，单位 mm，已对齐到 RGB 坐标系
  intrinsics         相机内参（fx, fy, cx, cy），来自 intrinsics.json

处理:
  1. 提取有效深度像素及其坐标:
     valid_mask = mask & (depth > 0) & (depth < MAX_DEPTH_MM)
     valid_depths = depth[valid_mask]          # 有效深度值列表
     ys, xs = np.where(valid_mask)             # 对应的像素坐标 (v, u)

  2. 计算代表深度（中位数）及对应像素坐标:
     if len(valid_depths) == 0:
         → 该帧填 nan，跳过
     else:
         Z = np.median(valid_depths)           # 单位 mm
         # 找到深度值最接近中位数的像素点
         idx = np.argmin(np.abs(valid_depths - Z))
         u, v = xs[idx], ys[idx]               # 中位数点的像素坐标

  3. 反投影到三维空间（针孔相机模型）:
     X = (u - cx) * Z / fx                    # 单位 mm
     Y = (v - cy) * Z / fy                    # 单位 mm
     # Z 已知

输出:
  (X, Y, Z): tuple[float, float, float]（单位 mm）或 (nan, nan, nan)
  (u, v):    中位数点像素坐标，用于 vis 标注
```

> **坐标系说明**：反投影后的 `(X, Y, Z)` 使用**相机坐标系**（以相机光心为原点，Z 轴朝前，X 轴朝右，Y 轴朝下）。不同帧的坐标系相同（相机固定），因此可以直接比较不同帧之间的坐标变化，画出三维轨迹。

#### 3.2 深度有效性过滤的重要性

**默认选择中位数，但也提供平均数的选项**

L515 的深度图存在以下无效点：
- **值为 0**：传感器无法测量（过近、反光、镜面材质）
- **值异常大**：超出量程或测量误差

如果不过滤直接取均值，会严重偏离真实深度。

#### 3.3 可选增强：使用中位数代替均值

```
mean_depth = np.median(valid_depths)   # 对异常值更鲁棒
```

**均值 vs 中位数对比**：

| 统计量 | 优点 | 缺点 | 推荐场景 |
|---|---|---|---|
| 均值 | 计算简单，反映整体 | 受少量离群点影响大 | 目标表面均匀、无遮挡 |
| 中位数 | 对异常值鲁棒 | 计算略慢 | 目标边缘有杂乱深度噪声 |

**推荐**：默认用中位数，提供参数切换。

#### 3.4 输出数据结构与磁盘存储

```python
# 最终输出：每个 prompt，每帧一个三维坐标点（相机坐标系，单位 mm）
# 无效帧（目标不可见）填 (nan, nan, nan)
trajectories_3d: dict[str, list[tuple[float, float, float]]]
```

**保存格式与路径：**

```
collect_data/session_YYYYMMDD_HHMMSS/
├── trajectories_3d.json       # 三维轨迹数据，格式见下
└── vis/
    └── depth/                 # 深度抠像相关可视化（debug 用）
        ├── door_handle/       # 每个 prompt 一个子目录
        │   ├── frame_0000.png # 标注中位数点的深度可视化图（见下方说明）
        │   ├── frame_0001.png
        │   └── ...
        └── person/
            └── ...
```

`trajectories_3d.json` 的文件结构：

```json
{
  "metadata": {
    "session": "session_20260326_153000",
    "fps": 30,
    "total_frames": 161,
    "coord_unit": "mm",
    "coord_system": "camera",
    "stat_method": "median",
    "intrinsics": {"fx": 910.0, "fy": 910.0, "cx": 640.0, "cy": 360.0}
  },
  "trajectories": {
    "door handle": [
      [120.3, -45.2, 1250.5],
      [119.8, -45.0, 1248.0],
      null,
      [121.1, -44.9, 1251.3]
    ],
    "person": [
      [-230.5, 80.1, 2100.0],
      [-228.9, 80.3, 2098.5],
      [-231.2, 79.8, 2102.1],
      null
    ]
  }
}
```

> - 每帧存为 `[X, Y, Z]`（单位 mm，相机坐标系），无效帧存为 `null`
> - `metadata.intrinsics` 中保存本次采集使用的内参，便于离线复核反投影结果
> - `null` 对应 Python 的 `(np.nan, np.nan, np.nan)`，JSON 序列化时整体替换为 `null`

**中位数点 vis 说明：**

每帧的 vis 图标注代表点 `(u, v)`（即反投影所用的中位数像素点）：

- 背景：深度伪彩色图（仅显示 mask 区域，其余置灰）
- 绿色圆点：中位数点 `(u, v)`
- 白色文字标注：`Z=1250mm  X=120mm  Y=-45mm`（三轴坐标）

保存路径：`vis/depth/{prompt_slug}/frame_{idx:04d}.png`
---

### Step 4：✅ 轨迹可视化（`visualize_trajectory.py`）

#### 4.1 可视化内容

输出**两张图**，分别从不同维度展示轨迹：

**图一：三维空间轨迹图（`trajectory_3d.png`）**
- 使用 `matplotlib` 的 `Axes3D` 绘制三维散点/折线图
- X/Y/Z 轴对应相机坐标系的三个轴（单位 mm）
- 每个 prompt 一条轨迹，不同颜色
- `null` 帧处折线断开
- 图例标注每个 prompt 的文字

**图二：各轴分量随时间变化图（`trajectory_components.png`）**
- 3 个子图（subplot），分别显示 X(t)、Y(t)、Z(t)
- X 轴：帧索引（或换算成时间，单位秒）
- Y 轴：对应坐标分量（mm）
- 便于单独分析目标在某一方向上的运动规律
- `null` 帧处折线自动断开

#### 4.2 方案对比

| 方案 | 库 | 优点 | 缺点 |
|---|---|---|---|
| **方案 A（推荐）** | `matplotlib` | 已安装，轻量，输出静态 PNG | 无交互 |
| 方案 B | `plotly` | 交互式缩放、hover 显示数值 | 需额外安装，需浏览器 |
| 方案 C | `opencv` 绘图 | 无需额外依赖 | 代码冗长，美观度差 |

**结论**：`matplotlib` 已安装，满足需求，使用方案 A。

#### 4.3 关键可视化细节

- **nan 处理**：`matplotlib` 的 `plot` / `plot3D` 会自动在 `nan` 处断线，无需额外处理
- **3D 图视角**：`ax.view_init(elev=20, azim=-60)` 设置初始视角，保存时固定视角以便对比不同次采集
- **坐标轴方向**：相机坐标系中 Y 轴朝下，可在绘图时做 `Y = -Y` 翻转使图中 Y 轴朝上（更直观）
- **平滑**：可选用滑动窗口均值对轨迹平滑（去抖动），窗口大小 5~15 帧，仅影响可视化，不修改原始 JSON
- **输出**：
  - `vis/trajectory/trajectory_3d.png`（高 DPI，固定视角）
  - `vis/trajectory/trajectory_components.png`（X/Y/Z 三子图）

---

## 三、文件依赖关系与调用顺序

```
l515_capture.py
    │  生成 collect_data/session_XXXX/color/ + depth/
    ↓
video_sam3.py（改造后）
    │  输入: color/ 目录路径 + prompts 列表
    │  输出: {prompt: {frame_idx: mask}} 字典
    ↓
depth_extract.py
    │  输入: mask 字典 + depth/ 目录路径 + intrinsics.json
    │  输出: {prompt: [(X,Y,Z),...]} 字典 + 保存 trajectories_3d.json
    ↓
visualize_trajectory.py
    │  输入: trajectories_3d.json
    │  输出: vis/trajectory/trajectory_3d.png
    │         vis/trajectory/trajectory_components.png
```

也可以写一个 `pipeline.py` 将以上四步串联起来，一键运行。

---

## 四、方案取舍总结

| 决策点 | 选择 | 理由 |
|---|---|---|
| 多 prompt 分割方式 | 单 Session 顺序推理（方案 B） | 显存受限，逻辑清晰 |
| 深度保存格式 | `.npy`（uint16） | 无损，保留完整精度 |
| 深度统计量 | 中位数（默认） | 对边缘噪声鲁棒，同时输出代表像素坐标 |
| 轨迹维度 | 3D `(X, Y, Z)`（方案 C） | 反投影获取完整空间坐标，信息量更丰富 |
| 反投影方式 | 针孔相机模型 + intrinsics.json | `rs.intrinsics` 直接提供，精度最高 |
| 可视化库 | matplotlib（Axes3D） | 已安装，支持 3D 绘图，够用 |
| RGB-Depth 对齐 | `rs.align(rs.stream.color)` | SDK 原生支持，精度最高 |
| mask 合并策略 | `np.any(masks, axis=0)` | 同一 prompt 多实例合并 |

---

## 五、潜在风险与注意事项

1. **L515 近距离盲区**：最近约 0.25m 无法测深，目标太近时深度全为 0，需在 UI/日志中提示。

2. **SAM3 在第 0 帧未检测到目标**：若目标在视频开头不可见，整个 session 的追踪都会失败。**解决方案**：扫描前 N 帧，选择第一个 `out_binary_masks.any()` 为 True 的帧作为 `frame_index`。

3. **多 prompt 推理耗时**：N 个 prompt 需要 N 次完整 propagate，时间 ×N。对于 161 帧视频，每次约 15 秒，3 个 prompt 约需 45 秒。若 prompt 数量多，可考虑方案 A（多 session 并行，但显存要求高）。

4. **深度图中的反光问题**：门把手、金属表面对 L515 的结构光可能产生干扰，导致深度缺失，中位数过滤可部分缓解。

5. **pyrealsense2 与 sam3 环境兼容性**：`pyrealsense2` 已安装在 sam3 环境中，验证可正常 import，无冲突。
