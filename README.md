# collect_scripts 使用指南

基于 **D435i（RealSense）+ SAM3** 的多目标深度轨迹采集管线，共四个步骤：

```
Step 1  capture.py          采集 RGB + 深度帧
Step 2  video_sam3.py       SAM3 多 Prompt 分割，生成逐帧 mask
Step 3  depth_extract.py    深度抠像 + 反投影，生成 3D 轨迹坐标
Step 4  visualize_trajectory.py   绘制轨迹图
        pipeline.py         一键串联 Step 2~4
```

---

## 环境准备

```bash
# 确保已激活 sam3 环境，pyrealsense2 已安装
conda activate sam3
python -c "import pyrealsense2; print('OK')"
```

---

## Step 1：采集数据（`capture.py`）

连接 D435i 后运行：

```bash
cd collect_scripts
python capture.py
```

### 按键操作

| 按键 | 功能 |
|------|------|
| `r` | **开始录制**（创建新 session 目录，开始保存帧） |
| `s` | **暂停 / 继续**录制（帧序号不重置，文件连续） |
| `q` | **退出**（自动保存 metadata.json） |

> 启动后先进入**预览模式**，摄像头窗口左半显示 RGB、右半显示深度伪彩色，不会写入任何文件。  
> 确认画面正常后再按 `r` 开始录制。

### 可选参数

```bash
# 最多录制 300 帧后自动停止
python capture.py --max-frames 300

# 指定保存根目录（默认 collect_data/）
python capture.py --base-dir /data/my_sessions
```

### 输出目录结构

录制完成后，在 `collect_data/` 下生成一个以时间戳命名的 session 目录：

```
collect_data/session_YYYYMMDD_HHMMSS/
├── color/
│   ├── frame_0000.png      # BGR 彩色图（640×480）
│   └── ...
├── depth/
│   ├── frame_0000.npy      # 对齐后深度图（uint16，单位 mm）
│   └── ...
├── vis/
│   └── depth/
│       ├── frame_0000.png  # RGB | 深度伪彩色 横向拼接，可直接查看
│       └── ...
├── intrinsics.json         # 相机内参（fx, fy, cx, cy）
└── metadata.json           # 帧数、FPS、时间戳等
```

> **`vis/depth/`**：每帧的 RGB 彩色图和深度伪彩色图横向拼接，无需任何工具即可用图片查看器浏览深度效果。深度色标：蓝色（近）→ 红色（远），灰色表示无深度数据。

---

## Step 2：SAM3 分割（`video_sam3.py`）

对采集好的帧序列运行文本驱动的目标分割。每个 prompt 独立推理，结果互不干扰。

```bash
python video_sam3.py <session_dir> --prompts "目标1" "目标2" ...
```

### 示例

```bash
python video_sam3.py collect_data/session_20260326_172631 \
    --prompts "door handle" "person" --debug-vis
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `session_dir` | （必填） | Step 1 生成的 session 目录路径 |
| `--prompts` | （必填） | 一个或多个文本 prompt，空格隔开，多词用引号括住 |
| `--frame-index` | `0` | 在第几帧添加 prompt（目标首次出现的帧） |
| `--debug-vis` | 关闭 | 开启后生成 `vis/mask/` 可视化图（RGB+mask 叠加） |

### 输出

```
session_dir/
├── masks/
│   ├── door_handle/
│   │   ├── frame_0000.npy   # bool 数组，shape=(H, W)
│   │   └── ...
│   └── person/
│       └── ...
└── vis/mask/                # 仅 --debug-vis 时生成
    ├── door_handle/
    │   ├── frame_0000.png   # 左：RGB+mask轮廓  右：mask区域深度
    │   └── ...
    └── person/
        └── ...
```

> **目标首帧选择**：若目标在第 0 帧不可见，用 `--frame-index N` 指定目标第一次出现的帧编号。

---

## Step 3：深度抠像 + 三维坐标（`depth_extract.py`）

读取 mask 和深度图，对每帧每目标计算相机坐标系下的三维位置 `(X, Y, Z)`（单位 mm）。

```bash
python depth_extract.py <session_dir> --prompts "目标1" "目标2" ...
```

### 示例

```bash
python depth_extract.py collect_data/session_20260326_172631 \
    --prompts "door handle" "person" \
    --stat-method median \
    --debug-vis
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `session_dir` | （必填） | session 目录路径 |
| `--prompts` | （必填） | 与 Step 2 保持一致 |
| `--stat-method` | `median` | 深度统计量：`median`（中位数，推荐）或 `mean`（均值） |
| `--debug-vis` | 关闭 | 生成 `vis/depth/{prompt}/` 标注图 |

### 输出

```
session_dir/
├── trajectories_3d.json         # 三维轨迹数据
└── vis/depth/                   # 仅 --debug-vis 时生成
    ├── door_handle/
    │   ├── frame_0000.png       # 深度伪彩色 + 绿点标注代表像素 + 坐标文字
    │   └── ...
    └── person/
        └── ...
```

`trajectories_3d.json` 格式示例：

```json
{
  "metadata": { "fps": 30, "total_frames": 150, "coord_unit": "mm", ... },
  "trajectories": {
    "door handle": [[120.3, -45.2, 1250.5], [119.8, -45.0, 1248.0], null, ...],
    "person":      [[-230.5, 80.1, 2100.0], null, ...]
  }
}
```

> `null` 表示该帧目标不可见或深度无效。

---

## Step 4：轨迹可视化（`visualize_trajectory.py`）

读取 `trajectories_3d.json`，输出两张静态图。

```bash
python visualize_trajectory.py <session_dir>
```

### 示例

```bash
python visualize_trajectory.py collect_data/session_20260326_172631 \
    --smooth-window 7
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `session_dir` | （必填） | session 目录路径 |
| `--smooth-window` | `1` | 轨迹平滑窗口帧数（1=不平滑，推荐 5~15 去抖动） |

### 输出

```
session_dir/vis/trajectory/
├── trajectory_3d.png           # 3D 空间轨迹图（相机坐标系）
└── trajectory_components.png   # X(t) / Y(t) / Z(t) 三子图
```

---

## 一键运行 Step 2~4（`pipeline.py`）

Step 1 需要实际摄像头，单独运行。Step 2~4 可用 `pipeline.py` 串联：

```bash
python pipeline.py <session_dir> --prompts "目标1" "目标2" [选项]
```

### 完整示例

```bash
# 最常用：中位数深度 + debug 可视化 + 轨迹平滑
python pipeline.py collect_data/session_20260326_172631 \
    --prompts "door handle" "person" \
    --frame-index 0 \
    --stat-method median \
    --smooth-window 7 \
    --debug-vis
```

### 所有参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `session_dir` | （必填） | session 目录路径 |
| `--prompts` | （必填） | 文本 prompt 列表 |
| `--frame-index` | `0` | SAM3 添加 prompt 的起始帧 |
| `--stat-method` | `median` | `median` 或 `mean` |
| `--smooth-window` | `1` | 轨迹平滑窗口大小 |
| `--debug-vis` | 关闭 | 生成所有 `vis/` 可视化图 |

---

## 典型完整流程

```bash
cd collect_scripts

# 1. 采集数据（按 r 开始，s 暂停，q 结束）
python capture.py
# → 生成 collect_data/session_20260326_172631/

# 2~4. 一键运行后处理
python pipeline.py collect_data/session_20260326_172631 \
    --prompts "door handle" "person" \
    --smooth-window 7 \
    --debug-vis
```

运行完毕后，查看结果：

```
collect_data/session_20260326_172631/
├── vis/depth/          ← 采集时的深度预览图（逐帧 RGB|深度）
├── vis/mask/           ← SAM3 分割结果叠加图（--debug-vis）
├── vis/depth/{prompt}/ ← 深度抠像代表点标注图（--debug-vis）
├── vis/trajectory/     ← 3D 轨迹图 + X/Y/Z 分量图
└── trajectories_3d.json
```

---

## 常见问题

**Q：按 `r` 后窗口无变化？**  
摄像头正常，录制已开始，帧会静默写入磁盘。观察终端每 30 帧打印一次进度。

**Q：SAM3 分割结果全为空 mask？**  
目标可能在第 0 帧不在画面内，尝试 `--frame-index 10`（或目标首次出现的帧号）。

**Q：`trajectories_3d.json` 中大量 `null`？**  
深度图该区域值为 0（反光/遮挡/超出量程），属正常现象。可尝试 `--stat-method mean` 或调整采集距离。

**Q：`pipeline.py` 导入报错 `No module named 'collect_scripts'`？**  
需要在项目根目录运行：
```bash
cd /home/ljc/Git/collect_demo_sam3
python -m collect_scripts.pipeline <session_dir> --prompts "..."
```
