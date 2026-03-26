# collect_scripts 目录深度研究报告

> 研究对象：`/home/ljc/Git/sam3/collect_scripts/`  
> 关联文件：`/home/ljc/Git/sam3/basic.py`  
> 日期：2026-03-26

---

## 一、目录结构

```
collect_scripts/
└── video_sam3.py   # 唯一脚本，核心数据采集逻辑
```

目录当前只有一个脚本文件。结合 `basic.py`（两者内容基本一致，`collect_scripts/video_sam3.py` 是 `basic.py` 的改良版），以及 `vibe/plan.md` 中描述的目标（基于 L515 深度相机的数据采集脚本），可以判断：

> **`collect_scripts/` 是为 SAM3 数据采集任务准备的脚本目录，当前处于早期开发阶段，只有一个视频语义分割+帧保存脚本。**

---

## 二、`video_sam3.py` 完整工作流分析

### 2.1 整体流程

```
配置 HuggingFace 登录
    ↓
build_sam3_video_predictor()  # 加载模型
    ↓
start_session(video_path)     # 开启推理会话，模型载入所有视频帧
    ↓
reset_session()               # 清除旧状态
    ↓
add_prompt(frame_index=0, text=text_prompt)  # 在第0帧添加文本 prompt
    ↓
cv2 读取所有视频帧到内存
    ↓
handle_stream_request("propagate_in_video")  # 流式传播，逐帧返回分割结果
    ↓
逐帧叠加绿色 mask → 保存为 PNG 序列（output_frames/frame_XXXX.png）
    ↓
close_session() + shutdown()  # 释放资源
```

---

### 2.2 逐段详解

#### 第 1 段：环境配置

```python
HF_TOKEN = ""  # ← 注意：collect_scripts 版本中 Token 为空！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
login(token=HF_TOKEN)
```

- 使用 HuggingFace 镜像站（`hf-mirror.com`）下载模型权重，适配中国网络环境。
- `collect_scripts/video_sam3.py` 中 `HF_TOKEN = ""`，表示该脚本还未填入实际 token，是一个模板/草稿状态。
- `basic.py` 中填入了具体 token。

---

#### 第 2 段：模型初始化

```python
video_predictor = build_sam3_video_predictor()
video_path = "/home/ljc/Git/sam3/data/test.mp4"
text_prompt = 'door handle'
```

- `build_sam3_video_predictor()` 不传 `gpus_to_use` 参数时，默认使用所有可用 GPU（单卡环境则用 GPU 0）。
- `text_prompt` 是**可配置的核心参数**，决定模型追踪什么物体。
  - `basic.py` 早期版本用了 `"door section"`，当前版本改为 `"door handle"`——说明作者在测试不同的分割目标。

---

#### 第 3 段：会话管理

```python
response = video_predictor.handle_request(
    request=dict(type="start_session", resource_path=video_path)
)
session_id = response["session_id"]

video_predictor.handle_request(
    request=dict(type="reset_session", session_id=session_id)
)

video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text=text_prompt,
    )
)
```

**`start_session`**：
- 传入视频路径（支持 `.mp4` 或 JPEG 帧目录）
- 模型预加载所有帧并缓存图像特征
- 返回 `session_id`（UUID 字符串）用于后续请求标识

**`reset_session`**：
- 清除该 session 内所有已添加的 prompt 和追踪状态
- 在重复使用同一 session 测试不同 prompt 时**必须调用**

**`add_prompt`**：
- `frame_index=0`：在视频第 0 帧上添加 prompt
- `text=text_prompt`：文本 prompt，模型自动检测该帧中所有匹配实例
- 返回的 `response["outputs"]` 包含第 0 帧的检测结果（可用于预览）

---

#### 第 4 段：视频帧读取

```python
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    video_frames.append(frame)  # BGR 格式
cap.release()
```

- 用 OpenCV 将整个视频读入内存（BGR numpy array 列表）
- **注意**：SAM3 内部也会读一遍视频帧用于模型推理，这里是**第二次读取**，专门用于可视化叠加
- 对于长视频（> 500 帧，4K 分辨率），全部读入内存可能消耗大量 RAM，需注意

---

#### 第 5 段：传播推理（核心）

```python
outputs_per_frame = {}
for response in video_predictor.handle_stream_request(
    request=dict(
        type="propagate_in_video",
        session_id=session_id,
    )
):
    outputs_per_frame[response["frame_index"]] = response["outputs"]
```

- `handle_stream_request` 是**生成器接口**，逐帧 yield 结果
- 每个 `response` 包含：
  - `response["frame_index"]`：当前帧索引（int）
  - `response["outputs"]`：当前帧的分割结果字典

**`response["outputs"]` 的完整结构**（来自 `sam3_video_inference.py` 第 515 行）：

```python
{
    "out_obj_ids":     np.ndarray,  # shape (N,), dtype=int64  — N 个追踪对象的 ID
    "out_binary_masks": np.ndarray, # shape (N, H, W), dtype=bool — 每个对象的二值 mask
    "out_probs":       np.ndarray,  # shape (N,), dtype=float32 — 每个对象的置信度
    "out_boxes_xywh":  np.ndarray,  # shape (N, 4), dtype=float32 — 边界框（归一化 XYWH）
    "frame_stats":     dict or None  # 帧统计信息（追踪对象数、丢弃数等）
}
```

**关键特性**：
- `out_binary_masks` 已经是**视频原始分辨率**（H×W），**布尔类型**，无需 resize 或阈值化
- N=0 时表示该帧没有检测到任何对象（dict 中数组为空）
- 对象 ID 从 0 开始，跨帧保持一致（同一物体在不同帧有相同 obj_id）

---

#### 第 6 段：mask 叠加与保存（与 basic.py 的关键差异）

```python
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

for frame_idx, frame in enumerate(video_frames):
    vis_frame = frame.copy()   # ← 关键：拷贝原帧，避免修改原始数据
    
    if frame_idx in outputs_per_frame:
        out = outputs_per_frame[frame_idx]
        obj_ids = out.get("out_obj_ids", [])
        binary_masks = out.get("out_binary_masks", [])

        for i, obj_id in enumerate(obj_ids):
            mask = binary_masks[i]   # shape (H, W), bool
            if mask.any():
                color_mask = np.zeros_like(vis_frame, dtype=np.uint8)
                color_mask[mask] = [0, 255, 0]   # 绿色 (BGR)
                vis_frame = cv2.addWeighted(vis_frame, 0.7, color_mask, 0.3, 0)
    
    save_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
    cv2.imwrite(save_path, vis_frame)
```

**与 `basic.py` 的核心差异**：

| 方面 | `basic.py`（旧） | `video_sam3.py`（新） |
|---|---|---|
| 输出格式 | MP4 视频文件 | **PNG 序列**（逐帧保存） |
| 输出目录 | 当前目录 `output_mask_video.mp4` | `output_frames/frame_XXXX.png` |
| 帧拷贝 | 直接修改 `frame` | `vis_frame = frame.copy()` 保护原始数据 |
| 数组访问 | `out["out_obj_ids"]`（直接索引） | `out.get("out_obj_ids", [])` 更安全 |

**为什么改为保存 PNG 序列？**

保存 PNG 序列而非 MP4 视频有以下优势：
1. **无损**：PNG 是无损压缩，mask 边界不会因视频编码产生模糊
2. **逐帧可访问**：可单独读取某一帧，便于标注、检查和后处理
3. **与深度数据对齐**：L515 相机同步采集 RGB+深度，两者都以帧为单位存储，PNG 序列便于按帧索引对齐
4. **灵活性**：后续可用 `ffmpeg` 自由合成为视频，也可直接用于训练数据集

---

#### 第 7 段：资源释放

```python
video_predictor.handle_request(
    request=dict(type="close_session", session_id=session_id)
)
video_predictor.shutdown()
```

- `close_session`：释放该会话占用的 GPU 内存（session 状态、缓存的帧特征等）
- `shutdown`：关闭多 GPU 进程组（`torch.distributed` 进程）
- **必须调用**，否则 GPU 内存不会释放，且进程不会正常退出

---

## 三、当前脚本的局限性

1. **只支持文本 prompt**：当前仅使用 `text=text_prompt` 在第 0 帧添加 prompt，不支持点/框 prompt，对于难以用文字描述的物体（如特定形状的零件）效果可能不好。

2. **prompt 只加在第 0 帧**：如果目标物体在第 0 帧没有出现或被遮挡，则整个视频的追踪都会失败。

3. **全帧读入内存**：`video_frames` 列表存储了所有视频帧的 BGR numpy array，对于长视频内存压力大。

4. **没有深度数据集成**：脚本目前只处理 RGB 视频，尚未集成 L515 的深度（Depth）和红外（IR）通道。

5. **单一颜色标注**：所有对象都用同一绿色，无法区分多个不同的追踪目标。

6. **没有置信度过滤**：`out_probs` 字段未使用，低置信度的噪声检测也会被渲染到输出图像上。

7. **输出路径硬编码**：`output_dir = "output_frames"` 和 `video_path` 都是硬编码路径，不适合批量采集。

---

## 四、与 L515 深度相机数据采集的关系

结合 `vibe/plan.md` 的目标，该脚本是**数据采集管线的 RGB 分割部分**，未来需要扩展以支持：

1. **实时采集**：从 L515 相机实时读取 RGB 流（而非离线 MP4），使用 `pyrealsense2` SDK
2. **深度数据同步**：同步保存对应的深度图（`frame_XXXX_depth.png` 或 `.npy`）
3. **相机内参保存**：保存 L515 的 RGB 和深度内参矩阵，用于后续 3D 重建
4. **mask 与深度的像素对齐**：利用 L515 的深度对齐功能（`rs.align`），将深度图对齐到 RGB 坐标系，再将 SAM3 mask 应用于深度图，提取目标物体的 3D 点云

---

## 五、关键依赖与运行环境

| 依赖 | 版本/说明 |
|---|---|
| Python 环境 | `conda activate sam3`（Python 3.12） |
| `sam3` | 本地安装（`pip install -e .`） |
| `torch` | CUDA 版本，支持 bfloat16 |
| `opencv-python` | 用于视频读写和图像处理 |
| `huggingface_hub` | 用于模型权重下载 |
| GPU 显存 | 约 8GB（161 帧 720p 视频峰值约 7.7GB） |

**运行命令**：
```bash
conda activate sam3
cd /home/ljc/Git/sam3
python collect_scripts/video_sam3.py
```

**输出**：
```
collect_scripts/../output_frames/
├── frame_0000.png
├── frame_0001.png
├── ...
└── frame_0160.png
```
