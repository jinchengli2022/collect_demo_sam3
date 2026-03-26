# SAM 3 Examples 目录深度研究报告

> 作者：GitHub Copilot  
> 日期：2026-03-26  
> 研究对象：`/home/ljc/Git/sam3/examples/` 下全部 11 个 Notebook 文件

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [推理 API 的两条主线](#2-推理-api-的两条主线)
3. [各脚本详细分析](#3-各脚本详细分析)
   - [sam3_image_predictor_example.ipynb](#31-sam3_image_predictor_exampleipynb)
   - [sam3_image_batched_inference.ipynb](#32-sam3_image_batched_inferenceipynb)
   - [sam3_image_interactive.ipynb](#33-sam3_image_interactiveipynb)
   - [sam3_for_sam1_task_example.ipynb](#34-sam3_for_sam1_task_exampleipynb)
   - [sam3_video_predictor_example.ipynb](#35-sam3_video_predictor_exampleipynb)
   - [sam3_for_sam2_video_task_example.ipynb](#36-sam3_for_sam2_video_task_exampleipynb)
   - [sam3_agent.ipynb](#37-sam3_agentipynb)
   - [saco_gold_silver_eval_example.ipynb](#38-saco_gold_silver_eval_exampleipynb)
   - [saco_gold_silver_vis_example.ipynb](#39-saco_gold_silver_vis_exampleipynb)
   - [saco_veval_eval_example.ipynb](#310-saco_veval_eval_exampleipynb)
   - [saco_veval_vis_example.ipynb](#311-saco_veval_vis_exampleipynb)
4. [关键数据结构与返回值格式](#4-关键数据结构与返回值格式)
5. [坐标系约定](#5-坐标系约定)
6. [模型构建入口对比](#6-模型构建入口对比)
7. [常见踩坑点](#7-常见踩坑点)

---

## 1. 整体架构概览

SAM 3（Segment Anything Model 3）是 Meta 发布的下一代分割模型，支持**图像**和**视频**两大场景，并提供两套不同层次的推理 API：

| 层次 | API 风格 | 适用场景 |
|---|---|---|
| **高层 API（推荐）** | `build_sam3_video_predictor` + `handle_request` / `handle_stream_request` | 视频交互分割（自动检测+跟踪） |
| **低层 API（兼容旧版）** | `build_sam3_video_model` + `predictor.add_new_points` / `propagate_in_video` | 兼容 SAM 2 的视频 VOS 任务 |
| **图像 API** | `build_sam3_image_model` + `Sam3Processor` | 图像实例分割 |
| **批量图像 API** | `build_sam3_image_model` + 手动构造 `Datapoint` + `collate` | 批量多图多查询推理 |
| **Agent API** | SAM 3 + MLLM（如 Qwen3-VL）组合 | 复杂自然语言查询分割 |

---

## 2. 推理 API 的两条主线

### 主线 A：高层视频 API（`sam3_video_predictor_example.ipynb`）

```
build_sam3_video_predictor()
    └── handle_request(type="start_session", resource_path=...)    → session_id
    └── handle_request(type="reset_session", session_id=...)
    └── handle_request(type="add_prompt", session_id=..., frame_index=..., text=...)
    └── handle_stream_request(type="propagate_in_video", session_id=...)
           → yield {"frame_index": int, "outputs": {obj_id: mask_tensor}}
    └── handle_request(type="remove_object", session_id=..., obj_id=...)
    └── handle_request(type="close_session", session_id=...)
    └── predictor.shutdown()
```

**核心特点**：
- 支持文本 prompt（自动检测多实例）和点 prompt（手动精确指定）
- 使用 `handle_stream_request` 流式返回每帧结果，适合大视频节省内存
- `outputs` 字段是 `{obj_id: mask_tensor}` 字典，mask 形状为 `(1, H, W)`，布尔类型
- 每次调用 `propagate_in_video` 默认从 frame 0 向后传播到视频末尾

### 主线 B：低层视频 API（`sam3_for_sam2_video_task_example.ipynb`）

```
build_sam3_video_model()
    └── sam3_model.tracker  →  predictor (Sam3TrackerPredictor)
    └── predictor.backbone = sam3_model.detector.backbone  # 重要！需手动挂载

predictor.init_state(video_path=...)
predictor.clear_all_points_in_video(inference_state)
predictor.add_new_points(inference_state, frame_idx, obj_id, points, labels)
predictor.add_new_points_or_box(inference_state, frame_idx, obj_id, points, labels, box)
predictor.propagate_in_video(inference_state, start_frame_idx, max_frame_num_to_track, reverse, propagate_preflight=True)
    → yield (frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores)
```

**核心特点**：
- 完全兼容 SAM 2 的 API 接口
- 点坐标必须转换为**相对坐标**（除以宽高，值域 0~1）
- 框坐标格式为 XYXY，也需转为相对坐标
- `propagate_in_video` 直接返回 5 元组，`video_res_masks[i]` 形状为 `(1, H, W)`
- 提取 mask：`(video_res_masks[i] > 0.0).cpu().numpy()`

---

## 3. 各脚本详细分析

### 3.1 `sam3_image_predictor_example.ipynb`

**功能**：图像分割推理，使用高层 `Sam3Processor` 接口。

**流程**：
1. `build_sam3_image_model(bpe_path=...)` 构建模型，bpe 文件在 `sam3/assets/bpe_simple_vocab_16e6.txt.gz`
2. `Sam3Processor(model, confidence_threshold=0.5)` 创建处理器
3. `processor.set_image(image)` → `inference_state`（图像嵌入缓存）
4. 添加文本 prompt：`processor.set_text_prompt(state, prompt="shoe")`
5. 添加几何 prompt（框）：`processor.add_geometric_prompt(state, box=norm_box_cxcywh, label=True/False)`
6. `plot_results(img, inference_state)` 可视化

**关键细节**：
- 框格式为 **CxCyWH 归一化坐标**（中心点+宽高，值域 0~1）
- 需要使用 `box_xywh_to_cxcywh()` 将 XYWH 转换为 CxCyWH
- 再用 `normalize_bbox()` 除以图像宽高
- `label=True` 表示正样本框，`label=False` 表示负样本框（排除）
- 支持多框混合正负 prompt

---

### 3.2 `sam3_image_batched_inference.ipynb`

**功能**：批量多图像、多查询推理，使用最底层的模型直接调用接口，适合批量离线推理。

**流程**：
1. `build_sam3_image_model(bpe_path=...)` 构建模型
2. 手动定义预处理变换（resize 到 1008×1008，归一化到 [-1,1]）
3. 手动定义后处理器 `PostProcessImage`（控制置信度、mask格式等）
4. 为每张图创建 `Datapoint` 对象，添加文本或视觉 prompt
5. 用 `collate` 函数将多个 `Datapoint` 打包成 batch
6. `batch = copy_data_to_device(batch, device)` 发送到 GPU
7. `output = model(batch)` 前向推理
8. `processed_results = postprocessor.process_results(output, batch.find_metadatas)` 后处理
9. 通过推理时返回的 `id`（如 `id1`、`id2`）索引具体查询的结果

**关键细节**：
- 每次 `add_text_prompt()` / `add_visual_prompt()` 返回一个唯一 `id`，后续通过 `processed_results[id]` 取结果
- 视觉 prompt 框格式为 **XYXY 绝对像素坐标**（左上角+右下角）
- 负样本框通过 `labels=[False]` 指定，可以排除不想分割的区域
- 第一次 forward 因为模型编译（torch.compile）会很慢
- `PostProcessImage` 的 `detection_threshold=0.5` 控制置信度过滤
- `max_dets_per_img=-1` 时不限制返回数量，否则只返回 top-k

---

### 3.3 `sam3_image_interactive.ipynb`

**功能**：在 Jupyter Notebook 中提供**可交互的图形界面**（基于 `ipywidgets`），用于实时图像分割演示。

**核心类**：`Sam3SegmentationWidget`

**界面功能**：
- 图像上传（本地文件 或 URL）
- 文本 prompt 输入框 + 触发按钮
- 框绘制模式切换（正/负样本）
- 置信度滑块（实时调整阈值）
- 图像尺寸滑块
- 清除所有 prompt 按钮

**工作原理**：
1. 用户上传图片后，`Sam3Processor.set_image()` 预计算图像嵌入
2. 用户绘制框时，通过 `matplotlib` 事件监听鼠标拖拽，记录框坐标
3. 每次 prompt 变化自动重新推理并更新可视化
4. 使用 `%matplotlib widget` 实现 canvas 交互

**注意**：需要安装 `ipywidgets`，且需在支持 widget 的 Jupyter 环境中运行。

---

### 3.4 `sam3_for_sam1_task_example.ipynb`

**功能**：兼容 SAM 1 的交互式图像分割接口，直接使用 `model.predict_inst()`。

**关键 API**：`model.predict_inst()`（注意是在 **model** 上调用，不是在 processor 上）

**构建模型的特殊参数**：
```python
model = build_sam3_image_model(bpe_path=bpe_path, enable_inst_interactivity=True)
```
> ⚠️ 必须传 `enable_inst_interactivity=True`，否则无法使用 `predict_inst`。

**流程**：
1. `processor.set_image(image)` → `inference_state`
2. `model.predict_inst(inference_state, point_coords, point_labels, multimask_output=True)` → `(masks, scores, logits)`
3. 对分数排序取最优 mask
4. 可传入上一轮的 `logits` 作为 `mask_input` 进行迭代细化

**支持的 prompt 类型**：
- **点 prompt**：`point_coords (N, 2)` + `point_labels (N,)`（1=正，0=负）
- **框 prompt**：`box` 格式为 XYXY 绝对坐标
- **框+点混合**：同时传入 box 和 point
- **批量多对象**：传入 `(B, 1, 2)` 的点矩阵，对应 B 个对象

**高级用法**：
- `model.predict_inst_batch()`：同时处理多图像批量推理
- `processor.set_image_batch(img_batch)`：为多张图像同时计算嵌入
- `multimask_output=True` 时返回 3 个候选 mask，取分数最高的作为最终结果
- 迭代细化：将上一次的 `logits[best_idx]` 传给下一次的 `mask_input`

---

### 3.5 `sam3_video_predictor_example.ipynb`

**功能**：高层视频交互分割（SAM 3 原生 API），支持文本 prompt 自动多实例检测和追踪。

**核心流程**：
```python
predictor = build_sam3_video_predictor(gpus_to_use=range(torch.cuda.device_count()))

# 1. 开启会话
session_id = predictor.handle_request({"type": "start_session", "resource_path": video_path})["session_id"]

# 2. 重置（可选，切换 prompt 前必须）
predictor.handle_request({"type": "reset_session", "session_id": session_id})

# 3. 添加文本 prompt（自动检测所有实例）
predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": "person"})

# 4. 流式传播
for response in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session_id}):
    frame_idx = response["frame_index"]
    outputs = response["outputs"]  # {obj_id: mask_tensor (1, H, W), ...}

# 5. 删除对象
predictor.handle_request({"type": "remove_object", "session_id": session_id, "obj_id": 2})

# 6. 添加点 prompt（手动指定对象）
predictor.handle_request({
    "type": "add_prompt", "session_id": session_id, "frame_index": 0,
    "points": points_tensor,      # float32，相对坐标 (N, 2)
    "point_labels": labels_tensor, # int32
    "obj_id": obj_id,
})

# 7. 关闭
predictor.handle_request({"type": "close_session", "session_id": session_id})
predictor.shutdown()
```

**坐标转换工具**：
```python
def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    # coord_type="point": [[x/W, y/H], ...]
    # coord_type="box":   [[x/W, y/H, w/W, h/H], ...]
```

**高层 API 与低层 API 的 propagate 结果差异**：

| | 高层 API | 低层 API |
|---|---|---|
| 调用方法 | `handle_stream_request` | `predictor.propagate_in_video` |
| 返回格式 | `{"frame_index": int, "outputs": dict}` | 5元组 `(frame_idx, obj_ids, low_res, video_res, scores)` |
| outputs 中 mask | `{obj_id: tensor(1,H,W)}` | 需手动 `video_res_masks[i] > 0.0` |
| mask 值域 | 布尔型（已阈值化） | float（需手动 `> 0.0` 阈值化） |

---

### 3.6 `sam3_for_sam2_video_task_example.ipynb`

**功能**：兼容 SAM 2 的低层视频 API，使用 `Sam3TrackerPredictor`，适用于 VOS（视频目标分割）标准评测场景。

**构建方式（与高层 API 不同！）**：
```python
from sam3.model_builder import build_sam3_video_model
sam3_model = build_sam3_video_model()
predictor = sam3_model.tracker
predictor.backbone = sam3_model.detector.backbone  # ⚠️ 必须手动挂载 backbone
```

**三个核心示例**：

**示例 1：单对象分割+追踪**
- `predictor.add_new_points()` 添加点 prompt（可多次调用细化）
- `predictor.propagate_in_video()` 传播
- 可在中间帧继续添加点进行修正，再次传播

**示例 2：框 prompt**
- `predictor.add_new_points_or_box(box=rel_box)` 使用框 prompt（相对坐标 XYXY）
- 可在框 prompt 后继续添加点来细化

**示例 3：多对象同时分割**
- 对不同 `obj_id` 分别调用 `add_new_points_or_box`
- 单次 `propagate_in_video` 同时跟踪所有对象
- 返回的 `video_res_masks` 按 `obj_ids` 的索引顺序对应

**提取 mask 的标准写法**：
```python
for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(...):
    video_segments[frame_idx] = {
        out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)  # 注意：用 out_obj_ids 不是 obj_ids
    }
```

**注意**：还支持 `predictor.add_new_mask()` 直接传入 mask 作为 prompt（用于半监督 VOS）。

---

### 3.7 `sam3_agent.ipynb`

**功能**：SAM 3 + MLLM 联合推理，处理**复杂自然语言查询**（如"最左边穿蓝色背心的孩子"）。

**架构**：
```
复杂文本查询
    ↓
MLLM（如 Qwen3-VL-8B）
    ↓ 生成子查询/推理
SAM 3 图像分割
    ↓
可视化输出
```

**关键组件**：
- `sam3.agent.client_llm.send_generate_request`：向 vLLM 服务器发请求
- `sam3.agent.client_sam3.call_sam_service`：调用 SAM 3 分割
- `sam3.agent.inference.run_single_image_inference`：端到端单图推理

**LLM 配置**：支持本地 vLLM 服务（需独立 conda 环境）或外部 API。

**vLLM 启动命令**（在独立 conda 环境中）：
```bash
vllm serve Qwen/Qwen3-VL-8B-Thinking --tensor-parallel-size 4 \
    --allowed-local-media-path / --enforce-eager --port 8001
```

**局限**：不支持视频，目前仅用于图像。

---

### 3.8 `saco_gold_silver_eval_example.ipynb`

**功能**：SA-Co 数据集的**离线评测**，计算 CGF1 / IL-MCC / pMF1 指标。

**数据集**：
- **SA-Co/Gold**：7 个子集（MetaCLIP NP、SA-1B NP、Crowded、FG Food、FG Sports、Attributes、Wiki Common），每个子集 3 个 split
- **SA-Co/Silver**：10 个子集（BDD100K、DROID、EGO4D、Food Rec、GeoDE、iNaturalist、NGA Art、SAV、YT1B、FathomNet）

**评测指标**：
- `cgF1`：Co-Grounded F1 分数（×100）
- `il_mcc`：Image-Level MCC（马修相关系数）
- `pmF1`：Positive Micro F1（×100）

**评测器用法**：
```python
from sam3.eval.cgf1_eval import CGF1Evaluator
evaluator = CGF1Evaluator(gt_path=gt_paths, verbose=True, iou_type="segm")  # 或 "bbox"
summary = evaluator.evaluate(pred_path)  # pred 为 COCO 格式 JSON
```

**前提**：必须先运行推理并生成预测文件 `coco_predictions_segm.json`。

---

### 3.9 `saco_gold_silver_vis_example.ipynb`

**功能**：可视化 SA-Co/Gold 和 SA-Co/Silver 数据集的标注数据。

**流程**：
1. 加载 COCO 格式标注文件 → `annot_dfs`（含 images、annotations 等 DataFrame）
2. 随机采样正样本（有 mask 标注的图-名词短语对）
3. 三列对比展示：原图 | 纯 mask（白底） | mask 叠加到原图

**工具函数**（来自 `sam3.visualization_utils`）：
- `get_annot_dfs(file_list)` 批量加载 JSON
- `draw_masks_to_frame(frame, masks, colors)` 将多个 mask 渲染到帧上
- `pascal_color_map()` 生成区分度高的颜色映射表

---

### 3.10 `saco_veval_eval_example.ipynb`

**功能**：SA-Co/Veval（视频评测集）的**离线评测**。

**数据集**：3 个子集（SAV、YT1B、SmartGlasses）

**评测指标**：
- `cgf1`（即 `video_mask_demo_cgf1_micro_50_95`）
- `pHOTA`（即 `video_mask_all_phrase_HOTA`，视频跟踪指标）

**评测器用法**：
```python
from sam3.eval.saco_veval_eval import VEvalEvaluator
veval_evaluator = VEvalEvaluator(gt_annot_file=gt_annot_file, eval_res_file=eval_res_file)
eval_res = veval_evaluator.run_eval(pred_file=pred_file)
```

**智能缓存**：如果 `eval_res_file` 已存在则直接读取，避免重复计算。

---

### 3.11 `saco_veval_vis_example.ipynb`

**功能**：可视化 SA-Co/Veval 视频数据集的标注数据。

**数据格式**（与图像数据集的主要区别）：
- 标注中含有 `videos`、`video_np_pairs`（视频-名词短语对）字段
- `num_masklets > 0` 表示正样本（该视频中确实存在该名词短语对应的对象）
- 可视化多帧：按 `every_n_frames` 间隔采样，展示分割 mask 随时间的变化

**辅助工具**（来自本地 `utils.py`）：
- `get_all_annotations_for_frame(annot_dfs, video_id, frame_idx, data_dir, dataset)` 读取指定视频指定帧的所有标注

---

## 4. 关键数据结构与返回值格式

### 高层视频 API（`handle_stream_request`）返回的 `outputs`

```python
# response["outputs"] 的结构：
{
    obj_id_1: tensor(1, H, W),   # 布尔类型，True=前景
    obj_id_2: tensor(1, H, W),
    ...
}
```

提取 mask 的正确方式：
```python
for obj_id, mask in frame_outputs.items():
    mask_np = mask.cpu().numpy().squeeze()  # → (H, W)
    mask_bool = mask_np > 0.0  # 如果是 float 则需阈值化；如是布尔则直接用
```

### 低层视频 API（`propagate_in_video`）返回的 5 元组

```python
frame_idx          # int：当前帧索引
obj_ids            # list：本次处理的对象 ID 列表
low_res_masks      # tensor(N, 1, H_low, W_low)：低分辨率 mask logits
video_res_masks    # tensor(N, 1, H_video, W_video)：视频分辨率 mask logits（float）
obj_scores         # tensor(N,)：每个对象的分数

# 提取二值 mask：
(video_res_masks[i] > 0.0).cpu().numpy()  # → (1, H, W)
```

### 图像 API（`predict_inst`）返回值

```python
masks   # ndarray (num_masks, H, W)，布尔型
scores  # ndarray (num_masks,)，质量分数
logits  # ndarray (num_masks, 256, 256)，低分辨率 logit，可用于迭代细化
```

---

## 5. 坐标系约定

| 场景 | 格式 | 值域 |
|---|---|---|
| 图像 API 框 prompt（`Sam3Processor`） | CxCyWH（中心点+宽高） | 归一化 0~1 |
| 图像 API 批量推理框 prompt（`add_visual_prompt`） | XYXY（左上+右下） | **绝对像素坐标** |
| 图像 API SAM 1 兼容（`predict_inst`）框 | XYXY | **绝对像素坐标** |
| 视频高层 API 点 prompt | XY | **归一化 0~1**（需除以宽高） |
| 视频高层 API 框 prompt（`add_prompt`中的 `box`） | XYWH | 归一化 0~1 |
| 视频低层 API 点 prompt | XY | **归一化 0~1**（需除以宽高） |
| 视频低层 API 框 prompt | XYXY | **归一化 0~1**（需除以宽高） |

---

## 6. 模型构建入口对比

| 函数 | 用途 | 特殊参数 |
|---|---|---|
| `build_sam3_video_predictor(gpus_to_use=...)` | 高层视频 API | `gpus_to_use=range(N)` 或 `[0]` |
| `build_sam3_video_model()` | 低层视频 API（SAM 2 兼容） | 需手动 `predictor.backbone = ...` |
| `build_sam3_image_model(bpe_path=...)` | 图像 API（标准） | `bpe_path` 必填 |
| `build_sam3_image_model(bpe_path=..., enable_inst_interactivity=True)` | 图像 SAM 1 兼容 | 多传 `enable_inst_interactivity=True` |

所有模型都需要提前下载权重（通过 HuggingFace Hub）。

---

## 7. 常见踩坑点

1. **`propagate` 不是有效 request type**：必须使用 `propagate_in_video`，且需用 `handle_stream_request` 而非 `handle_request`。

2. **mask 的 `obj_out` 不是字典**：高层 API 的 `outputs[obj_id]` 直接是 mask tensor，不是 `{"mask": tensor}` 结构。

3. **mask 形状问题**：返回的 mask 可能是 `(1, H, W)` 或更多维度，需用 `np.squeeze()` 处理，并用 `cv2.resize` 保证与视频帧尺寸一致后再做布尔索引。

4. **低层 API 需手动挂载 backbone**：`predictor.backbone = sam3_model.detector.backbone`，否则推理报错。

5. **`enable_inst_interactivity=True` 不能省略**：使用 SAM 1 兼容 `predict_inst` 接口时必须在构建模型时传入此参数。

6. **vLLM 需独立 conda 环境**：SAM 3 Agent 中用到的 vLLM 与 SAM 3 依赖可能冲突，需隔离安装。

7. **切换 text prompt 前必须 reset_session**：如果已有 text prompt 的传播结果，想换一个文本 prompt，必须先调用 `reset_session`，否则结果是错误的。

8. **视频低层 API 用 `out_obj_ids` 不是 `obj_ids`**：`propagate_in_video` 返回的第二个元素是当前帧有效的对象集合，要用 `enumerate(out_obj_ids)` 来索引 `video_res_masks`。

9. **批量图像 API 第一次 forward 很慢**：因为有 `torch.compile` JIT 编译，第一次需要几分钟，后续调用会快很多。

10. **bpe 路径**：`bpe_simple_vocab_16e6.txt.gz` 在 `sam3/assets/` 下，需用 `os.path.join(sam3_root, "assets/bpe_simple_vocab_16e6.txt.gz")`。
