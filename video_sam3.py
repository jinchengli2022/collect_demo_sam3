import torch
import os
import cv2
import numpy as np
from PIL import Image
from huggingface_hub import login

# ================= 1. 环境配置 =================
HF_TOKEN = ""
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    login(token=HF_TOKEN)
    print("HF 登录成功")
except:
    print("HF 登录失败")

from sam3.model_builder import build_sam3_video_predictor

# ================= 2. 初始化 =================
video_predictor = build_sam3_video_predictor()
video_path = "/home/ljc/Git/sam3/data/test.mp4"
output_video_path = "output_mask_video.mp4"
text_prompt = 'door handle'

# ================= 3. 开启会话 =================
response = video_predictor.handle_request(
    request=dict(type="start_session", resource_path=video_path)
)
session_id = response["session_id"]

# 重置会话（确保干净状态）
video_predictor.handle_request(
    request=dict(type="reset_session", session_id=session_id)
)

# 添加文本提示
video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text=text_prompt,
    )
)

# ================= 4. 读取原视频帧用于可视化 =================
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    video_frames.append(frame)  # BGR 格式，用于后续写入
cap.release()

print(f"视频共 {len(video_frames)} 帧，正在运行传播...")

# ================= 5. 使用 propagate_in_video + handle_stream_request 逐帧处理 =================
outputs_per_frame = {}
for response in video_predictor.handle_stream_request(
    request=dict(
        type="propagate_in_video",
        session_id=session_id,
    )
):
    outputs_per_frame[response["frame_index"]] = response["outputs"]

print(f"传播完成，共处理 {len(outputs_per_frame)} 帧，正在写入视频...")

# ================= 6. 将 mask 叠加到视频帧并写入输出视频 =================
output_dir = "output_frames"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"开始保存 PNG 序列到目录: {output_dir}...")

for frame_idx, frame in enumerate(video_frames):
    # 拷贝一份原图，避免在循环中重复修改同一张图
    vis_frame = frame.copy()
    
    if frame_idx in outputs_per_frame:
        out = outputs_per_frame[frame_idx]
        # out 结构: {"out_obj_ids": ndarray(N,), "out_binary_masks": ndarray(N, H, W)}
        obj_ids = out.get("out_obj_ids", [])
        binary_masks = out.get("out_binary_masks", [])

        for i, obj_id in enumerate(obj_ids):
            mask = binary_masks[i]  # shape (H, W), bool
            
            # 如果 mask 是布尔型且有内容
            if mask.any():
                # 创建彩色遮罩（绿色）
                color_mask = np.zeros_like(vis_frame, dtype=np.uint8)
                color_mask[mask] = [0, 255, 0]  # BGR 格式：绿色
                
                # 叠加到原图上 (0.7 原图权重, 0.3 遮罩权重)
                vis_frame = cv2.addWeighted(vis_frame, 0.7, color_mask, 0.3, 0)
    
    # 生成文件名，例如 frame_0000.png, frame_0001.png
    save_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
    cv2.imwrite(save_path, vis_frame)

print(f"处理完成！所有帧已保存至: {output_dir}")

# ================= 7. 关闭会话并释放资源 =================
video_predictor.handle_request(
    request=dict(type="close_session", session_id=session_id)
)
video_predictor.shutdown()

print(f"处理完成！视频已保存至: {output_video_path}")