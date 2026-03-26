"""
Step 1: D435i 数据采集脚本
修复了 AssertionError 并添加了深度滤镜优化
"""
from __future__ import annotations

import json
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs  # type: ignore[import-untyped]

# ================= D435i 推荐配置 =================
COLOR_WIDTH = 640   
COLOR_HEIGHT = 480  
DEPTH_WIDTH = 640   
DEPTH_HEIGHT = 480
FPS = 30
# =================================================

def create_session_dir(base: str = "collect_data") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(base) / f"session_{ts}"
    (session_dir / "color").mkdir(parents=True, exist_ok=True)
    (session_dir / "depth").mkdir(parents=True, exist_ok=True)
    (session_dir / "vis" / "depth").mkdir(parents=True, exist_ok=True)
    return session_dir


def make_depth_vis(depth_mm: np.ndarray, color_img: np.ndarray) -> np.ndarray:
    """
    生成深度可视化图：
    左半：RGB 彩色图
    右半：深度伪彩色图（COLORMAP_JET，0~4000 mm 映射）
    两图横向拼接，便于对比查看。
    """
    MAX_DEPTH_MM = 4000

    # 深度归一化到 0~255，超出范围截断
    depth_clipped = np.clip(depth_mm, 0, MAX_DEPTH_MM).astype(np.float32)
    depth_norm = (depth_clipped / MAX_DEPTH_MM * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    # 无深度的像素（值为 0）置为灰色，区分"有效近距"和"无数据"
    no_data_mask = (depth_mm == 0)
    depth_color[no_data_mask] = (80, 80, 80)

    # 在深度图右上角写深度范围说明
    cv2.putText(
        depth_color, "0~4000 mm  (COLORMAP_JET)",
        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )

    # 横向拼接 RGB | Depth
    vis = np.concatenate([color_img, depth_color], axis=1)
    return vis

def save_intrinsics(session_dir: Path, profile: rs.video_stream_profile) -> None:
    intr = profile.get_intrinsics()
    data = {
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.ppx,
        "cy": intr.ppy,
        "width": intr.width,
        "height": intr.height,
        "model": str(intr.model),
        "coeffs": list(intr.coeffs),
    }
    with open(session_dir / "intrinsics.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Step1] 内参已保存: {session_dir / 'intrinsics.json'}")

def save_metadata(session_dir: Path, fps: float, total_frames: int, w: int, h: int) -> None:
    data = {
        "session": session_dir.name,
        "fps": fps,
        "total_frames": total_frames,
        "color_width": w,
        "color_height": h,
        "depth_unit": "mm",
        "timestamp": datetime.now().isoformat(),
        "device": "D435i"
    }
    with open(session_dir / "metadata.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Step1] metadata 已保存: {session_dir / 'metadata.json'}")

def capture(max_frames: int = 0, base_dir: str = "collect_data") -> Path | None:
    print("[Step1] 打开摄像头预览，按键说明：")
    print("        [r] 开始录制")
    print("        [s] 暂停 / 继续录制")
    print("        [q] 退出")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)

    # 启动 Pipeline
    profile = pipeline.start(config)

    # 定义对齐和滤镜
    align = rs.align(rs.stream.color)
    hole_filling = rs.hole_filling_filter()  # 填充深度图黑洞

    # 状态机
    # "idle"      → 预览中，尚未录制
    # "recording" → 正在录制
    # "paused"    → 已暂停
    state = "idle"
    session_dir: Path | None = None
    frame_idx = 0
    stopped = False

    def _on_signal(sig: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, _on_signal)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    try:
        while not stopped:
            frames = pipeline.wait_for_frames(timeout_ms=5000)

            # 1. 空间对齐 (深度对齐到彩色)
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # 2. 深度滤镜处理
            depth_frame = hole_filling.process(depth_frame)

            # 3. 转换为 Numpy 数组
            color_img = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_mm = (depth_raw.astype(np.float32) * depth_scale * 1000.0).astype(np.uint16)

            # 4. 生成深度可视化图
            depth_vis = make_depth_vis(depth_mm, color_img)

            # 5. 在画面左上角叠加状态提示
            status_text = {
                "idle":      "PREVIEW  [r] record  [q] quit",
                "recording": f"REC  {frame_idx:04d} frames  [s] pause  [q] quit",
                "paused":    f"PAUSED   {frame_idx:04d} frames  [s] resume  [q] quit",
            }[state]
            status_color = {
                "idle": (180, 180, 180),
                "recording": (0, 60, 220),
                "paused": (0, 180, 220),
            }[state]
            cv2.putText(depth_vis, status_text, (8, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)

            # 录制中：绘制红色录制指示圆点
            if state == "recording":
                cv2.circle(depth_vis, (depth_vis.shape[1] - 20, 20), 8, (0, 0, 220), -1)

            cv2.imshow("D435i Capture", depth_vis)

            # 6. 按键处理（waitKey 返回低 8 位）
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("r"):
                if state == "idle":
                    # 第一次按 r：初始化 session 目录并开始录制
                    session_dir = create_session_dir(base_dir)
                    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
                    save_intrinsics(session_dir, color_stream)
                    state = "recording"
                    frame_idx = 0
                    print(f"[Step1] 开始录制，保存至: {session_dir}")
                elif state == "paused":
                    # 暂停状态下按 r 也可继续
                    state = "recording"
                    print("[Step1] 继续录制")

            elif key == ord("s"):
                if state == "recording":
                    state = "paused"
                    print(f"[Step1] 已暂停，当前已录制 {frame_idx} 帧")
                elif state == "paused":
                    state = "recording"
                    print("[Step1] 继续录制")

            # 7. 仅录制状态下保存数据
            if state == "recording":
                if max_frames > 0 and frame_idx >= max_frames:
                    print(f"[Step1] 已达到最大帧数 {max_frames}，自动停止录制")
                    break

                fname = f"frame_{frame_idx:04d}"
                cv2.imwrite(str(session_dir / "color" / f"{fname}.png"), color_img)
                np.save(str(session_dir / "depth" / f"{fname}.npy"), depth_mm)
                cv2.imwrite(str(session_dir / "vis" / "depth" / f"{fname}.png"), depth_vis)

                frame_idx += 1
                if frame_idx % 30 == 0:
                    print(f"[Step1] 已录制 {frame_idx} 帧")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        if session_dir is not None:
            save_metadata(session_dir, float(FPS), frame_idx, COLOR_WIDTH, COLOR_HEIGHT)
            print(f"[Step1] 录制结束，共保存 {frame_idx} 帧，路径: {session_dir}")
        else:
            print("[Step1] 未开始录制，退出。")

    return session_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="D435i 数据采集")
    parser.add_argument("--max-frames", type=int, default=0, help="最大采集帧数")
    parser.add_argument("--base-dir", type=str, default="collect_data")
    args = parser.parse_args()
    capture(max_frames=args.max_frames, base_dir=args.base_dir)