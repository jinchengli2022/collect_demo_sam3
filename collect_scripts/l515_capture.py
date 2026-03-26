"""
Step 1: L515 数据采集脚本
采集 RGB + 深度帧（对齐到 RGB 坐标系），保存到 collect_data/session_YYYYMMDD_HHMMSS/
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

COLOR_WIDTH = 1280
COLOR_HEIGHT = 720
DEPTH_WIDTH = 1024
DEPTH_HEIGHT = 768
FPS = 30


def create_session_dir(base: str = "collect_data") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(base) / f"session_{ts}"
    (session_dir / "color").mkdir(parents=True, exist_ok=True)
    (session_dir / "depth").mkdir(parents=True, exist_ok=True)
    return session_dir


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


def save_metadata(
    session_dir: Path,
    fps: float,
    total_frames: int,
    color_width: int,
    color_height: int,
) -> None:
    data = {
        "session": session_dir.name,
        "fps": fps,
        "total_frames": total_frames,
        "color_width": color_width,
        "color_height": color_height,
        "depth_unit": "mm",
        "timestamp": datetime.now().isoformat(),
    }
    with open(session_dir / "metadata.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Step1] metadata 已保存: {session_dir / 'metadata.json'}")


def capture(max_frames: int = 0, base_dir: str = "collect_data") -> Path:
    """
    采集 RGB + 深度帧并保存到 session 目录。
    max_frames=0 表示一直采集，直到按 q 或 Ctrl+C。
    返回 session 目录路径。
    """
    session_dir = create_session_dir(base_dir)
    print(f"[Step1] 开始采集，保存至: {session_dir}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    color_profile = profile.get_stream(rs.stream.color)
    assert isinstance(color_profile, rs.video_stream_profile)
    save_intrinsics(session_dir, color_profile)

    frame_idx = 0
    stopped = False

    def _on_signal(sig: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, _on_signal)

    try:
        while not stopped:
            if max_frames > 0 and frame_idx >= max_frames:
                break

            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_img: np.ndarray = np.asanyarray(color_frame.get_data())
            depth_raw: np.ndarray = np.asanyarray(depth_frame.get_data())

            depth_scale: float = profile.get_device().first_depth_sensor().get_depth_scale()
            depth_mm = (depth_raw.astype(np.float32) * depth_scale * 1000.0).astype(np.uint16)

            fname = f"frame_{frame_idx:04d}"
            cv2.imwrite(str(session_dir / "color" / f"{fname}.png"), color_img)
            np.save(str(session_dir / "depth" / f"{fname}.npy"), depth_mm)

            cv2.imshow("L515 capture (press q to stop)", color_img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"[Step1] 已采集 {frame_idx} 帧")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        save_metadata(session_dir, float(FPS), frame_idx, COLOR_WIDTH, COLOR_HEIGHT)
        print(f"[Step1] 采集完成，共 {frame_idx} 帧")

    return session_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="L515 RGB-D 数据采集")
    parser.add_argument("--max-frames", type=int, default=0, help="最大采集帧数，0=无限制")
    parser.add_argument("--base-dir", type=str, default="collect_data", help="输出根目录")
    args = parser.parse_args()

    capture(max_frames=args.max_frames, base_dir=args.base_dir)
