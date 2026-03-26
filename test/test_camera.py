"""
相机测试：同时显示对齐后的 RGB 图和深度伪彩色图
按 q 退出，按 s 保存当前帧到 test/snapshots/
"""
from __future__ import annotations

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
MAX_DEPTH_MM = 5000


def depth_to_colormap(depth_mm: np.ndarray) -> np.ndarray:
    clipped = np.clip(depth_mm, 0, MAX_DEPTH_MM).astype(np.float32)
    norm = (clipped / MAX_DEPTH_MM * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def run() -> None:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale: float = depth_sensor.get_depth_scale()

    snapshot_dir = Path(__file__).parent / "snapshots"

    print("相机已启动，按 q 退出，按 s 保存当前帧")

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_img: np.ndarray = np.asanyarray(color_frame.get_data())
            depth_raw: np.ndarray = np.asanyarray(depth_frame.get_data())
            depth_mm = (depth_raw.astype(np.float32) * depth_scale * 1000.0).astype(np.uint16)

            depth_vis = depth_to_colormap(depth_mm)

            h, w = color_img.shape[:2]
            combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
            combined[:, :w] = color_img
            combined[:, w:] = depth_vis

            cv2.putText(combined, "RGB (aligned)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined, f"Depth (0~{MAX_DEPTH_MM}mm)", (w + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("L515 Test  |  left: RGB  right: Depth  |  q=quit  s=save", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(str(snapshot_dir / f"{ts}_color.png"), color_img)
                np.save(str(snapshot_dir / f"{ts}_depth.npy"), depth_mm)
                cv2.imwrite(str(snapshot_dir / f"{ts}_depth_vis.png"), depth_vis)
                print(f"已保存: {snapshot_dir}/{ts}_*")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("相机已关闭")


if __name__ == "__main__":
    run()
