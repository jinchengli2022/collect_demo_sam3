"""
Pipeline: 串联 Step 2~4（Step 1 需要实际摄像头，单独运行）
用法:
  python pipeline.py <session_dir> --prompts "door handle" "person" [options]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from collect_scripts.video_sam3 import run_sam3_multi_prompt
from collect_scripts.depth_extract import extract_trajectories
from collect_scripts.visualize_trajectory import visualize


def run_pipeline(
    session_dir: Path,
    prompts: list[str],
    frame_index: int = 0,
    stat_method: Literal["median", "mean"] = "median",
    smooth_window: int = 1,
    debug_vis: bool = False,
) -> None:
    print("=" * 60)
    print(f"[Pipeline] session: {session_dir}")
    print(f"[Pipeline] prompts: {prompts}")
    print("=" * 60)

    print("\n--- Step 2: SAM3 分割 ---")
    run_sam3_multi_prompt(
        session_dir=session_dir,
        prompts=prompts,
        frame_index=frame_index,
        debug_vis=debug_vis,
    )

    print("\n--- Step 3: 深度抠像 + 三维坐标 ---")
    extract_trajectories(
        session_dir=session_dir,
        prompts=prompts,
        stat_method=stat_method,
        debug_vis=debug_vis,
    )

    print("\n--- Step 4: 轨迹可视化 ---")
    visualize(session_dir=session_dir, smooth_window=smooth_window)

    print("\n[Pipeline] 全部完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L515+SAM3 完整管线（Step 2~4）")
    parser.add_argument("session_dir", type=str, help="Step 1 采集的 session 目录")
    parser.add_argument("--prompts", nargs="+", required=True, help="文本 prompt 列表")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--stat-method", choices=["median", "mean"], default="median")
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--debug-vis", action="store_true", help="生成所有 vis/ 可视化图")
    args = parser.parse_args()

    run_pipeline(
        session_dir=Path(args.session_dir),
        prompts=args.prompts,
        frame_index=args.frame_index,
        stat_method=args.stat_method,
        smooth_window=args.smooth_window,
        debug_vis=args.debug_vis,
    )
