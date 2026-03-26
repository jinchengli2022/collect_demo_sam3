"""
Step 3: 深度抠像 + 三维坐标计算
输入: session_dir/masks/ + session_dir/depth/ + session_dir/intrinsics.json
输出: session_dir/trajectories_3d.json
      session_dir/vis/depth/{slug}/frame_XXXX.png  (debug, --debug-vis)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np

MAX_DEPTH_MM: int = 5000


def _load_intrinsics(session_dir: Path) -> dict[str, float]:
    with open(session_dir / "intrinsics.json") as f:
        data: dict[str, float] = json.load(f)
    return data


def _load_mask(mask_path: Path) -> np.ndarray:
    arr: np.ndarray = np.load(str(mask_path))
    return arr.astype(bool)


def _load_depth(depth_path: Path) -> np.ndarray:
    arr: np.ndarray = np.load(str(depth_path))
    return arr.astype(np.uint16)


def _compute_frame_point(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stat_method: Literal["median", "mean"],
) -> tuple[float, float, float, int, int]:
    """
    返回 (X, Y, Z, u_rep, v_rep)。
    X/Y/Z 单位 mm，相机坐标系。
    u_rep/v_rep 是代表点像素坐标（用于 vis 标注）。
    若无有效像素返回 (nan, nan, nan, 0, 0)。
    """
    valid_mask: np.ndarray = mask & (depth_mm > 0) & (depth_mm < MAX_DEPTH_MM)
    ys, xs = np.where(valid_mask)
    valid_depths: np.ndarray = depth_mm[valid_mask].astype(np.float64)

    if valid_depths.size == 0:
        return (math.nan, math.nan, math.nan, 0, 0)

    if stat_method == "median":
        z_val = float(np.median(valid_depths))
    else:
        z_val = float(np.mean(valid_depths))

    idx = int(np.argmin(np.abs(valid_depths - z_val)))
    u_rep = int(xs[idx])
    v_rep = int(ys[idx])

    x_val = (u_rep - cx) * z_val / fx
    y_val = (v_rep - cy) * z_val / fy

    return (x_val, y_val, z_val, u_rep, v_rep)


def _save_vis_depth(
    session_dir: Path,
    prompt: str,
    frame_idx: int,
    depth_mm: np.ndarray,
    mask: np.ndarray,
    u_rep: int,
    v_rep: int,
    x_val: float,
    y_val: float,
    z_val: float,
) -> None:
    slug = prompt.strip().replace(" ", "_").lower()
    out_dir = session_dir / "vis" / "depth" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    vis = np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
    region = depth_mm.astype(np.float32)
    region[~mask] = 0.0
    valid = region[mask]
    if valid.size > 0:
        dmin, dmax = float(valid.min()), float(valid.max())
        span = dmax - dmin if dmax > dmin else 1.0
        norm = ((region - dmin) / span * 255.0).clip(0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        vis[mask] = colored[mask]

    gray_bg = np.full_like(vis, 80)
    gray_bg[mask] = vis[mask]

    cv2.circle(gray_bg, (u_rep, v_rep), 6, (0, 255, 0), -1)
    label = f"Z={z_val:.0f}mm X={x_val:.0f}mm Y={y_val:.0f}mm"
    cv2.putText(gray_bg, label, (u_rep + 8, v_rep - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_dir / f"frame_{frame_idx:04d}.png"), gray_bg)


def extract_trajectories(
    session_dir: Path,
    prompts: list[str],
    stat_method: Literal["median", "mean"] = "median",
    debug_vis: bool = False,
) -> dict[str, list[Optional[tuple[float, float, float]]]]:
    """
    对每个 prompt 逐帧提取三维坐标。
    返回 {prompt: [  (X,Y,Z) or None, ...  ]}，列表长度=总帧数。
    同时保存 trajectories_3d.json。
    """
    intr = _load_intrinsics(session_dir)
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])

    with open(session_dir / "metadata.json") as f:
        meta: dict[str, object] = json.load(f)
    fps = float(meta["fps"])
    total_frames = int(meta["total_frames"])

    trajectories: dict[str, list[Optional[tuple[float, float, float]]]] = {}

    for prompt in prompts:
        slug = prompt.strip().replace(" ", "_").lower()
        mask_dir = session_dir / "masks" / slug
        depth_dir = session_dir / "depth"

        if not mask_dir.exists():
            raise FileNotFoundError(f"mask 目录不存在: {mask_dir}")
        if not depth_dir.exists():
            raise FileNotFoundError(f"depth 目录不存在: {depth_dir}")

        print(f"[Step3] 处理 prompt: '{prompt}'")
        traj: list[Optional[tuple[float, float, float]]] = []

        for fidx in range(total_frames):
            mask_path = mask_dir / f"frame_{fidx:04d}.npy"
            depth_path = depth_dir / f"frame_{fidx:04d}.npy"

            if not mask_path.exists() or not depth_path.exists():
                traj.append(None)
                continue

            mask = _load_mask(mask_path)
            depth_mm = _load_depth(depth_path)

            x_val, y_val, z_val, u_rep, v_rep = _compute_frame_point(
                mask, depth_mm, fx, fy, cx, cy, stat_method
            )

            if math.isnan(z_val):
                traj.append(None)
            else:
                traj.append((x_val, y_val, z_val))
                if debug_vis:
                    _save_vis_depth(
                        session_dir, prompt, fidx,
                        depth_mm, mask, u_rep, v_rep,
                        x_val, y_val, z_val,
                    )

        trajectories[prompt] = traj
        valid_count = sum(1 for p in traj if p is not None)
        print(f"[Step3] '{prompt}' 完成，有效帧 {valid_count}/{total_frames}")

    _save_json(session_dir, trajectories, fx, fy, cx, cy, fps, total_frames, stat_method, prompts)
    return trajectories


def _find_origin(
    trajectories: dict[str, list[Optional[tuple[float, float, float]]]],
    ref_prompt: str,
) -> tuple[float, float, float]:
    """返回 ref_prompt 轨迹中第一个有效帧的坐标，作为平移原点。"""
    for pt in trajectories[ref_prompt]:
        if pt is not None:
            return pt
    return (0.0, 0.0, 0.0)


def _save_json(
    session_dir: Path,
    trajectories: dict[str, list[Optional[tuple[float, float, float]]]],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    fps: float,
    total_frames: int,
    stat_method: str,
    prompts: list[str],
) -> None:
    ref_prompt = prompts[0]
    ox, oy, oz = _find_origin(trajectories, ref_prompt)
    print(f"[Step3] 坐标原点平移至 '{ref_prompt}' 起始帧: X={ox:.1f} Y={oy:.1f} Z={oz:.1f} mm")

    json_traj: dict[str, list[object]] = {}
    for prompt, traj in trajectories.items():
        json_traj[prompt] = [
            [round(p[0] - ox, 2), round(p[1] - oy, 2), round(p[2] - oz, 2)] if p is not None else None
            for p in traj
        ]

    output = {
        "metadata": {
            "session": session_dir.name,
            "fps": fps,
            "total_frames": total_frames,
            "coord_unit": "mm",
            "coord_system": "object_relative",
            "origin_prompt": ref_prompt,
            "origin_camera_mm": [round(ox, 2), round(oy, 2), round(oz, 2)],
            "stat_method": stat_method,
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        },
        "trajectories": json_traj,
    }
    out_path = session_dir / "trajectories_3d.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[Step3] 三维轨迹已保存: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="深度抠像 + 三维坐标计算")
    parser.add_argument("session_dir", type=str, help="session 目录路径")
    parser.add_argument("--prompts", nargs="+", required=True, help="文本 prompt 列表")
    parser.add_argument("--stat-method", choices=["median", "mean"], default="median")
    parser.add_argument("--debug-vis", action="store_true", help="生成 vis/depth/ 可视化图")
    args = parser.parse_args()

    extract_trajectories(
        session_dir=Path(args.session_dir),
        prompts=args.prompts,
        stat_method=args.stat_method,
        debug_vis=args.debug_vis,
    )
