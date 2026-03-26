"""
Step 2: SAM3 多 Prompt 分割
输入: session_dir/color/ 帧目录 + prompts 列表
输出: session_dir/masks/{slug}/frame_XXXX.npy  (bool H×W)
      session_dir/vis/mask/{slug}/frame_XXXX.png  (debug, --debug-vis)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from huggingface_hub import login

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
_HF_TOKEN = os.environ.get("HF_TOKEN", "")
if _HF_TOKEN:
    try:
        login(token=_HF_TOKEN)
    except Exception:
        pass

from sam3.model_builder import build_sam3_video_predictor


def _prompt_to_slug(prompt: str) -> str:
    return prompt.strip().replace(" ", "_").lower()


def _load_color_frames(color_dir: Path) -> list[np.ndarray]:
    paths = sorted(color_dir.glob("frame_*.png"))
    frames: list[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            frames.append(img)
    return frames


def _load_depth_frame(depth_dir: Path, frame_idx: int) -> Optional[np.ndarray]:
    p = depth_dir / f"frame_{frame_idx:04d}.npy"
    if p.exists():
        arr: np.ndarray = np.load(str(p))
        return arr
    return None


def _make_depth_colormap(depth_mm: np.ndarray, mask: np.ndarray) -> np.ndarray:
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
    return vis


def _save_vis_mask(
    session_dir: Path,
    prompt: str,
    frame_idx: int,
    color_frame: np.ndarray,
    mask: np.ndarray,
    depth_dir: Optional[Path],
) -> None:
    slug = _prompt_to_slug(prompt)
    out_dir = session_dir / "vis" / "mask" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    h, w = color_frame.shape[:2]

    overlay = color_frame.copy()
    if mask.any():
        green = np.zeros_like(overlay)
        green[mask] = [0, 255, 0]
        overlay = cv2.addWeighted(overlay, 0.7, green, 0.3, 0)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    panels: list[np.ndarray] = [overlay]
    if depth_dir is not None:
        depth_mm = _load_depth_frame(depth_dir, frame_idx)
        if depth_mm is not None:
            panels.append(_make_depth_colormap(depth_mm, mask))

    if len(panels) == 2:
        combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
        combined[:, :w] = panels[0]
        combined[:, w:] = panels[1]
    else:
        combined = panels[0]

    cv2.imwrite(str(out_dir / f"frame_{frame_idx:04d}.png"), combined)


def run_sam3_multi_prompt(
    session_dir: Path,
    prompts: list[str],
    frame_index: int = 0,
    debug_vis: bool = False,
) -> dict[str, dict[int, np.ndarray]]:
    """
    对 session_dir/color/ 中的帧序列，针对每个 prompt 独立运行 SAM3 propagation。
    返回 {prompt: {frame_idx: merged_mask (H, W, bool)}}
    同时将 mask 保存到 session_dir/masks/{slug}/frame_XXXX.npy
    """
    color_dir = session_dir / "color"
    depth_dir: Optional[Path] = session_dir / "depth" if (session_dir / "depth").exists() else None

    color_frames = _load_color_frames(color_dir)
    if not color_frames:
        raise FileNotFoundError(f"color 目录为空或不存在: {color_dir}")

    total = len(color_frames)
    h, w = color_frames[0].shape[:2]
    print(f"[Step2] 共 {total} 帧，{len(prompts)} 个 prompt")

    video_predictor = build_sam3_video_predictor()
    result: dict[str, dict[int, np.ndarray]] = {}

    for prompt in prompts:
        slug = _prompt_to_slug(prompt)
        mask_dir = session_dir / "masks" / slug
        mask_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Step2] 处理 prompt: '{prompt}'")

        response = video_predictor.handle_request(
            request=dict(type="start_session", resource_path=str(color_dir))
        )
        session_id: str = response["session_id"]

        video_predictor.handle_request(
            request=dict(type="reset_session", session_id=session_id)
        )

        actual_fi = min(frame_index, total - 1)
        video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=actual_fi,
                text=prompt,
            )
        )

        prompt_masks: dict[int, np.ndarray] = {}

        for resp in video_predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id)
        ):
            fidx: int = resp["frame_index"]
            outputs: dict[str, np.ndarray] = resp["outputs"]
            binary_masks: np.ndarray = outputs.get(
                "out_binary_masks", np.zeros((0,), dtype=bool)
            )
            if binary_masks.ndim == 3 and binary_masks.shape[0] > 0:
                merged: np.ndarray = np.any(binary_masks, axis=0)
            else:
                merged = np.zeros((h, w), dtype=bool)

            prompt_masks[fidx] = merged
            np.save(str(mask_dir / f"frame_{fidx:04d}.npy"), merged)

            if debug_vis and fidx < total:
                _save_vis_mask(
                    session_dir, prompt, fidx,
                    color_frames[fidx], merged, depth_dir,
                )

        video_predictor.handle_request(
            request=dict(type="close_session", session_id=session_id)
        )
        result[prompt] = prompt_masks
        print(f"[Step2] '{prompt}' 完成，mask 已保存至 {mask_dir}")

    video_predictor.shutdown()
    print("[Step2] 所有 prompt 分割完成")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM3 多 Prompt 分割")
    parser.add_argument("session_dir", type=str, help="session 目录路径")
    parser.add_argument("--prompts", nargs="+", required=True, help="文本 prompt 列表")
    parser.add_argument("--frame-index", type=int, default=0, help="在哪一帧添加 prompt")
    parser.add_argument("--debug-vis", action="store_true", help="生成 vis/mask/ 可视化图")
    args = parser.parse_args()

    run_sam3_multi_prompt(
        session_dir=Path(args.session_dir),
        prompts=args.prompts,
        frame_index=args.frame_index,
        debug_vis=args.debug_vis,
    )