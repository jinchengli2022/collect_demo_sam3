"""
Step 4: 轨迹可视化
输入: session_dir/trajectories_3d.json
输出: session_dir/vis/trajectory/trajectory_3d.png
      session_dir/vis/trajectory/trajectory_components.png
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3d projection


def _load_trajectories(
    session_dir: Path,
) -> tuple[dict[str, list[Optional[tuple[float, float, float]]]], dict[str, object]]:
    with open(session_dir / "trajectories_3d.json") as f:
        data: dict[str, object] = json.load(f)

    meta: dict[str, object] = data["metadata"]  # type: ignore[index]
    raw: dict[str, list[object]] = data["trajectories"]  # type: ignore[index]

    trajectories: dict[str, list[Optional[tuple[float, float, float]]]] = {}
    for prompt, frames in raw.items():
        parsed: list[Optional[tuple[float, float, float]]] = []
        for pt in frames:
            if pt is None:
                parsed.append(None)
            else:
                coords = list(pt)  # type: ignore[arg-type]
                parsed.append((float(coords[0]), float(coords[1]), float(coords[2])))
        trajectories[prompt] = parsed

    return trajectories, meta


def _to_nan_arrays(
    traj: list[Optional[tuple[float, float, float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.array([p[0] if p is not None else math.nan for p in traj])
    ys = np.array([p[1] if p is not None else math.nan for p in traj])
    zs = np.array([p[2] if p is not None else math.nan for p in traj])
    return xs, ys, zs


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    out = arr.copy()
    half = window // 2
    for i in range(len(arr)):
        if math.isnan(arr[i]):
            continue
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        segment = arr[lo:hi]
        valid = segment[~np.isnan(segment)]
        if valid.size > 0:
            out[i] = float(np.mean(valid))
    return out


def plot_3d(
    session_dir: Path,
    trajectories: dict[str, list[Optional[tuple[float, float, float]]]],
    smooth_window: int = 1,
) -> None:
    out_dir = session_dir / "vis" / "trajectory"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (prompt, traj) in enumerate(trajectories.items()):
        xs, ys, zs = _to_nan_arrays(traj)
        ys = -ys
        if smooth_window > 1:
            xs, ys, zs = _smooth(xs, smooth_window), _smooth(ys, smooth_window), _smooth(zs, smooth_window)

        color = colors[idx % len(colors)]
        ax.plot(xs, zs, ys, label=prompt, color=color, linewidth=1.5)
        valid = [(x, y, z) for x, y, z in zip(xs, ys, zs) if not math.isnan(x)]
        if valid:
            ax.scatter([valid[0][0]], [valid[0][2]], [valid[0][1]], color=color, s=30, zorder=5)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.set_zlabel("Y (mm)")  # type: ignore[attr-defined]
    ax.set_title("3D Trajectory (Camera Coordinate System)")
    ax.view_init(elev=20, azim=-60)
    ax.legend()

    out_path = out_dir / "trajectory_3d.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Step4] 3D 轨迹图已保存: {out_path}")


def plot_components(
    session_dir: Path,
    trajectories: dict[str, list[Optional[tuple[float, float, float]]]],
    fps: float,
    smooth_window: int = 1,
) -> None:
    out_dir = session_dir / "vis" / "trajectory"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = max(len(t) for t in trajectories.values()) if trajectories else 0
    time_axis = np.arange(n) / fps if fps > 0 else np.arange(n)
    x_label = "Time (s)" if fps > 0 else "Frame index"

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    component_labels = ["X (mm)", "Y (mm)", "Z (mm)"]

    for idx, (prompt, traj) in enumerate(trajectories.items()):
        xs, ys, zs = _to_nan_arrays(traj)
        ys = -ys
        if smooth_window > 1:
            xs = _smooth(xs, smooth_window)
            ys = _smooth(ys, smooth_window)
            zs = _smooth(zs, smooth_window)

        color = colors[idx % len(colors)]
        t = time_axis[: len(traj)]
        for ax, arr in zip(axes, [xs, ys, zs]):
            ax.plot(t, arr, label=prompt, color=color, linewidth=1.2)

    for ax, ylabel in zip(axes, component_labels):
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(x_label)
    fig.suptitle("Trajectory Components Over Time")
    fig.tight_layout()

    out_path = out_dir / "trajectory_components.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Step4] 分量图已保存: {out_path}")


def visualize(session_dir: Path, smooth_window: int = 1) -> None:
    trajectories, meta = _load_trajectories(session_dir)
    fps = float(meta.get("fps", 30))  # type: ignore[arg-type]
    plot_3d(session_dir, trajectories, smooth_window)
    plot_components(session_dir, trajectories, fps, smooth_window)
    print("[Step4] 可视化完成")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="轨迹可视化")
    parser.add_argument("session_dir", type=str, help="session 目录路径")
    parser.add_argument("--smooth-window", type=int, default=1, help="平滑窗口大小（1=不平滑）")
    args = parser.parse_args()

    visualize(
        session_dir=Path(args.session_dir),
        smooth_window=args.smooth_window,
    )
