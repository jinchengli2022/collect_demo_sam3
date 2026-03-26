#!/usr/bin/env bash
set -euo pipefail

SESSION_DIR="collect_data/mouse"
PROMPTS=("black mouse")     # 操作物体在前，参考物体在后

# 可按需取消注释
# FRAME_INDEX="--frame-index 0"
# STAT_METHOD="--stat-method median"
# SMOOTH_WINDOW="--smooth-window 7"
# DEBUG_VIS="--debug-vis"

python collect_scripts/video_sam3.py "$SESSION_DIR" \
    --prompts "${PROMPTS[@]}" \
    --frame-index 0 \
    --debug-vis

python collect_scripts/depth_extract.py "$SESSION_DIR" \
    --prompts "${PROMPTS[@]}" \
    --stat-method median \
    --debug-vis

python collect_scripts/visualize_trajectory.py "$SESSION_DIR" \
    --smooth-window 10
