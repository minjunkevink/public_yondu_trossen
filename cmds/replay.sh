#!/bin/bash

# replay_robot.sh - Script to replay Trossen AI Solo robot episodes

# Check if HF_USER environment variable is set
if [ -z "$HF_USER" ]; then
    echo "Error: HF_USER environment variable is not set"
    echo "Please set it with: export HF_USER=your_huggingface_username"
    exit 1
fi

echo "Starting replay with HF_USER: $HF_USER"
echo "Repository: $REPO_ID"
echo "Episode: $EPISODE"
if [ -n "$ROOT" ]; then
    echo "Local dataset path: $ROOT"
fi
echo "Replaying..."

# Build the command
CMD="python lerobot/scripts/control_robot.py \
--robot.type=trossen_ai_solo \
--robot.max_relative_target=null \
--control.type=replay \
--control.fps=30 \
--control.repo_id=trossen_pick_granola_bars_3cam_v1 \
--control.episode=0 \
--control.root=datasets/trossen_pick_granola_bars_3cam/trossen_pick_granola_bars_3cam_v1 "

# Execute the command
eval $CMD

echo "Replay completed!" 