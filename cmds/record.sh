#!/bin/bash

# record_robot.sh - Script to record Trossen AI Solo robot episodes

# Check if HF_USER environment variable is set
if [ -z "$HF_USER" ]; then
    echo "Error: HF_USER environment variable is not set"
    echo "Please set it with: export HF_USER=your_huggingface_username"
    exit 1
fi

echo "Starting recording with HF_USER: $HF_USER"
echo "Recording 2 episodes of 30 seconds each..."

# Run the recording command
python lerobot/scripts/control_robot.py \
--robot.type=trossen_ai_solo \
--robot.max_relative_target=null \
--control.type=record \
--control.fps=30 \
--control.single_task="Put the orange in the basket" \
--control.repo_id=${HF_USER}/trossen_put_orange_dataset_3_1 \
--control.tags='["tutorial"]' \
--control.warmup_time_s=10 \
--control.episode_time_s=8 \
--control.reset_time_s=30 \
--control.num_episodes=50 \
--control.push_to_hub=true \
--control.display_cameras=false

echo "Recording completed!"