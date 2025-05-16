#!/bin/bash

# record_robot.sh - Script to record Trossen AI Solo robot episodes

# Check if HF_USER environment variable is set
if [ -z "$HF_USER" ]; then
    echo "Error: HF_USER environment variable is not set"
    echo "Please set it with: export HF_USER=your_huggingface_username"
    exit 1
fi

echo "Starting recording with HF_USER: $HF_USER"
echo "Running evaluation ..."

# Run the recording command
python lerobot/scripts/control_robot.py \
  --robot.type=trossen_ai_solo \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Put the granolas in the basket" \
  --control.repo_id=${HF_USER}/eval_trossen_granola_200A_100B\
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=60 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=true \
  --control.policy.path=outputs/train/trossen_pick_granola/trossen_pick_granola_bars_3cam_200A_100B/200000/pretrained_model \
  --control.num_image_writer_processes=1 \
  --control.is_policy_in_radians=true
  


echo "Recording completed!"