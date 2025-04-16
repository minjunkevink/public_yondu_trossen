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
  --control.single_task="Put the orange in the basket" \
  --control.repo_id=${HF_USER}/eval_put_orange_run_9_v4\
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=120 \
  --control.reset_time_s=30 \
  --control.num_episodes=5 \
  --control.push_to_hub=true \
  --control.policy.path=outputs/train/trossen_put_orange_run_9/200000/pretrained_model \
  --control.num_image_writer_processes=1 \
  --control.start_pos="[-14, 59, 64, -47, 7.9, -4.5, 378]" \
  --control.display_cameras=false \
  


echo "Recording completed!"