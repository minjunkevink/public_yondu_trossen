#!/bin/bash

# vlm_record.sh - Script to record Trossen AI Solo robot episodes with VLM integration

# Check if HF_USER environment variable is set
if [ -z "$HF_USER" ]; then
    echo "Error: HF_USER environment variable is not set"
    echo "Please set it with: export HF_USER=your_huggingface_username"
    exit 1
fi

# Check if VLM_API_KEY is set
if [ -z "$VLM_API_KEY" ]; then
    echo "Error: VLM_API_KEY environment variable is not set"
    echo "Please set it with: export VLM_API_KEY=your_vlm_api_key"
    exit 1
fi

echo "Starting recording with HF_USER: $HF_USER"
echo "Recording with VLM integration..."

# Run the recording command with VLM integration
python lerobot/scripts/control_robot.py \
--robot.type=trossen_ai_solo \
--robot.max_relative_target=null \
--control.type=record \
--control.fps=30 \
--control.single_task="Put the orange in the basket" \
--control.repo_id=${HF_USER}/test_depth_2 \
--control.tags='["tutorial", "vlm_integrated"]' \
--control.warmup_time_s=3 \
--control.episode_time_s=3 \
--control.reset_time_s=3 \
--control.num_episodes=2 \
--control.push_to_hub=false \
--control.private=true \
--control.display_data=true \
--control.vlm.enabled=true \
--control.vlm.api_url="https://f5d72e224a29e2fc4b.gradio.live/" \
--control.vlm.api_key="${VLM_API_KEY}"

echo "Recording completed!" 