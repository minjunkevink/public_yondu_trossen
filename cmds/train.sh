#!/bin/bash

# Get the current timestamp for unique log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="training_${TIMESTAMP}.log"

# Default values
DATASET_REPO_ID="null"
POLICY_TYPE="act"
OUTPUT_DIR="outputs/train/trossen_pick_tshirt_3cam_v2m5"
JOB_NAME="trossen_pick_tshirt_3cam_v2m5"
DEVICE="cuda"
WANDB_ENABLE="false"
DATASET_ROOT="./datasets/trossen_pick_tshirt_3cam_v2m5"
STEPS="200000"
BATCH_SIZE="16"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset.repo_id=*)
      DATASET_REPO_ID="${1#*=}"
      shift
      ;;
    --policy.type=*)
      POLICY_TYPE="${1#*=}"
      shift
      ;;
    --output_dir=*)
      OUTPUT_DIR="${1#*=}"
      shift
      ;;
    --job_name=*)
      JOB_NAME="${1#*=}"
      shift
      ;;
    --policy.device=*)
      DEVICE="${1#*=}"
      shift
      ;;
    --wandb.enable=*)
      WANDB_ENABLE="${1#*=}"
      shift
      ;;
    --dataset.root=*)
      DATASET_ROOT="${1#*=}"
      shift
      ;;
    --steps=*)
      STEPS="${1#*=}"
      shift
      ;;
    --batch_size=*)
      BATCH_SIZE="${1#*=}"
      shift
      ;;
    --log_file=*)
      LOG_FILE="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Create logs directory if it doesn't exist
mkdir -p logs

# Construct the full command
TRAIN_CMD="python lerobot/scripts/train.py \
  --dataset.repo_id=\"${DATASET_REPO_ID}\" \
  --policy.type=${POLICY_TYPE} \
  --output_dir=${OUTPUT_DIR} \
  --job_name=${JOB_NAME} \
  --policy.device=${DEVICE} \
  --wandb.enable=${WANDB_ENABLE} \
  --dataset.root=${DATASET_ROOT} \
  --steps=${STEPS} \
  --batch_size=${BATCH_SIZE}"

# Print the command that will be executed
echo "Executing command:"
echo "$TRAIN_CMD"
echo "Logging output to logs/${LOG_FILE}"

# Run the command with nohup
nohup $TRAIN_CMD > "logs/${LOG_FILE}" 2>&1 &

# Get the process ID
PID=$!
echo "Started training process with PID: $PID"
echo "To monitor the logs, run: tail -f logs/${LOG_FILE}"
echo "To stop the process, run: kill $PID"

# Save the PID to a file for later reference
echo $PID > "logs/${JOB_NAME}.pid"
