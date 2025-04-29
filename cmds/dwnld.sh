#!/bin/bash

# download_datasets.sh - Script to download multiple versioned datasets

# Set the base directory where all datasets will be downloaded
BASE_DIR="./datasets"

# Create the base directory if it doesn't exist
mkdir -p $BASE_DIR

# Loop through all versions (0-25)
for i in {0..35}; do
  echo "=== Downloading dataset: Aravindh25/trossen_pick_tshirt_v$i ==="
  
  # Create a directory for this version
  DATASET_DIR="$BASE_DIR/trossen_pick_tshirt_v$i"
  mkdir -p $DATASET_DIR
  
  # Download the dataset
  huggingface-cli download \
    --repo-type dataset \
    Aravindh25/trossen_pick_tshirt_v$i \
    --local-dir $DATASET_DIR
    
  echo "=== Completed downloading v$i ==="
  echo ""
done

echo "All datasets downloaded successfully!"