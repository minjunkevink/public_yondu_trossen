#!/usr/bin/env python3

import pandas as pd
import pyarrow.parquet as pq
import json
import os
from pathlib import Path

# Path to the parquet file
parquet_path = "/home/yondu/.cache/huggingface/lerobot/Aravindh25/test_depth_4/data/chunk-000/episode_000000.parquet"

# Read the parquet file
table = pq.read_table(parquet_path)

# Extract path to a depth image from the first row
depth_column = "observation.depth.cam_wrist"
depth_data = table[depth_column].to_pandas()

# Print the first few depth paths
print("First 5 depth image paths:")
for i in range(min(5, len(depth_data))):
    path_dict = depth_data.iloc[i]
    path = path_dict.get('path')
    print(f"{i}: {path}")
    
    # Check if the file exists
    if path and os.path.exists(path):
        print(f"  File exists: {os.path.getsize(path)} bytes")
        
        # For the first file, print its parent directory contents
        if i == 0:
            parent_dir = os.path.dirname(path)
            print(f"\nContents of {parent_dir}:")
            try:
                for item in os.listdir(parent_dir):
                    item_path = os.path.join(parent_dir, item)
                    if os.path.isfile(item_path):
                        print(f"  {item}: {os.path.getsize(item_path)} bytes")
                    else:
                        print(f"  {item}/ (directory)")
            except Exception as e:
                print(f"  Error listing directory: {e}")
    else:
        print(f"  File does NOT exist")

# Try to find all png files in the dataset directory
print("\nSearching for all PNG files in the dataset directory:")
dataset_dir = "/home/yondu/.cache/huggingface/lerobot/Aravindh25/test_depth_4"
png_files = []

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".png"):
            full_path = os.path.join(root, file)
            png_files.append(full_path)

if png_files:
    print(f"Found {len(png_files)} PNG files:")
    for png in png_files[:10]:  # Limit to first 10 for brevity
        print(f"  {png}: {os.path.getsize(png)} bytes")
    if len(png_files) > 10:
        print(f"  ... and {len(png_files) - 10} more")
else:
    print("No PNG files found in the dataset directory") 