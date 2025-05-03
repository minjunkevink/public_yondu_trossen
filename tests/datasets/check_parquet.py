#!/usr/bin/env python3

import pandas as pd
import pyarrow.parquet as pq
import json

# Path to the parquet file
parquet_path = "/home/yondu/.cache/huggingface/lerobot/Aravindh25/test_depth_4/data/chunk-000/episode_000000.parquet"

# Read the parquet file
table = pq.read_table(parquet_path)
print("Columns in the parquet file:")
print(table.column_names)

# Check for depth-related columns
depth_columns = [col for col in table.column_names if "depth" in col]
print("\nDepth-related columns:", depth_columns)

# If there are depth columns, display some sample data
if depth_columns:
    for col in depth_columns:
        print(f"\nSample data from {col}:")
        sample_data = table[col].to_pandas().head()
        print(sample_data)

# Read the info.json file to understand the structure
info_path = "/home/yondu/.cache/huggingface/lerobot/Aravindh25/test_depth_4/meta/info.json"
try:
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    print("\nDepth features from info.json:")
    for feature_name, feature_info in info.get("features", {}).items():
        if "depth" in feature_name:
            print(f"\n{feature_name}:")
            print(json.dumps(feature_info, indent=2))
except Exception as e:
    print(f"Error reading info.json: {e}") 