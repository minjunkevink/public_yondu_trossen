#!/usr/bin/env python3
"""
Dataset conversion script for local datasets.
This script converts a LeRobot v2.1 dataset to LeRobot v2.1 Trossen v1.0 format:
- Converting joint angles from degrees to radians
- Reverting the scaling of gripper values to millimeters
- Updating the dataset metadata

Works with local datasets without requiring the LeRobot framework.

Usage:
python local_dataset_converter.py --dataset_path /path/to/your/dataset

Options:
--dataset_path: Path to the dataset directory
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# This scaling factor was used to prevent the gripper values from vanishing
# to zero in the dataset when converted to integer.
SCALING_FACTOR = 10000

def transform_arm_data(array):
    """
    Transforms arm data by converting joint angles to radians and scaling the gripper values.
    
    For the typical 7-DOF Trossen arm:
    - Indices 0-5 are joint angles (convert to radians)
    - Index 6 is the gripper value (divide by scaling factor)
    """
    if len(array) != 7:  # Basic validation for a standard Trossen arm
        print(f"Warning: Expected 7 values for arm data, got {len(array)}. Check if this is correct.")
    
    # Create a copy to avoid modifying the original
    result = np.copy(array)
    
    # Convert joint angles to radians (all except the gripper)
    result[:-1] = np.deg2rad(result[:-1])
    
    # Revert the scaling of the gripper
    result[-1] /= SCALING_FACTOR
    
    return result

def compute_stats(data_entries):
    """Compute statistics for a collection of data entries."""
    data = np.array(data_entries)
    return {
        "min": data.min(axis=0).tolist(),
        "max": data.max(axis=0).tolist(),
        "mean": data.mean(axis=0).tolist(),
        "std": data.std(axis=0).tolist(),
        "count": [len(data)],
    }

def update_info_json(info_file_path, new_subversion="v1.0"):
    """Update the info.json file with the new subversion."""
    try:
        with open(info_file_path, 'r') as f:
            info_data = json.load(f)
        
        current_version = info_data.get("codebase_version", "unknown")
        print(f"Current codebase version: {current_version}")
        
        # Create a new dict to maintain order with trossen_subversion after codebase_version
        updated_info_data = {}
        for key, value in info_data.items():
            updated_info_data[key] = value
            if key == "codebase_version":
                updated_info_data["trossen_subversion"] = new_subversion
        
        # If codebase_version wasn't found, add trossen_subversion at the beginning
        if "trossen_subversion" not in updated_info_data:
            updated_info_data = {"trossen_subversion": new_subversion, **info_data}
        
        with open(info_file_path, 'w') as f:
            json.dump(updated_info_data, f, indent=4)
        
        print(f"Updated trossen_subversion to: {new_subversion}")
        return True
    except Exception as e:
        print(f"Error updating info.json: {e}")
        return False

def convert_dataset(dataset_path):
    """Convert a local dataset to the new format."""
    dataset_path = Path(dataset_path)
    
    # Validate the dataset path
    if not dataset_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist")
        return False
    
    # Check for required directories and files
    meta_dir = dataset_path / "meta"
    data_dir = dataset_path / "data" / "chunk-000"
    info_file = meta_dir / "info.json"
    episodes_stats_file = meta_dir / "episodes_stats.jsonl"
    
    if not meta_dir.exists() or not data_dir.exists():
        print(f"Error: Required directories not found in {dataset_path}")
        return False
    
    if not info_file.exists() or not episodes_stats_file.exists():
        print(f"Error: Required metadata files not found in {meta_dir}")
        return False
    
    # Load the dataset info
    with open(info_file, 'r') as f:
        info_data = json.load(f)
    
    # Check if conversion has already been done
    if "trossen_subversion" in info_data:
        print(f"Dataset already has trossen_subversion: {info_data['trossen_subversion']}")
        yn = input("Do you want to proceed with conversion anyway? (y/n): ")
        if yn.lower() != 'y':
            return False
    
    # Get total episodes
    total_episodes = info_data.get("total_episodes", 0)
    print(f"Found {total_episodes} episodes in the dataset")
    
    # Load the episodes stats
    episodes_stats = []
    with open(episodes_stats_file, 'r') as f:
        for line in f:
            episodes_stats.append(json.loads(line))
    
    # Process each episode
    for episode_idx in range(total_episodes):
        print(f"Processing episode {episode_idx}/{total_episodes}...")
        episode_file = data_dir / f"episode_{episode_idx:06d}.parquet"
        
        if not episode_file.exists():
            print(f"Warning: Episode file {episode_file} not found, skipping")
            continue
        
        try:
            # Read the parquet file
            df = pd.read_parquet(episode_file)
            
            # Check if the expected columns exist
            if 'action' not in df.columns or 'observation.state' not in df.columns:
                print(f"Warning: Required columns not found in {episode_file}, skipping")
                continue
            
            # Apply transformations
            modified_actions = []
            modified_states = []
            
            # Process each row
            for i, row in df.iterrows():
                # Transform action and state
                action = np.array(row['action'])
                state = np.array(row['observation.state'])
                
                mod_action = transform_arm_data(action)
                mod_state = transform_arm_data(state)
                
                modified_actions.append(mod_action)
                modified_states.append(mod_state)
                
                # Update the row
                df.at[i, 'action'] = mod_action
                df.at[i, 'observation.state'] = mod_state
            
            # Save the updated parquet file
            df.to_parquet(episode_file)
            print(f"Saved modified data to: {episode_file}")
            
            # Update episode stats
            for ep_stat in episodes_stats:
                if ep_stat["episode_index"] == episode_idx:
                    ep_stat["stats"]["action"] = compute_stats(modified_actions)
                    ep_stat["stats"]["observation.state"] = compute_stats(modified_states)
                    break
        
        except Exception as e:
            print(f"Error processing episode {episode_idx}: {e}")
    
    # Save updated episodes stats
    with open(episodes_stats_file, 'w') as f:
        for ep_stat in episodes_stats:
            f.write(json.dumps(ep_stat) + '\n')
    print(f"Updated episodes stats saved to: {episodes_stats_file}")
    
    # Update info.json
    update_info_json(info_file)
    
    print(f"Dataset conversion completed successfully: {dataset_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert local LeRobot dataset to Trossen v1.0 format")
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Path to the local dataset directory")
    
    args = parser.parse_args()
    
    success = convert_dataset(args.dataset_path)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 