#!/usr/bin/env python3
import os
import json
import shutil
import argparse
from pathlib import Path
import jsonlines
import sys

def create_dataset_subset(source_dataset_path, target_dataset_path, num_episodes):
    """
    Create a subset of the dataset with the specified number of episodes.
    
    Args:
        source_dataset_path: Path to the source dataset directory
        target_dataset_path: Path to the target directory where the subset will be created
        num_episodes: Number of episodes to include in the subset
    """
    source_path = Path(source_dataset_path)
    target_path = Path(target_dataset_path)
    
    if not source_path.exists():
        print(f"Error: Source dataset path {source_path} does not exist", file=sys.stderr)
        return False
    
    # Print summary of the operation
    print(f"Creating dataset subset with {num_episodes} episodes...")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    
    # Create target directory structure
    print(f"Creating directory structure at {target_path}")
    target_path.mkdir(parents=True, exist_ok=True)
    for subdir in ['meta', 'data/chunk-000', 'videos/chunk-000']:
        (target_path / subdir).mkdir(parents=True, exist_ok=True)
    
    # Copy and filter episodes.jsonl and episodes_stats.jsonl
    print(f"Processing metadata files")
    if not process_meta_files(source_path, target_path, num_episodes):
        return False
    
    # Copy the actual episode data
    print(f"Copying data files for {num_episodes} episodes")
    copy_episode_data(source_path, target_path, num_episodes)
    
    # Copy video directory structure
    print(f"Setting up video directory structure")
    if not setup_video_structure(source_path, target_path):
        print("Warning: Could not set up video directory structure completely")
    
    # Copy videos for included episodes
    print(f"Copying video files for {num_episodes} episodes")
    total_videos = copy_episode_videos(source_path, target_path, num_episodes)
    
    # Update info.json with correct counts
    print(f"Updating info.json with new episode and video counts")
    update_info_json(source_path, target_path, num_episodes, total_videos)
    
    print(f"Dataset subset with {num_episodes} episodes created successfully at {target_path}")
    return True

def update_info_json(source_path, target_path, num_episodes, total_videos):
    """Update the info.json file with the correct episode and video counts."""
    info_file = source_path / 'meta' / 'info.json'
    target_info_file = target_path / 'meta' / 'info.json'
    
    if not info_file.exists():
        print(f"Warning: info.json not found at {info_file}, skipping update")
        return False
    
    try:
        with open(info_file, 'r') as f:
            info_data = json.load(f)
        
        # Update the total episodes count
        info_data['total_episodes'] = num_episodes
        
        # Update videos count if we have that information
        if total_videos is not None:
            info_data['total_videos'] = total_videos
        
        # Calculate total frames based on episodes.jsonl
        episodes_file = target_path / 'meta' / 'episodes.jsonl'
        if episodes_file.exists():
            total_frames = 0
            with jsonlines.open(episodes_file, 'r') as reader:
                for episode in reader:
                    total_frames += episode.get('length', 0)
            
            info_data['total_frames'] = total_frames
        
        # Update any splits information if necessary
        if 'splits' in info_data and 'train' in info_data['splits']:
            info_data['splits']['train'] = f"0:{num_episodes}"
        
        # Write the updated info file
        with open(target_info_file, 'w') as f:
            json.dump(info_data, f, indent=4)
        
        return True
    except Exception as e:
        print(f"Error updating info.json: {e}", file=sys.stderr)
        return False

def process_meta_files(source_path, target_path, num_episodes):
    """Process the metadata files to include only the specified number of episodes."""
    try:
        # Copy episodes.jsonl with only the first num_episodes
        episodes_file = source_path / 'meta' / 'episodes.jsonl'
        if not episodes_file.exists():
            print(f"Error: episodes.jsonl not found at {episodes_file}", file=sys.stderr)
            return False
            
        with jsonlines.open(episodes_file, 'r') as reader:
            episodes = list(reader)
        
        # Ensure we don't request more episodes than available
        if num_episodes > len(episodes):
            print(f"Warning: Requested {num_episodes} episodes but only {len(episodes)} are available.")
            num_episodes = len(episodes)
        
        subset_episodes = episodes[:num_episodes]
        
        with jsonlines.open(target_path / 'meta' / 'episodes.jsonl', 'w') as writer:
            for episode in subset_episodes:
                writer.write(episode)
        
        # Copy episodes_stats.jsonl with only the first num_episodes
        stats_file = source_path / 'meta' / 'episodes_stats.jsonl'
        if stats_file.exists():
            with jsonlines.open(stats_file, 'r') as reader:
                episodes_stats = list(reader)
            
            subset_episodes_stats = episodes_stats[:num_episodes]
            
            with jsonlines.open(target_path / 'meta' / 'episodes_stats.jsonl', 'w') as writer:
                for episode_stat in subset_episodes_stats:
                    writer.write(episode_stat)
        else:
            print(f"Warning: episodes_stats.jsonl not found at {stats_file}")
        
        # Copy other meta files
        for file_name in ['modality.json', 'tasks.jsonl']:
            source_file = source_path / 'meta' / file_name
            target_file = target_path / 'meta' / file_name
            
            if source_file.exists():
                shutil.copy2(source_file, target_file)
        
        return True
    except Exception as e:
        print(f"Error processing metadata files: {e}", file=sys.stderr)
        return False

def copy_episode_data(source_path, target_path, num_episodes):
    """Copy the data files for the specified number of episodes."""
    copied_count = 0
    missing_count = 0
    
    for episode_idx in range(num_episodes):
        episode_file = f"episode_{episode_idx:06d}.parquet"
        source_file = source_path / 'data' / 'chunk-000' / episode_file
        target_file = target_path / 'data' / 'chunk-000' / episode_file
        
        if source_file.exists():
            try:
                shutil.copy2(source_file, target_file)
                copied_count += 1
            except Exception as e:
                print(f"Error copying data file {episode_file}: {e}", file=sys.stderr)
        else:
            missing_count += 1
            if missing_count < 5:  # Limit warnings to avoid flooding the console
                print(f"Warning: Episode data file {episode_file} not found")
            elif missing_count == 5:
                print("Warning: Additional missing episode files not shown...")
    
    print(f"Copied {copied_count} episode data files, {missing_count} files were missing")
    return True

def setup_video_structure(source_path, target_path):
    """Set up the video directory structure."""
    source_video_dir = source_path / 'videos' / 'chunk-000'
    target_video_dir = target_path / 'videos' / 'chunk-000'
    
    if not source_video_dir.exists():
        print(f"Warning: Source video directory not found at {source_video_dir}")
        return False
    
    try:
        for camera_dir in source_video_dir.iterdir():
            if camera_dir.is_dir():
                camera_name = camera_dir.name
                (target_video_dir / camera_name).mkdir(exist_ok=True)
        return True
    except Exception as e:
        print(f"Error setting up video structure: {e}", file=sys.stderr)
        return False

def copy_episode_videos(source_path, target_path, num_episodes):
    """Copy video files for the specified number of episodes."""
    source_video_dir = source_path / 'videos' / 'chunk-000'
    target_video_dir = target_path / 'videos' / 'chunk-000'
    
    if not source_video_dir.exists():
        print(f"Warning: Source video directory not found at {source_video_dir}")
        return 0
    
    total_copied = 0
    
    # For each camera directory
    for camera_dir in source_video_dir.iterdir():
        if camera_dir.is_dir():
            camera_name = camera_dir.name
            camera_copied = 0
            
            # Copy video files for each episode
            for episode_idx in range(num_episodes):
                # Look for all files related to this episode
                episode_pattern = f"episode_{episode_idx:06d}."
                
                for video_file in camera_dir.glob(f"{episode_pattern}*"):
                    target_file = target_video_dir / camera_name / video_file.name
                    try:
                        shutil.copy2(video_file, target_file)
                        camera_copied += 1
                        total_copied += 1
                    except Exception as e:
                        print(f"Error copying video file {video_file.name}: {e}", file=sys.stderr)
            
            print(f"Copied {camera_copied} video files for camera {camera_name}")
    
    print(f"Total of {total_copied} video files copied")
    return total_copied

def main():
    parser = argparse.ArgumentParser(
        description="Create a subset of a dataset with a specified number of episodes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("source_dataset", 
                        help="Path to the source dataset directory")
    parser.add_argument("target_dataset", 
                        help="Path to the target directory where the subset will be created")
    parser.add_argument("num_episodes", type=int, 
                        help="Number of episodes to include in the subset")
    parser.add_argument("--skip-videos", action="store_true",
                        help="Skip copying video files (useful for large datasets)")
    
    args = parser.parse_args()
    
    # Validate the source exists
    if not os.path.isdir(args.source_dataset):
        print(f"Error: Source dataset directory does not exist: {args.source_dataset}", file=sys.stderr)
        sys.exit(1)
    
    # Modify function to respect the skip-videos flag
    original_copy_episode_videos = copy_episode_videos
    total_videos = None
    
    if args.skip_videos:
        def skip_videos(*args, **kwargs):
            print("Skipping video files as requested")
            return None
        
        # Replace the function temporarily
        globals()['copy_episode_videos'] = skip_videos
    
    # Create the dataset subset
    success = create_dataset_subset(args.source_dataset, args.target_dataset, args.num_episodes)
    
    # Restore the original function
    if args.skip_videos:
        globals()['copy_episode_videos'] = original_copy_episode_videos
    
    # Return appropriate exit code
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 