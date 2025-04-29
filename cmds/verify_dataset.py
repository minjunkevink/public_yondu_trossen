#!/usr/bin/env python3
"""
Verify dataset integrity by checking for missing files across all episodes.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set


def verify_dataset(dataset_path: str, verbose: bool = False) -> Dict:
    """
    Verify dataset integrity by checking for missing files.
    
    Args:
        dataset_path: Path to the dataset directory
        verbose: Whether to print detailed information during verification
        
    Returns:
        Dictionary with verification results
    """
    dataset_dir = Path(dataset_path)
    
    # Check if dataset exists
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Load info.json to get total episodes
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found at {info_path}")
    
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    total_episodes = info.get("total_episodes")
    if total_episodes is None:
        raise ValueError("'total_episodes' key not found in info.json")
    
    if verbose:
        print(f"Dataset should have {total_episodes} episodes")
    
    # Initialize results
    results = {
        "total_episodes": total_episodes,
        "missing_parquet_files": [],
        "camera_directories": [],
        "missing_video_files": {}
    }
    
    # Check parquet files (data/chunk-000/episode_*.parquet)
    parquet_dir = dataset_dir / "data" / "chunk-000"
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")
    
    # Find existing parquet files
    existing_parquet_files = set()
    for file in parquet_dir.glob("episode_*.parquet"):
        try:
            episode_num = int(file.stem.split('_')[1])
            existing_parquet_files.add(episode_num)
        except (IndexError, ValueError):
            if verbose:
                print(f"Warning: Unexpected parquet file name format: {file.name}")
    
    # Find missing parquet files
    for episode in range(total_episodes):
        if episode not in existing_parquet_files:
            results["missing_parquet_files"].append(episode)
    
    if verbose:
        if results["missing_parquet_files"]:
            print(f"Missing {len(results['missing_parquet_files'])} parquet files")
        else:
            print("All parquet files present")
    
    # Find camera directories
    videos_dir = dataset_dir / "videos" / "chunk-000"
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
    
    camera_dirs = [d for d in videos_dir.iterdir() if d.is_dir()]
    results["camera_directories"] = [d.name for d in camera_dirs]
    
    if verbose:
        print(f"Found {len(camera_dirs)} camera directories: {results['camera_directories']}")
    
    # Check video files for each camera
    for camera_dir in camera_dirs:
        camera_name = camera_dir.name
        results["missing_video_files"][camera_name] = []
        
        # Find existing video files
        existing_video_files = set()
        for file in camera_dir.glob("episode_*.mp4"):
            try:
                episode_num = int(file.stem.split('_')[1])
                existing_video_files.add(episode_num)
            except (IndexError, ValueError):
                if verbose:
                    print(f"Warning: Unexpected video file name format: {file.name}")
        
        # Find missing video files
        for episode in range(total_episodes):
            if episode not in existing_video_files:
                results["missing_video_files"][camera_name].append(episode)
        
        if verbose:
            if results["missing_video_files"][camera_name]:
                print(f"Missing {len(results['missing_video_files'][camera_name])} video files for {camera_name}")
            else:
                print(f"All video files present for {camera_name}")
    
    return results


def print_results(results: Dict, detailed: bool = False):
    """Print verification results in a human-readable format."""
    print("\n=== Dataset Verification Results ===")
    print(f"Total episodes: {results['total_episodes']}")
    
    # Print missing parquet files
    missing_parquet_count = len(results["missing_parquet_files"])
    print(f"\nMissing parquet files: {missing_parquet_count}")
    if missing_parquet_count > 0:
        # Always print missing episode indices, regardless of detailed flag
        if missing_parquet_count <= 10:
            print(f"  Missing episodes: {', '.join(map(str, results['missing_parquet_files']))}")
        else:
            print(f"  First 10 missing episodes: {', '.join(map(str, results['missing_parquet_files'][:10]))}")
            print(f"  ... and {missing_parquet_count - 10} more")
        
        # Show additional details if requested
        if detailed:
            # Additional detailed information about parquet files could go here
            pass
    
    # Print missing video files for each camera
    print(f"\nCamera directories found: {len(results['camera_directories'])}")
    for camera in results["camera_directories"]:
        missing_videos = results["missing_video_files"][camera]
        missing_count = len(missing_videos)
        print(f"\n  {camera}: {missing_count} missing videos")
        if missing_count > 0:
            # Always print missing episode indices, regardless of detailed flag
            if missing_count <= 10:
                print(f"    Missing episodes: {', '.join(map(str, missing_videos))}")
            else:
                print(f"    First 10 missing episodes: {', '.join(map(str, missing_videos[:10]))}")
                print(f"    ... and {missing_count - 10} more")
            
            # Show additional details if requested
            if detailed:
                # Additional detailed information about video files could go here
                pass
    
    # Print summary
    print("\n=== Summary ===")
    complete_episodes = results['total_episodes'] - max(
        len(results["missing_parquet_files"]),
        *[len(missing) for missing in results["missing_video_files"].values()]
    )
    print(f"Complete episodes: {complete_episodes}/{results['total_episodes']} " +
          f"({complete_episodes/results['total_episodes']*100:.1f}%)")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(description="Verify dataset integrity")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output")
    parser.add_argument("--detailed", "-d", action="store_true", help="Print detailed missing files")
    parser.add_argument("--output", "-o", type=str, help="Save results to JSON file")
    args = parser.parse_args()
    
    try:
        results = verify_dataset(args.dataset_path, args.verbose)
        print_results(results, args.detailed)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
                print(f"\nResults saved to {args.output}")
                
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
