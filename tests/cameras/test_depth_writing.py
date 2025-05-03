#!/usr/bin/env python3

import os
import numpy as np
import torch
import PIL.Image
from pathlib import Path

# This simulates our add_frame method's depth handling
def save_depth_image(depth_array, output_path):
    """
    Save a depth image using our new implementation.
    
    Args:
        depth_array: Numpy array or tensor representing depth data
        output_path: Path where to save the depth image
    """
    # Convert tensor to numpy if needed
    if isinstance(depth_array, torch.Tensor):
        depth_array = depth_array.numpy()
    
    # Handle potential channel dimension - could be in either (C, H, W) or (H, W, C) format
    if len(depth_array.shape) == 3:
        if depth_array.shape[0] == 1:  # Channel-first format (1, H, W)
            depth_array = depth_array[0]  # Remove channel dimension to get (H, W)
        elif depth_array.shape[2] == 1:  # Channel-last format (H, W, 1)
            depth_array = depth_array[:, :, 0]  # Remove channel dimension to get (H, W)
    
    try:
        # Create PIL Image directly from the array as uint16
        depth_img = PIL.Image.fromarray(depth_array.astype(np.uint16))
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save as 16-bit PNG
        depth_img.save(output_path)
        print(f"Successfully saved depth image to {output_path}")
        return True
    except Exception as e:
        print(f"Error writing depth image {output_path}: {e}")
        return False

# Create test directory
test_dir = Path("./test_depth_output")
os.makedirs(test_dir, exist_ok=True)

# Test case 1: Single-channel depth array [H, W]
print("\nTest case 1: 2D depth array [H, W]")
depth_2d = np.random.randint(0, 65535, size=(480, 640), dtype=np.uint16)
save_depth_image(depth_2d, test_dir / "depth_2d.png")

# Test case 2: Channel-first depth tensor [1, H, W]
print("\nTest case 2: 3D depth tensor with channel-first format [1, H, W]")
depth_3d_chfirst = torch.from_numpy(np.random.randint(0, 65535, size=(1, 480, 640), dtype=np.uint16))
save_depth_image(depth_3d_chfirst, test_dir / "depth_3d_chfirst.png")

# Test case 3: Channel-last depth tensor [H, W, 1]
print("\nTest case 3: 3D depth tensor with channel-last format [H, W, 1]")
depth_3d_chlast = np.random.randint(0, 65535, size=(480, 640, 1), dtype=np.uint16)
save_depth_image(depth_3d_chlast, test_dir / "depth_3d_chlast.png")

# Test case 4: Try loading the saved images
print("\nTest case 4: Loading saved depth images")
try:
    img_2d = PIL.Image.open(test_dir / "depth_2d.png")
    img_3d_chfirst = PIL.Image.open(test_dir / "depth_3d_chfirst.png")
    img_3d_chlast = PIL.Image.open(test_dir / "depth_3d_chlast.png")
    
    print(f"Successfully loaded 2D image: {img_2d.size}, mode: {img_2d.mode}")
    print(f"Successfully loaded channel-first image: {img_3d_chfirst.size}, mode: {img_3d_chfirst.mode}")
    print(f"Successfully loaded channel-last image: {img_3d_chlast.size}, mode: {img_3d_chlast.mode}")
    
    # Verify data type is preserved
    array_2d = np.array(img_2d)
    array_3d_chfirst = np.array(img_3d_chfirst)
    array_3d_chlast = np.array(img_3d_chlast)
    
    print(f"Loaded 2D image dtype: {array_2d.dtype}, shape: {array_2d.shape}")
    print(f"Loaded channel-first image dtype: {array_3d_chfirst.dtype}, shape: {array_3d_chfirst.shape}")
    print(f"Loaded channel-last image dtype: {array_3d_chlast.dtype}, shape: {array_3d_chlast.shape}")
except Exception as e:
    print(f"Error loading depth images: {e}")

print("\nTest completed.") 