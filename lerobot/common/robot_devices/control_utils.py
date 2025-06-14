# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########################################################################################
# Utilities
########################################################################################


import logging
import time
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image
import io

import rerun as rr
import torch
from deepdiff import DeepDiff
from termcolor import colored
import cv2

from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_features_from_robot
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, has_method
from lerobot.common.vlm.vlm_client import VLMClient


def log_control_info(robot: Robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1 / dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    # TODO(aliberts): move robot-specific logs logic in robot.print_logs()
    if not robot.robot_type.startswith("stretch"):
        for name in robot.leader_arms:
            key = f"read_leader_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRlead", robot.logs[key])

        for name in robot.follower_arms:
            key = f"write_follower_{name}_goal_pos_dt_s"
            if key in robot.logs:
                log_dt("dtWfoll", robot.logs[key])

            key = f"read_follower_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRfoll", robot.logs[key])

        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    # logging.info(info_str)


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def warmup_record(
    robot,
    events,
    enable_teleoperation,
    warmup_time_s,
    display_data,
    fps,
):
    control_loop(
        robot=robot,
        control_time_s=warmup_time_s,
        display_data=display_data,
        events=events,
        fps=fps,
        teleoperate=enable_teleoperation,
    )


def record_episode(
    robot,
    dataset,
    events,
    episode_time_s,
    display_data,
    policy,
    fps,
    single_task,
):
    control_loop(
        robot=robot,
        control_time_s=episode_time_s,
        display_data=display_data,
        dataset=dataset,
        events=events,
        policy=policy,
        fps=fps,
        teleoperate=policy is None,
        single_task=single_task,
    )


@safe_stop_image_writer
def control_loop(
    robot,
    control_time_s: float,
    display_data: bool,
    dataset,
    events,
    policy,
    fps: Optional[int] = None,
    teleoperate: bool = False,
    single_task: Optional[str] = None,
    vlm_config: Optional[Dict[str, Any]] = None,
):
    """
    Main control loop for robot operation.
    
    Args:
        robot: The robot instance
        control_time_s: Duration of control in seconds
        display_data: Whether to display camera data
        dataset: Dataset for recording
        events: Event dictionary for keyboard control
        policy: Policy for robot control
        fps: Target frames per second
        teleoperate: Whether to use teleoperation
        single_task: Task description for VLM
        vlm_config: VLM configuration dictionary
    """
    # Initialize VLM client if enabled
    vlm_client = None
    if vlm_config and vlm_config.get("enabled", False):
        vlm_client = VLMClient(
            api_url=vlm_config["api_url"],
            api_key=vlm_config["api_key"],
            query_interval_s=vlm_config.get("query_interval_s", 1.0),
            max_retries=vlm_config.get("max_retries", 3),
            timeout_s=vlm_config.get("timeout_s", 5.0),
            max_queue_size=vlm_config.get("max_queue_size", 5),
            response_timeout_s=vlm_config.get("response_timeout_s", 2.0)
        )
    
    start_time = time.perf_counter()
    frame_count = 0
    last_vlm_stats_time = 0
    vlm_stats_interval = 5.0  # Log VLM stats every 5 seconds
    
    try:
        while time.perf_counter() - start_time < control_time_s:
            frame_start_time = time.perf_counter()
            
            # Get robot state and camera images
            observation = robot.get_observation()
            
            # If VLM is enabled, process the latest camera image
            vlm_response = None
            if vlm_client and single_task:
                # Get the first available camera image
                camera_image = None
                for key, value in observation.items():
                    if isinstance(value, np.ndarray) and len(value.shape) == 3:
                        camera_image = value
                        break
                
                if camera_image is not None:
                    # Convert numpy array to JPEG bytes
                    image = Image.fromarray(camera_image)
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='JPEG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    # Query VLM
                    vlm_response = vlm_client.query_vlm(img_byte_arr, single_task)
                    if vlm_response:
                        processed_response = vlm_client.process_vlm_response(vlm_response)
                        if processed_response:
                            logger.info(f"VLM Response: {processed_response}")
                            # Here you can use the VLM response to influence robot control
                            # For example, you could modify the action based on the VLM's analysis
                
                # Log VLM latency statistics periodically
                current_time = time.perf_counter()
                if current_time - last_vlm_stats_time >= vlm_stats_interval:
                    mean_latency, std_latency, max_latency = vlm_client.get_latency_stats()
                    logger.info(f"VLM Latency Stats - Mean: {mean_latency:.3f}s, Std: {std_latency:.3f}s, Max: {max_latency:.3f}s")
                    last_vlm_stats_time = current_time
            
            # Get action from policy or teleoperation
            if policy is not None:
                action = policy.predict(observation)
            elif teleoperate:
                action = robot.teleop_step()
            else:
                action = None
            
            if action is not None:
                robot.send_action(action)
            
            # Record data if dataset is provided
            if dataset is not None:
                frame = {
                    "action": action,
                    "observation": observation,
                }
                if vlm_response:
                    frame["vlm_response"] = vlm_response
                dataset.add_frame(frame)
            
            # Display data if requested
            if display_data:
                for key, value in observation.items():
                    if isinstance(value, np.ndarray) and len(value.shape) == 3:
                        cv2.imshow(key, cv2.cvtColor(value, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            
            # Control loop timing
            frame_count += 1
            if fps is not None:
                dt = time.perf_counter() - frame_start_time
                if dt < 1.0 / fps:
                    time.sleep(1.0 / fps - dt)
            
            # Check for early exit
            if events.get("exit_early", False):
                break
                
    finally:
        # Clean shutdown of VLM client
        if vlm_client:
            vlm_client.shutdown()


def reset_environment(robot, events, reset_time_s, fps):
    # TODO(rcadene): refactor warmup_record and reset_environment
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    if robot.robot_type in ["trossen_ai_stationary", "trossen_ai_solo"]:
        time.sleep(reset_time_s)
    else:
        control_loop(
            robot=robot,
            control_time_s=reset_time_s,
            events=events,
            fps=fps,
            teleoperate=True,
        )


def restore_arm_positions(robot, saved_arm_positions):
    """
    Restore both leader and follower robot arms to previously saved positions and ensure they are in teleop mode afterward.
    Uses a slow movement speed for safety and smoothness.
    
    Args:
        robot: The robot instance that contains the arms to be positioned
        saved_arm_positions: Dictionary mapping arm names to their saved positions
        
    Returns:
        bool: True if successful, False if there was an error
    """
    if not saved_arm_positions or (not hasattr(robot, "leader_arms") and not hasattr(robot, "follower_arms")):
        return False
        
    # Hardcoded slow movement time (seconds)
    SLOW_MOVE_TIME = 3.0
    
    print(f"Restoring all arms to the saved position with slow movement ({SLOW_MOVE_TIME}s)...")
    
    try:
        # First enable position mode for precise positioning for both leader and follower arms
        
        # Handle leader arms - they need special treatment since they're normally in external effort mode
        if hasattr(robot, "leader_arms") and robot.leader_arms:
            for arm_name in robot.leader_arms:
                if arm_name in saved_arm_positions:
                    leader_arm = robot.leader_arms[arm_name]
                    if hasattr(leader_arm, "write"):
                        # Enable torque to ensure position control (Torque_Enable=1 sets position mode)
                        leader_arm.write("Torque_Enable", 1)
                        print(f"Enabled position mode for leader arm: {arm_name}")
        
        # Handle follower arms
        if hasattr(robot, "follower_arms") and robot.follower_arms:
            for arm_name in robot.follower_arms:
                if arm_name in saved_arm_positions:
                    follower_arm = robot.follower_arms[arm_name]
                    if hasattr(follower_arm, "write"):
                        # Follower arms should already be in position mode, but ensure it
                        follower_arm.write("Torque_Enable", 1)
                        print(f"Enabled position mode for follower arm: {arm_name}")
        
        # CRITICAL: Give leader arms time to fully transition to position mode
        print("Waiting for leader arms to fully transition to position mode...")
        time.sleep(1.0)  # Allow time for mode transition
                
        # Move leader arms using the new set_positions method for slow movement if available
        if hasattr(robot, "leader_arms") and robot.leader_arms:
            for arm_name in robot.leader_arms:
                if arm_name in saved_arm_positions:
                    leader_arm = robot.leader_arms[arm_name]
                    
                    # Try to use the new set_positions method first for better control of movement time
                    if hasattr(leader_arm, "set_positions"):
                        print(f"Using set_positions with slow movement time for leader arm {arm_name}")
                        leader_arm.set_positions(saved_arm_positions[arm_name], time_to_move=SLOW_MOVE_TIME)
                        print(f"Restored leader arm {arm_name} to saved position")
        
        # Move follower arms with same approach
        if hasattr(robot, "follower_arms") and robot.follower_arms:
            for arm_name in robot.follower_arms:
                if arm_name in saved_arm_positions:
                    follower_arm = robot.follower_arms[arm_name]
                    
                    # Try to use the new set_positions method first for better control of movement time
                    if hasattr(follower_arm, "set_positions"):
                        print(f"Using set_positions with slow movement time for follower arm {arm_name}")
                        follower_arm.set_positions(saved_arm_positions[arm_name], time_to_move=SLOW_MOVE_TIME)
                        print(f"Restored follower arm {arm_name} to saved position")
        
        # Wait for movement to complete
        print(f"Waiting {SLOW_MOVE_TIME}s for arms to reach saved positions...")
        time.sleep(SLOW_MOVE_TIME + 0.5)  # Critical buffer time to ensure the arms reach the saved positions
        
        # Switch back to teleop mode (external effort control) for ONLY the leader arms
        # Follower arms should remain in position mode
        if hasattr(robot, "leader_arms") and robot.leader_arms:
            print("Switching leader arms back to teleoperation mode (external effort)...")
            
            for arm_name in robot.leader_arms:
                leader_arm = robot.leader_arms[arm_name]
                if hasattr(leader_arm, "write"):
                    # Disable torque to enable external effort mode
                    leader_arm.write("Torque_Enable", 0)
                    print(f"Set leader arm {arm_name} back to external effort mode")
        
        # Give the arms a moment to settle in their final modes
        time.sleep(0.5)
        return True
        
    except Exception as e:
        print(f"Warning: Could not manage arm positions/modes: {e}")
        # Try to restore teleop mode in case of failure by disabling torque (only for leader arms)
        try:
            if hasattr(robot, "leader_arms") and robot.leader_arms:
                for arm_name in robot.leader_arms:
                    leader_arm = robot.leader_arms[arm_name]
                    if hasattr(leader_arm, "write"):
                        leader_arm.write("Torque_Enable", 0)
        except Exception as ex:
            print(f"Failed to restore external effort mode after error: {ex}")
        return False


def save_arm_positions(robot):
    """
    Save the current positions of both leader and follower robot arms.
    
    Args:
        robot: The robot instance that contains the arms whose positions will be saved
        
    Returns:
        dict: A dictionary mapping arm names to their current positions, or an empty dict if there was an error
    """
    saved_positions = {}
    
    try:
        # Save leader arm positions
        if hasattr(robot, "leader_arms") and robot.leader_arms:
            for arm_name in robot.leader_arms:
                leader_arm = robot.leader_arms[arm_name]
                if hasattr(leader_arm, "read"):
                    # Read the current position
                    saved_positions[arm_name] = leader_arm.read("Present_Position")
                    print(f"Saved leader arm {arm_name} position: {saved_positions[arm_name]}")
        
        # Save follower arm positions
        if hasattr(robot, "follower_arms") and robot.follower_arms:
            for arm_name in robot.follower_arms:
                follower_arm = robot.follower_arms[arm_name]
                if hasattr(follower_arm, "read"):
                    # Read the current position
                    saved_positions[arm_name] = follower_arm.read("Present_Position")
                    print(f"Saved follower arm {arm_name} position: {saved_positions[arm_name]}")
                    
        return saved_positions
    except Exception as e:
        print(f"Warning: Could not save arm positions: {e}")
        return {}  # Return empty dict if there was an error


def stop_recording(robot, listener, display_data):
    robot.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()


def sanity_check_dataset_name(repo_id, policy_cfg):
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided ({policy_cfg.type})."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, use_videos: bool
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, get_features_from_robot(robot, use_videos)),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
