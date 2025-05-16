import time
import traceback

import numpy as np
import trossen_arm as trossen

from lerobot.common.robot_devices.motors.configs import TrossenArmDriverConfig
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
)

PITCH_CIRCLE_RADIUS = 0.00875 # meters
VEL_LIMITS = [3.375, 3.375, 3.375, 7.0, 7.0, 7.0, 12.5 * PITCH_CIRCLE_RADIUS]

TROSSEN_ARM_MODELS = {
    "V0_LEADER": [trossen.Model.wxai_v0, trossen.StandardEndEffector.wxai_v0_leader],
    "V0_FOLLOWER": [trossen.Model.wxai_v0, trossen.StandardEndEffector.wxai_v0_follower],
}

class TrossenArmDriver:
    """
        The `TrossenArmDriver` class provides an interface for controlling
        Trossen Robotics' robotic arms. It leverages the trossen_arm for communication with arms.

        This class allows for configuration, torque management, and motion control of robotic arms. It includes features for handling connection states, moving the
        arm to specified poses, and logging timestamps for debugging and performance analysis.

        ### Key Features:
        - **Multi-motor Control:** Supports multiple motors connected to a bus.
        - **Mode Switching:** Enables switching between position and gravity compensation modes.
        - **Home and Sleep Pose Management:** Automatically transitions the arm to home and sleep poses for safe operation.
        - **Error Handling:** Raises specific exceptions for connection and operational errors.
        - **Logging:** Captures timestamps for operations to aid in debugging.

        ### Example Usage:
        ```python
        motors = {
            "joint_0": (1, "4340"),
            "joint_1": (2, "4340"),
            "joint_2": (4, "4340"),
            "joint_3": (6, "4310"),
            "joint_4": (7, "4310"),
            "joint_5": (8, "4310"),
            "joint_6": (9, "4310"),
        }
        arm_driver = TrossenArmDriver(
            motors=motors,
            ip="192.168.1.2",
            model="V0_LEADER",
        )
        arm_driver.connect()

        # Read motor positions
        positions = arm_driver.read("Present_Position")

        # Move to a new position (Home Pose)
        # Last joint is the gripper, which is in range [0, 450]
        arm_driver.write("Goal_Position", [0, 15, 15, 0, 0, 0, 200])

        # Disconnect when done
        arm_driver.disconnect()
        ```
    """


    def __init__(
        self,
        config: TrossenArmDriverConfig,
    ):
        self.ip = config.ip
        self.model = config.model
        self.mock = config.mock
        self.is_policy_in_radians = config.is_policy_in_radians
        self.driver = None
        self.calibration = None
        self.is_connected = False
        self.logs = {}
        self.fps = 30
        self.home_pose = [0, np.pi/12, np.pi/12, 0, 0, 0, 0]
        self.sleep_pose = [0, 0, 0, 0, 0, 0, 0]

        self.motors={
                    # name: (index, model)
                    "joint_0": [1, "4340"],
                    "joint_1": [2, "4340"],
                    "joint_2": [3, "4340"],
                    "joint_3": [4, "4310"],
                    "joint_4": [5, "4310"],
                    "joint_5": [6, "4310"],
                    "joint_6": [7, "4310"],
                }

        self.prev_write_time = 0
        self.current_write_time = None

        # To prevent DiscontinuityError due to large jumps in position in short time.
        # We scale the time to move based on the distance between the start and goal values and the maximum speed of the motors.
        # The below factor is used to scale the time to move.
        self.TIME_SCALING_FACTOR = 3.0

        # Minimum time to move for the arm (This is a tuning parameter)
        self.MIN_TIME_TO_MOVE = 6.0 / self.fps

    def connect(self):
        print(f"Connecting to {self.model} arm at {self.ip}...")
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"TrossenArmDriver({self.ip}) is already connected. Do not call `motors_bus.connect()` twice."
            )

        print("Initializing the drivers...")

        # Initialize the driver
        self.driver = trossen.TrossenArmDriver()

        # Get the model configuration
        try:
            model_name, model_end_effector = TROSSEN_ARM_MODELS[self.model]
        except KeyError:
            raise ValueError(f"Unsupported model: {self.model}")

        print("Configuring the drivers...")

        # Configure the driver
        try:
            self.driver.configure(model_name, model_end_effector, self.ip, True)
        except Exception:
            traceback.print_exc()
            print(
                f"Failed to configure the driver for the {self.model} arm at {self.ip}."
            )
            raise
        
        # Move the arms to the home pose
        self.driver.set_all_modes(trossen.Mode.position)
        self.driver.set_all_positions(self.home_pose, 2.0, False)

        # Allow to read and write - this must come BEFORE setting characteristics
        self.is_connected = True
            
        # Now that we're connected, apply custom joint characteristics if needed
        if self.model == "V0_LEADER":
            print("Setting custom joint characteristics for leader arm gripper...")
            try:
                self.set_custom_joint_characteristics()
            except Exception as e:
                print(f"Warning: Failed to set custom joint characteristics: {e}")
                traceback.print_exc()
                # Continue with connection even if this fails

        # Set a default gripper force limit for follower arm
        if self.model == "V0_FOLLOWER":
            print("Current gripper force limit scaling factor: ", self.get_gripper_force_limit_scaling_factor())
            print("Setting default gripper force limit for follower arm...")
            
            try:
                self.set_gripper_force_limit_scaling_factor(0.1)  # 50% of maximum force by default
                print("After setting force limit scaling factor: ", self.get_gripper_force_limit_scaling_factor())
            except Exception as e:
                print(f"Warning: Failed to set gripper force limit: {e}")
                traceback.print_exc()
                # Continue with connection even if this fails

    def reconnect(self):
        try:
            model_name, model_end_effector = TROSSEN_ARM_MODELS[self.model]
        except KeyError:
            raise ValueError(f"Unsupported model: {self.model}")
        try:
            self.driver.configure(model_name, model_end_effector, self.ip, True)
        except Exception:
            traceback.print_exc()
            print(
                f"Failed to configure the driver for the {self.model} arm at {self.ip}."
            )
            raise

        self.is_connected = True


    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def set_calibration(self, calibration: dict[str, list]):
        self.calibration = calibration

    def apply_calibration_autocorrect(self, values: np.ndarray | list, motor_names: list[str] | None):
        pass

    def apply_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        pass

    def autocorrect_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        pass

    def revert_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        pass

    def read(self, data_name, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"TrossenArmDriver({self.ip}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()

        # Read the present position of the motors
        if data_name == "Present_Position":
            # Get the positions of the motors
            values = self.driver.get_positions()
            if not self.is_policy_in_radians:
                values[:-1] = np.degrees(values[:-1])  # Convert all joints except gripper
                values[-1] = values[-1] * 10000  # Convert gripper to range (0-450)
        elif data_name == "Gripper_Force_Limit":
            # Read the current gripper force limit
            values = np.array([self.get_gripper_force_limit_scaling_factor()], dtype=np.float32)
        else:
            values = None
            print(f"Data name: {data_name} is not supported for reading.")

        # TODO: Add support for reading other data names as required

        self.logs["delta_timestamp_s_read"] = time.perf_counter() - start_time

        values = np.array(values, dtype=np.float32)
        return values

    def compute_time_to_move(self, goal_values: np.ndarray):
        # Compute the time to move based on the distance between the start and goal values
        # and the maximum speed of the motors
        current_pose = self.driver.get_positions()
        displacement = abs(goal_values - current_pose)
        time_to_move_all_joints = self.TIME_SCALING_FACTOR*displacement / VEL_LIMITS
        time_to_move = max(time_to_move_all_joints)
        time_to_move = max(time_to_move, self.MIN_TIME_TO_MOVE)
        return time_to_move

    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None, gripper_resistance: float | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"TrossenArmDriver({self.ip}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()

        # Write the goal position of the motors
        if data_name == "Goal_Position":
            values = np.array(values, dtype=np.float32)
            # Only convert if policy values are in degrees
            if not self.is_policy_in_radians:
                # Convert back to radians for joints
                values[:-1] = np.radians(values[:-1])  # Convert all joints except gripper
                values[-1] = values[-1] / 10000  # Convert gripper back to range (0-0.045)
            time_to_move = self.compute_time_to_move(values)
            # time_to_move = 0.02 # Setting to a hardcoded values to make the gripper more responsive
            self.driver.set_all_positions(values.tolist(), time_to_move, False)
            self.prev_write_time = self.current_write_time

        # Enable or disable the torque of the motors
        elif data_name == "Torque_Enable":
            # Set the arms to POSITION mode
            if values == 1:
                self.driver.set_all_modes(trossen.Mode.position)
            else:
                # If this is a leader arm, make sure custom joint characteristics are applied
                if self.model == "V0_LEADER":
                    print(f"Switching {self.ip} to external_effort mode with custom joint characteristics")
                    # First make sure joint characteristics are properly set
                    self.set_custom_joint_characteristics()
                    
                # Put the arm in external effort mode (gravity compensation)
                self.driver.set_all_modes(trossen.Mode.external_effort)
                
                # Use a non-zero goal time for smoother transition (2.0 seconds)
                self.driver.set_all_external_efforts([0.0] * 7, 0.0, True)
            
        # Set the gripper force limit scaling factor
        elif data_name == "Gripper_Force_Limit":
            # Convert input to float value between 0 and 1
            if isinstance(values, (list, np.ndarray)):
                values = float(values[0])
            else:
                values = float(values)
            # Ensure value is between 0 and 1
            values = max(0.0, min(1.0, values))
            self.set_gripper_force_limit_scaling_factor(values)
                
        elif data_name == "Reset":
            self.driver.set_all_modes(trossen.Mode.position)
            self.driver.set_all_positions(self.home_pose, 2.0, False)
        else:
            print(f"Data name: {data_name} value: {values} is not supported for writing.")

        self.logs["delta_timestamp_s_write"] = time.perf_counter() - start_time

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"TrossenArmDriver ({self.ip}) is not connected. Try running `motors_bus.connect()` first."
            )
        self.driver.set_all_modes(trossen.Mode.position)
        self.driver.set_all_positions(self.home_pose, 2.0, False)
        self.driver.set_all_positions(self.sleep_pose, 2.0, False)

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
            
    def set_custom_joint_characteristics(self):
        """
        Set custom joint characteristics for the gripper to prevent oscillation.
        
        This method modifies the friction and effort parameters of the last joint (gripper)
        to prevent oscillation during operation.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"TrossenArmDriver({self.ip}) is not connected. You need to run `motors_bus.connect()`."
            )
            
        try:
            # Only proceed for leader arms which have the oscillation issue
            if self.model != "V0_LEADER":
                print(f"Skipping joint characteristic customization for non-leader model: {self.model}")
                return False
                
            print(f"Setting custom joint characteristics for gripper on {self.ip}")
            
            # Get the current joint characteristics 
            joint_characteristics = self.driver.get_joint_characteristics()
            
            if not joint_characteristics or len(joint_characteristics) < 7:
                print(f"Warning: Could not get joint characteristics or unexpected number of joints")
                return False
                
            # The gripper is the last joint (index 6)
            num_joints = len(joint_characteristics)
            gripper_index = num_joints - 1
            
            # Modify the gripper's joint characteristics to prevent oscillation
            joint_characteristics[gripper_index].friction_viscous_coef = 1.0  # Damping to prevent oscillation
            joint_characteristics[gripper_index].friction_constant_term = 0.1  # Low constant friction
            joint_characteristics[gripper_index].friction_coulomb_coef = 0.05  # Small coulomb friction for stability
            joint_characteristics[gripper_index].friction_transition_velocity = 0.01  # Low transition velocity
            joint_characteristics[gripper_index].effort_correction = 2.0  # Maximum sensitivity
            
            # Set the modified joint characteristics
            success = self.driver.set_joint_characteristics(joint_characteristics)
            
            if success:
                print(f"Successfully set custom joint characteristics for gripper on {self.ip}")
                # Print the values that were set
                print(f"  friction_viscous_coef: {joint_characteristics[gripper_index].friction_viscous_coef}")
                print(f"  friction_constant_term: {joint_characteristics[gripper_index].friction_constant_term}")
                print(f"  friction_coulomb_coef: {joint_characteristics[gripper_index].friction_coulomb_coef}")
                print(f"  friction_transition_velocity: {joint_characteristics[gripper_index].friction_transition_velocity}")
                print(f"  effort_correction: {joint_characteristics[gripper_index].effort_correction}")
            else:
                print(f"Failed to set joint characteristics")
                
            return success
            
        except Exception as e:
            print(f"Error setting custom joint characteristics: {e}")
            return False
            
    def get_gripper_force_limit_scaling_factor(self):
        """
        Get the current gripper force limit scaling factor.
        
        Returns:
            float: A value between 0.0 and 1.0 where:
                - 0.0 means no force (gripper won't close)
                - 1.0 means maximum force as specified in the hardware specs
                
        Note:
            According to the WidowX AI specs, max gripping force is around 340N at full power.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"TrossenArmDriver({self.ip}) is not connected. You need to run `motors_bus.connect()`."
            )
            
        try:
            # Method 1: Direct API call (preferred)
            scaling_factor = self.driver.get_gripper_force_limit_scaling_factor()
            return scaling_factor
        except Exception as e1:
            print(f"Warning: Failed to get gripper force limit scaling factor using direct API: {e1}")
            try:
                # Method 2: Via end effector properties
                end_effector = self.driver.get_end_effector()
                scaling_factor = end_effector.t_max_factor
                return scaling_factor
            except Exception as e2:
                print(f"Error getting gripper force limit scaling factor: {e2}")
                return None
    
    def set_gripper_force_limit_scaling_factor(self, scaling_factor=0.5):
        """
        Set the maximum force that the gripper can apply.
        
        Args:
            scaling_factor (float): A value between 0.0 and 1.0 where:
                - 0.0 means no force (gripper won't close)
                - 1.0 means maximum force as specified in the hardware specs
                - Default is 0.5 (50% of maximum force)
                
        Returns:
            bool: True if successful, False otherwise
                
        Note:
            This is particularly useful for preventing damage to objects when grasping.
            According to the WidowX AI specs, max gripping force is around 340N at full power.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"TrossenArmDriver({self.ip}) is not connected. You need to run `motors_bus.connect()`."
            )
            
        # Ensure scaling factor is within valid range
        scaling_factor = max(0.0, min(1.0, scaling_factor))
        
        try:
            # Method 1: Direct API call (preferred)
            print(f"Setting gripper force limit to {scaling_factor*100:.1f}% of maximum")
            self.driver.set_gripper_force_limit_scaling_factor(scaling_factor)
            return True
        except Exception as e1:
            print(f"Warning: Failed to set gripper force limit using direct API: {e1}")
            try:
                # Method 2: Via end effector properties
                end_effector = self.driver.get_end_effector()
                end_effector.t_max_factor = scaling_factor
                self.driver.set_end_effector(end_effector)
                return True
            except Exception as e2:
                print(f"Error setting gripper force limit scaling factor: {e2}")
                return False

    def set_positions(self, positions, time_to_move=None, block=False):
        """
        Direct access to set all positions with a configurable movement time.
        
        Args:
            positions (list or np.ndarray): The goal positions for all joints
            time_to_move (float, optional): Time in seconds to complete the movement.
                                           If None, uses SLOW_MOVEMENT_TIME (default 3.0s)
            block (bool): Whether to block until the movement is complete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"TrossenArmDriver({self.ip}) is not connected. You need to run `motors_bus.connect()`."
            )
            
        # Convert to proper format
        positions = np.array(positions, dtype=np.float32).tolist()
        if not self.is_policy_in_radians:
            positions[:-1] = np.radians(positions[:-1])  # Convert all joints except gripper
            positions[-1] = positions[-1] / 10000  # Convert gripper back to range (0-0.045)
        
        # Use default slow movement time if not specified
        if time_to_move is None:
            time_to_move = self.SLOW_MOVEMENT_TIME
            
        print(f"Moving arm to position with time_to_move={time_to_move}s")
        
        try:
            # Call the underlying driver method directly
            self.driver.set_all_positions(positions, time_to_move, block)
            return True
        except Exception as e:
            print(f"Error setting positions: {e}")
            return False
