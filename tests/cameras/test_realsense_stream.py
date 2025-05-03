import pyrealsense2 as rs
import numpy as np
import time

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
config = rs.config()

# Get device serial number
ctx = rs.context()
devices = ctx.query_devices()
if len(devices) > 0:
    device = devices[0]
    serial_number = device.get_info(rs.camera_info.serial_number)
    print(f"Using device: {device.get_info(rs.camera_info.name)} (S/N: {serial_number})")
    
    # Enable the device
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        # Start streaming
        print("Starting pipeline...")
        pipeline.start(config)
        print("Pipeline started successfully!")
        
        # Wait for a few frames
        print("Trying to get frames...")
        for i in range(5):
            try:
                print(f"Waiting for frame {i+1}...")
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    print("Empty color frame, continuing...")
                    continue
                    
                # Convert frame to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                print(f"Frame received! Shape: {color_image.shape}")
                time.sleep(0.1)
            except Exception as e:
                print(f"Error getting frame: {e}")
        
        print("Test completed successfully!")
    except Exception as e:
        print(f"Error starting pipeline: {e}")
    finally:
        # Stop streaming
        print("Stopping pipeline...")
        pipeline.stop()
else:
    print("No devices found!") 