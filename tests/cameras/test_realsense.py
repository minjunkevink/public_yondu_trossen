import pyrealsense2 as rs

try:
    ctx = rs.context()
    devices = ctx.query_devices()
    print(f'Found {len(devices)} devices:')
    for dev in devices:
        print(f'  - {dev.get_info(rs.camera_info.name)} (S/N: {dev.get_info(rs.camera_info.serial_number)})')
except Exception as e:
    print(f'Error: {e}') 