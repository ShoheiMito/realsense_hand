import pyrealsense2 as rs
ctx = rs.context()
devs = ctx.query_devices()
if len(devs) == 0:
    print('ERROR: デバイスが見つかりません')
else:
    d = devs[0]
    print(f'デバイス: {d.get_info(rs.camera_info.name)}')
    print(f'USB: {d.get_info(rs.camera_info.usb_type_descriptor)}')
