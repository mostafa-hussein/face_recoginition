import pyrealsense2 as rs, cv2, numpy as np

pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipe.start(cfg)

try:
    while True:
        frames = pipe.wait_for_frames()
        color  = np.asanyarray(frames.get_color_frame().get_data())
        depth  = np.asanyarray(frames.get_depth_frame().get_data())

        depth_col = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03),
                                      cv2.COLORMAP_JET)
        combo = np.hstack((color, depth_col))
        cv2.imshow("RGB | Depth", combo)
        if cv2.waitKey(1) == 27:  # ESC
            break
finally:
    pipe.stop()
    cv2.destroyAllWindows()