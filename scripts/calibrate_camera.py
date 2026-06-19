from src.camera.realsense_manager import RealSenseManager
import pyrealsense2 as rs

def calibrate():
    cam = RealSenseManager()
    cam.start()
    
    frames = cam.get_frames()
    if frames:
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        c_profile = color_frame.profile.as_video_stream_profile()
        d_profile = depth_frame.profile.as_video_stream_profile()
        
        print("--- Color Intrinsics ---")
        print(c_profile.intrinsics)
        
        print("--- Depth Intrinsics ---")
        print(d_profile.intrinsics)
        
    cam.stop()

if __name__ == "__main__":
    calibrate()
