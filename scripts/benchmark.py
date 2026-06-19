import time
from src.camera.realsense_manager import RealSenseManager

def benchmark():
    cam = RealSenseManager()
    cam.start()
    
    start_time = time.time()
    frames_grabbed = 0
    while time.time() - start_time < 5.0:
        if cam.get_frames():
            frames_grabbed += 1
            
    print(f"Grabbed {frames_grabbed} frames in 5 seconds ({(frames_grabbed/5.0):.2f} FPS limit)")
    cam.stop()

if __name__ == "__main__":
    benchmark()
