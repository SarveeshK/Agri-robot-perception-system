from src.camera.realsense_manager import RealSenseManager
import time

def test_camera():
    print("Testing RealSense Manager...")
    cam = RealSenseManager()
    cam.start()
    
    print("Capturing 10 frames...")
    for i in range(10):
        frames = cam.get_frames()
        if frames:
            print(f"Captured frame {i+1}")
        time.sleep(0.1)
        
    cam.stop()
    print("Camera test passed.")

if __name__ == "__main__":
    test_camera()
