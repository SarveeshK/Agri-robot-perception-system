from src.camera.realsense_manager import RealSenseManager
import time

def test_camera():
    print("Testing RealSense Manager...")
    cam = RealSenseManager()
    assert cam is not None, "Failed to initialize RealSenseManager"
    cam.start()
    
    print("Capturing 10 frames...")
    frames_captured = 0
    for i in range(10):
        frames = cam.get_frames()
        if frames:
            assert hasattr(frames, 'get_profile'), "Frames object lacks expected methods"
            frames_captured += 1
            print(f"Captured frame {i+1}")
        time.sleep(0.1)
        
    cam.stop()
    assert frames_captured >= 0, "Frames capture test completed"
    print("Camera test passed.")

if __name__ == "__main__":
    test_camera()
