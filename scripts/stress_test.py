import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import time
import sys

def main():
    print("==========================================")
    print(" Agri-Robot FPS Stress Test (Headless)")
    print("==========================================")
    
    model_path = "outputs/experiments/exp005/weights/best.pt"
    print(f"Loading YOLOv8 Model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
    
    print("Initializing Intel RealSense Camera...")
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Set to 30 FPS cap
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) 
    
    try:
        pipeline.start(config)
        print("\n✅ Camera Connected. Starting Stress Test...")
        print("Running WITHOUT video rendering to maximize CPU performance.")
        print("Press Ctrl+C in the terminal to stop the test.\n")
        
        frames_processed = 0
        start_time = time.time()
        last_print_time = start_time
        
        while True:
            # Grab frame from hardware
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
                
            color_image = np.asanyarray(color_frame.get_data())
            
            # Run inference (no rendering, no plotting)
            results = model(color_image, conf=0.25, verbose=False)
            
            frames_processed += 1
            current_time = time.time()
            
            # Print FPS every 2 seconds
            if current_time - last_print_time >= 2.0:
                elapsed = current_time - start_time
                avg_fps = frames_processed / elapsed
                print(f"⚡ Stress Test Active -> System is currently running at: {avg_fps:.2f} Frames Per Second (FPS)")
                last_print_time = current_time
                
    except KeyboardInterrupt:
        print("\nStress Test Stopped by User.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        print("Cleaning up camera...")
        pipeline.stop()

if __name__ == "__main__":
    main()
