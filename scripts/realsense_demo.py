import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import sys

def main():
    print("==========================================")
    print(" Agri-Robot RealSense Live Demo")
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
    
    # Force RealSense to give us the Color feed (ignoring Infrared/Depth for this demo)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        pipeline.start(config)
        print("\n✅ RealSense Camera Connected and Streaming!")
        print("👉 Press 'q' or 'ESC' on your keyboard to close the video window.\n")
        
        while True:
            # Grab frames from the RealSense hardware
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
                
            # Convert raw RealSense data into a standard OpenCV image array
            color_image = np.asanyarray(color_frame.get_data())
            
            # Run YOLOv8 inference (verbose=False keeps the terminal clean)
            # Lowered conf to 0.25 to detect harder objects in harsh lighting
            results = model(color_image, conf=0.25, verbose=False)
            
            # Draw the colorful bounding boxes on the image
            annotated_frame = results[0].plot()
            
            # Pop open the video window
            cv2.imshow("Agri-Robot Perception (Intel RealSense)", annotated_frame)
            
            # Break loop if user presses 'q'
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                break
                
    except RuntimeError as e:
        print(f"\n❌ Hardware Error: {e}")
        print("Please ensure the RealSense camera is securely plugged into a blue USB 3.0 port.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        print("Stopping camera and cleaning up...")
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
