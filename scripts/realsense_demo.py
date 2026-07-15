import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import sys

def main():
    print("==========================================")
    print(" Agri-Robot Dual-Model Ensemble Demo")
    print("==========================================")
    
    # 1. Load your custom agricultural model
    model_path = "outputs/experiments/exp005/weights/best.pt"
    print(f"Loading Custom YOLOv8 Model from {model_path}...")
    try:
        model_agri = YOLO(model_path)
    except Exception as e:
        print(f"Failed to load custom model: {e}")
        sys.exit(1)
        
    # 2. Load the base COCO model for cars, trucks, and animals
    print(f"Loading Base YOLOv8 Nano for Cars/Animals...")
    model_coco = YOLO("yolov8n.pt")
    
    # COCO Class Indices for relevant agricultural/road obstacles:
    # 2: car, 3: motorcycle, 5: bus, 7: truck
    # 15: bird, 16: cat, 17: dog, 18: horse, 19: sheep, 20: cow
    coco_filter_classes = [2, 3, 5, 7, 15, 16, 17, 18, 19, 20]
    
    print("Initializing Intel RealSense Camera...")
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Force RealSense to give us HD resolution
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    
    try:
        pipeline.start(config)
        print("\n✅ RealSense Camera Connected!")
        print("⚡ Dual-Model Ensemble Active. FPS will drop to ~10-15.")
        print("👉 Press 'q' or 'ESC' to close the video window.\n")
        
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
                
            color_image = np.asanyarray(color_frame.get_data())
            
            # Run Inference on Custom Model (Rocks, Trees, Fences, etc.)
            res_agri = model_agri(color_image, conf=0.25, verbose=False)
            
            # Run Inference on Base Model (Filtered to Cars and Farm Animals only)
            res_coco = model_coco(color_image, conf=0.25, classes=coco_filter_classes, verbose=False)
            
            # Plot custom agricultural boxes first
            annotated_frame = res_agri[0].plot()
            
            # Clever trick: Overwrite the base image of the COCO results so it draws ON TOP of the first boxes
            res_coco[0].orig_img = annotated_frame
            final_frame = res_coco[0].plot()
            
            # Pop open the video window
            cv2.imshow("Agri-Robot Dual-Model Perception", final_frame)
            
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
