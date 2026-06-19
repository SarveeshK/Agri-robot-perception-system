import cv2
import numpy as np
from src.perception.yolo_detector import YoloDetector

def test_yolo():
    print("Testing YOLO Detector...")
    detector = YoloDetector()
    
    # Create dummy black image
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("Running inference on dummy image...")
    detections = detector.detect(dummy_img)
    
    print(f"Found {len(detections)} detections (expected 0 on blank image).")
    print("YOLO test passed.")

if __name__ == "__main__":
    test_yolo()
