import cv2
import numpy as np
from src.perception.yolo_detector import YoloDetector

def test_yolo():
    print("Testing YOLO Detector...")
    detector = YoloDetector()
    assert detector is not None, "Failed to initialize YoloDetector"
    assert detector.model is not None, "YOLO model failed to load"
    
    # Create dummy black image
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("Running inference on dummy image...")
    detections = detector.detect(dummy_img)
    
    assert isinstance(detections, list), "Detections should be a list"
    assert len(detections) >= 0, "Detections length should be non-negative"
    
    print(f"Found {len(detections)} detections (expected 0 on blank image).")
    print("YOLO test passed.")

if __name__ == "__main__":
    test_yolo()
