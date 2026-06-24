"""
Module for object detection using YOLOv8 models.
"""
import torch
from ultralytics import YOLO
from src.utils.logger import get_logger
from src.utils.config import settings

logger = get_logger("YoloDetector")

class YoloDetector:
    """
    Loads YOLOv8 and performs object detection on RGB frames.
    Filters out predictions based on confidence and allowed classes.
    """
    def __init__(self):
        """
        Initializes the YOLO model with configuration settings.
        """
        yolo_conf = settings.yolo
        self.model_path = yolo_conf.get("model_path", "models/yolov8n.pt")
        self.confidence_thresh = yolo_conf.get("confidence", 0.60)
        self.allowed_classes = yolo_conf.get("allowed_classes", [])
        
        logger.info(f"Loading YOLO model from {self.model_path}")
        self.model = YOLO(self.model_path)
        logger.info("YOLO model loaded successfully.")

    def detect(self, color_image):
        """
        Runs inference and returns filtered bounding boxes.
        
        Args:
            color_image (numpy.ndarray): The RGB image frame.
            
        Returns:
            list[dict]: A list of detections, where each detection is a dictionary
            containing 'class_name', 'class_id', 'confidence', and 'box' (x1, y1, x2, y2).
        """
        # Inference (with torch.inference_mode for speed)
        with torch.inference_mode():
            results = self.model(color_image, verbose=False)
            
        result = results[0]
        detections = []
        
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < self.confidence_thresh:
                continue
                
            class_id = int(box.cls[0])
            if self.allowed_classes and class_id not in self.allowed_classes:
                continue
                
            class_name = result.names[class_id]
            
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Avoid degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue
                
            detections.append({
                "class_name": class_name,
                "class_id": class_id,
                "confidence": confidence,
                "box": (x1, y1, x2, y2)
            })
            
        return detections
