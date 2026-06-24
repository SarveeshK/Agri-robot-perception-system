"""
Module for calculating 3D physical measurements of detected objects.
"""
from src.utils.logger import get_logger
from src.utils.config import settings

logger = get_logger("Measurement")

class ObjectMeasurement:
    """
    Computes 3D width, height, and distance, using EMA for temporal smoothing.
    """
    def __init__(self, calibration):
        """
        Initializes the measurement module with camera calibration.
        
        Args:
            calibration (CameraCalibration): The calibration object for deprojection.
        """
        self.calibration = calibration
        self.smoothing_alpha = settings.measurement.get("smoothing_alpha", 0.3)
        
        # Track history for EMA smoothing: {object_id: (dist, w, h)}
        # For now, we smooth based on center matching since we don't have tracking (Sprint 5)
        self.history = []
        
    def measure(self, box, distance_cm):
        """
        Calculates Width and Height in cm given the 2D bounding box and depth.
        
        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            distance_cm (float): The measured distance to the object in cm.
            
        Returns:
            tuple: A tuple containing (width_cm, height_cm).
        """
        if distance_cm <= 0:
            return 0.0, 0.0
            
        x1, y1, x2, y2 = box
        distance_m = distance_cm / 100.0
        
        # Deproject top-left and bottom-right points
        pt1 = self.calibration.deproject_pixel_to_point(x1, y1, distance_m)
        pt2 = self.calibration.deproject_pixel_to_point(x2, y2, distance_m)
        
        # Calculate 3D width and height
        # pt1 = [X1, Y1, Z], pt2 = [X2, Y2, Z]
        width_cm = abs(pt1[0] - pt2[0]) * 100.0
        height_cm = abs(pt1[1] - pt2[1]) * 100.0
        
        return width_cm, height_cm
