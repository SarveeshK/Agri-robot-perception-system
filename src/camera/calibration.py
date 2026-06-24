"""
Module for handling camera calibration and intrinsics for 3D deprojection.
"""
import pyrealsense2 as rs
from src.utils.logger import get_logger

logger = get_logger("Calibration")

class CameraCalibration:
    """
    Retrieves and holds camera intrinsics for 3D deprojection.
    """
    def __init__(self):
        """
        Initializes the camera calibration object.
        """
        self.color_intrinsics = None
        
    def update_intrinsics(self, color_frame):
        """
        Update camera intrinsics from the current color frame.
        Since we align depth to color, we use the color stream's intrinsics 
        to deproject aligned depth pixels.
        
        Args:
            color_frame (rs.video_frame): The color frame to extract intrinsics from.
        """
        profile = color_frame.profile.as_video_stream_profile()
        self.color_intrinsics = profile.intrinsics

    def deproject_pixel_to_point(self, pixel_x, pixel_y, depth):
        """
        Convert a 2D pixel coordinate and depth measurement into a 3D point.
        
        Args:
            pixel_x (float): The x-coordinate of the pixel.
            pixel_y (float): The y-coordinate of the pixel.
            depth (float): The depth distance in meters.
            
        Returns:
            list[float]: A list containing the 3D coordinates [X, Y, Z] in meters.
        """
        if not self.color_intrinsics:
            logger.warning("Intrinsics not set, cannot deproject.")
            return [0.0, 0.0, 0.0]
            
        point_3d = rs.rs2_deproject_pixel_to_point(self.color_intrinsics, [pixel_x, pixel_y], depth)
        return point_3d
