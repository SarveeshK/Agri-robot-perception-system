import pyrealsense2 as rs
from src.utils.logger import get_logger

logger = get_logger("Calibration")

class CameraCalibration:
    """
    Retrieves and holds camera intrinsics for 3D deprojection.
    """
    def __init__(self):
        self.color_intrinsics = None
        
    def update_intrinsics(self, color_frame):
        """
        Update intrinsics from the current color frame.
        Since we align depth to color, we use the color stream's intrinsics 
        to deproject aligned depth pixels.
        """
        profile = color_frame.profile.as_video_stream_profile()
        self.color_intrinsics = profile.intrinsics

    def deproject_pixel_to_point(self, pixel_x, pixel_y, depth):
        """
        Convert a 2D pixel (x, y) and depth (in meters) to a 3D point (X, Y, Z).
        Returns a list of 3 floats [X, Y, Z] in meters.
        """
        if not self.color_intrinsics:
            logger.warning("Intrinsics not set, cannot deproject.")
            return [0.0, 0.0, 0.0]
            
        point_3d = rs.rs2_deproject_pixel_to_point(self.color_intrinsics, [pixel_x, pixel_y], depth)
        return point_3d
