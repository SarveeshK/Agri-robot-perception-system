"""
Module for processing depth frames and extracting robust 3D measurements.
"""
import numpy as np
from src.utils.logger import get_logger
from src.utils.config import settings

logger = get_logger("DepthProcessor")

class DepthProcessor:
    """
    Optimizes depth extraction by carefully sampling depth values
    inside bounding boxes to avoid background bleed and noise.
    """
    def __init__(self):
        """
        Initializes the DepthProcessor using settings from the configuration.
        """
        meas_conf = settings.measurement
        self.roi_ratio = meas_conf.get("sampling_roi_ratio", 0.5)
        self.depth_percentile = meas_conf.get("depth_percentile", 10)
        self.min_samples = meas_conf.get("min_valid_samples", 5)

    def _validate_depth(self, depth_value_mm, valid_pixels, total_pixels):
        """
        Check if depth is in valid range and has enough confidence.
        
        Args:
            depth_value_mm (float): Measured depth in millimeters.
            valid_pixels (int): Count of valid depth pixels.
            total_pixels (int): Total pixels sampled.
            
        Returns:
            tuple: (valid_depth_mm or None, valid_ratio)
        """
        MIN_DEPTH = 100  # 10cm
        MAX_DEPTH = 3000 # 3m
        
        valid_ratio = valid_pixels / total_pixels if total_pixels > 0 else 0.0
        
        if depth_value_mm < MIN_DEPTH or depth_value_mm > MAX_DEPTH:
            return None, valid_ratio
            
        if valid_ratio < 0.5:
            return None, valid_ratio
            
        return depth_value_mm, valid_ratio

    def extract_robust_depth(self, depth_frame, box, depth_width, depth_height):
        """
        Given a bounding box (x1, y1, x2, y2), extracts a robust distance in cm.
        Uses a central ROI to sample depth and calculates a percentile.
        
        Args:
            depth_frame: The aligned depth frame from RealSense.
            box (tuple): Bounding box coordinates (x1, y1, x2, y2).
            depth_width (int): Width of the depth frame.
            depth_height (int): Height of the depth frame.
            
        Returns:
            tuple: (distance_cm, valid_samples_count)
        """
        x1, y1, x2, y2 = box
        
        # Calculate box dimensions
        w = x2 - x1
        h = y2 - y1
        
        # Shrink box to central ROI (e.g. inner 50%) to avoid edges
        shrink_x = int(w * (1.0 - self.roi_ratio) / 2.0)
        shrink_y = int(h * (1.0 - self.roi_ratio) / 2.0)
        
        roi_x1 = max(0, x1 + shrink_x)
        roi_y1 = max(0, y1 + shrink_y)
        roi_x2 = min(depth_width - 1, x2 - shrink_x)
        roi_y2 = min(depth_height - 1, y2 - shrink_y)
        
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return 0.0, 0
            
        # Sample points inside the ROI
        sample_xs = np.linspace(roi_x1, roi_x2, 5, dtype=int)
        sample_ys = np.linspace(roi_y1, roi_y2, 5, dtype=int)
        
        sample_depths = []
        
        # For each sample point, look at a small 3x3 patch for local robustness
        for sample_y in sample_ys:
            for sample_x in sample_xs:
                local_depths = []
                
                for yy in range(sample_y - 1, sample_y + 2):
                    for xx in range(sample_x - 1, sample_x + 2):
                        if xx < 0 or xx >= depth_width or yy < 0 or yy >= depth_height:
                            continue
                            
                        d = depth_frame.get_distance(xx, yy)
                        if d > 0: # Valid depth
                            local_depths.append(d)
                            
                if local_depths:
                    # Median of the local patch
                    sample_depths.append(float(np.median(local_depths)))
                    
        if len(sample_depths) < self.min_samples:
            return 0.0, len(sample_depths)
            
        # Percentile filter over all patch medians 
        # (10th percentile gets the closest robust part of the object)
        robust_distance_m = np.percentile(sample_depths, self.depth_percentile)
        
        return float(robust_distance_m * 100.0), len(sample_depths)
