"""
Module for aligning depth frames to color frames.
"""
import pyrealsense2 as rs
from src.utils.logger import get_logger

logger = get_logger("FrameAlignment")

class FrameAligner:
    """
    Aligns depth frames to color frames.
    """
    def __init__(self):
        """
        Initializes the aligner to map depth frames to color frame coordinates.
        """
        # We align depth to color so that our YOLO bounding boxes (on color) 
        # exactly match the depth map coordinates.
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        logger.info("Initialized Depth-to-Color alignment.")

    def process(self, frames):
        """
        Align the depth frames to color frames.
        
        Args:
            frames (rs.composite_frame): The raw frameset from the camera.
            
        Returns:
            rs.composite_frame: The aligned frameset.
        """
        aligned_frames = self.align.process(frames)
        return aligned_frames
