import pyrealsense2 as rs
from src.utils.logger import get_logger

logger = get_logger("FrameAlignment")

class FrameAligner:
    """
    Aligns depth frames to color frames.
    """
    def __init__(self):
        # We align depth to color so that our YOLO bounding boxes (on color) 
        # exactly match the depth map coordinates.
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        logger.info("Initialized Depth-to-Color alignment.")

    def process(self, frames):
        """
        Takes a raw frameset and returns aligned frames.
        """
        aligned_frames = self.align.process(frames)
        return aligned_frames
