import pyrealsense2 as rs
import numpy as np
from src.utils.logger import get_logger
from src.utils.config import settings

logger = get_logger("RealSenseManager")

class RealSenseManager:
    """
    Manages the Intel RealSense D456 camera connection, configuration,
    and post-processing filters.
    """
    def __init__(self):
        """
        Initialize the RealSense Manager and configure data streams.
        """
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.profile = None
        self.depth_sensor = None
        
        self.filters = []
        self._setup_streams()
        
    def _setup_streams(self):
        cam_conf = settings.camera
        w, h, fps = cam_conf.get('width', 640), cam_conf.get('height', 480), cam_conf.get('fps', 30)
        
        self.config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        logger.info(f"Configured streams: RGB {w}x{h} @ {fps}fps, Depth {w}x{h} @ {fps}fps")

    def start(self):
        """
        Start the RealSense pipeline and apply advanced settings and filters.
        """
        try:
            self.profile = self.pipeline.start(self.config)
            self.depth_sensor = self.profile.get_device().first_depth_sensor()
            logger.info("RealSense pipeline started successfully.")
            
            self._apply_advanced_settings()
            self._setup_post_processing()
            
        except Exception as e:
            logger.error(f"Failed to start RealSense pipeline: {e}")
            raise

    def _apply_advanced_settings(self):
        adv_conf = settings.camera.get("advanced", {})
        if not adv_conf:
            return
            
        disparity_shift = adv_conf.get("disparity_shift", 0)
        if disparity_shift > 0:
            try:
                # Disparity shift allows reducing the minimum depth distance
                # by shifting the disparity search range.
                device = self.profile.get_device()
                advnc_mode = rs.rs400_advanced_mode(device)
                
                if advnc_mode.is_enabled():
                    depth_table = advnc_mode.get_depth_table()
                    depth_table.disparityShift = disparity_shift
                    advnc_mode.set_depth_table(depth_table)
                    logger.info(f"Advanced Mode: Set disparity_shift to {disparity_shift} for close-range sensing.")
                else:
                    logger.warning("Advanced mode is not enabled. Cannot set disparity_shift.")
            except Exception as e:
                logger.error(f"Failed to apply advanced settings: {e}")

    def _setup_post_processing(self):
        pp_conf = settings.camera.get("post_processing", {})
        
        if pp_conf.get("decimation", {}).get("enabled", False):
            dec = rs.decimation_filter()
            dec.set_option(rs.option.filter_magnitude, pp_conf["decimation"].get("magnitude", 2))
            self.filters.append(("Decimation", dec))
            
        if pp_conf.get("spatial", {}).get("enabled", True):
            spatial = rs.spatial_filter()
            spatial.set_option(rs.option.filter_magnitude, pp_conf["spatial"].get("magnitude", 2))
            spatial.set_option(rs.option.filter_smooth_alpha, pp_conf["spatial"].get("smooth_alpha", 0.5))
            spatial.set_option(rs.option.filter_smooth_delta, pp_conf["spatial"].get("smooth_delta", 20))
            self.filters.append(("Spatial", spatial))
            
        if pp_conf.get("temporal", {}).get("enabled", True):
            temporal = rs.temporal_filter()
            temporal.set_option(rs.option.filter_smooth_alpha, pp_conf["temporal"].get("smooth_alpha", 0.4))
            temporal.set_option(rs.option.filter_smooth_delta, pp_conf["temporal"].get("smooth_delta", 20))
            self.filters.append(("Temporal", temporal))
            
        if pp_conf.get("hole_filling", {}).get("enabled", True):
            hole_filling = rs.hole_filling_filter()
            hole_filling.set_option(rs.option.holes_fill, pp_conf["hole_filling"].get("mode", 1))
            self.filters.append(("Hole Filling", hole_filling))

        if pp_conf.get("threshold", {}).get("enabled", True):
            threshold = rs.threshold_filter()
            threshold.set_option(rs.option.min_distance, pp_conf["threshold"].get("min_distance", 0.1))
            threshold.set_option(rs.option.max_distance, pp_conf["threshold"].get("max_distance", 3.0))
            # Insert threshold filter first if enabled
            self.filters.insert(0, ("Threshold", threshold))

        logger.info(f"Initialized {len(self.filters)} post-processing filters: {[name for name, _ in self.filters]}")

    def get_frames(self):
        """
        Wait for and retrieve a set of frames from the camera.
        
        Returns:
            rs.composite_frame or None: The captured frames if successful, None otherwise.
        """
        try:
            frames = self.pipeline.wait_for_frames()
            return frames
        except Exception as e:
            logger.warning(f"Error waiting for frames: {e}")
            return None

    def apply_filters(self, depth_frame):
        """
        Apply configured post-processing filters to the given depth frame.
        
        Args:
            depth_frame: The original depth frame.
            
        Returns:
            The filtered depth frame.
        """
        filtered_depth = depth_frame
        for name, filter_obj in self.filters:
            try:
                filtered_depth = filter_obj.process(filtered_depth)
            except Exception as e:
                logger.debug(f"Filter {name} failed: {e}")
        return filtered_depth

    def stop(self):
        """
        Stop the RealSense pipeline and release resources.
        """
        try:
            self.pipeline.stop()
            logger.info("RealSense pipeline stopped.")
        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")
