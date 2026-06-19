import cv2
import numpy as np

from src.utils.logger import get_logger
from src.utils.profiler import Profiler
from src.camera.realsense_manager import RealSenseManager
from src.camera.frame_alignment import FrameAligner
from src.camera.calibration import CameraCalibration
from src.perception.yolo_detector import YoloDetector
from src.perception.depth_processor import DepthProcessor
from src.perception.measurement import ObjectMeasurement
from src.perception.visualization import Visualizer

logger = get_logger("MainPipeline")

def main():
    logger.info("Starting AgriVision Perception System Phase 1...")
    
    # Initialize Utilities
    profiler = Profiler()
    
    # Initialize Camera Modules
    camera = RealSenseManager()
    camera.start()
    aligner = FrameAligner()
    calibration = CameraCalibration()
    
    # Initialize Perception Modules
    detector = YoloDetector()
    depth_proc = DepthProcessor()
    measurer = ObjectMeasurement(calibration)
    visualizer = Visualizer()
    
    try:
        while True:
            profiler.start("pipeline")
            
            # 1. Capture Frames
            frames = camera.get_frames()
            if not frames:
                continue
                
            # 2. Align Frames
            aligned_frames = aligner.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
                
            # Update camera calibration intrinsics once
            if not calibration.color_intrinsics:
                calibration.update_intrinsics(color_frame)
                
            # Apply post-processing filters to depth
            filtered_depth_frame = camera.apply_filters(depth_frame)
            filtered_depth_frame = filtered_depth_frame.as_depth_frame()
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_width = filtered_depth_frame.get_width()
            depth_height = filtered_depth_frame.get_height()
            
            # 3. YOLO Detection
            detections = detector.detect(color_image)
            
            # 4. Depth Processing & Object Measurement
            for det in detections:
                dist_cm, valid_samples = depth_proc.extract_robust_depth(
                    filtered_depth_frame, 
                    det["box"], 
                    depth_width, 
                    depth_height
                )
                
                if dist_cm > 0:
                    width_cm, height_cm = measurer.measure(det["box"], dist_cm)
                    det["measurement"] = (dist_cm, width_cm, height_cm)
                    det["valid_samples"] = valid_samples
            
            profiler.stop("pipeline")
            profiler.tick()
            
            # 5. Visualization
            output_image = visualizer.draw(color_image, detections, profiler.get_fps())
            
            cv2.imshow("AgriVision Perception", output_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Shutdown requested.")
                break
                
    except Exception as e:
        logger.error(f"Pipeline crashed: {e}", exc_info=True)
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
