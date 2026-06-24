"""
Module for rendering HUD and overlays on the camera feed.
"""
import cv2
from src.utils.config import settings

class Visualizer:
    """
    Renders bounding boxes, measurements, FPS, and collision warnings.
    """
    def __init__(self):
        """
        Initializes visualizer with settings from configuration.
        """
        self.show_fps = settings.visualization.get("show_fps", True)
        self.safety_distance_cm = settings.obstacle.get("safety_distance_cm", 50.0)

    def draw(self, image, detections, fps=None):
        """
        Draws bounding boxes, labels, and HUD overlays on the image.
        
        Args:
            image (numpy.ndarray): The RGB image frame to draw on.
            detections (list[dict]): Detections containing bounding boxes and optional measurements.
            fps (float, optional): The current frames per second to display.
            
        Returns:
            numpy.ndarray: The image with all visual overlays.
        """
        collision_warning = False
        nearest_distance = float('inf')
        nearest_class = "None"

        # Find the nearest obstacle anywhere in the frame
        for det in detections:
            meas = det.get("measurement")
            if meas:
                dist, _, _ = meas
                if dist > 0 and dist < nearest_distance:
                    nearest_distance = dist
                    nearest_class = det["class_name"]

        if nearest_distance <= self.safety_distance_cm:
            collision_warning = True

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            meas = det.get("measurement")
            
            # Highlight the nearest object in red
            is_nearest = (meas and meas[0] == nearest_distance)
            color = (0, 0, 255) if is_nearest else (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw labels
            label1 = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(image, label1, (x1, max(20, y1 - 25)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if meas:
                dist, w, h = meas
                if dist > 0:
                    label2 = f"D:{dist:.1f}cm W:{w:.1f}cm H:{h:.1f}cm"
                    cv2.putText(image, label2, (x1, max(40, y1 - 5)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Draw Global HUD panel
        self._draw_hud(image, nearest_class, nearest_distance, collision_warning, fps)
        
        return image

    def _draw_hud(self, image, nearest_class, nearest_distance, collision_warning, fps):
        """
        Helper method to draw the Global HUD panel.
        
        Args:
            image (numpy.ndarray): The image frame.
            nearest_class (str): Class name of the nearest object.
            nearest_distance (float): Distance to the nearest object in cm.
            collision_warning (bool): Whether a collision warning should be displayed.
            fps (float): Current FPS to display.
        """
        panel_color = (255, 255, 255)
        
        if fps and self.show_fps:
            cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, panel_color, 2)

        # Proximity info
        dist_text = f"{nearest_distance:.1f} cm" if nearest_distance != float('inf') else "--"
        
        cv2.putText(image, f"Nearest: {nearest_class}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, panel_color, 2)
        cv2.putText(image, f"Distance: {dist_text}", (10, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, panel_color, 2)

        # Collision warning irrespective of frame location
        if collision_warning:
            cv2.putText(image, "DON'T GO", (image.shape[1]//2 - 100, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            cv2.putText(image, "GO", (image.shape[1]//2 - 30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
