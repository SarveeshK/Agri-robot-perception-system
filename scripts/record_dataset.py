import cv2
import numpy as np
import time
import os
from src.camera.realsense_manager import RealSenseManager
from src.camera.frame_alignment import FrameAligner

def record():
    cam = RealSenseManager()
    cam.start()
    aligner = FrameAligner()
    
    save_dir = "data/recordings"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Press 's' to save a frame. Saved to {save_dir}")
    print("Press 'q' to quit.")
    
    count = 0
    try:
        while True:
            frames = cam.get_frames()
            if not frames:
                continue
                
            aligned = aligner.process(frames)
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()
            
            if not color or not depth:
                continue
                
            c_img = np.asanyarray(color.get_data())
            d_img = np.asanyarray(depth.get_data())
            
            # Apply color map to depth for visualization
            d_colormap = cv2.applyColorMap(cv2.convertScaleAbs(d_img, alpha=0.03), cv2.COLORMAP_JET)
            
            cv2.imshow("RGB", c_img)
            cv2.imshow("Depth", d_colormap)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                cv2.imwrite(f"{save_dir}/rgb_{count}.jpg", c_img)
                np.save(f"{save_dir}/depth_{count}.npy", d_img)
                print(f"Saved frame {count}")
                count += 1
            elif key == ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    record()
