import cv2
import yaml
import time
import sys
import os
from enum import Enum
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

class State(Enum):
    IDLE = 1
    MONITORING = 2
    DETECTING = 3
    CONFIRM_ACTION = 4
    ACTING = 5
    RECOVERING = 6

class ConfigLoader:
    def __init__(self, config_path="config/proximity.yaml"):
        if not os.path.exists(config_path):
            print(f"Error: {config_path} not found.")
            sys.exit(1)
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

class SensorNode:
    """Handles all hardware interfacing with the Intel RealSense camera"""
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # Laptop Safe Mode: Scaled down from 720p 30FPS to 480p 15FPS to prevent thermal throttling
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        self.align = rs.align(rs.stream.color)
        
    def start(self):
        self.pipeline.start(self.config)
        
    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color = aligned_frames.get_color_frame()
        depth = aligned_frames.get_depth_frame()
        if not color or not depth:
            return None, None
        return np.asanyarray(color.get_data()), depth

class FilterNode:
    """Processes depth maps into proximal clusters"""
    def __init__(self, config):
        self.cfg = config
        self.min_pixels = self.cfg['filters']['minimum_cluster_pixels']
        
    def check_zone(self, depth_frame):
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Convert config meters to millimeters for RealSense logic
        wake_mm = self.cfg['zones']['wake_distance'] * 1000
        stop_mm = self.cfg['zones']['stop_distance'] * 1000
        resume_mm = self.cfg['zones']['resume_distance'] * 1000
        
        valid_depth = depth_image > 0
        
        if np.sum((depth_image < stop_mm) & valid_depth) > self.min_pixels:
            return "DANGER"
        elif np.sum((depth_image < wake_mm) & valid_depth) > self.min_pixels:
            return "MONITOR"
        elif np.sum((depth_image > resume_mm) & valid_depth) > self.min_pixels:
            return "SAFE"
        return "UNKNOWN"

class PerceptionNode:
    """Wrapper for the YOLO inference engine, kept constantly loaded in RAM"""
    def __init__(self, model_path="outputs/experiments/exp005/weights/best.pt"):
        self.model = YOLO(model_path)
        
    def infer(self, color_image):
        return self.model(color_image, verbose=False)[0]

class DecisionNode:
    """Maps YOLO classes to Fused Threat/Action profiles defined in YAML"""
    def __init__(self, config):
        self.mapping = config.get('action_mapping', {})
        self.conf_cfg = config.get('confidence', {})
        
    def get_action(self, results):
        highest_threat_action = None
        highest_threat_level = 0
        
        threat_scores = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
        
        for box in results.boxes:
            cls_index = int(box.cls[0])
            if cls_index >= len(results.names): continue
            cls_name = results.names[cls_index].lower()
            conf = float(box.conf[0])
            
            if cls_name not in self.mapping:
                continue
                
            obj_map = self.mapping[cls_name]
            threat = obj_map.get('threat', 'Low')
            action = obj_map.get('action', 'Ignore')
            
            req_conf = self.conf_cfg.get(cls_name, self.conf_cfg.get('obstacle', 0.5))
            if threat == "Critical": req_conf = self.conf_cfg.get('human', 0.4)
            
            if conf >= req_conf:
                score = threat_scores.get(threat, 0)
                if score > highest_threat_level:
                    highest_threat_level = score
                    highest_threat_action = action
                    
        return highest_threat_action

class VisualDebugger:
    """Renders the diagnostic HUD"""
    @staticmethod
    def draw(image, state, action, fps):
        # Background box for readability
        cv2.rectangle(image, (20, 20), (500, 160), (0, 0, 0), -1)
        
        color = (0, 255, 0) # Green for IDLE
        if state in [State.MONITORING, State.CONFIRM_ACTION, State.RECOVERING]: color = (0, 255, 255) # Yellow
        elif state in [State.DETECTING, State.ACTING]: color = (0, 0, 255) # Red
        
        cv2.putText(image, f"STATE: {state.name}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(image, f"ACTION: {action}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f"FPS: {fps:.1f}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return image

def main():
    cfg = ConfigLoader().cfg
    sensor = SensorNode()
    filt = FilterNode(cfg)
    perception = PerceptionNode()
    decision = DecisionNode(cfg)
    
    state = State.IDLE
    sensor.start()
    
    action_buffer = []
    frames_processed = 0
    start_time = time.time()
    last_recovery_time = 0
    last_log_state = None
    last_valid_action = "NONE"
    
    print("✅ Proximity Perception Pipeline Active...")
    
    while True:
        color, depth = sensor.get_frames()
        if color is None: continue
        
        zone = filt.check_zone(depth)
        
        frames_processed += 1
        fps = frames_processed / (time.time() - start_time)
        
        # State Machine Logic
        if state in [State.IDLE, State.MONITORING]:
            last_valid_action = "NONE"
            
        if state == State.IDLE:
            if zone == "MONITOR": state = State.MONITORING
            elif zone == "DANGER": state = State.DETECTING
                
        elif state == State.MONITORING:
            if zone == "SAFE": state = State.IDLE
            elif zone == "DANGER": state = State.DETECTING
            
        elif state == State.DETECTING:
            results = perception.infer(color)
            color = results.plot() 
            proposed_action = decision.get_action(results)
            
            if proposed_action:
                action_buffer.append(proposed_action)
                if len(action_buffer) >= cfg['timing']['confirm_stop_frames']:
                    state = State.ACTING
                    last_valid_action = action_buffer[-1]
            else:
                action_buffer.clear()
                if zone != "DANGER":
                    state = State.RECOVERING
                    last_recovery_time = time.time()
                    
        elif state == State.ACTING:
            results = perception.infer(color)
            color = results.plot()
            proposed_action = decision.get_action(results)
            if proposed_action:
                last_valid_action = proposed_action
            
            if zone != "DANGER":
                state = State.RECOVERING
                last_recovery_time = time.time()
                action_buffer.clear()
                
        elif state == State.RECOVERING:
            if time.time() - last_recovery_time > cfg['timing']['recovery_time_seconds']:
                if zone == "SAFE": state = State.IDLE
                elif zone == "MONITOR": state = State.MONITORING
                else: state = State.DETECTING
        
        # Event Logging
        if state != last_log_state:
            print(f"[EVENT] State Transitioned -> {state.name}")
            if state == State.ACTING:
                print(f"[EVENT] Executing Motor Command -> {last_valid_action}")
            last_log_state = state
                    
        # Render HUD
        color = VisualDebugger.draw(color, state, last_valid_action, fps)
        
        cv2.imshow("Proximity Pipeline (Jetson v1.2)", color)
        if cv2.waitKey(1) == 27:
            break

if __name__ == "__main__":
    main()
