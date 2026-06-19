# Software Requirements Specification (SRS)

## 1. Introduction
### 1.1 Purpose
The purpose of this document is to outline the software requirements for the **AgriVision Perception System (Phase 1)**. This system is designed to provide an autonomous agricultural robot with real-time visual and depth understanding of its environment.

### 1.2 Intended Audience
This document is intended for software engineers, robotics researchers, and project stakeholders involved in the development and integration of the AgriVision robot platform.

### 1.3 Project Scope
Phase 1 focuses exclusively on the perception pipeline. It integrates an Intel RealSense D456 depth camera with the YOLOv8 object detection model. The software captures synchronized RGB and Depth frames, detects specified objects, computes their 3D real-world measurements (distance, width, height), and provides immediate proximity warnings. Navigation, decision-making, and robot kinematics are out of scope for Phase 1.

---

## 2. Overall Description
### 2.1 Product Perspective
The AgriVision Perception System is an independent, modular software component. It acts as the "eyes" of the agricultural robot, providing structured 3D spatial data about detected obstacles to be consumed by a future decision engine (Phase 2).

### 2.2 Operating Environment
- **OS**: Linux (Ubuntu recommended).
- **Hardware**: x86 or ARM architecture with USB 3.0 support.
- **Sensor**: Intel RealSense D456.
- **Language**: Python 3.12+.

### 2.3 Design Principles
- **Single Responsibility Principle**: Modules are decoupled (Camera, Perception, Utils).
- **Configuration over Code**: No hardcoded parameters; all limits and thresholds are defined in `settings.yaml`.
- **Stateless Execution**: The pipeline processes frames independently (with minor EMA smoothing) ensuring fault tolerance.

---

## 3. System Features

### 3.1 Synchronized Stream Acquisition
- **Description**: The system shall interface with the RealSense D456 to capture RGB and Depth frames.
- **Requirement**: Streams must be hardware-synchronized and running at 30 FPS.
- **Requirement**: The depth frame must be geometrically aligned to the color frame perspective.

### 3.2 Close-Range Depth Optimization
- **Description**: The system must overcome the physical minimum depth limit of the D456.
- **Requirement**: The system shall utilize the RealSense Advanced Mode API to adjust `disparity_shift` to sense objects closer than 40cm.
- **Requirement**: Hardware post-processing filters (Decimation, Spatial, Temporal, Hole Filling) must be applied to the depth stream to eliminate zero-depth noise.

### 3.3 Real-Time Object Detection
- **Description**: The system shall detect specific agricultural and obstacle classes.
- **Requirement**: Utilize YOLOv8 nano (`yolov8n.pt`) optimized for low-latency inference.
- **Requirement**: Filter detections strictly based on a configurable confidence threshold and a whitelist of allowed classes (e.g., humans, animals, plants, machinery).

### 3.4 Robust 3D Measurement
- **Description**: The system shall compute the real-world distance, width, and height of detected objects.
- **Requirement**: Depth sampling must be constrained to the central 50% Region of Interest (ROI) of the bounding box to prevent background bleed.
- **Requirement**: The system must apply percentile filtering (e.g., 10th percentile) to determine the closest robust surface of the object.
- **Requirement**: The system shall use the camera's intrinsic calibration matrix to deproject 2D pixels into 3D space to calculate absolute width and height in centimeters.

### 3.5 Global Collision Warning
- **Description**: The system shall alert the user/operator to impending collisions.
- **Requirement**: If any detected object's distance falls below the configurable `safety_distance_cm` limit, the system must trigger a global "DON'T GO" state.

---

## 4. Non-Functional Requirements

### 4.1 Performance Requirements
- **Target Framerate**: The pipeline must operate between 25 and 30 FPS.
- **Detection Latency**: Total pipeline latency (frame capture to output) must be < 100 ms.
- **Depth Accuracy**: Depth measurements must fall within a ±2 cm error margin.
- **Resource Utilization**: CPU usage must remain below 70% during continuous operation.

### 4.2 Reliability & Fault Tolerance
- **Requirement**: The system must not crash upon dropped frames. It must log a warning and continue to the next frame.
- **Requirement**: Invalid (zero) depth pixels must be discarded and must not crash the measurement mathematics.

### 4.3 Maintainability
- **Requirement**: All modules must utilize the unified `src.utils.logger` for debugging and traceability.
- **Requirement**: Performance profiling (`src.utils.profiler`) must wrap the execution pipeline to provide real-time latency feedback.
