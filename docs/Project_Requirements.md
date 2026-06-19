# Project Requirements

## Goal
Build a modular Real-Time Perception System for an agricultural robot.

## Scope (Phase 1)
- Capture RGB and Depth frames from Intel RealSense D456.
- Detect objects using YOLOv8.
- Measure the object's distance, width, and height.
- Convert image coordinates into real-world coordinates.
- Optimize close-range depth sensing.
- Whole-frame obstacle detection and warnings.
- Keep the system modular for future Phase 2 integration (Robot Controller, Tracking, etc).

## Performance Goals
- **Target FPS**: 25-30 FPS
- **Detection Latency**: <100 ms
- **Depth Error**: ±2 cm
- **CPU Usage**: <70%
- **Real-time Processing**: Yes
