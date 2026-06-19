# AgriVision Perception System

A modular, real-time perception system designed for an autonomous agricultural robot (Phase 1). This pipeline integrates Intel RealSense D456 streams and YOLOv8 object detection to provide robust real-world 3D object measurements and obstacle collision warnings.

## Features

- **Aligned RGB-Depth Stream**: Synchronized capture and alignment of color and depth streams.
- **YOLOv8 Detection**: High-speed object detection of specific configurable classes.
- **Robust 3D Measurement**: Measures object distance, width, and height in real-world dimensions (cm).
- **Depth Optimization**: Implements RealSense Advanced Mode (disparity shift) and Post-Processing filters (Decimation, Threshold, Spatial, Temporal, Hole Filling) to enhance sensing at close ranges and remove zero-depth noise.
- **Proximity HUD**: Whole-frame obstacle tracking that provides real-time "DON'T GO" warnings if any tracked object violates safety thresholds.

## Current Pipeline Architecture (Phase 1)
```text
Intel RealSense D456
        ↓
RGB Frame + Depth Frame
        ↓
Frame Alignment
        ↓
YOLO Detection
        ↓
Depth Processing
        ↓
Object Measurement
        ↓
Visualization
```

## Setup

Ensure your environment has the required packages:

```bash
pip install -r requirements.txt
```

*(Note: The provided `yolov8n.pt` weights must be placed in `models/` or they will be automatically downloaded).*

## Configuration

All system parameters (camera resolution, fps, YOLO thresholds, smoothing algorithms, safety distance limits) are located in `config/settings.yaml`.
**No hardcoded parameters exist in the source code.**

## Running the System

To start the perception pipeline:

```bash
python src/main.py
```

## Engineering Principles

- **Single Responsibility Principle**: Each module performs only one distinct function.
- **No Hardcoded Parameters**: Every parameter is configurable via `settings.yaml`.
- **Benchmarked Optimization**: All optimizations (especially Depth processing) are profiled for latency.
- **Independence**: The perception module operates completely independent of robot control logic.
- **Modularity**: Designed to easily accommodate future integration of Object Tracking, Decision Engine, and Robot Navigation.