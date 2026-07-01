# AgriVision Perception System: Comprehensive Technical Report

**Date:** July 2026  
**Project Objective:** To architect and develop a robust, modular, and AI-driven 3D perception and obstacle avoidance system for an autonomous agricultural robot operating in complex plantation environments (e.g., coconut tree plantations).

---

## 1. Executive Summary
The AgriVision Perception System has been engineered over two major architectural phases. The project successfully fuses physical depth-sensing hardware (Intel RealSense D456) with state-of-the-art Deep Learning (YOLOv8) to provide the robot with real-time 3D spatial awareness. 

Crucially, the system abandons monolithic scripting in favor of professional Object-Oriented Programming (OOP) and Machine Learning Operations (MLOps) principles. The architecture is highly decoupled, configuration-driven, and designed for immediate scalability from a local development laptop to a high-performance production Linux server.

---

## 2. Technical Stack & Libraries
The system relies on a strictly version-controlled Python environment to ensure deterministic behavior across platforms.

| Component | Library | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Hardware SDK** | `pyrealsense2` | `2.58.2.10647` | Directly interfaces with the Intel RealSense depth camera, managing streams, alignment, and hardware filters. |
| **Computer Vision** | `opencv-python` | `4.13.0.92` | Core image processing matrix operations, real-time bounding box rendering, and HUD (Heads-Up Display) generation. |
| **Matrix Math** | `numpy` | `2.4.6` | High-performance array manipulation, crucial for depth extraction and bounding box slicing. |
| **Deep Learning** | `ultralytics` | `8.4.69` | The backbone AI framework providing the YOLOv8 neural network architecture for object detection and transfer learning. |
| **Data Engineering**| `fiftyone` | `0.23.8` | Open-source dataset management API used to programmatically query, download, and format the Open Images V7 database. |
| **Configuration** | `PyYAML` | `6.0.3` | Parses `settings.yaml` and `class_mapping.yaml`, allowing non-programmatic modification of system parameters. |

---

## 3. System Architecture & Directory Structure
The repository is structured to strictly separate runtime inference (`src/`), machine learning pipelines (`scripts/`), configuration (`config/`), and artifacts (`models/`, `datasets/`).

```text
Agri-robot-perception-system/
├── config/                 # Centralized configuration (No hardcoded values)
│   ├── class_mapping.yaml  # Maps raw dataset classes to internal system IDs
│   └── settings.yaml       # Controls camera intrinsics, YOLO confidence, and safety logic
├── datasets/               # Local data lake
│   ├── processed/          # YOLO-formatted data (images/, labels/, data.yaml)
│   └── raw/                # Raw FiftyOne downloads and future proprietary images
├── docs/                   # System documentation and Standard Operating Procedures (SOPs)
│   ├── Phase1/             # RealSense & Inference documentation
│   └── Phase2/             # MLOps, Annotation, and Training guides
├── models/                 # Model registry
│   ├── pretrained/         # Foundation models (yolov8n.pt)
│   └── trained/            # Production-ready fine-tuned models (best.pt)
├── scripts/                # Phase 2: MLOps Pipeline Scripts (Download -> Train -> Export)
└── src/                    # Phase 1: Real-time Robot Inference Engine
    ├── camera/             # Hardware interfacing (RealSenseManager, FrameAligner)
    ├── perception/         # AI & Math (YoloDetector, DepthProcessor, ObjectMeasurement)
    ├── utils/              # System utilities (Logger, FPS Profiler)
    └── main.py             # The primary entry point connecting Camera to Perception
```

---

## 4. Phase 1: The Inference Engine (Hardware & Perception)
**Goal:** Build the runtime software necessary for the robot to "see" the world, measure physical distances, and prevent collisions in real-time (`src/`).

### 4.1 Hardware Integration (`src/camera/`)
- **Intel RealSense D456 Configuration:** Managed by `RealSenseManager`. Initializes RGB and Depth streams at 640x480 @ 30 FPS.
- **Hardware Frame Alignment:** The depth sensor and RGB sensor are physically offset. `FrameAligner` mathematically projects the 3D depth point cloud onto the 2D color image plane, ensuring perfect spatial synchronization.
- **Advanced Post-Processing:** Applies a cascade of hardware-accelerated filters (Spatial, Temporal, and Hole-Filling) to smooth depth noise, particularly for distant or highly reflective agricultural surfaces.

### 4.2 The AI Perception Pipeline (`src/perception/`)
- **YOLOv8 Object Detection:** `YoloDetector` loads the neural network weights into memory. It analyzes the aligned RGB frame and returns bounding boxes, class IDs, and confidence scores for detected obstacles.
- **Robust 3D Spatial Measurement:** `DepthProcessor` isolates the bounding box coordinates on the Depth frame. To prevent edge-bleed (where the depth sensor reads the background behind a tree instead of the tree itself), it extracts only the **central 50% ROI (Region of Interest)**. It then calculates the median valid depth to output a highly accurate distance in centimeters.
- **Physical Size Estimation:** `ObjectMeasurement` utilizes the camera's dynamically calculated intrinsics (focal length and principal point) to convert the 2D bounding box pixels and 3D depth into physical real-world Width and Height (in cm).

### 4.3 Safety Logic & Visualization
- **Obstacle Warning System:** If any measured object breaches the `safety_distance_cm` threshold defined in `config/settings.yaml`, the system triggers a global navigation warning.
- **Dynamic HUD:** `Visualizer` renders the bounding boxes, class names, 3D measurements (Distance, WxH), and FPS metrics directly onto a low-latency live video feed.

---

## 5. Phase 2: Dataset Engineering & MLOps (Model Training)
**Goal:** Build a complete, localized Machine Learning Operations pipeline to transition the robot's brain from recognizing generic internet objects to specific agricultural obstacles (coconut trees, weeds, etc.) (`scripts/`).

### 5.1 Automated Data Acquisition (`download_openimages.py`)
- Replaces manual web scraping by utilizing the `fiftyone` API to securely connect to the Open Images V7 database.
- Dynamically reads `config/class_mapping.yaml` to request exactly 60 samples of specific target classes (`Tree`, `Plant`), establishing a persistent, local data lake.

### 5.2 Dataset Preparation & Formatting (`prepare_dataset.py`)
- Automatically parses the FiftyOne database and implements an 80/20 random split (Train/Validation).
- Safely wipes stale directories (`datasets/processed/`) to prevent data contamination.
- Converts bounding boxes from absolute coordinates to the normalized center-x/center-y format required by YOLO.
- **Dataset Statistics Engine:** Generates a comprehensive `outputs/dataset_statistics.txt` report detailing total images, total objects, object density (Average Objects/Image), and class distributions to guarantee dataset health prior to neural network training.

### 5.3 Model Fine-Tuning / Transfer Learning (`train_model.py`)
- Implements Transfer Learning via the `ultralytics` API. Instead of randomly initializing weights (training from scratch), the script loads the pre-trained `yolov8n.pt` foundation model.
- Fine-tunes the network on the custom dataset for a specified number of epochs, intelligently adjusting its deeper layers to recognize agricultural patterns.
- Serializes training hyperparameters to `outputs/train/training_config.yaml` to ensure absolute experimental reproducibility.

### 5.4 Evaluation and Deployment (`evaluate_model.py` & `export_model.py`)
- **Mathematical Evaluation:** Runs the trained model against the unseen Validation dataset, calculating Precision, Recall, and mAP (Mean Average Precision).
- **Deployment:** The exporter isolates the highest-performing iteration (`best.pt`) and safely promotes it from the messy experimental output directory to the locked `models/trained/` production registry.

---

## 6. Next Steps (Phase 3: Production Training)
The architecture is now fully verified and operational. The upcoming Phase 3 will involve:
1. Transferring the MLOps pipeline to a high-performance Linux server.
2. Ingesting the proprietary company dataset representing the true coconut plantation environment.
3. Executing the complete Phase 2 pipeline to fine-tune the final production model.
4. Deploying the resulting model onto the physical robot hardware for real-world RealSense navigation testing.
