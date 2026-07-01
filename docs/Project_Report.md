# AgriVision Perception System: Comprehensive Project Report

**Date:** July 2026  
**Project Objective:** To develop a robust, modular, and AI-driven 3D perception and obstacle avoidance system for an agricultural robot operating in a plantation environment (e.g., coconut tree plantations).

---

## Executive Summary
The AgriVision Perception System has been successfully developed through two major architectural phases. The project integrates physical hardware (Intel RealSense D456) with state-of-the-art deep learning (YOLOv8) to provide the robot with real-time spatial awareness. 

Rather than relying on black-box, monolithic scripts, the system was engineered using professional Machine Learning Operations (MLOps) principles, ensuring that everything from camera calibration to dataset formatting and model training is modular, configurable, and ready for production deployment.

---

## Phase 1: The Inference System (Hardware & Perception)
**Goal:** Build the software necessary for the robot to "see" the world, measure distances, and prevent collisions in real-time.

### 1. Hardware Integration
- **Intel RealSense D456 Configuration:** Integrated the depth camera using the `pyrealsense2` library. 
- **Frame Alignment:** Built a custom `FrameAligner` to mathematically synchronize the RGB (color) video feed with the Depth video feed, ensuring that an object detected in the 2D image perfectly correlates to its 3D distance data.
- **Advanced Post-Processing:** Implemented spatial, temporal, and hole-filling filters to smooth out depth noise and improve accuracy at long ranges.

### 2. The AI Perception Pipeline
- **YOLOv8 Object Detection:** Integrated the Ultralytics YOLOv8 inference engine (`yolov8n.pt`) to draw bounding boxes around obstacles (trees, plants, people, etc.) in the color frame.
- **3D Spatial Measurement:** Developed a highly robust `DepthProcessor`. Instead of just reading a single pixel of depth (which is prone to errors), it extracts the central 50% of the object's bounding box and calculates the median valid depth to output highly accurate distances in centimeters.
- **Width/Height Estimation:** Using the camera's intrinsics (focal length), the system calculates the physical width and height of the detected obstacle in centimeters.

### 3. Safety & Visualization
- **Obstacle Warning System:** Implemented a global safety constraint in `config/settings.yaml`. If any object breaches the `safety_distance_cm` threshold, the UI triggers a critical warning for the robot navigation system.
- **Dynamic HUD:** Built a visualizer that renders bounding boxes, class names, FPS, and 3D measurements directly onto the live camera feed for easy debugging.

---

## Phase 2: Dataset Engineering & MLOps (Model Training)
**Goal:** Build a complete, localized Machine Learning pipeline to train the robot's brain to recognize specific agricultural obstacles (coconut trees, weeds, rocks) instead of generic internet objects.

### 1. Automated Data Acquisition
- Integrated **FiftyOne**, an open-source dataset management tool.
- Developed `scripts/download_openimages.py` to programmatically download thousands of specific images (e.g., "Tree", "Plant") from the massive Open Images V7 database without requiring manual web scraping.

### 2. Dataset Preparation & Formatting
- Built `scripts/prepare_dataset.py` to parse the raw FiftyOne database and automatically partition the images into an 80% Training Split and a 20% Validation Split.
- The script automatically converts the images and labels into the strict folder structure required by YOLOv8 and generates the critical `data.yaml` mapping file.
- **Dataset Statistics Engine:** Added an automated reporting feature that outputs `dataset_statistics.txt`, calculating the total objects, average objects per image, and class distributions to guarantee dataset health before training.

### 3. Model Fine-Tuning (Transfer Learning)
- Developed `scripts/train_model.py` to perform **Transfer Learning**. Rather than training a neural network from scratch, the script loads the pre-trained `yolov8n.pt` foundation and fine-tunes its weights exclusively on the new agricultural dataset.
- The pipeline tracks hyperparameters (saved in `training_config.yaml`) to ensure every experiment is reproducible.

### 4. Evaluation and Deployment
- **Model Evaluation (`evaluate_model.py`):** Calculates Precision, Recall, and mAP (Mean Average Precision) against the validation set to mathematically prove the model's intelligence.
- **Safe Export (`export_model.py`):** Automatically isolates the highest-performing model (`best.pt`) and promotes it to the `models/trained/` directory for live production use in Phase 1.

---

## Configuration & Architecture Standards
To ensure the codebase scales effectively when the company's proprietary dataset is introduced, strict software engineering standards were enforced:
1. **No Hardcoding:** Classes and parameters are managed in `config/class_mapping.yaml` and `config/settings.yaml`.
2. **Label Noise Prevention:** By mapping public classes (like generic "Plant") to internal IDs, the system ensures that future annotations don't break the model's logic.
3. **Comprehensive Documentation:** Generated Standard Operating Procedures (SOPs) for data collection and annotation (`docs/Phase2/`) to ensure the company's future dataset is meticulously structured.

---

## Next Steps (Phase 3)
The system is now fully verified and operational. The upcoming Phase 3 will involve:
1. Receiving the proprietary company dataset from the actual plantation.
2. Running the images through the Phase 2 MLOps pipeline.
3. Fine-tuning the final production model.
4. Deploying the model onto the physical robot hardware for real-world RealSense navigation testing.
