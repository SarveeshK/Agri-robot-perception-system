# Model Architecture & Selection Rationale

This document outlines the specific neural network architecture chosen for the AgriVision Perception System and details the engineering reasoning behind this selection.

---

## 1. Selected Model: YOLOv8 Nano (`yolov8n.pt`)

The perception engine of the robot is powered by the **YOLOv8 Nano** model, developed by Ultralytics.

### What is YOLO?
YOLO stands for **"You Only Look Once."** It is a state-of-the-art, single-stage object detection framework. Older AI architectures (like R-CNN) scan an image multiple times to propose regions and then classify them, which is highly accurate but too slow for real-time video. YOLO treats object detection as a single regression problem, looking at the entire image exactly once to simultaneously predict bounding boxes and class probabilities. This makes it exceptionally fast.

### Why the "Nano" Variant?
The YOLOv8 architecture comes in multiple scales (Nano, Small, Medium, Large, Extra Large). We explicitly selected the **Nano (n)** variant for the following engineering reasons:

1. **Edge Deployment constraints:** Agricultural robots typically operate on low-power, onboard edge computers (e.g., Raspberry Pi, Nvidia Jetson Nano, or Intel NUCs) running on battery power in the field. They do not have access to massive cloud GPUs.
2. **Real-Time Latency:** The perception system must process the Intel RealSense depth stream and RGB stream simultaneously. The Nano variant guarantees **30+ Frames Per Second (FPS)** inference speeds, ensuring the robot's collision avoidance system reacts instantly to obstacles.
3. **Memory Footprint:** The Nano model contains only ~3.2 million parameters (compared to ~68 million in the Large variant). It requires significantly less RAM and VRAM, leaving system resources available for navigation and motor control logic.

---

## 2. Training Methodology: Transfer Learning

We do not train the neural network from scratch (random weight initialization). Training a model from scratch requires millions of images, massive GPU clusters, and weeks of compute time. 

Instead, we employ a technique called **Transfer Learning (Fine-Tuning)**.

### How it Works
1. **The Foundation:** We begin with the pre-trained `yolov8n.pt` weights. This model has already been trained on the massive COCO dataset (Common Objects in Context). Because of this, its internal "brain" (the convolution layers) already understands fundamental visual concepts like edges, textures, shadows, and basic shapes.
2. **The Fine-Tuning:** We feed our custom agricultural dataset (Coconut Trees, Plants, Weeds, Rocks) into this foundation model. 
3. **The Result:** We "freeze" or leverage the foundational knowledge and only force the network to update its final classification layers to recognize our specific agricultural classes.

### Engineering Benefits of Transfer Learning
- **Data Efficiency:** We can achieve high accuracy with a fraction of the data (hundreds of images instead of hundreds of thousands).
- **Time Efficiency:** A model can converge and reach optimal accuracy in minutes or hours rather than days.
- **Cost Reduction:** It allows the entire MLOps pipeline to be run locally on standard hardware rather than renting expensive cloud compute.

---

## 3. Deployment Flow

1. **Base Model:** `models/pretrained/yolov8n.pt`
2. **Training Script:** `scripts/train_model.py` orchestrates the transfer learning process.
3. **Production Export:** The system evaluates the training run and automatically exports the most mathematically accurate iteration to `models/trained/best.pt`.
4. **Inference:** `src/main.py` loads `best.pt` into the active camera stream for real-time agricultural navigation.
