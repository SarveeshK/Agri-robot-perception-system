# Model Architecture & Selection Rationale

## Overview

This document describes the neural network architecture selected for the AgriVision Perception System and explains the engineering rationale behind the model selection, training methodology, and deployment workflow.

---

# 1. Selected Model: YOLOv8 Nano (`yolov8n.pt`)

The perception module of the AgriVision Perception System is built using **YOLOv8 Nano (`yolov8n.pt`)**, an object detection model developed by **Ultralytics**.

The model is responsible for detecting agricultural obstacles from RGB images captured by the Intel RealSense D456 camera. For every detected object, the system predicts:

* Object Class
* Bounding Box Coordinates
* Confidence Score

The detected objects are then combined with RealSense depth information to estimate their real-world distance from the robot.

---

## What is YOLO?

YOLO (**You Only Look Once**) is a real-time object detection architecture.

Unlike traditional two-stage detectors such as **R-CNN**, **Fast R-CNN**, and **Faster R-CNN**, which first generate region proposals and then classify each region, YOLO performs detection in a **single forward pass** through the neural network.

This single-stage detection approach significantly reduces inference time while maintaining high detection accuracy, making it well suited for robotic perception applications requiring real-time performance.

---

# 2. Why YOLOv8 Nano?

YOLOv8 is available in multiple model variants:

| Model   |    Parameters | Primary Use Case            |
| ------- | ------------: | --------------------------- |
| YOLOv8n |  ~3.2 Million | Embedded Systems & Robotics |
| YOLOv8s | ~11.2 Million | General Object Detection    |
| YOLOv8m | ~25.9 Million | High Accuracy Applications  |
| YOLOv8l | ~43.7 Million | Workstations & Servers      |
| YOLOv8x | ~68.2 Million | Research & High-End GPUs    |

For this project, the **Nano** variant was selected for the following engineering reasons.

### 2.1 Edge Deployment

The agricultural robot is intended to operate on embedded or edge computing hardware such as:

* NVIDIA Jetson series
* Intel NUC
* Industrial Mini-PC
* Raspberry Pi (with accelerator)

These systems have limited computational resources compared to workstation GPUs.

The lightweight Nano model provides efficient inference while operating within these hardware constraints.

---

### 2.2 Real-Time Performance

The perception pipeline continuously processes:

* RGB Frames
* Depth Frames
* Object Detection
* Distance Estimation

All of these operations must execute fast enough to support safe robot navigation.

YOLOv8 Nano provides low inference latency, enabling real-time perception suitable for obstacle avoidance.

> **Note:** Actual FPS depends on the deployment hardware (CPU/GPU), image resolution, and inference settings. It should not be stated as a guaranteed 30+ FPS across all platforms.

---

### 2.3 Lightweight Architecture

YOLOv8 Nano contains approximately **3.2 million trainable parameters**, making it significantly smaller than the larger YOLOv8 variants.

Advantages include:

* Lower RAM usage
* Reduced GPU memory requirements
* Faster inference
* Lower power consumption
* Easier deployment on embedded hardware

These characteristics make it an appropriate choice for an autonomous agricultural robot.

---

# 3. Training Methodology

The AgriVision Perception System employs **Transfer Learning (Fine-Tuning)** rather than training a neural network from random initialization.

Training an object detector entirely from scratch generally requires:

* Very large annotated datasets
* Significant computational resources
* Long training durations

Instead, a pretrained YOLOv8 Nano model is adapted for the agricultural domain.

---

## Pretrained Model

The project starts with:

```text
models/pretrained/yolov8n.pt
```

These pretrained weights have already learned generic visual features from a large public dataset (COCO), including:

* Edges
* Corners
* Texture patterns
* Shape representations
* Object boundaries

This provides a strong visual foundation.

---

## Fine-Tuning

The pretrained model is then fine-tuned using agricultural datasets.

Initial training uses a small foundation dataset containing:

* Tree
* Plant

Subsequent training will incorporate the company dataset containing:

* Coconut Tree
* Other Trees
* Rock
* Weed
* Small Stone

During fine-tuning, the model gradually adapts its learned weights to recognize agricultural obstacles while retaining the general visual knowledge learned during pretraining.

> **Technical Note:** During standard Ultralytics training, the entire network is typically fine-tuned unless layer freezing is explicitly configured.

---

# 3.1 How YOLO Learns

During training, the YOLOv8 model learns to detect agricultural objects by comparing its predictions with the annotated ground truth labels.

For every training image, the following sequence occurs:

1. The RGB image is resized and passed into the YOLOv8 neural network.
2. The network predicts:
   * Object Classes
   * Bounding Boxes
   * Confidence Scores
3. These predictions are compared against the ground truth annotations.
4. The prediction error (Loss) is calculated.
5. The optimizer updates the network weights using backpropagation.
6. This process repeats for every image in the dataset across multiple training epochs.

Each iteration gradually improves the model's ability to recognize agricultural obstacles.

```text
Input Image
      │
      ▼
YOLOv8 Neural Network
      │
      ▼
Predicted Objects
      │
      ▼
Ground Truth Comparison
      │
      ▼
Loss Calculation
      │
      ▼
Weight Update
      │
      ▼
Improved Model
```

---

# 3.2 Training Components

The YOLOv8 training process consists of several important components.

### Dataset

The dataset contains annotated RGB images representing agricultural obstacles.

Current foundation classes include:

* Tree
* Plant

Future production classes will include:

* Coconut Tree
* Other Trees
* Rock
* Weed
* Small Stone

---

### Epoch

An epoch represents one complete pass through the entire training dataset.

Example:

```text
Dataset Size : 60 Images

Epoch 1 → All 60 Images

Epoch 2 → All 60 Images Again

Epoch 3 → All 60 Images Again
```

Increasing the number of epochs allows the model to learn more refined visual patterns.

---

### Batch Size

Instead of processing the entire dataset simultaneously, images are processed in smaller groups called batches.

Example:

```text
Dataset : 60 Images

Batch Size : 4

15 Batches per Epoch
```

Smaller batches reduce memory usage, while larger batches may improve training stability depending on available hardware.

---

### Image Size

YOLOv8 requires all input images to have the same dimensions.

During training, every image is resized to a predefined resolution.

Example:

```text
640 × 640 pixels
```

Standardized image dimensions improve computational efficiency and maintain consistent input to the neural network.

---

# 3.3 Loss Function

The objective of training is to minimize prediction error, commonly referred to as **Loss**.

YOLOv8 optimizes several loss components simultaneously:

### Box Loss

Measures how accurately the predicted bounding box matches the annotated object.

Lower values indicate better localization.

---

### Classification Loss

Measures whether the object has been assigned the correct class label.

Example:

```text
Predicted : Plant
Actual : Tree
```

This contributes to the classification loss.

---

### Distribution Focal Loss (DFL)

Improves the precision of bounding box regression by learning more accurate object boundaries.

It helps YOLOv8 produce tighter and more reliable bounding boxes.

---

### Total Loss

The optimization process minimizes the combined value of:

```text
Total Loss
=
Box Loss
+
Classification Loss
+
Distribution Focal Loss
```

As training progresses, these loss values should gradually decrease.

---

# 3.4 Model Evaluation Metrics

After each training epoch, the model is evaluated using the validation dataset.

Several performance metrics are recorded.

### Precision

Precision measures how many predicted detections are correct.

High Precision indicates fewer false detections.

---

### Recall

Recall measures how many actual objects were successfully detected.

High Recall indicates fewer missed objects.

---

### mAP50

Mean Average Precision at an IoU threshold of 0.50.

This metric summarizes overall detection performance.

---

### mAP50-95

A stricter evaluation metric computed over multiple IoU thresholds.

This provides a more comprehensive measure of model quality.

---

# 3.5 Model Output

The training process produces several output files.

| File                   | Description                                    |
| ---------------------- | ---------------------------------------------- |
| `best.pt`              | Best-performing model selected during training |
| `last.pt`              | Model weights from the final training epoch    |
| `results.png`          | Training and validation curves                 |
| `confusion_matrix.png` | Visualization of classification performance    |
| `PR_curve.png`         | Precision–Recall curve                         |
| `F1_curve.png`         | F1 score across confidence thresholds          |

Among these, `best.pt` is used for deployment within the AgriVision Perception System.

---

# 4. Why Transfer Learning?

Transfer Learning provides several engineering advantages.

### Reduced Data Requirements

A relatively small domain-specific dataset can achieve useful detection performance.

---

### Faster Training

Training converges significantly faster than training a network from random initialization.

---

### Lower Computational Cost

Fine-tuning can be performed on a single GPU or even a capable CPU for small datasets, eliminating the need for large-scale compute infrastructure.

---

### Improved Performance

Starting from pretrained weights generally produces better accuracy than training from scratch when only limited training data is available.

---

# 5. Deployment Workflow

The deployment pipeline follows the sequence below.

```text
Pretrained Model
models/pretrained/yolov8n.pt
            │
            ▼
Transfer Learning
(train_model.py)
            │
            ▼
Training & Validation
            │
            ▼
Model Evaluation
            │
            ▼
Best Model Selection
            │
            ▼
models/trained/best.pt
            │
            ▼
Real-Time Inference
(src/main.py)
```

The training process automatically evaluates each epoch and saves the highest-performing model as `best.pt`.

This model is subsequently loaded by the perception pipeline for deployment.

---

# 6. Role Within the Perception System

The trained YOLO model is one component of the complete robotic perception pipeline.

```text
Intel RealSense D456
        │
        ▼
RGB Frame
        │
        ▼
YOLOv8 Nano (best.pt)
        │
        ▼
Detected Objects
(Tree, Plant,
Rock, Weed,
Small Stone)
        │
        ▼
Depth Processing
        │
        ▼
Distance Estimation
        │
        ▼
Obstacle Assessment
        │
        ▼
Robot Navigation
(GO / DON'T GO)
```

YOLO determines **what objects are present and where they are located**, while the RealSense depth camera determines **how far away those objects are**. Together, these components enable reliable obstacle perception for autonomous navigation.
