# YOLO Training Guide

This document explains the core concepts behind training the YOLOv8 model locally.

## Key Hyperparameters

### 1. Epochs
- **What it is:** One epoch means the model has seen the entire training dataset once.
- **Usage:** Set high enough for the model to learn (e.g., 100-300). YOLOv8 uses "Early Stopping," meaning if the model stops improving for 50 epochs, it will stop automatically to prevent overfitting.

### 2. Batch Size
- **What it is:** The number of images processed simultaneously before the model's weights are updated.
- **Usage:** Larger batches stabilize training but consume more GPU/CPU RAM. If your system crashes with "Out of Memory" (OOM), reduce the batch size (e.g., from 16 to 8).

### 3. Learning Rate (LR)
- **What it is:** Determines how much the model's weights change in response to the calculated error each time a batch is processed.
- **Usage:** Ultralytics YOLO manages this automatically very well. If you manually tune it, too high means the model overshoots the optimal solution; too low means training takes forever.

### 4. Image Size (`imgsz`)
- **What it is:** All images are resized to a square of this dimension before being passed into the neural network (e.g., 640).
- **Usage:** Higher resolution (e.g., 1280) detects smaller objects better but runs slower during both training and inference. 640 is standard.

## Understanding Output Metrics

### Loss
- **Box Loss:** How inaccurate the bounding box coordinates are.
- **Class Loss:** How inaccurate the class predictions are.
- **DFL Loss:** Distribution Focal Loss (fine-tunes box edges).
- *Goal:* You want these numbers to consistently go down over time.

### Confusion Matrix (`confusion_matrix.png`)
- Shows what classes the model is confusing with what. The diagonal should be dark (representing correct predictions). 
- If `weed` is often predicted as `background`, the model is failing to detect weeds (False Negatives).

### PR Curve (`PR_curve.png`)
- A graph plotting Precision against Recall at various confidence thresholds.
- *Goal:* A curve that bows up into the top right corner. The area under this curve is the mAP.
