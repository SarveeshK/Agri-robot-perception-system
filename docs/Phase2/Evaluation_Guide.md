# Evaluation Guide

Model evaluation determines whether the robot is safe to deploy in the plantation. A model that looks fine subjectively might fail statistically.

## Key Evaluation Metrics

### 1. Precision (P)
- **Definition:** Out of all the objects the model *claimed* were trees, how many were *actually* trees?
- **Formula:** True Positives / (True Positives + False Positives)
- **Impact:** Low precision means the robot stops for things that aren't obstacles (False Positives).

### 2. Recall (R)
- **Definition:** Out of all the trees that *actually exist* in the image, how many did the model *detect*?
- **Formula:** True Positives / (True Positives + False Negatives)
- **Impact:** Low recall means the robot fails to see obstacles and crashes into them (False Negatives). **In robotics, high recall is generally more important than high precision.**

### 3. mAP (Mean Average Precision)
- **mAP50:** Area under the Precision-Recall curve at an Intersection over Union (IoU) threshold of 0.50 (meaning the predicted bounding box overlaps the ground truth box by at least 50%).
- **mAP50-95:** The average of mAP over multiple strictness thresholds (from 0.50 to 0.95).
- **Impact:** mAP50 is your baseline accuracy metric. mAP50-95 tells you how *tight* and accurate the bounding boxes are.

### 4. Inference Speed & FPS
- Accuracy doesn't matter if the model is too slow.
- The model must process frames faster than the robot drives. If the RealSense runs at 30 FPS, inference should ideally take < 33ms.
- Always check the inference speed logs in `outputs/val/`.

## The False Positive / False Negative Trade-off
- You can tune the `confidence` threshold in `config/settings.yaml`.
- **Lower Confidence (e.g., 0.25):** Model detects more things. False Negatives go down, False Positives go up.
- **Higher Confidence (e.g., 0.70):** Model only predicts when sure. False Positives go down, False Negatives go up.
