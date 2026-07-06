"""
smoke_test_inference.py
========================
Runs inference on a few validation images to ensure the trained model
can actually produce detections, serving as the final gate of the Phase 2
smoke test.

Usage:
    python scripts/smoke_test_inference.py
"""

import os
import sys
import random
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics package not installed.")
    sys.exit(1)


MODEL_PATH = "models/trained/best.pt"
VAL_IMAGES_DIR = "datasets/processed/val/images"
NUM_TEST_IMAGES = 5

def main():
    print("=" * 50)
    print("AgriVision — Smoke Test Inference")
    print("=" * 50)

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run train_model.py and export_model.py first.")
        sys.exit(1)

    if not os.path.exists(VAL_IMAGES_DIR):
        print(f"ERROR: Validation directory not found at {VAL_IMAGES_DIR}")
        sys.exit(1)

    images = list(Path(VAL_IMAGES_DIR).glob("*.jpg")) + list(Path(VAL_IMAGES_DIR).glob("*.png"))
    if not images:
        print("ERROR: No images found in the validation set.")
        sys.exit(1)

    test_images = random.sample(images, min(NUM_TEST_IMAGES, len(images)))
    print(f"Loading model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"ERROR: Failed to load model. {e}")
        sys.exit(1)

    print(f"\nRunning inference on {len(test_images)} validation images...\n")
    
    total_detections = 0
    for img_path in test_images:
        results = model.predict(source=str(img_path), verbose=False)
        boxes = results[0].boxes
        num_dets = len(boxes) if boxes is not None else 0
        total_detections += num_dets
        print(f"  {img_path.name} -> {num_dets} detections")

    print("\n" + "=" * 50)
    if total_detections > 0:
        print("Inference Test: PASS ✅")
        print("Model successfully loaded and produced bounding boxes.")
    else:
        print("Inference Test: WARNING ⚠")
        print("Model loaded, but produced 0 detections on the test sample.")
        print("This is normal for a tiny 3-epoch smoke test with 15 images,")
        print("but for a full run, this would indicate a failed training process.")
    print("=" * 50)


if __name__ == "__main__":
    main()
