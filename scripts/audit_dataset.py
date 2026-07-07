#!/usr/bin/env python3
"""
audit_dataset.py
=================
Generates the comprehensive Dataset Sanity Report for Phase 4.
Runs prior to baseline training to ensure dataset health, balance,
and annotation quality.

Usage:
    python scripts/audit_dataset.py
"""

import os
import sys
import yaml
import json
import statistics
from pathlib import Path
from collections import defaultdict
from datetime import datetime

CONFIG_PATH = "config/class_mapping.yaml"
OUTPUTS_DIR = "outputs"
PROCESSED_DIR = "datasets/processed"
MANIFEST_PATH = os.path.join(PROCESSED_DIR, "build_manifest.yaml")
DEDUP_METRICS = os.path.join(PROCESSED_DIR, "dedup_metrics.json")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def load_class_names(config_path: str) -> list[str]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    classes = cfg["classes"]
    ordered = sorted(classes.items(), key=lambda x: x[1]["id"])
    return [name for name, _ in ordered]

def format_percentage(part, whole):
    if whole == 0: return "0.0%"
    return f"{(part/whole)*100:.1f}%"

def bar(count: int, max_count: int, width: int = 24) -> str:
    n = int((count / max_count) * width) if max_count > 0 else 0
    return "█" * n

def get_git_commit():
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "Unknown"

def main():
    print("=" * 50)
    print("AgriVision — Dataset Sanity Report Generator")
    print("=" * 50)

    if not os.path.exists(PROCESSED_DIR):
        print(f"ERROR: {PROCESSED_DIR} not found. Run build_dataset.py first.")
        sys.exit(2)

    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: {CONFIG_PATH} not found.")
        sys.exit(2)

    class_names = load_class_names(CONFIG_PATH)
    num_classes = len(class_names)
    
    # Target thresholds defined in Phase 4 spec
    targets = {
        "Tree": 2000,
        "Rock": 1500,
        "Fence": 800,
        "Bush": 1000,
        "Weed": 2000,
        "Stump": 700,
        "Small_Stone": 800,
        "Human": 0 # As available
    }

    # ── State Tracking ──
    total_images = 0
    total_annotations = 0
    
    empty_background_images = 0
    images_without_annotations = 0 # Corrupt or unreadable labels
    corrupt_skipped_images = 0
    invalid_labels = 0
    
    class_counts = defaultdict(int)
    bboxes_per_image = []
    
    # ── Parse Splits ──
    for split in ["train", "val"]:
        split_dir = os.path.join(PROCESSED_DIR, split)
        images_dir = os.path.join(split_dir, "images")
        labels_dir = os.path.join(split_dir, "labels")
        
        if not os.path.exists(images_dir): continue
        
        image_files = [f for f in Path(images_dir).iterdir() if f.suffix.lower() in IMAGE_EXTS]
        total_images += len(image_files)
        
        for img_path in image_files:
            lbl_path = Path(labels_dir) / (img_path.stem + ".txt")
            if not lbl_path.exists():
                images_without_annotations += 1
                continue
                
            try:
                with open(lbl_path, "r") as f:
                    lines = [l.strip() for l in f if l.strip()]
            except:
                corrupt_skipped_images += 1
                continue
                
            if not lines:
                empty_background_images += 1
                bboxes_per_image.append(0)
                continue
                
            valid_boxes = 0
            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    try:
                        cls_id = int(parts[0])
                        if 0 <= cls_id < num_classes:
                            class_counts[cls_id] += 1
                            total_annotations += 1
                            valid_boxes += 1
                        else:
                            invalid_labels += 1
                    except:
                        invalid_labels += 1
                else:
                    invalid_labels += 1
            bboxes_per_image.append(valid_boxes)

    # ── Calculate Annotation Quality ──
    filtered_bboxes = [b for b in bboxes_per_image if b > 0]
    avg_boxes = sum(filtered_bboxes) / len(filtered_bboxes) if filtered_bboxes else 0
    median_boxes = statistics.median(filtered_bboxes) if filtered_bboxes else 0
    max_boxes = max(filtered_bboxes) if filtered_bboxes else 0

    # ── Calculate Balance Metrics ──
    max_cls_count = max(class_counts.values()) if class_counts else 1
    min_cls_count = min([class_counts[i] for i in range(num_classes) if class_counts[i] > 0], default=1)
    majority_minority_ratio = max_cls_count / min_cls_count

    # ── Build Metadata ──
    manifest_data = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            manifest_data = yaml.safe_load(f) or {}

    dedup_data = {}
    if os.path.exists(DEDUP_METRICS):
        with open(DEDUP_METRICS, "r") as f:
            dedup_data = json.load(f)

    # ── Generate Report ──
    lines = []
    lines.append("==================================================")
    lines.append("          DATASET SANITY REPORT (Phase 4)         ")
    lines.append("==================================================")
    
    lines.append("\n[1] DATASET SUMMARY")
    lines.append("-" * 50)
    lines.append(f"Total Images        : {total_images}")
    lines.append(f"Total Annotations   : {total_annotations}")
    
    lines.append("\n[2] ANNOTATION QUALITY")
    lines.append("-" * 50)
    lines.append(f"Average Bounding Boxes / Image : {avg_boxes:.1f}")
    lines.append(f"Median Bounding Boxes / Image  : {median_boxes}")
    lines.append(f"Max Bounding Boxes in 1 Image  : {max_boxes}")
    
    lines.append("\n[3] DATASET HEALTH")
    lines.append("-" * 50)
    lines.append(f"Empty/Background Images   : {empty_background_images}")
    lines.append(f"Missing Labels (Corrupt)  : {images_without_annotations}")
    lines.append(f"Unreadable Files          : {corrupt_skipped_images}")
    lines.append(f"Invalid/OOB Labels        : {invalid_labels}")
    
    lines.append("\n[4] DEDUPLICATION STATISTICS")
    lines.append("-" * 50)
    if dedup_data:
        lines.append(f"Exact Duplicates Removed : {dedup_data.get('exact_removed', 0)}")
        lines.append(f"Near Duplicates Removed  : {dedup_data.get('near_removed', 0)}")
        lines.append(f"Algorithm                : {dedup_data.get('algorithm', 'Unknown')}")
    else:
        lines.append("No deduplication metrics found.")

    lines.append("\n[5] BALANCE METRICS & COVERAGE TARGETS")
    lines.append("-" * 50)
    lines.append(f"Minority/Majority Ratio : 1 : {majority_minority_ratio:.1f}")
    lines.append("")
    
    below_target = []
    for i, cls_name in enumerate(class_names):
        count = class_counts.get(i, 0)
        target = targets.get(cls_name, 0)
        pct = format_percentage(count, total_annotations)
        status = "✅" if count >= target else "❌"
        if target == 0: status = "ℹ️"
        
        if count < target and target > 0:
            below_target.append(cls_name)
            
        lines.append(f"{cls_name:<15} {bar(count, max_cls_count):<24} {count:>6} ({pct:>5}) Target: {target:<5} {status}")

    if below_target:
        lines.append("\nWARNING: The following classes are BELOW target thresholds:")
        for cls in below_target:
            lines.append(f"  - {cls}")
            
    lines.append("\n[6] BUILD METADATA")
    lines.append("-" * 50)
    lines.append(f"Pipeline Version : v1.0.0")
    lines.append(f"Build Timestamp  : {manifest_data.get('timestamp', datetime.now().isoformat())}")
    lines.append(f"Git Commit       : {get_git_commit()}")
    lines.append("==================================================")
    
    report_text = "\n".join(lines) + "\n"
    print(report_text)
    
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUTS_DIR, "dataset_sanity_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
        
    print(f"Report saved to: {report_path}")
    
    # Set exit code
    if invalid_labels > 0 or corrupt_skipped_images > 0:
        sys.exit(2)
    elif below_target:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
