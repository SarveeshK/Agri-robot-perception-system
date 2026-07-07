"""
providers/openimages.py
========================
Open Images V7 dataset provider.

Downloads images + bounding box annotations for requested canonical classes
using the OIDv7 public CSV files (no FiftyOne, no MongoDB required).

Dependencies:
    pip install openimages      # wraps OIDv7 CSV download cleanly

Output:
    datasets/raw/openimages/
        images/         ← downloaded JPEGs
        labels/         ← YOLO-format .txt annotation files
        data.yaml       ← provider class list (used by merge_dataset.py)
        dataset_info.yaml

Algorithm:
    1. Read target canonical classes from class_mapping.yaml
    2. Map canonical classes → OIDv7 label names via source_labels.openimages
    3. Download images + annotations via openimages package
    4. Convert annotations → YOLO normalized format
    5. Validate each image+label pair
    6. Move temp/ → datasets/raw/openimages/
    7. Write dataset_info.yaml manifest
"""

import os
import sys
import shutil
import yaml
import datetime
from pathlib import Path

RAW_DIR     = "datasets/raw/openimages"
TEMP_DIR    = "datasets/raw/temp/openimages"
CONFIG_PATH = "config/class_mapping.yaml"


def load_target_labels(config_path: str, targets: list | None) -> dict[str, list[str]]:
    """
    Returns {canonical_name: [oid_label, ...]} for requested targets.
    Skips classes with empty source_labels.openimages.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    result = {}
    for cls_name, cls_cfg in cfg["classes"].items():
        if targets and cls_name not in targets:
            continue
        oid_labels = cls_cfg.get("source_labels", {}).get("openimages", [])
        if oid_labels:
            result[cls_name] = oid_labels
        else:
            print(f"  [SKIP] {cls_name}: not available in Open Images V7 (source_labels.openimages is empty)")
    return result


def write_dataset_info(output_dir: str, requested: list[str], image_count: int) -> None:
    info = {
        "provider"          : "openimages",
        "name"              : "Open Images V7",
        "license"           : "CC BY 4.0",
        "homepage"          : "https://storage.googleapis.com/openimages/web/index.html",
        "download_date"     : datetime.date.today().isoformat(),
        "requested_classes" : requested,
        "image_count"       : image_count,
        "task"              : "detect",
        "notes"             : "Downloaded via openimages Python package",
    }
    out_path = os.path.join(output_dir, "dataset_info.yaml")
    with open(out_path, "w") as f:
        yaml.dump(info, f, default_flow_style=False, sort_keys=False)
    print(f"  Written: {out_path}")


def write_provider_data_yaml(output_dir: str, class_map: dict[str, list[str]]) -> None:
    """
    Write data.yaml listing OIDv7 label names per class.
    merge_dataset.py reads this to know the provider's own class names.
    """
    # names list: all OIDv7 label names in order
    all_labels = []
    for labels in class_map.values():
        all_labels.extend(labels)

    data = {
        "canonical_mapping": {
            oid_label: canonical
            for canonical, labels in class_map.items()
            for oid_label in labels
        }
    }
    out_path = os.path.join(output_dir, "data.yaml")
    with open(out_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def download(args) -> None:
    """Entry point called by download_dataset.py dispatcher."""
    try:
        from openimages.download import download_dataset
    except ImportError:
        print("ERROR: 'openimages' package not installed.")
        print("Install it with:  pip install openimages")
        sys.exit(1)

    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: {CONFIG_PATH} not found.")
        sys.exit(1)

    # ── 1. Resolve target labels ──────────────────────
    target_map = load_target_labels(CONFIG_PATH, args.target)
    if not target_map:
        print("No downloadable classes for Open Images. Check class_mapping.yaml.")
        sys.exit(1)

    print(f"Downloading for {len(target_map)} canonical class(es):")
    for cls, labels in target_map.items():
        print(f"  {cls} → OIDv7 labels: {labels}")
    print(f"Limit per class: {args.limit}\n")

    # ── 2. Download via openimages package ────────────
    os.makedirs(TEMP_DIR, exist_ok=True)
    all_oid_labels = [label for labels in target_map.values() for label in labels]

    print("Downloading images from Open Images V7...")
    for label in all_oid_labels:
        try:
            print(f"  -> Downloading {label}...")
            download_dataset(
                dest_dir=TEMP_DIR,
                class_labels=[label],
                annotation_format="darknet",
                limit=args.limit,
            )
        except Exception as e:
            print(f"  [ERROR] downloading {label}: {e}")
            continue

    # ── 3. Flatten and Validate ───────────────────────
    import cv2
    images_dir = os.path.join(TEMP_DIR, "images")
    labels_dir = os.path.join(TEMP_DIR, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # openimages saves to TEMP_DIR/<label>/images/ and TEMP_DIR/<label>/darknet/
    for oid_label in all_oid_labels:
        # The package often uses lowercase with underscores or exact label name
        # We search the whole TEMP_DIR for any images and darknet folders
        pass

    for img_file in Path(TEMP_DIR).rglob("*.jpg"):
        if img_file.parent.name == "images" and img_file.parent.parent.name != "temp":
            # Flatten image
            shutil.move(str(img_file), os.path.join(images_dir, img_file.name))
            
    for lbl_file in Path(TEMP_DIR).rglob("*.txt"):
        # The labels are in 'darknet' dir usually
        if lbl_file.parent.name == "darknet":
            shutil.move(str(lbl_file), os.path.join(labels_dir, lbl_file.name))

    # Clean up empty subdirectories
    for oid_label in all_oid_labels:
        label_dir = os.path.join(TEMP_DIR, oid_label.replace(" ", "_").lower())
        if os.path.exists(label_dir):
            shutil.rmtree(label_dir, ignore_errors=True)
        label_dir_exact = os.path.join(TEMP_DIR, oid_label)
        if os.path.exists(label_dir_exact):
            shutil.rmtree(label_dir_exact, ignore_errors=True)

    if not os.path.exists(images_dir):
        print("ERROR: No images downloaded. Check network and OIDv7 availability.")
        sys.exit(1)

    image_files = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))
    valid_count = 0
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [CORRUPT] {img_path.name} — removing")
            img_path.unlink(missing_ok=True)
            lbl = Path(labels_dir) / (img_path.stem + ".txt")
            lbl.unlink(missing_ok=True)
        else:
            valid_count += 1

    print(f"\nValidation: {valid_count} valid images out of {len(image_files)}")

    # ── 4. Move temp → raw/openimages ─────────────────
    if os.path.exists(RAW_DIR):
        shutil.rmtree(RAW_DIR)
    shutil.move(TEMP_DIR, RAW_DIR)
    print(f"Moved to: {RAW_DIR}")

    # ── 5. Write manifests ─────────────────────────────
    write_provider_data_yaml(RAW_DIR, target_map)
    write_dataset_info(RAW_DIR, list(target_map.keys()), valid_count)

    print(f"\nOpen Images download complete. {valid_count} images in {RAW_DIR}")
    print("Next: python scripts/merge_dataset.py")
