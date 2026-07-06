"""
providers/roboflow.py
=====================
Roboflow Universe dataset provider.

Downloads a public Roboflow dataset in YOLOv8 format.

Dependencies:
    pip install roboflow

Usage:
    python scripts/download_dataset.py \\
        --provider roboflow \\
        --url https://universe.roboflow.com/author/dataset-name \\
        --version 1 \\
        --api-key YOUR_KEY   (or set ROBOFLOW_API_KEY environment variable)

Output:
    datasets/raw/roboflow/
        images/
        labels/
        data.yaml
        dataset_info.yaml
"""

import os
import sys
import shutil
import yaml
import datetime
from pathlib import Path


RAW_DIR    = "datasets/raw/roboflow"
TEMP_DIR   = "datasets/raw/temp/roboflow"
CONFIG_PATH = "config/class_mapping.yaml"


def parse_roboflow_url(url: str) -> tuple[str, str]:
    """Extract workspace and project name from Roboflow Universe URL."""
    # https://universe.roboflow.com/<workspace>/<project>
    parts = url.rstrip("/").split("/")
    if len(parts) < 5:
        print("ERROR: Invalid Roboflow URL. Expected format:")
        print("  https://universe.roboflow.com/<workspace>/<project>")
        sys.exit(1)
    return parts[-2], parts[-1]     # workspace, project


def load_target_labels(config_path: str, targets: list | None) -> list[str]:
    """Return list of canonical class names to download."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    all_classes = list(cfg["classes"].keys())
    if targets:
        return [c for c in all_classes if c in targets]
    return all_classes


def write_dataset_info(output_dir: str, url: str, project_name: str, image_count: int) -> None:
    info = {
        "provider"          : "roboflow",
        "name"              : project_name,
        "license"           : "See Roboflow project page",
        "homepage"          : url,
        "download_date"     : datetime.date.today().isoformat(),
        "requested_classes" : "All available in project",
        "image_count"       : image_count,
        "task"              : "detect",
        "notes"             : f"Downloaded from Roboflow Universe: {url}",
    }
    out_path = os.path.join(output_dir, "dataset_info.yaml")
    with open(out_path, "w") as f:
        yaml.dump(info, f, default_flow_style=False, sort_keys=False)
    print(f"  Written: {out_path}")


def download(args) -> None:
    """Entry point called by download_dataset.py dispatcher."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: 'roboflow' package not installed.")
        print("Install it with:  pip install roboflow")
        sys.exit(1)

    if not args.url:
        print("ERROR: --url is required for the roboflow provider.")
        print("Example: --url https://universe.roboflow.com/author/dataset-name")
        sys.exit(1)

    # Resolve API key
    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ERROR: Roboflow API key not provided.")
        print("Use --api-key YOUR_KEY or set the ROBOFLOW_API_KEY environment variable.")
        print("Get a free key at: https://roboflow.com")
        sys.exit(1)

    workspace, project_name = parse_roboflow_url(args.url)
    version = args.version or 1

    print(f"Connecting to Roboflow...")
    print(f"  Workspace : {workspace}")
    print(f"  Project   : {project_name}")
    print(f"  Version   : {version}")

    os.makedirs(TEMP_DIR, exist_ok=True)

    try:
        rf      = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)
        dataset = project.version(version).download("yolov8", location=TEMP_DIR)
    except Exception as e:
        print(f"ERROR during Roboflow download: {e}")
        sys.exit(1)

    # Roboflow creates train/valid/test subdirs — flatten to images/ + labels/
    images_out = os.path.join(TEMP_DIR, "images")
    labels_out = os.path.join(TEMP_DIR, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
    image_count = 0

    for split in ["train", "valid", "test"]:
        split_dir = Path(TEMP_DIR) / split
        if not split_dir.exists():
            continue
        for img in (split_dir / "images").glob("*"):
            if img.suffix.lower() in IMAGE_EXTS:
                shutil.move(str(img), os.path.join(images_out, img.name))
                lbl = split_dir / "labels" / (img.stem + ".txt")
                if lbl.exists():
                    shutil.move(str(lbl), os.path.join(labels_out, lbl.name))
                image_count += 1

    # Copy data.yaml from Roboflow download
    rf_yaml = Path(TEMP_DIR) / "data.yaml"
    if rf_yaml.exists():
        shutil.copy2(rf_yaml, os.path.join(TEMP_DIR, "data.yaml"))

    # Move temp → raw/roboflow
    if os.path.exists(RAW_DIR):
        shutil.rmtree(RAW_DIR)
    shutil.move(TEMP_DIR, RAW_DIR)
    print(f"Moved to: {RAW_DIR}")

    write_dataset_info(RAW_DIR, args.url, project_name, image_count)

    print(f"\nRoboflow download complete. {image_count} images in {RAW_DIR}")
    print("Next: python scripts/merge_dataset.py")
