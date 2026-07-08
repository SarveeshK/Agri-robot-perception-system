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
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    if line.startswith("ROBOFLOW_API_KEY="):
                        api_key = line.strip().split("=", 1)[1].strip('"\'')
                        break
                        
    if not api_key:
        print("ERROR: Roboflow API key not provided.")
        print("Use --api-key YOUR_KEY, set the ROBOFLOW_API_KEY env var, or add it to the .env file.")
        print("Get a free key at: https://roboflow.com")
        sys.exit(1)

    workspace, project_name = parse_roboflow_url(args.url)
    version = args.version or 1

    print(f"Connecting to Roboflow...")
    print(f"  Workspace : {workspace}")
    print(f"  Project   : {project_name}")
    print(f"  Version   : {version}")

    # Remove TEMP_DIR if it exists so Roboflow doesn't skip the download
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    try:
        rf      = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)
        try:
            dataset = project.version(version).download("yolov8", location=TEMP_DIR)
        except Exception as e:
            print(f"  Version {version} not found. Auto-detecting latest version...")
            versions = project.versions()
            if not versions:
                raise Exception("No versions found for this project.")
            latest_version = versions[0].version.split("/")[-1]
            print(f"  Auto-detected version: {latest_version}")
            dataset = project.version(latest_version).download("yolov8", location=TEMP_DIR)
            
        download_path = dataset.location
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
        split_dir = Path(download_path) / split
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
    rf_yaml = Path(download_path) / "data.yaml"
    dest_yaml = Path(TEMP_DIR) / "data.yaml"
    if rf_yaml.exists() and rf_yaml.resolve() != dest_yaml.resolve():
        shutil.copy2(rf_yaml, dest_yaml)

    # Move temp → raw/roboflow_<project_name>
    dynamic_raw_dir = f"datasets/raw/roboflow_{project_name}"
    if os.path.exists(dynamic_raw_dir):
        shutil.rmtree(dynamic_raw_dir)
    shutil.move(TEMP_DIR, dynamic_raw_dir)
    print(f"Moved to: {dynamic_raw_dir}")

    write_dataset_info(dynamic_raw_dir, args.url, project_name, image_count)

    print(f"\nRoboflow download complete. {image_count} images in {dynamic_raw_dir}")
    print("Next: python scripts/merge_dataset.py")
