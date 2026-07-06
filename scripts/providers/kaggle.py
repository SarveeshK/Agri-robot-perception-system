"""
providers/kaggle.py
===================
Kaggle dataset provider.

Downloads a Kaggle dataset by slug and extracts it into YOLO format.

Dependencies:
    pip install kaggle
    Kaggle API credentials must be configured:
        ~/.kaggle/kaggle.json   or
        KAGGLE_USERNAME + KAGGLE_KEY environment variables

Usage:
    python scripts/download_dataset.py \\
        --provider kaggle \\
        --dataset username/weed-detection-dataset

Output:
    datasets/raw/kaggle/
        images/
        labels/
        data.yaml   (if present in the dataset)
        dataset_info.yaml
"""

import os
import sys
import shutil
import subprocess
import yaml
import datetime
from pathlib import Path


RAW_DIR  = "datasets/raw/kaggle"
TEMP_DIR = "datasets/raw/temp/kaggle"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_yolo_dirs(root: Path) -> tuple[Path | None, Path | None]:
    """
    Search the extracted directory tree for images/ and labels/ folders.
    Returns (images_dir, labels_dir) or (None, None) if not found.
    """
    images_dir = labels_dir = None
    for d in root.rglob("images"):
        if d.is_dir():
            images_dir = d
            break
    for d in root.rglob("labels"):
        if d.is_dir():
            labels_dir = d
            break
    return images_dir, labels_dir


def write_dataset_info(output_dir: str, slug: str, image_count: int) -> None:
    info = {
        "provider"      : "kaggle",
        "name"          : slug,
        "license"       : "See Kaggle dataset page",
        "homepage"      : f"https://www.kaggle.com/datasets/{slug}",
        "download_date" : datetime.date.today().isoformat(),
        "image_count"   : image_count,
        "task"          : "detect",
        "notes"         : f"Downloaded from Kaggle: {slug}",
    }
    out_path = os.path.join(output_dir, "dataset_info.yaml")
    with open(out_path, "w") as f:
        yaml.dump(info, f, default_flow_style=False, sort_keys=False)
    print(f"  Written: {out_path}")


def download(args) -> None:
    """Entry point called by download_dataset.py dispatcher."""
    if not args.dataset:
        print("ERROR: --dataset is required for the kaggle provider.")
        print("Example: --dataset username/weed-detection-dataset")
        sys.exit(1)

    # Check kaggle CLI availability
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise FileNotFoundError
    except FileNotFoundError:
        print("ERROR: 'kaggle' CLI not found.")
        print("Install it with:  pip install kaggle")
        print("Configure credentials: https://www.kaggle.com/docs/api")
        sys.exit(1)

    os.makedirs(TEMP_DIR, exist_ok=True)

    print(f"Downloading Kaggle dataset: {args.dataset}")
    result = subprocess.run(
        ["kaggle", "datasets", "download",
         "-d", args.dataset,
         "-p", TEMP_DIR,
         "--unzip"],
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        print(f"ERROR: Kaggle download failed (exit code {result.returncode}).")
        sys.exit(1)

    # Find YOLO images/ and labels/ directories
    temp_path  = Path(TEMP_DIR)
    images_src, labels_src = find_yolo_dirs(temp_path)

    images_out = temp_path / "images"
    labels_out = temp_path / "labels"
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    image_count = 0

    if images_src and images_src != images_out:
        for img in images_src.rglob("*"):
            if img.suffix.lower() in IMAGE_EXTS:
                shutil.move(str(img), str(images_out / img.name))
                image_count += 1

    if labels_src and labels_src != labels_out:
        for lbl in labels_src.rglob("*.txt"):
            shutil.move(str(lbl), str(labels_out / lbl.name))

    if image_count == 0:
        print("WARNING: No images found in standard YOLO structure.")
        print("The dataset may use a different format.")
        print("Try manually placing images in datasets/raw/manual/ instead.")

    # Copy data.yaml if present
    for yaml_file in temp_path.rglob("data.yaml"):
        shutil.copy2(yaml_file, temp_path / "data.yaml")
        break

    # Move temp → raw/kaggle
    if os.path.exists(RAW_DIR):
        shutil.rmtree(RAW_DIR)
    shutil.move(TEMP_DIR, RAW_DIR)
    print(f"Moved to: {RAW_DIR}")

    write_dataset_info(RAW_DIR, args.dataset, image_count)

    print(f"\nKaggle download complete. {image_count} images in {RAW_DIR}")
    print("Next: python scripts/merge_dataset.py")
