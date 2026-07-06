"""
providers/manual.py
====================
Manual ZIP dataset provider.

Extracts a local ZIP file containing a YOLO-format dataset into
datasets/raw/manual/. Use this for any dataset downloaded manually
from a browser, GitHub release, or company share.

Usage:
    python scripts/download_dataset.py \\
        --provider manual \\
        --zip /path/to/dataset.zip \\
        --name my_dataset_name

Output:
    datasets/raw/manual/
        images/
        labels/
        data.yaml   (if present)
        dataset_info.yaml   (you will be prompted to fill in license info)
"""

import os
import sys
import shutil
import zipfile
import yaml
import datetime
from pathlib import Path


RAW_DIR    = "datasets/raw/manual"
TEMP_DIR   = "datasets/raw/temp/manual"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_yolo_dirs(root: Path) -> tuple[Path | None, Path | None]:
    """Search extracted tree for images/ and labels/ folders."""
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


def write_dataset_info(output_dir: str, name: str, zip_path: str, image_count: int) -> None:
    info = {
        "provider"      : "manual",
        "name"          : name,
        "license"       : "FILL IN — check dataset source",
        "homepage"      : "FILL IN — add source URL",
        "download_date" : datetime.date.today().isoformat(),
        "source_zip"    : zip_path,
        "image_count"   : image_count,
        "task"          : "detect",
        "notes"         : "Manually provided dataset. Update license and homepage fields.",
    }
    out_path = os.path.join(output_dir, "dataset_info.yaml")
    with open(out_path, "w") as f:
        yaml.dump(info, f, default_flow_style=False, sort_keys=False)
    print(f"\n  Written: {out_path}")
    print("  ⚠ ACTION REQUIRED: Open dataset_info.yaml and fill in 'license' and 'homepage'.")


def download(args) -> None:
    """Entry point called by download_dataset.py dispatcher."""
    if not args.zip:
        print("ERROR: --zip is required for the manual provider.")
        print("Example: --zip /path/to/dataset.zip")
        sys.exit(1)

    if not os.path.exists(args.zip):
        print(f"ERROR: ZIP file not found: {args.zip}")
        sys.exit(1)

    dataset_name = args.name or Path(args.zip).stem

    print(f"Extracting: {args.zip}")
    print(f"Dataset name: {dataset_name}")

    os.makedirs(TEMP_DIR, exist_ok=True)

    # Extract
    with zipfile.ZipFile(args.zip, "r") as z:
        z.extractall(TEMP_DIR)
    print(f"Extracted to: {TEMP_DIR}")

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
    else:
        # No images/ subdir — scan top-level
        for img in temp_path.glob("*"):
            if img.suffix.lower() in IMAGE_EXTS:
                shutil.move(str(img), str(images_out / img.name))
                image_count += 1

    if labels_src and labels_src != labels_out:
        for lbl in labels_src.rglob("*.txt"):
            shutil.move(str(lbl), str(labels_out / lbl.name))

    if image_count == 0:
        print("WARNING: No images found in the ZIP file.")
        print("Make sure the ZIP contains a YOLO-format dataset with images/ and labels/ folders.")

    # Copy data.yaml if present
    for yaml_file in temp_path.rglob("data.yaml"):
        shutil.copy2(yaml_file, temp_path / "data.yaml")
        break

    # Validate images
    import cv2
    valid_count = 0
    for img in (temp_path / "images").glob("*"):
        if img.suffix.lower() in IMAGE_EXTS:
            frame = cv2.imread(str(img))
            if frame is None:
                print(f"  [CORRUPT] {img.name} — removing")
                img.unlink(missing_ok=True)
                lbl = (temp_path / "labels") / (img.stem + ".txt")
                lbl.unlink(missing_ok=True)
            else:
                valid_count += 1

    print(f"Validation: {valid_count} valid images")

    # Move temp → raw/manual
    if os.path.exists(RAW_DIR):
        shutil.rmtree(RAW_DIR)
    shutil.move(TEMP_DIR, RAW_DIR)
    print(f"Moved to: {RAW_DIR}")

    write_dataset_info(RAW_DIR, dataset_name, args.zip, valid_count)

    print(f"\nManual import complete. {valid_count} images in {RAW_DIR}")
    print("Next: python scripts/merge_dataset.py")
