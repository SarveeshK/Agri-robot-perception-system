"""
prepare_dataset.py
==================
Reads from datasets/merged/, validates every image-label pair,
performs a stratified 80/20 train/val split, copies files into
datasets/processed/, and generates:
  - data.yaml             (consumed by train_model.py)
  - dataset_statistics.txt
  - coverage_report.txt   (ASCII bar chart)

Before overwriting processed/, it auto-archives the previous version
to datasets/archive/vN/.

No FiftyOne. No database. Pure filesystem + OpenCV.
"""

import os
import sys
import shutil
import random
import yaml
import cv2
from pathlib import Path
from collections import defaultdict


# ── Config ────────────────────────────────────────────────────────────────────

MERGED_DIR    = "datasets/merged"
PROCESSED_DIR = "datasets/processed"
ARCHIVE_DIR   = "datasets/archive"
CONFIG_PATH   = "config/class_mapping.yaml"
IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_RATIO   = 0.80
RANDOM_SEED   = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_class_names(config_path: str) -> list[str]:
    """Return class names ordered by id from class_mapping.yaml."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    classes = cfg["classes"]
    ordered = sorted(classes.items(), key=lambda x: x[1]["id"])
    return [name for name, _ in ordered]


def validate_image(img_path: Path) -> bool:
    """True if OpenCV can read the image and it has non-zero dimensions."""
    img = cv2.imread(str(img_path))
    return img is not None and img.shape[0] > 0 and img.shape[1] > 0


def validate_label(lbl_path: Path, num_classes: int) -> bool:
    """
    True if every line in the YOLO .txt file is valid detection format:
      class_id  x_center  y_center  width  height
    All values must be numeric; coords must be in [0.0, 1.0].
    Empty file = background image = valid.
    """
    try:
        with open(lbl_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                return False
            cls_id = int(parts[0])
            if cls_id < 0 or cls_id >= num_classes:
                return False
            coords = [float(v) for v in parts[1:]]
            if not all(0.0 <= c <= 1.0 for c in coords):
                return False
        return True
    except Exception:
        return False


def archive_processed(processed_dir: str, archive_dir: str) -> None:
    """Copy current processed/ into archive/vN/ before overwriting."""
    data_yaml = os.path.join(processed_dir, "data.yaml")
    if not os.path.exists(data_yaml):
        return                          # nothing to archive yet

    os.makedirs(archive_dir, exist_ok=True)
    existing = [
        d for d in os.listdir(archive_dir)
        if d.startswith("v") and os.path.isdir(os.path.join(archive_dir, d))
    ]
    version_num = len(existing) + 1
    dest = os.path.join(archive_dir, f"v{version_num}")
    shutil.copytree(processed_dir, dest)
    print(f"  Archived previous datasets/processed/ → archive/v{version_num}/")


def stratified_split(
    pairs: list[tuple],
    num_classes: int,
    ratio: float = SPLIT_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[list, list]:
    """
    Split image-label pairs so every class is represented in both
    train and val, even for rare classes.

    Groups images by dominant class (first annotation in label file),
    then splits each group independently at `ratio`.
    Images with no annotations go into a shared background pool.
    """
    random.seed(seed)
    groups: dict[int, list] = defaultdict(list)
    bg_pool: list = []

    for img_path, lbl_path in pairs:
        dominant = None
        if lbl_path and os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                dominant = int(lines[0].split()[0])
        if dominant is not None:
            groups[dominant].append((img_path, lbl_path))
        else:
            bg_pool.append((img_path, lbl_path))

    train_pairs: list = []
    val_pairs: list   = []

    for cls_id, cls_pairs in groups.items():
        random.shuffle(cls_pairs)
        cut = max(1, int(len(cls_pairs) * ratio))
        train_pairs.extend(cls_pairs[:cut])
        val_pairs.extend(cls_pairs[cut:])

    # Background pool: plain random split
    random.shuffle(bg_pool)
    cut = int(len(bg_pool) * ratio)
    train_pairs.extend(bg_pool[:cut])
    val_pairs.extend(bg_pool[cut:])

    return train_pairs, val_pairs


def copy_split(pairs: list[tuple], split: str, processed_dir: str) -> None:
    """Copy image + label files into processed/<split>/images|labels/."""
    img_dir = os.path.join(processed_dir, split, "images")
    lbl_dir = os.path.join(processed_dir, split, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for img_path, lbl_path in pairs:
        shutil.copy2(img_path, os.path.join(img_dir, img_path.name))
        dest_lbl = os.path.join(lbl_dir, img_path.stem + ".txt")
        if lbl_path and os.path.exists(lbl_path):
            shutil.copy2(lbl_path, dest_lbl)
        else:
            open(dest_lbl, "w").close()     # empty label = background


def count_annotations(pairs: list[tuple], num_classes: int) -> dict[int, int]:
    """Count annotations per class id across a list of (img, lbl) pairs."""
    counts: dict[int, int] = defaultdict(int)
    for _, lbl_path in pairs:
        if lbl_path and os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        cls_id = int(line.split()[0])
                        if 0 <= cls_id < num_classes:
                            counts[cls_id] += 1
    return counts


def generate_coverage_report(
    label_dir: str,
    class_names: list[str],
    report_path: str,
) -> dict[str, int]:
    """
    Write an ASCII bar chart showing annotation counts per class.
    Returns class_counts dict for downstream use.
    """
    class_counts: dict[str, int] = defaultdict(int)

    for lbl_file in Path(label_dir).glob("*.txt"):
        with open(lbl_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    cls_id = int(line.split()[0])
                    if cls_id < len(class_names):
                        class_counts[class_names[cls_id]] += 1

    total        = sum(class_counts.values())
    max_count    = max(class_counts.values(), default=1)
    bar_width    = 24
    low_warnings = []

    lines = [
        "AgriVision Dataset Coverage Report",
        "=" * 42,
        f"Total Annotations : {total:,}",
        "",
        "Class Distribution",
        "─" * 55,
    ]

    for cls in class_names:
        count = class_counts.get(cls, 0)
        bars  = int((count / max_count) * bar_width) if max_count else 0
        pct   = (count / total * 100) if total else 0.0
        warn  = "  ⚠ low" if count < 100 else ""
        lines.append(f"{cls:<15} {'█' * bars:<24} {count:>5}  ({pct:>5.1f}%){warn}")
        if count < 100:
            low_warnings.append((cls, count))

    if low_warnings:
        lines += ["", "Imbalance Warnings", "─" * 34]
        for cls, count in low_warnings:
            lines.append(f"  {cls}: {count} annotations — collect more data")

    lines += ["", "Recommended Next Steps", "─" * 34]
    if low_warnings:
        for cls, _ in low_warnings:
            lines.append(f"  - Download more {cls} images (check dataset_sources.yaml)")
    else:
        lines.append("  - All classes have sufficient data. Proceed to training.")

    content = "\n".join(lines) + "\n"
    with open(report_path, "w") as f:
        f.write(content)

    # Also print to console
    print("\n" + content)
    return dict(class_counts)


def generate_data_yaml(
    processed_dir: str,
    class_names: list[str],
) -> None:
    """
    Write data.yaml in dict format (explicit class ordering).
    Consumed unchanged by train_model.py via ultralytics.
    """
    data = {
        "path"  : os.path.abspath(processed_dir),
        "train" : "train/images",
        "val"   : "val/images",
        "nc"    : len(class_names),
        "names" : {i: name for i, name in enumerate(class_names)},
    }
    out_path = os.path.join(processed_dir, "data.yaml")
    with open(out_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"Generated: {out_path}")


def generate_statistics(
    processed_dir : str,
    class_names   : list[str],
    train_pairs   : list[tuple],
    val_pairs     : list[tuple],
    corrupt_images: int,
    missing_labels: int,
    invalid_labels: int,
) -> None:
    """Write dataset_statistics.txt with per-class counts for each split."""
    nc           = len(class_names)
    train_counts = count_annotations(train_pairs, nc)
    val_counts   = count_annotations(val_pairs, nc)

    lines = [
        "Dataset Statistics",
        "=" * 40,
        f"Total Valid Images  : {len(train_pairs) + len(val_pairs)}",
        f"Training Images     : {len(train_pairs)}",
        f"Validation Images   : {len(val_pairs)}",
        "",
        f"Corrupt Images      : {corrupt_images}",
        f"Missing Labels      : {missing_labels}",
        f"Invalid Labels      : {invalid_labels}",
        "",
        "Class Distribution — Train",
        "─" * 30,
    ]
    for i, name in enumerate(class_names):
        lines.append(f"  {name:<15} : {train_counts.get(i, 0)}")

    lines += ["", "Class Distribution — Val", "─" * 30]
    for i, name in enumerate(class_names):
        lines.append(f"  {name:<15} : {val_counts.get(i, 0)}")

    out_path = os.path.join(processed_dir, "dataset_statistics.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Generated: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def prepare_data() -> None:
    print("=" * 50)
    print("AgriVision — prepare_dataset.py")
    print("=" * 50)

    # ── 1. Load config ────────────────────────────────
    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: {CONFIG_PATH} not found.")
        sys.exit(1)

    class_names = load_class_names(CONFIG_PATH)
    num_classes = len(class_names)
    print(f"\nClasses ({num_classes}): {class_names}")

    # ── 2. Locate merged images ───────────────────────
    merged_images_dir = os.path.join(MERGED_DIR, "images")
    merged_labels_dir = os.path.join(MERGED_DIR, "labels")

    if not os.path.exists(merged_images_dir):
        print("\n0 images found. datasets/merged/images/ does not exist.")
        print("Run merge_dataset.py first.")
        return

    all_images = sorted([
        f for f in Path(merged_images_dir).iterdir()
        if f.suffix.lower() in IMAGE_EXTS
    ])

    if not all_images:
        print("\n0 images found in datasets/merged/images/. Exiting cleanly.")
        return

    print(f"\nFound {len(all_images)} images. Validating...")

    # ── 3. Validate ───────────────────────────────────
    valid_pairs  : list[tuple] = []
    corrupt_images = missing_labels = invalid_labels = 0

    for img_path in all_images:
        lbl_path = Path(merged_labels_dir) / (img_path.stem + ".txt")

        if not validate_image(img_path):
            print(f"  [CORRUPT IMAGE]  {img_path.name}")
            corrupt_images += 1
            continue

        if not lbl_path.exists():
            print(f"  [MISSING LABEL]  {lbl_path.name}")
            missing_labels += 1
            continue

        if not validate_label(lbl_path, num_classes):
            print(f"  [INVALID LABEL]  {lbl_path.name}")
            invalid_labels += 1
            continue

        valid_pairs.append((img_path, lbl_path))

    print(f"\nValidation complete:")
    print(f"  Valid pairs      : {len(valid_pairs)}")
    print(f"  Corrupt images   : {corrupt_images}")
    print(f"  Missing labels   : {missing_labels}")
    print(f"  Invalid labels   : {invalid_labels}")

    if not valid_pairs:
        print("\nNo valid pairs found. Nothing to prepare. Exiting.")
        return

    # ── 4. Archive previous processed/ ───────────────
    print("\nArchiving previous dataset...")
    archive_processed(PROCESSED_DIR, ARCHIVE_DIR)

    # ── 5. Clean + create output directories ─────────
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ── 6. Stratified split ───────────────────────────
    print("\nPerforming stratified 80/20 split...")
    train_pairs, val_pairs = stratified_split(valid_pairs, num_classes)
    print(f"  Train : {len(train_pairs)} images")
    print(f"  Val   : {len(val_pairs)} images")

    if not val_pairs:
        print("  WARNING: Val set is empty (too few images for split).")
        print("  Training set used as val fallback.")
        val_pairs = train_pairs

    # ── 7. Copy files ─────────────────────────────────
    print("\nCopying files to processed/...")
    copy_split(train_pairs, "train", PROCESSED_DIR)
    copy_split(val_pairs,   "val",   PROCESSED_DIR)
    print("  Done.")

    # ── 8. Generate outputs ───────────────────────────
    print("\nGenerating outputs...")
    generate_data_yaml(PROCESSED_DIR, class_names)
    generate_statistics(
        PROCESSED_DIR, class_names,
        train_pairs, val_pairs,
        corrupt_images, missing_labels, invalid_labels,
    )

    coverage_path = os.path.join(PROCESSED_DIR, "coverage_report.txt")
    generate_coverage_report(merged_labels_dir, class_names, coverage_path)
    print(f"Generated: {coverage_path}")

    print("\n" + "=" * 50)
    print("Dataset preparation complete.")
    print("Next: python scripts/audit_dataset.py")
    print("=" * 50)


if __name__ == "__main__":
    prepare_data()
