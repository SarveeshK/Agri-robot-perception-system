"""
merge_dataset.py
================
Reads all datasets from datasets/raw/<provider>/, harmonizes their labels
into the 7 canonical classes defined in class_mapping.yaml, deduplicates
images using SHA256 hashing, tracks provenance in metadata.csv, and writes
everything to datasets/merged/.

No FiftyOne. No database. Pure filesystem.

Output:
    datasets/merged/
        images/                  ← all deduplicated images
        labels/                  ← harmonized YOLO .txt label files
        metadata.csv             ← provenance: filename, provider, label, license
        dataset_quality_report.txt

Algorithm per provider:
    1. Read dataset_info.yaml   → provider name, license, homepage
    2. Read data.yaml           → OIDv7/source label → canonical class mapping
    3. Read class_mapping.yaml  → source_labels per provider (fallback lookup)
    4. For each image:
         a. Compute SHA256 hash
         b. If hash seen → SKIP (duplicate)
         c. For each label line → map source label → canonical class id
         d. Skip unmapped annotations (log warning)
         e. Copy image + write harmonized labels to merged/
         f. Append row to metadata.csv
    5. Write dataset_quality_report.txt
"""

import os
import sys
import csv
import hashlib
import shutil
import yaml
import datetime
from pathlib import Path
from collections import defaultdict
import json


# ── Config paths ──────────────────────────────────────────────────────────────
RAW_DIR        = "datasets/raw"
MERGED_DIR     = "datasets/merged"
CONFIG_PATH    = "config/class_mapping.yaml"
SOURCES_PATH   = "config/dataset_sources.yaml"
IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".bmp"}


# ── Helpers ───────────────────────────────────────────────────────────────────



def load_class_mapping(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_sources(sources_path: str) -> dict:
    if not os.path.exists(sources_path):
        return {}
    with open(sources_path, "r") as f:
        return yaml.safe_load(f).get("providers", {})


def build_label_lookup(class_cfg: dict, provider_name: str, base_provider: str) -> dict[str, int]:
    """
    Build {source_label_lowercase: canonical_class_id} for a given provider.
    Reads source_labels.<provider_name> (dataset-specific) and source_labels.<base_provider> (fallback) from class_mapping.yaml.
    """
    lookup: dict[str, int] = {}
    for cls_name, cls_data in class_cfg["classes"].items():
        cls_id = cls_data["id"]
        source_labels_dict = cls_data.get("source_labels", {})
        
        # 1. Base provider fallback (e.g., roboflow)
        base_labels = source_labels_dict.get(base_provider, [])
        for label in base_labels:
            lookup[label.lower()] = cls_id
            
        # 2. Exact provider override (e.g., roboflow_rocks-detection-govch)
        # This overwrites the base mapping if there's a conflict
        exact_labels = source_labels_dict.get(provider_name, [])
        for label in exact_labels:
            lookup[label.lower()] = cls_id
            
    return lookup


def load_provider_info(provider_dir: Path) -> dict:
    """Load dataset_info.yaml from a provider directory."""
    info_path = provider_dir / "dataset_info.yaml"
    if not info_path.exists():
        return {
            "provider": provider_dir.name,
            "name"    : provider_dir.name,
            "license" : "Unknown",
            "homepage": "",
        }
    with open(info_path, "r") as f:
        return yaml.safe_load(f)


def load_provider_data_yaml(provider_dir: Path) -> list[str]:
    """Load data.yaml which contains the names of the source classes."""
    data_yaml = provider_dir / "data.yaml"
    if not data_yaml.exists():
        return []
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)
    return data.get("names", [])


def process_label_file(
    lbl_path    : Path,
    label_lookup: dict[str, int],
    provider_names: list[str],
    class_cfg   : dict,
    provider_name: str,
) -> tuple[list[str], list[str]]:
    """
    Read source label file, map each annotation to canonical class id.
    Returns (harmonized_lines, unmapped_labels).
    """
    if not lbl_path.exists():
        return [], []

    harmonized   : list[str] = []
    unmapped     : list[str] = []

    with open(lbl_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        src_class_id = int(parts[0])
        coords       = parts[1:]

        canonical_id = None
        src_label_name = None

        if isinstance(provider_names, dict):
            if src_class_id in provider_names:
                src_label_name = str(provider_names[src_class_id]).lower()
            elif str(src_class_id) in provider_names:
                src_label_name = str(provider_names[str(src_class_id)]).lower()
        elif isinstance(provider_names, list):
            if src_class_id < len(provider_names):
                src_label_name = str(provider_names[src_class_id]).lower()

        if src_label_name is not None:
            canonical_id = label_lookup.get(src_label_name)

        if canonical_id is None:
            # Fallback only for manual datasets
            if provider_name == "manual" and 0 <= src_class_id < len(class_cfg["classes"]):
                canonical_id = src_class_id

        if canonical_id is not None:
            harmonized.append(f"{canonical_id} {' '.join(coords)}")
        else:
            unmapped.append(str(src_class_id))

    return harmonized, unmapped


# ── Main ──────────────────────────────────────────────────────────────────────

def merge_datasets() -> None:
    print("=" * 50)
    print("AgriVision — merge_dataset.py")
    print("=" * 50)

    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: {CONFIG_PATH} not found.")
        sys.exit(1)

    class_cfg   = load_class_mapping(CONFIG_PATH)
    sources_cfg = load_sources(SOURCES_PATH)

    # Provider priority: lower = higher priority for duplicate resolution
    provider_priority = {
        name: cfg.get("priority", 99)
        for name, cfg in sources_cfg.items()
    }

    # Create output dirs
    merged_images = os.path.join(MERGED_DIR, "images")
    merged_labels = os.path.join(MERGED_DIR, "labels")
    os.makedirs(merged_images, exist_ok=True)
    os.makedirs(merged_labels, exist_ok=True)

    raw_path = Path(RAW_DIR)
    if not raw_path.exists():
        print("\n0 images merged. datasets/raw/ does not exist. Exiting cleanly.")
        return

    provider_dirs = sorted(
        [d for d in raw_path.iterdir() if d.is_dir() and d.name != "temp"],
        key=lambda d: provider_priority.get(d.name.split("_")[0], 99),
    )

    if not provider_dirs:
        print("\n0 images merged. No provider folders in datasets/raw/. Exiting cleanly.")
        return

    # State tracking
    # Load deduplication skip list if it exists
    skip_list_path = Path("datasets/processed/duplicates_to_skip.json")
    skip_list = set()
    if skip_list_path.exists():
        with open(skip_list_path, "r") as f:
            skip_list = set(json.load(f))

    metadata_rows   : list[dict]     = []
    provider_counts : dict[str, int] = defaultdict(int)
    dup_count       = 0
    unmapped_total  : dict[str, int] = defaultdict(int)
    corrupt_count   = 0

    print(f"\nProcessing {len(provider_dirs)} provider(s): {[d.name for d in provider_dirs]}\n")

    for provider_dir in provider_dirs:
        # Strict directory validation
        info_yaml = provider_dir / "dataset_info.yaml"
        data_yaml = provider_dir / "data.yaml"
        classes_txt = provider_dir / "classes.txt"
        
        if not info_yaml.exists():
            print(f"  [{provider_dir.name}] Skipping (missing dataset_info.yaml)")
            continue
            
        if not (data_yaml.exists() or classes_txt.exists()):
            print(f"  [{provider_dir.name}] Skipping (missing data.yaml or classes.txt)")
            continue
            
        provider_name = provider_dir.name
        base_provider = provider_name.split("_")[0]
        info          = load_provider_info(provider_dir)
        data_mapping  = load_provider_data_yaml(provider_dir)
        label_lookup  = build_label_lookup(class_cfg, provider_name, base_provider)

        images_dir = provider_dir / "images"
        labels_dir = provider_dir / "labels"

        if not images_dir.exists():
            print(f"  [{provider_name}] No images/ directory — skipping.")
            continue

        image_files = sorted([
            f for f in images_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTS
        ])

        print(f"[{provider_name}] Processing {len(image_files)} images...")

        for img_path in image_files:
            # Check skip list
            rel_path = f"{provider_name}/images/{img_path.name}"
            if rel_path in skip_list:
                dup_count += 1
                continue

            # Process label
            lbl_path              = labels_dir / (img_path.stem + ".txt")
            harmonized, unmapped  = process_label_file(
                lbl_path, label_lookup, data_mapping, class_cfg, provider_name
            )

            for u in unmapped:
                unmapped_total[u] += 1

            # Copy image
            dest_img = os.path.join(merged_images, img_path.name)
            # Handle name collision from different providers
            if os.path.exists(dest_img):
                stem = img_path.stem + f"_{provider_name}"
                dest_img = os.path.join(merged_images, stem + img_path.suffix)
            shutil.copy2(img_path, dest_img)

            # Write harmonized label
            dest_name    = Path(dest_img).stem
            dest_lbl     = os.path.join(merged_labels, dest_name + ".txt")
            with open(dest_lbl, "w") as f:
                f.write("\n".join(harmonized) + ("\n" if harmonized else ""))

            provider_counts[provider_name] += 1

            # Metadata row
            # Determine primary source label for this image (first annotation)
            original_label = "background"
            if harmonized:
                canonical_id = int(harmonized[0].split()[0])
                class_names  = [
                    name for name, _ in
                    sorted(class_cfg["classes"].items(), key=lambda x: x[1]["id"])
                ]
                original_label = class_names[canonical_id] if canonical_id < len(class_names) else "unknown"

            metadata_rows.append({
                "filename"       : Path(dest_img).name,
                "provider"       : provider_name,
                "original_label" : original_label,
                "canonical_class": original_label,
                "license"        : info.get("license", "Unknown"),
                "homepage"       : info.get("homepage", ""),
                "sha256"         : img_hash,
            })

        print(f"  Done: {provider_counts[provider_name]} images added from {provider_name}")

    # ── Write metadata.csv ────────────────────────────
    csv_path = os.path.join(MERGED_DIR, "metadata.csv")
    if metadata_rows:
        fieldnames = ["filename", "provider", "original_label", "canonical_class",
                      "license", "homepage", "sha256"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata_rows)
        print(f"\nWritten: {csv_path} ({len(metadata_rows)} rows)")

    # ── Write quality report ──────────────────────────
    total_merged = sum(provider_counts.values())
    report_lines = [
        "AgriVision Dataset Merge Report",
        "=" * 40,
        f"Run Date : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Images by Provider",
        "─" * 30,
    ]
    for p, count in sorted(provider_counts.items(), key=lambda x: -x[1]):
        report_lines.append(f"  {p:<15} : {count}")
    report_lines += [
        "",
        f"Duplicates Removed      : {dup_count}",
        f"Unmapped Annotations    : {sum(unmapped_total.values())}",
        "",
        f"Final Image Count       : {total_merged}",
    ]

    if unmapped_total:
        report_lines += ["", "Unmapped Labels (consider adding to class_mapping.yaml)", "─" * 50]
        for label, count in sorted(unmapped_total.items(), key=lambda x: -x[1]):
            report_lines.append(f"  class_id '{label}' → {count} annotations skipped")

    report_content = "\n".join(report_lines) + "\n"
    report_path    = os.path.join(MERGED_DIR, "dataset_quality_report.txt")
    with open(report_path, "w") as f:
        f.write(report_content)

    print("\n" + report_content)
    print(f"Written: {report_path}")
    print("\n" + "=" * 50)
    print(f"Merge complete. {total_merged} images in datasets/merged/")
    print("Next: python scripts/prepare_dataset.py")
    print("=" * 50)


if __name__ == "__main__":
    merge_datasets()
