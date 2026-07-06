"""
audit_dataset.py
=================
Runs a comprehensive audit on datasets/processed/ (or a versioned archive).
Produces outputs/dataset_audit.txt and exits with:
    0 → PASSED
    1 → PASSED WITH WARNINGS
    2 → FAILED

Run this before every training session.

Usage:
    python scripts/audit_dataset.py                  # audit processed/
    python scripts/audit_dataset.py --version v1     # audit archive/v1/
"""

import os
import sys
import argparse
import yaml
import cv2
from pathlib import Path
from collections import defaultdict


CONFIG_PATH = "config/class_mapping.yaml"
OUTPUTS_DIR = "outputs"
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_class_names(config_path: str) -> list[str]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    classes = cfg["classes"]
    ordered = sorted(classes.items(), key=lambda x: x[1]["id"])
    return [name for name, _ in ordered]


def bar(count: int, max_count: int, width: int = 24) -> str:
    n = int((count / max_count) * width) if max_count > 0 else 0
    return "█" * n


def validate_label_line(line: str, num_classes: int) -> str | None:
    """Returns None if valid, error string if invalid."""
    parts = line.strip().split()
    if len(parts) != 5:
        return f"wrong field count ({len(parts)}, expected 5)"
    try:
        cls_id = int(parts[0])
        coords = [float(v) for v in parts[1:]]
    except ValueError:
        return "non-numeric value"
    if cls_id < 0 or cls_id >= num_classes:
        return f"class_id {cls_id} out of range [0, {num_classes-1}]"
    if not all(0.0 <= c <= 1.0 for c in coords):
        return f"coordinate out of [0,1] range: {coords}"
    return None


# ── Audit Logic ───────────────────────────────────────────────────────────────

def audit(dataset_dir: str, class_names: list[str]) -> dict:
    """
    Audit a processed dataset directory.
    Returns a results dict consumed by the report generator.
    """
    num_classes = len(class_names)
    results = {
        "dataset_dir"   : dataset_dir,
        "splits"        : {},
        "errors"        : [],
        "warnings"      : [],
    }

    for split in ["train", "val"]:
        split_dir    = os.path.join(dataset_dir, split)
        images_dir   = os.path.join(split_dir, "images")
        labels_dir   = os.path.join(split_dir, "labels")

        split_results = {
            "image_count"   : 0,
            "corrupt_images": 0,
            "missing_labels": 0,
            "empty_labels"  : 0,
            "invalid_labels": 0,
            "class_counts"  : defaultdict(int),
            "out_of_range"  : 0,
        }

        if not os.path.exists(images_dir):
            results["errors"].append(f"{split}/images/ directory missing")
            results["splits"][split] = split_results
            continue

        image_files = sorted([
            f for f in Path(images_dir).iterdir()
            if f.suffix.lower() in IMAGE_EXTS
        ])
        split_results["image_count"] = len(image_files)

        for img_path in image_files:
            # Validate image
            img = cv2.imread(str(img_path))
            if img is None:
                split_results["corrupt_images"] += 1
                results["errors"].append(f"Corrupt image: {split}/images/{img_path.name}")
                continue

            # Validate label
            lbl_path = Path(labels_dir) / (img_path.stem + ".txt")
            if not lbl_path.exists():
                split_results["missing_labels"] += 1
                results["errors"].append(f"Missing label: {split}/labels/{img_path.stem}.txt")
                continue

            with open(lbl_path, "r") as f:
                lines = [l.strip() for l in f if l.strip()]

            if not lines:
                split_results["empty_labels"] += 1
                # Empty = background image, not an error
                continue

            for line in lines:
                err = validate_label_line(line, num_classes)
                if err:
                    split_results["invalid_labels"] += 1
                    results["errors"].append(
                        f"Invalid label in {split}/labels/{lbl_path.name}: {err}"
                    )
                else:
                    cls_id = int(line.split()[0])
                    split_results["class_counts"][cls_id] += 1

        results["splits"][split] = split_results

    # Check stratification — every class with > 0 train samples should appear in val
    train_classes = set(results["splits"].get("train", {}).get("class_counts", {}).keys())
    val_classes   = set(results["splits"].get("val",   {}).get("class_counts", {}).keys())
    missing_in_val = train_classes - val_classes
    for cls_id in missing_in_val:
        cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        results["warnings"].append(
            f"Class '{cls_name}' present in train but MISSING from val — "
            "model cannot be evaluated on this class"
        )

    # Check low-count classes
    for split_name, split_data in results["splits"].items():
        for cls_id, count in split_data["class_counts"].items():
            if count < 10:
                cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                results["warnings"].append(
                    f"[{split_name}] '{cls_name}': only {count} annotations (< 10)"
                )

    return results


def generate_report(results: dict, class_names: list[str]) -> tuple[str, int]:
    """Generate human-readable report. Returns (report_text, exit_code)."""
    import datetime

    lines = [
        "AgriVision Dataset Audit",
        "=" * 50,
        f"Date      : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Directory : {results['dataset_dir']}",
        "",
    ]

    total_images = sum(
        s.get("image_count", 0) for s in results["splits"].values()
    )
    lines += [
        "Image Counts",
        "─" * 30,
    ]
    for split, data in results["splits"].items():
        lines.append(f"  {split:<8} : {data.get('image_count', 0)}")
    lines.append(f"  {'total':<8} : {total_images}")

    lines += ["", "Label Health", "─" * 30]
    for split, data in results["splits"].items():
        lines.append(f"  [{split}]")
        lines.append(f"    Corrupt images  : {data.get('corrupt_images', 0)}")
        lines.append(f"    Missing labels  : {data.get('missing_labels', 0)}")
        lines.append(f"    Empty labels    : {data.get('empty_labels', 0)}  (background images)")
        lines.append(f"    Invalid labels  : {data.get('invalid_labels', 0)}")

    # Class distribution per split with bar chart
    for split, data in results["splits"].items():
        counts   = data.get("class_counts", {})
        max_cnt  = max(counts.values(), default=1)
        lines   += ["", f"Class Distribution — {split}", "─" * 45]
        for i, cls_name in enumerate(class_names):
            count = counts.get(i, 0)
            warn  = "  ⚠" if count < 10 else ""
            lines.append(
                f"  {cls_name:<15} {bar(count, max_cnt):<24} {count:>5}{warn}"
            )

    # Stratification check
    lines += ["", "Split Verification", "─" * 30]
    if not results["warnings"]:
        lines.append("  All classes present in both splits ✅")
    else:
        for w in results["warnings"]:
            lines.append(f"  ⚠ {w}")

    # Errors
    lines += ["", "Errors", "─" * 30]
    if results["errors"]:
        for e in results["errors"][:20]:      # cap at 20
            lines.append(f"  ✗ {e}")
        if len(results["errors"]) > 20:
            lines.append(f"  ... and {len(results['errors']) - 20} more errors")
    else:
        lines.append("  No errors found ✅")

    # Verdict
    has_errors   = bool(results["errors"])
    has_warnings = bool(results["warnings"])
    critical_errors = [
        e for e in results["errors"]
        if "Corrupt" in e or "Missing label" in e or "Invalid label" in e
    ]

    lines += ["", "=" * 50]
    if critical_errors:
        verdict  = "FAILED"
        exit_code = 2
        lines.append(f"Audit Result : {verdict}")
        lines.append("Action       : Fix all errors before training.")
    elif has_warnings:
        verdict  = "PASSED WITH WARNINGS"
        exit_code = 1
        lines.append(f"Audit Result : {verdict}")
        lines.append("Action       : Review warnings. Training may proceed with caution.")
    else:
        verdict  = "PASSED"
        exit_code = 0
        lines.append(f"Audit Result : {verdict} ✅")
        lines.append("Action       : Dataset is ready for training.")

    lines.append(f"Errors       : {len(results['errors'])}")
    lines.append(f"Warnings     : {len(results['warnings'])}")

    return "\n".join(lines) + "\n", exit_code


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AgriVision Dataset Audit — validates processed/ before training"
    )
    parser.add_argument(
        "--version",
        default=None,
        metavar="vN",
        help="Audit an archived version, e.g. --version v1",
    )
    args = parser.parse_args()

    if args.version:
        dataset_dir = os.path.join("datasets", "archive", args.version)
    else:
        dataset_dir = "datasets/processed"

    if not os.path.exists(dataset_dir):
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        sys.exit(2)

    data_yaml = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(data_yaml):
        print(f"ERROR: data.yaml not found in {dataset_dir}")
        print("Run prepare_dataset.py first.")
        sys.exit(2)

    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: {CONFIG_PATH} not found.")
        sys.exit(2)

    print(f"Auditing: {dataset_dir}\n")
    class_names = load_class_names(CONFIG_PATH)
    results     = audit(dataset_dir, class_names)
    report, exit_code = generate_report(results, class_names)

    print(report)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUTS_DIR, "dataset_audit.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved: {report_path}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
