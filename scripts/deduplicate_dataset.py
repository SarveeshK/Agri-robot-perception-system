#!/usr/bin/env python3
import os
import sys
import yaml
import hashlib
import csv
import json
import argparse
import time
from pathlib import Path
from PIL import Image
import imagehash
import pybktree
from datetime import datetime
from tqdm import tqdm

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

class ImageMeta:
    def __init__(self, path, provider_name, priority_str, last_verified, ann_count, pixels, filesize, canonical_classes, phash=None, sha256=None):
        self.path = path
        self.provider_name = provider_name
        self.ann_count = ann_count
        self.pixels = pixels
        self.filesize = filesize
        self.canonical_classes = canonical_classes
        self.phash = phash
        self.sha256 = sha256
        
        # Parse priority (e.g. P0 -> 3, P1 -> 2, P2 -> 1, else 0)
        p = str(priority_str).upper()
        if p == "P0": self.priority = 3
        elif p == "P1": self.priority = 2
        elif p == "P2": self.priority = 1
        else: self.priority = 0
        
        # Parse recency
        try:
            # simple timestamp
            self.recency = datetime.fromisoformat(str(last_verified).replace("Z", "+00:00")).timestamp()
        except:
            try:
                self.recency = datetime.strptime(str(last_verified), "%Y-%m-%d").timestamp()
            except:
                self.recency = 0

    def compute_score(self, weights):
        return (
            self.ann_count * weights.get("annotations", 1000) +
            self.pixels * weights.get("pixels", 1) +
            self.filesize * weights.get("filesize", 0.001) +
            self.priority * weights.get("priority", 5000) +
            self.recency * weights.get("recency", 0.001) # recency is a huge timestamp, scale it down or just let it be a tiny factor if weight is low
        )
        
    def __repr__(self):
        return f"ImageMeta({self.path.name})"

def tree_dist(a: ImageMeta, b: ImageMeta) -> int:
    return a.phash - b.phash

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run without generating duplicates_to_skip.json")
    args = parser.parse_args()

    print("==================================================")
    print("AgriVision — deduplicate_dataset.py")
    print("==================================================")

    ssot = load_yaml("config/dataset_sources.yaml")
    dedup_cfg = ssot.get("deduplication", {})
    if not dedup_cfg.get("enabled", False):
        print("Deduplication is disabled in config.")
        return
        
    threshold = dedup_cfg.get("phash_threshold", 5)
    weights = dedup_cfg.get("weights", {})
    
    class_mapping = load_yaml("config/class_mapping.yaml")
    
    # Pre-build lookup: dataset -> native_label_lowercase -> canonical_class_name
    ds_label_map = {}
    for ds in ssot.get("active_datasets", []):
        provider_name = ds["id"] # wait, the folder is e.g. roboflow_{project} or openimages. We use the folder name.
        # Actually we can just build a flat lookup for each known provider_dir name.
        pass

    # A better way to map is to use the logic from merge_dataset:
    def build_label_lookup(provider_name: str, base_provider: str) -> dict[str, str]:
        lookup = {}
        for canonical_name, cls_data in class_mapping["classes"].items():
            source_labels_dict = cls_data.get("source_labels", {})
            base_labels = source_labels_dict.get(base_provider, [])
            for label in base_labels:
                lookup[label.lower()] = canonical_name
            exact_labels = source_labels_dict.get(provider_name, [])
            for label in exact_labels:
                lookup[label.lower()] = canonical_name
        return lookup

    # 1. Discover all images
    print("Scanning datasets/raw...")
    raw_dir = Path("datasets/raw")
    if not raw_dir.exists(): return
    
    all_images = []
    
    # We also need dataset_sources mapping to get priority and recency
    ds_metadata = {}
    for ds in ssot.get("active_datasets", []):
        if ds.get("provider") == "roboflow":
            folder = f"roboflow_{ds['project']}"
        elif ds.get("provider") == "kaggle":
            folder = f"kaggle_{ds['dataset'].split('/')[-1]}"
        else:
            folder = ds.get("provider")
        ds_metadata[folder] = {
            "priority": ds.get("priority", "P2"),
            "last_verified": ds.get("last_verified", 0)
        }
        
    # Gather image metadata
    t0 = time.time()
    for provider_dir in raw_dir.iterdir():
        if not provider_dir.is_dir() or provider_dir.name in ["temp", "legacy"]:
            continue
            
        images_dir = provider_dir / "images"
        labels_dir = provider_dir / "labels"
        data_yaml_path = provider_dir / "data.yaml"
        
        if not images_dir.exists(): continue
        
        provider_name = provider_dir.name
        base_provider = provider_name.split("_")[0]
        label_lookup = build_label_lookup(provider_name, base_provider)
        
        # Load data.yaml to map YOLO integer IDs to string names
        yolo_names = []
        if data_yaml_path.exists():
            with open(data_yaml_path, "r") as f:
                d = yaml.safe_load(f)
                yolo_names = d.get("names", [])
        
        meta = ds_metadata.get(provider_name, {"priority": "P2", "last_verified": 0})
        
        for img_path in images_dir.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]: continue
            
            # Count annotations and find canonical classes
            ann_count = 0
            canonical_classes = set()
            lbl_path = labels_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                with open(lbl_path, "r") as f:
                    lines = f.read().strip().split('\n')
                    for line in lines:
                        if not line.strip(): continue
                        ann_count += 1
                        try:
                            parts = line.strip().split()
                            src_class_id = int(parts[0])
                            
                            src_label_name = None
                            if isinstance(yolo_names, dict):
                                if src_class_id in yolo_names:
                                    src_label_name = str(yolo_names[src_class_id]).lower()
                                elif str(src_class_id) in yolo_names:
                                    src_label_name = str(yolo_names[str(src_class_id)]).lower()
                            elif isinstance(yolo_names, list):
                                if src_class_id < len(yolo_names):
                                    src_label_name = str(yolo_names[src_class_id]).lower()
                                    
                            if src_label_name and src_label_name in label_lookup:
                                canonical_classes.add(label_lookup[src_label_name])
                        except:
                            pass
                            
            # Get image size and filesize
            filesize = os.path.getsize(img_path)
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                    pixels = w * h
                    phash = imagehash.phash(im)
            except:
                continue
                
            sha256 = get_sha256(str(img_path))
            
            img_meta = ImageMeta(
                path=img_path,
                provider_name=provider_name,
                priority_str=meta["priority"],
                last_verified=meta["last_verified"],
                ann_count=ann_count,
                pixels=pixels,
                filesize=filesize,
                canonical_classes=canonical_classes,
                phash=phash,
                sha256=sha256
            )
            all_images.append(img_meta)
            
    t1 = time.time()
    time_hash = t1 - t0
    print(f"Loaded {len(all_images)} images and computed hashes in {time_hash:.2f}s")
    
    # Score all images
    for img in all_images:
        img.score = img.compute_score(weights)
        
    # Sort descending by score so we always keep the first one we encounter
    all_images.sort(key=lambda x: x.score, reverse=True)
    
    exact_removed = 0
    near_removed = 0
    duplicates = [] # list of dicts for report
    
    # STAGE 1: Exact Duplicates (SHA256)
    t2 = time.time()
    seen_sha = {}
    stage2_images = []
    
    for img in all_images:
        if img.sha256 in seen_sha:
            kept_img = seen_sha[img.sha256]
            exact_removed += 1
            duplicates.append({
                "original": kept_img.path.name,
                "duplicate": img.path.name,
                "type": "sha256",
                "distance": 0,
                "kept_reason": "higher_score",
                "kept_dataset": kept_img.provider_name,
                "removed_dataset": img.provider_name,
                "duplicate_path": str(img.path)
            })
        else:
            seen_sha[img.sha256] = img
            stage2_images.append(img)
            
    t3 = time.time()
    time_exact = t3 - t2
    print(f"Stage 1 (Exact): removed {exact_removed} duplicates in {time_exact:.2f}s")
    
    # STAGE 2: Near Duplicates (BK-Tree)
    t4 = time.time()
    bktree = pybktree.BKTree(tree_dist)
    
    # Since stage2_images is already sorted by score (descending), 
    # we just insert into BKTree if no match is found. If match is found, it's a duplicate.
    
    for img in tqdm(stage2_images, desc="Stage 2 (BKTree)"):
        # find matches <= threshold
        matches = bktree.find(img, threshold)
        
        is_dup = False
        kept_by = None
        min_dist = 999
        
        for dist, match_img in matches:
            if dist <= 2:
                # Very High Confidence
                is_dup = True
                kept_by = match_img
                min_dist = dist
                break
            elif 3 <= dist <= threshold:
                # High Confidence - Semantic Check
                if img.canonical_classes == match_img.canonical_classes:
                    is_dup = True
                    kept_by = match_img
                    min_dist = dist
                    break
                    
        if is_dup:
            near_removed += 1
            duplicates.append({
                "original": kept_by.path.name,
                "duplicate": img.path.name,
                "type": "phash",
                "distance": min_dist,
                "kept_reason": "higher_score",
                "kept_dataset": kept_by.provider_name,
                "removed_dataset": img.provider_name,
                "duplicate_path": str(img.path)
            })
        else:
            bktree.add(img)
            
    t5 = time.time()
    time_bktree = t5 - t4
    print(f"Stage 2 (Near) : removed {near_removed} duplicates in {time_bktree:.2f}s")
    
    time_total = t5 - t0
    
    # Generate CSV Report
    report_path = Path("datasets/processed/duplicate_report.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["original", "duplicate", "type", "distance", "kept_reason", "kept_dataset", "removed_dataset"])
        writer.writeheader()
        for d in duplicates:
            writer.writerow({k: v for k, v in d.items() if k != "duplicate_path"})
            
    print(f"Report written to: {report_path}")
    
    if args.dry_run:
        print("DRY RUN: skipping generation of duplicates_to_skip.json")
    else:
        skip_list = []
        for d in duplicates:
            # relative path from raw_dir e.g. roboflow_yolov8tree/images/img.jpg
            p = Path(d["duplicate_path"])
            rel_path = f"{p.parent.parent.name}/{p.parent.name}/{p.name}"
            skip_list.append(rel_path)
            
        skip_path = Path("datasets/processed/duplicates_to_skip.json")
        with open(skip_path, "w") as f:
            json.dump(skip_list, f, indent=2)
        print(f"Skip list written to: {skip_path}")

    # Write metrics for build_dataset.py to read
    metrics = {
        "enabled": True,
        "exact_removed": exact_removed,
        "near_removed": near_removed,
        "threshold": threshold,
        "algorithm": "BKTree",
        "kept_images": len(all_images) - exact_removed - near_removed,
        "time_seconds": round(time_total, 2)
    }
    with open("datasets/processed/dedup_metrics.json", "w") as f:
        json.dump(metrics, f)

    print("\nPerformance Logging:")
    print(f"  Hash computation : {time_hash:.2f} sec")
    print(f"  Exact dedup      : {time_exact:.2f} sec")
    print(f"  BKTree           : {time_bktree:.2f} sec")
    print(f"  Total            : {time_total:.2f} sec")
    
    print(f"\nDeduplication complete. Kept {metrics['kept_images']} images.")

if __name__ == "__main__":
    main()
