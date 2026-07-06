import yaml
import subprocess
import os
import shutil
import hashlib
import sys
import json
from pathlib import Path
from datetime import datetime

SSOT_FILE = Path("config/dataset_sources.yaml")
MERGED_DIR = Path("datasets/merged")
PROCESSED_DIR = Path("datasets/processed")

def get_file_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return ""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("==================================================")
    print("AgriVision Dataset Build Orchestrator")
    print("==================================================")
    
    if not SSOT_FILE.exists():
        print(f"ERROR: {SSOT_FILE} not found!")
        return

    with open(SSOT_FILE, "r") as f:
        ssot = yaml.safe_load(f)
    
    active_datasets = ssot.get("active_datasets", [])
    if not active_datasets:
        print("No active datasets defined in SSOT.")
        return

    print("Cleaning up legacy folders...")
    legacy_roboflow = Path("datasets/raw/roboflow")
    if legacy_roboflow.exists():
        shutil.rmtree(legacy_roboflow)
        print("  Deleted legacy datasets/raw/roboflow directory.")

    print("Clearing previous merged dataset...")
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "pipeline": {"version": "1.0.0"},
        "build_date": datetime.now().isoformat(),
        "datasets": {},
        "total_images_merged": 0,
        "total_duplicates_removed": 0
    }
    
    datasets_included = []

    for ds in active_datasets:
        if not ds.get("enabled", False):
            continue
        if ds.get("status") != "Approved":
            print(f"Skipping {ds['id']} (Status: {ds.get('status')})")
            continue
            
        print(f"\n---> Processing {ds['id']} ({ds['target_class']})")
        
        provider = ds.get("provider")
        
        cmd = [sys.executable, "scripts/download_dataset.py", "--provider", provider]
        
        if provider == "roboflow":
            cmd.extend(["--url", f"https://universe.roboflow.com/{ds['workspace']}/{ds['project']}"])
            if "version" in ds and ds["version"] != "latest":
                cmd.extend(["--version", str(ds["version"])])
        elif provider == "kaggle":
            cmd.extend(["--dataset", ds["dataset"]])
            
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"ERROR downloading {ds['id']}")
            continue
            
        # Fingerprinting
        raw_dir = None
        if provider == "roboflow":
            raw_dir = Path(f"datasets/raw/roboflow_{ds['project']}")
        elif provider == "kaggle":
            dataset_name = ds['dataset'].split('/')[-1]
            raw_dir = Path(f"datasets/raw/kaggle_{dataset_name}")
            
        fingerprint = {}
        if raw_dir and raw_dir.exists():
            data_yaml = raw_dir / "data.yaml"
            fingerprint["data_yaml_sha256"] = get_file_sha256(data_yaml)
            images_dir = raw_dir / "images"
            if images_dir.exists():
                fingerprint["downloaded_images"] = len(list(images_dir.glob("*.*")))
            
        manifest["datasets"][ds["id"]] = {
            "version": ds.get("version"),
            "fingerprint": fingerprint
        }
        
        datasets_included.append(ds)

    print("\n==================================================")
    print("Phase 3.1.5: Global Deduplication")
    print("==================================================")
    print(f"Running: python scripts/deduplicate_dataset.py")
    result = subprocess.run([sys.executable, "scripts/deduplicate_dataset.py"])
    if result.returncode != 0:
        print(f"ERROR running deduplication. Check deduplicate_dataset.py output.")
        return

    print("\n==================================================")
    print("Phase 3.2: Merging Datasets")
    print("==================================================")
    print(f"Running: python scripts/merge_dataset.py")
    result = subprocess.run([sys.executable, "scripts/merge_dataset.py"])
    if result.returncode != 0:
        print(f"ERROR merging datasets. Check merge_dataset.py output.")
        return

    print("\n==================================================")
    print("Phase 3.3: Preparing Final YOLO Training Corpus")
    print("==================================================")
    subprocess.run([sys.executable, "scripts/prepare_dataset.py"])

    # Extract stats from dataset_statistics.txt if it exists
    stats_file = PROCESSED_DIR / "dataset_statistics.txt"
    total_train, total_val = 0, 0
    if stats_file.exists():
        with open(stats_file, "r") as f:
            for line in f:
                if line.strip().startswith("Training Images"):
                    total_train = int(line.split(":")[1].strip())
                elif line.strip().startswith("Validation Images"):
                    total_val = int(line.split(":")[1].strip())
    
    manifest["total_images"] = total_train + total_val

    dedup_metrics = PROCESSED_DIR / "dedup_metrics.json"
    if dedup_metrics.exists():
        with open(dedup_metrics, "r") as f:
            manifest["deduplication"] = json.load(f)

    # Write manifest
    manifest_path = PROCESSED_DIR / "build_manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, sort_keys=False)

    print("\n========================================")
    print("AgriVision Dataset Build Summary")
    print("========================================")
    print("\nDatasets Included:")
    for ds in datasets_included:
        print(f"  ✓ {ds['target_class']} ({ds['provider']} v{ds.get('version', 'unknown')})")
        
    print(f"\nImages:")
    print(f"  Total: {total_train + total_val}")
    print(f"  Train: {total_train}")
    print(f"  Val: {total_val}")
    
    print("\nManifest:")
    print(f"  {manifest_path}")
    print("========================================")

if __name__ == "__main__":
    main()
