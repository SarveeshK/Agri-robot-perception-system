import os
import shutil
import datetime
import argparse
import yaml
import json
import hashlib

def generate_checksums(dest_dir):
    checksum_file = os.path.join(dest_dir, 'SHA256SUMS')
    with open(checksum_file, 'w') as out_f:
        for root, _, files in os.walk(dest_dir):
            for file in sorted(files):
                if file == 'SHA256SUMS': continue
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, dest_dir)
                
                sha256 = hashlib.sha256()
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256.update(chunk)
                out_f.write(f"{sha256.hexdigest()}  {rel_path}\n")

def get_class_names(data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('names', [])

def count_class_distribution(dataset_dir, class_names):
    counts = {name: 0 for name in class_names}
    labels_dir = os.path.join(dataset_dir, 'labels')
    
    for split in ['train', 'val']:
        split_dir = os.path.join(labels_dir, split)
        if not os.path.exists(split_dir): continue
        
        for file in os.listdir(split_dir):
            if file.endswith('.txt'):
                with open(os.path.join(split_dir, file), 'r') as f:
                    for line in f:
                        if line.strip():
                            cls_id = int(line.split()[0])
                            if cls_id < len(class_names):
                                counts[class_names[cls_id]] += 1
    
    total = sum(counts.values())
    if total == 0: return counts
    
    # Convert to percentages
    return {name: round((count / total) * 100, 2) for name, count in counts.items()}

def check_drift(new_dist, releases_dir):
    """Compare new_dist against the most recent release to detect >20% drift."""
    if not os.path.exists(releases_dir): return
    
    # Find most recent release
    releases = [d for d in os.listdir(releases_dir) if d.startswith('v') and os.path.isdir(os.path.join(releases_dir, d))]
    releases.sort(reverse=True)
    
    if not releases: return
    
    last_release = releases[0]
    last_dist_path = os.path.join(releases_dir, last_release, 'class_distribution.json')
    if not os.path.exists(last_dist_path): return
    
    with open(last_dist_path, 'r') as f:
        old_dist = json.load(f)
        
    print(f"\n--- Drift Detection (vs {last_release}) ---")
    for cls_name, new_pct in new_dist.items():
        old_pct = old_dist.get(cls_name, 0)
        diff = abs(new_pct - old_pct)
        if diff > 20.0:
            print(f"⚠️ WARNING: Massive drift in {cls_name}! Old: {old_pct}% -> New: {new_pct}%")
        else:
            print(f"OK: {cls_name} ({old_pct}% -> {new_pct}%)")

def freeze_dataset(output_base="datasets/releases"):
    print("=" * 50)
    print("AgriVision — Dataset Freezer & Drift Detection")
    print("=" * 50)
    
    src_dir = "datasets/processed"
    if not os.path.exists(src_dir):
        print(f"ERROR: {src_dir} does not exist.")
        return
        
    # Determine version tag
    version_tag = f"v{datetime.datetime.now().strftime('%Y.%m.%d')}"
    dest_dir = os.path.join(output_base, version_tag)
    suffix = 1
    original_dest = dest_dir
    while os.path.exists(dest_dir):
        dest_dir = f"{original_dest}.{suffix}"
        suffix += 1
        
    print(f"Freezing {src_dir} -> {dest_dir}...")
    shutil.copytree(src_dir, dest_dir)
    
    # 1. Drift Detection (class_distribution.json)
    data_yaml_path = os.path.join(dest_dir, 'data.yaml')
    if os.path.exists(data_yaml_path):
        class_names = get_class_names(data_yaml_path)
        dist = count_class_distribution(src_dir, class_names)
        
        check_drift(dist, output_base)
        
        with open(os.path.join(dest_dir, 'class_distribution.json'), 'w') as f:
            json.dump(dist, f, indent=4)
            
    # 2. Release Manifest (release.yaml)
    # Count total images
    total_images = 0
    for split in ['train', 'val']:
        img_dir = os.path.join(dest_dir, 'images', split)
        if os.path.exists(img_dir):
            total_images += len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
            
    manifest = {
        "release": {
            "version": os.path.basename(dest_dir),
            "pipeline": "v1.1.0"
        },
        "datasets": [], # To be filled by reading metadata.csv in the future if needed
        "statistics": {
            "images": total_images
        },
        "build_time": datetime.datetime.now().isoformat()
    }
    
    with open(os.path.join(dest_dir, 'release.yaml'), 'w') as f:
        yaml.dump(manifest, f, sort_keys=False)
        
    # 3. Immutability Checksums
    print("Generating SHA256 checksums...")
    generate_checksums(dest_dir)
        
    print(f"\nDataset successfully frozen at {dest_dir}")
    print("Manifest, checksums, and distribution drift metrics generated.")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freeze processed dataset for training")
    parser.add_argument("--output", default="datasets/releases", help="Output release directory")
    args = parser.parse_args()
    freeze_dataset(args.output)
