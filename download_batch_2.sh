#!/bin/bash
source testenv/bin/activate

echo "=================================================="
echo "Downloading robust Human & Stump datasets"
echo "Source: Open Images V7 (Google)"
echo "=================================================="

# We explicitly target the exact canonical classes mapped in class_mapping.yaml
# Downloading 1000 instances of each to completely cure the bottlenecks.
python scripts/download_dataset.py --provider openimages --target Human Stump --limit 1000

echo "Download complete! Ready for build."
