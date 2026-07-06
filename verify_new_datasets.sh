#!/bin/bash
source testenv/bin/activate
export ROBOFLOW_API_KEY="OBqA5KlUenPRf8rQQTzz"

echo "========================================"
echo "Verifying Weed Dataset"
echo "========================================"
python scripts/download_dataset.py --provider roboflow --url https://universe.roboflow.com/project-weeds/weed-detection-5jm0z --version 2
cat datasets/raw/roboflow_weed-detection-5jm0z/data.yaml

echo "========================================"
echo "Verifying Bush Dataset"
echo "========================================"
python scripts/download_dataset.py --provider roboflow --url https://universe.roboflow.com/mtlworkspace/bush-detection --version 1
cat datasets/raw/roboflow_bush-detection/data.yaml

echo "========================================"
echo "Verifying Stump Dataset"
echo "========================================"
python scripts/download_dataset.py --provider roboflow --url https://universe.roboflow.com/minbin/tree-stump --version 1
cat datasets/raw/roboflow_tree-stump/data.yaml
