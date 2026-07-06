#!/bin/bash
# Continuing on errors so if one fails, others merge
source testenv/bin/activate
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
export ROBOFLOW_API_KEY="OBqA5KlUenPRf8rQQTzz"

echo "=================================================="
echo "Phase 3.2: Building Unified Dataset"
echo "=================================================="

echo "Clearing previous merged dataset to ensure clean build..."
rm -rf datasets/merged
mkdir -p datasets/merged

# 1. Tree Dataset (Roboflow)
echo -e "\n---> Downloading Tree Dataset (Roboflow)..."
python scripts/download_dataset.py --provider roboflow --url "https://universe.roboflow.com/trees-sam/yolov8tree" --version 1
python scripts/merge_dataset.py

# 2. Rock Dataset (Roboflow)
echo -e "\n---> Downloading Rock Dataset (Roboflow)..."
python scripts/download_dataset.py --provider roboflow --url "https://universe.roboflow.com/rocks-ebmeq/rocks-detection-govch" --version 1
python scripts/merge_dataset.py

# 3. Fence Dataset (Roboflow)
echo -e "\n---> Downloading Fence Dataset (Roboflow)..."
python scripts/download_dataset.py --provider roboflow --url "https://universe.roboflow.com/ayoub-9grd0/fence-detection-bkrx1" --version 2
python scripts/merge_dataset.py

echo -e "\n=================================================="
echo "Phase 3.3: Preparing Final YOLO Training Corpus"
echo "=================================================="
python scripts/prepare_dataset.py

echo -e "\nDataset is ready for Phase 4: Training!"
