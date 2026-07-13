#!/bin/bash
export ROBOFLOW_API_KEY=OBqA5KlUenPRf8rQQTzz
source testenv/bin/activate

echo "Downloading Stump (jump889/tree-trunk-train-luh1h)..."
python scripts/download_dataset.py --provider roboflow --url https://universe.roboflow.com/jump889/tree-trunk-train-luh1h

echo "Downloading Background 1 (university-of-maryland/farm-ljomy)..."
python scripts/download_dataset.py --provider roboflow --url https://universe.roboflow.com/university-of-maryland/farm-ljomy

echo "Downloading Background 2 (bernard-tu/b2_w-instance-segmentation-prllf)..."
python scripts/download_dataset.py --provider roboflow --url https://universe.roboflow.com/bernard-tu/b2_w-instance-segmentation-prllf
