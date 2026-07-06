#!/bin/bash
# Continuing on errors so we can test Roboflow even if Kaggle fails

# Secure the Kaggle token
chmod 600 ~/.kaggle/access_token

# Activate virtual environment
source testenv/bin/activate

# Secure the Kaggle token and explicitly export it to the environment
chmod 600 ~/.kaggle/access_token
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)

# Install required provider packages
echo "Installing Kaggle and Roboflow pip packages..."
pip install kaggle kagglehub roboflow --quiet

echo "=================================================="
echo "Phase 3.1: Candidate Dataset Verification"
echo "=================================================="

# Export the Roboflow API Key you provided
export ROBOFLOW_API_KEY="OBqA5KlUenPRf8rQQTzz"

# Clean up any existing temp downloads
rm -rf datasets/raw/kaggle datasets/raw/roboflow datasets/raw/temp

echo -e "\n[1/3] Verifying Kaggle 'Tree' Dataset..."
# We use --limit 50 to only grab a sample for verification
python scripts/download_dataset.py --provider kaggle --dataset skurski/tree-counting-image-dataset --limit 50

echo -e "\n[2/3] Verifying Roboflow 'Rock' Dataset..."
python scripts/download_dataset.py --provider roboflow --url "https://universe.roboflow.com/rocks-ebmeq/rocks-detection-govch" --version 1 --limit 50

echo -e "\n[3/3] Verifying Roboflow 'Fence' Dataset..."
python scripts/download_dataset.py --provider roboflow --url "https://universe.roboflow.com/ayoub-9grd0/fence-detection-bkrx1" --version 2 --limit 50

echo -e "\n=================================================="
echo "Verification Downloads Complete!"
echo "Please inspect the downloaded data.yaml files in datasets/raw/"
echo "=================================================="
