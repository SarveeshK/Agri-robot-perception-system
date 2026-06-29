import fiftyone as fo
import fiftyone.zoo as foz
import os
import yaml

def download_data():
    # Read mapping to know which classes to download
    with open('config/class_mapping.yaml', 'r') as f:
        mapping = yaml.safe_load(f)
    
    classes_to_download = list(mapping['foundation_dataset'].keys())
    
    output_dir = os.path.abspath("datasets/raw/openimages")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting download for classes: {classes_to_download}")
    
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train", # using train split as it's best practice for building datasets
        label_types=["detections"],
        classes=classes_to_download,
        max_samples=60, # increased from 15 to 60 for a slightly more robust foundation model
        dataset_name="agri-robot-openimages"
    )
    
    # Make dataset persistent so it can be loaded by prepare_dataset.py
    dataset.persistent = True
    
    print(f"Dataset downloaded successfully into FiftyOne database as '{dataset.name}'")

if __name__ == "__main__":
    download_data()
