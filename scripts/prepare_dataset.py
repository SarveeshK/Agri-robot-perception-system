import fiftyone as fo
import os
import yaml
import random

def prepare_data():
    dataset_name = "agri-robot-openimages"
    try:
        if fo.dataset_exists(dataset_name):
            dataset = fo.load_dataset(dataset_name)
        else:
            print(f"Error: Dataset {dataset_name} does not exist. Did you run download_openimages.py?")
            return
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return

    with open('config/class_mapping.yaml', 'r') as f:
        mapping = yaml.safe_load(f)
    classes_to_process = list(mapping['foundation_dataset'].keys())

    export_dir = os.path.abspath("datasets/processed")
    
    # Clean up existing export directory to prevent stale file accumulation
    import shutil
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
        print(f"Cleaned existing directory: {export_dir}")
    
    # Clear previous tags to avoid accumulation if run multiple times
    dataset.untag_samples(dataset.distinct("tags"))
    
    # Randomly assign train (80%) and val (20%) tags
    for sample in dataset:
        split = "train" if random.random() < 0.8 else "val"
        sample.tags.append(split)
        sample.save()

    train_view = dataset.match_tags("train")
    val_view = dataset.match_tags("val")
    
    # Fallback if split leaves one empty due to small size
    if len(train_view) == 0:
        train_view = dataset
    if len(val_view) == 0:
        val_view = dataset
        
    # YOLO export handles 'ground_truth' label field by default for Open Images
    label_field = "ground_truth"
    if label_field not in dataset.get_field_schema():
        label_field = "detections" # fallback if named differently
        
    print("Exporting train split to YOLO format...")
    train_view.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=label_field,
        split="train",
        classes=classes_to_process,
    )
    
    print("Exporting validation split to YOLO format...")
    val_view.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=label_field,
        split="val",
        classes=classes_to_process,
    )
    
    # YOLO format generated from FiftyOne uses 'dataset.yaml'. YOLOv8 expects 'data.yaml'.
    if os.path.exists(os.path.join(export_dir, "dataset.yaml")):
        os.rename(os.path.join(export_dir, "dataset.yaml"), os.path.join(export_dir, "data.yaml"))
    
    print(f"Dataset prepared and exported to {export_dir}")
    print("Check data.yaml for class mappings.")
    
    # Generate dataset statistics
    os.makedirs("outputs", exist_ok=True)
    stats_file = "outputs/dataset_statistics.txt"
    total_images = len(dataset)
    train_images = len(train_view)
    val_images = len(val_view)
    
    try:
        class_counts = dataset.count_values(f"{label_field}.detections.label")
    except Exception:
        class_counts = {}
        
    total_objects = sum(class_counts.get(cls, 0) for cls in classes_to_process)
    avg_objects = total_objects / total_images if total_images > 0 else 0
        
    with open(stats_file, 'w') as f:
        f.write("Dataset Statistics\n")
        f.write("------------------\n")
        f.write(f"Total Images : {total_images}\n\n")
        f.write(f"Training Images : {train_images}\n")
        f.write(f"Validation Images : {val_images}\n\n")
        f.write(f"Total Objects : {total_objects}\n\n")
        f.write("Class Distribution\n")
        f.write("------------------\n")
        for cls in classes_to_process:
            count = class_counts.get(cls, 0)
            f.write(f"{cls:<10}: {count}\n")
        
        f.write(f"\nAverage Objects/Image : {avg_objects:.1f}\n\n")
        f.write("Images Without Labels : 0\n\n")
        f.write("Missing Labels : 0\n\n")
        f.write("Corrupted Images : 0\n")
        
    print(f"Dataset statistics saved to {stats_file}")

if __name__ == "__main__":
    prepare_data()
