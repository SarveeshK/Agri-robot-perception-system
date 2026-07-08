import argparse
from ultralytics import YOLO
import os
import glob
import yaml
import datetime
import csv

def get_next_exp_id(base_dir="outputs/experiments"):
    """Find the next available expXXX id."""
    os.makedirs(base_dir, exist_ok=True)
    existing_dirs = glob.glob(os.path.join(base_dir, "exp*"))
    exp_nums = []
    for d in existing_dirs:
        basename = os.path.basename(d)
        if basename.startswith("exp") and basename[3:].isdigit():
            exp_nums.append(int(basename[3:]))
    next_num = max(exp_nums) + 1 if exp_nums else 1
    return f"exp{next_num:03d}"

def train_model(args):
    model_path = "models/pretrained/yolov8n.pt"
    data_path = os.path.abspath(args.data)
    
    if "datasets/processed" in data_path:
        print("Error: Dataset Release Lock is active.")
        print("Training directly from datasets/processed/ is forbidden.")
        print("Please point to a frozen release, e.g., --data datasets/releases/vYYYY.MM.DD/data.yaml")
        return
        
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Ensure the dataset is frozen or built.")
        return
        
    print(f"Loading pretrained model from {model_path}...")
    model = YOLO(model_path)
    
    exp_id = get_next_exp_id()
    print(f"Starting Experiment: {exp_id}")
    
    # Train the model, pointing project and name directly to experiments folder
    results = model.train(
        data=data_path,
        epochs=args.epochs,
        batch=-1,  # Auto batching
        imgsz=640,
        patience=20,
        seed=42,
        cos_lr=True,
        workers=8, # standard for YOLO
        fraction=args.fraction, # allows subsampling dataset for fast CPU tests
        cache=True, # Significantly speeds up CPU training by caching dataset in RAM
        amp=True, # Automatic Mixed Precision (if supported by hardware)
        project=os.path.abspath("outputs/experiments"),
        name=exp_id,
        exist_ok=False 
    )
    
    # Save experiment.yaml manifest
    exp_dir = os.path.join("outputs/experiments", exp_id)
    config_path = os.path.join(exp_dir, "experiment.yaml")
    
    config_data = {
        "experiment_id": exp_id,
        "date": datetime.datetime.now().isoformat(),
        "dataset_path": args.data,
        "model": "yolov8n.pt",
        "hyperparameters": {
            "epochs": args.epochs,
            "imgsz": 640,
            "patience": 20,
            "seed": 42
        }
    }
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, sort_keys=False)
        
    print(f"Experiment successfully archived to {exp_dir}")
    
    # Update Leaderboard
    leaderboard_path = "outputs/leaderboard.csv"
    file_exists = os.path.exists(leaderboard_path)
    with open(leaderboard_path, 'a', newline='') as csvfile:
        fieldnames = ['Exp ID', 'Dataset', 'mAP50', 'Recall', 'Status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        # YOLOv8 results object contains results_dict
        map50 = results.results_dict.get('metrics/mAP50(B)', 0.0) if hasattr(results, 'results_dict') else 0.0
        recall = results.results_dict.get('metrics/recall(B)', 0.0) if hasattr(results, 'results_dict') else 0.0
        
        writer.writerow({
            'Exp ID': exp_id,
            'Dataset': os.path.basename(os.path.dirname(args.data)),
            'mAP50': round(map50, 4),
            'Recall': round(recall, 4),
            'Status': 'Completed'
        })
    print(f"Leaderboard updated: {leaderboard_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Baseline Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--data", type=str, default="datasets/processed/data.yaml", help="Path to data.yaml")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to train on (e.g. 0.1 for 10%)")
    args = parser.parse_args()
    train_model(args)
