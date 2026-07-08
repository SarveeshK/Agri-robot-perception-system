import argparse
from ultralytics import YOLO
import os

def evaluate_model(args):
    model_path = args.weights
    data_path = os.path.abspath(args.data)
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
        
    print(f"Loading trained model from {model_path}...")
    model = YOLO(model_path)
    
    # Extract the experiment directory (e.g. outputs/experiments/exp001) from the weights path
    exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(model_path)))
    
    print(f"Starting evaluation on validation set...")
    metrics = model.val(data=data_path, project=exp_dir, name="val", exist_ok=True)
    
    print("\n--- Evaluation Metrics ---")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.p.mean():.4f}")
    print(f"Recall: {metrics.box.r.mean():.4f}")
    
    val_out_dir = os.path.join(exp_dir, "val")
    print(f"\nEvaluation complete. Detailed graphs (including confusion_matrix.png) saved in {val_out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 Model")
    parser.add_argument("--weights", type=str, default="outputs/experiments/exp001/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--data", type=str, default="datasets/processed/data.yaml", help="Path to data.yaml")
    args = parser.parse_args()
    evaluate_model(args)
