from ultralytics import YOLO
import os

def evaluate_model():
    model_path = "outputs/train/weights/best.pt"
    data_path = os.path.abspath("datasets/processed/data.yaml")
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Run train_model.py first.")
        return
        
    print(f"Loading trained model from {model_path}...")
    model = YOLO(model_path)
    
    print("Starting evaluation on validation set...")
    metrics = model.val(data=data_path, project=os.path.abspath("outputs"), name="val", exist_ok=True)
    
    print("\n--- Evaluation Metrics ---")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.p.mean():.4f}")
    print(f"Recall: {metrics.box.r.mean():.4f}")
    
    print("\nEvaluation complete. Detailed graphs and metrics saved in outputs/val/")

if __name__ == "__main__":
    evaluate_model()
