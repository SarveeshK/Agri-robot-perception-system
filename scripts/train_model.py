from ultralytics import YOLO
import os
import shutil

def train_model():
    model_path = "models/pretrained/yolov8n.pt"
    data_path = os.path.abspath("datasets/processed/data.yaml")
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run prepare_dataset.py first.")
        return
        
    print(f"Loading pretrained model from {model_path}...")
    model = YOLO(model_path)
    
    print("Starting training...")
    # Train the model
    results = model.train(
        data=data_path,
        epochs=3, # Lightweight training for pipeline verification
        batch=2,
        imgsz=640,
        project=os.path.abspath("outputs"),
        name="train",
        exist_ok=True # Overwrite for testing purposes
    )
    
    print("Training complete.")
    
    # Save training configuration
    import yaml
    import datetime
    os.makedirs("outputs/train", exist_ok=True)
    config_path = "outputs/train/training_config.yaml"
    config_data = {
        "model": "yolov8n.pt",
        "epochs": 3,
        "batch": 2,
        "imgsz": 640,
        "optimizer": "auto",
        "learning_rate": "auto",
        "dataset": "datasets/processed",
        "date": datetime.datetime.now().strftime("%Y-%m-%d")
    }
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, sort_keys=False)
        
    print(f"Training configuration saved to {config_path}")
    print("Artifacts (best.pt, last.pt, results.png, etc.) saved in outputs/train/")

if __name__ == "__main__":
    train_model()
