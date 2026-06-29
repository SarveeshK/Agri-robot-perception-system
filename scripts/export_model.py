import os
import shutil

def export_model():
    source = "outputs/train/weights/best.pt"
    destination_dir = "models/trained"
    destination = os.path.join(destination_dir, "best.pt")
    
    if not os.path.exists(source):
        print(f"Error: Trained model not found at {source}. Run train_model.py first.")
        return
        
    os.makedirs(destination_dir, exist_ok=True)
    shutil.copy2(source, destination)
    
    print(f"Model successfully exported to {destination}")
    print("\nTo deploy this model, update config/settings.yaml:")
    print("yolo:")
    print('  model_path: "models/trained/best.pt"')

if __name__ == "__main__":
    export_model()
