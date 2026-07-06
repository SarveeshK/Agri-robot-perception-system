import os
from roboflow import Roboflow

rf = Roboflow(api_key="OBqA5KlUenPRf8rQQTzz")
project = rf.workspace("ayoub-9grd0").project("fence-detection-bkrx1")
dataset = project.version(2).download("yolov8", location="test_robo_output")

print(f"Dataset location property: {dataset.location}")

print("\n--- Directory Tree ---")
for root, dirs, files in os.walk("test_robo_output"):
    print(f"{root} (dirs: {len(dirs)}, files: {len(files)})")
    if len(files) > 0 and len(files) < 10:
        for f in files:
            print(f"  - {f}")
