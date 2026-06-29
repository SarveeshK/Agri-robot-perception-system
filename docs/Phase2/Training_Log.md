# Training Log Format

Maintain a log of all significant models trained for production.

---

### [Date] - [Model Version Name]
- **Base Model:** yolov8[n/s/m].pt
- **Dataset Version:** [e.g., v1 - OpenImages subset, v2 - Company Plantation]
- **Classes Count:** [number]
- **Epochs Trained:** [number]
- **mAP50:** [value]
- **mAP50-95:** [value]
- **Inference Time (ms):** [value]
- **Deployment Status:** [e.g., Deployed to robot, Failed testing, Kept as backup]
- **Notes:** [e.g., Added 50 images of low-light conditions. Recall improved by 5%.]
