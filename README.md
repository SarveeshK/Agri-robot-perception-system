# Agro-Robot Perception System

A modular, real-time AI perception system for an autonomous agricultural robot. The system integrates an Intel RealSense D456 depth camera with a custom-trained YOLOv8 model to detect agricultural obstacles in real time and provide 3D distance measurements for safe navigation.

---

## 🔒 Development Rule: Architecture Frozen (v1.1.0)
The architecture is frozen. New pull requests must satisfy one of:
* Dataset improvement
* Annotation improvement
* Training improvement
* Evaluation improvement
* Hardware integration
* Robot integration
* Bug fix

Architectural changes require a documented justification:
- reproducibility issue
- deployment blocker
- critical bug

---

## Project Architecture

```
Internet (Public Datasets)
         │
         ▼
download_dataset.py       ← multi-provider dataset downloader
         │
         ▼
datasets/raw/             ← source-separated raw data
         │
         ▼
merge_dataset.py          ← label harmonization + SHA256 deduplication
         │
         ▼
datasets/merged/          ← unified YOLO labels + provenance metadata
         │
         ▼
prepare_dataset.py        ← validate, stratified split, data.yaml
         │
         ▼
datasets/processed/       ← train/ val/ data.yaml
         │
         ▼
train_model.py → evaluate_model.py → export_model.py
         │
         ▼
best.pt
         │
         ▼
RealSense D456  +  YOLO  →  Obstacle Detection  →  Navigation
```

---

## Detection Classes

| ID | Class | Navigation Behavior |
|---|---|---|
| 0 | Tree | Hard stop at 1.5 m |
| 1 | Stump | Hard stop at 1.0 m |
| 2 | Rock | Hard stop at 1.2 m |
| 3 | Small_Stone | Conditional (size-based) |
| 4 | Weed | Traversable |
| 5 | Bush | Hard stop at 1.0 m |
| 6 | Fence | Hard stop at 2.0 m |

---

## Setup

```bash
pip install -r requirements.txt
```

Optional provider dependencies (install only for the provider you use):
```bash
pip install openimages    # for --provider openimages
pip install roboflow      # for --provider roboflow
pip install kaggle        # for --provider kaggle
```

---

## Dataset Pipeline

```bash
# Step 1 — Download (repeat for each provider)
python scripts/download_dataset.py --provider openimages --target Tree Rock --limit 200
python scripts/download_dataset.py --provider roboflow --url <url> --version 1

# Step 2 — Merge + harmonize labels
python scripts/merge_dataset.py

# Step 3 — Validate + split + prepare
python scripts/prepare_dataset.py

# Step 4 — Audit (must PASS before training)
python scripts/audit_dataset.py

# Step 5 — Train
python scripts/train_model.py

# Step 6 — Evaluate
python scripts/evaluate_model.py

# Step 7 — Export to models/trained/
python scripts/export_model.py
```

See [docs/Dataset_Strategy.md](docs/Dataset_Strategy.md) for full dataset documentation.

---

## Running the Perception System (Phase 1)

```bash
python src/main.py
```

---

## Configuration

| File | Purpose |
|---|---|
| `config/settings.yaml` | Camera, YOLO inference, visualization, safety distance |
| `config/class_mapping.yaml` | 7 classes with per-provider source labels |
| `config/obstacle_properties.yaml` | Navigation behavior per class |
| `config/dataset_sources.yaml` | Provider registry for downloader |

---

## Repository Structure

```
Agri-robot-perception-system/
├── config/                   # All configuration — no hardcoded values
│   ├── class_mapping.yaml    # Canonical classes + provider label mappings
│   ├── obstacle_properties.yaml  # Navigation behavior per class
│   ├── dataset_sources.yaml  # Dataset provider registry
│   └── settings.yaml         # Camera, YOLO, safety parameters
├── datasets/
│   ├── raw/                  # Downloaded datasets (per provider)
│   ├── merged/               # Harmonized + deduplicated dataset
│   ├── processed/            # Train/val split ready for training
│   ├── exports/              # Model exports
│   └── archive/              # Versioned dataset snapshots (v1, v2, ...)
├── docs/
│   ├── Dataset_Strategy.md   # Full dataset design documentation
│   ├── Phase1/               # RealSense + perception docs
│   └── Phase2/               # MLOps, training, annotation guides
├── models/
│   ├── pretrained/           # Foundation models (yolov8n.pt)
│   └── trained/              # Production models (best.pt)
├── scripts/                  # MLOps pipeline
│   ├── download_dataset.py   # Provider dispatcher
│   ├── providers/            # One file per dataset provider
│   ├── merge_dataset.py      # Label harmonization
│   ├── prepare_dataset.py    # Validation + split
│   ├── audit_dataset.py      # Pre-training quality gate
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── export_model.py
└── src/                      # Real-time inference engine (Phase 1)
    ├── camera/               # RealSense hardware interface
    ├── perception/           # YOLO + depth + measurement
    ├── utils/                # Logger, profiler
    └── main.py
```

---

## Engineering Principles

- **Single Responsibility**: Each script has exactly one job.
- **Configuration-Driven**: Every parameter in YAML — no hardcoded values.
- **Provider-Agnostic**: Adding a dataset source = one new file + one config entry.
- **Reproducible**: SHA256 deduplication + versioned dataset archives.
- **Modular**: Perception engine is independent of navigation logic.