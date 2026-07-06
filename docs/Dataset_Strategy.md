# Dataset Strategy — AgriVision Perception System

## Project Goal

> **Develop a real-time perception system for an autonomous agricultural robot that detects obstacles using YOLOv8 and an Intel RealSense D456 camera.**

The dataset is the foundation of the entire system. A well-constructed dataset directly determines the robot's ability to navigate safely. This document explains every decision in the dataset pipeline.

---

## Target Classes

Seven classes are defined by **navigation relevance**, not by general taxonomy.

| ID | Class | Navigation Role | Stop Distance |
|---|---|---|---|
| 0 | `Tree` | Large immovable structure — hard stop | 1.5 m |
| 1 | `Stump` | Low-profile but rigid — coconut/palm/bamboo stumps | 1.0 m |
| 2 | `Rock` | Large boulder — stop and reroute | 1.2 m |
| 3 | `Small_Stone` | Conditional: depth-estimated size determines traversability | 0.5 m |
| 4 | `Weed` | Traversable — log GPS for treatment planning | — |
| 5 | `Bush` | Blocks path and can occlude depth sensing | 1.0 m |
| 6 | `Fence` | Hard boundary — robot must never cross | 2.0 m |

Navigation behavior per class is defined in `config/obstacle_properties.yaml`, not in the model. This keeps perception independent of navigation logic.

---

## Dataset Sources

| Source | Best Classes | License | Notes |
|---|---|---|---|
| Open Images V7 | Tree, Rock, Small_Stone | CC BY 4.0 | Large general dataset |
| Roboflow Universe | Stump, Bush, Fence, Weed | Per-project | Best for agricultural classes |
| Kaggle | Rock, Small_Stone, Weed | Per-dataset | Research datasets |
| Manual (ZIP) | Any | Must verify | Own captures or company data |

### Dataset Coverage

| Class | Open Images | Roboflow | Kaggle | Manual |
|---|---|---|---|---|
| Tree | ✅ | ✅ | ✅ | — |
| Stump | ❌ | ✅ | ✅ | — |
| Rock | ✅ | ✅ | ✅ | — |
| Small_Stone | ✅ | ✅ | ✅ | — |
| Weed | ❌ | ✅ | ✅ | — |
| Bush | ❌ | ✅ | — | — |
| Fence | ❌ | ✅ | — | — |

❌ means the class is not reliably labeled in that provider. The pipeline documents this via `source_labels: []` in `class_mapping.yaml`.

---

## Full Pipeline

```
Internet
    │
    ▼
download_dataset.py  --provider <name> --target <classes> --limit <N>
    │
    ▼
datasets/raw/<provider>/  +  dataset_info.yaml
    │
    ▼
merge_dataset.py
    │  ← SHA256 duplicate detection
    │  ← label harmonization via class_mapping.yaml source_labels
    │  ← provenance tracking in metadata.csv
    ▼
datasets/merged/  +  metadata.csv  +  dataset_quality_report.txt
    │
    ▼
prepare_dataset.py
    │  ← image + label validation (YOLO format, value ranges)
    │  ← stratified 80/20 split (every class in both splits)
    │  ← auto-archives previous processed/ to archive/vN/
    ▼
datasets/processed/  +  data.yaml  +  coverage_report.txt
    │
    ▼
audit_dataset.py  ← MUST PASS before training
    │
    ▼
train_model.py → evaluate_model.py → export_model.py
```

---

## CLI Quickstart

```bash
# 1. Download from Open Images (Tree and Rock, 200 images each)
python scripts/download_dataset.py \
    --provider openimages \
    --target Tree Rock \
    --limit 200

# 2. Download from Roboflow (Stump, Bush, Fence)
python scripts/download_dataset.py \
    --provider roboflow \
    --url https://universe.roboflow.com/author/agri-obstacles \
    --version 1

# 3. Merge all downloaded datasets
python scripts/merge_dataset.py

# 4. Validate, split, and prepare
python scripts/prepare_dataset.py

# 5. Audit (must PASS before training)
python scripts/audit_dataset.py

# 6. Train
python scripts/train_model.py

# 7. Evaluate
python scripts/evaluate_model.py

# 8. Export to models/trained/
python scripts/export_model.py
```

---

## Aliasing Rules

Class aliasing maps source dataset labels to canonical class names. Rules:

1. **Only add an alias if you are confident it refers specifically to this class.**
2. **When in doubt, leave it out.** A missed annotation is better than a wrong one.
3. **Never use broad labels** like `"plant"` for `Tree` — they cover too many object types.
4. **All aliases are in `config/class_mapping.yaml`** under `source_labels.<provider>`.
5. **Adding a new alias requires no code changes** — `merge_dataset.py` reads the config at runtime.

Example of what NOT to do:
```yaml
# BAD — "plant" in Open Images includes corn, banana, sugarcane, ferns
Tree:
  source_labels:
    openimages:
      - plant      # ← too broad, will pollute Tree class with non-tree objects
```

---

## How to Add a New Dataset Source

### Option A — New provider (Roboflow, Kaggle, etc.)
```bash
python scripts/download_dataset.py --provider roboflow --url <url> --version 1
python scripts/merge_dataset.py
python scripts/prepare_dataset.py
python scripts/audit_dataset.py
```

### Option B — Manual ZIP file
```bash
python scripts/download_dataset.py --provider manual --zip /path/to/file.zip --name my_data
# Then open datasets/raw/manual/dataset_info.yaml and fill in license + homepage
python scripts/merge_dataset.py
python scripts/prepare_dataset.py
python scripts/audit_dataset.py
```

### Adding new label aliases
If the merge report shows "Unmapped Labels", add them to `config/class_mapping.yaml`:
```yaml
Tree:
  source_labels:
    roboflow:
      - Tree
      - Coconut Tree
      - new_label_from_dataset   # ← add here
```
Re-run `merge_dataset.py` — no code changes needed.

---

## Dataset Versioning

`prepare_dataset.py` automatically archives the previous `datasets/processed/` before overwriting it:

```
datasets/archive/
    v1/   ← first training dataset
    v2/   ← after adding Roboflow data
    v3/   ← after adding Kaggle data
```

To audit or reproduce a past version:
```bash
python scripts/audit_dataset.py --version v1
```

---

## Future Classes (not in current scope)

The architecture supports adding new classes by editing `class_mapping.yaml` only — no pipeline code changes needed.

Potential future additions for safety:
- `Human` (farm workers — safety critical)
- `Animal` (livestock wandering into path)
- `Farm_Equipment` (machinery obstacles)

These are deferred to a future phase pending mentor discussion.
