# AgriVision Dataset Catalog

This document tracks all datasets considered for the AgriVision perception pipeline. 

Every dataset undergoes a rigorous evaluation process to ensure only high-quality, agricultural-relevant data enters the final merged training corpus.

## Status Workflow
`Pending` ➔ `Downloaded` ➔ `Verified` ➔ `Approved` ➔ `Merged` ➔ `Training`

## Acceptance Criteria
A dataset is only marked as **Approved** if it meets the following objective rules:
1. Active and accessible.
2. License permits intended use.
3. Provides YOLO bounding boxes natively.
4. Agricultural or closely related domain (Farm/Orchard > Forest >> Urban).
5. Annotation quality ≥ 4/5 (based on 50 image random sample).
6. At least 200 useful images for the target class (except Stumps).
7. No major annotation errors.
8. Minimal duplicate images.

## Scoring Metrics
* **Annotation**: Out of 5 (Bounding box accuracy, Missing annotations, Label consistency).
* **Diversity**: Out of 5 (Morning, Afternoon, Cloudy, Shadows, Angles, Seasons).
* **RealSense Fit**: 5 (Ground robot), 4 (Handheld), 2 (Drone), 1 (Satellite).

---

# Curated Datasets

| Priority | Class | Dataset | Domain | Source | Version | Images | Target Class | License | Annotation | Diversity | RealSense Fit | Purpose | Keep | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P0 | Tree | [Tree Counting Image Dataset](https://www.kaggle.com/datasets/skurski/tree-counting-image-dataset) | Forest/Urban | Kaggle | v1 | 91 | Tree | CC BY 4.0 | TBD | TBD | TBD | Extract Trees | TBD | Pending |
| P0 | Tree | [Tree Yolo Annotated](https://www.kaggle.com/datasets/ahmadheshammahmoud/tree-yolo-annotated) | Farm/Forest | Kaggle | v1 | ~1500 | Tree | Unknown | TBD | TBD | TBD | Extract Trees | No | **Rejected (Deleted)** |
| P0 | Rock | [MARGALLA TREE ROCKS](https://universe.roboflow.com/asim-cheema/margalla-tree-rocks) | Forest | Roboflow | v1 | ~1300 | Rock, Stone | CC BY 4.0 | TBD | TBD | TBD | Extract Rocks | TBD | Pending |
| P0 | Rock | [Rocks Detection (govch)](https://universe.roboflow.com/rocks-ebmeq/rocks-detection-govch) | Mixed | Roboflow | v1 | ~1000 | Float_Rock | CC BY 4.0 | TBD | TBD | TBD | Extract Rocks | TBD | Pending |
| P0 | Fence | [Fence Detection (ayoub-9grd0)](https://universe.roboflow.com/ayoub-9grd0/fence-detection-bkrx1) | Farm/Urban | Roboflow | v2 | ~700 | Fence | CC BY 4.0 | TBD | TBD | TBD | Extract Fence | TBD | Pending |
| P0 | Fence | [Broken Fence Detection](https://universe.roboflow.com/) | Farm | Roboflow | v1 | ~500 | Fence | CC BY 4.0 | TBD | TBD | TBD | Extract Fence | TBD | Pending |
