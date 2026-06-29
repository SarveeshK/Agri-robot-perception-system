# Annotation Guide

Proper annotation is the most critical factor in model accuracy. Poor annotations will always result in a poor model ("garbage in, garbage out").

## 1. Bounding Box Rules
- **Tightness:** The bounding box should tightly encapsulate the visible pixels of the object. Do not leave excessive empty space around the object.
- **Incompleteness:** Do not guess where the object ends if it is cut off by the edge of the image. Only bound the visible pixels.

## 2. Class IDs
Refer to `config/class_mapping.yaml` for your final production classes. 
All annotators must use the exact same spelling and capitalization.

## 3. Handling Occlusion
- If an object is split into two visible pieces by a foreground obstacle (e.g., a tree behind a thin pole), draw **one** bounding box covering the entire extent of the tree if you are confident it's a single object.
- If the occlusion is massive and separates the object into two visually distinct and distant pieces, draw two separate boxes.

## 4. Multi-object Annotation
- Annotate *every* instance of the target class in the image. 
- If you label a tree in the foreground but ignore three trees in the background, the model learns that background trees are "not trees", destroying recall.

## 5. Label Consistency
- An object must be labeled exactly the same way across all images. 
- If a "small stone" is labeled as `rock` in one image and `small_stone` in another, the model will be confused and loss will not converge.

## 6. Quality Checks
- Before exporting the dataset to YOLO format, have a second person review a random 10% sample of the annotations.
- Look for: missing objects, loose boxes, and incorrect class assignments.
