# Data Collection Guide

When collecting the production dataset using the Intel RealSense D456 inside the coconut plantation, strictly adhere to the following variation principles. Diversity in your dataset ensures the YOLOv8 model generalizes well to new, unseen conditions.

## 1. Viewpoint Variation
Do not capture all images from the exact same angle.
- **Why:** The robot will approach obstacles from different trajectories.
- **How:** Take photos straight-on, from the left, from the right, and slightly angled.

## 2. Distance Variation
Capture obstacles from near and far.
- **Why:** The model needs to recognize a tree both when it's 5 meters away and when it's 0.5 meters away (imminent collision).
- **How:** Step backward and forward while collecting data for the same object.

## 3. Scale Variation
Include small and large examples of the same class.
- **Why:** A "rock" can be the size of a fist or the size of a boulder. 
- **How:** Ensure your dataset naturally captures these scale differences.

## 4. Illumination Variation
Lighting drastically changes pixel values.
- **Why:** Shadows, harsh sunlight, overcast skies, and dusk all affect object appearance.
- **How:** Collect data at different times of the day (morning, noon, late afternoon). Do not collect only on sunny days.

## 5. Background Variation
The background shouldn't always look the same.
- **Why:** If every rock you photograph is on dirt, the model might learn that "dirt = rock".
- **How:** Capture rocks on grass, rocks on dirt, rocks near trees.

## 6. Occlusion
Objects in the real world are often partially hidden.
- **Why:** If the model only ever sees whole, unobstructed trees, it will fail to detect a tree partially hidden behind a large leaf or another tree.
- **How:** Intentionally capture objects that are partially blocked.

## 7. Positive and Negative Samples
- **Positive Samples:** Images containing the target objects (trees, rocks, weeds).
- **Negative Samples (Background Images):** Images containing *zero* target objects (e.g., just an empty dirt path or sky).
- **Why:** Including about 1-10% negative samples helps reduce False Positives.

## 8. Domain-specific Acquisition
- **RealSense D456 Constraints:** Since the robot uses this specific sensor, you must use it to capture the data. Do not use a high-end DSLR or an iPhone to collect the final production dataset. The model must learn the specific noise profile, color balance, and lens distortion of the RealSense RGB camera.
