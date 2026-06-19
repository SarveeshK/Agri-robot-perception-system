# Research Notes

## Intel RealSense D456 Limitations
The D456 uses stereo vision with a baseline distance. This creates a physical limitation on the minimum depth distance (Min-Z), typically around 40-50 cm.

## Disparity Shift
By enabling the `rs400_advanced_mode` API in `pyrealsense2`, we can modify the depth table's `disparityShift`.
Increasing this parameter allows the camera to see closer than its default optical minimum, at the expense of its maximum range. This is a highly effective optimization for agricultural robots needing to manipulate or avoid close-range objects.
