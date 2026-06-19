# System Architecture

## Pipeline Overview (Phase 1)
```text
Intel RealSense D456 -> RGB + Depth -> Frame Alignment -> YOLO -> Depth Processing -> 3D Measurement -> Visualization
```

## Future Pipeline (Phase 2)
```text
Phase 1 -> Object Tracking -> Decision Engine -> Robot Controller
```

## Depth Optimization Approaches
1. **Advanced Mode / Disparity Shift**: Used to lower the minimum observable depth distance by shifting the sensor search space.
2. **Post-Processing Filters**:
   - Decimation: Reduces noise.
   - Threshold: Discards invalid distances.
   - Spatial: Smooths planar regions.
   - Temporal: Stabilizes depth over time.
   - Hole Filling: Corrects missing pixels.
3. **Robust Depth Extraction**: Instead of sampling single pixels, we sample a 5x5 grid from the central 50% ROI of the bounding box and use median filters and percentiles.
