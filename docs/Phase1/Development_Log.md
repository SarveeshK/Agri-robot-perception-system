# Development Log

## Sprint 1: RealSense Integration
- [x] Frame capture working
- [x] RGB-Depth alignment
- [x] Intrinsics extraction
- [x] Filters working (spatial, temporal)

## Sprint 2: YOLOv8 Integration
- [x] Model loading
- [x] Inference working
- [x] GPU warmup
- [x] Confidence filtering

## Sprint 3: Measurement
- [x] Distance calculation
- [x] Width measurement
- [x] Height measurement
- [x] Accuracy validation

## Sprint 4: Camera Optimization (Current)
- [ ] Close-range detection testing
- [ ] Depth accuracy at different distances
- [ ] Latency profiling
- [ ] FPS optimization

### Issues & Findings
- Issue #1: Depth invalid below 30cm → need disparity shift
- Finding #1: Close objects have lower confidence
- Optimization #1: Reduced temporal smoothing → faster response

### Next Steps
- Test with real obstacles
- Measure distance accuracy at 10cm, 30cm, 50cm, 100cm
- Document optimal settings
