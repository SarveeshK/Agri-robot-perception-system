import math
from src.perception.measurement import ObjectMeasurement

class DummyCalibration:
    def deproject_pixel_to_point(self, x, y, depth):
        # simple mock deprojection
        return [x * depth / 1000.0, y * depth / 1000.0, depth]

def test_measurement_accuracy():
    print("Testing Object Measurement...")
    calib = DummyCalibration()
    measurer = ObjectMeasurement(calib)
    
    # Simulate a bounding box and a 100cm distance
    box = (100, 100, 200, 200)
    distance_cm = 100.0
    
    w, h = measurer.measure(box, distance_cm)
    
    assert w > 0, "Width should be positive"
    assert h > 0, "Height should be positive"
    
    print(f"Measured Object: Width={w:.2f}cm, Height={h:.2f}cm at Distance={distance_cm}cm")
    print("Measurement test passed.")

if __name__ == "__main__":
    test_measurement_accuracy()
