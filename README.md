# PixelFlow

[![PyPI version](https://badge.fury.io/py/pixelflow.svg)](https://badge.fury.io/py/pixelflow)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**The computer vision library that gets out of your way.**

PixelFlow provides a unified, intuitive API for object detection, tracking, annotation, and video processing. Write clean, readable pipelines that work with any ML framework.

```python
import pixelflow as pf
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
media = pf.Media("traffic.mp4")
tracker = pf.tracker.ByteTracker()
zones = pf.Zones()
zones.add_zone([(100, 400), (500, 400), (500, 600), (100, 600)], zone_id="entrance")

for frame in media.frames:
    detections = pf.detections.from_ultralytics(model.predict(frame))
    detections = tracker.update(detections)
    zones.update(detections)

    frame = pf.annotate.box(frame, detections)
    frame = pf.annotate.label(frame, detections)
    frame = pf.annotate.zones(frame, zones)

    pf.show_frame("Live", frame)
```

## Installation

```bash
pip install pixelflow
```

## Core Concepts

### Detections - One Format, Every Framework

Convert outputs from any ML framework into a unified format:

```python
import pixelflow as pf

# Ultralytics YOLO
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
detections = pf.detections.from_ultralytics(model.predict(image))

# Detectron2
from detectron2.engine import DefaultPredictor
predictor = DefaultPredictor(cfg)
detections = pf.detections.from_detectron2(predictor(image), class_names=["person", "car"])

# HuggingFace Transformers
from transformers import pipeline
detector = pipeline("object-detection")
detections = pf.detections.from_transformers(detector(image))

# Florence-2
detections = pf.detections.from_florence2(model_output)

# SAM (Segment Anything)
detections = pf.detections.from_sam(masks, scores)

# OCR Engines
detections = pf.detections.from_tesseract(tesseract_output)
detections = pf.detections.from_paddleocr(paddleocr_output)
detections = pf.detections.from_easyocr(easyocr_output)
```

### Powerful Filtering

Chain filters for complex queries with zero overhead:

```python
# Get high-confidence people in the parking zone, tracked for 5+ seconds
results = (detections
    .filter_by_confidence(min_confidence=0.7)
    .filter_by_class_id("person")
    .filter_by_zones(["parking_lot"])
    .filter_by_tracking_duration(min_seconds=5.0))

# Size-based filtering
large_objects = detections.filter_by_size(min_area=10000)
tall_objects = detections.filter_by_dimensions(min_height=200)
squares = detections.filter_by_aspect_ratio(min_ratio=0.9, max_ratio=1.1)

# Spatial filtering
left_side = detections.filter_by_position(max_x=frame_width // 2)

# Remove overlapping detections
unique = detections.remove_duplicates(iou_threshold=0.5)

# OCR-specific filters
titles = detections.filter_by_text_level("title")
english_text = detections.filter_by_text_language("en")
```

### Media Handling

Unified interface for videos, images, webcams, and streams:

```python
import pixelflow as pf

# Video file with automatic resizing
media = pf.Media("video.mp4", width=640)
print(media.info)  # MediaInfo(resolution=1920x1080, fps=30.00, frames=900, duration=30.00s)

for frame in media.frames:  # Lazy loading, memory efficient
    # Process frame
    pf.show_frame("Preview", frame)

# Webcam
webcam = pf.Media(0)

# Network stream
stream = pf.Media("rtsp://camera.local/stream")

# Write processed video
for frame in pf.Media("input.mp4").frames:
    processed = process(frame)
    pf.write_frame("output.mp4", processed, media.info)
```

### Zone-Based Analytics

Define spatial regions and track what enters them:

```python
import pixelflow as pf

zones = pf.Zones()

# Define zones with different trigger strategies
zones.add_zone(
    polygon=[(100, 400), (300, 400), (300, 600), (100, 600)],
    zone_id="entrance",
    name="Main Entrance",
    trigger_strategy="bottom_center"  # Trigger when bottom-center of bbox enters
)

zones.add_zone(
    polygon=[(500, 200), (700, 200), (700, 500), (500, 500)],
    zone_id="restricted",
    trigger_strategy="percentage",
    overlap_threshold=0.3  # Trigger when 30% of bbox is inside
)

# Update detections with zone info
zones.update(detections)

# Access zone statistics
print(zones.get_zone_counts())  # {'entrance': 3, 'restricted': 1}
print(zones.get_zone_stats())   # Detailed stats including total_entered

# Filter by zone
entrance_only = zones.filter_by_zones(detections, ["entrance"])
not_restricted = zones.filter_by_zones(detections, ["restricted"], exclude=True)
```

**Trigger Strategies:**
- `center` - Bounding box center point (default)
- `bottom_center` - Bottom-center point (great for people/vehicles)
- `percentage` - Percentage of bbox overlap
- `overlap` - Any intersection
- Multiple strategies with `mode="any"` or `mode="all"`

### Object Tracking

Built-in ByteTrack for multi-object tracking:

```python
import pixelflow as pf

tracker = pf.tracker.ByteTracker()

for frame in media.frames:
    detections = pf.detections.from_ultralytics(model.predict(frame))
    detections = tracker.update(detections)

    for det in detections:
        print(f"Track ID: {det.tracker_id}, Class: {det.class_name}")
        print(f"First seen: {det.first_seen_time}, Duration: {det.total_time}s")
```

### Rich Annotations

Beautiful, customizable visualizations:

```python
import pixelflow as pf

# Bounding boxes and labels
frame = pf.annotate.box(frame, detections, thickness=2)
frame = pf.annotate.label(frame, detections, font_scale=0.6)

# Segmentation masks
frame = pf.annotate.mask(frame, detections, opacity=0.4)

# Keypoints and skeletons (pose estimation)
frame = pf.annotate.keypoint(frame, detections)
frame = pf.annotate.keypoint_skeleton(frame, detections)

# Zone visualization
frame = pf.annotate.zones(frame, zones)
frame = pf.annotate.crossings(frame, crossings)

# Privacy protection
frame = pf.annotate.blur(frame, detections, kernel_size=51)
frame = pf.annotate.pixelate(frame, detections, pixel_size=15)

# Shapes
frame = pf.annotate.oval(frame, detections)
frame = pf.annotate.polygon(frame, detections)
```

### Image Transforms

Comprehensive image and detection-aware transformations:

```python
import pixelflow as pf

# Image-only transforms
rotated = pf.transform.rotate(image, 45)
flipped = pf.transform.flip_horizontal(image)
cropped = pf.transform.crop(image, [100, 50, 500, 400])

# Enhancement
enhanced = pf.transform.clahe(image)
gray = pf.transform.to_grayscale(image)
corrected = pf.transform.gamma_correction(image, gamma=1.2)

# Detection-aware transforms (coordinates update automatically)
rotated_img, rotated_dets = pf.transform.rotate_detections(image, detections, 45)
flipped_img, flipped_dets = pf.transform.flip_horizontal_detections(image, detections)
cropped_img, cropped_dets = pf.transform.crop_detections(image, detections, bbox=[100, 50, 500, 400])

# Automatic inverse transforms (undo all transforms)
original_coords = pf.transform.inverse_transforms(transformed_detections)
```

### Sliced Inference for Large Images

Detect small objects in high-resolution images:

```python
import pixelflow as pf

slicer = pf.SlicedInference(
    slice_height=640,
    slice_width=640,
    overlap_ratio_h=0.2,
    overlap_ratio_w=0.2
)

def detector(image):
    return pf.detections.from_ultralytics(model.predict(image))

# Run on 4K image - automatically slices, detects, and merges
large_image = cv2.imread("satellite.jpg")  # 4000x3000
detections = slicer.predict(large_image, detector)
```

### Line Crossing Detection

Track objects crossing defined lines:

```python
import pixelflow as pf

crossings = pf.Crossings()
crossings.add_line(
    start=(100, 300),
    end=(500, 300),
    line_id="entry_line",
    direction="down"  # Only count downward crossings
)

for frame in media.frames:
    detections = tracker.update(pf.detections.from_ultralytics(model.predict(frame)))
    crossings.update(detections)

    print(f"Crossed: {crossings.get_counts()}")
    frame = pf.annotate.crossings(frame, crossings)
```

### Serialization

Export and import detection data:

```python
# To JSON
json_str = detections.to_json()

# To dictionary (for pandas, etc.)
data = detections.to_dict()
df = pd.DataFrame(data)

# With metrics
report = detections.to_json_with_metrics()
```

## Supported Frameworks

| Framework | Converter | Features |
|-----------|-----------|----------|
| Ultralytics (YOLO) | `from_ultralytics()` | Boxes, masks, keypoints |
| Detectron2 | `from_detectron2()` | Boxes, masks, keypoints |
| HuggingFace Transformers | `from_transformers()` | Boxes, scores |
| Florence-2 | `from_florence2()` | Boxes, phrases |
| SAM | `from_sam()` | Masks, scores |
| Tesseract OCR | `from_tesseract()` | Text, boxes, confidence |
| PaddleOCR | `from_paddleocr()` | Text, boxes, structure |
| EasyOCR | `from_easyocr()` | Text, boxes |
| PP-Structure | `from_ppstructure()` | Tables, formulas, layouts |

## Documentation

Full documentation available at [https://datamarkin.com/docs/pixelflow/](https://datamarkin.com/docs/pixelflow/)

## Contributing

Contributions welcome! Please read our contributing guidelines.

## License

MIT License - see LICENSE file for details.

---

Built with love by [Datamarkin](https://datamarkin.com)