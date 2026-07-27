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
video = pf.VideoReader("traffic.mp4", width=640)
tracker = pf.tracker.ByteTracker()
zones = pf.Zones()
zones.add_zone([(100, 400), (500, 400), (500, 600), (100, 600)], zone_id="entrance")

for frame in video:
    detections = pf.detections.from_ultralytics(model.predict(frame))
    detections = tracker.update(detections)
    zones.update(detections)

    frame = pf.annotate.box(frame, detections)
    frame = pf.annotate.label(frame, detections)
    frame = pf.annotate.zones(frame, zones)

    if pf.show_frame("Live", frame) == ord('q'):
        break
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
detections = pf.detections.from_detectron2(predictor(image), labels=["person", "car"])

# Mayaku (Detectron2 reimplementation — RGB-native, returns Instances directly)
from mayaku.inference import Predictor
from mayaku.utils.image import read_image
detections = pf.detections.from_mayaku(predictor(read_image("photo.jpg")), labels=pf.COCO_LABELS)

# HuggingFace Transformers
from transformers import pipeline
detector = pipeline("object-detection")
detections = pf.detections.from_transformers(detector(image))

# Florence-2
detections = pf.detections.from_florence2(model_output)

# SAM (Segment Anything)
detections = pf.detections.from_sam(masks, scores)

# RF-DETR
detections = pf.detections.from_rfdetr(model_output)

# Supervision
detections = pf.detections.from_supervision(sv_detections)
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

Purpose-built classes for videos, cameras, and images:

```python
import pixelflow as pf

# Read a video file
video = pf.VideoReader("video.mp4", width=640)
print(f"{video.width}x{video.height}, {video.fps}fps, {len(video)} frames")

for frame in video:       # Replayable — resets on each iteration
    if pf.show_frame("Preview", frame) == ord('q'):
        break

# Load a single image
image = pf.read_image("photo.jpg", width=640)

# Webcam / network stream
cam = pf.CameraStream(0, width=640)
cam = pf.CameraStream("rtsp://camera.local/stream")

# Write processed video
writer = pf.VideoWriter("output.mp4", fps=video.fps)
for frame in video:
    processed = process(frame)
    writer.write(processed)
writer.close()
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
| Mayaku | `from_mayaku()` | Boxes, masks, keypoints |
| HuggingFace Transformers | `from_transformers()` | Boxes, scores |
| Florence-2 | `from_florence2()` | Boxes, polygons, phrases |
| SAM | `from_sam()` | Masks, scores |
| EfficientTAM | `from_efficienttam()` | Masks, scores |
| RF-DETR | `from_rfdetr()` | Boxes, masks |
| Supervision | `from_supervision()` | Boxes, masks, keypoints |
| Falcon Perception | `from_falcon_perception()` | Boxes, masks |
| Datamarkin API | `from_datamarkin()` | Boxes, masks, keypoints |
| Datamarkin CSV | `from_datamarkin_csv()` | Boxes, polygons |

## Documentation

Full documentation available at [https://datamarkin.com/docs/pixelflow/](https://datamarkin.com/docs/pixelflow/)

## Contributing

Contributions welcome! Please read our contributing guidelines.

## License

MIT License - see LICENSE file for details.

---

Built with love by [Datamarkin](https://datamarkin.com)