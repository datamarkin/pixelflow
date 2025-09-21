# PixelFlow

[![PyPI version](https://badge.fury.io/py/pixelflow.svg)](https://badge.fury.io/py/pixelflow)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A computer vision post-processing library for visualizing and analyzing detection results from ML models. PixelFlow takes detection outputs from frameworks like YOLO, Detectron2, and others, then provides powerful tools for annotation, tracking, filtering, and analysis.

## ✨ Features

- 🎨 **Flexible Annotation Tools**: Box, mask, keypoint, heatmap, blur, pixelate, and motion trail annotations
- 🎥 **Efficient Video Processing**: Lazy frame loading generator for memory-efficient video handling
- 🔍 **Zone-based Filtering**: Spatial region management for targeted analysis
- 🔄 **Multi-framework Support**: Seamless integration with Detectron2, YOLO, Ultralytics, and more
- ⚡ **High-performance Drawing**: Optimized OpenCV-based rendering
- 📊 **Unified Data Structures**: Standardized prediction and results formats across frameworks
- 🎯 **Object Tracking**: Built-in tracking capabilities for video sequences

## 🚀 Quick Start

### Installation

```bash
pip install pixelflow
```

### Basic Usage

```python
import cv2
import pixelflow as pf

# Load an image
image = cv2.imread("image.jpg")

# Create detection results (example with dummy data)
predictions = [
    pf.Prediction(
        bbox=[100, 100, 200, 200],
        confidence=0.95,
        class_name="person"
    )
]
results = pf.Results(predictions=predictions)

# Annotate the image
annotated = pf.annotate_boxes(image, results)

# Save the result
cv2.imwrite("annotated_image.jpg", annotated)
```

### Video Processing

```python
import pixelflow as pf

# Process video with lazy loading
for frame in pf.video_loader("input_video.mp4"):
    # Your processing logic here
    processed_frame = pf.annotate_boxes(frame, results)
    # Save or display the frame
```

### Zone-based Filtering

```python
import pixelflow as pf

# Define a zone
zone = pf.PolygonZone(polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])

# Filter predictions within the zone
filtered_results = pf.filter_by_zone(results, zone)
```

## 📖 Documentation

- **API Reference**: [pixelflow.datamarkin.com](https://pixelflow.datamarkin.com)
- **Examples**: Check the `/examples` directory for more detailed usage examples
- **Contributing**: See `CONTRIBUTING.md` for development guidelines

## 🔧 Supported Frameworks

PixelFlow provides seamless integration with popular ML frameworks:

- **Detectron2**: `Results.from_detectron2()`
- **Ultralytics/YOLO**: `Results.from_ultralytics()`
- **Custom Models**: Easy adapter pattern for any framework

## 📋 Requirements

- Python 3.9+
- OpenCV (cv2)
- NumPy
- Shapely
- SciPy

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Homepage**: [https://github.com/datamarkin/pixelflow](https://github.com/datamarkin/pixelflow)
- **Issues**: [https://github.com/datamarkin/pixelflow/issues](https://github.com/datamarkin/pixelflow/issues)
- **Documentation**: [https://pixelflow.datamarkin.com](https://pixelflow.datamarkin.com)