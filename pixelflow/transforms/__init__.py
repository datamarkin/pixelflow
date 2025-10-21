"""
Image and Detection Transformations.

Provides comprehensive transformation functions for images and detections,
organized into two categories:

1. Image Operations (image.py) - Image-only transforms (geometric + enhancement)
2. Detection Operations (detections.py) - Image + detection transforms together

## Usage Patterns

### Image-only transformations:
```python
import pixelflow as pf

# Geometric operations
rotated = pf.transform.rotate(image, 45)
flipped = pf.transform.flip_horizontal(image)
cropped = pf.transform.crop(image, [100, 50, 500, 400])

# Enhancement operations
enhanced = pf.transform.clahe(image)
gray = pf.transform.to_grayscale(image)
```

### Detection-aware transformations:
```python
# Transform both image and detections together
rotated_img, rotated_detections = pf.transform.rotate_detections(
    image, detections, angle=45
)

flipped_img, flipped_detections = pf.transform.flip_horizontal_detections(
    image, detections
)

cropped_img, cropped_detections = pf.transform.crop_detections(
    image, detections, bbox=[100, 50, 500, 400]
)
```

### Additional detection utilities:
```python
# Align keypoints to horizontal
aligned_img, aligned_detections = pf.transform.rotate_to_align(
    image, detections, 'p0', 'p9', target_angle=0
)

# Add padding to detection bboxes
padded_detections = pf.transform.add_padding(detections, padding=0.1)

# Update bbox from keypoints
updated_detections = pf.transform.update_bbox_from_keypoints(
    detections, keypoint_names=['p0', 'p9', 'p21', 'p3']
)
```

### Automated inverse transformations:
```python
# Apply transform chain
img, dets = pf.transform.rotate_detections(image, detections, 45)
img, dets = pf.transform.crop_detections(img, dets, [100, 50, 500, 400])

# Run inference on transformed image
results = model.predict(img)

# Automatically undo ALL transforms to get original coordinates
original_coords = pf.transform.inverse_transforms(results)

# Now coordinates match original image
annotated = pf.annotate.box(original_image, original_coords)
```
"""

# Import all functions from submodules
from .image import (
    # Geometric operations
    rotate,
    flip_horizontal,
    flip_vertical,
    crop,

    # Enhancement operations
    clahe,
    to_grayscale,
    auto_contrast,
    normalize,
    gamma_correction,
    standardize
)

from .detections import (
    rotate_detections,
    flip_horizontal_detections,
    flip_vertical_detections,
    crop_detections,
    crop_around_detections,
    rotate_to_align,
    update_bbox_from_keypoints,
    add_padding
)

from .inverse import (
    inverse_transforms
)

__all__ = [
    # Image-only geometric operations
    'rotate',
    'flip_horizontal',
    'flip_vertical',
    'crop',

    # Image-only enhancement operations
    'clahe',
    'to_grayscale',
    'auto_contrast',
    'normalize',
    'gamma_correction',
    'standardize',

    # Detection-aware operations (always return tuple)
    'rotate_detections',
    'flip_horizontal_detections',
    'flip_vertical_detections',
    'crop_detections',
    'crop_around_detections',

    # Additional detection utilities
    'rotate_to_align',
    'update_bbox_from_keypoints',
    'add_padding',

    # Inverse transformations
    'inverse_transforms',
]
