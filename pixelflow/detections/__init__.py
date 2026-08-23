# detections/__init__.py
# Export the detection result type and its converters

from .detections import KeyPoint, Detection, Detections

from .converters import (
    from_datamarkin,
    from_florence2,
    from_detectron2,
    from_mayaku,
    from_ultralytics,
    from_transformers,
    from_sam,
    from_arrays,
    from_supervision,
    from_rfdetr,
    from_falcon_perception,
    from_efficienttam,
    from_easyocr
)

# Filters are not exported as free functions: every one of them is attached to
# Detections as a method, and two ways to call the same thing is one too many.

__all__ = [
    # Core classes
    'KeyPoint',
    'Detection',
    'Detections',

    # Converter functions
    'from_datamarkin',
    'from_florence2',
    'from_detectron2',
    'from_mayaku',
    'from_ultralytics',
    'from_transformers',
    'from_sam',
    'from_arrays',
    'from_supervision',
    'from_rfdetr',
    'from_falcon_perception',
    'from_efficienttam',
    'from_easyocr',
]
