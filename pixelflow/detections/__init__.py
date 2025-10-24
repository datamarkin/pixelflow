# detections/__init__.py
# Export all detection classes and functions for public API

# Import core classes
from .detections import KeyPoint, Detection, Detections

# Import converter functions
from .converters import (
    from_datamarkin,
    from_florence2,
    from_detectron2,
    from_ultralytics,
    from_transformers,
    from_sam,
    from_datamarkin_csv,
    # OCR converters
    from_tesseract,
    from_easyocr,
    from_paddleocr,
    from_ppstructure
)

# Import filter functions (now public API)
from .filters import (
    filter_by_confidence,
    filter_by_class_id,
    remap_class_ids,
    filter_by_size,
    filter_by_dimensions,
    filter_by_aspect_ratio,
    filter_by_zones,
    filter_by_position,
    filter_by_relative_size,
    filter_by_tracking_duration,
    filter_by_first_seen_time,
    filter_tracked_objects,
    remove_duplicates,
    filter_overlapping,
    # OCR filter functions
    filter_by_text_confidence,
    filter_by_text_level,
    filter_by_text_language,
    filter_by_text_contains,
    sort_by_text_order,
    filter_by_text_parent
)

# All public exports - maintains exact same API as before
__all__ = [
    # Core classes
    'KeyPoint',
    'Detection', 
    'Detections',
    
    # Converter functions
    'from_datamarkin',
    'from_florence2',
    'from_detectron2',
    'from_ultralytics',
    'from_transformers',
    'from_sam',
    'from_datamarkin_csv',
    # OCR converters
    'from_tesseract',
    'from_easyocr',
    'from_paddleocr',
    'from_ppstructure',
    
    # Filter functions
    'filter_by_confidence',
    'filter_by_class_id',
    'remap_class_ids',
    'filter_by_size',
    'filter_by_dimensions',
    'filter_by_aspect_ratio',
    'filter_by_zones',
    'filter_by_position',
    'filter_by_relative_size',
    'filter_by_tracking_duration',
    'filter_by_first_seen_time',
    'filter_tracked_objects',
    'remove_duplicates',
    'filter_overlapping',
    # OCR filter functions
    'filter_by_text_confidence',
    'filter_by_text_level',
    'filter_by_text_language',
    'filter_by_text_contains',
    'sort_by_text_order',
    'filter_by_text_parent'
]