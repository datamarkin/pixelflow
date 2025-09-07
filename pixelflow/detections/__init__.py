# detections/__init__.py
# Export all detection classes and functions for public API

# Import core classes
from .detections import KeyPoint, Detection, Detections

# Import converter functions  
from .converters import (
    from_datamarkin_api,
    from_detectron2, 
    from_ultralytics,
    from_transformers,
    from_sam,
    from_datamarkin_csv
)

# Import filter functions (expose as public API without underscore prefix)
from .filters import (
    _filter_by_confidence as filter_by_confidence,
    _filter_by_class_id as filter_by_class_id,
    _remap_class_ids as remap_class_ids,
    _filter_by_size as filter_by_size,
    _filter_by_dimensions as filter_by_dimensions,
    _filter_by_aspect_ratio as filter_by_aspect_ratio,
    _filter_by_zones as filter_by_zones,
    _filter_by_position as filter_by_position,
    _filter_by_relative_size as filter_by_relative_size,
    _filter_by_tracking_duration as filter_by_tracking_duration,
    _filter_by_first_seen_time as filter_by_first_seen_time,
    _filter_tracked_objects as filter_tracked_objects,
    _remove_duplicates as remove_duplicates,
    _filter_overlapping as filter_overlapping,
    _calculate_iou as calculate_iou
)

# All public exports - maintains exact same API as before
__all__ = [
    # Core classes
    'KeyPoint',
    'Detection', 
    'Detections',
    
    # Converter functions
    'from_datamarkin_api',
    'from_detectron2',
    'from_ultralytics', 
    'from_transformers',
    'from_sam',
    'from_datamarkin_csv',
    
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
    'calculate_iou'
]