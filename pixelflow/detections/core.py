# results/__init__.py

import json
import ast
import cv2
import numpy as np
from pixelflow.validators import (validate_bbox,
                                  validate_masks,
                                  round_to_decimal,
                                  convert_datamarkin_masks,
                                  simplify_polygon)
from typing import (List,
                    Iterator)


# Object-oriented approach instead of a NumPy array-based approach
# Let's see how it goes


class KeyPoint:
    def __init__(self, x: int, y: int, name: str, visibility: bool):
        self.x = x
        self.y = y
        self.name = name
        self.visibility = visibility

    def to_dict(self):
        """
        Convert the KeyPoint object to a dictionary that can be easily converted to JSON.
        """
        return {
            "x": self.x,
            "y": self.y,
            "name": self.name,
            "visibility": self.visibility
        }


class Detection:
    def __init__(self, inference_id=None, bbox=None, masks=None, segments=None, keypoints: List[KeyPoint] = None, class_id=None,
                 class_name=None, labels=None, confidence=None, tracker_id=None, data=None, zones=None, zone_names=None,
                 line_crossings=None, first_seen_time=None, total_time=0.0):
        self.inference_id = inference_id
        self.bbox = validate_bbox(bbox) if bbox is not None else None
        self.masks = masks
        self.segments = segments
        self.keypoints = keypoints if keypoints is not None else None
        self.class_id = class_id
        self.class_name = class_name
        self.labels = labels
        self.confidence = round_to_decimal(confidence)
        self.tracker_id = tracker_id
        self.data = data
        self.zones = zones if zones is not None else []  # List of zone IDs
        self.zone_names = zone_names if zone_names is not None else []  # List of zone names
        self.line_crossings = line_crossings if line_crossings is not None else []  # List of line crossing events
        self.first_seen_time = first_seen_time  # Timestamp/frame when first detected
        self.total_time = total_time  # Total time since first detection (in seconds)

    def to_dict(self):
        """
        Convert the Prediction object to a dictionary that can be easily converted to JSON.
        """
        return {
            "inference_id": self.inference_id,
            "bbox": self.bbox,
            "mask": self.masks,
            "segments": self.segments,
            "keypoints": [kp.to_dict() for kp in self.keypoints] if self.keypoints is not None else None,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "labels": self.labels,
            "confidence": self.confidence,
            "tracker_id": self.tracker_id,
            "data": self.data,
            "zones": self.zones,
            "zone_names": self.zone_names,
            "line_crossings": self.line_crossings,
            "first_seen_time": self.first_seen_time,
            "total_time": self.total_time
        }

    def simplify_masks(self, tolerance: float = 2.0, preserve_topology: bool = True):
        """
        Simplifies the polygon masks using Shapely.

        Args:
            tolerance (float): The tolerance factor for simplification (higher = more simplified).
            preserve_topology (bool): If True, the function will try to preserve the polygon's topology.
        """
        if self.masks:
            # Apply the simplify function to each mask (assuming self.masks is a list of polygons)
            self.masks = [simplify_polygon(mask, tolerance, preserve_topology) for mask in self.masks]


class Detections:
    def __init__(self):
        self.detections: List[Detection] = []

    def show(self):
        # Display the annotated image
        pass

    def add_detection(self, detection: Detection):
        """
        Add a detection to the list.
        """
        self.detections.append(detection)
    
    def update_zones(self, zone_manager):
        """
        Update all predictions with zone information.
        
        Args:
            zone_manager: ZoneManager instance to check zones against
            
        Returns:
            self: Returns self for method chaining
        """
        if zone_manager is not None:
            zone_manager.update(self)
        return self

    def __len__(self):
        return len(self.detections)

    def __iter__(self) -> Iterator[Detection]:
        return iter(self.detections)

    def __getitem__(self, index: int) -> Detection:
        return self.detections[index]


    def simplify(self, tolerance: float = 2.0, preserve_topology: bool = True):
        """
        Simplifies the masks of all detections in the Detections object.

        Args:
            tolerance (float): The tolerance factor for simplification.
            preserve_topology (bool): Whether to preserve the topology.
        """
        for detection in self.detections:
            detection.simplify_masks(tolerance=tolerance, preserve_topology=preserve_topology)
        return self

    def to_json(self):
        """
        Converts the list of detections into a JSON string.
        """
        detections_dict = [detection.to_dict() for detection in self.detections]
        return json.dumps(detections_dict, indent=4)

    def to_dict(self):
        """
        Converts the list of detections into a dictionary.
        """
        return [detection.to_dict() for detection in self.detections]

    def to_json_with_metrics(self) -> str:
        """
        Converts the list of detections into a JSON string.
        More to come here
        """
        detections_dict = [detection.to_dict() for detection in self.detections]
        return json.dumps(detections_dict, indent=4)


# Import filter methods and attach them to Detections class for zero overhead
from .filters import (
    _filter_by_confidence,
    _filter_by_class_id,
    _remap_class_ids,
    _filter_by_size,
    _filter_by_dimensions,
    _filter_by_aspect_ratio,
    _filter_by_zones,
    _filter_by_position,
    _filter_by_relative_size,
    _filter_by_tracking_duration,
    _filter_by_first_seen_time,
    _filter_tracked_objects,
    _remove_duplicates,
    _filter_overlapping,
    _calculate_iou
)

# Import converter functions
from .converters import (
    from_datamarkin_api,
    from_detectron2,
    from_ultralytics,
    from_transformers,
    from_sam,
    from_datamarkin_csv
)

# Attach filter methods directly to Detections class - zero overhead method injection
Detections.filter_by_confidence = _filter_by_confidence
Detections.filter_by_class_id = _filter_by_class_id
Detections.remap_class_ids = _remap_class_ids
Detections.filter_by_size = _filter_by_size
Detections.filter_by_dimensions = _filter_by_dimensions
Detections.filter_by_aspect_ratio = _filter_by_aspect_ratio
Detections.filter_by_zones = _filter_by_zones
Detections.filter_by_position = _filter_by_position
Detections.filter_by_relative_size = _filter_by_relative_size
Detections.filter_by_tracking_duration = _filter_by_tracking_duration
Detections.filter_by_first_seen_time = _filter_by_first_seen_time
Detections.filter_tracked_objects = _filter_tracked_objects
Detections.remove_duplicates = _remove_duplicates
Detections.filter_overlapping = _filter_overlapping
Detections._calculate_iou = lambda self, bbox1, bbox2: _calculate_iou(bbox1, bbox2)