"""
Core Detection Data Structures for Computer Vision Processing.

Provides unified data structures for representing detection results from various
computer vision frameworks, including bounding boxes, masks, keypoints, and tracking
information. Designed for efficient processing and seamless integration across
different ML frameworks and visualization tools.
"""

import json
from pixelflow.validators import (validate_bbox,
                                  round_to_decimal,
                                  simplify_polygon)
from typing import (List,
                    Iterator, Optional, Union, Any, Dict)

__all__ = ["KeyPoint", "Detection", "Detections"]


# Object-oriented approach instead of a NumPy array-based approach
# Let's see how it goes


class KeyPoint:
    """
    Represents a single keypoint with coordinate and visibility information.
    
    Used for pose estimation and object keypoint detection, storing both spatial
    coordinates and visibility state for structured keypoint data representation.
    
    Args:
        x (int): X coordinate in pixels.
        y (int): Y coordinate in pixels.
        name (str): Descriptive name or label for the keypoint (e.g., "nose", "left_eye").
        visibility (bool): Whether the keypoint is visible and detectable in the image.
    
    Example:
        >>> import pixelflow as pf
        >>> # Create a keypoint for pose estimation
        >>> nose_point = pf.detections.KeyPoint(x=320, y=240, name="nose", visibility=True)
        >>> print(f"Nose at ({nose_point.x}, {nose_point.y})")
        >>> 
        >>> # Create keypoint with occlusion
        >>> hidden_point = pf.detections.KeyPoint(x=150, y=200, name="left_elbow", visibility=False)
    """
    
    def __init__(self, x: int, y: int, name: str, visibility: bool):
        self.x = x
        self.y = y
        self.name = name
        self.visibility = visibility

    def to_dict(self) -> Dict[str, Union[int, str, bool]]:
        """
        Convert KeyPoint to dictionary format for JSON serialization.
        
        Returns:
            Dict[str, Union[int, str, bool]]: Dictionary containing x, y, name, and visibility fields.
        
        Example:
            >>> keypoint = pf.detections.KeyPoint(100, 200, "nose", True)
            >>> data = keypoint.to_dict()
            >>> print(data)  # {'x': 100, 'y': 200, 'name': 'nose', 'visibility': True}
        """
        return {
            "x": self.x,
            "y": self.y,
            "name": self.name,
            "visibility": self.visibility
        }


class Detection:
    """
    Unified representation of a single object detection with comprehensive metadata.
    
    Stores all detection information including bounding boxes, segmentation masks,
    keypoints, classification data, tracking information, and spatial analytics.
    Provides a standardized interface for detection data across different ML frameworks.
    
    Args:
        inference_id (Optional[str]): Unique identifier for the inference session.
        bbox (Optional[List[float]]): Bounding box coordinates in XYXY format [x1, y1, x2, y2].
        masks (Optional[List[Any]]): Segmentation masks in various formats (binary, polygon).
        segments (Optional[List[Any]]): Polygon segments for object boundaries.
        keypoints (Optional[List[KeyPoint]]): List of KeyPoint objects for pose/structure data.
        class_id (Optional[Union[int, str]]): Numeric or string class identifier.
        class_name (Optional[str]): Human-readable class name.
        labels (Optional[List[str]]): Additional classification labels.
        confidence (Optional[float]): Detection confidence score, automatically rounded.
        tracker_id (Optional[int]): Unique tracking identifier for multi-frame tracking.
        data (Optional[Dict[str, Any]]): Additional custom metadata.
        zones (Optional[List[str]]): List of zone identifiers the detection intersects.
        zone_names (Optional[List[str]]): Human-readable names for intersected zones.
        line_crossings (Optional[List[Dict]]): Line crossing events for this detection.
        first_seen_time (Optional[float]): Timestamp when detection first appeared.
        total_time (float): Total duration since first detection in seconds. Default is 0.0.
    
    Example:
        >>> import pixelflow as pf
        >>> # Create basic detection with bounding box
        >>> detection = pf.detections.Detection(
        ...     bbox=[100, 50, 200, 150],
        ...     class_name="person",
        ...     confidence=0.85
        ... )
        >>> 
        >>> # Create detection with tracking and zones
        >>> tracked_detection = pf.detections.Detection(
        ...     bbox=[150, 75, 250, 175],
        ...     class_name="vehicle",
        ...     confidence=0.92,
        ...     tracker_id=42,
        ...     zones=["parking_lot"],
        ...     first_seen_time=1234567890.5
        ... )
        >>> 
        >>> # Detection with keypoints for pose estimation
        >>> pose_detection = pf.detections.Detection(
        ...     bbox=[200, 100, 300, 400],
        ...     class_name="person",
        ...     keypoints=[pf.detections.KeyPoint(250, 120, "nose", True)]
        ... )
    
    Notes:
        - Bounding box coordinates are automatically validated using validate_bbox
        - Confidence scores are automatically rounded using round_to_decimal
        - Zone and line crossing lists are initialized as empty lists if None
        - Compatible with all major ML framework outputs through converter functions
    """
    
    def __init__(self, 
                 inference_id: Optional[str] = None, 
                 bbox: Optional[List[float]] = None, 
                 masks: Optional[List[Any]] = None, 
                 segments: Optional[List[Any]] = None, 
                 keypoints: Optional[List[KeyPoint]] = None, 
                 class_id: Optional[Union[int, str]] = None,
                 class_name: Optional[str] = None, 
                 labels: Optional[List[str]] = None, 
                 confidence: Optional[float] = None, 
                 tracker_id: Optional[int] = None, 
                 data: Optional[Dict[str, Any]] = None, 
                 zones: Optional[List[str]] = None, 
                 zone_names: Optional[List[str]] = None,
                 line_crossings: Optional[List[Dict]] = None, 
                 first_seen_time: Optional[float] = None, 
                 total_time: float = 0.0):
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Detection to dictionary format for JSON serialization.
        
        Returns:
            Dict[str, Any]: Dictionary containing all detection fields with keypoints
                          converted to dictionaries and proper type formatting.
        
        Example:
            >>> detection = pf.detections.Detection(bbox=[100, 50, 200, 150], class_name="car")
            >>> data = detection.to_dict()
            >>> import json
            >>> json_str = json.dumps(data, indent=2)
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

    def simplify_masks(self, tolerance: float = 2.0, preserve_topology: bool = True) -> None:
        """
        Simplify polygon masks to reduce complexity while preserving shape.
        
        Args:
            tolerance (float): Simplification tolerance in pixels. Higher values create
                             more simplified polygons. Range: [0.1, 10.0]. Default is 2.0.
            preserve_topology (bool): Whether to preserve polygon topology during
                                    simplification. Default is True.
        
        Example:
            >>> # Create detection with complex polygon mask
            >>> detection = pf.detections.Detection(masks=[complex_polygon])
            >>> # Simplify with default settings
            >>> detection.simplify_masks()
            >>> # Aggressive simplification
            >>> detection.simplify_masks(tolerance=5.0, preserve_topology=False)
        
        Notes:
            - Modifies masks in-place for memory efficiency
            - Uses Shapely's Douglas-Peucker algorithm for simplification
            - Only processes polygon-format masks, binary masks are unchanged
        """
        if self.masks:
            # Apply the simplify function to each mask (assuming self.masks is a list of polygons)
            self.masks = [simplify_polygon(mask, tolerance, preserve_topology) for mask in self.masks]


class Detections:
    """
    Container for multiple Detection objects with filtering and processing capabilities.
    
    Provides a unified interface for managing collections of detections with support
    for iteration, indexing, filtering, zone management, and serialization. Includes
    dynamically attached filter methods for comprehensive detection processing.
    
    Example:
        >>> import pixelflow as pf
        >>> # Create empty detections container
        >>> detections = pf.detections.Detections()
        >>> 
        >>> # Add detections
        >>> detection1 = pf.detections.Detection(bbox=[100, 50, 200, 150], class_name="person")
        >>> detection2 = pf.detections.Detection(bbox=[300, 100, 400, 200], class_name="car")
        >>> detections.add_detection(detection1)
        >>> detections.add_detection(detection2)
        >>> 
        >>> # Use container features
        >>> print(f"Found {len(detections)} objects")
        >>> for detection in detections:
        ...     print(f"Class: {detection.class_name}")
        >>> 
        >>> # Apply filters (dynamically attached methods)
        >>> high_conf = detections.filter_by_confidence(0.8)
        >>> people_only = detections.filter_by_class_id("person")
    
    Notes:
        - Implements standard Python container protocols (__len__, __iter__, __getitem__)
        - Filter methods are dynamically attached from filters module for zero overhead
        - Supports method chaining for complex filtering workflows
        - Zone management integration for spatial filtering
    """
    
    def __init__(self):
        self.detections: List[Detection] = []

    def show(self) -> None:
        """
        Display annotated image with detections (placeholder for future implementation).
        
        Notes:
            - Placeholder method for future visualization features
            - Will integrate with annotation and display modules
        """
        # Display the annotated image
        pass

    def add_detection(self, detection: Detection) -> None:
        """
        Add a Detection object to the collection.
        
        Args:
            detection (Detection): Detection object to add to the collection.
        
        Example:
            >>> detections = pf.detections.Detections()
            >>> detection = pf.detections.Detection(bbox=[100, 50, 200, 150])
            >>> detections.add_detection(detection)
        """
        self.detections.append(detection)
    
    def update_zones(self, zone_manager: Any) -> 'Detections':
        """
        Update all detections with zone intersection information.
        
        Args:
            zone_manager (Any): ZoneManager instance to check zone intersections.
                              If None, no zone updates are performed.
        
        Returns:
            Detections: Returns self for method chaining.
        
        Example:
            >>> zone_manager = pf.zones.ZoneManager()
            >>> zone_manager.add_polygon_zone("parking", [(0, 0), (100, 0), (100, 100), (0, 100)])
            >>> detections.update_zones(zone_manager)
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


    def simplify(self, tolerance: float = 2.0, preserve_topology: bool = True) -> 'Detections':
        """
        Simplify polygon masks for all detections in the collection.
        
        Args:
            tolerance (float): Simplification tolerance in pixels. Default is 2.0.
            preserve_topology (bool): Whether to preserve polygon topology. Default is True.
        
        Returns:
            Detections: Returns self for method chaining.
        
        Example:
            >>> detections.simplify()  # Use default settings
            >>> detections.simplify(tolerance=5.0, preserve_topology=False)  # Aggressive simplification
        """
        for detection in self.detections:
            detection.simplify_masks(tolerance=tolerance, preserve_topology=preserve_topology)
        return self

    def to_json(self) -> str:
        """
        Convert all detections to JSON string format.
        
        Returns:
            str: JSON string representation of all detections with proper formatting.
        
        Example:
            >>> json_data = detections.to_json()
            >>> with open("detections.json", "w") as f:
            ...     f.write(json_data)
        """
        detections_dict = [detection.to_dict() for detection in self.detections]
        return json.dumps(detections_dict, indent=4)

    def to_dict(self) -> List[Dict[str, Any]]:
        """
        Convert all detections to list of dictionaries.
        
        Returns:
            List[Dict[str, Any]]: List of detection dictionaries for programmatic access.
        
        Example:
            >>> data = detections.to_dict()
            >>> for detection_dict in data:
            ...     print(f"Class: {detection_dict['class_name']}")
        """
        return [detection.to_dict() for detection in self.detections]

    def to_json_with_metrics(self) -> str:
        """
        Convert detections to JSON with additional analytics metrics.
        
        Returns:
            str: JSON string with detections and computed metrics (placeholder for future features).
        
        Example:
            >>> json_with_stats = detections.to_json_with_metrics()
        
        Notes:
            - Currently identical to to_json(), future versions will include analytics
            - Planned metrics: detection counts, confidence distributions, zone statistics
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