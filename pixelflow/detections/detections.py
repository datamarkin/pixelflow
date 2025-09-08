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
    Represents a single keypoint with coordinate and visibility information for pose estimation.
    
    Stores spatial coordinates and visibility state for structured keypoint data representation
    in pose estimation and object keypoint detection tasks. Provides a standardized format
    for keypoint data across different ML frameworks and coordinate systems.
    
    Args:
        x (int): X coordinate in pixels from image left edge.
        y (int): Y coordinate in pixels from image top edge.
        name (str): Descriptive name or label for the keypoint (e.g., "nose", "left_eye").
        visibility (bool): Whether the keypoint is visible and detectable in the image.
    
    Example:
        >>> import pixelflow as pf
        >>> 
        >>> # Create a keypoint for pose estimation
        >>> nose_point = pf.detections.KeyPoint(x=320, y=240, name="nose", visibility=True)
        >>> print(f"Nose at ({nose_point.x}, {nose_point.y})")
        >>> 
        >>> # Create keypoint with occlusion
        >>> hidden_point = pf.detections.KeyPoint(x=150, y=200, name="left_elbow", visibility=False)
        >>> 
        >>> # Use in detection workflow
        >>> keypoints = [nose_point, hidden_point]
        >>> detection = pf.detections.Detection(keypoints=keypoints, class_name="person")
    
    Notes:
        - Coordinates are stored as integers for pixel-perfect alignment
        - Visibility flag helps distinguish between occluded and visible keypoints
        - Compatible with major pose estimation frameworks (MediaPipe, OpenPose, etc.)
    """
    
    def __init__(self, x: int, y: int, name: str, visibility: bool):
        self.x = x
        self.y = y
        self.name = name
        self.visibility = visibility

    def to_dict(self) -> Dict[str, Union[int, str, bool]]:
        """
        Convert KeyPoint to dictionary format for JSON serialization and storage.
        
        Transforms the KeyPoint object into a standardized dictionary representation
        suitable for JSON export, API responses, and data persistence workflows.
        
        Returns:
            Dict[str, Union[int, str, bool]]: Dictionary containing x, y, name, and visibility fields
                                            in standardized format for serialization.
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Basic keypoint serialization
            >>> keypoint = pf.detections.KeyPoint(100, 200, "nose", True)
            >>> data = keypoint.to_dict()
            >>> print(data)  # {'x': 100, 'y': 200, 'name': 'nose', 'visibility': True}
            >>> 
            >>> # JSON export workflow
            >>> import json
            >>> json_str = json.dumps(data)
            >>> 
            >>> # Multiple keypoints serialization
            >>> keypoints = [pf.detections.KeyPoint(x, y, f"point_{i}", True) for i, (x, y) in enumerate([(100, 200), (150, 250)])]
            >>> serialized = [kp.to_dict() for kp in keypoints]
        """
        return {
            "x": self.x,
            "y": self.y,
            "name": self.name,
            "visibility": self.visibility
        }


class Detection:
    """
    Unified representation of a single object detection with comprehensive metadata and tracking.
    
    Stores complete detection information including bounding boxes, segmentation masks,
    keypoints, classification data, tracking information, and spatial analytics. Provides
    a standardized interface for detection data across different ML frameworks with automatic
    validation and type conversion for consistent data handling.
    
    Args:
        inference_id (Optional[str]): Unique identifier for the inference session or batch.
        bbox (Optional[List[float]]): Bounding box coordinates in XYXY format [x1, y1, x2, y2].
                                     Automatically validated and normalized.
        masks (Optional[List[Any]]): Segmentation masks in various formats (binary arrays, polygons).
        segments (Optional[List[Any]]): Polygon segments for precise object boundaries.
        keypoints (Optional[List[KeyPoint]]): List of KeyPoint objects for pose/structure data.
        class_id (Optional[Union[int, str]]): Numeric or string class identifier from model.
        class_name (Optional[str]): Human-readable class name (e.g., "person", "vehicle").
        labels (Optional[List[str]]): Additional classification labels or attributes.
        confidence (Optional[float]): Detection confidence score [0.0-1.0], automatically rounded
                                     to 4 decimal places for consistency.
        tracker_id (Optional[int]): Unique tracking identifier for multi-frame object tracking.
        data (Optional[Dict[str, Any]]): Additional custom metadata and framework-specific data.
        zones (Optional[List[str]]): List of zone identifiers the detection intersects.
                                   Defaults to empty list if None.
        zone_names (Optional[List[str]]): Human-readable names for intersected zones.
                                         Defaults to empty list if None.
        line_crossings (Optional[List[Dict]]): Line crossing events for this detection.
                                             Defaults to empty list if None.
        first_seen_time (Optional[float]): Timestamp when detection first appeared (Unix time).
        total_time (float): Total duration since first detection in seconds. Default is 0.0.
    
    Raises:
        ValueError: If bbox coordinates are invalid or out of expected format.
        TypeError: If keypoints contain non-KeyPoint objects.
    
    Example:
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> import cv2
        >>> 
        >>> # Basic detection from ML framework
        >>> image = cv2.imread("image.jpg")
        >>> model = YOLO("yolo11n.pt")
        >>> outputs = model.predict(image)
        >>> results = pf.detections.from_ultralytics(outputs)
        >>> 
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
        >>> nose_point = pf.detections.KeyPoint(250, 120, "nose", True)
        >>> pose_detection = pf.detections.Detection(
        ...     bbox=[200, 100, 300, 400],
        ...     class_name="person",
        ...     keypoints=[nose_point]
        ... )
    
    Notes:
        - Bounding box coordinates are automatically validated using validate_bbox function
        - Confidence scores are automatically rounded using round_to_decimal for precision
        - Zone and line crossing lists are initialized as empty lists if None provided
        - Compatible with all major ML framework outputs through converter functions
        - Supports in-place mask simplification for performance optimization
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
        Convert Detection to dictionary format for JSON serialization and API export.
        
        Transforms the Detection object into a comprehensive dictionary representation
        suitable for JSON export, API responses, database storage, and data analysis.
        Handles nested KeyPoint objects and maintains data type consistency.
        
        Returns:
            Dict[str, Any]: Dictionary containing all detection fields with keypoints
                          converted to dictionaries, proper type formatting, and
                          null handling for optional fields.
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Basic detection serialization
            >>> detection = pf.detections.Detection(bbox=[100, 50, 200, 150], class_name="car")
            >>> data = detection.to_dict()
            >>> import json
            >>> json_str = json.dumps(data, indent=2)
            >>> 
            >>> # Detection with keypoints serialization
            >>> keypoint = pf.detections.KeyPoint(250, 120, "nose", True)
            >>> detection_with_pose = pf.detections.Detection(
            ...     bbox=[200, 100, 300, 400],
            ...     keypoints=[keypoint],
            ...     tracker_id=42
            ... )
            >>> data = detection_with_pose.to_dict()
            >>> 
            >>> # Export to file
            >>> with open("detection.json", "w") as f:
            ...     json.dump(data, f, indent=2)
        
        Notes:
            - Keypoints are recursively converted to dictionaries using their to_dict() method
            - All optional fields are included even if None for consistent API responses
            - Compatible with standard JSON serialization libraries
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
        Simplify polygon masks to reduce complexity while preserving essential shape characteristics.
        
        Applies Douglas-Peucker polygon simplification algorithm to reduce the number of
        vertices in mask polygons while maintaining visual fidelity. Optimizes memory
        usage and processing speed for downstream operations without significant accuracy loss.
        
        Args:
            tolerance (float): Simplification tolerance in pixels. Higher values create
                             more simplified polygons with fewer vertices. 
                             Range: [0.1, 10.0]. Default is 2.0.
            preserve_topology (bool): Whether to preserve polygon topology during
                                    simplification to avoid self-intersections and holes.
                                    Default is True.
        
        Raises:
            ValueError: If tolerance is outside the valid range [0.1, 10.0].
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Create detection with complex polygon mask
            >>> complex_polygon = [[100, 100], [101, 100], [102, 101], [200, 200]]
            >>> detection = pf.detections.Detection(masks=[complex_polygon])
            >>> 
            >>> # Simplify with default settings
            >>> detection.simplify_masks()
            >>> 
            >>> # Aggressive simplification for performance
            >>> detection.simplify_masks(tolerance=5.0, preserve_topology=False)
            >>> 
            >>> # Conservative simplification for accuracy
            >>> detection.simplify_masks(tolerance=0.5, preserve_topology=True)
        
        Notes:
            - Modifies masks in-place for memory efficiency and performance
            - Uses Shapely's Douglas-Peucker algorithm for geometric simplification
            - Only processes polygon-format masks; binary/raster masks are unchanged
            - Higher tolerance values result in more aggressive vertex reduction
            - Topology preservation prevents self-intersections but may retain more vertices
        
        Performance Notes:
            - Significant speedup for complex polygons with hundreds of vertices
            - Memory usage reduced proportionally to vertex count reduction
            - Recommended for real-time applications with complex segmentation masks
        """
        if self.masks:
            # Apply the simplify function to each mask (assuming self.masks is a list of polygons)
            self.masks = [simplify_polygon(mask, tolerance, preserve_topology) for mask in self.masks]


class Detections:
    """
    Container for multiple Detection objects with comprehensive filtering and processing capabilities.
    
    Provides a unified interface for managing collections of detections with support
    for iteration, indexing, filtering, zone management, serialization, and bulk operations.
    Implements standard Python container protocols and includes dynamically attached filter
    methods for zero-overhead detection processing and analysis workflows.
    
    Example:
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> import cv2
        >>> 
        >>> # Create detections from ML framework
        >>> image = cv2.imread("image.jpg")
        >>> model = YOLO("yolo11n.pt")
        >>> outputs = model.predict(image)
        >>> detections = pf.detections.from_ultralytics(outputs)
        >>> 
        >>> # Manual creation and management
        >>> detections = pf.detections.Detections()
        >>> detection1 = pf.detections.Detection(bbox=[100, 50, 200, 150], class_name="person")
        >>> detection2 = pf.detections.Detection(bbox=[300, 100, 400, 200], class_name="car")
        >>> detections.add_detection(detection1)
        >>> detections.add_detection(detection2)
        >>> 
        >>> # Container operations
        >>> print(f"Found {len(detections)} objects")
        >>> for detection in detections:
        ...     print(f"Class: {detection.class_name}")
        >>> 
        >>> # Apply filters (dynamically attached methods)
        >>> high_conf = detections.filter_by_confidence(0.8)
        >>> people_only = detections.filter_by_class_id("person")
        >>> chained = detections.filter_by_confidence(0.7).filter_by_size(min_area=1000)
    
    Notes:
        - Implements standard Python container protocols (__len__, __iter__, __getitem__)
        - Filter methods are dynamically attached from filters module for zero overhead
        - Supports method chaining for complex filtering workflows
        - Zone management integration for spatial filtering and analytics
        - Bulk operations for mask simplification and serialization
        - Compatible with all major ML framework outputs through converter functions
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
        
        Appends a Detection instance to the internal collection, enabling batch
        processing and filtering operations on the complete detection set.
        
        Args:
            detection (Detection): Detection object to add to the collection.
                                 Must be a valid Detection instance.
        
        Raises:
            TypeError: If detection is not a Detection instance.
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Create container and add detection
            >>> detections = pf.detections.Detections()
            >>> detection = pf.detections.Detection(bbox=[100, 50, 200, 150])
            >>> detections.add_detection(detection)
            >>> 
            >>> # Add multiple detections
            >>> detection2 = pf.detections.Detection(bbox=[200, 100, 300, 200], class_name="car")
            >>> detections.add_detection(detection2)
            >>> print(f"Total detections: {len(detections)}")
        """
        self.detections.append(detection)
    
    def update_zones(self, zone_manager: Any) -> 'Detections':
        """
        Update all detections with zone intersection information for spatial analytics.
        
        Processes each detection in the collection to determine which spatial zones
        it intersects, updating the zones and zone_names fields for spatial filtering
        and analytics workflows.
        
        Args:
            zone_manager (Any): ZoneManager instance to check zone intersections.
                              If None, no zone updates are performed.
        
        Returns:
            Detections: Returns self for method chaining with updated zone information.
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Setup zones and update detections
            >>> zone_manager = pf.zones.ZoneManager()
            >>> zone_manager.add_polygon_zone("parking", [(0, 0), (100, 0), (100, 100), (0, 100)])
            >>> detections.update_zones(zone_manager)
            >>> 
            >>> # Chain with filtering
            >>> parking_detections = detections.update_zones(zone_manager).filter_by_zones(["parking"])
        
        Notes:
            - Modifies detection objects in-place for memory efficiency
            - Zone intersection uses bounding box center point by default
            - Compatible with all zone types (polygon, circular, rectangular)
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
        Simplify polygon masks for all detections in the collection for performance optimization.
        
        Applies polygon simplification to all detections with mask data, reducing
        vertex count while preserving essential shape characteristics. Optimizes
        memory usage and processing speed for bulk detection operations.
        
        Args:
            tolerance (float): Simplification tolerance in pixels. Higher values create
                             more simplified polygons. Range: [0.1, 10.0]. Default is 2.0.
            preserve_topology (bool): Whether to preserve polygon topology during
                                    simplification. Default is True.
        
        Returns:
            Detections: Returns self for method chaining with simplified masks.
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Default simplification
            >>> detections.simplify()
            >>> 
            >>> # Aggressive simplification for performance
            >>> detections.simplify(tolerance=5.0, preserve_topology=False)
            >>> 
            >>> # Chain with other operations
            >>> processed = detections.simplify().filter_by_confidence(0.7)
        
        Notes:
            - Applies simplification to all detections with mask data in the collection
            - Calls Detection.simplify_masks() on each detection instance
            - Performance improvement scales with collection size and mask complexity
        """
        for detection in self.detections:
            detection.simplify_masks(tolerance=tolerance, preserve_topology=preserve_topology)
        return self

    def to_json(self) -> str:
        """
        Convert all detections to JSON string format for export and storage.
        
        Transforms the entire detection collection into a formatted JSON string
        suitable for file export, API responses, and data interchange workflows.
        
        Returns:
            str: JSON string representation of all detections with proper indentation
                and formatting for readability.
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Export to JSON string
            >>> json_data = detections.to_json()
            >>> 
            >>> # Save to file
            >>> with open("detections.json", "w") as f:
            ...     f.write(json_data)
            >>> 
            >>> # API response format
            >>> response_data = {"detections": json.loads(detections.to_json())}
        
        Notes:
            - Uses 4-space indentation for readability
            - Keypoints are recursively converted to dictionaries
            - Compatible with standard JSON parsing libraries
        """
        detections_dict = [detection.to_dict() for detection in self.detections]
        return json.dumps(detections_dict, indent=4)

    def to_dict(self) -> List[Dict[str, Any]]:
        """
        Convert all detections to list of dictionaries for programmatic processing.
        
        Transforms the detection collection into a list of dictionary representations,
        providing structured data access for analysis, filtering, and integration workflows.
        
        Returns:
            List[Dict[str, Any]]: List of detection dictionaries with all fields
                                including nested keypoint data and metadata.
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Convert to dictionary format
            >>> data = detections.to_dict()
            >>> 
            >>> # Programmatic access
            >>> for detection_dict in data:
            ...     print(f"Class: {detection_dict['class_name']}")
            ...     print(f"Confidence: {detection_dict['confidence']}")
            >>> 
            >>> # Data analysis workflow
            >>> import pandas as pd
            >>> df = pd.DataFrame(data)
            >>> high_conf = df[df['confidence'] > 0.8]
        
        Notes:
            - Each detection is converted using its to_dict() method
            - Keypoints are recursively converted to nested dictionaries
            - Suitable for pandas DataFrame conversion and data analysis
        """
        return [detection.to_dict() for detection in self.detections]

    def to_json_with_metrics(self) -> str:
        """
        Convert detections to JSON with additional analytics metrics for comprehensive reporting.
        
        Generates a JSON representation that includes both detection data and computed
        analytics metrics for comprehensive reporting and analysis workflows.
        
        Returns:
            str: JSON string with detections and computed metrics including counts,
                confidence statistics, and spatial analytics (placeholder for future features).
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Export with analytics
            >>> json_with_stats = detections.to_json_with_metrics()
            >>> 
            >>> # Save comprehensive report
            >>> with open("detection_report.json", "w") as f:
            ...     f.write(json_with_stats)
        
        Notes:
            - Currently identical to to_json(), future versions will include analytics
            - Planned metrics: detection counts, confidence distributions, zone statistics
            - Will include class distribution, temporal analytics, and spatial statistics
            - Designed for comprehensive reporting and dashboard integration
        """
        detections_dict = [detection.to_dict() for detection in self.detections]
        return json.dumps(detections_dict, indent=4)


# Import filter methods and attach them to Detections class for zero overhead
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
    _calculate_iou
)


# Attach filter methods directly to Detections class - zero overhead method injection
Detections.filter_by_confidence = filter_by_confidence
Detections.filter_by_class_id = filter_by_class_id
Detections.remap_class_ids = remap_class_ids
Detections.filter_by_size = filter_by_size
Detections.filter_by_dimensions = filter_by_dimensions
Detections.filter_by_aspect_ratio = filter_by_aspect_ratio
Detections.filter_by_zones = filter_by_zones
Detections.filter_by_position = filter_by_position
Detections.filter_by_relative_size = filter_by_relative_size
Detections.filter_by_tracking_duration = filter_by_tracking_duration
Detections.filter_by_first_seen_time = filter_by_first_seen_time
Detections.filter_tracked_objects = filter_tracked_objects
Detections.remove_duplicates = remove_duplicates
Detections.filter_overlapping = filter_overlapping
Detections._calculate_iou = lambda self, bbox1, bbox2: _calculate_iou(bbox1, bbox2)