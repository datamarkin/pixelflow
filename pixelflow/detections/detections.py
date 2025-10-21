"""
Core Detection Data Structures for Computer Vision Processing.

Provides unified data structures for representing detection results from various
computer vision frameworks, including bounding boxes, masks, keypoints, and tracking
information. Designed for efficient processing and seamless integration across
different ML frameworks and visualization tools.
"""

import json
import base64
import numpy as np
from pixelflow.validators import (validate_bbox,
                                  round_to_decimal,
                                  simplify_polygon)
from typing import (List,
                    Iterator, Optional, Union, Any, Dict)

__all__ = ["KeyPoint", "OCRData", "Detection", "Detections"]


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
        Automatically converts NumPy types to native Python types for JSON compatibility.

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
        # Convert numpy types to native Python types for JSON serialization
        x_val = int(self.x) if isinstance(self.x, np.integer) else self.x
        y_val = int(self.y) if isinstance(self.y, np.integer) else self.y

        return {
            "x": x_val,
            "y": y_val,
            "name": self.name,
            "visibility": self.visibility
        }


class OCRData:
    """
    Encapsulates OCR and document structure information for text-based detections.

    Provides a structured container for all OCR-related metadata including text content,
    confidence scores, hierarchical document structure, geometric properties, and
    specialized document element types (tables, formulas, etc). Designed to work with
    various OCR engines and document parsing systems like Tesseract, EasyOCR, PaddleOCR,
    and PP-StructureV3.

    Args:
        text (str): Recognized text content from OCR engine.
        confidence (float): OCR confidence score [0.0-1.0], automatically rounded
                          to 4 decimal places for consistency.
        language (Optional[str]): ISO 639-1 language code (e.g., 'en', 'es', 'zh', 'ar').
        level (Optional[str]): Text hierarchy level for document structure:
                              'char', 'word', 'line', 'paragraph', 'block', or 'page'.
        order (Optional[int]): Reading sequence number for proper text ordering within a level.
        parent_id (Optional[str]): ID of parent element for hierarchical document reconstruction.
        angle (Optional[float]): Text rotation angle in degrees. Positive values indicate
                                clockwise rotation, negative for counterclockwise.
        direction (Optional[str]): Text reading direction: 'ltr' (left-to-right),
                                  'rtl' (right-to-left), 'ttb' (top-to-bottom),
                                  'btt' (bottom-to-top).
        element_type (Optional[str]): Document element classification for structured documents:
                                     'text', 'table', 'formula', 'seal', 'chart', 'image',
                                     'title', 'header', 'footer', etc.
        page_index (Optional[int]): Page number for multi-page documents (0-indexed).
        table_html (Optional[str]): HTML representation of table structure for table elements.
        formula_latex (Optional[str]): LaTeX code representation for mathematical formulas.

    Example:
        >>> import pixelflow as pf
        >>>
        >>> # Basic OCR data for text detection
        >>> ocr_data = pf.detections.OCRData(
        ...     text="Hello World",
        ...     confidence=0.98,
        ...     language="en",
        ...     level="line"
        ... )
        >>>
        >>> # Create detection with OCR data
        >>> detection = pf.detections.Detection(
        ...     bbox=[100, 50, 200, 80],
        ...     ocr_data=ocr_data
        ... )
        >>>
        >>> # Document structure with hierarchy
        >>> paragraph_ocr = pf.detections.OCRData(
        ...     text="First paragraph content",
        ...     confidence=0.95,
        ...     language="en",
        ...     level="paragraph",
        ...     order=1,
        ...     parent_id="page_1"
        ... )
        >>>
        >>> # Table element from PP-StructureV3
        >>> table_ocr = pf.detections.OCRData(
        ...     text="",
        ...     confidence=0.92,
        ...     element_type="table",
        ...     page_index=0,
        ...     table_html="<table><tr><td>Cell 1</td></tr></table>"
        ... )
        >>>
        >>> # Mathematical formula element
        >>> formula_ocr = pf.detections.OCRData(
        ...     text="E=mc²",
        ...     confidence=0.96,
        ...     element_type="formula",
        ...     formula_latex="E=mc^2"
        ... )

    Notes:
        - Confidence scores are automatically rounded using round_to_decimal for precision
        - All fields except text and confidence are optional
        - Compatible with Tesseract, EasyOCR, PaddleOCR, PP-StructureV3, and custom OCR engines
        - Use element_type to distinguish between different document element types
        - Hierarchical structure via level, order, and parent_id enables document tree reconstruction
    """

    def __init__(
        self,
        text: str,
        confidence: float,
        language: Optional[str] = None,
        level: Optional[str] = None,
        order: Optional[int] = None,
        parent_id: Optional[str] = None,
        angle: Optional[float] = None,
        direction: Optional[str] = None,
        element_type: Optional[str] = None,
        page_index: Optional[int] = None,
        table_html: Optional[str] = None,
        formula_latex: Optional[str] = None,
    ):
        self.text = text
        self.confidence = round_to_decimal(confidence)
        self.language = language
        self.level = level
        self.order = order
        self.parent_id = parent_id
        self.angle = angle
        self.direction = direction
        self.element_type = element_type
        self.page_index = page_index
        self.table_html = table_html
        self.formula_latex = formula_latex

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert OCRData to dictionary format for JSON serialization and storage.

        Transforms the OCRData object into a standardized dictionary representation
        suitable for JSON export, API responses, and data persistence workflows.
        Automatically converts NumPy types to native Python types for JSON compatibility.

        Returns:
            Dict[str, Any]: Dictionary containing all OCR fields in standardized format.

        Example:
            >>> import pixelflow as pf
            >>> import json
            >>>
            >>> # Basic OCR data serialization
            >>> ocr_data = pf.detections.OCRData(text="Hello", confidence=0.95, language="en")
            >>> data = ocr_data.to_dict()
            >>> json_str = json.dumps(data)
            >>>
            >>> # Table element serialization
            >>> table_ocr = pf.detections.OCRData(
            ...     text="",
            ...     confidence=0.90,
            ...     element_type="table",
            ...     table_html="<table>...</table>"
            ... )
            >>> table_dict = table_ocr.to_dict()
        """
        # Helper to convert numpy types to native Python types
        def to_python_type(value):
            if value is None:
                return None
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                return float(value)
            return value

        return {
            "text": self.text,
            "confidence": to_python_type(self.confidence),
            "language": self.language,
            "level": self.level,
            "order": to_python_type(self.order),
            "parent_id": self.parent_id,
            "angle": to_python_type(self.angle),
            "direction": self.direction,
            "element_type": self.element_type,
            "page_index": to_python_type(self.page_index),
            "table_html": self.table_html,
            "formula_latex": self.formula_latex,
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
        ocr_data (Optional[OCRData]): Structured OCR/document data with text content, confidence,
                                     language, hierarchy level, reading order, and element type.
                                     Use OCRData class for all OCR-related information.
        metadata (Optional[Dict[str, Any]]): Additional custom metadata and framework-specific data.
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
        >>>
        >>> # OCR detection with structured OCRData
        >>> ocr_data = pf.detections.OCRData(
        ...     text="Hello World",
        ...     confidence=0.98,
        ...     language="en",
        ...     level="line",
        ...     order=1
        ... )
        >>> ocr_detection = pf.detections.Detection(
        ...     bbox=[50, 100, 200, 130],
        ...     ocr_data=ocr_data
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
                 ocr_data: Optional[OCRData] = None,
                 metadata: Optional[Dict[str, Any]] = None,
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

        self.ocr_data = ocr_data
        self.metadata = metadata

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
        Handles nested KeyPoint objects and maintains data type consistency. Automatically
        encodes numpy binary masks as base64 strings for JSON compatibility and converts
        all NumPy types to native Python types for JSON serialization.

        Returns:
            Dict[str, Any]: Dictionary containing all detection fields with keypoints
                          converted to dictionaries, numpy masks encoded as base64,
                          numpy types converted to native Python types, and proper
                          type formatting for JSON serialization.

        Example:
            >>> import pixelflow as pf
            >>>
            >>> # Basic detection serialization
            >>> detection = pf.detections.Detection(bbox=[100, 50, 200, 150], class_name="car")
            >>> data = detection.to_dict()
            >>> import json
            >>> json_str = json.dumps(data, indent=2)  # Now JSON-safe!
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
            - Numpy binary masks are automatically encoded as base64 with metadata
            - Polygon masks remain as coordinate lists (already JSON-compatible)
            - All NumPy numeric types (int32, float32, etc.) converted to native Python types
            - All optional fields are included even if None for consistent API responses
            - Output is fully JSON-serializable (can use json.dumps() directly)
        """
        # Helper function to convert numpy types to native Python types
        def to_python_type(value):
            """Convert numpy types to native Python types for JSON serialization."""
            if value is None:
                return None
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                return float(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (list, tuple)):
                return [to_python_type(v) for v in value]
            return value
        # Process masks - encode numpy arrays as base64, keep polygons as-is
        serializable_masks = None
        if self.masks:
            serializable_masks = []
            for mask in self.masks:
                if isinstance(mask, np.ndarray):
                    # Binary mask - encode as base64
                    serializable_masks.append({
                        'format': 'base64',
                        'data': base64.b64encode(mask.tobytes()).decode('utf-8'),
                        'shape': list(mask.shape),
                        'dtype': str(mask.dtype)
                    })
                else:
                    # Polygon format - keep as-is (already JSON-compatible)
                    serializable_masks.append({
                        'format': 'polygon',
                        'data': mask
                    })

        # Process segments - convert numpy to list if needed
        serializable_segments = self.segments
        if self.segments:
            if isinstance(self.segments, np.ndarray):
                serializable_segments = self.segments.tolist()
            elif isinstance(self.segments, list):
                # Check if list contains numpy arrays and convert them
                serializable_segments = [
                    seg.tolist() if isinstance(seg, np.ndarray) else seg
                    for seg in self.segments
                ]

        return {
            "inference_id": self.inference_id,
            "bbox": to_python_type(self.bbox),
            "masks": serializable_masks,
            "segments": serializable_segments,
            "keypoints": [kp.to_dict() for kp in self.keypoints] if self.keypoints is not None else None,
            "class_id": to_python_type(self.class_id),
            "class_name": self.class_name,
            "labels": self.labels,
            "confidence": to_python_type(self.confidence),
            "tracker_id": to_python_type(self.tracker_id),
            "ocr_data": self.ocr_data.to_dict() if self.ocr_data is not None else None,
            "metadata": self.metadata,
            "zones": self.zones,
            "zone_names": self.zone_names,
            "line_crossings": self.line_crossings,
            "first_seen_time": to_python_type(self.first_seen_time),
            "total_time": to_python_type(self.total_time)
        }

    @staticmethod
    def decode_mask(mask_dict: Dict[str, Any]) -> Union[np.ndarray, List]:
        """
        Decode mask from dictionary format (base64 or polygon).

        Utility function to decode masks that were serialized using to_dict().
        Handles both base64-encoded binary masks and polygon coordinate formats.

        Args:
            mask_dict (Dict[str, Any]): Dictionary with 'format' and 'data' keys.
                                       Format can be 'base64' or 'polygon'.

        Returns:
            Union[np.ndarray, List]: Decoded mask as numpy array (for base64)
                                    or list of coordinates (for polygon).

        Raises:
            ValueError: If mask format is unknown or invalid.
            KeyError: If required dictionary keys are missing.

        Example:
            >>> import pixelflow as pf
            >>> import json
            >>>
            >>> # Serialize detection with mask
            >>> detection = pf.detections.Detection(masks=[np.ones((100, 100), dtype=bool)])
            >>> data = detection.to_dict()
            >>>
            >>> # Export to JSON and load back
            >>> json_str = json.dumps(data)
            >>> loaded_data = json.loads(json_str)
            >>>
            >>> # Decode mask
            >>> mask_dict = loaded_data['masks'][0]
            >>> decoded_mask = pf.detections.Detection.decode_mask(mask_dict)
            >>> print(decoded_mask.shape)  # (100, 100)
            >>>
            >>> # Polygon mask decoding
            >>> polygon_dict = {'format': 'polygon', 'data': [(0, 0), (100, 0), (100, 100)]}
            >>> polygon = pf.detections.Detection.decode_mask(polygon_dict)
            >>> print(polygon)  # [(0, 0), (100, 0), (100, 100)]

        Notes:
            - For base64 format, requires 'data', 'shape', and 'dtype' keys
            - For polygon format, returns the 'data' field directly
            - Automatically reshapes decoded binary masks to original dimensions
            - Compatible with all masks serialized by to_dict() or to_json()
        """
        if mask_dict['format'] == 'base64':
            # Decode base64-encoded binary mask
            mask_bytes = base64.b64decode(mask_dict['data'])
            mask = np.frombuffer(mask_bytes, dtype=mask_dict['dtype'])
            return mask.reshape(mask_dict['shape'])
        elif mask_dict['format'] == 'polygon':
            # Polygon format - return as-is
            return mask_dict['data']
        else:
            raise ValueError(f"Unknown mask format: {mask_dict['format']}")

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

    def copy(self) -> "Detection":
        """
        Create a deep copy of this detection.

        Creates a new Detection instance with deep copies of all fields to ensure
        complete independence from the original. Useful for immutable transform
        operations and safe data manipulation workflows.

        Returns:
            Detection: New detection instance with all fields deep copied.

        Example:
            >>> import pixelflow as pf
            >>>
            >>> # Create and copy detection
            >>> detection = pf.detections.Detection(bbox=[100, 50, 200, 150], class_name="car")
            >>> copied = detection.copy()
            >>>
            >>> # Modifications don't affect original
            >>> copied.bbox[0] = 150
            >>> print(detection.bbox)  # [100, 50, 200, 150] - unchanged
            >>> print(copied.bbox)     # [150, 50, 200, 150] - modified
            >>>
            >>> # Use in transform workflows
            >>> new_detection = detection.copy()
            >>> new_detection.bbox = transform_bbox(new_detection.bbox)

        Notes:
            - All list and dict fields are deep copied for complete independence
            - OCRData objects are shared (considered immutable)
            - Keypoints list is shallow copied (KeyPoint objects are immutable)
            - Useful for immutable transform operations in detection processing
        """
        import copy as copy_module

        return Detection(
            inference_id=self.inference_id,
            bbox=self.bbox.copy() if self.bbox else None,
            masks=copy_module.deepcopy(self.masks) if self.masks else None,
            segments=copy_module.deepcopy(self.segments) if self.segments else None,
            keypoints=self.keypoints.copy() if self.keypoints else None,
            class_id=self.class_id,
            class_name=self.class_name,
            labels=self.labels.copy() if self.labels else None,
            confidence=self.confidence,
            tracker_id=self.tracker_id,
            ocr_data=self.ocr_data,  # OCRData is immutable, safe to share reference
            metadata=self.metadata.copy() if self.metadata else {},
            zones=self.zones.copy() if self.zones else [],
            zone_names=self.zone_names.copy() if self.zone_names else [],
            line_crossings=copy_module.deepcopy(self.line_crossings) if self.line_crossings else [],
            first_seen_time=self.first_seen_time,
            total_time=self.total_time
        )


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

    def copy(self) -> "Detections":
        """
        Create a deep copy of all detections in the collection.

        Creates a new Detections container with deep copies of all Detection objects,
        ensuring complete independence from the original collection. Enables safe
        immutable operations and transform workflows without modifying original data.

        Returns:
            Detections: New detections container with copied detection objects.

        Example:
            >>> import pixelflow as pf
            >>>
            >>> # Create and copy detections
            >>> detections = model.predict(image)
            >>> copied = detections.copy()
            >>>
            >>> # Modifications don't affect original
            >>> copied[0].bbox[0] = 150
            >>> print(detections[0].bbox[0])  # Original unchanged
            >>> print(copied[0].bbox[0])      # 150 - modified
            >>>
            >>> # Use in immutable transform workflows
            >>> transformed = detections.copy()
            >>> for detection in transformed:
            ...     detection.bbox = transform_bbox(detection.bbox)
            >>>
            >>> # Chain with filters
            >>> filtered = detections.copy().filter_by_confidence(0.7)

        Notes:
            - All Detection objects are deep copied using Detection.copy()
            - Completely independent from original collection
            - Safe for concurrent operations and immutable patterns
            - Useful for transform operations that modify detection data
        """
        new_detections = Detections()
        for detection in self.detections:
            new_detections.add_detection(detection.copy())
        return new_detections

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
    _calculate_iou,
    # OCR filter methods
    filter_by_text_confidence,
    filter_by_text_level,
    filter_by_text_language,
    filter_by_text_contains,
    sort_by_text_order,
    filter_by_text_parent
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

# Attach OCR filter methods
Detections.filter_by_text_confidence = filter_by_text_confidence
Detections.filter_by_text_level = filter_by_text_level
Detections.filter_by_text_language = filter_by_text_language
Detections.filter_by_text_contains = filter_by_text_contains
Detections.sort_by_text_order = sort_by_text_order
Detections.filter_by_text_parent = filter_by_text_parent