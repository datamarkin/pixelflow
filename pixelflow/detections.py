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

    def filter_by_confidence(self, threshold: float) -> 'Detections':
        """
        Returns a new Detections object containing only detections
        with a confidence score greater than or equal to the given threshold.
        """
        filtered_detections = Detections()
        for detection in self.detections:
            if detection.confidence is not None and detection.confidence >= threshold:
                filtered_detections.add_detection(detection)
        return filtered_detections

    def filter_by_class_id(self, class_ids) -> 'Detections':
        """
        Returns a new Detections object containing only detections
        with class_id matching one of the provided class_ids.
        
        Args:
            class_ids: Single class_id or list of class_ids to filter by
        """
        # Handle single class_id or list of class_ids
        if not isinstance(class_ids, (list, tuple)):
            class_ids = [class_ids]
            
        filtered_detections = Detections()
        for detection in self.detections:
            if detection.class_id is not None and detection.class_id in class_ids:
                filtered_detections.add_detection(detection)
        return filtered_detections

    def remap_class_ids(self, from_ids, to_id: int) -> 'Detections':
        """
        Returns a new Detections object with class IDs remapped.
        
        Args:
            from_ids: Single class_id (int) or list of class_ids to remap from
            to_id: Target class_id to remap to
            
        Returns:
            Detections: New Detections object with remapped class IDs
            
        Example:
            # Remap truck(7) and bus(5) to car(2)
            results = results.remap_class_ids([7, 5], 2)
        """
        # Handle single from_id or list of from_ids
        if not isinstance(from_ids, (list, tuple)):
            from_ids = [from_ids]
            
        remapped_detections = Detections()
        for detection in self.detections:
            # Create a copy of the detection
            new_detection = Detection(
                inference_id=detection.inference_id,
                bbox=detection.bbox,
                masks=detection.masks,
                segments=detection.segments,
                keypoints=detection.keypoints,
                class_id=detection.class_id,
                class_name=detection.class_name,
                labels=detection.labels,
                confidence=detection.confidence,
                tracker_id=detection.tracker_id,
                data=detection.data,
                zones=detection.zones,
                zone_names=detection.zone_names,
                line_crossings=detection.line_crossings,
                first_seen_time=detection.first_seen_time,
                total_time=detection.total_time
            )
            
            # Remap class_id if it matches
            if new_detection.class_id is not None and new_detection.class_id in from_ids:
                new_detection.class_id = to_id
                # Clear class_name since it may no longer be accurate
                new_detection.class_name = None
            
            remapped_detections.add_detection(new_detection)
        
        return remapped_detections

    def filter_by_size(self, min_area=None, max_area=None) -> 'Detections':
        """
        Returns a new Detections object containing only detections
        within the specified area range.
        
        Args:
            min_area: Minimum bounding box area in pixels (inclusive)
            max_area: Maximum bounding box area in pixels (inclusive)
            
        Returns:
            Detections: New Detections object with size-filtered detections
            
        Example:
            # Keep only detections with area between 1000 and 50000 pixels
            results = results.filter_by_size(min_area=1000, max_area=50000)
        """
        filtered_detections = Detections()
        for detection in self.detections:
            if detection.bbox is None:
                continue
                
            # Calculate bounding box area
            x1, y1, x2, y2 = detection.bbox
            area = (x2 - x1) * (y2 - y1)
            
            # Check area constraints
            if min_area is not None and area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue
                
            filtered_detections.add_detection(detection)
        return filtered_detections

    def filter_by_dimensions(self, min_width=None, max_width=None, min_height=None, max_height=None) -> 'Detections':
        """
        Returns a new Detections object containing only detections
        within the specified width and height range.
        
        Args:
            min_width: Minimum bounding box width in pixels (inclusive)
            max_width: Maximum bounding box width in pixels (inclusive)
            min_height: Minimum bounding box height in pixels (inclusive)
            max_height: Maximum bounding box height in pixels (inclusive)
            
        Returns:
            Detections: New Detections object with dimension-filtered detections
            
        Example:
            # Keep only detections with width 50-200px and height 100-300px
            results = results.filter_by_dimensions(min_width=50, max_width=200, min_height=100, max_height=300)
        """
        filtered_detections = Detections()
        for detection in self.detections:
            if detection.bbox is None:
                continue
                
            # Calculate bounding box dimensions
            x1, y1, x2, y2 = detection.bbox
            width = x2 - x1
            height = y2 - y1
            
            # Check width constraints
            if min_width is not None and width < min_width:
                continue
            if max_width is not None and width > max_width:
                continue
                
            # Check height constraints
            if min_height is not None and height < min_height:
                continue
            if max_height is not None and height > max_height:
                continue
                
            filtered_detections.add_detection(detection)
        return filtered_detections

    def filter_by_aspect_ratio(self, min_ratio=None, max_ratio=None) -> 'Detections':
        """
        Returns a new Detections object containing only detections
        within the specified aspect ratio range.
        
        Args:
            min_ratio: Minimum aspect ratio (width/height) (inclusive)
            max_ratio: Maximum aspect ratio (width/height) (inclusive)
            
        Returns:
            Detections: New Detections object with aspect ratio-filtered detections
            
        Example:
            # Keep only square-ish objects (aspect ratio between 0.8 and 1.2)
            results = results.filter_by_aspect_ratio(min_ratio=0.8, max_ratio=1.2)
            
            # Keep only wide objects (aspect ratio > 2.0)
            results = results.filter_by_aspect_ratio(min_ratio=2.0)
        """
        filtered_detections = Detections()
        for detection in self.detections:
            if detection.bbox is None:
                continue
                
            # Calculate aspect ratio
            x1, y1, x2, y2 = detection.bbox
            width = x2 - x1
            height = y2 - y1
            
            # Avoid division by zero
            if height == 0:
                continue
                
            aspect_ratio = width / height
            
            # Check aspect ratio constraints
            if min_ratio is not None and aspect_ratio < min_ratio:
                continue
            if max_ratio is not None and aspect_ratio > max_ratio:
                continue
                
            filtered_detections.add_detection(detection)
        return filtered_detections

    def filter_by_zones(self, zone_ids, exclude=False) -> 'Detections':
        """
        Returns a new Detections object containing only detections
        that are inside (or outside) the specified zones.
        
        Args:
            zone_ids: Single zone_id or list of zone_ids to filter by
            exclude: If True, exclude detections in specified zones (default: include)
            
        Returns:
            Detections: New Detections object with zone-filtered detections
            
        Example:
            # Keep only detections in parking zones
            results = results.filter_by_zones(["parking_1", "parking_2"])
            
            # Exclude detections in restricted zones
            results = results.filter_by_zones([1, 2], exclude=True)
        """
        # Handle single zone_id or list of zone_ids
        if not isinstance(zone_ids, (list, tuple)):
            zone_ids = [zone_ids]
            
        filtered_detections = Detections()
        for detection in self.detections:
            if not hasattr(detection, 'zones') or detection.zones is None:
                # If no zone info, include only if we're excluding zones
                if exclude:
                    filtered_detections.add_detection(detection)
            else:
                # Check if detection is in any of the specified zones
                in_specified_zones = any(z in zone_ids for z in detection.zones)
                
                if (in_specified_zones and not exclude) or (not in_specified_zones and exclude):
                    filtered_detections.add_detection(detection)
                    
        return filtered_detections

    def filter_by_position(self, region, margin_percent=0.1, frame_width=None, frame_height=None) -> 'Detections':
        """
        Returns a new Detections object containing only detections
        in the specified region of the frame.
        
        Args:
            region: Region to filter by ("center", "edge", "top", "bottom", "left", "right", "corners")
            margin_percent: Margin as percentage of frame size (0.0 to 0.5)
            frame_width: Frame width in pixels (required)
            frame_height: Frame height in pixels (required)
            
        Returns:
            Detections: New Detections object with position-filtered detections
            
        Example:
            # Keep only detections in center 60% of frame
            results = results.filter_by_position("center", margin_percent=0.2, frame_width=1920, frame_height=1080)
            
            # Keep only detections near edges
            results = results.filter_by_position("edge", margin_percent=0.1, frame_width=1920, frame_height=1080)
        """
        if frame_width is None or frame_height is None:
            raise ValueError("frame_width and frame_height must be provided")
            
        margin_percent = max(0.0, min(0.5, margin_percent))
        margin_x = int(frame_width * margin_percent)
        margin_y = int(frame_height * margin_percent)
        
        filtered_detections = Detections()
        for detection in self.detections:
            if detection.bbox is None:
                continue
                
            # Calculate detection center
            x1, y1, x2, y2 = detection.bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Check region constraints
            if region == "center":
                # Center region (excluding margins from all sides)
                if (margin_x <= center_x <= frame_width - margin_x and
                    margin_y <= center_y <= frame_height - margin_y):
                    filtered_detections.add_detection(detection)
                    
            elif region == "edge":
                # Edge region (within margins from any side)
                if (center_x < margin_x or center_x > frame_width - margin_x or
                    center_y < margin_y or center_y > frame_height - margin_y):
                    filtered_detections.add_detection(detection)
                    
            elif region == "top":
                if center_y < margin_y:
                    filtered_detections.add_detection(detection)
                    
            elif region == "bottom":
                if center_y > frame_height - margin_y:
                    filtered_detections.add_detection(detection)
                    
            elif region == "left":
                if center_x < margin_x:
                    filtered_detections.add_detection(detection)
                    
            elif region == "right":
                if center_x > frame_width - margin_x:
                    filtered_detections.add_detection(detection)
                    
            elif region == "corners":
                # Corners are both edge horizontally AND vertically
                if ((center_x < margin_x or center_x > frame_width - margin_x) and
                    (center_y < margin_y or center_y > frame_height - margin_y)):
                    filtered_detections.add_detection(detection)
            else:
                raise ValueError(f"Invalid region '{region}'. Valid options are: center, edge, top, bottom, left, right, corners")
                
        return filtered_detections

    def filter_by_relative_size(self, min_percent=None, max_percent=None, frame_width=None, frame_height=None) -> 'Detections':
        """
        Returns a new Detections object containing only detections
        within the specified size range relative to frame size.
        
        Args:
            min_percent: Minimum size as percentage of frame area (0.0 to 1.0)
            max_percent: Maximum size as percentage of frame area (0.0 to 1.0)
            frame_width: Frame width in pixels (required)
            frame_height: Frame height in pixels (required)
            
        Returns:
            Detections: New Detections object with relative size-filtered detections
            
        Example:
            # Keep only detections that are 0.1% to 20% of frame size
            results = results.filter_by_relative_size(min_percent=0.001, max_percent=0.2, 
                                                    frame_width=1920, frame_height=1080)
        """
        if frame_width is None or frame_height is None:
            raise ValueError("frame_width and frame_height must be provided")
            
        frame_area = frame_width * frame_height
        
        filtered_detections = Detections()
        for detection in self.detections:
            if detection.bbox is None:
                continue
                
            # Calculate bounding box area
            x1, y1, x2, y2 = detection.bbox
            detection_area = (x2 - x1) * (y2 - y1)
            relative_size = detection_area / frame_area
            
            # Check relative size constraints
            if min_percent is not None and relative_size < min_percent:
                continue
            if max_percent is not None and relative_size > max_percent:
                continue
                
            filtered_detections.add_detection(detection)
        return filtered_detections

    def filter_by_tracking_duration(self, min_seconds=None, max_seconds=None) -> 'Detections':
        """
        Returns a new Detections object containing only detections
        that have been tracked for the specified duration range.
        
        Args:
            min_seconds: Minimum tracking duration in seconds (inclusive)
            max_seconds: Maximum tracking duration in seconds (inclusive)
            
        Returns:
            Detections: New Detections object with duration-filtered detections
            
        Example:
            # Keep only objects tracked for at least 5 seconds
            results = results.filter_by_tracking_duration(min_seconds=5.0)
            
            # Keep objects tracked between 2-10 seconds
            results = results.filter_by_tracking_duration(min_seconds=2.0, max_seconds=10.0)
        """
        filtered_detections = Detections()
        for detection in self.detections:
            duration = detection.total_time
            
            # Check duration constraints
            if min_seconds is not None and duration < min_seconds:
                continue
            if max_seconds is not None and duration > max_seconds:
                continue
                
            filtered_detections.add_detection(detection)
        return filtered_detections

    def filter_by_first_seen_time(self, start_time=None, end_time=None) -> 'Detections':
        """
        Returns a new Detections object containing only detections
        that were first seen within the specified time range.
        
        Args:
            start_time: Earliest first seen time (inclusive)
            end_time: Latest first seen time (inclusive)
            
        Returns:
            Detections: New Detections object with time-filtered detections
            
        Example:
            # Keep only objects first detected between frame 100-500
            results = results.filter_by_first_seen_time(start_time=100, end_time=500)
        """
        filtered_detections = Detections()
        for detection in self.detections:
            if detection.first_seen_time is None:
                continue
                
            first_seen = detection.first_seen_time
            
            # Check time constraints
            if start_time is not None and first_seen < start_time:
                continue
            if end_time is not None and first_seen > end_time:
                continue
                
            filtered_detections.add_detection(detection)
        return filtered_detections

    def filter_tracked_objects(self, require_tracker_id=True) -> 'Detections':
        """
        Returns a new Detections object containing only detections
        that have tracker IDs (or optionally, no tracker IDs).
        
        Args:
            require_tracker_id: If True, keep only tracked objects. If False, keep only untracked objects.
            
        Returns:
            Detections: New Detections object with tracking-filtered detections
            
        Example:
            # Keep only tracked objects
            results = results.filter_tracked_objects(require_tracker_id=True)
            
            # Keep only untracked objects  
            results = results.filter_tracked_objects(require_tracker_id=False)
        """
        filtered_detections = Detections()
        for detection in self.detections:
            has_tracker = detection.tracker_id is not None
            
            if (require_tracker_id and has_tracker) or (not require_tracker_id and not has_tracker):
                filtered_detections.add_detection(detection)
                
        return filtered_detections

    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            float: IoU value between 0.0 and 1.0
        """
        # Calculate intersection coordinates
        x1_inter = max(bbox1[0], bbox2[0])
        y1_inter = max(bbox1[1], bbox2[1])
        x2_inter = min(bbox1[2], bbox2[2])
        y2_inter = min(bbox1[3], bbox2[3])
        
        # Check if there's any intersection
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
            
        # Calculate intersection area
        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union area
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Avoid division by zero
        if union_area == 0:
            return 0.0
            
        return intersection_area / union_area

    def remove_duplicates(self, iou_threshold=0.8, keep='first') -> 'Detections':
        """
        Returns a new Detections object with duplicate/overlapping detections removed.
        
        Args:
            iou_threshold: IoU threshold for considering detections as duplicates (0.0 to 1.0)
            keep: Which detection to keep when duplicates found ('first', 'last', 'highest_confidence')
            
        Returns:
            Detections: New Detections object with duplicates removed
            
        Example:
            # Remove highly overlapping detections, keep first occurrence
            results = results.remove_duplicates(iou_threshold=0.8, keep='first')
            
            # Remove duplicates, keep highest confidence
            results = results.remove_duplicates(iou_threshold=0.7, keep='highest_confidence')
        """
        if keep not in ['first', 'last', 'highest_confidence']:
            raise ValueError("keep must be 'first', 'last', or 'highest_confidence'")
            
        filtered_detections = Detections()
        processed_indices = set()
        
        for i, detection_i in enumerate(self.detections):
            if i in processed_indices or detection_i.bbox is None:
                continue
                
            # Find all detections that overlap with this one
            overlapping_group = [i]
            
            for j, detection_j in enumerate(self.detections[i+1:], start=i+1):
                if j in processed_indices or detection_j.bbox is None:
                    continue
                    
                iou = self._calculate_iou(detection_i.bbox, detection_j.bbox)
                if iou >= iou_threshold:
                    overlapping_group.append(j)
            
            # Decide which detection to keep from the overlapping group
            if keep == 'first':
                keep_index = overlapping_group[0]
            elif keep == 'last':
                keep_index = overlapping_group[-1]
            else:  # highest_confidence
                best_detection = None
                keep_index = overlapping_group[0]
                best_confidence = -1
                
                for idx in overlapping_group:
                    det = self.detections[idx]
                    conf = det.confidence if det.confidence is not None else 0
                    if conf > best_confidence:
                        best_confidence = conf
                        keep_index = idx
            
            # Add the chosen detection and mark all others as processed
            filtered_detections.add_detection(self.detections[keep_index])
            processed_indices.update(overlapping_group)
            
        return filtered_detections

    def filter_overlapping(self, min_overlap=0.5, target_class_ids=None) -> 'Detections':
        """
        Returns a new Detections object containing only detections
        that overlap with other detections by at least the specified amount.
        
        Args:
            min_overlap: Minimum IoU overlap required (0.0 to 1.0)
            target_class_ids: Optional list of class IDs to check overlap against
            
        Returns:
            Detections: New Detections object with overlapping detections
            
        Example:
            # Keep only detections that overlap significantly with others
            results = results.filter_overlapping(min_overlap=0.3)
            
            # Keep only detections that overlap with person class (class_id=0)
            results = results.filter_overlapping(min_overlap=0.2, target_class_ids=[0])
        """
        filtered_detections = Detections()
        
        for i, detection_i in enumerate(self.detections):
            if detection_i.bbox is None:
                continue
                
            has_overlap = False
            
            for j, detection_j in enumerate(self.detections):
                if i == j or detection_j.bbox is None:
                    continue
                    
                # If target_class_ids specified, only check overlap with those classes
                if target_class_ids is not None:
                    if detection_j.class_id not in target_class_ids:
                        continue
                
                iou = self._calculate_iou(detection_i.bbox, detection_j.bbox)
                if iou >= min_overlap:
                    has_overlap = True
                    break
            
            if has_overlap:
                filtered_detections.add_detection(detection_i)
                
        return filtered_detections

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


def from_datamarkin_api(api_response: dict) -> Detections:
    """
    Converts the Datamarkin API response to a `Detections` object.

    Args:
        api_response (dict): The API response in dictionary format.

    Returns:
        Detections: The corresponding Detections object.
    """

    detections_obj = Detections()

    for obj in api_response.get("predictions", {}).get("objects", []):
        bbox = obj.get("bbox", [])
        mask = obj.get("mask", [])
        keypoints = obj.get("keypoints", [])
        class_name = obj.get("class", "")
        confidence = obj.get("bbox_score", None)

        # Create the Detection object
        detection = Detection(
            bbox=bbox,
            masks=mask,
            keypoints=keypoints,
            class_id=class_name,
            confidence=confidence,
        )

        # Add the prediction to the list
        detections_obj.add_detection(detection)

    return detections_obj



def from_detectron2(detectron2_results) -> Detections:
    """
    Converts Detectron2 results to a custom Detections object.

    Args:
        detectron2_results: Detectron2 inference results containing instances with prediction data.

    Returns:
        Detections: A unified Detections object containing detections.
    """
    detections_obj = Detections()
    
    # Get instances and ensure they're on CPU for processing
    instances = detectron2_results["instances"].to("cpu")
    
    # Check if we have any instances
    if len(instances) == 0:
        return detections_obj

    # Extract prediction data
    # Bounding boxes - Detectron2 uses XYXY format
    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else None
    
    # Confidence scores
    scores = instances.scores.numpy() if instances.has("scores") else None
    
    # Class IDs  
    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else None
    
    # Segmentation masks
    masks = None
    if instances.has("pred_masks"):
        masks = instances.pred_masks.numpy()
    
    # Keypoints if available
    keypoints = None
    if instances.has("pred_keypoints"):
        keypoints = instances.pred_keypoints.numpy()

    # Iterate over each detection
    for i in range(len(instances)):
        # Extract bounding box in XYXY format
        bbox = boxes[i].tolist() if boxes is not None else None
        
        # Extract confidence score
        confidence = float(scores[i]) if scores is not None else None
        
        # Extract class ID  
        class_id = int(classes[i]) if classes is not None else None
        
        # Handle segmentation masks
        mask = None
        if masks is not None:
            mask_data = masks[i].astype(bool)
            mask = mask_data
        
        # Handle keypoints if available
        kpts = None
        if keypoints is not None:
            # Detectron2 keypoints are in format (x, y, visibility) 
            kpt_data = keypoints[i]
            # Convert to PixelFlow KeyPoint format if needed
            # This would need to be implemented based on your KeyPoint class
        
        # Create a Detection object
        detection = Detection(
            bbox=bbox,
            masks=[mask] if mask is not None else None,
            segments=None,
            keypoints=kpts,
            class_id=class_id,
            confidence=confidence
        )

        # Add the detection to the Detections object
        detections_obj.add_detection(detection)

    return detections_obj

# TODO check/verify & improve the mask part
def from_ultralytics(ultralytics_results) -> Detections:
    """
    Converts Ultralytics YOLO results to a custom Detections object.
    
    Supports both detection and segmentation models.
    
    Args:
        ultralytics_results: YOLO results from the Ultralytics library (single result object or list).

    Returns:
        Detections: A unified Detections object containing detections.
    """
    detections_obj = Detections()
    
    # Handle empty results
    if not ultralytics_results:
        return detections_obj
    
    # Handle both single result and list of results
    if isinstance(ultralytics_results, list):
        # Get the first result (YOLO returns a list with one result per image)
        result = ultralytics_results[0]
    else:
        # Already a single result object
        result = ultralytics_results
    
    # Handle case where there are no detections
    if result.boxes is None or len(result.boxes) == 0:
        return detections_obj
    
    # Get all box data in one tensor transfer (more efficient)
    boxes_data = result.boxes.data.cpu().numpy()
    
    # Extract components from the tensor
    # Format: [x1, y1, x2, y2, conf, class_id, ...] or [x1, y1, x2, y2, conf, class_id, track_id]
    xyxy = boxes_data[:, :4]  # Bounding boxes
    confidences = boxes_data[:, 4]  # Confidence scores  
    class_ids = boxes_data[:, 5].astype(int)  # Class IDs
    
    # Check if tracker IDs are available (when using model.track())
    tracker_ids = None
    if hasattr(result.boxes, 'id') and result.boxes.id is not None:
        tracker_ids = result.boxes.id.cpu().numpy().astype(int)
    
    # Check if we have segmentation masks
    has_masks = hasattr(result, 'masks') and result.masks is not None
    
    # Process each detection
    num_detections = len(xyxy)
    
    # Get binary masks if available (shape: [num_masks, height, width])
    binary_masks = None
    orig_shape = None
    if has_masks and hasattr(result.masks, 'data'):
        # Convert masks to numpy arrays
        binary_masks = result.masks.data.cpu().numpy()
        # Get original image shape from masks or result
        if hasattr(result.masks, 'orig_shape'):
            orig_shape = result.masks.orig_shape  # (height, width)
        elif hasattr(result, 'orig_shape'):
            orig_shape = result.orig_shape  # (height, width)
    
    for i in range(num_detections):
        # Basic detection info
        bbox = xyxy[i].tolist()
        confidence = float(confidences[i])
        class_id = int(class_ids[i])
        
        # Extract class name from result if available
        class_name = None
        if hasattr(result, 'names') and result.names:
            if class_id in result.names:
                class_name = result.names[class_id]
        
        # Get tracker ID if available
        tracker_id = None
        if tracker_ids is not None:
            tracker_id = int(tracker_ids[i])
        
        # Handle masks if available
        masks = None
        segments = None
        if has_masks:
            # Store polygon format (xy) for segments
            segments = result.masks.xy[i]
            if segments is not None and len(segments) > 0:
                segments = segments.astype(int).tolist()
            
            # Store binary mask if available
            if binary_masks is not None:
                # Get the binary mask for this detection
                mask = binary_masks[i]
                
                # Handle letterbox padding and resize mask to original shape
                if orig_shape is not None and mask.shape[:2] != orig_shape:
                    # YOLO uses letterboxing: it pads to square then resizes
                    # We need to remove padding and resize to original dimensions
                    mask_h, mask_w = mask.shape[:2]  # Should be 640x640
                    orig_h, orig_w = orig_shape  # Original image dimensions
                    
                    # Calculate the scale and padding used by YOLO
                    scale = min(mask_h / orig_h, mask_w / orig_w)
                    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
                    
                    # Calculate padding
                    pad_h = (mask_h - new_h) // 2
                    pad_w = (mask_w - new_w) // 2
                    
                    # Remove padding
                    mask = mask[pad_h:pad_h + new_h, pad_w:pad_w + new_w]
                    
                    # Resize to original dimensions
                    mask = cv2.resize(mask.astype(np.uint8), 
                                    (orig_w, orig_h),  # cv2 uses (width, height)
                                    interpolation=cv2.INTER_NEAREST)
                
                mask = mask.astype(bool)
                masks = [mask]  # Wrap in list for consistency with API
            elif segments is not None:
                # Fallback to polygon format if binary not available
                masks = [segments]
        
        # Create detection object
        detection = Detection(
            bbox=bbox,
            masks=masks,  # Can be either binary mask or polygon coordinates
            segments=segments,  # Always polygon coordinates
            keypoints=None,
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            tracker_id=tracker_id
        )
        
        detections_obj.add_detection(detection)
    
    # Store the original YOLO masks data for later use if needed
    # This avoids processing masks until they're actually used
    if has_masks:
        detections_obj._ultralytics_masks = result.masks
    
    return detections_obj


def from_transformers(transformers_results):
    pass


def from_sam(sam_results):
    pass


def from_datamarkin_csv(group, height, width) -> Detections:
    """
    Converts CSV data to a `Detections` object.

    Args:
        group: The pandas DataFrame group with the CSV rows.
        height: Image height to denormalize the bounding box and segmentation coordinates.
        width: Image width to denormalize the bounding box and segmentation coordinates.

    Returns:
        Detections: The corresponding Detections object.
    """

    detections_obj = Detections()

    for index, row in group.iterrows():
        # Get the bounding box coordinates and denormalize them
        xmin = int(row['xmin'] * width)
        ymin = int(row['ymin'] * height)
        xmax = int(row['xmax'] * width)
        ymax = int(row['ymax'] * height)

        # Convert normalized points to pixel coordinates for the mask
        segmentation_list = ast.literal_eval(row['segmentation'])
        segmentation_points = []
        for i in range(0, len(segmentation_list), 2):
            x = int(segmentation_list[i] * width)
            y = int(segmentation_list[i + 1] * height)
            segmentation_points.append((x, y))  # Convert to tuple for polygon points

        # Create the Detection object
        detection = Detection(
            bbox=[xmin, ymin, xmax, ymax],
            masks=[segmentation_points],  # Add mask as list of lists of tuples
            keypoints=None,  # TODO
            class_id=row['class'],
            confidence=row.get('confidence', None)  # Add confidence if available
        )

        # Add the prediction to the predictions list
        detections_obj.add_detection(detection)

    return detections_obj

