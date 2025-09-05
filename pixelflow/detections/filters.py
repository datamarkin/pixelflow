# _detection_filters.py

from typing import List, Union


def _filter_by_confidence(self, threshold: float):
    """
    Returns a new Detections object containing only detections
    with a confidence score greater than or equal to the given threshold.
    """
    filtered_detections = self.__class__()
    for detection in self.detections:
        if detection.confidence is not None and detection.confidence >= threshold:
            filtered_detections.add_detection(detection)
    return filtered_detections


def _filter_by_class_id(self, class_ids):
    """
    Returns a new Detections object containing only detections
    with class_id matching one of the provided class_ids.
    
    Args:
        class_ids: Single class_id or list of class_ids to filter by
    """
    # Handle single class_id or list of class_ids
    if not isinstance(class_ids, (list, tuple)):
        class_ids = [class_ids]
        
    filtered_detections = self.__class__()
    for detection in self.detections:
        if detection.class_id is not None and detection.class_id in class_ids:
            filtered_detections.add_detection(detection)
    return filtered_detections


def _remap_class_ids(self, from_ids, to_id: int):
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
    from .core import Detection
    
    # Handle single from_id or list of from_ids
    if not isinstance(from_ids, (list, tuple)):
        from_ids = [from_ids]
        
    remapped_detections = self.__class__()
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


def _filter_by_size(self, min_area=None, max_area=None):
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
    filtered_detections = self.__class__()
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


def _filter_by_dimensions(self, min_width=None, max_width=None, min_height=None, max_height=None):
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
    filtered_detections = self.__class__()
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


def _filter_by_aspect_ratio(self, min_ratio=None, max_ratio=None):
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
    filtered_detections = self.__class__()
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


def _filter_by_zones(self, zone_ids, exclude=False):
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
        
    filtered_detections = self.__class__()
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


def _filter_by_position(self, region, margin_percent=0.1, frame_width=None, frame_height=None):
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
    
    filtered_detections = self.__class__()
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


def _filter_by_relative_size(self, min_percent=None, max_percent=None, frame_width=None, frame_height=None):
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
    
    filtered_detections = self.__class__()
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


def _filter_by_tracking_duration(self, min_seconds=None, max_seconds=None):
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
    filtered_detections = self.__class__()
    for detection in self.detections:
        duration = detection.total_time
        
        # Check duration constraints
        if min_seconds is not None and duration < min_seconds:
            continue
        if max_seconds is not None and duration > max_seconds:
            continue
            
        filtered_detections.add_detection(detection)
    return filtered_detections


def _filter_by_first_seen_time(self, start_time=None, end_time=None):
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
    filtered_detections = self.__class__()
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


def _filter_tracked_objects(self, require_tracker_id=True):
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
    filtered_detections = self.__class__()
    for detection in self.detections:
        has_tracker = detection.tracker_id is not None
        
        if (require_tracker_id and has_tracker) or (not require_tracker_id and not has_tracker):
            filtered_detections.add_detection(detection)
            
    return filtered_detections


def _calculate_iou(bbox1, bbox2):
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


def _remove_duplicates(self, iou_threshold=0.8, keep='first'):
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
        
    filtered_detections = self.__class__()
    processed_indices = set()
    
    for i, detection_i in enumerate(self.detections):
        if i in processed_indices or detection_i.bbox is None:
            continue
            
        # Find all detections that overlap with this one
        overlapping_group = [i]
        
        for j, detection_j in enumerate(self.detections[i+1:], start=i+1):
            if j in processed_indices or detection_j.bbox is None:
                continue
                
            iou = _calculate_iou(detection_i.bbox, detection_j.bbox)
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


def _filter_overlapping(self, min_overlap=0.5, target_class_ids=None):
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
    filtered_detections = self.__class__()
    
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
            
            iou = _calculate_iou(detection_i.bbox, detection_j.bbox)
            if iou >= min_overlap:
                has_overlap = True
                break
        
        if has_overlap:
            filtered_detections.add_detection(detection_i)
            
    return filtered_detections