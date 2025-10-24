"""
Detection-Aware Transformations.

Functions that transform both images and detections together, keeping coordinates synchronized.
All functions return (image, detections) tuple except crop_around_detections.
"""

import cv2
import numpy as np
import math
from typing import Tuple, Union, Dict, List, Optional
from ..detections.detections import Detections, Detection, KeyPoint
from . import image as image_transforms

__all__ = [
    'rotate_detections',
    'flip_horizontal_detections',
    'flip_vertical_detections',
    'crop_detections',
    'crop_around_detections',
    'rotate_to_align',
    'update_bbox_from_keypoints',
    'add_padding'
]


def rotate_detections(
    image: np.ndarray,
    detections: Detections,
    angle: float,
    center: Tuple[float, float] = None,
    fillcolor: Tuple[int, int, int] = None,
    track_metadata: bool = True
) -> Tuple[np.ndarray, Detections]:
    """
    Rotate image and update detection coordinates.

    Args:
        image: Input image (H, W, 3) BGR format
        detections: Detections to transform (modified in-place)
        angle: Rotation angle in degrees (positive = counter-clockwise)
        center: Rotation center (x, y). If None, uses image center.
        fillcolor: Fill color for areas outside original image. If None, uses edge pixels.
        track_metadata: If True, writes transform metadata to detection.metadata

    Returns:
        (rotated_image, rotated_detections) tuple

    Example:
        ```python
        import pixelflow as pf
        import cv2

        image = cv2.imread("image.jpg")
        detections = model.predict(image)

        # Rotate 45 degrees (modifies detections in-place)
        rotated_img, rotated_dets = pf.transform.rotate_detections(
            image, detections, angle=45
        )

        # To preserve original, copy first
        rotated_img, rotated_dets = pf.transform.rotate_detections(
            image, detections.copy(), angle=45
        )
        ```

    Note:
        - Rotation is performed around image center by default
        - Uses edge pixel replication if fillcolor not specified
        - Detections are modified IN-PLACE for performance
        - Use detections.copy() if you need to preserve the original
    """
    h, w = image.shape[:2]

    # Use image center if center not provided
    if center is None:
        center = (w / 2.0, h / 2.0)

    # Rotate image using image.py function
    rotated_img = image_transforms.rotate(image, angle, center=center, fillcolor=fillcolor)
    new_h, new_w = rotated_img.shape[:2]

    # Get rotation matrix for coordinate transformation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate canvas expansion
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # Transform detections in-place using rotation matrix
    for detection in detections:
        # Transform bbox
        if detection.bbox is not None:
            x1, y1, x2, y2 = detection.bbox
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            # Apply rotation matrix using cv2.transform (handles rotation + translation)
            rotated_corners = cv2.transform(corners.reshape(1, -1, 2), rotation_matrix)[0]

            xs = rotated_corners[:, 0]
            ys = rotated_corners[:, 1]
            detection.bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

        # Transform keypoints
        if detection.keypoints is not None:
            # Collect all keypoint coordinates
            keypoint_coords = np.array([[kp.x, kp.y] for kp in detection.keypoints], dtype=np.float32)
            # Apply rotation matrix using cv2.transform
            transformed_coords = cv2.transform(keypoint_coords.reshape(1, -1, 2), rotation_matrix)[0]

            # Create new keypoints with transformed coordinates
            new_keypoints = []
            for i, kp in enumerate(detection.keypoints):
                new_kp = KeyPoint(
                    x=int(transformed_coords[i, 0]),
                    y=int(transformed_coords[i, 1]),
                    name=kp.name,
                    visibility=kp.visibility
                )
                new_keypoints.append(new_kp)
            detection.keypoints = new_keypoints

        # Transform segments (polygons)
        if detection.segments is not None:
            if isinstance(detection.segments, np.ndarray):
                # Single segment as numpy array
                segment_coords = detection.segments.astype(np.float32).reshape(1, -1, 2)
                transformed = cv2.transform(segment_coords, rotation_matrix)[0]
                detection.segments = transformed
            elif isinstance(detection.segments, list):
                # Multiple segments as list
                new_segments_list = []
                for segment in detection.segments:
                    if isinstance(segment, np.ndarray):
                        # Segment is numpy array
                        segment_coords = segment.astype(np.float32).reshape(1, -1, 2)
                        transformed = cv2.transform(segment_coords, rotation_matrix)[0]
                        new_segments_list.append(transformed)
                    elif isinstance(segment, list):
                        # Segment is list of [x, y] coordinates
                        segment_coords = np.array(segment, dtype=np.float32).reshape(1, -1, 2)
                        transformed = cv2.transform(segment_coords, rotation_matrix)[0]
                        new_segments_list.append(transformed.tolist())
                detection.segments = new_segments_list

        # Transform masks
        if detection.masks is not None:
            new_masks = []
            for mask in detection.masks:
                if isinstance(mask, np.ndarray):
                    # Binary mask - use warpAffine
                    rotated_mask = cv2.warpAffine(
                        mask.astype(np.uint8),
                        rotation_matrix,
                        (new_w, new_h),
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0
                    ).astype(bool)
                    new_masks.append(rotated_mask)
                else:
                    if isinstance(mask, list):
                        # Polygon mask - use cv2.transform
                        mask_coords = np.array(mask, dtype=np.float32).reshape(1, -1, 2)
                        transformed = cv2.transform(mask_coords, rotation_matrix)[0]
                        new_masks.append(transformed.tolist())
                    else:
                        new_masks.append(mask)
            detection.masks = new_masks

        # Track metadata
        if track_metadata:
            if detection.metadata is None:
                detection.metadata = {}
            if 'transforms' not in detection.metadata:
                detection.metadata['transforms'] = []

            detection.metadata['transforms'].append({
                'type': 'rotate',
                'angle': angle,
                'center': center,
                'original_size': (h, w)
            })

    return rotated_img, detections


def flip_horizontal_detections(
    image: np.ndarray,
    detections: Detections,
    track_metadata: bool = True
) -> Tuple[np.ndarray, Detections]:
    """
    Flip image horizontally and update detection coordinates.

    Args:
        image: Input image (H, W, 3) BGR format
        detections: Detections to transform (modified in-place)
        track_metadata: If True, writes transform metadata to detection.metadata

    Returns:
        (flipped_image, flipped_detections) tuple

    Example:
        ```python
        import pixelflow as pf
        import cv2

        image = cv2.imread("image.jpg")
        detections = model.predict(image)

        # Flip horizontally (modifies detections in-place)
        flipped_img, flipped_dets = pf.transform.flip_horizontal_detections(
            image, detections
        )

        # To preserve original, copy first
        flipped_img, flipped_dets = pf.transform.flip_horizontal_detections(
            image, detections.copy()
        )
        ```

    Note:
        - Left-right mirror transformation
        - Detections are modified IN-PLACE for performance
        - Use detections.copy() if you need to preserve the original
    """
    h, w = image.shape[:2]

    # Flip image using image.py function
    flipped_img = image_transforms.flip_horizontal(image)

    # Transform detections in-place
    for detection in detections:
        # Transform bbox
        if detection.bbox is not None:
            x1, y1, x2, y2 = detection.bbox
            detection.bbox = [w - x2, y1, w - x1, y2]

        # Transform keypoints
        if detection.keypoints is not None:
            new_keypoints = []
            for kp in detection.keypoints:
                new_kp = KeyPoint(
                    x=int(w - kp.x),
                    y=int(kp.y),
                    name=kp.name,
                    visibility=kp.visibility
                )
                new_keypoints.append(new_kp)
            detection.keypoints = new_keypoints

        # Transform segments
        if detection.segments is not None:
            if isinstance(detection.segments, np.ndarray):
                detection.segments[:, 0] = w - detection.segments[:, 0]
            elif isinstance(detection.segments, list):
                new_segments_list = []
                for segment in detection.segments:
                    if isinstance(segment, np.ndarray):
                        segment[:, 0] = w - segment[:, 0]
                        new_segments_list.append(segment)
                    elif isinstance(segment, list):
                        new_segment = [[w - x, y] for x, y in segment]
                        new_segments_list.append(new_segment)
                detection.segments = new_segments_list

        # Transform masks
        if detection.masks is not None:
            new_masks = []
            for mask in detection.masks:
                if isinstance(mask, np.ndarray):
                    flipped_mask = cv2.flip(mask.astype(np.uint8), 1).astype(bool)
                    new_masks.append(flipped_mask)
                else:
                    if isinstance(mask, list):
                        new_mask = [[w - x, y] for x, y in mask]
                        new_masks.append(new_mask)
                    else:
                        new_masks.append(mask)
            detection.masks = new_masks

        # Track metadata
        if track_metadata:
            if detection.metadata is None:
                detection.metadata = {}
            if 'transforms' not in detection.metadata:
                detection.metadata['transforms'] = []

            detection.metadata['transforms'].append({
                'type': 'flip_horizontal',
                'original_size': (h, w)
            })

    return flipped_img, detections


def flip_vertical_detections(
    image: np.ndarray,
    detections: Detections,
    track_metadata: bool = True
) -> Tuple[np.ndarray, Detections]:
    """
    Flip image vertically and update detection coordinates.

    Args:
        image: Input image (H, W, 3) BGR format
        detections: Detections to transform (modified in-place)
        track_metadata: If True, writes transform metadata to detection.metadata

    Returns:
        (flipped_image, flipped_detections) tuple

    Example:
        ```python
        import pixelflow as pf
        import cv2

        image = cv2.imread("image.jpg")
        detections = model.predict(image)

        # Flip vertically (modifies detections in-place)
        flipped_img, flipped_dets = pf.transform.flip_vertical_detections(
            image, detections
        )

        # To preserve original, copy first
        flipped_img, flipped_dets = pf.transform.flip_vertical_detections(
            image, detections.copy()
        )
        ```

    Note:
        - Top-bottom mirror transformation
        - Detections are modified IN-PLACE for performance
        - Use detections.copy() if you need to preserve the original
    """
    h, w = image.shape[:2]

    # Flip image using image.py function
    flipped_img = image_transforms.flip_vertical(image)

    # Transform detections in-place
    for detection in detections:
        # Transform bbox
        if detection.bbox is not None:
            x1, y1, x2, y2 = detection.bbox
            detection.bbox = [x1, h - y2, x2, h - y1]

        # Transform keypoints
        if detection.keypoints is not None:
            new_keypoints = []
            for kp in detection.keypoints:
                new_kp = KeyPoint(
                    x=int(kp.x),
                    y=int(h - kp.y),
                    name=kp.name,
                    visibility=kp.visibility
                )
                new_keypoints.append(new_kp)
            detection.keypoints = new_keypoints

        # Transform segments
        if detection.segments is not None:
            if isinstance(detection.segments, np.ndarray):
                detection.segments[:, 1] = h - detection.segments[:, 1]
            elif isinstance(detection.segments, list):
                new_segments_list = []
                for segment in detection.segments:
                    if isinstance(segment, np.ndarray):
                        segment[:, 1] = h - segment[:, 1]
                        new_segments_list.append(segment)
                    elif isinstance(segment, list):
                        new_segment = [[x, h - y] for x, y in segment]
                        new_segments_list.append(new_segment)
                detection.segments = new_segments_list

        # Transform masks
        if detection.masks is not None:
            new_masks = []
            for mask in detection.masks:
                if isinstance(mask, np.ndarray):
                    flipped_mask = cv2.flip(mask.astype(np.uint8), 0).astype(bool)
                    new_masks.append(flipped_mask)
                else:
                    if isinstance(mask, list):
                        new_mask = [[x, h - y] for x, y in mask]
                        new_masks.append(new_mask)
                    else:
                        new_masks.append(mask)
            detection.masks = new_masks

        # Track metadata
        if track_metadata:
            if detection.metadata is None:
                detection.metadata = {}
            if 'transforms' not in detection.metadata:
                detection.metadata['transforms'] = []

            detection.metadata['transforms'].append({
                'type': 'flip_vertical',
                'original_size': (h, w)
            })

    return flipped_img, detections


def crop_detections(
    image: np.ndarray,
    detections: Detections,
    bbox: List[float],
    track_metadata: bool = True
) -> Tuple[np.ndarray, Detections]:
    """
    Crop image to bounding box and update detection coordinates.

    Args:
        image: Input image (H, W, 3) BGR format
        detections: Detections to transform
        bbox: Crop region [x1, y1, x2, y2] in pixels
        track_metadata: If True, writes transform metadata to detection.metadata

    Returns:
        (cropped_image, cropped_detections) tuple

    Example:
        ```python
        import pixelflow as pf
        import cv2

        image = cv2.imread("image.jpg")
        detections = model.predict(image)

        # Crop to region
        cropped_img, cropped_dets = pf.transform.crop_detections(
            image, detections, bbox=[100, 50, 500, 400]
        )
        ```

    Note:
        - Coordinates are automatically clipped to image boundaries
        - Detections outside crop region are filtered out
        - Remaining detections have coordinates translated to new origin
    """
    # Crop image using image.py function
    cropped_img = image_transforms.crop(image, bbox)

    x1, y1, x2, y2 = bbox
    crop_w = x2 - x1
    crop_h = y2 - y1

    # Transform detections (filter and transform)
    cropped_detections = Detections()

    for detection in detections:
        # Check if detection intersects crop region
        if detection.bbox is not None:
            det_x1, det_y1, det_x2, det_y2 = detection.bbox

            # Skip if completely outside crop
            if det_x2 < x1 or det_x1 > x2 or det_y2 < y1 or det_y1 > y2:
                continue

        # Copy detection (only ones inside crop region)
        new_detection = detection.copy()

        # Transform bbox
        if new_detection.bbox is not None:
            det_x1, det_y1, det_x2, det_y2 = new_detection.bbox
            # Translate to new coordinate system
            new_detection.bbox = [
                det_x1 - x1,
                det_y1 - y1,
                det_x2 - x1,
                det_y2 - y1
            ]

        # Transform keypoints
        if new_detection.keypoints is not None:
            new_keypoints = []
            for kp in new_detection.keypoints:
                # Check if keypoint is within crop
                if x1 <= kp.x <= x2 and y1 <= kp.y <= y2:
                    new_kp = KeyPoint(
                        x=int(kp.x - x1),
                        y=int(kp.y - y1),
                        name=kp.name,
                        visibility=kp.visibility
                    )
                    new_keypoints.append(new_kp)
            new_detection.keypoints = new_keypoints if new_keypoints else None

        # Transform segments
        if new_detection.segments is not None:
            if isinstance(new_detection.segments, np.ndarray):
                new_detection.segments[:, 0] -= x1
                new_detection.segments[:, 1] -= y1
            elif isinstance(new_detection.segments, list):
                new_segments_list = []
                for segment in new_detection.segments:
                    if isinstance(segment, np.ndarray):
                        segment[:, 0] -= x1
                        segment[:, 1] -= y1
                        new_segments_list.append(segment)
                    elif isinstance(segment, list):
                        new_segment = [[x - x1, y - y1] for x, y in segment]
                        new_segments_list.append(new_segment)
                new_detection.segments = new_segments_list

        # Transform masks
        if new_detection.masks is not None:
            new_masks = []
            for mask in new_detection.masks:
                if isinstance(mask, np.ndarray):
                    # Crop mask to same region
                    h, w = mask.shape[:2]
                    crop_x1 = int(max(0, min(x1, w)))
                    crop_y1 = int(max(0, min(y1, h)))
                    crop_x2 = int(max(0, min(x2, w)))
                    crop_y2 = int(max(0, min(y2, h)))
                    cropped_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]
                    new_masks.append(cropped_mask)
                else:
                    if isinstance(mask, list):
                        new_mask = [[x - x1, y - y1] for x, y in mask]
                        new_masks.append(new_mask)
                    else:
                        new_masks.append(mask)
            new_detection.masks = new_masks

        # Track metadata
        if track_metadata:
            if new_detection.metadata is None:
                new_detection.metadata = {}
            if 'transforms' not in new_detection.metadata:
                new_detection.metadata['transforms'] = []

            new_detection.metadata['transforms'].append({
                'type': 'crop',
                'bbox': bbox,
                'original_size': (image.shape[0], image.shape[1])
            })

        cropped_detections.add_detection(new_detection)

    return cropped_img, cropped_detections


def crop_around_detections(
    image: np.ndarray,
    detections: Detections,
    padding: Union[float, Dict[str, float]] = 0.0
) -> List[np.ndarray]:
    """
    Crop image around each detection's bounding box.

    Args:
        image: Input image (H, W, 3) BGR format
        detections: Detections to crop around
        padding: Padding to add around bbox
            - Float: uniform padding as fraction of bbox shorter side
            - Dict: {'left': 0.2, 'right': 0.2, 'top': 0.1, 'bottom': 0.1}

    Returns:
        List of cropped images, one per detection (same order as detections)

    Example:
        ```python
        import pixelflow as pf
        import cv2

        image = cv2.imread("image.jpg")
        detections = model.predict(image)

        # Crop around each detection with 20% padding
        crops = pf.transform.crop_around_detections(image, detections, padding=0.2)

        # Process each crop
        for crop, detection in zip(crops, detections):
            cv2.imshow(f"{detection.class_name}", crop)
            cv2.waitKey(0)
        ```

    Note:
        - Detections themselves are NOT modified (coordinates stay in original space)
        - Crop regions are clipped to image boundaries (no extension)
        - If detection has no bbox, it is skipped
        - No metadata tracking (detections unchanged)
    """
    crops = []
    h, w = image.shape[:2]

    for detection in detections:
        if detection.bbox is None:
            continue

        x1, y1, x2, y2 = detection.bbox
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # Calculate padding
        if isinstance(padding, dict):
            pad_left = padding.get('left', 0) * bbox_w
            pad_right = padding.get('right', 0) * bbox_w
            pad_top = padding.get('top', 0) * bbox_h
            pad_bottom = padding.get('bottom', 0) * bbox_h
        else:
            # Uniform padding based on shorter side
            reference = min(bbox_w, bbox_h)
            pad_amount = padding * reference
            pad_left = pad_right = pad_top = pad_bottom = pad_amount

        # Apply padding
        crop_x1 = x1 - pad_left
        crop_y1 = y1 - pad_top
        crop_x2 = x2 + pad_right
        crop_y2 = y2 + pad_bottom

        # Clip to image boundaries (no extension)
        crop_bbox = [
            max(0, crop_x1),
            max(0, crop_y1),
            min(w, crop_x2),
            min(h, crop_y2)
        ]

        # Crop directly using image.py function
        crop = image_transforms.crop(image, crop_bbox)
        crops.append(crop)

    return crops


def rotate_to_align(
    image: np.ndarray,
    detections: Detections,
    point1_name: str,
    point2_name: str,
    target_angle: float = 0.0,
    detection_index: int = 0
) -> Tuple[np.ndarray, Detections]:
    """
    Rotate image so two keypoints form specified angle.

    Internally calls rotate_detections() with calculated angle to align two keypoints
    to a target orientation.

    Args:
        image: Input image (H, W, 3) BGR format
        detections: Detections containing keypoints (modified in-place)
        point1_name: First keypoint name (e.g., 'p0')
        point2_name: Second keypoint name (e.g., 'p9')
        target_angle: Target angle in degrees (0 = horizontal, 90 = vertical)
        detection_index: Which detection to use for alignment (default: first)

    Returns:
        (rotated_image, rotated_detections) with metadata tracking

    Raises:
        ValueError: If keypoints not found or detection index invalid

    Example:
        ```python
        import pixelflow as pf
        import cv2

        image = cv2.imread("image.jpg")
        detections = model.predict(image)

        # Align p0-p9 horizontally (modifies detections in-place)
        image, detections = pf.transform.rotate_to_align(
            image, detections, 'p0', 'p9', target_angle=0
        )

        # To preserve original, copy first
        image, detections = pf.transform.rotate_to_align(
            image, detections.copy(), 'p0', 'p9', target_angle=90
        )
        ```

    Note:
        - Keypoints must exist in specified detection
        - Rotation is performed around image center
        - Metadata is automatically tracked
        - Detections are modified IN-PLACE for performance
        - Use detections.copy() if you need to preserve the original
    """
    # Get keypoints
    if detection_index >= len(detections.detections):
        raise ValueError(f"Detection index {detection_index} out of range")

    detection = detections.detections[detection_index]

    if detection.keypoints is None:
        raise ValueError("Detection has no keypoints")

    # Find keypoints by name
    point1 = None
    point2 = None

    for kp in detection.keypoints:
        if kp.name == point1_name:
            point1 = (kp.x, kp.y)
        if kp.name == point2_name:
            point2 = (kp.x, kp.y)

    if point1 is None:
        raise ValueError(f"Keypoint '{point1_name}' not found")
    if point2 is None:
        raise ValueError(f"Keypoint '{point2_name}' not found")

    # Calculate current angle
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]

    if dx == 0 and dy == 0:
        raise ValueError("Keypoints are at the same location")

    current_angle = math.degrees(math.atan2(dy, dx))

    # Calculate rotation needed
    rotation_angle = target_angle - current_angle

    # Rotate image and detections (in-place)
    return rotate_detections(image, detections, rotation_angle)


def update_bbox_from_keypoints(
    detections: Detections,
    keypoint_names: Optional[List[str]] = None,
    track_metadata: bool = True
) -> Detections:
    """
    Update detection bboxes based on keypoints for all detections.

    Calculates bounding box from specified keypoints (or all keypoints if not specified)
    and updates each detection's bbox field in-place.

    Args:
        detections: Detections to update (modified in-place)
        keypoint_names: Which keypoints to include. If None, uses all keypoints.
        track_metadata: If True, writes transform metadata to detection.metadata

    Returns:
        Same detections object with updated bboxes

    Example:
        ```python
        import pixelflow as pf

        detections = model.predict(image)

        # Update bbox from specific keypoints (modifies in-place)
        detections = pf.transform.update_bbox_from_keypoints(
            detections, keypoint_names=['p0', 'p9', 'p21', 'p3']
        )

        # Update bbox from all keypoints
        detections = pf.transform.update_bbox_from_keypoints(detections)

        # To preserve original, copy first
        updated = pf.transform.update_bbox_from_keypoints(
            detections.copy(), keypoint_names=['p0', 'p9']
        )
        ```

    Note:
        - Operates on ALL detections in the container
        - Skips detections without keypoints
        - If keypoint_names specified but not found, detection is skipped
        - Detections are modified IN-PLACE for performance
        - Use detections.copy() if you need to preserve the original
    """
    for detection in detections:
        # Skip if no keypoints
        if detection.keypoints is None or len(detection.keypoints) == 0:
            continue

        # Filter keypoints
        if keypoint_names is not None:
            keypoints = [kp for kp in detection.keypoints if kp.name in keypoint_names]
            if not keypoints:
                # Skip this detection if none of the specified keypoints found
                continue
        else:
            keypoints = detection.keypoints

        # Calculate bbox from keypoints
        xs = [kp.x for kp in keypoints]
        ys = [kp.y for kp in keypoints]
        detection.bbox = [min(xs), min(ys), max(xs), max(ys)]

        # Track metadata
        if track_metadata:
            if detection.metadata is None:
                detection.metadata = {}
            if 'transforms' not in detection.metadata:
                detection.metadata['transforms'] = []
            detection.metadata['transforms'].append({
                'type': 'update_bbox_from_keypoints',
                'keypoint_names': keypoint_names
            })

    return detections


def add_padding(
    detections: Detections,
    padding: Union[float, Dict[str, float]],
    reference: str = 'shorter',
    track_metadata: bool = True
) -> Detections:
    """
    Add padding to detection bboxes for all detections.

    Args:
        detections: Detections to pad (modified in-place)
        padding:
            - Float: uniform padding as fraction of reference dimension
            - Dict: {'left': 0.2, 'right': 0.2, 'top': 0.1, 'bottom': 0.1}
        reference: Which dimension to base percentage on
            - 'shorter': Use shorter side (bbox width or height)
            - 'longer': Use longer side
            - 'width': Always use bbox width
            - 'height': Always use bbox height
        track_metadata: If True, writes transform metadata to detection.metadata

    Returns:
        Same detections object with padded bboxes

    Example:
        ```python
        import pixelflow as pf

        detections = model.predict(image)

        # 10% uniform padding based on shorter side (modifies in-place)
        padded = pf.transform.add_padding(detections, padding=0.1, reference='shorter')

        # Asymmetric padding
        padded = pf.transform.add_padding(
            detections,
            padding={'left': 0.3, 'right': 0.2, 'top': 0, 'bottom': 0}
        )

        # 20% padding based on width
        padded = pf.transform.add_padding(detections, padding=0.2, reference='width')

        # To preserve original, copy first
        padded = pf.transform.add_padding(detections.copy(), padding=0.1)
        ```

    Note:
        - Operates on ALL detections in the container
        - Padding can extend bbox beyond image boundaries
        - Detections are modified IN-PLACE for performance
        - Use detections.copy() if you need to preserve the original
    """
    for detection in detections:
        if detection.bbox is None:
            # No bbox, skip this detection
            continue

        x1, y1, x2, y2 = detection.bbox
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # Calculate reference dimension
        if reference == 'shorter':
            ref_dim = min(bbox_w, bbox_h)
        elif reference == 'longer':
            ref_dim = max(bbox_w, bbox_h)
        elif reference == 'width':
            ref_dim = bbox_w
        elif reference == 'height':
            ref_dim = bbox_h
        else:
            raise ValueError(f"Invalid reference: {reference}. Must be 'shorter', 'longer', 'width', or 'height'")

        # Calculate padding
        if isinstance(padding, dict):
            pad_left = padding.get('left', 0) * ref_dim
            pad_right = padding.get('right', 0) * ref_dim
            pad_top = padding.get('top', 0) * ref_dim
            pad_bottom = padding.get('bottom', 0) * ref_dim
        else:
            # Uniform padding
            pad_left = pad_right = pad_top = pad_bottom = padding * ref_dim

        # Apply padding (in-place)
        detection.bbox = [
            x1 - pad_left,
            y1 - pad_top,
            x2 + pad_right,
            y2 + pad_bottom
        ]

        # Track metadata
        if track_metadata:
            if detection.metadata is None:
                detection.metadata = {}
            if 'transforms' not in detection.metadata:
                detection.metadata['transforms'] = []
            detection.metadata['transforms'].append({
                'type': 'add_padding',
                'padding': padding,
                'reference': reference
            })

    return detections
