"""
Automated Inverse Transformations.

Automatically reverses detection transforms by reading transform history from metadata.
Supports chained transforms with LIFO (last-in-first-out) reversal order.
"""

import cv2
import numpy as np
import math
from typing import Optional
from ..detections.detections import Detections, KeyPoint

__all__ = ['inverse_transforms']


def inverse_transforms(
    detections: Detections,
    clear_history: bool = True
) -> Detections:
    """
    Automatically reverse all transforms to map detections to original image space.

    Reads transform history from detection metadata and applies inverse operations
    in reverse order (LIFO) to undo all transformations. Essential for production
    pipelines where transforms are applied before inference and results must be
    mapped back to original image coordinates.

    Args:
        detections: Detections with transform history in metadata['transforms']
        clear_history: If True, clears transform history after inverse (default: True)

    Returns:
        New Detections object with coordinates in original image space

    Example:
        ```python
        import pixelflow as pf
        import cv2

        # Load original image
        original_img = cv2.imread("image.jpg")
        detections = model.predict(original_img)

        # Apply transform chain
        img, dets = pf.transform.rotate_detections(original_img, detections, 45)
        img, dets = pf.transform.crop_detections(img, dets, [100, 50, 500, 400])
        img, dets = pf.transform.flip_horizontal_detections(img, dets)

        # Run inference on transformed image
        results = model.predict(img)

        # Automatically undo ALL transforms
        original_coords = pf.transform.inverse_transforms(results)

        # Now coordinates match original image
        annotated = pf.annotate.box(original_img, original_coords)
        ```

    Note:
        - Transforms are reversed in LIFO order (last applied, first undone)
        - Skips non-spatial transforms (e.g., update_bbox_from_keypoints)
        - Uses detections.copy() for immutable operation
        - Modifies coordinates in-place for performance
        - Clear history to prevent double-inversion if applied twice
    """
    # Create deep copy to avoid modifying original
    inverse_detections = detections.copy()

    # Process each detection
    for detection in inverse_detections:
        # Check if detection has transform history
        if detection.metadata is None or 'transforms' not in detection.metadata:
            continue

        transforms = detection.metadata['transforms']
        if not transforms:
            continue

        # Reverse transforms in LIFO order (last applied, first undone)
        for transform in reversed(transforms):
            transform_type = transform.get('type')

            if transform_type == 'rotate':
                _inverse_rotate(detection, transform)

            elif transform_type == 'flip_horizontal':
                _inverse_flip_horizontal(detection, transform)

            elif transform_type == 'flip_vertical':
                _inverse_flip_vertical(detection, transform)

            elif transform_type == 'crop':
                _inverse_crop(detection, transform)

            elif transform_type == 'add_padding':
                _inverse_add_padding(detection, transform)

            elif transform_type == 'update_bbox_from_keypoints':
                # Skip - not a spatial transform, no inverse needed
                pass

        # Clear transform history if requested
        if clear_history:
            detection.metadata['transforms'] = []

    return inverse_detections


def _inverse_rotate(detection, metadata):
    """Inverse rotation - rotate by negative angle."""
    angle = metadata['angle']
    center = metadata['center']
    original_size = metadata['original_size']
    h, w = original_size

    # Calculate inverse rotation (negative angle)
    inverse_angle = -angle
    angle_rad = math.radians(inverse_angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Get rotation matrix (same as forward transform setup)
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

    # Current image dimensions (after rotation)
    # Calculate what they would be
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust rotation matrix for canvas expansion (inverse)
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # Transform bbox
    if detection.bbox is not None:
        x1, y1, x2, y2 = detection.bbox
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        rotated_corners = []

        for x, y in corners:
            # Reverse the forward rotation offsets first
            x_centered = x - rotation_matrix[0, 2]
            y_centered = y - rotation_matrix[1, 2]

            # Apply inverse rotation
            x_rot = (x_centered - center[0]) * cos_a - (y_centered - center[1]) * sin_a + center[0]
            y_rot = (x_centered - center[0]) * sin_a + (y_centered - center[1]) * cos_a + center[1]

            rotated_corners.append((x_rot, y_rot))

        xs = [pt[0] for pt in rotated_corners]
        ys = [pt[1] for pt in rotated_corners]
        detection.bbox = [min(xs), min(ys), max(xs), max(ys)]

    # Transform keypoints
    if detection.keypoints is not None:
        new_keypoints = []
        for kp in detection.keypoints:
            x, y = kp.x, kp.y

            # Reverse forward offsets
            x_centered = x - rotation_matrix[0, 2]
            y_centered = y - rotation_matrix[1, 2]

            # Apply inverse rotation
            x_rot = (x_centered - center[0]) * cos_a - (y_centered - center[1]) * sin_a + center[0]
            y_rot = (x_centered - center[0]) * sin_a + (y_centered - center[1]) * cos_a + center[1]

            new_kp = KeyPoint(
                x=int(x_rot),
                y=int(y_rot),
                name=kp.name,
                visibility=kp.visibility
            )
            new_keypoints.append(new_kp)
        detection.keypoints = new_keypoints

    # Transform segments
    if detection.segments is not None:
        if isinstance(detection.segments, np.ndarray):
            for i in range(len(detection.segments)):
                x, y = detection.segments[i]
                x_centered = x - rotation_matrix[0, 2]
                y_centered = y - rotation_matrix[1, 2]
                x_rot = (x_centered - center[0]) * cos_a - (y_centered - center[1]) * sin_a + center[0]
                y_rot = (x_centered - center[0]) * sin_a + (y_centered - center[1]) * cos_a + center[1]
                detection.segments[i] = [x_rot, y_rot]
        elif isinstance(detection.segments, list):
            new_segments_list = []
            for segment in detection.segments:
                if isinstance(segment, np.ndarray):
                    for i in range(len(segment)):
                        x, y = segment[i]
                        x_centered = x - rotation_matrix[0, 2]
                        y_centered = y - rotation_matrix[1, 2]
                        x_rot = (x_centered - center[0]) * cos_a - (y_centered - center[1]) * sin_a + center[0]
                        y_rot = (x_centered - center[0]) * sin_a + (y_centered - center[1]) * cos_a + center[1]
                        segment[i] = [x_rot, y_rot]
                    new_segments_list.append(segment)
                elif isinstance(segment, list):
                    new_segment = []
                    for x, y in segment:
                        x_centered = x - rotation_matrix[0, 2]
                        y_centered = y - rotation_matrix[1, 2]
                        x_rot = (x_centered - center[0]) * cos_a - (y_centered - center[1]) * sin_a + center[0]
                        y_rot = (x_centered - center[0]) * sin_a + (y_centered - center[1]) * cos_a + center[1]
                        new_segment.append([x_rot, y_rot])
                    new_segments_list.append(new_segment)
            detection.segments = new_segments_list

    # Transform masks (rotate back)
    if detection.masks is not None:
        new_masks = []
        for mask in detection.masks:
            if isinstance(mask, np.ndarray):
                # Calculate original dimensions for mask rotation
                rotated_mask = cv2.warpAffine(
                    mask.astype(np.uint8),
                    cv2.getRotationMatrix2D(center, -angle, 1.0),
                    (w, h),
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                ).astype(bool)
                new_masks.append(rotated_mask)
            else:
                # Polygon mask
                if isinstance(mask, list):
                    new_mask = []
                    for x, y in mask:
                        x_centered = x - rotation_matrix[0, 2]
                        y_centered = y - rotation_matrix[1, 2]
                        x_rot = (x_centered - center[0]) * cos_a - (y_centered - center[1]) * sin_a + center[0]
                        y_rot = (x_centered - center[0]) * sin_a + (y_centered - center[1]) * cos_a + center[1]
                        new_mask.append([x_rot, y_rot])
                    new_masks.append(new_mask)
                else:
                    new_masks.append(mask)
        detection.masks = new_masks


def _inverse_flip_horizontal(detection, metadata):
    """Inverse horizontal flip - flip is self-inverse."""
    original_size = metadata['original_size']
    h, w = original_size

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


def _inverse_flip_vertical(detection, metadata):
    """Inverse vertical flip - flip is self-inverse."""
    original_size = metadata['original_size']
    h, w = original_size

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


def _inverse_crop(detection, metadata):
    """Inverse crop - add crop offset back to coordinates."""
    crop_bbox = metadata['bbox']
    x1_offset, y1_offset = crop_bbox[0], crop_bbox[1]

    # Transform bbox (add crop offset)
    if detection.bbox is not None:
        x1, y1, x2, y2 = detection.bbox
        detection.bbox = [
            x1 + x1_offset,
            y1 + y1_offset,
            x2 + x1_offset,
            y2 + y1_offset
        ]

    # Transform keypoints
    if detection.keypoints is not None:
        new_keypoints = []
        for kp in detection.keypoints:
            new_kp = KeyPoint(
                x=int(kp.x + x1_offset),
                y=int(kp.y + y1_offset),
                name=kp.name,
                visibility=kp.visibility
            )
            new_keypoints.append(new_kp)
        detection.keypoints = new_keypoints

    # Transform segments
    if detection.segments is not None:
        if isinstance(detection.segments, np.ndarray):
            detection.segments[:, 0] += x1_offset
            detection.segments[:, 1] += y1_offset
        elif isinstance(detection.segments, list):
            new_segments_list = []
            for segment in detection.segments:
                if isinstance(segment, np.ndarray):
                    segment[:, 0] += x1_offset
                    segment[:, 1] += y1_offset
                    new_segments_list.append(segment)
                elif isinstance(segment, list):
                    new_segment = [[x + x1_offset, y + y1_offset] for x, y in segment]
                    new_segments_list.append(new_segment)
            detection.segments = new_segments_list

    # Masks remain in cropped space (can't expand back without original image size)


def _inverse_add_padding(detection, metadata):
    """Inverse add_padding - subtract padding from bbox."""
    padding = metadata['padding']
    reference = metadata['reference']

    if detection.bbox is None:
        return

    x1, y1, x2, y2 = detection.bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    # Calculate reference dimension (same logic as add_padding)
    if reference == 'shorter':
        ref_dim = min(bbox_w, bbox_h)
    elif reference == 'longer':
        ref_dim = max(bbox_w, bbox_h)
    elif reference == 'width':
        ref_dim = bbox_w
    elif reference == 'height':
        ref_dim = bbox_h
    else:
        return  # Unknown reference, skip

    # Calculate padding values
    if isinstance(padding, dict):
        pad_left = padding.get('left', 0) * ref_dim
        pad_right = padding.get('right', 0) * ref_dim
        pad_top = padding.get('top', 0) * ref_dim
        pad_bottom = padding.get('bottom', 0) * ref_dim
    else:
        # Uniform padding
        pad_left = pad_right = pad_top = pad_bottom = padding * ref_dim

    # Subtract padding (inverse of add_padding)
    detection.bbox = [
        x1 + pad_left,
        y1 + pad_top,
        x2 - pad_right,
        y2 - pad_bottom
    ]
