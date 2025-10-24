from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..detections import Detections

import cv2
import numpy as np
from .utils import _get_adaptive_params
from ..colors import _get_color_for_prediction


# COCO keypoint connections for human pose estimation
# Format: (start_keypoint_name, end_keypoint_name)
COCO_SKELETON = [
    # Head connections
    ('nose', 'left_eye'),
    ('nose', 'right_eye'),
    ('left_eye', 'left_ear'),
    ('right_eye', 'right_ear'),
    # Upper body
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    # Torso
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    # Lower body
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
]


def keypoint_skeleton(
    image: np.ndarray,
    detections: 'Detections',
    connections: Optional[List[Tuple[str, str]]] = None,
    thickness: Optional[int] = None,
    colors: Optional[List[tuple]] = None
) -> np.ndarray:
    """
    Draw skeleton connections between keypoints.

    Visualizes the structural relationships between keypoints by drawing lines
    connecting related points. Useful for pose estimation visualization, showing
    body structure, wing structure, or any connected landmark system.

    Args:
        image (np.ndarray): Input image to draw skeleton on (BGR format).
                           Modified in-place with drawn skeleton lines.
        detections (Detections): Detections object containing keypoints.
                                Each detection may have a 'keypoints' attribute with
                                KeyPoint objects (x, y, name, visibility).
        connections (Optional[List[Tuple[str, str]]]): List of keypoint name pairs
                                                       defining which points to connect.
                                                       Format: [("nose", "left_eye"), ...]
                                                       If None, uses COCO human pose skeleton.
        thickness (Optional[int]): Line thickness for skeleton connections in pixels.
                                  If None, automatically determined based on image size.
        colors (Optional[List[tuple]]): List of BGR color tuples for custom colors.
                                       Colors mapped to unique class_ids in order.
                                       If None, uses default ColorManager colors.

    Returns:
        np.ndarray: Image with skeleton drawn. The input image is modified in-place.

    Raises:
        AttributeError: If keypoint objects lack required attributes (x, y, name).

    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>>
        >>> # Load image and run pose estimation
        >>> image = cv2.imread("person.jpg")
        >>> model = YOLO("yolo11n-pose.pt")
        >>> outputs = model.predict(image)
        >>> detections = pf.detections.from_ultralytics(outputs)
        >>>
        >>> # Draw COCO skeleton (default for human pose)
        >>> annotated = pf.annotate.keypoint_skeleton(image, detections)
        >>>
        >>> # Define custom connections for wing keypoints
        >>> wing_skeleton = [
        ...     ("p0", "p1"), ("p1", "p2"), ("p2", "p3"),
        ...     ("p3", "p4"), ("p4", "p5"), ("p5", "p6"),
        ...     ("p6", "p7"), ("p7", "p8"), ("p8", "p9"),
        ...     ("p0", "p9")  # Close the loop
        ... ]
        >>> annotated = pf.annotate.keypoint_skeleton(
        ...     image, detections,
        ...     connections=wing_skeleton,
        ...     thickness=3
        ... )
        >>>
        >>> # Combine with keypoint markers for complete visualization
        >>> annotated = pf.annotate.keypoint_skeleton(image, detections, thickness=2)
        >>> annotated = pf.annotate.keypoint(image, detections, radius=4)
        >>>
        >>> # Custom styling with green skeleton
        >>> annotated = pf.annotate.keypoint_skeleton(
        ...     image, detections,
        ...     thickness=3,
        ...     colors=[(0, 255, 0)]
        ... )

    Notes:
        - Only draws connections where both keypoints are visible
        - Default connections follow COCO human pose skeleton format
        - Missing keypoint names are skipped gracefully
        - Input image is modified in-place for memory efficiency
        - Line thickness automatically adapts to image size if not specified
        - Works with any keypoint format by providing custom connections
        - Skips detections without keypoints gracefully

    See Also:
        keypoint : Draw individual keypoint markers
        COCO_SKELETON : Default connection pattern for human pose
    """
    # Get adaptive thickness if not specified
    if thickness is None:
        params = _get_adaptive_params(image)
        thickness = params['thickness']

    # Use COCO skeleton by default
    if connections is None:
        connections = COCO_SKELETON

    # Draw skeleton for each detection
    for detection in detections:
        # Skip if no keypoints
        if detection.keypoints is None or len(detection.keypoints) == 0:
            continue

        color = _get_color_for_prediction(detection, colors)

        # Build keypoint lookup dictionary for fast access
        kp_dict = {kp.name: kp for kp in detection.keypoints}

        # Draw each connection
        for start_name, end_name in connections:
            # Check if both keypoints exist
            if start_name not in kp_dict or end_name not in kp_dict:
                continue

            start_kp = kp_dict[start_name]
            end_kp = kp_dict[end_name]

            # Only draw if both keypoints are visible
            if not start_kp.visibility or not end_kp.visibility:
                continue

            # Get coordinates
            start_point = (int(start_kp.x), int(start_kp.y))
            end_point = (int(end_kp.x), int(end_kp.y))

            # Draw connection line
            cv2.line(image, start_point, end_point, color, thickness, cv2.LINE_AA)

    return image
