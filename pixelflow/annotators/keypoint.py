from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..detections import Detections

import cv2
import numpy as np
from .utils import _get_adaptive_params
from ..colors import _get_color_for_prediction


def keypoint(
    image: np.ndarray,
    detections: 'Detections',
    radius: Optional[int] = None,
    thickness: Optional[int] = None,
    colors: Optional[List[tuple]] = None,
    show_names: bool = False
) -> np.ndarray:
    """
    Draw keypoint markers on detected objects.

    Visualizes keypoint locations as colored circles on the image. Respects keypoint
    visibility flags and automatically adapts marker size based on image dimensions.
    Useful for pose estimation, landmark detection, and any application involving
    point annotations.

    Args:
        image (np.ndarray): Input image to draw keypoints on (BGR format).
                           Modified in-place with drawn keypoint markers.
        detections (Detections): Detections object containing keypoints.
                                Each detection may have a 'keypoints' attribute with
                                KeyPoint objects (x, y, name, visibility).
        radius (Optional[int]): Radius of keypoint circles in pixels.
                               If None, automatically scaled based on image size.
        thickness (Optional[int]): Thickness of circle outline in pixels.
                                  Use -1 for filled circles (default).
                                  If None, defaults to -1 (filled).
        colors (Optional[List[tuple]]): List of BGR color tuples for custom colors.
                                       Colors mapped to unique class_ids in order.
                                       If None, uses default ColorManager colors.
        show_names (bool): If True, draws keypoint names as text labels next to points.
                          Default is False.

    Returns:
        np.ndarray: Image with keypoints drawn. The input image is modified in-place.

    Raises:
        AttributeError: If keypoint objects lack required attributes (x, y).

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
        >>> # Draw keypoints with default settings
        >>> annotated = pf.annotate.keypoint(image, detections)
        >>>
        >>> # Draw with custom styling and keypoint names
        >>> annotated = pf.annotate.keypoint(
        ...     image, detections,
        ...     radius=5,
        ...     thickness=2,
        ...     colors=[(0, 255, 0)],  # Green
        ...     show_names=True
        ... )
        >>>
        >>> # Combine with skeleton for complete pose visualization
        >>> annotated = pf.annotate.keypoint(image, detections, radius=4)
        >>> annotated = pf.annotate.keypoint_skeleton(image, detections, thickness=2)

    Notes:
        - Only draws keypoints with visibility=True
        - Radius is automatically scaled with image size if not specified
        - Input image is modified in-place for memory efficiency
        - Keypoint names can be displayed for debugging or annotation purposes
        - Works with any detection format that includes keypoint data
        - Skips detections without keypoints gracefully

    See Also:
        keypoint_skeleton : Draw skeleton connections between keypoints
        anchors : Draw anchor points on bounding boxes
    """
    # Get adaptive parameters if not specified
    if radius is None:
        params = _get_adaptive_params(image)
        radius = max(3, params['thickness'] * 2)  # Scale radius with image size

    # Default to filled circles
    if thickness is None:
        thickness = -1

    # Draw keypoints for each detection
    for detection in detections:
        # Skip if no keypoints
        if detection.keypoints is None or len(detection.keypoints) == 0:
            continue

        color = _get_color_for_prediction(detection, colors)

        # Draw each keypoint
        for kp in detection.keypoints:
            # Skip invisible keypoints
            if not kp.visibility:
                continue

            x, y = int(kp.x), int(kp.y)

            # Draw keypoint circle
            cv2.circle(image, (x, y), radius, color, thickness)

            # Optionally draw keypoint name
            if show_names and kp.name:
                # Position text slightly above and to the right of the keypoint
                text_x = x + radius + 2
                text_y = y - radius - 2
                cv2.putText(
                    image,
                    kp.name,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA
                )

    return image
