from typing import List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..detections import Detections

import cv2
import numpy as np
from .utils import _get_adaptive_params
from ..colors import _get_color_for_prediction
from ..strategies import (
    get_anchor_position, 
    STRATEGY_CENTER,
    STRATEGY_BOTTOM_CENTER,
    STRATEGY_TOP_LEFT,
    STRATEGY_TOP_RIGHT,
    STRATEGY_BOTTOM_LEFT,
    STRATEGY_BOTTOM_RIGHT,
    STRATEGY_TOP_CENTER,
    STRATEGY_LEFT_CENTER,
    STRATEGY_RIGHT_CENTER
)


def anchors(
    image: np.ndarray, 
    detections: 'Detections',
    strategy: Union[str, List[str]] = None,
    radius: Optional[int] = None,
    thickness: Optional[int] = None,
    colors: Optional[List[tuple]] = None
) -> np.ndarray:
    """
    Draw anchor points on detected objects based on trigger strategies.
    
    Visualizes the anchor points used by the trigger strategy system to determine
    if detections are within zones or crossing lines. Each anchor point is drawn
    as a small filled circle at the calculated position on the bounding box.
    Supports drawing individual anchor points or multiple points simultaneously.
    
    Args:
        image (np.ndarray): Input image to draw anchor points on in BGR format.
                           Image is modified in-place.
        detections (Detections): Detection results containing bounding boxes.
                                Each detection must have a 'bbox' attribute with (x1, y1, x2, y2) coordinates.
        strategy (Union[str, List[str]], optional): Strategy for determining anchor points to draw.
                                                   Single string: "center", "bottom_center", "top_left", etc.
                                                   List of strings: Multiple anchor points.
                                                   None draws all 9 main anchor points (center, corners, edge centers).
        radius (Optional[int]): Radius of anchor point circles in pixels.
                               If None, automatically scaled based on image size (minimum 2 pixels).
        thickness (Optional[int]): Thickness of circle outline in pixels.
                                  Use -1 for filled circles.
                                  If None, defaults to -1 (filled).
        colors (Optional[List[tuple]]): List of BGR color tuples for custom colors.
                                       Colors mapped to unique class_ids in order of appearance.
                                       If None, uses default ColorManager colors.
    
    Returns:
        np.ndarray: Image with anchor points drawn. The input image is modified in-place.
        
    Raises:
        AttributeError: If detections don't have required 'bbox' attribute.
        ValueError: If invalid strategy string is provided.
        
    Examples:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> 
        >>> # Load image and get detections
        >>> image = cv2.imread("path/to/image.jpg")
        >>> model = YOLO("yolo11n.pt")
        >>> outputs = model.predict(image)
        >>> detections = pf.results.from_ultralytics(outputs)
        >>> 
        >>> # Draw all main anchor points (default)
        >>> annotated = pf.annotators.anchors(image, detections)
        >>> 
        >>> # Draw bottom center points (useful for ground-based tracking)
        >>> annotated = pf.annotators.anchors(image, detections, strategy="bottom_center")
        >>> 
        >>> # Draw multiple specific anchor points
        >>> corners = ["top_left", "top_right", "bottom_left", "bottom_right"]
        >>> annotated = pf.annotators.anchors(image, detections, strategy=corners)
        >>> 
        >>> # Custom styling with larger green circles
        >>> annotated = pf.annotators.anchors(
        ...     image, detections, 
        ...     strategy="center", 
        ...     radius=8, 
        ...     thickness=2,
        ...     colors=[(0, 255, 0)]
        ... )
    
    Notes:
        - Anchor points are calculated using the get_anchor_position function from strategies module
        - Invalid anchor strategies are skipped gracefully to prevent errors
        - Radius is automatically scaled with image size if not specified
        - All 9 anchor points include: center, 4 corners, and 4 edge centers
        - Colors are applied consistently across detections with the same class_id
        
    See Also:
        get_anchor_position : Calculate anchor coordinates for bounding boxes
        box : Draw bounding boxes around detections
    """
    # Get adaptive parameters if not specified
    if radius is None:
        params = _get_adaptive_params(image)
        radius = max(2, params['thickness'] * 2)  # Scale radius with image size
    
    # Default to filled circles
    if thickness is None:
        thickness = -1
    
    # Convert strategy to list of strategies for uniform processing
    strategies_to_draw = []
    
    if strategy is None:
        # Default: draw all main anchor points
        strategies_to_draw = [
            STRATEGY_CENTER,
            STRATEGY_BOTTOM_CENTER,
            STRATEGY_TOP_LEFT,
            STRATEGY_TOP_RIGHT,
            STRATEGY_BOTTOM_LEFT,
            STRATEGY_BOTTOM_RIGHT,
            STRATEGY_TOP_CENTER,
            STRATEGY_LEFT_CENTER,
            STRATEGY_RIGHT_CENTER
        ]
    elif isinstance(strategy, (list, tuple)):
        # Use list of strategies directly
        strategies_to_draw = strategy
    else:
        # Single strategy
        strategies_to_draw = [strategy]
    
    # Draw anchor points for each detection
    for result in detections:
        bbox = result.bbox
        color = _get_color_for_prediction(result, colors)
        
        # Draw each anchor point
        for anchor_strategy in strategies_to_draw:
            # Get anchor position
            try:
                x, y = get_anchor_position(bbox, anchor_strategy)
                x, y = int(x), int(y)
                
                # Draw circle at anchor point
                cv2.circle(image, (x, y), radius, color, thickness)
                
            except Exception:
                # Skip invalid anchor strategies gracefully
                continue
    
    return image

