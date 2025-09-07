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
    
    Args:
        image (np.ndarray): Input image to draw anchor points on (BGR format)
        detections (Detections): Detections object containing bounding boxes.
                                Each detection must have a 'bbox' attribute with (x1, y1, x2, y2) coordinates.
        strategy: Strategy for determining which anchor points to draw. Options:
                 - None: Draw all main anchor points (center, corners, edge centers)
                 - Single string (e.g., "center", "bottom_center")
                 - List of strings for multiple anchor points
                 Default: None (draws all main anchor points)
        radius (Optional[int]): Radius of anchor point circles in pixels.
                               If None, automatically determined based on image size.
        thickness (Optional[int]): Thickness of circle outline. Use -1 for filled circles.
                                  If None, circles are filled by default.
        colors (Optional[List[tuple]]): List of BGR color tuples to override default colors.
                                       Colors are mapped to unique class_ids in order of appearance.
                                       If None, uses default ColorManager colors.
    
    Returns:
        np.ndarray: Image with anchor points drawn. The input image is modified in-place.
        
    Examples:
        >>> import cv2
        >>> import pixelflow as pf
        >>> 
        >>> # Load image and get detections
        >>> image = cv2.imread("path/to/image.jpg")
        >>> detections = pf.results.from_ultralytics(model(image))
        >>> 
        >>> # Draw all main anchor points (default)
        >>> annotated = pf.annotators.anchors(image, detections)
        >>> 
        >>> # Draw bottom center points (useful for ground-based tracking)
        >>> annotated = pf.annotators.anchors(image, detections, strategy="bottom_center")
        >>> 
        >>> # Draw all four corners
        >>> corners = ["top_left", "top_right", "bottom_left", "bottom_right"]
        >>> annotated = pf.annotators.anchors(image, detections, strategy=corners)
        >>> 
        >>> 
        >>> # Custom styling
        >>> annotated = pf.annotators.anchors(
        ...     image, detections, 
        ...     strategy="center", 
        ...     radius=8, 
        ...     thickness=2,
        ...     colors=[(0, 255, 0)]  # Green circles
        ... )
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

