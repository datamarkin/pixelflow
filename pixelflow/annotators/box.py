from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..detections import Detections

import cv2
import numpy as np
from .utils import _get_adaptive_params
from ..colors import get_color_for_prediction


def box(
    image: np.ndarray, 
    detections: 'Detections', 
    thickness: Optional[int] = None, 
    colors: Optional[List[tuple]] = None
) -> np.ndarray:
    """
    Draw bounding boxes on detected objects.
    
    Simple and efficient bounding box visualization for object detection results.
    Automatically adapts line thickness based on image dimensions for optimal visibility.
    
    Args:
        image (np.ndarray): Input image to draw boxes on (BGR format)
        detections (Detections): Detections object containing bounding boxes.
                                Each detection must have a 'bbox' attribute with (x1, y1, x2, y2) coordinates.
        thickness (Optional[int]): Line thickness for bounding boxes in pixels.
                                  If None, automatically determined based on image size.
        colors (Optional[List[tuple]]): List of BGR color tuples to override default colors.
                                       Colors are mapped to unique class_ids in order of appearance.
                                       If None, uses default ColorManager colors.
    
    Returns:
        np.ndarray: Image with bounding boxes drawn. The input image is modified in-place.
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> 
        >>> # Load image and get model predictions
        >>> image = cv2.imread("path/to/image.jpg")
        >>> outputs = model.predict(image)  # Raw model outputs
        >>> detections = pf.results.from_ultralytics(outputs)  # Convert to PixelFlow format
        >>> 
        >>> # Draw boxes with default colors
        >>> annotated = pf.annotate.box(image, detections)
        >>> 
        >>> # Override with custom colors for specific classes
        >>> custom_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red
        >>> annotated = pf.annotate.box(image, detections, colors=custom_colors)
        >>> 
        >>> # Use custom thickness for smaller images
        >>> annotated = pf.annotate.box(image, detections, thickness=1)
    """
    # Get adaptive thickness if not specified
    if thickness is None:
        params = _get_adaptive_params(image)
        thickness = params['thickness']
    
    for result in detections:
        bbox = result.bbox
        x1, y1, x2, y2 = map(int, bbox)

        color = get_color_for_prediction(result, colors)

        cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=thickness)

    return image