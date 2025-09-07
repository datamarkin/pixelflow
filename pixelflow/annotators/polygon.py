from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..detections import Detections

import cv2
import numpy as np
from .utils import _get_adaptive_params
from ..colors import _get_color_for_prediction


def polygon(image: np.ndarray, detections: 'Detections', thickness=None, colors=None) -> np.ndarray:
    """
    Draw polygon outlines on detected objects.
    
    Args:
        image (np.ndarray): Input image to draw polygons on
        detections (Detections): Detections object containing segments
        thickness (int): Line thickness for polygon outlines. Default is 2.
        colors (list, optional): List of BGR color tuples to override default colors.
                                Colors are mapped to unique class_ids in order of appearance.
                                If None, uses default ColorManager colors.
    
    Returns:
        np.ndarray: Image with polygon outlines drawn
        
    Examples:
        # Use default colors
        annotated = polygon(image, detections)
        
        # Override with custom colors
        custom_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        annotated = polygon(image, detections, colors=custom_colors)
    """
    assert isinstance(image, np.ndarray), "Input image must be a NumPy array."
    
    # Get adaptive thickness if not specified
    if thickness is None:
        params = _get_adaptive_params(image)
        thickness = params['thickness']

    for result in detections:
        # Iterate over the segments in the result
        # Convert the points to a NumPy array and reshape for OpenCV
        polygon = np.array(result.segments, dtype=np.int32).reshape((-1, 1, 2))
        # Draw the polygon on the canvas
        color = _get_color_for_prediction(result, colors)
        cv2.polylines(image, [polygon], isClosed=True, color=color, thickness=thickness)

    return image