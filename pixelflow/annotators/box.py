import cv2
import numpy as np
from .utils import _get_adaptive_params


def box(image, results, thickness=None, colors=None):
    """
    Draw bounding boxes on detected objects.
    
    Args:
        image (np.ndarray): Input image to draw boxes on
        results: List of detection results containing bounding boxes
        thickness (int): Line thickness for bounding boxes. Default is 2.
        colors (list, optional): List of BGR color tuples to override default colors.
                                Colors are mapped to unique class_ids in order of appearance.
                                If None, uses default ColorManager colors.
    
    Returns:
        np.ndarray: Image with bounding boxes drawn
        
    Examples:
        # Use default colors
        annotated = box(image, results)
        
        # Override with custom colors
        custom_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        annotated = box(image, results, colors=custom_colors)
    """
    from ..colors import get_color_for_prediction
    from .. import draw
    
    # Get adaptive thickness if not specified
    if thickness is None:
        params = _get_adaptive_params(image)
        thickness = params['thickness']
    
    for result in results:
        box = result.bbox
        x1, y1, x2, y2 = map(int, box)

        color = get_color_for_prediction(result, colors)

        draw.rectangle(image, (x1, y1), (x2, y2), line_color=color, thickness=thickness)

    return image