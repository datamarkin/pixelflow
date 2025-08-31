from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..results import Detections

import cv2
import numpy as np
from .utils import _get_adaptive_params


def oval(image: np.ndarray, detections: 'Detections', thickness=None, start_angle: int = -45, end_angle: int = 235, colors=None):
    """
    Draws elliptical footprints at the bottom of detected objects.
    
    Creates ground-plane footprint visualization by drawing partial ellipses
    at the bottom center of each bounding box. Useful for showing object
    presence on the ground plane or creating shadow-like effects.
    
    Args:
        image (np.ndarray): Input image to draw footprints on
        detections (Detections): Detections object containing bounding boxes
        thickness (int): Thickness of the ellipse lines. Default is 2.
        start_angle (int): Starting angle of the ellipse in degrees. 
                          Default is -45 (bottom-left).
        end_angle (int): Ending angle of the ellipse in degrees.
                        Default is 235 (bottom-right, creating bottom arc).
        colors (list, optional): List of BGR color tuples to override default colors.
                               Colors are mapped to unique class_ids in order of appearance.
                               If None, uses default ColorManager colors.
        
    Returns:
        np.ndarray: Image with elliptical footprints drawn at object bases
    
    Notes:
        - Ellipse width matches the bounding box width
        - Ellipse height is 25% of the width for natural proportions
        - Center point is at bottom-center of bounding box
        - Useful for ground plane visualization and spatial awareness
        
    Examples:
        # Use default colors
        annotated = oval(image, detections)
        
        # Override with custom colors
        custom_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        annotated = oval(image, detections, colors=custom_colors)
    """
    assert isinstance(image, np.ndarray), "Input image must be a NumPy array."
    
    # Get adaptive thickness if not specified
    if thickness is None:
        params = _get_adaptive_params(image)
        thickness = params['thickness']
    
    from ..colors import get_color_for_prediction
    
    for result in detections:
        box = result.bbox
        x1, y1, x2, y2 = map(int, box)
        
        # Get color for this detection
        color = get_color_for_prediction(result, colors)
        
        # Calculate ellipse parameters
        center = (int((x1 + x2) / 2), y2)  # Bottom center of bbox
        width = x2 - x1
        height = int(0.25 * width)  # Height is 25% of width for natural look
        
        # Draw the ellipse (partial arc from start_angle to end_angle)
        cv2.ellipse(
            image,
            center=center,
            axes=(int(width / 2), height),  # Semi-major and semi-minor axes
            angle=0.0,
            startAngle=start_angle,
            endAngle=end_angle,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA  # Anti-aliased for smooth curves
        )
    
    return image




# Alias for backward compatibility
footprint = oval