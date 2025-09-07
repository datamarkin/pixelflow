from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..detections import Detections

import cv2
import numpy as np
from ..colors import _get_color_for_prediction


def mask(frame: np.ndarray,
         detections: 'Detections',
         opacity: float = 0.5,
         colors=None) -> np.ndarray:
    """
    Overlays masks on a video frame, supporting both binary masks and polygon formats.

    Args:
        frame (np.ndarray): The video frame (BGR format).
        detections (Detections): Detections object containing masks.
        opacity (float): Opacity level for blending masks with the frame (0.0 to 1.0).
        colors (list, optional): List of BGR color tuples to override default colors.
                               Colors are mapped to unique class_ids in order of appearance.
                               If None, uses default ColorManager colors.

    Returns:
        np.ndarray: The frame with masks overlaid.
        
    Examples:
        # Use default colors
        annotated = mask(image, detections)
        
        # Override with custom colors
        custom_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        annotated = mask(image, detections, colors=custom_colors)
    """
    # Create a single overlay for all masks
    overlay = np.zeros_like(frame, dtype=np.uint8)
    has_mask = np.zeros(frame.shape[:2], dtype=bool)
    
    # Draw all masks to the overlay in a single pass
    for result in detections:
        if result.masks is None:
            continue
            
        color = _get_color_for_prediction(result, colors)
        
        for mask_data in result.masks:
            binary_mask = None
            
            if isinstance(mask_data, np.ndarray):
                # Binary mask format
                if mask_data.dtype == bool:
                    if mask_data.shape[:2] != frame.shape[:2]:
                        raise ValueError(f"Mask dimensions {mask_data.shape[:2]} do not match frame dimensions {frame.shape[:2]}.")
                    binary_mask = mask_data
                else:
                    binary_mask = mask_data.astype(bool)
                    if binary_mask.shape[:2] != frame.shape[:2]:
                        raise ValueError(f"Mask dimensions {binary_mask.shape[:2]} do not match frame dimensions {frame.shape[:2]}.")
            
            elif isinstance(mask_data, list) and len(mask_data) > 0:
                # Polygon format - convert to binary mask
                mask_img = np.zeros(frame.shape[:2], dtype=np.uint8)
                points = np.array(mask_data, dtype=np.int32)
                if len(points.shape) == 2 and points.shape[1] == 2:
                    points = points.reshape((-1, 1, 2))
                    cv2.fillPoly(mask_img, [points], 1)
                    binary_mask = mask_img.astype(bool)
            
            if binary_mask is not None:
                # Draw mask to overlay and track which pixels have masks
                overlay[binary_mask] = color
                has_mask |= binary_mask
    
    # Single blend operation for all masks
    if has_mask.any():  # Only blend if there are masks
        if opacity >= 1.0:
            # Direct copy for full opacity
            frame[has_mask] = overlay[has_mask]
        else:
            # Use OpenCV's optimized blending for partial opacity
            blended = cv2.addWeighted(frame, 1 - opacity, overlay, opacity, 0)
            frame[has_mask] = blended[has_mask]
    
    return frame