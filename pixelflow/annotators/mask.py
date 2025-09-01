from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..detections import Detections

import cv2
import numpy as np
from ..colors import get_color_for_prediction


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
    # Work with a copy to avoid modifying the original frame
    result_frame = frame.copy()

    # Process all masks
    for result in detections:
        if result.masks is None:
            continue
            
        color = get_color_for_prediction(result, colors)
        
        for mask_data in result.masks:
            if isinstance(mask_data, np.ndarray):
                # Binary mask format
                if mask_data.dtype == bool:
                    # Direct boolean mask
                    if mask_data.shape[:2] != frame.shape[:2]:
                        raise ValueError(f"Mask dimensions {mask_data.shape[:2]} do not match frame dimensions {frame.shape[:2]}.")
                    binary_mask = mask_data
                else:
                    # Convert to boolean if needed
                    binary_mask = mask_data.astype(bool)
                    if binary_mask.shape[:2] != frame.shape[:2]:
                        raise ValueError(f"Mask dimensions {binary_mask.shape[:2]} do not match frame dimensions {frame.shape[:2]}.")
                
                # Apply color only to masked regions with opacity blending
                result_frame[binary_mask] = (
                    opacity * np.array(color) + 
                    (1 - opacity) * result_frame[binary_mask]
                ).astype(np.uint8)
                
            elif isinstance(mask_data, list):
                # Polygon format - convert to binary mask
                if len(mask_data) > 0:
                    # Create a binary mask from polygon points
                    mask_img = np.zeros(frame.shape[:2], dtype=np.uint8)
                    points = np.array(mask_data, dtype=np.int32)
                    if len(points.shape) == 2 and points.shape[1] == 2:
                        # Reshape for cv2.fillPoly which expects [num_polygons, num_points, 2]
                        points = points.reshape((-1, 1, 2))
                        cv2.fillPoly(mask_img, [points], 1)
                        binary_mask = mask_img.astype(bool)
                        
                        # Apply color only to masked regions with opacity blending
                        result_frame[binary_mask] = (
                            opacity * np.array(color) + 
                            (1 - opacity) * result_frame[binary_mask]
                        ).astype(np.uint8)

    return result_frame