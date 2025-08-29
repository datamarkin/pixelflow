import cv2
import numpy as np


def mask(frame: np.ndarray,
         results,
         opacity: float = 0.5,
         colors=None) -> np.ndarray:
    """
    Overlays masks on a video frame, supporting both binary masks and polygon formats.

    Args:
        frame (np.ndarray): The video frame (BGR format).
        results: List of results containing masks (e.g., from a model's output).
        opacity (float): Opacity level for blending masks with the frame (0.0 to 1.0).
        colors (list, optional): List of BGR color tuples to override default colors.
                               Colors are mapped to unique class_ids in order of appearance.
                               If None, uses default ColorManager colors.

    Returns:
        np.ndarray: The frame with masks overlaid.
        
    Examples:
        # Use default colors
        annotated = mask(image, results)
        
        # Override with custom colors
        custom_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        annotated = mask(image, results, colors=custom_colors)
    """
    from ..colors import get_color_for_prediction
    
    # Create a shared overlay array (same as frame) for all masks
    overlay = np.zeros_like(frame, dtype=np.uint8)

    # Process all masks
    for result in results:
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
                    overlay[mask_data] = color
                else:
                    # Convert to boolean if needed
                    binary_mask = mask_data.astype(bool)
                    if binary_mask.shape[:2] != frame.shape[:2]:
                        raise ValueError(f"Mask dimensions {binary_mask.shape[:2]} do not match frame dimensions {frame.shape[:2]}.")
                    overlay[binary_mask] = color
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
                        overlay[binary_mask] = color

    # Blend the overlay with the original frame in a single operation
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, dst=frame)

    return frame