import cv2
import numpy as np


def mask(frame: np.ndarray,
         results,
         opacity: float = 0.5,
         colors=None) -> np.ndarray:
    """
    Overlays binary masks on a video frame with improved performance.

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

    # Process all masks at once
    for result in results:
        for mask in result.masks:
            # Ensure mask dimensions match the frame
            if mask.shape[:2] != frame.shape[:2]:
                raise ValueError("Mask dimensions do not match frame dimensions.")

            # Apply the color only to the masked regions
            color = get_color_for_prediction(result, colors)
            overlay[mask] = color

    # Blend the overlay with the original frame in a single operation
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, dst=frame)

    return frame