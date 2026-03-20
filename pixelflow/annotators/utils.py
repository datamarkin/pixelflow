"""
Shared utilities for annotator modules.
"""

import numpy as np


# Adaptive sizing cache and configuration
_adaptive_cache = {}
ADAPTIVE_SCALE_MULTIPLIER = 1.0  # Users can adjust this globally


def _get_adaptive_params(image):
    """
    Get cached adaptive parameters based on image dimensions.
    Calculates once per unique resolution and caches for reuse.
    
    Args:
        image (np.ndarray): Input image to get dimensions from
    
    Returns:
        dict: Adaptive parameters for annotation sizing
    """
    shape_key = image.shape[:2]  # (height, width) as cache key
    
    if shape_key in _adaptive_cache:
        return _adaptive_cache[shape_key]
    
    # Calculate once per unique resolution
    height, width = shape_key
    # Base scale normalized around 1000x1000 images
    base_scale = np.sqrt(width * height) / 1000 * ADAPTIVE_SCALE_MULTIPLIER
    
    params = {
        'thickness': max(1, int(base_scale * 2)),
        'font_scale': max(0.3, base_scale * 0.5),
        'font_thickness': max(1, int(base_scale * 1.5)),
        'padding': max(2, int(base_scale * 5)),
        'margin': max(1, int(base_scale * 2)),
        'text_offset': max(10, int(base_scale * 20)),
        'blur_kernel': max(3, int(base_scale * 15)) | 1,  # ensure odd number
        'pixel_size': max(2, int(base_scale * 10)),
        'corner_radius': max(0, int(base_scale * 5)),
        'shadow_offset': max(1, int(base_scale * 2)),
    }
    
    _adaptive_cache[shape_key] = params
    return params