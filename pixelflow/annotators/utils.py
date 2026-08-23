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
        'thickness': max(1, int(base_scale * 3)),
        'font_scale': max(0.3, base_scale * 0.6),
        'font_thickness': max(1, int(base_scale * 2.5)),
        'padding': max(2, int(base_scale * 8)),
        'margin': max(1, int(base_scale * 3)),
        'text_offset': max(10, int(base_scale * 25)),
        'blur_kernel': max(3, int(base_scale * 25)) | 1,  # ensure odd number
        'pixel_size': max(2, int(base_scale * 10)),
        'corner_radius': max(0, int(base_scale * 8)),
        'shadow_offset': max(1, int(base_scale * 3)),
    }
    
    _adaptive_cache[shape_key] = params
    return params


def _meets_confidence(keypoint, min_confidence):
    """Whether `keypoint` clears the caller's threshold and should be drawn.

    A keypoint carrying no score cannot fail a threshold, so it draws. Keeping the
    comparison in one place stops the renderers from drifting on the boundary the way
    the old `visibility` flag drifted between `> 0` and `> 0.5` across converters.
    """
    return keypoint.confidence is None or keypoint.confidence >= min_confidence


def _caption_with_score(caption, confidence):
    """Join a caption and its score into one label, dropping whichever is absent.

    Joining the parts that exist rather than formatting both keeps a missing caption
    from rendering as the literal "None: 0.87", and a detection carrying only a score
    - anything from `from_sam` - still gets a label. Every annotator that writes one
    formats it here, so the score precision and the separator cannot drift apart
    between them.
    """
    score = f"{confidence:.2f}" if confidence is not None else ''
    return ': '.join(part for part in (caption, score) if part)
