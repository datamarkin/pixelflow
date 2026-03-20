from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..detections import Detections

import cv2
import time
import numpy as np
from .utils import _get_adaptive_params


def fps_counter(image: np.ndarray, detections: 'Detections', thickness=None):
    # Get current time
    current_time = time.time()

    # Calculate FPS
    if not hasattr(fps_counter, 'prev_time'):
        fps_counter.prev_time = current_time
        fps_counter.fps = 0
    else:
        # Calculate time difference
        time_diff = current_time - fps_counter.prev_time
        if time_diff > 0:
            fps_counter.fps = 1 / time_diff
        fps_counter.prev_time = current_time

    # Get adaptive parameters if thickness not specified
    if thickness is None:
        params = _get_adaptive_params(image)
        font_scale = params['font_scale'] * 1.5  # FPS counter slightly larger
        thickness = params['font_thickness']
    else:
        font_scale = 1
    
    # Compute adaptive position based on image dimensions
    params = _get_adaptive_params(image)
    padding = params['padding']
    text_size = cv2.getTextSize(f"FPS: 000", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    position = (padding, padding + text_size[1])

    # Draw FPS text in green (RGB)
    fps_color = (0, 255, 0)
    fps_text = f"FPS: {int(fps_counter.fps)}"
    cv2.putText(image, fps_text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, fps_color, thickness)

    return image