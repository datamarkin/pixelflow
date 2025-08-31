from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..detections import Detections

import numpy as np


def motion_trails(image: np.ndarray, detections: 'Detections', thickness=None):
    # TODO: Implement motion trail visualization for tracked objects
    return image