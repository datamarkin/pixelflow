from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..detections import Detections

import numpy as np


def keypoint(image: np.ndarray, detections: 'Detections', thickness=None):
    # TODO: Implement keypoint visualization (e.g., pose estimation)
    return image