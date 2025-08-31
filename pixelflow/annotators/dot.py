from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..detections import Detections

import numpy as np


def dot(image: np.ndarray, detections: 'Detections', thickness=None):
    # TODO: Implement center dot annotation for detected objects
    return image