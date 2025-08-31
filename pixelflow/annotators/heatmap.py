from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..detections import Detections

import numpy as np


def heatmap(image: np.ndarray, detections: 'Detections', thickness=None):
    # TODO: Implement heatmap visualization for detection density
    return image