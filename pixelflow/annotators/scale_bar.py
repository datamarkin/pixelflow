from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..detections import Detections

import numpy as np


def scale_bar(image: np.ndarray, detections: 'Detections', thickness=None):
    # TODO: Implement scale bar for size reference
    return image