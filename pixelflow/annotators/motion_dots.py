from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..detections import Detections

import numpy as np


def motion_dots(image: np.ndarray, detections: 'Detections', thickness=None):
    # TODO: Implement motion dots/breadcrumbs for object paths
    return image