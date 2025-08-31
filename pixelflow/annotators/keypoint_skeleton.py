from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..detections import Detections

import numpy as np


def keypoint_skeleton(image: np.ndarray, detections: 'Detections', thickness=None):
    # TODO: Implement skeleton connections between keypoints
    return image