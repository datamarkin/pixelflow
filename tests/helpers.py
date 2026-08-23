"""
Shared builders for tests that need detections with specific geometry.

`conftest.py` holds fixtures sized for realistic scenes; these are for tests that
pick their own coordinates -- annotator tests asserting on pixels, tracker tests
driving a trajectory, transform tests probing edge shapes. Keeping the
construction here means a change to `from_arrays` (which recently gained `texts`
and `segments`, and made `class_ids` optional) is one edit rather than three.
"""

import numpy as np

import pixelflow as pf


def make_detections(boxes, scores=None, class_ids=None, tracker_ids=None, **kwargs):
    """Build a Detections from `boxes`, defaulting everything a test does not pin.

    Args:
        boxes: Sequence of [x1, y1, x2, y2], or anything reshapeable to (N, 4).
        scores: Per-box confidences. Defaults to 0.9 for every box.
        class_ids: Per-box class ids. Defaults to 0 for every box.
        tracker_ids: Per-box tracker ids, assigned after construction.
        **kwargs: Passed through to `pf.from_arrays` (texts, segments, labels).

    Returns:
        Detections: The built collection. Note that invalid geometry is dropped on
            the way in, so this can legitimately return fewer rows than `boxes`.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    count = len(boxes)
    result = pf.from_arrays(boxes,
                            scores=[0.9] * count if scores is None else scores,
                            class_ids=[0] * count if class_ids is None else class_ids,
                            **kwargs)
    if tracker_ids is not None:
        for detection, tracker_id in zip(result, tracker_ids):
            detection.tracker_id = tracker_id
    return result


def black_canvas(height=240, width=320):
    """A black image, so anything an annotator draws is unambiguous.

    conftest's `blank_image` is white, which is the right ground for eyeballing an
    annotation but the wrong one for asserting that a pixel changed.
    """
    return np.zeros((height, width, 3), dtype=np.uint8)
