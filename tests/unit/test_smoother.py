"""
Unit tests for pixelflow.smoother.

`smooth` divides its work in two: a tracker seen in the current frame is
smoothed from its neighbours, one missing from it is interpolated between them.
Most of what can go wrong here is the two halves disagreeing about which case a
tracker falls into, so that boundary is what these tests pin.
"""

import warnings

import numpy as np
import pytest

import pixelflow as pf


def blank():
    return np.zeros((240, 320, 3), dtype=np.uint8)


def detections(boxes, tracker_ids=None):
    """Detections at `boxes`, optionally carrying tracker ids."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    result = pf.from_arrays(boxes, scores=[0.9] * len(boxes), class_ids=[0] * len(boxes))
    if tracker_ids is not None:
        for detection, tracker_id in zip(result, tracker_ids):
            detection.tracker_id = tracker_id
    return result


def fill(buffer, frames):
    """Push `frames` through `buffer`, yielding (raw, smoothed) once it is full."""
    out = []
    for detections_in in frames:
        raw, _ = buffer.update(detections_in, blank())
        if buffer.get_temporal_context() is not None:
            out.append((raw, pf.smooth(buffer)))
    return out


class TestGapHandling:
    """A tracker missing from the current frame is the interesting case."""

    def test_gap_frame_emits_each_tracker_once(self):
        """Regression: the gap was both smoothed and interpolated, emitting it twice.

        A tracker absent from the current frame but present either side belongs to
        the interpolation path alone. When the smoothing path also claimed it, one
        object came back as two rows sharing a tracker_id.
        """
        buffer = pf.Buffer(frames=5)
        frames = []
        for i in range(12):
            x = 100 + 10 * i
            frames.append(detections([[x, 50, x + 40, 90]], [1]) if i != 5
                          else detections([]))

        for raw, smoothed in fill(buffer, frames):
            ids = [d.tracker_id for d in smoothed]
            assert len(ids) == len(set(ids)), f"duplicate tracker_id in {ids}"

    def test_gap_is_interpolated(self):
        """The point of the interpolation path: a one-frame dropout is filled."""
        buffer = pf.Buffer(frames=5)
        frames = []
        for i in range(12):
            x = 100 + 10 * i
            frames.append(detections([[x, 50, x + 40, 90]], [1]) if i != 5
                          else detections([]))

        recovered = [(len(raw), len(smoothed)) for raw, smoothed in fill(buffer, frames)]
        assert (0, 1) in recovered, "the dropped frame was not interpolated"

    def test_tracker_ids_stay_unique_across_a_sequence(self):
        """The invariant, stated plainly: one id, one row, per frame."""
        buffer = pf.Buffer(frames=5)
        frames = []
        for i in range(20):
            boxes, ids = [], []
            if i != 7:
                boxes.append([100 + 10 * i, 50, 140 + 10 * i, 90]); ids.append(1)
            if i not in (11, 12):
                boxes.append([300 - 8 * i, 150, 340 - 8 * i, 190]); ids.append(2)
            frames.append(detections(boxes, ids))

        for _, smoothed in fill(buffer, frames):
            ids = [d.tracker_id for d in smoothed]
            assert len(ids) == len(set(ids)), f"duplicate tracker_id in {ids}"


class TestUntrackedDetections:
    """Smoothing needs tracking; saying so beats returning nothing."""

    def test_warns(self):
        """Regression: untracked detections used to disappear without a word."""
        buffer = pf.Buffer(frames=5)
        for i in range(7):
            buffer.update(detections([[100 + 5 * i, 50, 140 + 5 * i, 90]]), blank())

        with pytest.warns(UserWarning, match="tracker_id"):
            pf.smooth(buffer)

    def test_returns_them_unsmoothed(self):
        """Nothing can be smoothed, so the frame comes back as it went in."""
        buffer = pf.Buffer(frames=5)
        for i in range(7):
            buffer.update(detections([[100 + 5 * i, 50, 140 + 5 * i, 90]]), blank())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert len(pf.smooth(buffer)) == 1

    def test_tracked_detections_do_not_warn(self):
        """The warning fires on the real mistake only."""
        buffer = pf.Buffer(frames=5)
        for i in range(7):
            buffer.update(detections([[100 + 5 * i, 50, 140 + 5 * i, 90]], [1]), blank())

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pf.smooth(buffer)


class TestSmoothing:
    """What the module is for."""

    def test_reduces_jitter(self):
        """A tracked box wobbling around a straight line comes back closer to it."""
        buffer = pf.Buffer(frames=5)
        true_x = [100 + 10 * i for i in range(12)]
        jitter = [0, 6, -6, 5, -5, 6, -6, 5, -5, 6, -6, 0]
        frames = [detections([[x + j, 50, x + j + 40, 90]], [1])
                  for x, j in zip(true_x, jitter)]

        results = fill(buffer, frames)
        ideal = true_x[2:2 + len(results)]
        raw_error = np.mean([abs(raw[0].bbox[0] - x) for (raw, _), x in zip(results, ideal)])
        smoothed_error = np.mean([abs(sm[0].bbox[0] - x) for (_, sm), x in zip(results, ideal)])
        assert smoothed_error < raw_error

    def test_partial_buffer_returns_raw_results(self):
        """Before the window fills there is no temporal context to smooth with."""
        buffer = pf.Buffer(frames=5)
        buffer.update(detections([[0, 0, 10, 10]], [1]), blank())
        assert len(pf.smooth(buffer)) == 0

    def test_rejects_out_of_range_decay(self):
        """The weight decay is a documented range, enforced rather than assumed."""
        buffer = pf.Buffer(frames=5)
        with pytest.raises(ValueError, match="temporal_weight_decay"):
            pf.smooth(buffer, temporal_weight_decay=2.0)
