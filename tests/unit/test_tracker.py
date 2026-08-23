"""
Unit and scenario tests for pixelflow.tracker.

The unit tests pin the three pieces the tracker is built from -- the Kalman
filter, the association maths, and the track state machine. The scenario tests
matter more: a multi-object tracker can pass every unit test and still switch
identities the moment two objects cross, so the second half of this file drives
whole sequences through `ByteTracker` and asserts on the identities that come
out the other end.

Scenarios use synthetic boxes rather than a real sequence, which keeps them
deterministic and fast at the cost of realism. They are a regression net, not a
benchmark: MOTA/HOTA numbers need real footage.
"""

import numpy as np
import pytest

import pixelflow as pf
from pixelflow.tracker import KalmanFilter, STrack, TrackState, matching


# ============================================================================
# Helpers
# ============================================================================

def box(x, y, w=50, h=50):
    """One [x1, y1, x2, y2] box at (x, y)."""
    return [x, y, x + w, y + h]


def moving(start_x, y, step, frames, w=50, h=50):
    """One object translating `step` px per frame."""
    return [box(start_x + step * i, y, w, h) for i in range(frames)]


def track_sequence(tracker, frames, scores=None):
    """Drive `frames` through `tracker`, returning the tracker_ids seen per frame."""
    seen = []
    for i, boxes in enumerate(frames):
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        frame_scores = scores[i] if scores is not None else [0.9] * len(boxes)
        detections = pf.from_arrays(boxes, scores=frame_scores,
                                    class_ids=[0] * len(boxes))
        seen.append([d.tracker_id for d in tracker.update(detections)])
    return seen


def assigned(frame_ids):
    """The ids actually assigned in one frame, as a set (None means unconfirmed)."""
    return {i for i in frame_ids if i is not None}


# ============================================================================
# KalmanFilter
# ============================================================================

class TestKalmanFilter:
    """The constant-velocity model every track carries."""

    def test_initiate_shapes_and_position(self):
        """A new track starts at the measurement with zero velocity."""
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([100.0, 200.0, 0.5, 80.0]))
        assert mean.shape == (8,)
        assert cov.shape == (8, 8)
        np.testing.assert_allclose(mean[:4], [100.0, 200.0, 0.5, 80.0])
        np.testing.assert_allclose(mean[4:], 0.0)

    def test_predict_advances_by_velocity(self):
        """Position moves by velocity; a stationary track stays put."""
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([100.0, 200.0, 0.5, 80.0]))
        mean[4:6] = [10.0, -5.0]  # vx, vy
        moved, _ = kf.predict(mean, cov)
        assert moved[0] == pytest.approx(110.0)
        assert moved[1] == pytest.approx(195.0)

    def test_predict_grows_uncertainty(self):
        """Predicting without a measurement makes the estimate less certain."""
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([100.0, 200.0, 0.5, 80.0]))
        _, predicted_cov = kf.predict(mean, cov)
        assert np.trace(predicted_cov) > np.trace(cov)

    def test_update_pulls_toward_measurement(self):
        """Correcting with an observation moves the estimate toward it."""
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([100.0, 200.0, 0.5, 80.0]))
        mean, cov = kf.predict(mean, cov)
        observation = np.array([150.0, 200.0, 0.5, 80.0])
        corrected, corrected_cov = kf.update(mean, cov, observation)
        assert mean[0] < corrected[0] <= observation[0]
        assert np.trace(corrected_cov) < np.trace(cov)

    def test_project_returns_measurement_space(self):
        """Projection drops the velocity half of the state."""
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([100.0, 200.0, 0.5, 80.0]))
        projected_mean, projected_cov = kf.project(mean, cov)
        assert projected_mean.shape == (4,)
        assert projected_cov.shape == (4, 4)

    def test_gating_distance_ranks_by_closeness(self):
        """A measurement near the prediction gates closer than a far one."""
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([100.0, 200.0, 0.5, 80.0]))
        distances = kf.gating_distance(
            mean, cov, np.array([[101.0, 201.0, 0.5, 80.0],
                                 [900.0, 900.0, 0.5, 80.0]]))
        assert distances.shape == (2,)
        assert distances[0] < distances[1]


# ============================================================================
# Association maths
# ============================================================================

class TestMatching:
    """IoU, cost matrices and the assignment step."""

    def test_iou_identical_boxes(self):
        """A box fully overlaps itself."""
        b = np.array([[0.0, 0.0, 10.0, 10.0]])
        assert matching.box_iou_batch(b, b)[0, 0] == pytest.approx(1.0)

    def test_iou_disjoint_boxes(self):
        """Boxes that do not touch score zero."""
        a = np.array([[0.0, 0.0, 10.0, 10.0]])
        b = np.array([[100.0, 100.0, 110.0, 110.0]])
        assert matching.box_iou_batch(a, b)[0, 0] == pytest.approx(0.0)

    def test_iou_partial_overlap_exact_value(self):
        """A known overlap: 25 intersection over 175 union."""
        a = np.array([[0.0, 0.0, 10.0, 10.0]])
        b = np.array([[5.0, 5.0, 15.0, 15.0]])
        assert matching.box_iou_batch(a, b)[0, 0] == pytest.approx(25.0 / 175.0)

    def test_iou_matrix_shape(self):
        """Every box in a is scored against every box in b."""
        a = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
        b = np.array([[0.0, 0.0, 10.0, 10.0]])
        assert matching.box_iou_batch(a, b).shape == (2, 1)

    def test_linear_assignment_pairs_the_obvious_match(self):
        """Two tracks and two detections, one clearly cheaper pairing each."""
        cost = np.array([[0.05, 0.9], [0.9, 0.05]])
        matches, unmatched_a, unmatched_b = matching.linear_assignment(cost, thresh=0.5)
        assert sorted(map(tuple, matches)) == [(0, 0), (1, 1)]
        assert len(unmatched_a) == 0 and len(unmatched_b) == 0

    def test_linear_assignment_rejects_costly_pairs(self):
        """A pairing above the threshold is left unmatched rather than forced."""
        cost = np.array([[0.99]])
        matches, unmatched_a, unmatched_b = matching.linear_assignment(cost, thresh=0.5)
        assert len(matches) == 0
        assert list(unmatched_a) == [0] and list(unmatched_b) == [0]

    def test_linear_assignment_handles_empty(self):
        """No tracks or no detections is not an error."""
        matches, unmatched_a, unmatched_b = matching.linear_assignment(
            np.empty((0, 0)), thresh=0.5)
        assert len(matches) == 0

    def test_iou_distance_is_one_minus_iou(self):
        """Distance and overlap are complements, so identical boxes cost nothing."""
        tracks = [STrack(np.array([0.0, 0.0, 10.0, 10.0]), 0.9, 0)]
        assert matching.iou_distance(tracks, tracks)[0, 0] == pytest.approx(0.0, abs=1e-6)


# ============================================================================
# Track state machine
# ============================================================================

class TestSTrack:
    """One track's coordinate conversions and lifecycle."""

    def test_coordinate_round_trip(self):
        """tlwh -> tlbr -> tlwh returns the original box."""
        tlwh = np.array([10.0, 20.0, 30.0, 40.0])
        np.testing.assert_allclose(STrack.tlbr_to_tlwh(STrack.tlwh_to_tlbr(tlwh)), tlwh)

    def test_tlwh_to_xyah(self):
        """The Kalman state is centre x, centre y, aspect ratio, height."""
        np.testing.assert_allclose(
            STrack.tlwh_to_xyah(np.array([10.0, 20.0, 30.0, 60.0])),
            [25.0, 50.0, 0.5, 60.0])

    def test_new_track_starts_unconfirmed(self):
        """A track is NEW until it is activated."""
        assert STrack(np.array([0.0, 0.0, 10.0, 10.0]), 0.9, 0).state == TrackState.NEW

    def test_activate_marks_tracked_and_assigns_id(self):
        """Activation gives the track its identity."""
        track = STrack(np.array([0.0, 0.0, 10.0, 10.0]), 0.9, 0)
        track.activate(KalmanFilter(), frame_id=1, track_id=7)
        assert track.track_id == 7
        assert track.state == TrackState.TRACKED

    def test_mark_lost_and_removed(self):
        """A track that stops matching goes lost, then removed."""
        track = STrack(np.array([0.0, 0.0, 10.0, 10.0]), 0.9, 0)
        track.activate(KalmanFilter(), frame_id=1, track_id=1)
        track.mark_lost()
        assert track.state == TrackState.LOST
        track.mark_removed()
        assert track.state == TrackState.REMOVED


# ============================================================================
# ByteTracker basics
# ============================================================================

class TestByteTrackerBasics:
    """Construction, resetting and degenerate input."""

    def test_defaults(self):
        """The documented defaults are what the constructor actually uses."""
        tracker = pf.tracker.ByteTracker()
        assert tracker.track_activation_threshold == 0.25
        assert tracker.max_time_lost == 30
        assert tracker.minimum_consecutive_frames == 3

    def test_empty_detections_do_not_raise(self):
        """A frame with nothing in it is normal, not an error."""
        tracker = pf.tracker.ByteTracker()
        out = tracker.update(pf.from_arrays(np.empty((0, 4)), scores=[], class_ids=[]))
        assert len(out) == 0

    def test_many_empty_frames_do_not_raise(self):
        """A tracker that has never seen anything keeps working."""
        tracker = pf.tracker.ByteTracker()
        for _ in range(5):
            tracker.update(pf.from_arrays(np.empty((0, 4)), scores=[], class_ids=[]))
        assert len(track_sequence(tracker, [[box(0, 0)]])[0]) == 1

    def test_reset_clears_identities(self):
        """After a reset the next object starts from the first id again."""
        tracker = pf.tracker.ByteTracker()
        track_sequence(tracker, moving(0, 0, 10, 8))
        first_run_id = max(assigned(track_sequence(tracker, [[box(80, 0)]])[0]), default=None)
        tracker.reset()
        second_run = track_sequence(tracker, moving(0, 0, 10, 8))
        assert first_run_id is not None
        assert max(assigned(second_run[-1])) == 1

    def test_metrics_report_frames_and_tracks(self):
        """The metrics counter follows what the tracker actually did."""
        tracker = pf.tracker.ByteTracker()
        track_sequence(tracker, moving(0, 0, 10, 10))
        metrics = tracker.get_metrics()
        assert metrics["total_frames"] == 10
        assert metrics["total_tracks"] >= 1

    def test_detection_count_is_preserved(self):
        """Tracking assigns identities; it never adds or drops detections."""
        tracker = pf.tracker.ByteTracker()
        for frame_ids in track_sequence(tracker, [[box(0, 0), box(200, 200)]] * 6):
            assert len(frame_ids) == 2


# ============================================================================
# Scenarios -- the behaviour that actually matters
# ============================================================================

class TestTrackingScenarios:
    """Whole sequences, asserting on identity rather than internals."""

    def test_single_object_keeps_one_id(self):
        """Linear motion is the easiest case and must be rock solid."""
        tracker = pf.tracker.ByteTracker()
        frames = track_sequence(tracker, moving(0, 100, 8, 20))
        settled = [assigned(f) for f in frames[5:]]
        assert all(s == settled[0] for s in settled)
        assert len(settled[0]) == 1

    def test_two_objects_get_distinct_ids(self):
        """Separate objects are never conflated."""
        tracker = pf.tracker.ByteTracker()
        left = moving(0, 100, 8, 20)
        right = moving(400, 300, -8, 20)
        frames = track_sequence(tracker, [[l[0], r[0]] for l, r in
                                          zip([[b] for b in left], [[b] for b in right])])
        assert len(assigned(frames[-1])) == 2

    def test_short_occlusion_preserves_identity(self):
        """A gap well inside lost_track_buffer must not mint a new id."""
        tracker = pf.tracker.ByteTracker()
        frames = [[b] for b in moving(0, 100, 10, 10)]
        frames += [[]] * 5
        frames += [[b] for b in moving(150, 100, 10, 10)]
        seen = track_sequence(tracker, frames)
        before = assigned(seen[8])
        after = assigned(seen[-1])
        assert before and after
        assert before == after

    def test_long_occlusion_starts_a_new_track(self):
        """Past the buffer the track is gone, and honesty beats a stale id."""
        tracker = pf.tracker.ByteTracker(lost_track_buffer=5)
        frames = [[b] for b in moving(0, 100, 10, 10)]
        frames += [[]] * 15
        frames += [[b] for b in moving(500, 100, 10, 10)]
        seen = track_sequence(tracker, frames)
        before = assigned(seen[8])
        after = assigned(seen[-1])
        assert before and after
        assert before != after

    def test_crossing_objects_do_not_swap_ids(self):
        """Two objects passing through each other is the classic switch case."""
        tracker = pf.tracker.ByteTracker()
        frames = [[box(10 + 12 * i, 100), box(290 - 12 * i, 100)] for i in range(24)]
        seen = track_sequence(tracker, frames)
        settled = [assigned(f) for f in seen[5:]]
        assert all(s == settled[0] for s in settled), "identity set changed across the crossing"
        assert len(settled[0]) == 2

    @pytest.mark.parametrize("step,expected", [
        (5, 1), (10, 1), (15, 1), (20, 1), (25, 1), (30, 1),
    ])
    def test_tracks_up_to_the_association_limit(self, step, expected):
        """Association is IoU-only, so it holds while boxes still overlap frame to frame.

        Documents the working envelope for a 50px box: everything here overlaps
        enough to associate, and the Kalman prediction extends the range past
        what raw IoU alone would allow.
        """
        tracker = pf.tracker.ByteTracker()
        seen = track_sequence(tracker, [[b] for b in moving(100, 100, step, 15)])
        assert len(assigned(seen[-1])) == expected

    @pytest.mark.xfail(
        strict=True,
        reason="No camera motion compensation. Association is IoU-only, so once "
               "frame-to-frame displacement exceeds roughly 0.6x the box width the "
               "first match never happens, velocity is never learned, and no track "
               "is ever confirmed. Remove this marker when CMC lands.")
    def test_camera_pan_keeps_identities(self):
        """A translating camera must not look like every object vanishing."""
        tracker = pf.tracker.ByteTracker()
        frames = [[box(100 + 40 * i, 100), box(200 + 40 * i, 300)] for i in range(15)]
        seen = track_sequence(tracker, frames)
        settled = [assigned(f) for f in seen[4:]]
        assert all(s == settled[0] for s in settled)
        assert len(settled[0]) == 2

    def test_fast_motion_is_dropped_not_misattributed(self):
        """Beyond the association limit the tracker reports nothing.

        Pins the current failure mode: it declines to confirm a track rather
        than assigning a stream of new ids to the same object. Silence is the
        better of the two wrong answers, and this test says which one we get.
        """
        tracker = pf.tracker.ByteTracker()
        seen = track_sequence(tracker, [[b] for b in moving(100, 100, 40, 15)])
        assert assigned(seen[-1]) == set()

    def test_crowd_keeps_every_object_separate(self):
        """Eight objects in a grid, all moving together."""
        tracker = pf.tracker.ByteTracker()
        frames = []
        for i in range(12):
            frames.append([box(60 * col + 4 * i, 60 * row) for row in range(2) for col in range(4)])
        seen = track_sequence(tracker, frames)
        assert len(assigned(seen[-1])) == 8

    def test_low_confidence_detections_sustain_a_track(self):
        """The second association stage is ByteTrack's whole point."""
        tracker = pf.tracker.ByteTracker()
        frames = [[b] for b in moving(0, 100, 8, 16)]
        # Confident enough to start, then a run of weak detections mid-sequence.
        scores = [[0.9]] * 6 + [[0.35]] * 5 + [[0.9]] * 5
        seen = track_sequence(tracker, frames, scores=scores)
        assert assigned(seen[5]) == assigned(seen[-1]) != set()

    def test_new_object_entering_gets_a_fresh_id(self):
        """An object that was never there before is not an existing track."""
        tracker = pf.tracker.ByteTracker()
        frames = [[box(0 + 8 * i, 100)] for i in range(10)]
        frames += [[box(80 + 8 * i, 100), box(400, 300)] for i in range(10)]
        seen = track_sequence(tracker, frames)
        assert len(assigned(seen[9])) == 1
        assert len(assigned(seen[-1])) == 2

    def test_minimum_consecutive_frames_delays_confirmation(self):
        """Tracks are unconfirmed until they have been seen enough times."""
        tracker = pf.tracker.ByteTracker(minimum_consecutive_frames=4)
        seen = track_sequence(tracker, [[b] for b in moving(0, 100, 5, 8)])
        assert assigned(seen[0]) == set()
        assert assigned(seen[-1]) != set()

    def test_stationary_object_holds_its_id(self):
        """Zero velocity is a legitimate trajectory."""
        tracker = pf.tracker.ByteTracker()
        seen = track_sequence(tracker, [[box(100, 100)]] * 15)
        settled = [assigned(f) for f in seen[4:]]
        assert all(s == settled[0] for s in settled)
        assert len(settled[0]) == 1
