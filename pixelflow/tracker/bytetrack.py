"""
ByteTrack: Multi-Object Tracking by Associating Every Detection Box

This module implements the ByteTrack algorithm for multi-object tracking,
which achieves high performance by associating both high and low confidence detections.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .kalman_filter import KalmanFilter
from .track import STrack, TrackState
from . import matching


@dataclass
class TrackerMetrics:
    """Metrics for tracking performance analysis."""
    total_tracks: int = 0
    active_tracks: int = 0
    lost_tracks: int = 0
    removed_tracks: int = 0
    total_frames: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'total_tracks': self.total_tracks,
            'active_tracks': self.active_tracks,
            'lost_tracks': self.lost_tracks,
            'removed_tracks': self.removed_tracks,
            'total_frames': self.total_frames
        }


class ByteTracker:
    """
    ByteTrack tracker for multi-object tracking.
    
    ByteTrack associates every detection box instead of only high-confidence ones,
    utilizing similarities with existing tracklets to recover true objects from
    low-confidence detections while filtering out background.
    
    Args:
        track_activation_threshold: Detection confidence threshold for track activation (default: 0.25)
        lost_track_buffer: Number of frames to buffer when a track is lost (default: 30)
        minimum_matching_threshold: IoU threshold for first-stage matching with high confidence detections (default: 0.7)
        minimum_consecutive_frames: Minimum frames before considering a track valid (default: 3)
        second_match_threshold: IoU threshold for second-stage matching with low confidence detections (default: 0.5)
        assignment_threshold: IoU threshold for assigning tracker IDs to predictions (default: 0.3)
    """
    
    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.7,
        minimum_consecutive_frames: int = 3,
        second_match_threshold: float = 0.5,
        assignment_threshold: float = 0.3
    ):
        self.track_activation_threshold = track_activation_threshold
        self.minimum_matching_threshold = minimum_matching_threshold
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.second_match_threshold = second_match_threshold
        self.assignment_threshold = assignment_threshold
        
        # Frame and threshold management
        self.frame_id = 0
        self.det_thresh = track_activation_threshold + 0.1
        self.max_time_lost = lost_track_buffer
        
        # Kalman filters
        self.kalman_filter = KalmanFilter()
        
        # Track lists
        self.tracked_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []
        self.removed_tracks: List[STrack] = []
        
        # Track ID counter
        self.next_id = 1
        
        # Metrics
        self.metrics = TrackerMetrics()
        
    def update(self, results: 'Results') -> 'Results':
        """
        Update tracker with new detections from Results object.
        
        Args:
            results: Results object containing predictions with bboxes and confidences
        
        Returns:
            Results object with tracker_id assigned to each prediction
        """
        self.frame_id += 1
        self.metrics.total_frames = self.frame_id
        
        activated_tracks = []
        refind_tracks = []
        lost_tracks = []
        removed_tracks = []
        
        # Extract detection data from Results
        if len(results) == 0:
            # No detections, just update existing tracks
            self._update_tracks_no_detections()
            return results
        
        # Convert predictions to numpy arrays
        bboxes = []
        scores = []
        class_ids = []
        
        for pred in results.predictions:
            if pred.bbox is not None and pred.confidence is not None:
                # Convert bbox from [x1, y1, x2, y2] to [x1, y1, x2, y2]
                bboxes.append(pred.bbox)
                scores.append(pred.confidence)
                class_ids.append(pred.class_id if pred.class_id is not None else -1)
        
        if len(bboxes) == 0:
            self._update_tracks_no_detections()
            return results
        
        bboxes = np.array(bboxes)
        scores = np.array(scores)
        class_ids = np.array(class_ids)
        
        # Split detections into high and low confidence
        remain_inds = scores > self.track_activation_threshold
        inds_low = scores > 0.1
        inds_high = scores > self.track_activation_threshold
        
        inds_second = np.logical_and(inds_low, ~inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        class_ids_keep = class_ids[remain_inds]
        class_ids_second = class_ids[inds_second]
        
        # Create STrack objects for high confidence detections
        if len(dets) > 0:
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbr), score, class_id)
                for tlbr, score, class_id in zip(dets, scores_keep, class_ids_keep)
            ]
        else:
            detections = []
        
        # Separate tracked and unconfirmed tracks
        unconfirmed = []
        tracked_tracks = []
        
        for track in self.tracked_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_tracks.append(track)
        
        # Combine tracked and lost tracks for matching
        track_pool = self._joint_tracks(tracked_tracks, self.lost_tracks)
        
        # Predict current location with Kalman filter
        STrack.multi_predict(track_pool)
        
        # First association with high score detection boxes
        dists = matching.iou_distance(track_pool, detections)
        dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.minimum_matching_threshold
        )
        
        for itracked, idet in matches:
            track = track_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.TRACKED:
                track.update(detections[idet], self.frame_id)
                activated_tracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracks.append(track)
        
        # Second association with low score detection boxes
        if len(dets_second) > 0:
            detections_second = [
                STrack(STrack.tlbr_to_tlwh(tlbr), score, class_id)
                for tlbr, score, class_id in zip(dets_second, scores_second, class_ids_second)
            ]
        else:
            detections_second = []
        
        r_tracked_tracks = [
            track_pool[i]
            for i in u_track
            if track_pool[i].state == TrackState.TRACKED
        ]
        
        dists = matching.iou_distance(r_tracked_tracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=self.second_match_threshold)
        
        for itracked, idet in matches:
            track = r_tracked_tracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.TRACKED:
                track.update(det, self.frame_id)
                activated_tracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracks.append(track)
        
        # Handle lost tracks
        for it in u_track:
            track = r_tracked_tracks[it]
            if track.state != TrackState.LOST:
                track.mark_lost()
                lost_tracks.append(track)
        
        # Deal with unconfirmed tracks
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_tracks.append(unconfirmed[itracked])
        
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_tracks.append(track)
        
        # Initialize new tracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            
            track_id = self._next_id()
            track.activate(self.kalman_filter, self.frame_id, track_id)
            activated_tracks.append(track)
        
        # Update track states
        for track in self.lost_tracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_tracks.append(track)
        
        # Update track lists
        self.tracked_tracks = [
            t for t in self.tracked_tracks if t.state == TrackState.TRACKED
        ]
        self.tracked_tracks = self._joint_tracks(self.tracked_tracks, activated_tracks)
        self.tracked_tracks = self._joint_tracks(self.tracked_tracks, refind_tracks)
        self.lost_tracks = self._sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_tracks)
        self.lost_tracks = self._sub_tracks(self.lost_tracks, self.removed_tracks)
        self.removed_tracks.extend(removed_tracks)
        self.tracked_tracks, self.lost_tracks = self._remove_duplicate_tracks(
            self.tracked_tracks, self.lost_tracks
        )
        
        # Filter by minimum consecutive frames
        output_tracks = [
            track for track in self.tracked_tracks
            if track.is_activated and track.tracklet_len >= self.minimum_consecutive_frames
        ]
        
        # Update metrics
        self._update_metrics()
        
        # Assign tracker IDs to predictions
        self._assign_tracker_ids(results, output_tracks)
        
        return results
    
    def _assign_tracker_ids(self, results: 'Results', tracks: List[STrack]):
        """
        Assign tracker IDs to predictions in Results object.
        
        Args:
            results: Results object to update
            tracks: List of active tracks
        """
        if len(tracks) == 0:
            # No tracks, clear all tracker IDs
            for pred in results.predictions:
                pred.tracker_id = None
            return
        
        # Get bounding boxes from predictions and tracks
        pred_boxes = []
        for pred in results.predictions:
            if pred.bbox is not None:
                pred_boxes.append(pred.bbox)
            else:
                pred_boxes.append([0, 0, 0, 0])
        
        if len(pred_boxes) == 0:
            return
        
        pred_boxes = np.array(pred_boxes)
        track_boxes = np.array([track.tlbr for track in tracks])
        
        # Calculate IoU between predictions and tracks
        ious = matching.box_iou_batch(pred_boxes, track_boxes)
        
        # Assign tracker IDs based on best IoU match
        for i, pred in enumerate(results.predictions):
            if np.max(ious[i]) > self.assignment_threshold:  # Minimum IoU threshold for assignment
                best_track_idx = np.argmax(ious[i])
                pred.tracker_id = tracks[best_track_idx].track_id
            else:
                pred.tracker_id = None
    
    def _update_tracks_no_detections(self):
        """Update tracks when no detections are present."""
        for track in self.tracked_tracks:
            track.predict()
            if track.time_since_update > self.max_time_lost:
                track.mark_lost()
                self.lost_tracks.append(track)
        
        self.tracked_tracks = [
            t for t in self.tracked_tracks if t.state == TrackState.TRACKED
        ]
    
    def _joint_tracks(self, tracks_a: List[STrack], tracks_b: List[STrack]) -> List[STrack]:
        """
        Join two lists of tracks, removing duplicates.
        
        Args:
            tracks_a: First list of tracks
            tracks_b: Second list of tracks
        
        Returns:
            Combined list of unique tracks
        """
        exists = set()
        res = []
        
        for t in tracks_a:
            exists.add(t.track_id)
            res.append(t)
        
        for t in tracks_b:
            if t.track_id not in exists:
                exists.add(t.track_id)
                res.append(t)
        
        return res
    
    def _sub_tracks(self, tracks_a: List[STrack], tracks_b: List[STrack]) -> List[STrack]:
        """
        Subtract tracks_b from tracks_a.
        
        Args:
            tracks_a: List to subtract from
            tracks_b: List to subtract
        
        Returns:
            tracks_a minus tracks_b
        """
        tracks = {}
        for t in tracks_a:
            tracks[t.track_id] = t
        
        for t in tracks_b:
            if t.track_id in tracks:
                del tracks[t.track_id]
        
        return list(tracks.values())
    
    def _remove_duplicate_tracks(
        self, tracks_a: List[STrack], tracks_b: List[STrack]
    ) -> Tuple[List[STrack], List[STrack]]:
        """
        Remove duplicate tracks based on IoU.
        
        Args:
            tracks_a: First list of tracks
            tracks_b: Second list of tracks
        
        Returns:
            Cleaned lists without duplicates
        """
        pdist = matching.iou_distance(tracks_a, tracks_b)
        pairs = np.where(pdist < 0.15)
        
        dupa, dupb = [], []
        for p, q in zip(pairs[0], pairs[1]):
            timep = tracks_a[p].frame_id - tracks_a[p].start_frame
            timeq = tracks_b[q].frame_id - tracks_b[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        
        resa = [t for i, t in enumerate(tracks_a) if i not in dupa]
        resb = [t for i, t in enumerate(tracks_b) if i not in dupb]
        
        return resa, resb
    
    def _next_id(self) -> int:
        """Get next available track ID."""
        track_id = self.next_id
        self.next_id += 1
        self.metrics.total_tracks = self.next_id - 1
        return track_id
    
    def _update_metrics(self):
        """Update tracking metrics."""
        self.metrics.active_tracks = len([t for t in self.tracked_tracks if t.is_activated])
        self.metrics.lost_tracks = len(self.lost_tracks)
        self.metrics.removed_tracks = len(self.removed_tracks)
    
    def reset(self):
        """Reset tracker to initial state."""
        self.frame_id = 0
        self.next_id = 1
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.metrics = TrackerMetrics()
    
    def get_metrics(self) -> Dict:
        """Get current tracking metrics."""
        return self.metrics.to_dict()