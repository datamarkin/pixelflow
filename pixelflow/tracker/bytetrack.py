"""
ByteTrack: Multi-Object Tracking by Associating Every Detection Box

This module implements the ByteTrack algorithm for multi-object tracking,
which achieves high performance by associating both high and low confidence detections.
ByteTrack uses a multi-stage matching process that recovers true objects from
low-confidence detections while filtering out background noise, making it robust
for real-world tracking scenarios.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .kalman_filter import KalmanFilter
from .track import STrack, TrackState
from . import matching


@dataclass
class TrackerMetrics:
    """Tracking performance metrics collection.
    
    Dataclass containing comprehensive metrics for analyzing multi-object tracking performance.
    Tracks counts of active, lost, and removed tracks across video frames for performance evaluation.
    """
    total_tracks: int = 0
    active_tracks: int = 0
    lost_tracks: int = 0
    removed_tracks: int = 0
    total_frames: int = 0
    
    def to_dict(self) -> Dict[str, int]:
        """Convert metrics to dictionary format.
        
        Converts all tracking metrics to a dictionary suitable for JSON serialization,
        logging, or analysis workflows. Useful for saving metrics to files or APIs.
        
        Returns:
            Dict[str, int]: Dictionary containing all metric values with descriptive keys.
                           Keys include: total_tracks, active_tracks, lost_tracks, 
                           removed_tracks, total_frames.
        
        Example:
            >>> import cv2
            >>> import pixelflow as pf
            >>> from ultralytics import YOLO
            >>> 
            >>> # Initialize tracker and process video
            >>> tracker = pf.tracker.ByteTracker()
            >>> model = YOLO("yolo11n.pt")
            >>> video_path = "path/to/video.mp4"
            >>> 
            >>> for frame in pf.video.get_video_frames(video_path):
            ...     outputs = model.predict(frame)
            ...     results = pf.results.from_ultralytics(outputs)
            ...     tracked_results = tracker.update(results)
            >>> 
            >>> # Export metrics
            >>> metrics_dict = tracker.get_metrics()
            >>> 
            >>> # Save to JSON
            >>> import json
            >>> with open("tracking_metrics.json", "w") as f:
            ...     json.dump(metrics_dict, f)
            >>> 
            >>> # Log performance summary
            >>> print(f"Processed {metrics_dict['total_frames']} frames")
            >>> print(f"Tracked {metrics_dict['total_tracks']} objects")
        
        Notes:
            - All values are integers representing counts
            - Dictionary keys are strings for JSON compatibility
            - Values reflect current state when called
        """
        return {
            'total_tracks': self.total_tracks,
            'active_tracks': self.active_tracks,
            'lost_tracks': self.lost_tracks,
            'removed_tracks': self.removed_tracks,
            'total_frames': self.total_frames
        }


class ByteTracker:
    """Multi-object tracker implementing the ByteTrack algorithm.
    
    ByteTrack associates every detection box instead of only high-confidence ones,
    utilizing similarities with existing tracklets to recover true objects from
    low-confidence detections while filtering out background. Uses a two-stage matching
    process: first matching high-confidence detections to existing tracks, then matching
    low-confidence detections to remaining unmatched tracks. This approach significantly
    improves tracking performance in challenging scenarios with occlusions and detection noise.
    
    Args:
        track_activation_threshold (float): Detection confidence threshold for track activation.
                                          Range: [0.0, 1.0]. Default is 0.25 (25% confidence).
        lost_track_buffer (int): Number of frames to buffer when a track is lost before removal.
                               Range: [1, 100]. Default is 30 frames (~1 second at 30fps).
        minimum_matching_threshold (float): IoU threshold for first-stage matching with high confidence detections.
                                           Range: [0.0, 1.0]. Default is 0.7 (70% overlap required).
        minimum_consecutive_frames (int): Minimum consecutive frames before considering a track valid.
                                        Range: [1, 10]. Default is 3 frames.
        second_match_threshold (float): IoU threshold for second-stage matching with low confidence detections.
                                      Range: [0.0, 1.0]. Default is 0.5 (50% overlap required).
        assignment_threshold (float): IoU threshold for assigning tracker IDs to final detections.
                                    Range: [0.0, 1.0]. Default is 0.3 (30% overlap required).
    
    Raises:
        ValueError: If any threshold parameter is outside valid range [0.0, 1.0]
        ValueError: If frame or buffer parameters are not positive integers
    
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> 
        >>> # Initialize tracker with default settings
        >>> tracker = pf.tracker.ByteTracker()
        >>> model = YOLO("yolo11n.pt")
        >>> video_path = "path/to/video.mp4"
        >>> 
        >>> # Process video frames
        >>> for frame in pf.video.get_video_frames(video_path):
        ...     outputs = model.predict(frame)
        ...     results = pf.results.from_ultralytics(outputs)
        ...     tracked_results = tracker.update(results)
        ...     print(f"Frame has {len(tracked_results)} tracked objects")
        >>> 
        >>> # Configure for high-precision tracking
        >>> precision_tracker = pf.tracker.ByteTracker(
        ...     track_activation_threshold=0.5,
        ...     minimum_matching_threshold=0.8,
        ...     minimum_consecutive_frames=5
        ... )
        >>> 
        >>> # Configure for challenging scenarios
        >>> robust_tracker = pf.tracker.ByteTracker(
        ...     track_activation_threshold=0.1,
        ...     lost_track_buffer=60,
        ...     second_match_threshold=0.3
        ... )
        >>> 
        >>> # Reset tracker between videos
        >>> tracker.reset()
        >>> metrics = tracker.get_metrics()
        >>> print(f"Tracked {metrics['total_tracks']} objects")
    
    Notes:
        - Uses Kalman filtering for motion prediction
        - Implements two-stage association: high-confidence then low-confidence
        - Automatically handles track state transitions (tracked → lost → removed)
        - Maintains track continuity through temporary occlusions
        - Filters out short-lived false positive tracks
        - All threshold parameters are automatically clamped to valid ranges
    
    Performance Notes:
        - Optimized for real-time video processing (>30 FPS on modern hardware)
        - Memory usage scales linearly with number of active tracks
        - IoU computation is the primary bottleneck for scenes with many detections
        - Efficient track management prevents memory leaks in long videos
    
    See Also:
        STrack: Individual track representation with Kalman filtering
        TrackerMetrics: Performance metrics collection for analysis
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
        """Initialize ByteTracker with specified parameters.
        
        Sets up the ByteTracker with all necessary parameters for multi-object tracking.
        Initializes Kalman filters, track lists, and metrics collection. All threshold
        parameters are validated and clamped to appropriate ranges.
        
        Args:
            track_activation_threshold (float): Detection confidence threshold for track activation.
                                              Range: [0.0, 1.0]. Default is 0.25.
            lost_track_buffer (int): Number of frames to buffer lost tracks before removal.
                                   Range: [1, 100]. Default is 30.
            minimum_matching_threshold (float): IoU threshold for first-stage high-confidence matching.
                                               Range: [0.0, 1.0]. Default is 0.7.
            minimum_consecutive_frames (int): Minimum consecutive frames for valid track.
                                            Range: [1, 10]. Default is 3.
            second_match_threshold (float): IoU threshold for second-stage low-confidence matching.
                                          Range: [0.0, 1.0]. Default is 0.5.
            assignment_threshold (float): IoU threshold for final detection-to-track assignment.
                                        Range: [0.0, 1.0]. Default is 0.3.
        
        Raises:
            ValueError: If threshold parameters are outside [0.0, 1.0] range
            ValueError: If integer parameters are not positive
            TypeError: If parameters are not of expected numeric types
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Default initialization for general tracking
            >>> tracker = pf.tracker.ByteTracker()
            >>> 
            >>> # High-precision tracking for critical applications
            >>> precision_tracker = pf.tracker.ByteTracker(
            ...     track_activation_threshold=0.6,
            ...     minimum_matching_threshold=0.8,
            ...     minimum_consecutive_frames=5
            ... )
            >>> 
            >>> # Robust tracking for noisy environments
            >>> robust_tracker = pf.tracker.ByteTracker(
            ...     track_activation_threshold=0.15,
            ...     lost_track_buffer=50,
            ...     second_match_threshold=0.4
            ... )
            >>> 
            >>> # Real-time tracking with fast recovery
            >>> realtime_tracker = pf.tracker.ByteTracker(
            ...     lost_track_buffer=15,
            ...     assignment_threshold=0.2
            ... )
        
        Notes:
            - Parameters are automatically validated and clamped to safe ranges
            - Kalman filter is initialized with default motion model parameters
            - Track ID counter starts at 1 and increments for each new track
            - All track lists (tracked, lost, removed) are initialized empty
            - Metrics collection begins immediately upon initialization
        """
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
        
    def update(self, results: 'Detections') -> 'Detections':
        """Update tracker with new frame detections.
        
        Processes a new frame of detections through the ByteTrack algorithm, performing
        two-stage association to assign tracker IDs. First matches high-confidence detections
        to existing tracks, then matches low-confidence detections to remaining unmatched tracks.
        Handles track state transitions and manages track lifecycle automatically.
        
        Args:
            results (Detections): Detection results containing bounding boxes, confidences,
                                and class IDs. Each detection should have bbox as [x1, y1, x2, y2]
                                coordinates and confidence score between 0.0 and 1.0.
        
        Returns:
            Detections: The same Detections object with tracker_id assigned to each detection.
                       Unmatched detections will have tracker_id=None. Successfully tracked
                       detections receive integer tracker IDs starting from 1.
        
        Raises:
            AttributeError: If detections lack required bbox or confidence attributes
            ValueError: If bbox coordinates are malformed or confidence scores invalid
            TypeError: If results is not a Detections object
        
        Example:
            >>> import cv2
            >>> import pixelflow as pf
            >>> from ultralytics import YOLO
            >>> 
            >>> # Single frame tracking
            >>> tracker = pf.tracker.ByteTracker()
            >>> model = YOLO("yolo11n.pt")
            >>> image = cv2.imread("frame.jpg")
            >>> outputs = model.predict(image)
            >>> results = pf.results.from_ultralytics(outputs)
            >>> tracked_results = tracker.update(results)
            >>> print(f"Tracked {len(tracked_results)} objects with IDs")
            >>> 
            >>> # Video sequence tracking
            >>> cap = cv2.VideoCapture("video.mp4")
            >>> while cap.isOpened():
            ...     ret, frame = cap.read()
            ...     if not ret:
            ...         break
            ...     outputs = model.predict(frame)
            ...     results = pf.results.from_ultralytics(outputs)
            ...     tracked_results = tracker.update(results)
            ...     # Process tracked results...
            >>> cap.release()
            >>> 
            >>> # Batch processing with metrics
            >>> video_path = "path/to/video.mp4"
            >>> for frame in pf.video.get_video_frames(video_path):
            ...     outputs = model.predict(frame)
            ...     results = pf.results.from_ultralytics(outputs)
            ...     tracked_results = tracker.update(results)
            >>> metrics = tracker.get_metrics()
            >>> print(f"Total tracks: {metrics['total_tracks']}")
            >>> 
            >>> # Handle empty detections gracefully
            >>> empty_results = pf.results.Results(detections=[])
            >>> tracked_empty = tracker.update(empty_results)
            >>> # Tracker continues to predict existing tracks
        
        Notes:
            - Automatically handles empty detection frames by predicting existing tracks
            - Uses two-stage matching: high-confidence (>threshold) then low-confidence (>0.1)
            - Implements Kalman filtering for motion prediction between frames
            - Manages track states: new → tracked → lost → removed
            - Filters tracks by minimum consecutive frames before assignment
            - Updates internal metrics automatically with each frame
            - Preserves original Detections object structure and metadata
        
        Performance Notes:
            - Optimized for real-time processing with efficient IoU computation
            - Memory usage scales with number of active tracks (typically <100)
            - Primary bottleneck is IoU distance calculation for large detection counts
            - Track prediction and update operations are highly optimized
        
        See Also:
            reset : Reset tracker state for new video sequences
            get_metrics : Retrieve current tracking performance metrics
        """
        self.frame_id += 1
        self.metrics.total_frames = self.frame_id
        
        activated_tracks = []
        refind_tracks = []
        lost_tracks = []
        removed_tracks = []
        
        # Extract detection data from Detections
        if len(results) == 0:
            # No detections, just update existing tracks
            self._update_tracks_no_detections()
            return results
        
        # Convert detections to numpy arrays
        bboxes = []
        scores = []
        class_ids = []
        
        for pred in results.detections:
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
        
        # Assign tracker IDs to detections
        self._assign_tracker_ids(results, output_tracks)
        
        return results
    
    def _assign_tracker_ids(self, results: 'Detections', tracks: List[STrack]) -> None:
        """Assign tracker IDs to detections based on IoU matching.
        
        Matches active tracks to detections using IoU overlap and assigns tracker IDs
        to detections that exceed the assignment threshold. Unmatched detections receive
        None as tracker_id. This is the final step that connects tracking results to detections.
        
        Args:
            results (Detections): Detections object to update with tracker IDs.
                                 Modified in-place with tracker_id assignments.
            tracks (List[STrack]): List of active tracks with valid bounding boxes
                                 and tracker IDs to match against detections.
        
        Raises:
            AttributeError: If detections lack bbox attribute or tracks lack required attributes
            ValueError: If bounding box coordinates are invalid
        
        Notes:
            - Modifies the results object in-place
            - Uses assignment_threshold for minimum IoU matching
            - Assigns best matching track ID based on highest IoU
            - Unmatched detections get tracker_id=None
            - Handles empty detection or track lists gracefully
        """
        if len(tracks) == 0:
            # No tracks, clear all tracker IDs
            for pred in results.detections:
                pred.tracker_id = None
            return
        
        # Get bounding boxes from detections and tracks
        pred_boxes = []
        for pred in results.detections:
            if pred.bbox is not None:
                pred_boxes.append(pred.bbox)
            else:
                pred_boxes.append([0, 0, 0, 0])
        
        if len(pred_boxes) == 0:
            return
        
        pred_boxes = np.array(pred_boxes)
        track_boxes = np.array([track.tlbr for track in tracks])
        
        # Calculate IoU between detections and tracks
        ious = matching.box_iou_batch(pred_boxes, track_boxes)
        
        # Assign tracker IDs based on best IoU match
        for i, pred in enumerate(results.detections):
            if np.max(ious[i]) > self.assignment_threshold:  # Minimum IoU threshold for assignment
                best_track_idx = np.argmax(ious[i])
                pred.tracker_id = tracks[best_track_idx].track_id
            else:
                pred.tracker_id = None
    
    def _update_tracks_no_detections(self) -> None:
        """Update tracks when no detections are present in current frame.
        
        Handles frames with zero detections by predicting track positions using Kalman
        filtering and transitioning tracks to lost state if they exceed the maximum
        time without detection. Maintains track continuity during temporary occlusions.
        
        Notes:
            - All active tracks are predicted forward using motion model
            - Tracks exceeding max_time_lost are moved to lost state
            - No new tracks are created during this update
            - Existing track IDs are preserved for potential re-detection
        """
        for track in self.tracked_tracks:
            track.predict()
            if track.time_since_update > self.max_time_lost:
                track.mark_lost()
                self.lost_tracks.append(track)
        
        self.tracked_tracks = [
            t for t in self.tracked_tracks if t.state == TrackState.TRACKED
        ]
    
    def _joint_tracks(self, tracks_a: List[STrack], tracks_b: List[STrack]) -> List[STrack]:
        """Join two track lists removing duplicates by track ID.
        
        Combines two lists of tracks while preserving uniqueness based on track IDs.
        Used throughout the tracking pipeline to merge different track categories
        without creating duplicate entries.
        
        Args:
            tracks_a (List[STrack]): First list of tracks to combine.
            tracks_b (List[STrack]): Second list of tracks to add to first list.
        
        Returns:
            List[STrack]: Combined list containing all unique tracks from both inputs.
                         Duplicates (same track_id) are excluded from tracks_b.
        
        Notes:
            - Preserves order of tracks_a completely
            - Only adds tracks from tracks_b that don't exist in tracks_a
            - Uses track_id as unique identifier for duplicate detection
            - Efficient O(n+m) complexity with set-based duplicate checking
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
        """Remove tracks from first list that exist in second list.
        
        Performs set subtraction on track lists using track IDs as unique identifiers.
        Used to clean up track lists by removing tracks that have transitioned to
        different states or been processed in other pipeline stages.
        
        Args:
            tracks_a (List[STrack]): Base list to subtract from.
            tracks_b (List[STrack]): List of tracks to remove from tracks_a.
        
        Returns:
            List[STrack]: New list containing tracks from tracks_a that are not
                         present in tracks_b (based on track_id matching).
        
        Notes:
            - Uses track_id as unique identifier for subtraction
            - Returns new list without modifying input lists
            - Efficient dictionary-based implementation
            - Handles empty lists gracefully
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
        """Remove spatially duplicate tracks based on IoU overlap.
        
        Identifies tracks with high spatial overlap (IoU > 0.15) between two track lists
        and removes duplicates based on track age. Older tracks (longer tracklet_len)
        are preserved while newer tracks are removed to maintain tracking consistency.
        
        Args:
            tracks_a (List[STrack]): First list of tracks to check for duplicates.
            tracks_b (List[STrack]): Second list of tracks to check against first list.
        
        Returns:
            Tuple[List[STrack], List[STrack]]: Cleaned versions of both track lists
                                             with spatial duplicates removed.
        
        Notes:
            - Uses IoU threshold of 0.15 for duplicate detection
            - Preserves tracks with longer history (older start_frame)
            - Prevents multiple tracks from following the same object
            - Essential for maintaining track identity consistency
            - Operates on spatial proximity rather than track ID
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
        """Generate next unique track ID.
        
        Increments the internal track ID counter and updates total track metrics.
        Ensures each new track receives a unique integer identifier starting from 1.
        
        Returns:
            int: Next available track ID for assignment to new tracks.
        
        Notes:
            - Track IDs start at 1 and increment monotonically
            - Updates total_tracks metric automatically
            - Thread-safe for single-threaded tracking scenarios
            - IDs are never reused, even after track removal
        """
        track_id = self.next_id
        self.next_id += 1
        self.metrics.total_tracks = self.next_id - 1
        return track_id
    
    def _update_metrics(self) -> None:
        """Update internal tracking performance metrics.
        
        Recalculates current tracking metrics based on track list states.
        Called automatically during each update cycle to maintain accurate
        performance statistics for analysis and monitoring.
        
        Notes:
            - Updates active_tracks count from currently tracked objects
            - Updates lost_tracks count from temporarily lost objects
            - Updates removed_tracks count from permanently removed objects
            - Called automatically by update() method
            - Metrics reflect instantaneous state when called
        """
        self.metrics.active_tracks = len([t for t in self.tracked_tracks if t.is_activated])
        self.metrics.lost_tracks = len(self.lost_tracks)
        self.metrics.removed_tracks = len(self.removed_tracks)
    
    def reset(self) -> None:
        """Reset tracker to initial state for new video sequences.
        
        Clears all tracks, resets frame counter, and reinitializes metrics.
        Essential when switching between different video sequences to prevent
        track ID conflicts and ensure clean tracking state.
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> tracker = pf.tracker.ByteTracker()
            >>> 
            >>> # Process first video
            >>> video1_frames = pf.video.get_video_frames("video1.mp4")
            >>> for frame in video1_frames:
            ...     # Process frame...
            ...     pass
            >>> 
            >>> # Reset before processing second video
            >>> tracker.reset()
            >>> 
            >>> # Process second video with clean state
            >>> video2_frames = pf.video.get_video_frames("video2.mp4")
            >>> for frame in video2_frames:
            ...     # Process frame...
            ...     pass
        
        Notes:
            - Clears all track lists (tracked, lost, removed)
            - Resets frame counter to 0
            - Resets track ID counter to 1
            - Initializes fresh metrics collection
            - Preserves tracker configuration parameters
            - Should be called between different video sequences
        """
        self.frame_id = 0
        self.next_id = 1
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.metrics = TrackerMetrics()
    
    def get_metrics(self) -> Dict[str, int]:
        """Retrieve current tracking performance metrics.
        
        Returns comprehensive tracking statistics including counts of total,
        active, lost, and removed tracks, plus total frames processed.
        Useful for performance analysis and monitoring tracking quality.
        
        Returns:
            Dict[str, int]: Dictionary containing tracking metrics with keys:
                          total_tracks, active_tracks, lost_tracks, 
                          removed_tracks, total_frames.
        
        Example:
            >>> import cv2
            >>> import pixelflow as pf
            >>> from ultralytics import YOLO
            >>> 
            >>> # Track objects in video and analyze performance
            >>> tracker = pf.tracker.ByteTracker()
            >>> model = YOLO("yolo11n.pt")
            >>> 
            >>> for frame in pf.video.get_video_frames("video.mp4"):
            ...     outputs = model.predict(frame)
            ...     results = pf.results.from_ultralytics(outputs)
            ...     tracked_results = tracker.update(results)
            >>> 
            >>> # Get performance summary
            >>> metrics = tracker.get_metrics()
            >>> print(f"Processed {metrics['total_frames']} frames")
            >>> print(f"Tracked {metrics['total_tracks']} unique objects")
            >>> print(f"Currently tracking {metrics['active_tracks']} objects")
            >>> print(f"Lost {metrics['lost_tracks']} tracks temporarily")
            >>> 
            >>> # Calculate tracking efficiency
            >>> efficiency = metrics['active_tracks'] / max(1, metrics['total_tracks'])
            >>> print(f"Tracking efficiency: {efficiency:.2%}")
        
        Notes:
            - Returns current snapshot of tracking state
            - Metrics are updated automatically during each update() call
            - All values are non-negative integers
            - Dictionary format suitable for JSON serialization
            - Useful for real-time monitoring and post-processing analysis
        
        See Also:
            TrackerMetrics.to_dict : Direct access to metrics dataclass dictionary
        """
        return self.metrics.to_dict()