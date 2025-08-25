"""
Single object tracking module for PixelFlow ByteTrack implementation.

This module defines the STrack class which represents a single tracked object
and manages its state throughout the tracking process.
"""

import numpy as np
from enum import Enum
from typing import Optional, List
from .kalman_filter import KalmanFilter


class TrackState(Enum):
    """Enumeration of possible track states."""
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class STrack:
    """
    Single object track class for ByteTrack.
    
    Represents a single tracked object with its state, motion model,
    and tracking history.
    """
    
    shared_kalman = KalmanFilter()
    
    def __init__(self, tlwh: np.ndarray, score: float, class_id: Optional[int] = None):
        """
        Initialize a new track.
        
        Args:
            tlwh: Bounding box in format [top, left, width, height]
            score: Detection confidence score
            class_id: Optional class ID for multi-class tracking
        """
        # Convert tlwh to tlbr for internal use
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score
        self.class_id = class_id
        
        # Kalman filter states
        self.mean = None
        self.covariance = None
        
        # Track management
        self.is_activated = False
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0
        self.state = TrackState.NEW
        
        # Track metrics
        self.time_since_update = 0
        
    @property
    def tlwh(self) -> np.ndarray:
        """Get bounding box in tlwh format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]  # Convert aspect ratio to width
        ret[:2] -= ret[2:] / 2  # Convert center to top-left
        return ret
    
    @property
    def tlbr(self) -> np.ndarray:
        """Get bounding box in tlbr format (top-left, bottom-right)."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]  # Convert width-height to bottom-right
        return ret
    
    @property
    def xywh(self) -> np.ndarray:
        """Get bounding box in xywh format (center x, center y, width, height)."""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2  # Convert top-left to center
        return ret
    
    @property
    def xyxy(self) -> np.ndarray:
        """Get bounding box in xyxy format (x1, y1, x2, y2)."""
        return self.tlbr
    
    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """
        Convert tlwh box to xyah format (center x, center y, aspect ratio, height).
        
        Args:
            tlwh: Box in [top, left, width, height] format
            
        Returns:
            Box in [center_x, center_y, aspect_ratio, height] format
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2  # Convert to center
        ret[2] /= ret[3]  # width/height = aspect ratio
        return ret
    
    @staticmethod
    def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        """
        Convert tlbr box to tlwh format.
        
        Args:
            tlbr: Box in [top, left, bottom, right] format
            
        Returns:
            Box in [top, left, width, height] format
        """
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]  # Convert bottom-right to width-height
        return ret
    
    @staticmethod
    def tlwh_to_tlbr(tlwh: np.ndarray) -> np.ndarray:
        """
        Convert tlwh box to tlbr format.
        
        Args:
            tlwh: Box in [top, left, width, height] format
            
        Returns:
            Box in [top, left, bottom, right] format
        """
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]  # Convert width-height to bottom-right
        return ret
    
    def activate(self, kalman_filter: KalmanFilter, frame_id: int, track_id: int):
        """
        Activate a new track.
        
        Args:
            kalman_filter: Kalman filter instance for state estimation
            frame_id: Current frame ID
            track_id: Assigned track ID
        """
        self.track_id = track_id
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 0
        
        # Initialize Kalman filter
        self.mean, self.covariance = kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )
        
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.time_since_update = 0
    
    def re_activate(self, new_track: 'STrack', frame_id: int, new_id: bool = False):
        """
        Reactivate a lost track.
        
        Args:
            new_track: New detection to reactivate with
            frame_id: Current frame ID
            new_id: Whether to assign a new track ID
        """
        # Smooth velocity transition during reactivation
        if self.time_since_update > 2:
            # If track was lost for multiple frames, reduce trust in velocity
            self.mean[4:8] *= 0.3  # Keep only 30% of velocity
            
        self.mean, self.covariance = self.shared_kalman.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.frame_id = frame_id
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.score = new_track.score
        self.time_since_update = 0
        
        if new_id:
            self.track_id = new_track.track_id
    
    def update(self, new_track: 'STrack', frame_id: int):
        """
        Update track with a new detection.
        
        Args:
            new_track: New detection matched to this track
            frame_id: Current frame ID
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        # Update Kalman filter
        self.mean, self.covariance = self.shared_kalman.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.score = new_track.score
        self.time_since_update = 0
    
    def predict(self):
        """Predict the track's next state using Kalman filter."""
        if self.state != TrackState.TRACKED:
            # Reduce velocity gradually instead of hard reset for smoother transitions
            self.mean[4:8] *= 0.5  # Dampen all velocities by 50%
            
        self.mean, self.covariance = self.shared_kalman.predict(
            self.mean, self.covariance
        )
        self.time_since_update += 1
    
    def mark_lost(self):
        """Mark the track as lost."""
        self.state = TrackState.LOST
    
    def mark_removed(self):
        """Mark the track as removed."""
        self.state = TrackState.REMOVED
    
    @staticmethod
    def multi_predict(tracks: List['STrack']):
        """
        Predict states for multiple tracks.
        
        Args:
            tracks: List of tracks to predict
        """
        if len(tracks) > 0:
            for track in tracks:
                track.predict()
    
    def __repr__(self):
        return f"STrack(id={self.track_id}, state={self.state.name}, score={self.score:.2f})"