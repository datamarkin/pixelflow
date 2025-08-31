"""
Buffer module for temporal frame and results management.

This module provides a rolling window buffer that collects frames and their
corresponding detection results, enabling temporal context for downstream processing.
"""

from typing import Optional, Tuple, List
import numpy as np


class Buffer:
    """
    A rolling window buffer for frames and detection results.
    
    The buffer collects a specified number of frames and their results,
    always returning the middle frame once the buffer is full. This provides
    temporal context (past and future frames) for advanced processing like
    smoothing, interpolation, or motion analysis.
    
    Attributes:
        buffer_size: Number of frames to buffer (should be odd for clean middle)
        frame_buffer: List storing the buffered frames
        results_buffer: List storing the buffered results
        is_full: Whether the buffer has reached capacity
    """
    
    def __init__(self, frames: int = 5):
        """
        Initialize the Buffer.
        
        Args:
            frames: Number of frames to buffer (default: 5).
                   Should be odd for a clean middle frame.
                   With frames=5, you get 2 past, 1 current, 2 future frames.
        """
        if frames < 1:
            raise ValueError("Buffer size must be at least 1")
        
        if frames % 2 == 0:
            print(f"Warning: Buffer size {frames} is even. Consider using odd number for clean middle frame.")
        
        self.buffer_size = frames
        self.frame_buffer: List[np.ndarray] = []
        self.results_buffer: List = []
        self.is_full = False
        self._frame_count = 0
    
    def update(self, results, frame: np.ndarray) -> Tuple:
        """
        Add new frame and results to buffer, return middle frame and results.
        
        This method maintains a rolling window of frames and results.
        Until the buffer is full, it returns a black frame with empty results.
        Once full, it returns the middle frame and its corresponding results.
        
        Args:
            results: Detection results for the current frame
            frame: Current video frame as numpy array
            
        Returns:
            Tuple of (results, frame):
            - (empty_results, black_frame) if buffer not yet full
            - (middle_results, middle_frame) once buffer is full
        """
        # Add new frame and results to buffers
        self.frame_buffer.append(frame.copy())
        self.results_buffer.append(results)
        self._frame_count += 1
        
        # Maintain buffer size by removing oldest if exceeded
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
            self.results_buffer.pop(0)
        
        # Check if buffer is full
        if len(self.frame_buffer) >= self.buffer_size:
            self.is_full = True
            # Return middle frame and results
            middle_idx = self.buffer_size // 2
            return self.results_buffer[middle_idx], self.frame_buffer[middle_idx]
        else:
            # Buffer not full yet, return black frame with empty results
            black_frame = np.zeros_like(frame)
            # Try to create empty results of same type as input
            try:
                # Import here to avoid circular dependency
                from pixelflow.results import Detections
                empty_results = Detections()
            except:
                # If can't import Detections, return None
                empty_results = None
            
            return empty_results, black_frame
    
    def get_buffer_contents(self) -> Tuple[List, List[np.ndarray]]:
        """
        Get the current contents of the buffer.
        
        Returns:
            Tuple of (results_buffer, frame_buffer) containing all buffered items.
        """
        return self.results_buffer.copy(), self.frame_buffer.copy()
    
    def get_temporal_context(self) -> Optional[dict]:
        """
        Get temporal context around the current (middle) frame.
        
        Returns:
            Dictionary with past, current, and future frames/results,
            or None if buffer is not full yet.
        """
        if not self.is_full:
            return None
        
        middle_idx = self.buffer_size // 2
        
        return {
            'past_frames': self.frame_buffer[:middle_idx],
            'past_results': self.results_buffer[:middle_idx],
            'current_frame': self.frame_buffer[middle_idx],
            'current_results': self.results_buffer[middle_idx],
            'future_frames': self.frame_buffer[middle_idx + 1:],
            'future_results': self.results_buffer[middle_idx + 1:],
        }
    
    def reset(self):
        """Reset the buffer to initial state."""
        self.frame_buffer.clear()
        self.results_buffer.clear()
        self.is_full = False
        self._frame_count = 0
    
    @property
    def frames_processed(self) -> int:
        """Total number of frames processed by the buffer."""
        return self._frame_count
    
    @property
    def current_size(self) -> int:
        """Current number of frames in the buffer."""
        return len(self.frame_buffer)
    
    @property
    def delay(self) -> int:
        """Frame delay introduced by the buffer (in frames)."""
        return self.buffer_size // 2 if self.buffer_size > 1 else 0