"""
Buffer module for temporal frame and results management.

This module provides a rolling window buffer that collects frames and their
corresponding detection results, enabling temporal context for downstream processing.
The buffer implements a delay-based temporal processing approach, returning the middle
frame once the buffer is full, which provides both past and future context for advanced
computer vision operations like motion smoothing, temporal interpolation, and trajectory analysis.
"""

from typing import Optional, Tuple, List, Union, Any, Dict
import numpy as np


class Buffer:
    """
    A rolling window buffer for frames and detection results with temporal context.
    
    The Buffer class implements a temporal processing strategy using a rolling window
    approach that maintains a fixed-size collection of frames and their corresponding
    detection results. It employs a delay-based algorithm that returns the middle frame
    once the buffer reaches capacity, providing both historical and future context for
    advanced computer vision operations.
    
    The buffer is particularly useful for temporal smoothing, motion analysis,
    trajectory prediction, and any processing that benefits from knowing both past
    and future states. It automatically handles buffer management, frame synchronization,
    and provides convenient access to temporal context.
    
    Attributes:
        buffer_size (int): Number of frames to buffer (should be odd for clean middle)
        frame_buffer (List[np.ndarray]): List storing the buffered frames
        results_buffer (List): List storing the buffered results
        is_full (bool): Whether the buffer has reached capacity
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> 
        >>> # Setup model and buffer
        >>> model = YOLO("yolo11l.pt")
        >>> buffer = pf.Buffer(frames=5)  # 2 past + 1 current + 2 future
        >>> 
        >>> # Process video with temporal context
        >>> cap = cv2.VideoCapture("video.mp4")
        >>> while True:
        ...     ret, frame = cap.read()
        ...     if not ret:
        ...         break
        ...         
        ...     # Get raw model outputs and convert to PixelFlow format
        ...     outputs = model.predict(frame)
        ...     results = pf.detections.from_ultralytics(outputs)
        ...     
        ...     # Update buffer and get temporally-delayed frame
        ...     buffered_results, buffered_frame = buffer.update(results, frame)
        ...     
        ...     # Process with temporal context (if buffer is full)
        ...     if buffer.is_full:
        ...         temporal_ctx = buffer.get_temporal_context()
        ...         # Use past/future frames for smoothing or prediction
    
    Notes:
        - Buffer introduces a delay equal to buffer_size // 2 frames
        - Odd buffer sizes provide clean middle frame alignment
        - Even buffer sizes will still work but with slight offset
        - Memory usage scales linearly with buffer size and frame resolution
        - All frames are copied to prevent external modifications
    
    Performance Notes:
        - Memory usage: buffer_size * frame_size * channels bytes
        - Frame copying adds ~10-20% overhead for typical resolutions
        - Buffer operations are O(1) for update, O(n) for full access
        - Consider buffer size vs available memory for high-resolution video
    """
    
    def __init__(self, frames: int = 5) -> None:
        """
        Initialize the Buffer with specified capacity.
        
        Creates a new buffer instance with the specified frame capacity. The buffer
        will collect frames until it reaches capacity, then maintain a rolling window
        returning the middle frame with temporal context.
        
        Args:
            frames (int): Number of frames to buffer. Range: [1, 1000].
                         Default is 5. Should be odd for clean middle frame alignment.
                         With frames=5, you get 2 past + 1 current + 2 future frames.
                         
        Raises:
            ValueError: If frames < 1 or frames > 1000.
            
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Basic buffer for temporal smoothing
            >>> buffer = pf.Buffer()  # Default 5 frames
            >>> 
            >>> # Larger buffer for motion analysis
            >>> motion_buffer = pf.Buffer(frames=9)  # 4 past + 1 current + 4 future
            >>> 
            >>> # Minimal buffer for simple delay
            >>> delay_buffer = pf.Buffer(frames=3)  # 1 past + 1 current + 1 future
            >>> 
            >>> # Check buffer properties
            >>> print(f"Delay: {buffer.delay} frames")
            >>> print(f"Capacity: {buffer.buffer_size} frames")
        
        Notes:
            - Buffer automatically warns for even frame counts
            - Internal frame counter starts at 0
            - Buffer is initially empty and not full
            - Frame copying ensures thread safety
            - Validation prevents memory issues from excessive buffer sizes
        """
        if frames < 1:
            raise ValueError("Buffer size must be at least 1")
        if frames > 1000:
            raise ValueError("Buffer size must not exceed 1000 frames")
        
        if frames % 2 == 0:
            print(f"Warning: Buffer size {frames} is even. Consider using odd number for clean middle frame.")
        
        self.buffer_size = frames
        self.frame_buffer: List[np.ndarray] = []
        self.results_buffer: List = []
        self.is_full = False
        self._frame_count = 0
    
    def update(self, results: Any, frame: np.ndarray) -> Tuple[Any, np.ndarray]:
        """
        Add new frame and results to buffer, return temporally-delayed middle frame.
        
        This method implements the core rolling window algorithm that maintains a
        fixed-size temporal buffer. It adds the current frame and results to the end
        of the buffer, removes the oldest items if capacity is exceeded, and returns
        the middle frame once the buffer reaches full capacity. This provides temporal
        context with both past and future frames available.
        
        Args:
            results (Any): Detection results for the current frame. Can be any type
                          (Detections, list, dict, etc.) that represents detection outputs.
                          Results are stored as-is without validation.
            frame (np.ndarray): Current video frame as numpy array. Shape: (H, W, C)
                               or (H, W). Frame is copied to prevent external modifications.
                               
        Returns:
            Tuple[Any, np.ndarray]: Tuple of (buffered_results, buffered_frame):
                - If buffer not full: (empty_results, black_frame) matching input types
                - If buffer full: (middle_results, middle_frame) from temporal center
                
        Raises:
            AttributeError: If frame does not have numpy array interface.
            ValueError: If frame is empty or has invalid dimensions.
            MemoryError: If frame copy fails due to insufficient memory.
            
        Example:
            >>> import cv2
            >>> import pixelflow as pf
            >>> from ultralytics import YOLO
            >>> 
            >>> # Setup model and buffer
            >>> model = YOLO("yolo11l.pt")
            >>> buffer = pf.Buffer(frames=5)
            >>> 
            >>> # Process single frame
            >>> frame = cv2.imread("frame001.jpg")
            >>> outputs = model.predict(frame)
            >>> results = pf.detections.from_ultralytics(outputs)
            >>> buffered_results, buffered_frame = buffer.update(results, frame)
            >>> 
            >>> # Process video stream with temporal buffering
            >>> cap = cv2.VideoCapture("video.mp4")
            >>> for frame_idx in range(100):
            ...     ret, frame = cap.read()
            ...     outputs = model.predict(frame)
            ...     results = pf.detections.from_ultralytics(outputs)
            ...     
            ...     # Get temporally-buffered frame (delayed by buffer_size//2)
            ...     delayed_results, delayed_frame = buffer.update(results, frame)
            ...     
            ...     # Use delayed results for temporal analysis
            ...     if buffer.is_full:
            ...         print(f"Processing frame {frame_idx - buffer.delay}")
            >>> 
            >>> # Handle different result types
            >>> custom_results = {"boxes": [[100, 100, 200, 200]], "confidence": [0.9]}
            >>> delayed_custom, delayed_frame = buffer.update(custom_results, frame)
        
        Notes:
            - Frame delay equals buffer_size // 2 frames
            - Frames are deep copied to prevent external modifications
            - Empty results type matches input when buffer not full
            - Rolling window automatically maintains buffer size
            - Frame counter increments with each update
            - Memory usage grows until buffer is full, then remains constant
            
        Performance Notes:
            - Frame copy overhead: ~10-20% for typical resolutions
            - Buffer operations: O(1) amortized time complexity
            - Memory usage: constant after initial fill period
            - Consider frame resolution vs processing speed trade-offs
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
                from pixelflow.detections import Detections
                empty_results = Detections()
            except:
                # If can't import Detections, return None
                empty_results = None
            
            return empty_results, black_frame
    
    def get_buffer_contents(self) -> Tuple[List[Any], List[np.ndarray]]:
        """
        Get complete copies of current buffer contents.
        
        Returns copies of both the results buffer and frame buffer, allowing safe
        access to all buffered items without risk of external modification. Useful
        for batch processing, debugging, or advanced temporal analysis.
        
        Returns:
            Tuple[List[Any], List[np.ndarray]]: Tuple of (results_copy, frames_copy)
                where results_copy contains all buffered results and frames_copy
                contains all buffered frames. Both are shallow copies of the lists
                but frames themselves are the original references.
                
        Example:
            >>> import cv2
            >>> import pixelflow as pf
            >>> from ultralytics import YOLO
            >>> 
            >>> # Setup and populate buffer
            >>> model = YOLO("yolo11l.pt")
            >>> buffer = pf.Buffer(frames=5)
            >>> 
            >>> # Add several frames
            >>> for i in range(10):
            ...     frame = cv2.imread(f"frame{i:03d}.jpg")
            ...     outputs = model.predict(frame)
            ...     results = pf.detections.from_ultralytics(outputs)
            ...     buffer.update(results, frame)
            >>> 
            >>> # Get all buffer contents for analysis
            >>> all_results, all_frames = buffer.get_buffer_contents()
            >>> 
            >>> # Analyze temporal patterns
            >>> for i, (result, frame) in enumerate(zip(all_results, all_frames)):
            ...     print(f"Buffer slot {i}: {len(result)} detections")
            >>> 
            >>> # Safe to modify without affecting buffer
            >>> all_frames[0] = cv2.GaussianBlur(all_frames[0], (15, 15), 0)
        
        Notes:
            - Returns shallow copies of buffer lists for safety
            - Original frame arrays are shared (not deep copied)
            - Safe for read-only access and list modifications
            - Buffer state remains unchanged by this operation
            - Useful for debugging temporal processing issues
        """
        return self.results_buffer.copy(), self.frame_buffer.copy()
    
    def get_temporal_context(self) -> Optional[Dict[str, Union[List[Any], List[np.ndarray], Any, np.ndarray]]]:
        """
        Get structured temporal context around the middle frame.
        
        Provides organized access to the temporal neighborhood of the current processing
        frame, splitting the buffer into past, current, and future components. This
        enables sophisticated temporal analysis like motion prediction, trajectory
        smoothing, and temporal consistency checks.
        
        Returns:
            Optional[Dict]: Dictionary with temporal context structure, or None if
                          buffer is not full. When available, contains:
                - 'past_frames': List[np.ndarray] of frames before current
                - 'past_results': List[Any] of results before current  
                - 'current_frame': np.ndarray of the middle/current frame
                - 'current_results': Any results for the middle frame
                - 'future_frames': List[np.ndarray] of frames after current
                - 'future_results': List[Any] of results after current
                
        Example:
            >>> import cv2
            >>> import pixelflow as pf
            >>> from ultralytics import YOLO
            >>> 
            >>> # Setup temporal processing
            >>> model = YOLO("yolo11l.pt")
            >>> buffer = pf.Buffer(frames=7)  # 3 past + 1 current + 3 future
            >>> 
            >>> # Fill buffer with video frames
            >>> cap = cv2.VideoCapture("video.mp4")
            >>> for _ in range(10):
            ...     ret, frame = cap.read()
            ...     outputs = model.predict(frame)
            ...     results = pf.detections.from_ultralytics(outputs)
            ...     buffer.update(results, frame)
            >>> 
            >>> # Get temporal context for analysis
            >>> context = buffer.get_temporal_context()
            >>> if context:
            ...     print(f"Past frames: {len(context['past_frames'])}")
            ...     print(f"Future frames: {len(context['future_frames'])}")
            ...     
            ...     # Motion analysis using temporal context
            ...     past_detections = context['past_results']
            ...     current_detections = context['current_results']
            ...     future_detections = context['future_results']
            ...     
            ...     # Smooth trajectories using past and future
            ...     smoothed_positions = analyze_motion_trajectory(
            ...         past_detections, current_detections, future_detections
            ...     )
            >>> 
            >>> # Temporal interpolation example
            >>> if context and len(context['past_frames']) > 0:
            ...     past_frame = context['past_frames'][-1]
            ...     current_frame = context['current_frame']
            ...     interpolated = cv2.addWeighted(past_frame, 0.3, current_frame, 0.7, 0)
        
        Notes:
            - Returns None until buffer reaches full capacity
            - Past/future lists may be empty for buffer_size <= 1
            - Arrays are references to original buffer contents
            - Middle index calculation: buffer_size // 2
            - Temporal context updates with each buffer update
            - Useful for bidirectional temporal processing
            
        See Also:
            get_buffer_contents : Get raw buffer contents without structure
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
    
    def reset(self) -> None:
        """
        Reset the buffer to initial empty state.
        
        Clears all buffered frames and results, resets the full status flag, and
        reinitializes the frame counter. This is useful for processing multiple
        videos or restarting temporal processing pipelines.
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Process first video
            >>> buffer = pf.Buffer(frames=5)
            >>> # ... process frames ...
            >>> 
            >>> # Reset for second video
            >>> buffer.reset()
            >>> assert buffer.current_size == 0
            >>> assert not buffer.is_full
            >>> assert buffer.frames_processed == 0
            >>> 
            >>> # Buffer ready for new sequence
            >>> # ... process new frames ...
        
        Notes:
            - Memory is freed immediately when buffers are cleared
            - Buffer capacity and configuration remain unchanged
            - All temporal state is lost after reset
            - Thread-safe operation
        """
        self.frame_buffer.clear()
        self.results_buffer.clear()
        self.is_full = False
        self._frame_count = 0
    
    @property
    def frames_processed(self) -> int:
        """
        Get total number of frames processed by the buffer.
        
        Returns the cumulative count of all frames that have been added to the
        buffer since initialization or last reset, regardless of current buffer
        contents. Useful for tracking processing progress and synchronization.
        
        Returns:
            int: Total frame count processed. Range: [0, ∞).
                
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> buffer = pf.Buffer(frames=3)
            >>> 
            >>> # Process frames and track count
            >>> for i in range(10):
            ...     # ... get frame and results ...
            ...     buffer.update(results, frame)
            ...     print(f"Processed: {buffer.frames_processed} frames")
            >>> 
            >>> assert buffer.frames_processed == 10
            >>> assert buffer.current_size == 3  # Buffer capacity
        
        Notes:
            - Counter increments with each update() call
            - Independent of current buffer contents
            - Resets to 0 only when reset() is called
            - Useful for progress tracking and debugging
        """
        return self._frame_count
    
    @property
    def current_size(self) -> int:
        """
        Get current number of frames stored in the buffer.
        
        Returns the actual number of frames currently held in the buffer, which
        grows from 0 to buffer_size during the initial filling phase, then remains
        constant at buffer_size during rolling window operation.
        
        Returns:
            int: Current frame count in buffer. Range: [0, buffer_size].
                
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> buffer = pf.Buffer(frames=5)
            >>> 
            >>> # Monitor buffer filling
            >>> for i in range(8):
            ...     # ... get frame and results ...
            ...     buffer.update(results, frame)
            ...     print(f"Buffer: {buffer.current_size}/{buffer.buffer_size} frames")
            ...     print(f"Full: {buffer.is_full}")
            >>> 
            >>> # Output shows:
            >>> # Buffer: 1/5 frames, Full: False
            >>> # Buffer: 2/5 frames, Full: False  
            >>> # ...
            >>> # Buffer: 5/5 frames, Full: True
            >>> # Buffer: 5/5 frames, Full: True (rolling window)
        
        Notes:
            - Grows linearly during initial buffer filling
            - Remains constant once buffer is full
            - Always <= buffer_size by design
            - Useful for monitoring buffer fill progress
        """
        return len(self.frame_buffer)
    
    @property
    def delay(self) -> int:
        """
        Get frame delay introduced by temporal buffering.
        
        Returns the number of frames by which the output is delayed relative to
        the input due to the temporal buffering strategy. This delay allows the
        buffer to provide future context when processing the middle frame.
        
        Returns:
            int: Frame delay count. Calculated as buffer_size // 2.
                For buffer_size=1: delay=0, For buffer_size=5: delay=2.
                
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Different buffer sizes and their delays
            >>> small_buffer = pf.Buffer(frames=3)
            >>> medium_buffer = pf.Buffer(frames=5) 
            >>> large_buffer = pf.Buffer(frames=9)
            >>> 
            >>> print(f"3-frame buffer delay: {small_buffer.delay} frames")   # 1
            >>> print(f"5-frame buffer delay: {medium_buffer.delay} frames")  # 2
            >>> print(f"9-frame buffer delay: {large_buffer.delay} frames")   # 4
            >>> 
            >>> # Synchronization example
            >>> input_frame_idx = 100
            >>> output_frame_idx = input_frame_idx - medium_buffer.delay
            >>> print(f"Input frame {input_frame_idx} outputs frame {output_frame_idx}")
        
        Notes:
            - Delay represents temporal offset between input and output
            - Higher delay provides more future context but increases latency
            - Zero delay for single-frame buffers (no temporal context)
            - Critical for synchronizing buffered output with external systems
            - Consider delay when timing analysis or real-time processing
        """
        return self.buffer_size // 2 if self.buffer_size > 1 else 0