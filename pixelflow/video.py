"""
PixelFlow video utilities for handling video files, streams, and performance monitoring.

Simple, clean API for video processing with support for:
- Video metadata extraction
- File/RTSP/HTTP/webcam sources
- Frame generators with stride and resolution support
- Video writing with custom resolution
- Performance monitoring
"""

from __future__ import annotations

import time
from collections import deque
from typing import Generator, Union, Tuple, Optional

import cv2
import numpy as np


class VideoInfo:
    """
    Video metadata container.
    
    Attributes:
        width (int): Video width in pixels
        height (int): Video height in pixels
        fps (int): Frames per second
        total_frames (int): Total number of frames (None for streams)
    """
    
    def __init__(self, width: int, height: int, fps: int, total_frames: Optional[int] = None):
        self.width = width
        self.height = height
        self.fps = fps
        self.total_frames = total_frames
    
    @classmethod
    def from_source(cls, source: Union[str, int]) -> VideoInfo:
        """
        Extract video information from source.
        
        Args:
            source: Video file path, URL, or webcam index
            
        Returns:
            VideoInfo: Video metadata
            
        Raises:
            Exception: If source cannot be opened
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise Exception(f"Could not open video source: {source}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # For streams, total_frames might be 0 or invalid
        if total_frames <= 0:
            total_frames = None
            
        cap.release()
        return cls(width, height, fps, total_frames)
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get resolution as (width, height) tuple."""
        return (self.width, self.height)
    
    def __repr__(self) -> str:
        return f"VideoInfo(width={self.width}, height={self.height}, fps={self.fps}, total_frames={self.total_frames})"


class VideoWriter:
    """
    Context manager for writing video files with optional resolution scaling.
    
    Example:
        ```python
        import pixelflow as pf
        
        video_info = pf.VideoInfo.from_source("input.mp4")
        
        with pf.VideoWriter("output.mp4", video_info) as writer:
            for frame in pf.get_video_frames("input.mp4"):
                # Process frame here
                writer.write_frame(frame)
        ```
    """
    
    def __init__(self, 
                 path: str, 
                 video_info: VideoInfo, 
                 codec: str = "mp4v",
                 resolution: Optional[Tuple[int, int]] = None):
        """
        Initialize VideoWriter.
        
        Args:
            path: Output video file path
            video_info: Source video information
            codec: Video codec (default: mp4v)
            resolution: Custom output resolution (width, height). If None, uses video_info resolution
        """
        self.path = path
        self.video_info = video_info
        self.codec = codec
        self.resolution = resolution or video_info.resolution
        self._writer = None
    
    def __enter__(self):
        try:
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
        except TypeError as e:
            print(f"{e}. Defaulting to mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        
        self._writer = cv2.VideoWriter(
            self.path,
            fourcc,
            self.video_info.fps,
            self.resolution
        )
        return self
    
    def write_frame(self, frame: np.ndarray):
        """
        Write a single frame to the video.
        
        Args:
            frame: BGR frame to write
        """
        if self._writer is None:
            raise RuntimeError("VideoWriter not initialized. Use within 'with' statement.")
        
        # Resize frame if needed
        if frame.shape[1] != self.resolution[0] or frame.shape[0] != self.resolution[1]:
            frame = cv2.resize(frame, self.resolution)
        
        self._writer.write(frame)
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self._writer:
            self._writer.release()


def get_video_frames(source: Union[str, int],
                    stride: int = 1,
                    start: int = 0,
                    end: Optional[int] = None,
                    resolution: Optional[Tuple[int, int]] = None) -> Generator[np.ndarray, None, None]:
    """
    Generator that yields frames from video source.
    
    Supports files, RTSP streams, HTTP streams, and webcams.
    
    Args:
        source: Video file path, RTSP/HTTP URL, or webcam index (0, 1, etc.)
        stride: Process every nth frame (default: 1)
        start: Starting frame number (default: 0, ignored for streams)
        end: Ending frame number (default: None = all frames, ignored for streams)
        resolution: Resize frames to (width, height). If None, original size
        
    Yields:
        np.ndarray: BGR frame
        
    Example:
        ```python
        # Process video file
        for frame in get_video_frames("video.mp4", stride=2):
            # Process every 2nd frame
            pass
            
        # Process RTSP stream
        for frame in get_video_frames("rtsp://192.168.1.100:554/stream"):
            # Process live stream
            pass
            
        # Process webcam
        for frame in get_video_frames(0, resolution=(640, 480)):
            # Process webcam with custom resolution
            pass
        ```
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise Exception(f"Could not open video source: {source}")
    
    try:
        # For files, handle start/end frames
        if isinstance(source, str) and not source.startswith(('rtsp://', 'http://', 'https://')):
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if end is not None and end > total_frames:
                raise Exception("Requested end frame is beyond video length")
            
            start = max(start, 0)
            end = min(end, total_frames) if end is not None else total_frames
            
            # Seek to start frame
            if start > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            
            frame_position = start
        else:
            # For streams, ignore start/end
            frame_position = 0
            end = float('inf')
        
        while True:
            ret, frame = cap.read()
            if not ret or frame_position >= end:
                break
            
            # Resize if requested
            if resolution is not None:
                frame = cv2.resize(frame, resolution)
            
            yield frame
            
            # Skip frames for stride
            for _ in range(stride - 1):
                success = cap.grab()
                if not success:
                    return
            
            frame_position += stride
            
    finally:
        cap.release()


class Monitor:
    """
    Performance monitor for FPS and timing measurements.
    
    Example:
        ```python
        monitor = Monitor()
        
        for frame in get_video_frames("video.mp4"):
            # Process frame
            monitor.tick()
            print(f"FPS: {monitor.fps:.1f}")
        ```
    """
    
    def __init__(self, sample_size: int = 30):
        """
        Initialize monitor.
        
        Args:
            sample_size: Number of samples to keep for FPS calculation
        """
        self.timestamps = deque(maxlen=sample_size)
        self.start_time = time.time()
    
    @property
    def fps(self) -> float:
        """
        Get current FPS based on recent timestamps.
        
        Returns:
            float: Current FPS, 0.0 if no timestamps
        """
        if len(self.timestamps) < 2:
            return 0.0
        
        time_span = self.timestamps[-1] - self.timestamps[0]
        return len(self.timestamps) / time_span if time_span > 0 else 0.0
    
    @property
    def elapsed_time(self) -> float:
        """Get total elapsed time since creation."""
        return time.time() - self.start_time
    
    def tick(self):
        """Record a timestamp for FPS calculation."""
        self.timestamps.append(time.time())
    
    def reset(self):
        """Clear all timestamps and reset start time."""
        self.timestamps.clear()
        self.start_time = time.time()
