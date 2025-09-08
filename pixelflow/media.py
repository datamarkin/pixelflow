"""
Media handling and display utilities for PixelFlow.

Provides unified media handling for videos, images, and streams with built-in 
resizing, frame iteration, and video writing capabilities. Includes display 
utilities with graceful exit handling and automatic resource cleanup.
"""

import atexit
from pathlib import Path
from typing import Union, Iterator, Optional, Dict
import cv2
import numpy as np

__all__ = [
    "DisplayExit",
    "MediaInfo", 
    "Media",
    "write_frame",
    "show_frame",
    "close_display"
]


class DisplayExit(Exception):
    """Exception raised when user wants to exit display (e.g., presses 'q').
    
    This custom exception allows for graceful handling of user-initiated
    exit requests from display windows, enabling proper cleanup of resources
    before program termination.
    
    Example:
        >>> try:
        ...     show_frame("preview", frame)
        ... except DisplayExit:
        ...     print("User requested exit")
        ...     close_display()
    """
    pass


class MediaInfo:
    """Media metadata container for video and image information.
    
    Stores essential media properties including resolution, framerate, 
    duration, and codec information. Provides structured access to
    media metadata with automatic duration calculation.
    
    Args:
        width (int): Media width in pixels
        height (int): Media height in pixels  
        fps (float): Frames per second for video content
        frame_count (int): Total number of frames in the media
        duration (float): Duration in seconds
        codec (Optional[str]): Codec identifier string. Default is None.
        
    Example:
        >>> import pixelflow as pf
        >>> media = pf.Media("video.mp4")
        >>> info = media.info
        >>> print(f"Resolution: {info.width}x{info.height}")
        Resolution: 1920x1080
        >>> print(f"Duration: {info.duration:.1f}s")
        Duration: 30.5s
    """
    
    def __init__(self, width: int, height: int, fps: float, frame_count: int, 
                 duration: float, codec: str = None):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = frame_count
        self.duration = duration
        self.codec = codec
    
    def __repr__(self):
        return (f"MediaInfo(resolution={self.width}x{self.height}, "
                f"fps={self.fps:.2f}, frames={self.frame_count}, "
                f"duration={self.duration:.2f}s)")


class Media:
    """Unified media handler for videos, images, streams, and webcams.
    
    Provides a consistent interface for accessing various media sources
    including local files, network streams, and webcam feeds. Supports
    automatic resizing and lazy frame iteration for memory efficiency.
    
    Args:
        source (Union[str, int]): Media source path, URL, or webcam index.
                                 Supports local files, RTSP/RTMP streams, 
                                 HTTP URLs, and webcam indices (0, 1, etc.)
        width (Optional[int]): Optional width for automatic frame resizing.
                              Maintains aspect ratio. Default is None (no resize).
                              
    Example:
        >>> import pixelflow as pf
        >>> import cv2
        >>> 
        >>> # Load video file
        >>> media = pf.Media("video.mp4")
        >>> print(f"Video info: {media.info}")
        Video info: MediaInfo(resolution=1920x1080, fps=30.00, frames=900, duration=30.00s)
        >>> 
        >>> # Process frames with resizing
        >>> media_resized = pf.Media("video.mp4", width=640)
        >>> for frame in media_resized.frames:
        ...     # Process each frame (resized to 640px width)
        ...     cv2.imshow("Frame", frame)
        ...     if cv2.waitKey(1) & 0xFF == ord('q'):
        ...         break
        >>> 
        >>> # Use webcam
        >>> webcam = pf.Media(0)  # First webcam
        >>> for frame in webcam.frames:
        ...     # Process webcam frames
        ...     break
        
    Notes:
        - Automatically detects source type (file, stream, webcam)
        - Uses lazy loading for memory efficiency with large videos
        - Maintains aspect ratio when resizing frames
        - Resources are automatically cleaned up on object destruction
        
    See Also:
        MediaInfo : Container for media metadata
        show_frame : Display frames with automatic resizing
    """
    
    def __init__(self, source: Union[str, int], width: int = None):
        self.source = source
        self._info = None
        self._cap = None
        self._resize_width = width
        self._source_type = self._detect_source_type(source)
    
    def _detect_source_type(self, source: Union[str, int]) -> str:
        """Detect the type of media source."""
        if isinstance(source, int):
            return "webcam"
        elif isinstance(source, str):
            if source.startswith(('rtsp://', 'rtmp://', 'udp://')):
                return "stream"
            elif source.startswith(('http://', 'https://')):
                return "url"
            elif Path(source).exists():
                return "file"
            else:
                # Try to find in local directory with same name
                local_file = Path(source).name
                if Path(local_file).exists():
                    self.source = local_file
                    return "file"
                return "mapped"  # Will need resource manager
        return "unknown"
    
    def _get_capture(self) -> cv2.VideoCapture:
        """Get or create cached VideoCapture."""
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.source)
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open video source: {self.source}")
        return self._cap
    
    @property
    def info(self) -> MediaInfo:
        """Get media metadata."""
        if self._info is None:
            cap = self._get_capture()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            self._info = MediaInfo(width, height, fps, frame_count, duration, codec)
        return self._info
    
    @property
    def frames(self) -> Iterator[np.ndarray]:
        """Generate frames from the media source."""
        cap = self._get_capture()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize if width specified
            if self._resize_width:
                h, w = frame.shape[:2]
                height = int(h * self._resize_width / w)
                frame = cv2.resize(frame, (self._resize_width, height))
            
            yield frame
    
    def __del__(self):
        """Clean up VideoCapture on destruction."""
        if self._cap is not None:
            self._cap.release()


# Global writer cache with specs
_writers: Dict[str, cv2.VideoWriter] = {}
_writer_specs: Dict[str, MediaInfo] = {}

def write_frame(output_path: str, frame: np.ndarray, video_info: MediaInfo = None, width: int = None):
    """Write a single frame to video file with automatic writer management.
    
    Automatically manages VideoWriter lifecycle, creating writers as needed and
    caching them for subsequent frames. Supports optional resizing and uses
    MP4V codec for broad compatibility. Writers are automatically released
    on program exit.
    
    Args:
        output_path (str): Path to output video file. Should end with .mp4
                          extension for best compatibility.
        frame (np.ndarray): Frame to write as BGR numpy array with shape (H, W, 3).
        video_info (Optional[MediaInfo]): Video metadata containing fps, resolution,
                                        and other properties. Required on first call
                                        for each output path, cached afterward.
        width (Optional[int]): Optional width for resizing output frame.
                              Maintains aspect ratio. Default is None (no resize).
                              
    Raises:
        ValueError: If video_info is None on first call to a new output path
        RuntimeError: If VideoWriter fails to open or initialize
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> 
        >>> # Get source video info  
        >>> source = pf.Media("input.mp4")
        >>> info = source.info
        >>> 
        >>> # Write processed frames
        >>> for frame in source.frames:
        ...     # Process frame (e.g., add annotations)
        ...     processed_frame = frame.copy()  # Your processing here
        ...     pf.write_frame("output.mp4", processed_frame, info)
        >>> 
        >>> # Write with resizing
        >>> for frame in source.frames:
        ...     pf.write_frame("resized.mp4", frame, info, width=640)
        >>> 
        >>> # Multiple output files (each needs video_info on first call)
        >>> pf.write_frame("version1.mp4", frame1, info)  # First call needs info
        >>> pf.write_frame("version1.mp4", frame2)        # Subsequent calls use cache
        >>> pf.write_frame("version2.mp4", frame1, info)  # New file needs info again
        
    Notes:
        - VideoWriter instances are automatically cached and reused per output path
        - All writers are automatically released on program exit via atexit handler
        - Uses MP4V codec which provides good compatibility across platforms
        - Frame resizing maintains aspect ratio when width parameter is provided
        - Video metadata is cached per output path to avoid repeated specification
        
    Performance Notes:
        - Writer creation has overhead; reusing writers for multiple frames is efficient
        - Frame resizing is performed before writing, affecting output video dimensions
        - Large frame buffers may impact memory usage with many concurrent writers
        
    See Also:
        MediaInfo : Container for video metadata
        Media : Source for obtaining video_info from existing media
    """
    # Resize frame if width specified
    if width:
        h, w = frame.shape[:2]
        height = int(h * width / w)
        frame = cv2.resize(frame, (width, height))
    
    if output_path not in _writers:
        # Get video_info from parameter or cache
        if video_info is None:
            if output_path in _writer_specs:
                video_info = _writer_specs[output_path]
            else:
                raise ValueError(f"First call to write_frame('{output_path}') requires video_info parameter")
        
        # Adjust video_info if frame was resized
        if width:
            h, w = frame.shape[:2]
            video_info = MediaInfo(w, h, video_info.fps, video_info.frame_count, 
                                 video_info.duration, video_info.codec)
        
        # Cache the specs for future calls
        _writer_specs[output_path] = video_info
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_path, fourcc, video_info.fps, 
            (video_info.width, video_info.height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")
        _writers[output_path] = writer
    
    _writers[output_path].write(frame)

@atexit.register
def _cleanup_writers():
    """Clean up all video writers on exit."""
    for writer in _writers.values():
        if writer.isOpened():
            writer.release()
    _writers.clear()
    _writer_specs.clear()




# Display functions
def show_frame(window_name: str, frame: np.ndarray, wait_key: int = 1, width: int = None) -> None:
    """Display a frame in a window with automatic exit handling and resizing.
    
    Shows a frame in an OpenCV window with automatic graceful exit when 'q'
    is pressed. Supports optional resizing for performance optimization and
    better display scaling. Automatically handles program termination and
    window cleanup on exit.
    
    Args:
        window_name (str): Name of the display window. Used as identifier
                          for the OpenCV window.
        frame (np.ndarray): Frame to display as BGR numpy array with shape (H, W, 3).
        wait_key (int): Milliseconds to wait for key press. Default is 1.
                       Use 0 to wait indefinitely until key press.
        width (Optional[int]): Optional width for display resize. Maintains
                              aspect ratio. Provides huge performance boost
                              for large frames. Default is None (no resize).
                              
    Raises:
        SystemExit: Program exits cleanly when 'q' key is pressed
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> 
        >>> # Basic frame display
        >>> image = cv2.imread("image.jpg")
        >>> pf.show_frame("Preview", image)
        >>> 
        >>> # Display with resizing for performance
        >>> large_frame = cv2.imread("large_image.jpg")  # e.g., 4K image
        >>> pf.show_frame("Preview", large_frame, width=800)  # Much faster display
        >>> 
        >>> # Video playback with frame rate control
        >>> media = pf.Media("video.mp4")
        >>> for frame in media.frames:
        ...     pf.show_frame("Video", frame, wait_key=33)  # ~30 FPS display
        >>> 
        >>> # Wait for user input before continuing
        >>> pf.show_frame("Result", processed_frame, wait_key=0)  # Wait indefinitely
        
    Notes:
        - Pressing 'q' triggers clean program exit with proper window cleanup
        - Frame resizing is performed only for display; original frame is unchanged
        - Resizing provides significant performance improvement for large frames
        - Window names are persistent; reusing names updates the same window
        - All OpenCV windows are automatically closed on program exit
        
    Performance Notes:
        - Displaying large frames (>1920x1080) without resizing can be slow
        - Using width parameter can improve display performance by 10x or more
        - Lower wait_key values result in smoother playback but higher CPU usage
        
    See Also:
        close_display : Manual window cleanup function
        write_frame : Save frames to video files
    """
    # Resize for display if width specified (performance optimization)
    if width:
        h, w = frame.shape[:2]
        height = int(h * width / w)
        display_frame = cv2.resize(frame, (width, height))
    else:
        display_frame = frame
    
    cv2.imshow(window_name, display_frame)
    key = cv2.waitKey(wait_key) & 0xFF
    
    if key == ord('q'):
        close_display()
        exit(0)  # Clean program exit


def close_display():
    """Close all OpenCV display windows.
    
    Closes all currently open OpenCV windows and releases associated
    resources. Useful for manual cleanup or when handling exceptions
    in display code.
    
    Example:
        >>> import pixelflow as pf
        >>> import cv2
        >>> 
        >>> # Display some frames
        >>> image = cv2.imread("image.jpg")
        >>> pf.show_frame("Window1", image)
        >>> pf.show_frame("Window2", image)
        >>> 
        >>> # Manual cleanup
        >>> pf.close_display()
        >>> 
        >>> # Exception handling
        >>> try:
        ...     for frame in media.frames:
        ...         pf.show_frame("Preview", frame)
        ... except KeyboardInterrupt:
        ...     pf.close_display()
        ...     print("Display closed by user")
        
    Notes:
        - Automatically called when 'q' is pressed in show_frame()
        - Safe to call multiple times; no-op if no windows are open
        - Does not affect video writers or other non-display resources
        
    See Also:
        show_frame : Display frames with automatic 'q' key handling
    """
    cv2.destroyAllWindows()


